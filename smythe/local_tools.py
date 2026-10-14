"""Local tools — Python functions offered as model tools to the agents you name.

A :class:`LocalTool` pairs a :class:`~smythe.tools.ToolSpec` with a handler
your code supplies. :class:`LocalToolRuntime` offers each tool only to the
agent ids you list for it, never by agent name: a generated plan names its
agents freely, so a name can repeat yours, but its agents never get your ids.
:class:`CompositeToolRuntime` merges several runtimes, such as local tools
beside MCP servers, into one session per node.

:func:`python_skill` makes a tool of an Agent Skills record and a Python
handler. It never reads the skill's directory, so its ``scripts/`` are never
run or imported.

The model's tool loop is the only retry owner: each call runs its handler
once, and a failure or timeout returns to the model as an error result.
Calls are not journaled, so a retried node or a resumed run may call a
handler again, as it may an MCP tool.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from smythe._canonical import sha256_hex
from smythe._failures import SecurityStop
from smythe.agent_skills import AgentSkill
from smythe.budget import BudgetValidationError
from smythe.provider import ProviderResponseError
from smythe.tools import ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec
from smythe.workflow_store import WorkflowError

if TYPE_CHECKING:
    from smythe.agent import Agent

__all__ = [
    "CompositeToolRuntime", "LocalTool", "LocalToolRuntime", "LocalToolSession", "python_skill",
]

# The most characters of a result's text, or an error's message, the model sees.
MAX_RESULT_CHARS = 64 * 1024

# What the executor's tool loop stops the node on instead of showing the
# model: unusable or unsettled provider responses, invalid accounting, journal
# failures and security stops.
_STOPS = (ProviderResponseError, BudgetValidationError, WorkflowError, SecurityStop)


def _stops(error: BaseException) -> bool:
    if isinstance(error, _STOPS):
        return True
    return isinstance(error, BaseExceptionGroup) and any(map(_stops, error.exceptions))


def _positive_seconds(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:  # an integer too large to become a float
        return False


@dataclass(frozen=True)
class LocalTool:
    """A tool whose calls run *handler* in this process.

    *handler* takes the call's arguments, a copy of the dict the model sent,
    and returns the result text; it may be ``async``. A synchronous handler
    runs in a thread of its own, so it never blocks the event loop. The
    arguments come from the model: check them as you would any untrusted
    input. *version* is your name for what the handler does;
    ``schema_sha256`` is the SHA-256 of the input schema as canonical JSON.
    The spec keeps its own copy of the schema.

    With *timeout_s*, a call that has not returned in time becomes an error
    result. An ``async`` handler is cancelled then; a synchronous one cannot
    be, and its thread runs on until the handler returns, though the run does
    not wait for it.
    """

    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Any]
    version: str
    timeout_s: float | None = None
    schema_sha256: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ToolSpec):
            raise TypeError(f"spec must be a ToolSpec, not {type(self.spec).__name__}")
        name, schema = self.spec.name, self.spec.input_schema
        if not callable(self.handler):
            raise TypeError(f"Tool {name!r}: handler must be callable")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(f"Tool {name!r}: version must be non-blank text")
        if self.timeout_s is not None and not _positive_seconds(self.timeout_s):
            raise ValueError(f"Tool {name!r}: timeout_s must be a positive, finite number")
        if not isinstance(schema, dict):
            raise TypeError(f"Tool {name!r}: input_schema must be a JSON Schema object (a dict)")
        try:
            digest = sha256_hex(schema)
        except ValueError as error:
            raise ValueError(f"Tool {name!r}: input_schema {error}") from None
        spec = ToolSpec(name, self.spec.description, copy.deepcopy(schema))
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "schema_sha256", digest)

    @property
    def name(self) -> str:
        return self.spec.name

    def describe(self) -> dict[str, str]:
        """``{"name", "version", "schema_sha256"}``, the tool's identity for records."""
        return {"name": self.name, "version": self.version, "schema_sha256": self.schema_sha256}


def _is_async(handler: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(
        getattr(handler, "__call__", None))


async def _in_thread(tool: LocalTool, arguments: dict[str, Any]) -> Any:
    """Run a synchronous handler in a daemon thread of its own and await its outcome.

    Unlike the loop's default executor, which ``asyncio.run`` waits for when
    it closes, a handler that outlives its timeout cannot hold up the end of
    the run.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = contextvars.copy_context()

    def settle(outcome: Any, error: BaseException | None) -> None:
        if future.done():  # The call timed out or was cancelled.
            return
        if error is None:
            future.set_result(outcome)
        else:
            future.set_exception(error)

    def work() -> None:
        outcome, error = None, None
        try:
            outcome = context.run(tool.handler, arguments)
        except StopIteration:  # A future cannot hold one; a generator would convert it too.
            error = RuntimeError("handler raised StopIteration")
        except BaseException as raised:  # Handed to the awaiting task, as asyncio.to_thread does.
            error = raised
        try:
            loop.call_soon_threadsafe(settle, outcome, error)
        except RuntimeError:  # The loop closed while the handler ran.
            pass

    threading.Thread(target=work, name=f"smythe-tool-{tool.name}", daemon=True).start()
    return await future


class LocalToolSession(ToolSession):
    """The local tools offered to one node: those granted to its agent's id."""

    def __init__(self, tools: Mapping[str, LocalTool],
                 max_result_chars: int = MAX_RESULT_CHARS) -> None:
        self._tools = dict(tools)
        self._limit = max_result_chars

    @property
    def tools(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def call(self, tool_call: ToolCall) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return self._result(tool_call, f"Unknown tool {tool_call.name!r}", True)
        arguments = copy.deepcopy(tool_call.arguments)
        scope = asyncio.timeout(tool.timeout_s)
        try:
            async with scope:
                if _is_async(tool.handler):
                    text = await tool.handler(arguments)
                else:
                    text = await _in_thread(tool, arguments)
        except Exception as error:
            if _stops(error):
                raise
            if isinstance(error, TimeoutError) and scope.expired():
                return self._result(tool_call, f"Tool {tool.name!r} did not finish within "
                                               f"{tool.timeout_s:g} seconds", True)
            return self._result(tool_call, str(error) or type(error).__name__, True)
        if not isinstance(text, str):
            shown = type(text).__name__
            return self._result(tool_call, f"Tool {tool.name!r} returned {shown}, not text", True)
        return self._result(tool_call, text, False)

    def _result(self, tool_call: ToolCall, text: str, is_error: bool) -> ToolResult:
        if len(text) > self._limit:
            text = text[:self._limit] + f"\n[{len(text) - self._limit} more characters not shown]"
        return ToolResult(tool_call_id=tool_call.id, content=text, is_error=is_error)


class LocalToolRuntime(ToolRuntime):
    """Offers local tools to the agents whose ids *agents* lists.

        runtime = LocalToolRuntime([search], agents={"developer": ["notes.search"]})
        swarm = Swarm(tool_runtime=runtime, ...)

    *agents* maps an agent id to the names of the tools its nodes are
    offered. Only the id counts: an agent with any other id, including one a
    generated plan created under the same name, and a node without an agent
    are offered none, whatever capability tags or skills the agent has. Tool
    names must be unique and every name *agents* lists must be defined. A
    result's text and an error's message are cut to *max_result_chars*
    characters, with a note of how many more there were.
    """

    def __init__(self, tools: Sequence[LocalTool], *, agents: Mapping[str, Sequence[str]],
                 max_result_chars: int = MAX_RESULT_CHARS) -> None:
        if (isinstance(tools, str) or not isinstance(tools, Sequence)
                or not all(isinstance(tool, LocalTool) for tool in tools)):
            raise TypeError("tools must be a sequence of LocalTool")
        defined = _unique(tool.name for tool in tools)
        if not isinstance(agents, Mapping):
            raise TypeError("agents must map agent ids to sequences of tool names")
        grants: dict[str, tuple[str, ...]] = {}
        for agent_id, names in agents.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise TypeError("agents must be keyed by non-empty agent ids")
            if (isinstance(names, (str, bytes)) or not isinstance(names, Sequence)
                    or not all(isinstance(name, str) for name in names)):
                raise TypeError(f"agents[{agent_id!r}] must be a sequence of tool names")
            undefined = sorted(set(names) - defined)
            if undefined:
                raise ValueError(f"agents[{agent_id!r}] names undefined tools {undefined}")
            grants[agent_id] = tuple(dict.fromkeys(names))
        if type(max_result_chars) is not int or max_result_chars < 1:
            raise ValueError("max_result_chars must be a positive integer")
        self._tools = MappingProxyType({tool.name: tool for tool in tools})
        self._grants = MappingProxyType(grants)
        self._limit = max_result_chars

    def open(self, agent: Agent | None):
        return self._open(agent)

    @asynccontextmanager
    async def _open(self, agent: Agent | None):
        names = () if agent is None else self._grants.get(agent.id, ())
        yield LocalToolSession({name: self._tools[name] for name in names}, self._limit)

    def describe(self) -> list[dict[str, str]]:
        """Each tool's name, version and ``schema_sha256``, sorted by name."""
        return [self._tools[name].describe() for name in sorted(self._tools)]


def _unique(names: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Tool name {name!r} is defined more than once")
        seen.add(name)
    return seen


def _defined(runtime: ToolRuntime) -> list[str]:
    """The tool names *runtime* defines before any session opens; none for other runtimes."""
    if isinstance(runtime, LocalToolRuntime):
        return list(runtime._tools)
    if isinstance(runtime, CompositeToolRuntime):
        return [name for child in runtime._runtimes for name in _defined(child)]
    return []


class _CompositeSession(ToolSession):
    def __init__(self, sessions: list[ToolSession]) -> None:
        self._tools: list[ToolSpec] = []
        self._routes: dict[str, ToolSession] = {}
        for session in sessions:
            for spec in session.tools:
                if spec.name in self._routes:
                    raise ValueError(
                        f"Tool name {spec.name!r} is offered by more than one tool runtime")
                self._routes[spec.name] = session
                self._tools.append(spec)

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    async def call(self, tool_call: ToolCall) -> ToolResult:
        session = self._routes.get(tool_call.name)
        if session is None:
            return ToolResult(tool_call_id=tool_call.id, content=f"Unknown tool {tool_call.name!r}",
                              is_error=True)
        return await session.call(tool_call)


class CompositeToolRuntime(ToolRuntime):
    """Merges the sessions of several tool runtimes into one per node.

        runtime = CompositeToolRuntime(LocalToolRuntime(...), MCPToolRuntime())

    Each runtime opens its own session for the node's agent. The node is
    offered all of their tools, and each call goes to the session that
    offered it. A tool name may come from one runtime only: local runtimes
    that define the same name are refused here, and sessions that offer the
    same name raise ``ValueError`` when they open, before the node's first
    provider call.
    """

    def __init__(self, *runtimes: ToolRuntime) -> None:
        if not runtimes:
            raise ValueError("CompositeToolRuntime needs at least one tool runtime")
        for runtime in runtimes:
            if not isinstance(runtime, ToolRuntime):
                raise TypeError("CompositeToolRuntime merges ToolRuntime objects, "
                                f"not {type(runtime).__name__}")
        self._runtimes = runtimes
        _unique(_defined(self))

    @property
    def runtimes(self) -> tuple[ToolRuntime, ...]:
        """The tool runtimes this one merges, in the order they were given."""
        return self._runtimes

    def open(self, agent: Agent | None):
        return self._open(agent)

    @asynccontextmanager
    async def _open(self, agent: Agent | None):
        async with AsyncExitStack() as stack:
            sessions = [await stack.enter_async_context(runtime.open(agent))
                        for runtime in self._runtimes]
            yield _CompositeSession(sessions)


def python_skill(
    skill: AgentSkill, handler: Callable[[dict[str, Any]], Any], *, input_schema: dict[str, Any],
    version: str, timeout_s: float | None = None,
) -> LocalTool:
    """Return a :class:`LocalTool` that runs *handler* for an Agent Skill.

    The tool is named ``skill.<name>`` and described by the skill's
    ``description``, as its ``SKILL.md`` states it; *input_schema*, *version*
    and *timeout_s* are yours. Nothing is read from the skill's directory:
    its ``scripts/`` are never run and nothing is imported from it. The
    skill's declared ``allowed-tools`` grant nothing, and only a
    :class:`LocalToolRuntime` that lists the tool for an agent id offers it.
    """
    if not isinstance(skill, AgentSkill):
        raise TypeError(f"skill must be an AgentSkill, not {type(skill).__name__}")
    spec = ToolSpec(f"skill.{skill.name}", skill.description, input_schema)
    return LocalTool(spec, handler, version, timeout_s=timeout_s)
