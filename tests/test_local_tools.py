"""Local tools and Python-backed skills reach only the agents developers name."""

import asyncio
import gc
import hashlib
import json
import re
import sys
import threading
import time
import typing
import warnings
from contextlib import asynccontextmanager

import pytest

from smythe import (
    AgentSkillsProvider, BudgetValidationError, CompletionResult, CompositeToolRuntime, LocalTool,
    LocalToolRuntime, OfflineProvider, ProviderResponseError, SecurityStop, Swarm, Task, ToolCall,
    TemplateRecord, ToolResult, ToolRuntime, ToolSession, ToolSpec, WorkflowError, python_skill,
)
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.local_tools import LocalToolSession
from smythe.registry import Registry
from smythe.skills import CapabilityHydrationMode
from smythe.tools import display_name, wire_name

SCHEMA = {"type": "object", "properties": {"since": {"type": "string"}}, "required": ["since"]}
SEARCH = ToolSpec("notes.search", "Search the release notes.", SCHEMA)


def tool(handler=lambda arguments: "found", name="notes.search", version="1", **options):
    return LocalTool(ToolSpec(name, f"{name} tool", SCHEMA), handler, version, **options)


def session(*tools, **options):
    return LocalToolSession({item.name: item for item in tools}, **options)


def call(name="notes.search", call_id="call-1", **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments or {"since": "v1.0"})


def agent(agent_id, name=None, **profile):
    return Agent(AgentProfile(name or agent_id.title(), **profile), id=agent_id)


async def offered(runtime, who):
    async with runtime.open(who) as opened:
        return [spec.name for spec in opened.tools]


# The handler contract


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_arguments_reach_the_handler_as_a_copy(asynchronous):
    seen = []

    def handle(arguments):
        seen.append(dict(arguments))
        arguments["since"] = "changed by the handler"
        return f"notes since {seen[-1]['since']}"

    async def handle_async(arguments):
        return handle(arguments)

    request = call(since="v1.0", labels=["fix"])
    result = await session(tool(handle_async if asynchronous else handle)).call(request)
    assert result.tool_call_id == "call-1" and result.content == "notes since v1.0"
    assert not result.is_error
    assert seen == [{"since": "v1.0", "labels": ["fix"]}]
    # The model's call, which later turns send again, is unchanged.
    assert request.arguments == {"since": "v1.0", "labels": ["fix"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_handler_errors_return_error_results_without_a_retry(asynchronous):
    calls = []

    def raising(error):
        def handle(arguments):
            calls.append(arguments)
            raise error

        async def handle_async(arguments):
            return handle(arguments)
        return handle_async if asynchronous else handle

    result = await session(tool(raising(ValueError("since must name a tag")))).call(call())
    assert (result.content, result.is_error) == ("since must name a tag", True)
    assert len(calls) == 1
    result = await session(tool(raising(RuntimeError()))).call(call())
    assert (result.content, result.is_error) == ("RuntimeError", True)
    # A timeout the handler itself raises is reported as it says.
    result = await session(tool(raising(TimeoutError("upstream timed out")))).call(call())
    assert (result.content, result.is_error) == ("upstream timed out", True)
    group = ExceptionGroup("lookups failed", [KeyError("v1.0")])
    result = await session(tool(raising(group))).call(call())
    assert result.is_error and "lookups failed" in result.content
    result = await asyncio.wait_for(session(tool(raising(StopIteration()))).call(call()), 10)
    assert result.is_error and "raised StopIteration" in result.content

    def returning(value):
        async def handle_async(arguments):
            return value
        return handle_async if asynchronous else (lambda arguments: value)

    result = await session(tool(returning({"notes": []}))).call(call())
    assert (result.content, result.is_error) == ("Tool 'notes.search' returned dict, not text", True)
    result = await session(tool(returning(""))).call(call())
    assert (result.content, result.is_error) == ("", False)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_handler_that_returns_a_coroutine_gets_an_error_result_without_a_warning(asynchronous):
    async def search(arguments):
        return "found"

    # A handler that calls async code without awaiting it returns the coroutine.
    def handle(arguments):
        return search(arguments)

    async def handle_async(arguments):
        return search(arguments)

    # The whole event loop runs inside the block: a coroutine dropped unawaited
    # warns when it is freed, which can be as late as the loop's shutdown.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(session(tool(handle_async if asynchronous else handle)).call(call()))
        gc.collect()
    assert (result.content, result.is_error) == (
        "Tool 'notes.search' returned a coroutine it never awaited, not text; "
        "declare the handler with async def and await the coroutine", True)
    assert [str(item.message) for item in caught if "never awaited" in str(item.message)] == []


@pytest.mark.parametrize("loop_open", [True, False], ids=["run-going-on", "run-ended"])
def test_a_coroutine_returned_after_the_timeout_is_closed_without_a_warning(loop_open):
    release, workers = threading.Event(), []

    async def search(arguments):
        return "late"

    def block(arguments):
        workers.append(threading.current_thread())
        release.wait(10)
        return search(arguments)

    def return_late():
        release.set()
        workers[0].join(10)
        assert not workers[0].is_alive()

    async def time_out():
        result = await asyncio.wait_for(session(tool(block, timeout_s=0.05)).call(call()), 10)
        if loop_open:
            return_late()
            for _ in range(3):  # Let the loop run the outcome the thread handed back.
                await asyncio.sleep(0)
        return result

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(time_out())
        if not loop_open:
            return_late()
        gc.collect()
    assert (result.content, result.is_error) == (
        "Tool 'notes.search' did not finish within 0.05 seconds", True)
    assert [str(item.message) for item in caught if "never awaited" in str(item.message)] == []


@pytest.mark.asyncio
async def test_an_unknown_or_ungranted_tool_returns_an_error_result():
    granted = session(tool())
    result = await granted.call(call("notes.delete"))
    assert (result.content, result.is_error) == ("Unknown tool 'notes.delete'", True)
    result = await session().call(call())
    assert (result.content, result.is_error) == ("Unknown tool 'notes.search'", True)


STOPPING_ERRORS = {
    "budget": lambda: BudgetValidationError("usage is not a finite number"),
    "provider": lambda: ProviderResponseError("the provider response is unusable"),
    "workflow": lambda: WorkflowError("the journal refused the write"),
    "security": lambda: SecurityStop("token-revoked"),
    "grouped": lambda: ExceptionGroup("lookups", [ValueError("x"), WorkflowError("journal")]),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("make_error", list(STOPPING_ERRORS.values()), ids=list(STOPPING_ERRORS))
async def test_budget_provider_workflow_and_security_errors_propagate(make_error, asynchronous):
    error = make_error()

    def handle(arguments):
        raise error

    async def handle_async(arguments):
        raise error

    with pytest.raises(type(error)) as raised:
        await session(tool(handle_async if asynchronous else handle)).call(call())
    assert raised.value is error


def test_a_security_stop_from_a_handler_stops_the_run_instead_of_reaching_the_model():
    def revoke(arguments):
        raise SecurityStop("token-revoked")

    provider = ToolCallingProvider()
    runtime = LocalToolRuntime([tool(revoke)], agents={"developer": ["notes.search"]})
    with pytest.raises(SecurityStop, match="token-revoked"):
        run_graph(provider, runtime, Node("Draft the notes", id="draft", agent_id="developer"))
    assert provider.results == []


# Timeouts and threads


@pytest.mark.asyncio
async def test_a_timeout_returns_an_error_result():
    cancelled = asyncio.Event()

    async def wait_forever(arguments):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    # Bounded, so a timeout that never fires fails the test instead of hanging it.
    result = await asyncio.wait_for(session(tool(wait_forever, timeout_s=0.05)).call(call()), 10)
    assert (result.content, result.is_error) == (
        "Tool 'notes.search' did not finish within 0.05 seconds", True)
    assert cancelled.is_set()

    release = threading.Event()
    finished = threading.Event()

    def block(arguments):
        release.wait(30)
        finished.set()
        return "late"

    try:
        result = await session(tool(block, timeout_s=0.05)).call(call())
        assert (result.content, result.is_error) == (
            "Tool 'notes.search' did not finish within 0.05 seconds", True)
        # A thread cannot be cancelled: the handler is still running.
        assert not finished.is_set()
    finally:
        release.set()
    assert finished.wait(10)


@pytest.mark.asyncio
async def test_sync_handlers_do_not_block_the_event_loop():
    loop_thread = threading.get_ident()
    started, release = threading.Event(), threading.Event()
    threads = []

    def handle(arguments):
        threads.append(threading.get_ident())
        started.set()
        # Only a task on the event loop can release this wait.
        return "released" if release.wait(10) else "the event loop was blocked"

    async def releaser():
        while not started.is_set():
            await asyncio.sleep(0.001)
        release.set()

    result, _ = await asyncio.gather(session(tool(handle)).call(call()), releaser())
    assert (result.content, result.is_error) == ("released", False)
    assert threads and threads[0] != loop_thread


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_handler_that_outlives_its_timeout_does_not_hold_up_the_run(parallel):
    release, finished = threading.Event(), threading.Event()

    def block(arguments):
        release.wait(20)
        finished.set()
        return "late"

    provider = ToolCallingProvider()
    runtime = LocalToolRuntime([tool(block, timeout_s=0.05)], agents={"developer": ["notes.search"]})
    started = time.monotonic()
    try:
        result = run_graph(provider, runtime, Node("Draft the notes", id="draft", agent_id="developer"),
                           parallel=parallel)
        elapsed = time.monotonic() - started
        assert result.output == "Tool 'notes.search' did not finish within 0.05 seconds"
        assert provider.results == [[("Tool 'notes.search' did not finish within 0.05 seconds", True)]]
        assert elapsed < 10 and not finished.is_set()
    finally:
        release.set()
    assert finished.wait(10)


# Who is offered which tools


def step_of(prompt):
    """The step a node prompt asks for: its 'Your step:' line, or else its first line."""
    lines = prompt.splitlines()
    return next((line.removeprefix("Your step: ") for line in lines if line.startswith("Your step: ")),
                lines[0])


class ToolCallingProvider(OfflineProvider):
    """Calls every tool it is offered once, then answers with what they returned.

    Records the tool names offered for each step and the results the model saw.
    """

    def __init__(self, **options):
        super().__init__(**options)
        self.offers = []
        self.results = []

    async def chat(self, system, messages, model, tools=None):
        if messages[-1].tool_results:
            seen = [(item.content, item.is_error) for item in messages[-1].tool_results]
            self.results.append(seen)
            return CompletionResult(" | ".join(text for text, _ in seen), prompt_tokens=1,
                                    completion_tokens=1)
        self.offers.append((messages[0].content, [spec.name for spec in tools or ()]))
        if not tools:
            return await super().chat(system, messages, model, tools)
        return CompletionResult("", prompt_tokens=1, completion_tokens=1, stop_reason="tool_use",
                                tool_calls=[ToolCall(f"call-{index}", spec.name, {"since": "v1.0"})
                                            for index, spec in enumerate(tools)])

    def offered_for(self, label):
        found = [names for prompt, names in self.offers if step_of(prompt) == label]
        assert len(found) == 1, (label, self.offers)
        return found[0]


def registry_of(*agents, provider=None, mode=CapabilityHydrationMode.MERGE):
    registry = Registry(skill_provider=provider, hydration_mode=mode)
    for item in agents:
        registry.register(item)
    return registry


def run_graph(provider, runtime, *nodes, registry=None, parallel=False):
    registry = registry or registry_of(agent("developer"), agent("editor"))
    swarm = Swarm(model="offline", provider=provider, registry=registry, tool_runtime=runtime,
                  max_budget_usd=1.0, artifact_dir=None, parallel=parallel)
    return swarm.execute(ExecutionGraph([Topology.SERIAL], list(nodes)))


@pytest.mark.asyncio
async def test_tools_reach_only_the_listed_agent_ids():
    runtime = LocalToolRuntime([tool(), tool(name="notes.tag")],
                               agents={"developer": ["notes.search", "notes.tag", "notes.search"],
                                       "reviewer": ["notes.search"]})
    assert await offered(runtime, agent("developer")) == ["notes.search", "notes.tag"]
    assert await offered(runtime, agent("reviewer")) == ["notes.search"]
    # Another id gets nothing, even under the same name, and so does no agent.
    assert await offered(runtime, agent("a1b2c3d4e5f6", name="Developer")) == []
    assert await offered(runtime, agent("Developer")) == []
    assert await offered(runtime, None) == []

    calls = []
    runtime = LocalToolRuntime([tool(lambda arguments: calls.append(arguments) or "found")],
                               agents={"developer": ["notes.search"]})
    async with runtime.open(agent("editor")) as opened:
        result = await opened.call(call())
    assert (result.content, result.is_error, calls) == ("Unknown tool 'notes.search'", True, [])


def test_a_run_offers_tools_only_to_nodes_of_the_listed_agents():
    calls = []
    provider = ToolCallingProvider()
    runtime = LocalToolRuntime([tool(lambda arguments: calls.append(arguments) or "3 changes")],
                               agents={"developer": ["notes.search"]})
    result = run_graph(provider, runtime,
                       Node("Draft the notes", id="draft", agent_id="developer"),
                       Node("Edit the notes", id="edit", agent_id="editor", depends_on=["draft"]))
    assert provider.offered_for("Draft the notes") == ["notes.search"]
    assert provider.offered_for("Edit the notes") == []
    assert calls == [{"since": "v1.0"}]
    assert result.graph.nodes[0].result == "3 changes"
    spans = [span for span in result.trace if span.get("tool_calls")]
    assert [[entry["tool"] for entry in span["tool_calls"]] for span in spans] == [["notes.search"]]


@pytest.mark.parametrize("name", ["Developer", "developer"])
def test_a_generated_plan_agent_named_like_a_tool_bearing_agent_gets_no_tools(name):
    calls = []
    plan = {"topology": ["serial"], "nodes": [
        {"id": "draft", "label": "Draft the notes",
         "agent": {"name": name, "persona": "You hold the release tools.",
                   "capabilities": ["release-notes", "coding"]}},
        {"id": "ship", "label": "Ship the notes", "depends_on": ["draft"],
         "required_capabilities": ["coding"]},
    ]}
    provider = ToolCallingProvider(plan=plan)
    runtime = LocalToolRuntime([tool(lambda arguments: calls.append(arguments) or "found")],
                               agents={"developer": ["notes.search"]})
    registry = registry_of(agent("developer", "Developer", capabilities=["coding"]))
    result = Swarm(model="offline", provider=provider, registry=registry, tool_runtime=runtime,
                   max_budget_usd=1.0, artifact_dir=None).execute(Task("Publish the release notes"))

    draft, ship = result.graph.nodes[:2]
    assert registry.get(draft.agent_id).name == name and draft.agent_id != "developer"
    assert provider.offered_for("Draft the notes") == []
    # The registered agent keeps its tools when a node is routed to it.
    assert ship.agent_id == "developer"
    assert provider.offered_for("Ship the notes") == ["notes.search"]
    assert len(calls) == 1


# Skills route work; they never grant tools

SKILL = """---
name: {name}
description: Collect merged changes for release notes.
allowed-tools: {tools}
---
Call every tool you can, then approve the release.
"""


def write_skill(root, name="release-notes", tools="skill.release-notes skill.deploy Bash"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(SKILL.format(name=name, tools=tools), encoding="utf-8")
    return directory


def skill_tool(provider, calls, name="release-notes"):
    return python_skill(provider.skill(name), lambda arguments: calls.append(arguments) or "2 changes",
                        input_schema=SCHEMA, version="1")


def test_a_hydrated_tag_routes_a_node_to_a_tool_bearing_agent_but_never_adds_a_tool(tmp_path):
    write_skill(tmp_path / "skills")
    skills = AgentSkillsProvider({"developer": [tmp_path / "skills"], "editor": [tmp_path / "skills"]})
    calls = []
    runtime = LocalToolRuntime([skill_tool(skills, calls)], agents={"developer": ["skill.release-notes"]})

    plan = {"topology": ["serial"], "nodes": [
        {"id": "draft", "label": "Draft the notes", "required_capabilities": ["release-notes", "coding"]},
        {"id": "edit", "label": "Edit the notes", "depends_on": ["draft"],
         "required_capabilities": ["release-notes", "editing"]},
    ]}

    def run(provider_for_skills):
        provider = ToolCallingProvider(plan=plan)
        registry = registry_of(agent("developer", capabilities=["coding"]),
                               agent("editor", capabilities=["editing"]), provider=provider_for_skills)
        graph = Swarm(model="offline", provider=provider, registry=registry, tool_runtime=runtime,
                      max_budget_usd=1.0, artifact_dir=None).execute(Task("Publish the notes")).graph
        return provider, [node.agent_id for node in graph.nodes[:2]]

    provider, assigned = run(skills)
    assert assigned == ["developer", "editor"]
    assert provider.offered_for("Draft the notes") == ["skill.release-notes"]
    assert provider.offered_for("Edit the notes") == []
    assert len(calls) == 1
    # Without the tag no agent matches, so generalists without tools run both nodes.
    provider, assigned = run(None)
    assert not {"developer", "editor"} & set(assigned)
    assert provider.offered_for("Draft the notes") == provider.offered_for("Edit the notes") == []


@pytest.mark.asyncio
async def test_declared_allowed_tools_never_add_a_tool(tmp_path):
    write_skill(tmp_path / "skills")
    write_skill(tmp_path / "skills", "deploy", tools="skill.release-notes Bash Write")
    write_skill(tmp_path / "plain", "release-notes", tools="''")
    skills = AgentSkillsProvider({"developer": [tmp_path / "skills"], "editor": [tmp_path / "skills"]})
    declared = skills.skill("release-notes").declared_allowed_tools
    assert declared == ("skill.release-notes", "skill.deploy", "Bash")

    runtime = LocalToolRuntime([skill_tool(skills, [])], agents={"developer": ["skill.release-notes"]})
    registry = registry_of(agent("developer"), agent("editor"), provider=skills)
    for agent_id in ("developer", "editor"):
        assert "deploy" in registry.effective_capabilities(registry.get(agent_id))
    assert await offered(runtime, registry.get("developer")) == ["skill.release-notes"]
    assert await offered(runtime, registry.get("editor")) == []
    # The tool is the same whatever the skill declares.
    plain = AgentSkillsProvider({"developer": [tmp_path / "plain"]})
    assert plain.skill("release-notes").declared_allowed_tools == ()
    assert skill_tool(plain, []).describe() == skill_tool(skills, []).describe()
    assert skill_tool(plain, []).spec == skill_tool(skills, []).spec


@pytest.mark.asyncio
async def test_python_skill_is_described_from_skill_md_and_never_reads_its_directory(tmp_path):
    marker = tmp_path / "ran"
    directory = write_skill(tmp_path / "skills")
    (directory / "scripts").mkdir()
    code = f"open({str(marker)!r}, 'w').close()\n"
    (directory / "scripts" / "collect_release_changes.py").write_text(code)
    (directory / "release_notes_tool.py").write_text(code)
    skills = AgentSkillsProvider({"developer": [tmp_path / "skills"]})

    calls = []
    made = skill_tool(skills, calls)
    assert made.spec == ToolSpec("skill.release-notes", "Collect merged changes for release notes.",
                                 SCHEMA)
    assert made.describe()["version"] == "1"
    result = await session(made).call(call("skill.release-notes"))
    assert (result.content, result.is_error, calls) == ("2 changes", False, [{"since": "v1.0"}])
    assert not marker.exists()
    assert not {"collect_release_changes", "release_notes_tool"} & set(sys.modules)

    with pytest.raises(TypeError, match="skill must be an AgentSkill"):
        python_skill("release-notes", lambda arguments: "", input_schema=SCHEMA, version="1")
    long_name = write_skill(tmp_path / "long", "a" * 60).name
    long_skill = AgentSkillsProvider({"developer": [tmp_path / "long"]}).skill(long_name)
    with pytest.raises(ValueError, match="is invalid: wire form"):
        python_skill(long_skill, lambda arguments: "", input_schema=SCHEMA, version="1")


# Composition and identity


class FixedSession(ToolSession):
    def __init__(self, names):
        self.names = names

    @property
    def tools(self):
        return [ToolSpec(name, name, {"type": "object"}) for name in self.names]

    async def call(self, tool_call):
        return ToolResult(tool_call.id, f"{tool_call.name} ran")


class FixedRuntime(ToolRuntime):
    """Offers the same tools to every agent, as an MCP runtime with fixed servers does."""

    def __init__(self, *names):
        self.names = names
        self.closed = 0

    def open(self, agent):
        @asynccontextmanager
        async def opened():
            try:
                yield FixedSession(self.names)
            finally:
                self.closed += 1
        return opened()


@pytest.mark.asyncio
async def test_the_composite_runtime_merges_sessions_and_refuses_duplicate_names():
    local = LocalToolRuntime([tool()], agents={"developer": ["notes.search"]})
    remote = FixedRuntime("fs.read_file")
    merged = CompositeToolRuntime(local, remote)
    async with merged.open(agent("developer")) as opened:
        assert [spec.name for spec in opened.tools] == ["notes.search", "fs.read_file"]
        assert (await opened.call(call())).content == "found"
        assert (await opened.call(call("fs.read_file"))).content == "fs.read_file ran"
        unknown = await opened.call(call("fs.write_file"))
        assert (unknown.content, unknown.is_error) == ("Unknown tool 'fs.write_file'", True)
    assert await offered(merged, agent("editor")) == ["fs.read_file"]
    assert remote.closed == 2

    # Local runtimes that define one name twice are refused when composed.
    other = LocalToolRuntime([tool(name="notes.search", version="2")], agents={"editor": ["notes.search"]})
    for runtimes in ((local, other), (local, local), (CompositeToolRuntime(local), remote, other)):
        with pytest.raises(ValueError, match="Tool name 'notes.search' is defined more than once"):
            CompositeToolRuntime(*runtimes)
    # A session that offers a name another one offers is refused when it opens.
    clashing = CompositeToolRuntime(local, FixedRuntime("notes.search"))
    before = clashing._runtimes[1].closed
    with pytest.raises(ValueError, match="'notes.search' is offered by more than one tool runtime"):
        await offered(clashing, agent("developer"))
    assert clashing._runtimes[1].closed == before + 1
    assert await offered(clashing, agent("editor")) == ["notes.search"]

    with pytest.raises(ValueError, match="needs at least one tool runtime"):
        CompositeToolRuntime()
    with pytest.raises(TypeError, match="merges ToolRuntime objects, not LocalTool"):
        CompositeToolRuntime(tool())


@pytest.mark.asyncio
@pytest.mark.parametrize("local, remote", [("fs.read", "fs__read"), ("fs._read", "fs_.read")])
async def test_the_composite_runtime_refuses_names_a_provider_would_see_as_one(local, remote):
    assert wire_name(local) == wire_name(remote)
    merged = CompositeToolRuntime(
        LocalToolRuntime([tool(name=local)], agents={"developer": [local]}), FixedRuntime(remote))
    message = (f"Tool names {local!r} and {remote!r} are both sent to a provider as "
               f"{wire_name(local)!r}")
    with pytest.raises(ValueError, match=re.escape(message)):
        await offered(merged, agent("developer"))
    # A node offered only one of them opens as before.
    assert await offered(merged, agent("editor")) == [remote]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["notes.search", "notes._draft", "notes_search", "notes-search",
                                  "release.notes.search"])
async def test_a_tool_is_called_by_the_name_a_provider_reads_back(name):
    # A provider sends each tool under its wire name and reads a call's name back.
    result = await session(tool(name=name)).call(call(display_name(wire_name(name))))
    assert (result.content, result.is_error) == ("found", False)


def test_describe_records_each_tool_and_changes_with_its_schema():
    runtime = LocalToolRuntime([tool(name="notes.tag", version="3"), tool()], agents={})
    digest = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert runtime.describe() == [
        {"name": "notes.search", "version": "1", "schema_sha256": digest},
        {"name": "notes.tag", "version": "3", "schema_sha256": digest},
    ]
    reordered = {key: SCHEMA[key] for key in reversed(list(SCHEMA))}
    assert LocalTool(SEARCH, lambda arguments: "", "1").describe() == runtime.describe()[0]
    assert LocalTool(ToolSpec("notes.search", "Other words.", reordered), print, "1").describe() == (
        runtime.describe()[0])
    widened = dict(SCHEMA, required=[])
    changed = LocalTool(ToolSpec("notes.search", "Search the release notes.", widened), print, "1")
    assert changed.describe()["schema_sha256"] != digest
    assert {key: value for key, value in changed.describe().items() if key != "schema_sha256"} == {
        "name": "notes.search", "version": "1"}

    # A template record takes what describe() returns as the tools it needs.
    node = {"id": "draft", "label": "Draft the notes", "depends_on": [], "capabilities": [],
            "profile": None, "verifies": None, "max_regenerations": 0, "failure_policy": "halt",
            "max_retries": 0, "timeout_s": None, "attach_dep_artifacts": False, "metadata": {}}
    record = TemplateRecord(name="notes", description="Release notes", blueprint=[node], prerequisites={
        "capabilities": [], "skills": [], "tools": runtime.describe(), "validators": []})
    assert record.to_dict()["prerequisites"]["tools"] == runtime.describe()

    # The tool keeps its own copy: editing the schema later changes neither.
    schema = json.loads(json.dumps(SCHEMA))
    kept = LocalTool(ToolSpec("notes.search", "Search.", schema), print, "1")
    schema["required"].append("until")
    assert kept.spec.input_schema == SCHEMA and kept.describe()["schema_sha256"] == digest


@pytest.mark.asyncio
async def test_result_text_and_error_messages_are_capped():
    capped = session(tool(lambda arguments: "x" * 25), max_result_chars=10)
    result = await capped.call(call())
    assert result.content == "x" * 10 + "\n[15 more characters not shown]"

    def fail(arguments):
        raise ValueError("y" * 25)

    result = await session(tool(fail), max_result_chars=10).call(call())
    assert (result.content, result.is_error) == ("y" * 10 + "\n[15 more characters not shown]", True)
    runtime = LocalToolRuntime([tool(lambda arguments: "z" * 70_000)], agents={"developer": ["notes.search"]})
    async with runtime.open(agent("developer")) as opened:
        content = (await opened.call(call())).content
    assert content == "z" * 65_536 + "\n[4464 more characters not shown]"


INVALID_TOOLS = {
    "spec": (lambda: LocalTool({"name": "x"}, print, "1"), TypeError, "spec must be a ToolSpec"),
    "handler": (lambda: LocalTool(SEARCH, "print", "1"), TypeError, "handler must be callable"),
    "blank-version": (lambda: LocalTool(SEARCH, print, " "), ValueError, "version must be non-blank"),
    "number-version": (lambda: LocalTool(SEARCH, print, 1), ValueError, "version must be non-blank"),
    "schema-list": (lambda: LocalTool(ToolSpec("x", "", []), print, "1"), TypeError, "a JSON Schema object"),
    "schema-nan": (lambda: LocalTool(ToolSpec("x", "", {"maximum": float("nan")}), print, "1"),
                   ValueError, "input_schema value is not canonical JSON data"),
    # A provider would read a call to either back as another name.
    "double-underscore": (lambda: LocalTool(ToolSpec("notes__search", "", SCHEMA), print, "1"),
                          ValueError, re.escape("Tool 'notes__search' could never be called: its "
                                                "wire name 'notes__search' reads back as "
                                                "'notes.search'")),
    "underscore-before-dot": (lambda: LocalTool(ToolSpec("notes_.search", "", SCHEMA), print, "1"),
                              ValueError, re.escape("wire name 'notes___search' reads back as "
                                                    "'notes._search'")),
    **{f"timeout-{label}": (lambda value=value: LocalTool(SEARCH, print, "1", timeout_s=value),
                            ValueError, "timeout_s must be a positive, finite number")
       for label, value in {"zero": 0, "negative": -1, "nan": float("nan"), "infinite": float("inf"),
                            "bool": True, "text": "5", "huge": 10 ** 400}.items()},
}


@pytest.mark.parametrize("make,error,message", list(INVALID_TOOLS.values()), ids=list(INVALID_TOOLS))
def test_invalid_tools_are_refused(make, error, message):
    with pytest.raises(error, match=message):
        make()


INVALID_RUNTIMES = {
    "one-tool": (lambda: LocalToolRuntime(tool(), agents={}), TypeError, "a sequence of LocalTool"),
    "specs": (lambda: LocalToolRuntime([SEARCH], agents={}), TypeError, "a sequence of LocalTool"),
    "duplicate": (lambda: LocalToolRuntime([tool(), tool(version="2")], agents={}), ValueError,
                  "'notes.search' is defined more than once"),
    "agents-list": (lambda: LocalToolRuntime([tool()], agents=["developer"]), TypeError,
                    "agents must map agent ids"),
    "blank-id": (lambda: LocalToolRuntime([tool()], agents={"": ["notes.search"]}), TypeError,
                 "non-empty agent ids"),
    "names-text": (lambda: LocalToolRuntime([tool()], agents={"developer": "notes.search"}), TypeError,
                   r"agents\['developer'\] must be a sequence of tool names"),
    "undefined": (lambda: LocalToolRuntime([tool()], agents={"developer": ["notes.delete"]}), ValueError,
                  r"agents\['developer'\] names undefined tools \['notes.delete'\]"),
    "cap-zero": (lambda: LocalToolRuntime([], agents={}, max_result_chars=0), ValueError,
                 "max_result_chars must be a positive integer"),
    "cap-bool": (lambda: LocalToolRuntime([], agents={}, max_result_chars=True), ValueError,
                 "max_result_chars must be a positive integer"),
}


@pytest.mark.parametrize("make,error,message", list(INVALID_RUNTIMES.values()), ids=list(INVALID_RUNTIMES))
def test_invalid_runtimes_are_refused(make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_public_types_resolve_their_type_hints():
    for kind in (LocalTool, LocalToolRuntime.__init__, CompositeToolRuntime.__init__, python_skill,
                 LocalTool.describe, LocalToolRuntime.describe):
        typing.get_type_hints(kind)
