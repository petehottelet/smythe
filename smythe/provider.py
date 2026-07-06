"""Provider abstraction — async LLM backends for node execution.

Two entry points:

- ``complete(system, prompt, model)`` — single-turn, text-only.  The
  simple path; kept for backwards compatibility and implemented on top
  of ``chat()`` in the built-in providers.
- ``chat(system, messages, model, tools)`` — multi-turn and tool-aware.
  The tool loop (executor layer) drives this.  Custom providers that
  only implement ``complete()`` keep working for tool-less execution
  via the base-class default.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4

from smythe.tools import ChatMessage, ToolCall, ToolSpec, display_name, wire_name


@dataclass
class CompletionResult:
    """Response from a provider call, including token usage for budget tracking."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(ABC):
    """Abstract base for LLM providers.

    All providers are async-native so the parallel executor can
    fan out calls concurrently.  The serial executor wraps calls
    with asyncio.run().
    """

    @abstractmethod
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        """Send a prompt to the LLM and return text + token usage."""

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        """Multi-turn, tool-aware completion.

        The default delegates the simple case (no tools, one plain user
        message) to complete(), so custom providers that predate the
        tool contract keep working for tool-less execution.
        """
        if (
            not tools
            and len(messages) == 1
            and messages[0].role == "user"
            and not messages[0].tool_results
        ):
            return await self.complete(system, messages[0].content, model)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement tool-aware chat(). "
            "Override chat() to support tools or multi-turn messages."
        )


class OfflineProvider(Provider):
    """Deterministic provider that runs entirely offline.

    No API keys, no network, stable output — evaluate the full
    framework (or run demos and CI) without spending a token:

    - Planning calls (recognized by the planning system prompt) return
      ``plan`` as JSON, so the full Architect -> executor -> synthesizer
      pipeline runs offline.
    - Other calls consume ``responses`` in order (the last one repeats),
      or echo the prompt's first line when no script is given.
    """

    def __init__(
        self,
        *,
        plan: dict | None = None,
        responses: list[str] | None = None,
        echo_prefix: str = "offline: ",
    ) -> None:
        self._plan = plan
        self._responses = list(responses) if responses else []
        self._echo_prefix = echo_prefix
        self._cursor = 0
        self.calls: list[str] = []  # first line of every prompt, for assertions

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        from smythe.prompts import PLANNING_SYSTEM_PROMPT

        # Root nodes carry the overall task above their label; echo the
        # step label so offline output still names the work being done.
        step_line = next(
            (line for line in prompt.splitlines() if line.startswith("Your step: ")),
            None,
        )
        first_line = (
            step_line.removeprefix("Your step: ") if step_line
            else prompt.split("\n")[0]
        )
        self.calls.append(first_line)

        if self._plan is not None and system == PLANNING_SYSTEM_PROMPT:
            return CompletionResult(
                text=json.dumps(self._plan), prompt_tokens=250, completion_tokens=400,
            )
        if self._responses:
            text = self._responses[min(self._cursor, len(self._responses) - 1)]
            self._cursor += 1
        else:
            text = f"{self._echo_prefix}{first_line[:60]}"
        return CompletionResult(text=text, prompt_tokens=40, completion_tokens=60)


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API."""

    def __init__(self, *, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "Install the anthropic extra: pip install smythe[anthropic]"
                ) from exc
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        client = self._get_client()

        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {
                    "name": wire_name(t.name),
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = await client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            messages=self._to_api_messages(messages),
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=display_name(block.name),
                        arguments=dict(block.input or {}),
                    )
                )

        stop = getattr(response, "stop_reason", None)
        if not isinstance(stop, str):
            stop = "tool_use" if tool_calls else "end_turn"

        return CompletionResult(
            text="\n".join(text_parts),
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
            tool_calls=tool_calls,
            stop_reason=stop,
        )

    @staticmethod
    def _to_api_messages(messages: list[ChatMessage]) -> list[dict]:
        api_messages: list[dict] = []
        for m in messages:
            if m.role == "assistant":
                content: list[dict] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": wire_name(tc.name),
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content})
            elif m.tool_results:
                # All results for a turn go in ONE user message — splitting
                # them degrades the model's parallel tool calling.
                content = []
                for tr in m.tool_results:
                    block = {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                    if tr.is_error:
                        block["is_error"] = True
                    content.append(block)
                if m.content:
                    content.append({"type": "text", "text": m.content})
                api_messages.append({"role": "user", "content": content})
            else:
                api_messages.append({"role": "user", "content": m.content})
        return api_messages


class OpenAIProvider(Provider):
    """Provider backed by the OpenAI Chat Completions API.

    Pass ``base_url`` to target any OpenAI-compatible endpoint —
    Ollama (http://localhost:11434/v1), LM Studio, vLLM, etc.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "Install the openai extra: pip install smythe[openai]"
                ) from exc
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        client = self._get_client()

        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": wire_name(t.name),
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        response = await client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=self._to_api_messages(system, messages),
            **kwargs,
        )

        text = ""
        tool_calls: list[ToolCall] = []
        stop = "end_turn"
        if response.choices:
            message = response.choices[0].message
            text = message.content or ""
            raw_calls = getattr(message, "tool_calls", None)
            if isinstance(raw_calls, (list, tuple)):
                for rc in raw_calls:
                    fn = rc.function
                    try:
                        arguments = json.loads(fn.arguments) if fn.arguments else {}
                        if not isinstance(arguments, dict):
                            arguments = {"_raw_arguments": fn.arguments}
                    except (json.JSONDecodeError, TypeError):
                        # Malformed model output must never crash the loop;
                        # the tool sees the raw string and can error cleanly.
                        arguments = {"_raw_arguments": fn.arguments}
                    tool_calls.append(
                        ToolCall(id=rc.id, name=display_name(fn.name), arguments=arguments)
                    )
            finish = getattr(response.choices[0], "finish_reason", None)
            stop = {
                "stop": "end_turn",
                "tool_calls": "tool_use",
                "length": "max_tokens",
            }.get(finish, "tool_use" if tool_calls else "end_turn")

        usage = response.usage
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            tool_calls=tool_calls,
            stop_reason=stop,
        )

    @staticmethod
    def _to_api_messages(system: str, messages: list[ChatMessage]) -> list[dict]:
        api_messages: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "assistant":
                entry: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": wire_name(tc.name),
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                api_messages.append(entry)
            elif m.tool_results:
                # OpenAI wants one role:"tool" message per result, keyed by id.
                # There is no error flag; errors are conveyed in the content.
                for tr in m.tool_results:
                    content = f"ERROR: {tr.content}" if tr.is_error else tr.content
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": content,
                    })
                if m.content:
                    api_messages.append({"role": "user", "content": m.content})
            else:
                api_messages.append({"role": "user", "content": m.content})
        return api_messages


class GeminiProvider(Provider):
    """Provider backed by the Google Gen AI SDK (Gemini models)."""

    def __init__(self, *, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "Install the gemini extra: pip install smythe[gemini]"
                ) from exc
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = genai.Client(**kwargs)
        return self._client

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        client = self._get_client()

        config: dict = {
            "system_instruction": system,
            "max_output_tokens": self._max_tokens,
        }
        if tools:
            config["tools"] = [{
                "function_declarations": [
                    {
                        "name": wire_name(t.name),
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                    for t in tools
                ],
            }]

        response = await client.aio.models.generate_content(
            model=model,
            contents=self._to_api_contents(messages),
            config=config,
        )

        text = getattr(response, "text", None)
        text = text if isinstance(text, str) else ""

        # Gemini function calls carry no ids — synthesize them so the
        # neutral contract holds; results are matched by name on the way
        # back (see _to_api_contents).
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(response, "function_calls", None)
        if isinstance(raw_calls, (list, tuple)):
            for rc in raw_calls:
                name = getattr(rc, "name", "") or ""
                args = getattr(rc, "args", None)
                tool_calls.append(
                    ToolCall(
                        id=f"gemini-{uuid4().hex[:8]}",
                        name=display_name(name),
                        arguments=dict(args) if args else {},
                    )
                )

        usage = response.usage_metadata
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    @staticmethod
    def _to_api_contents(messages: list[ChatMessage]) -> list[dict]:
        # Gemini matches function responses to calls by name and order,
        # not by id, so the ToolCall id is dropped here.  The loop keeps
        # results in call order, which preserves the pairing.
        contents: list[dict] = []
        call_names: dict[str, str] = {}
        for m in messages:
            if m.role == "assistant":
                parts: list[dict] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls:
                    call_names[tc.id] = wire_name(tc.name)
                    parts.append({
                        "function_call": {
                            "name": wire_name(tc.name),
                            "args": tc.arguments,
                        },
                    })
                contents.append({"role": "model", "parts": parts})
            elif m.tool_results:
                parts = []
                for tr in m.tool_results:
                    name = call_names.get(tr.tool_call_id, tr.tool_call_id)
                    payload = (
                        {"error": tr.content} if tr.is_error else {"output": tr.content}
                    )
                    parts.append({
                        "function_response": {"name": name, "response": payload},
                    })
                if m.content:
                    parts.append({"text": m.content})
                contents.append({"role": "user", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": m.content}]})
        return contents
