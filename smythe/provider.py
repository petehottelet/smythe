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

import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4

from smythe.tools import ChatMessage, ToolCall, ToolSpec, display_name, wire_name

_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


@dataclass
class Artifact:
    """A binary output from a provider call (e.g. a generated image).

    Artifacts travel on the CompletionResult only; the executor persists
    them to disk and records paths on the node, so raw bytes never enter
    checkpoints or planner memory.
    """

    data: bytes
    mime_type: str = "image/png"

    @property
    def suffix(self) -> str:
        """File extension for this artifact's mime type ('.bin' if unknown)."""
        return _MIME_SUFFIXES.get(self.mime_type, ".bin")


@dataclass
class CompletionResult:
    """Response from a provider call, including token usage for budget tracking."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    artifacts: list[Artifact] = field(default_factory=list)
    cost_usd: float | None = None
    """Explicit cost for this call, when the provider can price it better
    than the Sentinel's blended token rate (e.g. per-image billing).
    None means cost is derived from token counts as usual."""

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
            and not messages[0].attachments
        ):
            return await self.complete(system, messages[0].content, model)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement tool-aware chat(). "
            "Override chat() to support tools, attachments, or multi-turn "
            "messages."
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
    - With ``artifacts_per_call`` > 0, every non-planning call also
      returns that many deterministic 1x1 PNG artifacts, so image
      pipelines can be exercised offline in CI.
    """

    def __init__(
        self,
        *,
        plan: dict | None = None,
        responses: list[str] | None = None,
        echo_prefix: str = "offline: ",
        artifacts_per_call: int = 0,
    ) -> None:
        self._plan = plan
        self._responses = list(responses) if responses else []
        self._echo_prefix = echo_prefix
        self._artifacts_per_call = artifacts_per_call
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
        artifacts = [
            Artifact(data=_OFFLINE_PNG, mime_type="image/png")
            for _ in range(self._artifacts_per_call)
        ]
        return CompletionResult(
            text=text, prompt_tokens=40, completion_tokens=60, artifacts=artifacts,
        )

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        """Attachment-tolerant chat: vision nodes stay runnable offline.

        Attached images are acknowledged in the deterministic text so
        tests can assert the multimodal path was exercised.
        """
        if (
            not tools
            and len(messages) == 1
            and messages[0].role == "user"
            and not messages[0].tool_results
        ):
            result = await self.complete(system, messages[0].content, model)
            if messages[0].attachments:
                result.text += f" [saw {len(messages[0].attachments)} image(s)]"
            return result
        return await super().chat(system, messages, model, tools)


# A valid 1x1 transparent PNG — the deterministic artifact payload for
# OfflineProvider image runs.
_OFFLINE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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
            elif m.attachments:
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": a.mime_type,
                            "data": base64.b64encode(a.data).decode(),
                        },
                    }
                    for a in m.attachments
                ]
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

        # Current OpenAI models reject the legacy max_tokens parameter;
        # max_completion_tokens is its documented replacement.
        response = await client.chat.completions.create(
            model=model,
            max_completion_tokens=self._max_tokens,
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
            elif m.attachments:
                parts: list[dict] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{a.mime_type};base64,"
                                f"{base64.b64encode(a.data).decode()}"
                            ),
                        },
                    }
                    for a in m.attachments
                ]
                if m.content:
                    parts.append({"type": "text", "text": m.content})
                api_messages.append({"role": "user", "content": parts})
            else:
                api_messages.append({"role": "user", "content": m.content})
        return api_messages


class OpenAIImageProvider(OpenAIProvider):
    """OpenAI Image API provider that returns generated files as artifacts.

    This provider is intentionally separate from :class:`OpenAIProvider`:
    GPT Image models use ``images.generate`` rather than a text completion
    endpoint, and image output has its own size, quality, format, and billing
    controls.  ``cost_per_image_usd`` is an optional caller-supplied estimate
    for the selected model/quality/size; when omitted, the Sentinel falls back
    to the usage-token fields returned by the API.
    """

    _OUTPUT_MIME_TYPES = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        size: str = "auto",
        quality: str = "medium",
        output_format: str = "png",
        output_compression: int | None = None,
        moderation: str = "auto",
        n: int = 1,
        cost_per_image_usd: float | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        if not 1 <= n <= 10:
            raise ValueError(f"n must be between 1 and 10, got {n}")
        if output_format not in self._OUTPUT_MIME_TYPES:
            raise ValueError(
                "output_format must be one of "
                f"{sorted(self._OUTPUT_MIME_TYPES)}, got {output_format!r}"
            )
        if quality not in {"low", "medium", "high", "auto"}:
            raise ValueError(
                "quality must be one of ['auto', 'high', 'low', 'medium'], "
                f"got {quality!r}"
            )
        if moderation not in {"auto", "low"}:
            raise ValueError(
                f"moderation must be one of ['auto', 'low'], got {moderation!r}"
            )
        if output_compression is not None and not 0 <= output_compression <= 100:
            raise ValueError(
                "output_compression must be between 0 and 100, "
                f"got {output_compression}"
            )
        if output_compression is not None and output_format == "png":
            raise ValueError("output_compression is supported only for JPEG or WebP")
        if cost_per_image_usd is not None and cost_per_image_usd < 0:
            raise ValueError(
                f"cost_per_image_usd must be non-negative, got {cost_per_image_usd}"
            )

        self._image_size = size
        self._image_quality = quality
        self._image_output_format = output_format
        self._image_output_compression = output_compression
        self._image_moderation = moderation
        self._images_per_call = n
        self._cost_per_image_usd = cost_per_image_usd

    @property
    def cost_estimate_per_call(self) -> float | None:
        """Reservation hint for parallel budgeting (n images x per-image price)."""
        if self._cost_per_image_usd is None:
            return None
        return self._images_per_call * self._cost_per_image_usd

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        if not model.lower().startswith("gpt-image-"):
            raise ValueError(
                "OpenAIImageProvider requires a GPT Image model "
                f"(gpt-image-*), got {model!r}"
            )
        if tools:
            raise ValueError("OpenAIImageProvider does not support tool calls")

        content = "\n\n".join(m.content for m in messages if m.content).strip()
        if not content:
            raise ValueError("OpenAIImageProvider requires a non-empty image prompt")
        prompt = f"{system.strip()}\n\n{content}" if system.strip() else content

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": self._images_per_call,
            "size": self._image_size,
            "quality": self._image_quality,
            "output_format": self._image_output_format,
            "moderation": self._image_moderation,
        }
        if self._image_output_compression is not None:
            payload["output_compression"] = self._image_output_compression

        response = await self._get_client().images.generate(**payload)
        mime_type = self._OUTPUT_MIME_TYPES[self._image_output_format]
        artifacts: list[Artifact] = []
        for item in getattr(response, "data", None) or []:
            encoded = getattr(item, "b64_json", None)
            if encoded:
                artifacts.append(
                    Artifact(data=base64.b64decode(encoded), mime_type=mime_type)
                )
        if not artifacts:
            raise RuntimeError("OpenAI Image API response contained no base64 image data")

        usage = getattr(response, "usage", None)
        cost_usd = None
        if self._cost_per_image_usd is not None:
            cost_usd = len(artifacts) * self._cost_per_image_usd
        return CompletionResult(
            text=f"Generated {len(artifacts)} image artifact(s) with {model}.",
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            artifacts=artifacts,
            cost_usd=cost_usd,
        )


class GeminiProvider(Provider):
    """Provider backed by the Google Gen AI SDK (Gemini models).

    Image generation: for image-output models (``gemini-*-image*``),
    ``response_modalities`` is requested automatically and returned
    inline images become ``CompletionResult.artifacts``.  Pass
    ``cost_per_image_usd`` to bill those calls per image instead of the
    Sentinel's blended token rate — image models price output per image,
    which token math badly underestimates.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        response_modalities: list[str] | None = None,
        cost_per_image_usd: float | None = None,
        image_config: dict | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._max_tokens = max_tokens
        self._response_modalities = response_modalities
        self._cost_per_image_usd = cost_per_image_usd
        self._image_config = dict(image_config) if image_config else None
        self._client = None

    @property
    def cost_estimate_per_call(self) -> float | None:
        """Reservation hint for parallel budgeting: per-image price when set.

        The AsyncExecutor consults this so budget reservations reflect
        per-image billing instead of the blended token estimate.
        """
        return self._cost_per_image_usd

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
        modalities = self._response_modalities
        # Auto-detect stays off when tools are in play: Gemini rejects
        # function declarations combined with IMAGE output. Explicit
        # response_modalities are still honored (caller's responsibility).
        if modalities is None and not tools and any(
            hint in model.lower() for hint in ("image", "banana")
        ):
            modalities = ["TEXT", "IMAGE"]
        if modalities:
            config["response_modalities"] = list(modalities)
        if self._image_config:
            config["image_config"] = dict(self._image_config)
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

        artifacts = self._extract_artifacts(response)
        cost_usd = None
        if artifacts and self._cost_per_image_usd is not None:
            cost_usd = len(artifacts) * self._cost_per_image_usd

        usage = response.usage_metadata
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            artifacts=artifacts,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _extract_artifacts(response) -> list[Artifact]:
        """Collect inline image parts from the response, defensively.

        ``response.text`` concatenates only text parts, so image bytes
        are reachable solely through the candidate parts.
        """
        artifacts: list[Artifact] = []
        candidates = getattr(response, "candidates", None)
        if not isinstance(candidates, (list, tuple)) or not candidates:
            return artifacts
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None)
        if not isinstance(parts, (list, tuple)):
            return artifacts
        for part in parts:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None)
            if data is None:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)
            mime = getattr(blob, "mime_type", None) or "application/octet-stream"
            artifacts.append(Artifact(data=data, mime_type=mime))
        return artifacts

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
                parts = [
                    {"inline_data": {"mime_type": a.mime_type, "data": a.data}}
                    for a in m.attachments
                ]
                if m.content or not parts:
                    parts.append({"text": m.content})
                contents.append({"role": "user", "parts": parts})
        return contents
