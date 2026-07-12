"""Tests for the Provider abstraction and auto-detection."""

import asyncio
import os

import pytest

from smythe.provider import (
    AnthropicProvider,
    CompletionResult,
    GeminiProvider,
    OpenAIImageProvider,
    OpenAIProvider,
    Provider,
)
from smythe.swarm import Swarm, _auto_detect_provider


class MockProvider(Provider):
    """Test provider that returns a canned response."""

    def __init__(self, response: str = "mock response") -> None:
        self._response = response

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult(text=self._response, prompt_tokens=5, completion_tokens=3)


def test_mock_provider_implements_abc():
    p = MockProvider()
    assert isinstance(p, Provider)


def test_mock_provider_complete():
    p = MockProvider("hello")
    result = asyncio.run(p.complete("sys", "prompt", "model"))
    assert isinstance(result, CompletionResult)
    assert result.text == "hello"
    assert result.total_tokens == 8


def test_anthropic_provider_constructable():
    p = AnthropicProvider(api_key="test-key")
    assert isinstance(p, Provider)


def test_openai_provider_constructable():
    p = OpenAIProvider(api_key="test-key")
    assert isinstance(p, Provider)


def test_auto_detect_claude():
    p = _auto_detect_provider("claude-opus-4-8")
    assert isinstance(p, AnthropicProvider)


def test_auto_detect_gpt():
    p = _auto_detect_provider("gpt-4o")
    assert isinstance(p, OpenAIProvider)


def test_auto_detect_gpt_image():
    p = _auto_detect_provider("gpt-image-2")
    assert isinstance(p, OpenAIImageProvider)


def test_auto_detect_o1():
    p = _auto_detect_provider("o1-preview")
    assert isinstance(p, OpenAIProvider)


def test_auto_detect_gemini():
    p = _auto_detect_provider("gemini-3-pro-image-preview")
    assert isinstance(p, GeminiProvider)


def test_auto_detect_unknown_raises():
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        _auto_detect_provider("llama-3-70b")


def test_swarm_accepts_explicit_provider():
    mock = MockProvider()
    swarm = Swarm(model="anything-custom", provider=mock)
    assert swarm._provider is mock


def test_anthropic_guard_empty_content():
    """AnthropicProvider handles empty content blocks without crashing."""
    from unittest.mock import AsyncMock, MagicMock

    p = AnthropicProvider(api_key="fake")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "model"))
    assert result.text == ""
    assert result.prompt_tokens == 10


def test_anthropic_guard_non_text_content():
    """AnthropicProvider skips non-text content blocks."""
    from unittest.mock import AsyncMock, MagicMock

    p = AnthropicProvider(api_key="fake")
    mock_client = MagicMock()
    tool_block = MagicMock(spec=[])
    text_block = MagicMock()
    text_block.text = "actual text"
    mock_response = MagicMock()
    mock_response.content = [tool_block, text_block]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "model"))
    assert result.text == "actual text"


def test_openai_guard_empty_choices():
    """OpenAIProvider handles empty choices without crashing."""
    from unittest.mock import AsyncMock, MagicMock

    p = OpenAIProvider(api_key="fake")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "model"))
    assert result.text == ""
    assert result.prompt_tokens == 10


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_anthropic_integration():
    p = AnthropicProvider()
    result = asyncio.run(p.complete("You are a test.", "Say hello.", "claude-sonnet-4-20250514"))
    assert isinstance(result, CompletionResult)
    assert len(result.text) > 0
    assert result.prompt_tokens > 0


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_integration():
    p = OpenAIProvider()
    result = asyncio.run(p.complete("You are a test.", "Say hello.", "gpt-4o-mini"))
    assert isinstance(result, CompletionResult)
    assert len(result.text) > 0
    assert result.prompt_tokens > 0


def test_gemini_provider_constructable():
    p = GeminiProvider(api_key="test-key")
    assert isinstance(p, Provider)


def test_gemini_provider_missing_sdk():
    """GeminiProvider raises an actionable ImportError when the SDK is missing."""
    import unittest.mock as um

    p = GeminiProvider(api_key="fake")
    with um.patch.dict("sys.modules", {"google": None, "google.genai": None}):
        p._client = None
        with pytest.raises(ImportError, match="pip install smythe\\[gemini\\]"):
            p._get_client()


def test_gemini_provider_complete():
    """GeminiProvider returns text and token fields from a mocked SDK."""
    from unittest.mock import AsyncMock, MagicMock

    p = GeminiProvider(api_key="fake")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "gemini says hello"
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=12,
        candidates_token_count=7,
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "gemini-3-pro"))
    assert result.text == "gemini says hello"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7
    assert result.total_tokens == 19


def test_gemini_provider_no_usage_metadata():
    """GeminiProvider handles missing usage_metadata gracefully."""
    from unittest.mock import AsyncMock, MagicMock

    p = GeminiProvider(api_key="fake")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "response"
    mock_response.usage_metadata = None
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "gemini-3-pro"))
    assert result.text == "response"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_gemini_provider_empty_text():
    """GeminiProvider handles None text in the response."""
    from unittest.mock import AsyncMock, MagicMock

    p = GeminiProvider(api_key="fake")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=5,
        candidates_token_count=0,
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    p._client = mock_client

    result = asyncio.run(p.complete("sys", "prompt", "gemini-3-pro"))
    assert result.text == ""
    assert result.prompt_tokens == 5


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
def test_gemini_integration():
    p = GeminiProvider()
    result = asyncio.run(p.complete("You are a test.", "Say hello.", "gemini-2.5-flash"))
    assert isinstance(result, CompletionResult)
    assert len(result.text) > 0
    assert result.prompt_tokens > 0


# ---------------------------------------------------------------------------
# Tool-aware chat()
# ---------------------------------------------------------------------------

from smythe.tools import ChatMessage, ToolCall, ToolResult, ToolSpec  # noqa: E402

WEATHER_TOOL = ToolSpec(
    name="wx.get_weather",
    description="Get the weather",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)


def test_base_chat_delegates_simple_case_to_complete():
    p = MockProvider("delegated")
    result = asyncio.run(p.chat("sys", [ChatMessage(role="user", content="hi")], "m"))
    assert result.text == "delegated"


def test_base_chat_with_tools_raises_not_implemented():
    p = MockProvider()
    with pytest.raises(NotImplementedError, match="tool-aware"):
        asyncio.run(
            p.chat("sys", [ChatMessage(role="user", content="hi")], "m", tools=[WEATHER_TOOL])
        )


def _anthropic_with_mock(response):
    from unittest.mock import AsyncMock, MagicMock

    p = AnthropicProvider(api_key="fake")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=response)
    p._client = mock_client
    return p, mock_client


def _make_anthropic_response(blocks, stop_reason="end_turn"):
    from unittest.mock import MagicMock

    response = MagicMock()
    response.content = blocks
    response.stop_reason = stop_reason
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    return response


def test_anthropic_chat_parses_tool_use_blocks():
    from unittest.mock import MagicMock

    tool_block = MagicMock(spec=["type", "id", "name", "input"])
    tool_block.type = "tool_use"
    tool_block.id = "toolu_1"
    tool_block.name = "wx__get_weather"
    tool_block.input = {"city": "Paris"}

    p, _ = _anthropic_with_mock(_make_anthropic_response([tool_block], "tool_use"))
    result = asyncio.run(
        p.chat("sys", [ChatMessage(role="user", content="weather?")], "m", tools=[WEATHER_TOOL])
    )

    assert result.stop_reason == "tool_use"
    [call] = result.tool_calls
    assert call.id == "toolu_1"
    assert call.name == "wx.get_weather"       # display name restored
    assert call.arguments == {"city": "Paris"}


def test_anthropic_chat_sends_wire_names_and_schema():
    p, mock_client = _anthropic_with_mock(_make_anthropic_response([]))
    asyncio.run(
        p.chat("sys", [ChatMessage(role="user", content="q")], "m", tools=[WEATHER_TOOL])
    )
    sent = mock_client.messages.create.call_args.kwargs
    [tool] = sent["tools"]
    assert tool["name"] == "wx__get_weather"
    assert tool["input_schema"]["type"] == "object"


def test_anthropic_tool_results_go_in_one_user_message():
    p, mock_client = _anthropic_with_mock(_make_anthropic_response([]))
    messages = [
        ChatMessage(role="user", content="weather in two cities?"),
        ChatMessage(role="assistant", tool_calls=[
            ToolCall(id="t1", name="wx.get_weather", arguments={"city": "Paris"}),
            ToolCall(id="t2", name="wx.get_weather", arguments={"city": "Oslo"}),
        ]),
        ChatMessage(role="user", tool_results=[
            ToolResult(tool_call_id="t1", content="20C"),
            ToolResult(tool_call_id="t2", content="5C", is_error=True),
        ]),
    ]
    asyncio.run(p.chat("sys", messages, "m", tools=[WEATHER_TOOL]))

    sent = mock_client.messages.create.call_args.kwargs["messages"]
    assert len(sent) == 3
    result_msg = sent[2]
    assert result_msg["role"] == "user"
    assert [b["type"] for b in result_msg["content"]] == ["tool_result", "tool_result"]
    assert result_msg["content"][1]["is_error"] is True
    assert "is_error" not in result_msg["content"][0]


def _openai_with_mock(response):
    from unittest.mock import AsyncMock, MagicMock

    p = OpenAIProvider(api_key="fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    p._client = mock_client
    return p, mock_client


def _make_openai_response(content=None, tool_calls=None, finish_reason="stop"):
    from unittest.mock import MagicMock

    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _openai_tool_call(call_id, name, arguments_json):
    from unittest.mock import MagicMock

    rc = MagicMock()
    rc.id = call_id
    rc.function = MagicMock()
    rc.function.name = name           # NB: name= kwarg would configure the mock itself
    rc.function.arguments = arguments_json
    return rc


def test_openai_chat_parses_tool_calls():
    rc = _openai_tool_call("call_1", "wx__get_weather", '{"city": "Paris"}')
    p, _ = _openai_with_mock(_make_openai_response(tool_calls=[rc], finish_reason="tool_calls"))
    result = asyncio.run(
        p.chat("sys", [ChatMessage(role="user", content="q")], "m", tools=[WEATHER_TOOL])
    )
    assert result.stop_reason == "tool_use"
    [call] = result.tool_calls
    assert call.name == "wx.get_weather"
    assert call.arguments == {"city": "Paris"}


def test_openai_malformed_arguments_never_raise():
    rc = _openai_tool_call("call_1", "wx__get_weather", '{"city": TRUNC')
    p, _ = _openai_with_mock(_make_openai_response(tool_calls=[rc], finish_reason="tool_calls"))
    result = asyncio.run(
        p.chat("sys", [ChatMessage(role="user", content="q")], "m", tools=[WEATHER_TOOL])
    )
    [call] = result.tool_calls
    assert call.arguments == {"_raw_arguments": '{"city": TRUNC'}


def test_openai_tool_results_become_tool_role_messages():
    p, mock_client = _openai_with_mock(_make_openai_response(content="done"))
    messages = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", tool_calls=[
            ToolCall(id="call_1", name="wx.get_weather", arguments={"city": "Paris"}),
        ]),
        ChatMessage(role="user", tool_results=[
            ToolResult(tool_call_id="call_1", content="boom", is_error=True),
        ]),
    ]
    asyncio.run(p.chat("sys", messages, "m", tools=[WEATHER_TOOL]))

    sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assistant = sent[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "wx__get_weather"
    tool_msg = sent[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "ERROR: boom"


def _gemini_with_mock(response):
    from unittest.mock import AsyncMock, MagicMock

    p = GeminiProvider(api_key="fake")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    p._client = mock_client
    return p, mock_client


def test_gemini_chat_synthesizes_call_ids():
    from unittest.mock import MagicMock

    fc = MagicMock()
    fc.name = "wx__get_weather"
    fc.args = {"city": "Paris"}
    response = MagicMock()
    response.text = None
    response.function_calls = [fc]
    response.usage_metadata = MagicMock(prompt_token_count=8, candidates_token_count=3)

    p, _ = _gemini_with_mock(response)
    result = asyncio.run(
        p.chat("sys", [ChatMessage(role="user", content="q")], "m", tools=[WEATHER_TOOL])
    )

    assert result.stop_reason == "tool_use"
    [call] = result.tool_calls
    assert call.id.startswith("gemini-")
    assert call.name == "wx.get_weather"
    assert call.arguments == {"city": "Paris"}


def test_gemini_tool_results_matched_by_name():
    from unittest.mock import MagicMock

    response = MagicMock()
    response.text = "done"
    response.function_calls = None
    response.usage_metadata = None

    p, mock_client = _gemini_with_mock(response)
    messages = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", tool_calls=[
            ToolCall(id="gemini-abc", name="wx.get_weather", arguments={"city": "Paris"}),
        ]),
        ChatMessage(role="user", tool_results=[
            ToolResult(tool_call_id="gemini-abc", content="20C"),
        ]),
    ]
    asyncio.run(p.chat("sys", messages, "m", tools=[WEATHER_TOOL]))

    sent = mock_client.aio.models.generate_content.call_args.kwargs
    declarations = sent["config"]["tools"][0]["function_declarations"]
    assert declarations[0]["name"] == "wx__get_weather"
    contents = sent["contents"]
    fr = contents[2]["parts"][0]["function_response"]
    assert fr["name"] == "wx__get_weather"     # id resolved back to wire name
    assert fr["response"] == {"output": "20C"}


def test_openai_provider_accepts_base_url():
    p = OpenAIProvider(api_key="k", base_url="http://localhost:11434/v1")
    assert p._base_url == "http://localhost:11434/v1"


def test_openai_provider_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    p = OpenAIProvider(api_key="k")
    assert p._base_url == "http://localhost:1234/v1"


def test_openai_provider_no_base_url_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    p = OpenAIProvider(api_key="k")
    assert p._base_url is None


# ---------------------------------------------------------------------------
# Artifacts (image generation)
# ---------------------------------------------------------------------------

from smythe.provider import Artifact, OfflineProvider  # noqa: E402


def test_artifact_suffix_mapping():
    assert Artifact(data=b"x", mime_type="image/png").suffix == ".png"
    assert Artifact(data=b"x", mime_type="image/jpeg").suffix == ".jpg"
    assert Artifact(data=b"x", mime_type="application/x-unknown").suffix == ".bin"


def test_completion_result_defaults_have_no_artifacts_or_cost():
    r = CompletionResult(text="hi")
    assert r.artifacts == []
    assert r.cost_usd is None


def test_offline_provider_artifacts_per_call():
    p = OfflineProvider(artifacts_per_call=2)
    result = asyncio.run(p.complete("sys", "make images", "demo"))
    assert len(result.artifacts) == 2
    # Deterministic, valid PNG payload (magic bytes).
    assert result.artifacts[0].data.startswith(b"\x89PNG")
    assert result.artifacts[0].mime_type == "image/png"


def test_offline_provider_no_artifacts_by_default():
    p = OfflineProvider()
    result = asyncio.run(p.complete("sys", "plain text", "demo"))
    assert result.artifacts == []


def _image_response(images: int = 1, text: str = "made it"):
    """Duck-typed Gemini response carrying inline image parts."""
    from unittest.mock import MagicMock

    parts = []
    for _ in range(images):
        blob = MagicMock()
        blob.data = b"\x89PNG-fake-bytes"
        blob.mime_type = "image/png"
        part = MagicMock()
        part.inline_data = blob
        parts.append(part)
    text_part = MagicMock()
    text_part.inline_data = None
    parts.append(text_part)

    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.text = text
    response.candidates = [candidate]
    response.function_calls = None
    response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=1290)
    return response


def _gemini_with_response(response, **kwargs) -> GeminiProvider:
    from unittest.mock import AsyncMock, MagicMock

    p = GeminiProvider(api_key="fake", **kwargs)
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    p._client = mock_client
    return p


def test_gemini_extracts_image_artifacts():
    p = _gemini_with_response(_image_response(images=2))
    result = asyncio.run(p.complete("sys", "two images please", "gemini-2.5-flash-image"))
    assert len(result.artifacts) == 2
    assert result.artifacts[0].data == b"\x89PNG-fake-bytes"
    assert result.artifacts[0].mime_type == "image/png"
    assert result.text == "made it"


def test_gemini_auto_requests_image_modalities_for_image_models():
    p = _gemini_with_response(_image_response())
    asyncio.run(p.complete("sys", "prompt", "gemini-2.5-flash-image"))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert sent["config"]["response_modalities"] == ["TEXT", "IMAGE"]


def test_gemini_no_modalities_for_text_models():
    p = _gemini_with_response(_image_response(images=0))
    asyncio.run(p.complete("sys", "prompt", "gemini-3-flash"))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert "response_modalities" not in sent["config"]


def test_gemini_explicit_modalities_override():
    p = _gemini_with_response(_image_response(), response_modalities=["IMAGE"])
    asyncio.run(p.complete("sys", "prompt", "gemini-3-flash"))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert sent["config"]["response_modalities"] == ["IMAGE"]


def test_gemini_per_image_cost():
    p = _gemini_with_response(_image_response(images=3), cost_per_image_usd=0.039)
    result = asyncio.run(p.complete("sys", "prompt", "gemini-2.5-flash-image"))
    assert result.cost_usd == pytest.approx(3 * 0.039)


def test_gemini_no_explicit_cost_without_price_or_images():
    p = _gemini_with_response(_image_response(images=0), cost_per_image_usd=0.039)
    result = asyncio.run(p.complete("sys", "prompt", "gemini-2.5-flash-image"))
    assert result.cost_usd is None
    p2 = _gemini_with_response(_image_response(images=2))
    result2 = asyncio.run(p2.complete("sys", "prompt", "gemini-2.5-flash-image"))
    assert result2.cost_usd is None


def test_gemini_decodes_base64_string_inline_data():
    import base64

    response = _image_response(images=1)
    blob = response.candidates[0].content.parts[0].inline_data
    blob.data = base64.b64encode(b"raw-bytes").decode()
    p = _gemini_with_response(response)
    result = asyncio.run(p.complete("sys", "prompt", "gemini-2.5-flash-image"))
    assert result.artifacts[0].data == b"raw-bytes"


def test_gemini_auto_modalities_for_nano_banana_models():
    p = _gemini_with_response(_image_response())
    asyncio.run(p.complete("sys", "prompt", "nano-banana-pro-preview"))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert sent["config"]["response_modalities"] == ["TEXT", "IMAGE"]


# ---------------------------------------------------------------------------
# OpenAI Image API
# ---------------------------------------------------------------------------


def _openai_image_response(images: int = 1):
    import base64
    from unittest.mock import MagicMock

    response = MagicMock()
    response.data = []
    for index in range(images):
        item = MagicMock()
        item.b64_json = base64.b64encode(f"image-{index}".encode()).decode()
        response.data.append(item)
    response.usage = MagicMock(input_tokens=11, output_tokens=22)
    return response


def _openai_image_with_response(response, **kwargs) -> OpenAIImageProvider:
    from unittest.mock import AsyncMock, MagicMock

    provider = OpenAIImageProvider(api_key="fake", **kwargs)
    client = MagicMock()
    client.images.generate = AsyncMock(return_value=response)
    provider._client = client
    return provider


def test_openai_image_provider_extracts_artifacts_usage_and_cost():
    provider = _openai_image_with_response(
        _openai_image_response(images=2),
        size="1536x1024",
        quality="medium",
        output_format="jpeg",
        output_compression=90,
        n=2,
        cost_per_image_usd=0.041,
    )

    result = asyncio.run(
        provider.complete("creative director", "Golden Gate Bridge", "gpt-image-2")
    )

    assert [artifact.data for artifact in result.artifacts] == [b"image-0", b"image-1"]
    assert all(artifact.mime_type == "image/jpeg" for artifact in result.artifacts)
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 22
    assert result.cost_usd == pytest.approx(0.082)
    sent = provider._client.images.generate.call_args.kwargs
    assert sent == {
        "model": "gpt-image-2",
        "prompt": "creative director\n\nGolden Gate Bridge",
        "n": 2,
        "size": "1536x1024",
        "quality": "medium",
        "output_format": "jpeg",
        "moderation": "auto",
        "output_compression": 90,
    }


def test_openai_image_provider_rejects_non_image_model_and_tools():
    from smythe.tools import ToolSpec

    provider = _openai_image_with_response(_openai_image_response())
    with pytest.raises(ValueError, match="requires a GPT Image model"):
        asyncio.run(provider.complete("sys", "prompt", "gpt-5"))

    tool = ToolSpec(name="lookup", description="test", input_schema={})
    with pytest.raises(ValueError, match="does not support tool calls"):
        asyncio.run(
            provider.chat(
                "sys",
                [ChatMessage(role="user", content="prompt")],
                "gpt-image-2",
                tools=[tool],
            )
        )


def test_openai_image_provider_rejects_empty_image_data():
    provider = _openai_image_with_response(_openai_image_response(images=0))
    with pytest.raises(RuntimeError, match="no base64 image data"):
        asyncio.run(provider.complete("sys", "prompt", "gpt-image-2"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quality": "ultra"}, "quality must be one of"),
        ({"moderation": "strict"}, "moderation must be one of"),
        (
            {"output_format": "png", "output_compression": 90},
            "supported only for JPEG or WebP",
        ),
    ],
)
def test_openai_image_provider_validates_image_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OpenAIImageProvider(api_key="fake", **kwargs)


def test_gemini_image_config_passthrough():
    p = _gemini_with_response(_image_response(), image_config={"aspect_ratio": "16:9"})
    asyncio.run(p.complete("sys", "wide banner", "gemini-2.5-flash-image"))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert sent["config"]["image_config"] == {"aspect_ratio": "16:9"}


def test_gemini_tools_suppress_auto_image_modality():
    p = _gemini_with_response(_image_response(images=0))
    asyncio.run(p.chat(
        "sys", [ChatMessage(role="user", content="go")],
        "gemini-2.5-flash-image", tools=[WEATHER_TOOL],
    ))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert "response_modalities" not in sent["config"]
    assert "tools" in sent["config"]


def test_gemini_explicit_modalities_still_sent_with_tools():
    p = _gemini_with_response(_image_response(images=0), response_modalities=["TEXT"])
    asyncio.run(p.chat(
        "sys", [ChatMessage(role="user", content="go")],
        "gemini-2.5-flash-image", tools=[WEATHER_TOOL],
    ))
    sent = p._client.aio.models.generate_content.call_args.kwargs
    assert sent["config"]["response_modalities"] == ["TEXT"]


def test_provider_cost_estimate_hints():
    assert GeminiProvider(api_key="k").cost_estimate_per_call is None
    assert GeminiProvider(api_key="k", cost_per_image_usd=0.04).cost_estimate_per_call == 0.04
