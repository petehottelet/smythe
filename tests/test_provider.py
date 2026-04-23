"""Tests for the Provider abstraction and auto-detection."""

import asyncio
import os

import pytest

from smythe.provider import (
    AnthropicProvider,
    CompletionResult,
    GeminiProvider,
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
    p = _auto_detect_provider("claude-mythos")
    assert isinstance(p, AnthropicProvider)


def test_auto_detect_gpt():
    p = _auto_detect_provider("gpt-4o")
    assert isinstance(p, OpenAIProvider)


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
