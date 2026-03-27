"""Tests for the Provider abstraction and auto-detection."""

import asyncio
import os

import pytest

from smythe.provider import AnthropicProvider, OpenAIProvider, Provider
from smythe.swarm import Swarm, _auto_detect_provider


class MockProvider(Provider):
    """Test provider that returns a canned response."""

    def __init__(self, response: str = "mock response") -> None:
        self._response = response

    async def complete(self, system: str, prompt: str, model: str) -> str:
        return self._response


def test_mock_provider_implements_abc():
    p = MockProvider()
    assert isinstance(p, Provider)


def test_mock_provider_complete():
    p = MockProvider("hello")
    result = asyncio.run(p.complete("sys", "prompt", "model"))
    assert result == "hello"


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


def test_auto_detect_unknown_raises():
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        _auto_detect_provider("gemini-3-pro")


def test_swarm_accepts_explicit_provider():
    mock = MockProvider()
    swarm = Swarm(model="anything-custom", provider=mock)
    assert swarm._provider is mock


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_anthropic_integration():
    p = AnthropicProvider()
    result = asyncio.run(p.complete("You are a test.", "Say hello.", "claude-sonnet-4-20250514"))
    assert len(result) > 0


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_integration():
    p = OpenAIProvider()
    result = asyncio.run(p.complete("You are a test.", "Say hello.", "gpt-4o-mini"))
    assert len(result) > 0
