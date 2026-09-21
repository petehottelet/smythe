"""The manual connectivity tool has no credential-activated execution path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import provider_probe
from smythe.provider import CompletionResult


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_probe_requires_explicit_paid_flag_with_credentials_present(monkeypatch, provider):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "fake-key")
    dispatch = AsyncMock()
    monkeypatch.setattr(provider_probe, "probe", dispatch)
    with pytest.raises(SystemExit) as error:
        provider_probe.main(["--provider", provider, "--model", "text-model"])
    assert error.value.code == 2
    dispatch.assert_not_called()


def test_single_bounded_openai_probe_closes_client_even_on_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    close = AsyncMock()
    complete = AsyncMock(side_effect=RuntimeError("failed request"))
    config = {}

    def adapter(**kwargs):
        config.update(kwargs)
        return SimpleNamespace(_get_client=lambda: SimpleNamespace(close=close), complete=complete)

    monkeypatch.setattr(provider_probe, "OpenAIProvider", adapter)
    with pytest.raises(RuntimeError, match="failed request"):
        asyncio.run(provider_probe.probe("openai", "text-model"))
    assert config["max_tokens"] == 128
    assert config["max_retries"] == 0
    assert config["request_timeout_s"] == 30
    assert config["base_url"] == "https://api.openai.com/v1"
    complete.assert_awaited_once_with("You are a test.", "Say hello.", "text-model")
    close.assert_awaited_once()


def test_successful_probe_retains_output_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = SimpleNamespace(close=AsyncMock())
    adapter = SimpleNamespace(_get_client=lambda: client,
                              complete=AsyncMock(return_value=CompletionResult(
                                  text="hello", prompt_tokens=5, completion_tokens=2)))
    monkeypatch.setattr(provider_probe, "OpenAIProvider", lambda **kwargs: adapter)
    result = asyncio.run(provider_probe.probe("openai", "text-model"))
    assert result == {"provider": "openai", "model": "text-model", "text": "hello",
                      "prompt_tokens": 5, "completion_tokens": 2}


@pytest.mark.parametrize("provider,key", [("anthropic", "ANTHROPIC_API_KEY"),
                                          ("openai", "OPENAI_API_KEY"), ("gemini", "GOOGLE_API_KEY")])
def test_probe_checks_credentials_before_sdk_dispatch(monkeypatch, provider, key):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match=f"Set {key}"):
        asyncio.run(provider_probe.probe(provider, "text-model"))
