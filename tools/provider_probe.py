"""Explicit, paid, single-request connectivity probe; never run by pytest or CI.

Uses a fixed short prompt, at most 128 output tokens, no SDK retries and a
30-second deadline. These are request bounds, not an exact dollar quote.
"""

import argparse
import asyncio
import json
import os

from smythe.provider import AnthropicProvider, GeminiProvider, OpenAIProvider


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--provider", required=True, choices=("anthropic", "openai", "gemini"))
    result.add_argument("--model", required=True, help="Exact text model to probe")
    result.add_argument("--allow-paid", action="store_true", help="Authorize this one paid request")
    return result


async def probe(provider, model):
    key_name = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY"}[provider]
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f"Set {key_name} before authorizing a paid probe")
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        adapter = AnthropicProvider(api_key=key, max_tokens=128)
        client = AsyncAnthropic(api_key=key, base_url="https://api.anthropic.com",
                                max_retries=0, timeout=30)
    elif provider == "openai":
        adapter = OpenAIProvider(api_key=key, max_tokens=128,
                                 base_url="https://api.openai.com/v1",
                                 max_retries=0, request_timeout_s=30)
        client = adapter._get_client()
    else:
        from google import genai
        from google.genai import types

        adapter = GeminiProvider(api_key=key, max_tokens=128, response_modalities=["TEXT"])
        client = genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
            base_url="https://generativelanguage.googleapis.com", timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
    adapter._client = client
    try:
        async with asyncio.timeout(30):
            result = await adapter.complete("You are a test.", "Say hello.", model)
        return {"provider": provider, "model": model, "text": result.text,
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}
    finally:
        if provider == "gemini":
            await client.aio.aclose()
            client.close()
        else:
            await client.close()


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not args.allow_paid:
        cli.error("No request sent: --allow-paid is required")
    print(json.dumps(asyncio.run(probe(args.provider, args.model)), indent=2))


if __name__ == "__main__":
    main()
