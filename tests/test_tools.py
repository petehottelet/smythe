"""Tests for the neutral tool-calling types and helpers."""

import pytest

from smythe.tools import (
    ChatMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
    content_to_text,
    display_name,
    wire_name,
)


# ---------------------------------------------------------------------------
# Wire-name mapping
# ---------------------------------------------------------------------------


def test_wire_name_roundtrip():
    assert wire_name("fs.read_file") == "fs__read_file"
    assert display_name("fs__read_file") == "fs.read_file"


def test_wire_name_passthrough_without_namespace():
    assert wire_name("get_weather") == "get_weather"
    assert display_name("get_weather") == "get_weather"


# ---------------------------------------------------------------------------
# ToolSpec validation
# ---------------------------------------------------------------------------


def test_toolspec_accepts_namespaced_name():
    spec = ToolSpec(name="fs.read_file", description="d", input_schema={"type": "object"})
    assert spec.name == "fs.read_file"


@pytest.mark.parametrize("bad", ["", "has space", "emoji🙂", "x" * 100, "a/b"])
def test_toolspec_rejects_invalid_names(bad):
    with pytest.raises(ValueError, match="Tool name"):
        ToolSpec(name=bad, description="d", input_schema={})


# ---------------------------------------------------------------------------
# ChatMessage defaults
# ---------------------------------------------------------------------------


def test_chat_message_defaults():
    m = ChatMessage(role="user", content="hi")
    assert m.tool_calls == []
    assert m.tool_results == []


def test_tool_result_defaults():
    r = ToolResult(tool_call_id="c1", content="ok")
    assert r.is_error is False


def test_tool_call_fields():
    c = ToolCall(id="c1", name="fs.read_file", arguments={"path": "x"})
    assert c.arguments["path"] == "x"


# ---------------------------------------------------------------------------
# content_to_text — never raises, placeholders for non-text
# ---------------------------------------------------------------------------


class _TextPart:
    text = "hello"


class _ImagePart:
    type = "image"
    mimeType = "image/png"
    data = "x" * 4096


class _WeirdPart:
    """No text, no type, no data — placeholder falls back to class name."""


def test_content_to_text_concatenates_text_parts():
    assert content_to_text([_TextPart(), _TextPart()]) == "hello\nhello"


def test_content_to_text_image_placeholder():
    out = content_to_text([_ImagePart()])
    assert out.startswith("[image: image/png, 4KB")
    assert "not passed to the model" in out


def test_content_to_text_mixed():
    out = content_to_text([_TextPart(), _ImagePart()])
    assert out.splitlines()[0] == "hello"
    assert "[image" in out


def test_content_to_text_unknown_part_uses_class_name():
    assert "_WeirdPart" in content_to_text([_WeirdPart()])


def test_content_to_text_never_raises_on_odd_inputs():
    assert content_to_text(None) == ""
    assert content_to_text("already text") == "already text"
    assert content_to_text(42) == "42"
    assert content_to_text([]) == ""
