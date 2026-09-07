"""Build-only PyPI README links; the repository README is never modified.

The supported markup is deliberately the README's inline Markdown links and
quoted HTML a/img attributes. Unexpected link syntax fails the build instead
of silently publishing a relative destination. No runtime imports are needed.
The registry badge alone follows main with a generic label, so its pre-release
version is not permanently frozen into the new distribution description.
"""

from __future__ import annotations

import html
from pathlib import Path
import posixpath
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit


_REPOSITORY = "https://github.com/petehottelet/smythe"
_RAW = "https://raw.githubusercontent.com/petehottelet/smythe"
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LINK = re.compile(r"(?P<image>!)?\[[^\]\n]*\]\((?P<url>[^()\s<>]+)\)")
_TOKEN = re.compile(r"`+|!?\[[^\]\n]*\]\(|<(?:a|img)\b", re.I)
_HTML = re.compile(r"<(?P<tag>a|img)\b[^<>]*>", re.I)
_ATTRIBUTE = re.compile(r"\b(?P<name>href|src)\s*=\s*(?P<quote>[\"'])(?P<url>.*?)(?P=quote)", re.I)
_ALT = re.compile(r"\balt\s*=\s*(?P<quote>[\"'])(?P<text>.*?)(?P=quote)", re.I)


def _registry_badge(value: str) -> bool:
    parsed = urlsplit(value)
    return not (parsed.scheme or parsed.netloc) and posixpath.normpath(unquote(parsed.path)) == "assets/badges/pypi.svg"


def _destination(value: str, version: str, *, image: bool) -> str:
    if not value or value.startswith("#"):
        return value
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return value
    path = unquote(parsed.path)
    if not path or path.startswith("/") or "\\" in path or any(ord(char) < 32 for char in path):
        raise ValueError(f"Unsupported README destination: {value!r}")
    normalized = posixpath.normpath(path)
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"README destination escapes repository: {value!r}")
    tag = quote(f"v{version}", safe="")
    if image:
        base = f"{_RAW}/{'main' if _registry_badge(value) else tag}/"
    else:
        kind = "tree" if path.endswith("/") else "blob"
        base = f"{_REPOSITORY}/{kind}/{tag}/"
    encoded = quote(normalized, safe="/") + ("/" if path.endswith("/") else "")
    return urlunsplit((*urlsplit(base + encoded)[:3], parsed.query, parsed.fragment))


def _rewrite_inline(line: str, version: str) -> str:
    chunks, cursor = [], 0
    while match := _TOKEN.search(line, cursor):
        chunks.append(line[cursor:match.start()])
        token = match.group()
        if token.startswith("`"):
            closing = re.search(rf"(?<!`){re.escape(token)}(?!`)", line[match.end():])
            if closing is None:
                # An unmatched backtick is literal Markdown text.
                chunks.append(token)
                cursor = match.end()
            else:
                cursor = match.end() + closing.end()
                chunks.append(line[match.start():cursor])
        elif token.startswith("<"):
            tag = _HTML.match(line, match.start())
            if tag is None:
                raise ValueError("README a/img tags must occupy one line with quoted attributes")
            image = tag["tag"].lower() == "img"
            name = "src" if image else "href"
            attributes = [attr for attr in _ATTRIBUTE.finditer(tag.group()) if attr["name"].lower() == name]
            if len(attributes) != 1:
                raise ValueError(f"README {tag['tag']} must have exactly one quoted {name}")
            attr = attributes[0]
            target = _destination(html.unescape(attr["url"]), version, image=image)
            original = tag.group()
            if target == html.unescape(attr["url"]):
                rewritten = original
            else:
                rewritten = (original[:attr.start("url")] + html.escape(target, quote=True)
                             + original[attr.end("url"):])
            if image and _registry_badge(html.unescape(attr["url"])):
                alt = _ALT.search(rewritten)
                if alt is None:
                    raise ValueError("Registry badge requires a quoted alt attribute")
                rewritten = rewritten[:alt.start("text")] + "Latest PyPI release" + rewritten[alt.end("text"):]
            chunks.append(rewritten)
            cursor = tag.end()
        else:
            link = _LINK.match(line, match.start())
            if link is None:
                raise ValueError("README links require a simple inline destination without titles")
            prefix = ("![Latest PyPI release](" if link["image"] and _registry_badge(link["url"])
                      else line[link.start():link.start("url")])
            chunks.append(prefix
                          + _destination(link["url"], version, image=bool(link["image"]))
                          + line[link.end("url"):link.end()])
            cursor = link.end()
    chunks.append(line[cursor:])
    return "".join(chunks)


def render_pypi_readme(text: str, version: str) -> str:
    """Resolve local links from README markup to the matching release tag."""
    if not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.!+-]*", version):
        raise ValueError("A static distribution version is required for README release links")
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if fence is not None:
            lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
            lines.append(line)
        else:
            if re.match(r"^ {0,3}\[[^\]]+\]:", line):
                raise ValueError("README reference-style links require explicit build-hook support")
            lines.append(_rewrite_inline(line, version))
    if fence is not None:
        raise ValueError("README has an unclosed code fence")
    return "".join(lines)


def get_metadata_hook():
    # Hatchling's documented custom-plugin selector keeps this dependency out
    # of imports used by pure formatting tests and the installed Smythe package.
    from hatchling.metadata.plugin.interface import MetadataHookInterface

    class PypiReadmeHook(MetadataHookInterface):
        def update(self, metadata):
            text = (Path(self.root) / "README.md").read_text(encoding="utf-8")
            metadata["readme"] = {"text": render_pypi_readme(text, metadata["version"]),
                                  "content-type": "text/markdown"}

    return PypiReadmeHook
