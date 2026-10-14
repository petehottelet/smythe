"""Local paths are redacted whole from the error text a record keeps.

Every path here is built at runtime, so no tracked file holds a user's home
folder.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from smythe._redact import redact_paths

BACKSLASH = "\\"
WINDOWS_HOME = PureWindowsPath("C:/", "Users", "Example Person")
POSIX_HOME = PurePosixPath("/", "home", "Example Person")


def windows(*parts: str) -> str:
    return BACKSLASH.join(parts)


@pytest.fixture
def home(monkeypatch):
    """Make the home folder the given path, whatever this host's own is."""
    def use(value):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: value))
        return str(value)
    return use


@pytest.mark.parametrize("path", [
    windows("C:", "Program Files", "Tool", "run.exe"),
    windows("D:", "My Data", "Runs 2026", "out.json"),
    windows("D:", "data", "report final.txt"),
    windows("D:", "data", "My Report"),
    windows("C:", "Users", "Other Person"),
    windows("C:", "Users", "jane doe"),
    windows("D:", "projects", "acme merger plan"),
    windows("", "", "fileserver", "shared drive", "run.txt"),
    "/srv/My Data/out.json",
    "/srv/clients/acme secret",
    "file:///srv/My Data/key.json",
], ids=["program-files", "two-spaces", "final-name", "capitalized-name", "another-user",
        "lowercase-user", "lowercase-words", "unc", "posix", "posix-lowercase", "file-url"])
def test_an_absolute_path_with_a_space_is_redacted_whole(home, path):
    home(WINDOWS_HOME)
    escaped = path.replace(BACKSLASH, BACKSLASH * 2)
    for spelling in (path, escaped):
        # Unquoted, the words after a path cannot be told from the rest of its
        # last name, so they are removed with it.
        assert redact_paths(f"open {spelling} failed") == "open <path>"
        assert redact_paths(f"open {spelling} is locked") == "open <path>"
        assert redact_paths(f"open {spelling}: access denied") == "open <path>: access denied"
        assert redact_paths(f"Permission denied: {spelling}") == "Permission denied: <path>"
        assert redact_paths(f"open {spelling} failed\nretrying") == "open <path>\nretrying"
    for quote in ("'", '"', "`"):
        quoted = f"{quote}{path}{quote}"
        assert redact_paths(f"open {quoted} failed") == f"open {quote}<path>{quote} failed"
    # A repr quotes a path and doubles its backslashes.
    assert redact_paths(f"No such file or directory: {path!r}") == (
        "No such file or directory: '<path>'")
    assert redact_paths(repr({"path": path})) == "{'path': '<path>'}"


@pytest.mark.parametrize("value", [WINDOWS_HOME, POSIX_HOME], ids=["windows", "posix"])
def test_every_spelling_of_the_home_folder_is_written_as_home(home, value):
    native = home(value)
    key_file = str(value / "keys" / "service.json")
    tail = key_file[len(native):]  # the separator and the names below the home folder
    texts = {
        f"read {key_file} failed": f"read ~{tail} failed",
        # A dict's repr and JSON text double each backslash.
        repr({"key_file": key_file}): repr({"key_file": "~" + tail}),
        json.dumps({"key_file": key_file}): json.dumps({"key_file": "~" + tail}),
        f"read {key_file.replace(BACKSLASH, '/')}": "read ~/keys/service.json",
        f"working in {native}": "working in ~",
        repr(f"working in {native}"): repr("working in ~"),
    }
    for text, expected in texts.items():
        assert redact_paths(text) == expected
    # A folder whose name only begins with the home folder's is another absolute path.
    sibling = str(value.parent / (value.name + "ality") / "notes.txt")
    assert redact_paths(f"read {sibling}") == "read <path>"
    assert redact_paths(repr(sibling)) == "'<path>'"


def test_routes_and_url_paths_stay_readable(home):
    home(WINDOWS_HOME)
    kept = (
        "POST /v1/chat/completions returned 404",
        "GET /v1/models/list timed out",
        "DELETE /v1/files/upload-7 failed",
        "see https://api.example.com/v1/models/list",
        "see http://[::1]/v1/models/list",
        "relative/dir/file.txt",
        windows("relative", "dir", "file.txt"),
        repr(windows("relative", "dir", "file.txt")),
        "./local/file.txt",
        "~/project/out.json",
        "ratio 1/2/3",
    )
    for text in kept:
        assert redact_paths(text) == text
    # A route never shields a local path that follows it.
    assert redact_paths("POST /v1/files failed for /srv/app/key.json") == (
        "POST /v1/files failed for <path>")
    assert redact_paths("open /srv/app/key.json then GET /v1/models") == (
        "open <path> GET /v1/models")
    assert redact_paths("open /srv/app/key.json POST /v1/models") == (
        "open <path> POST /v1/models")


def test_an_unquoted_path_takes_in_every_word_up_to_where_it_must_end(home):
    home(WINDOWS_HOME)
    path = windows("D:", "data", "out.json")
    # Any word can be the rest of a name with a space, so each is taken in, in
    # preference to keeping part of a path.
    for after in ("on request 7", "and/or more", "Permission denied", "failed, retrying"):
        assert redact_paths(f"open {path} {after}") == "open <path>"
    assert redact_paths(f"open {windows('D:', 'data', 'my report')} failed") == "open <path>"
    # A comma does not end a path, as folder names hold commas.
    clients = windows("D:", "clients", "Smith, John", "notes.txt")
    assert redact_paths(f"open {clients} failed") == "open <path>"
    # A word that ends in a colon or a semicolon ends it, and its mark stays.
    assert redact_paths(f"open {path}: denied") == "open <path>: denied"
    assert redact_paths(f"open {windows('D:', 'data', 'my report')}: denied") == (
        "open <path>: denied")
    assert redact_paths(f"open {path} now; retrying") == "open <path>; retrying"
    # So do the end of a line, a quote, a URL, and an HTTP method and its route.
    assert redact_paths(f"open {path} failed\nretrying") == "open <path>\nretrying"
    assert redact_paths(f"open {path} failed 'x'") == "open <path> 'x'"
    assert redact_paths(repr(f"open {path} failed")) == "'open <path>'"
    assert redact_paths(f"open {path} see https://example.com/a") == (
        "open <path> https://example.com/a")
    assert redact_paths(f"open {path} then GET /v1/models") == "open <path> GET /v1/models"
