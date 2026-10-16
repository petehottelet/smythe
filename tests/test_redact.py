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
    # The names below it stay in every spelling, though elsewhere a repr's
    # doubled backslashes start a UNC root.
    deeper = str(value / "keys" / "team" / "service.json")
    below = "~" + deeper[len(native):]
    for spell in (repr, json.dumps, lambda text: repr(repr(text))):
        assert redact_paths(spell(deeper)) == spell(below)
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
        "./local/file.txt",
        "~/project/out.json",
        "ratio 1/2/3",
        "request failed (POST /v1/chat/completions): 404",
        "retrying 'GET /v1/models/list'",
    )
    for text in kept:
        assert redact_paths(text) == text
    # A method is one only as a word of its own, so a path after a word that
    # merely ends in a method's name is redacted.
    for word in ("INPUT", "OUTPUT", "TARGET", "BUDGET", "DISPATCH", "XPOST", "re-PUT", "app.GET"):
        assert redact_paths(f"{word} /srv/app/key.json locked") == f"{word} <path>"
        assert redact_paths(f"{word} /srv/app/key.json: locked") == f"{word} <path>: locked"
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
    # A method's name with no route after it can be part of a folder's name.
    for named in (windows("D:", "My POST Box", "x.txt"), "/srv/My GET Data/x.txt"):
        assert redact_paths(f"open {named} failed") == "open <path>"
    assert redact_paths(f"open {path} then GET") == "open <path>"


@pytest.mark.parametrize("value", [WINDOWS_HOME, POSIX_HOME], ids=["windows", "posix"])
def test_the_home_folder_is_written_as_home_after_any_character(home, value):
    native = home(value)
    include = str(value / "include")
    tail = include[len(native):]
    # Compiler flags, URLs and elided paths put a character right before it.
    for text, kept in ((f"cc -I{include} failed", f"cc -I~{tail} failed"),
                       (f"get https://h{native}{tail} failed", f"get https://h~{tail} failed"),
                       (f"path ...{include} not found", f"path ...~{tail} not found")):
        redacted = redact_paths(text)
        assert "Example" not in redacted and "Person" not in redacted
        assert redacted == kept


@pytest.mark.parametrize("value", [WINDOWS_HOME, POSIX_HOME], ids=["windows", "posix"])
def test_a_unc_path_is_redacted_after_any_character(home, value):
    home(value)
    paths = (windows("", "", "fileserver", "clients", "Acme Merger", "include"),
             windows("", "", "fileserver", "jdoe", "include", "secret.h"))
    # Compiler flags and elided paths put a character right before its root.
    for path in paths:
        for spelling in (path, path.replace(BACKSLASH, BACKSLASH * 2)):
            for before in ("cc -I", "cl /I", "cl.exe /I", "mapped_", "path ...", "x", "~"):
                redacted = redact_paths(f"{before}{spelling} failed: no such file")
                assert redacted == f"{before}<path>: no such file"
    # A repr doubles the separators of a path with no drive, and of a relative
    # one, so each reads as a UNC root and is redacted, in preference to
    # keeping a network path that follows a word.
    rooted = windows("", "home", "Other Person", "include")
    assert redact_paths(repr(f"-I{rooted}")) == "'-I<path>'"
    assert redact_paths(repr(windows("relative", "dir", "file.txt"))) == "'relative<path>'"


def test_a_path_after_a_closing_bracket_is_redacted(home):
    home(WINDOWS_HOME)
    # A log line's tag puts a ']' right before a path.
    other_user = "/".join(["", "home", "other.user", ".ssh", "id_rsa"])
    for tag in ("[worker-1]", "[ERR]", "[tag]", "]"):
        for path in ("/srv/clients/acme-merger/notes.txt", other_user):
            assert redact_paths(f"{tag}{path}: denied") == f"{tag}<path>: denied"
            assert redact_paths(f"{tag}{path} missing") == f"{tag}<path>"
    # Only a URL's bracketed IPv6 host is kept with the path after it, as any
    # other URL's path is.
    for url in ("http://[::1]/v1/models/list", "https://[2001:db8::7]/v1/files/upload-7",
                "http://[fe80::1%25eth0]/status/check"):
        assert redact_paths(f"see {url} now") == f"see {url} now"
    assert redact_paths("see http://[::1]/v1/models for /srv/app/key.json") == (
        "see http://[::1]/v1/models for <path>")
    drive = "/".join(["C:", "Users", "Other Person", "key.json"])
    assert redact_paths(f"get http://[::1]/{drive} failed") == "get http://[::1]/<path>"



def test_a_name_that_holds_a_quote_mark_is_redacted_whole(home):
    home(WINDOWS_HOME)
    # A backtick, which a name can hold, does not end an unquoted path.
    for path in (windows("D:", "clients", "it`s here", "notes.txt"), "/srv/it`s/key.json"):
        assert redact_paths(f"open {path} failed") == "open <path>"
    redacted = redact_paths("cl /I`//fileserver/share/include` failed")
    assert "fileserver" not in redacted and "share" not in redacted
    # A quote mark with a letter or a digit after it, as in O'Brien, is part
    # of a quoted name rather than its closing quote.
    for quote, name in (("`", "it`s"), ("'", "O'Brien")):
        quoted = f"{quote}/srv/{name}/key.json{quote}"
        assert redact_paths(f"see {quoted} now") == f"see {quote}<path>{quote} now"
