"""Bundled sources and a source library on the loopback address, for the research examples.

The research examples read their sources with journaled read steps from a
small HTTP server inside the example's process: there is no other network
traffic and no API key. The sources are text files in
examples/source_to_artifact/sources/. A source has one or more versions,
oldest first: ``<source id>.txt`` holds its first version, and each later
one is ``<version>.txt`` in a folder named for the source, such as
``renewals-2026/2026-q3.txt``.

The library serves each source's current version and counts every read it
answers. Releasing a source makes its next version current, as when a
source is revised.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from smythe import ContextSource, ReadProfile, SourceStatus

SOURCE_DIR = Path(__file__).resolve().parent / "source_to_artifact" / "sources"


def read_versions(versions: Mapping[str, Sequence[str]],
                  directory: Path = SOURCE_DIR) -> dict[str, tuple[ContextSource, ...]]:
    """Each source's bundled versions, oldest first, by source id.

    *versions* maps each source id to its version labels, oldest first.
    """
    found = {}
    for source_id, labels in versions.items():
        paths = [directory / f"{source_id}.txt",
                 *(directory / source_id / f"{label}.txt" for label in labels[1:])]
        found[source_id] = tuple(ContextSource(source_id, path.read_text(encoding="utf-8"), label)
                                 for path, label in zip(paths, labels))
    return found


def read_sources(versions: Mapping[str, str],
                 directory: Path = SOURCE_DIR) -> tuple[ContextSource, ...]:
    """The first version of each source *versions* names, labeled as it says, sorted by id."""
    found = read_versions({source_id: [label] for source_id, label in versions.items()}, directory)
    return tuple(found[source_id][0] for source_id in sorted(found))


class SourceLibrary:
    """Serves each source's current version on the loopback address, counting each read.

    It runs while it is entered as a context manager. ``reads`` lists the
    source id of each read it answered, in order. It is also a source
    catalog for repair plans: :meth:`describe` answers from memory, without
    a read or a cost.
    """

    metered = False

    def __init__(self, versions: Mapping[str, Sequence[ContextSource]]) -> None:
        self._versions = {source_id: tuple(found) for source_id, found in versions.items()}
        self._current = dict.fromkeys(self._versions, 0)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self.reads: list[str] = []

    def current(self, source_id: str) -> ContextSource:
        with self._lock:
            return self._versions[source_id][self._current[source_id]]

    def release(self, source_id: str) -> ContextSource:
        """Make the source's next bundled version current, and return it."""
        with self._lock:
            if self._current[source_id] + 1 >= len(self._versions[source_id]):
                raise ValueError(f"{source_id} has no later version")
            self._current[source_id] += 1
        return self.current(source_id)

    def describe(self, source_id: str) -> SourceStatus | None:
        """The current version, by the SHA-256 of its text.

        A run that is not given a source as one of its context sources records
        a read of it by the digest of the text received, without a version
        label. The current version is described the same way, so a new version
        is a changed source.
        """
        if source_id not in self._versions:
            return None
        return SourceStatus(sha256=self.current(source_id).sha256)

    def profile(self) -> ReadProfile:
        """One versioned, free read that is safe to repeat; loopback is allowed explicitly."""
        return library_profile(self.port)

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("the library is not running")
        return self._server.server_address[1]

    def __enter__(self) -> SourceLibrary:
        library = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                source_id = self.path.removeprefix("/sources/")
                body = (library.current(source_id).text.encode("utf-8")
                        if source_id in library._versions else None)
                with library._lock:
                    library.reads.append(source_id)
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


@contextmanager
def source_library(sources: Sequence[ContextSource]):
    """Serve one version of each source; yield the port and the list of reads answered."""
    with SourceLibrary({source.source_id: [source] for source in sources}) as library:
        yield library.port, library.reads


def library_profile(port: int) -> ReadProfile:
    """One versioned, free read that is safe to repeat; loopback is allowed explicitly."""
    return ReadProfile(name="library", version="1", url=f"http://127.0.0.1:{port}/sources/{{source_id}}",
                       input_schema={"source_id": "string"}, metered=False, repeat_safe=True,
                       timeout_s=10.0, allow_private_networks=True)
