"""Independent-writer and persistence-failure boundaries for file checkpoints."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import tempfile
import threading

import pytest

from smythe import checkpoint
from smythe.checkpoint import FileCheckpointStore


def test_independent_stores_publish_complete_snapshots_without_shared_temporary_files(
    tmp_path, monkeypatch,
):
    stores = [FileCheckpointStore(tmp_path), FileCheckpointStore(tmp_path)]
    payloads = [{"writer": index, "body": str(index) * 100_000} for index in range(2)]
    barrier = threading.Barrier(2)
    publication_lock = threading.Lock()
    writer = threading.local()
    original = os.replace
    sources = []
    published = []

    def publish(source, destination):
        sources.append(Path(source))
        barrier.wait(timeout=60)
        with publication_lock:
            original(source, destination)
            published.append((writer.index, json.loads(Path(destination).read_bytes())))

    def save(index):
        writer.index = index
        stores[index].save("shared", payloads[index])

    monkeypatch.setattr(checkpoint.os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, index) for index in range(2)]
        failures = [future.exception(timeout=90) for future in futures]
    assert failures == [None, None]
    assert sorted(published) == list(enumerate(payloads))
    assert len(set(sources)) == 2
    assert all(source.parent == tmp_path for source in sources)
    assert stores[0].load("shared") in payloads
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shared.json"]


def test_file_bytes_are_flushed_before_the_checkpoint_is_published(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})
    original_fsync, original_replace = os.fsync, os.replace
    synced = []

    def flush(descriptor):
        synced.append(stat.S_ISREG(os.fstat(descriptor).st_mode))
        original_fsync(descriptor)

    def publish(source, destination):
        assert synced == [True]
        assert store.load("run") == {"old": True}
        assert json.loads(Path(source).read_bytes()) == {"new": True}
        original_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "fsync", flush)
    monkeypatch.setattr(checkpoint.os, "replace", publish)
    store.save("run", {"new": True})
    assert synced == ([True] if os.name == "nt" else [True, False])
    assert store.load("run") == {"new": True}


@pytest.mark.parametrize("failure", ["serialization", "file_flush", "replace"])
def test_failed_save_preserves_previous_bytes_and_removes_only_its_own_temp(
    tmp_path, monkeypatch, failure,
):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})
    previous = (tmp_path / "run.json").read_bytes()
    # A stranded older writer's file is not this save's cleanup authority.
    stranded = tmp_path / ".run.json.earlier.tmp"
    stranded.write_bytes(b"preserve previous interrupted writer")
    state = {"new": True}

    def fail(*args, **kwargs):
        raise OSError("injected persistence failure")

    if failure == "serialization":
        state["opaque"] = object()
        expected = TypeError
    else:
        monkeypatch.setattr(checkpoint.os, "fsync" if failure == "file_flush" else "replace", fail)
        expected = OSError
    with pytest.raises(expected):
        store.save("run", state)
    assert (tmp_path / "run.json").read_bytes() == previous
    assert stranded.read_bytes() == b"preserve previous interrupted writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [stranded.name, "run.json"]
    assert store.list_ids() == ["run"]


@pytest.mark.skipif(os.name == "nt", reason="Windows stdlib cannot flush directory entries")
def test_directory_flush_failure_reports_error_after_atomic_publication(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})
    original = os.fsync

    def flush(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory flush failed")
        original(descriptor)

    monkeypatch.setattr(checkpoint.os, "fsync", flush)
    with pytest.raises(OSError, match="directory flush failed"):
        store.save("run", {"new": True})
    assert store.load("run") == {"new": True}
    assert [path.name for path in tmp_path.iterdir()] == ["run.json"]


def test_interruption_during_publication_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})

    def interrupted(*args):
        raise KeyboardInterrupt("interrupted before replace")

    monkeypatch.setattr(checkpoint.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.save("run", {"new": True})
    assert store.load("run") == {"old": True}
    assert [path.name for path in tmp_path.iterdir()] == ["run.json"]


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_stream_failure_preserves_previous_checkpoint_and_original_exception(
    tmp_path, monkeypatch, failure,
):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})
    previous = (tmp_path / "run.json").read_bytes()
    original = tempfile.NamedTemporaryFile
    error = OSError(f"injected stream {failure} failure")

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream
            self.name = stream.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            if failure == "write":
                self.stream.write(data[:5])
                raise error
            return self.stream.write(data)

        def flush(self):
            if failure == "flush":
                raise error
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *args, **kwargs: FailingStream(original(*args, **kwargs)))
    with pytest.raises(OSError) as caught:
        store.save("run", {"new": "complete snapshot"})
    assert caught.value is error
    assert (tmp_path / "run.json").read_bytes() == previous
    assert [path.name for path in tmp_path.iterdir()] == ["run.json"]


def test_serialization_failure_does_not_create_a_temporary_file(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})

    def unexpected(*args, **kwargs):
        pytest.fail("serialization failure reached temporary-file creation")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", unexpected)
    with pytest.raises(TypeError):
        store.save("run", {"unserializable": object()})
    assert store.load("run") == {"old": True}


def test_cleanup_failure_does_not_replace_the_original_publication_error(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("run", {"old": True})
    stranded = tmp_path / ".run.json.earlier.tmp"
    stranded.write_bytes(b"earlier writer")
    original_unlink = Path.unlink
    attempted = []
    error = OSError("original replacement failure")

    def replace(*args):
        raise error

    def unlink(path, *args, **kwargs):
        attempted.append(path)
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(checkpoint.os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    try:
        with pytest.raises(OSError) as caught:
            store.save("run", {"new": True})
        assert caught.value is error
        assert store.load("run") == {"old": True}
        assert len(attempted) == 1 and attempted[0] != stranded
        assert attempted[0].parent == tmp_path and attempted[0].suffix == ".tmp"
        assert stranded.read_bytes() == b"earlier writer"
    finally:
        # The injected cleanup failure intentionally left this save's temp.
        for path in attempted:
            original_unlink(path, missing_ok=True)
