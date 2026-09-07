"""WAL admission, retry-phase, and connection-setting regression coverage."""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

import smythe._sqlite as sqlite_module
from smythe._sqlite import enable_wal


@pytest.fixture
def connection(tmp_path):
    class Observed(sqlite3.Connection):
        action = None

        def execute(self, sql, *args):
            if self.action is not None:
                self.action(sql)
            return super().execute(sql, *args)

    with closing(sqlite3.connect(tmp_path / "wal.db", isolation_level=None,
                                 factory=Observed)) as database:
        database.execute("PRAGMA busy_timeout = 137").close()
        yield database


def _timeout(connection):
    with closing(sqlite3.Connection.execute(connection, "PRAGMA busy_timeout")) as cursor:
        return cursor.fetchone()[0]


def _busy(code=sqlite3.SQLITE_BUSY):
    error = sqlite3.OperationalError("test contention")
    error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_BUSY | (2 << 8)])
def test_busy_retry_preserves_original_timeout_for_each_schema_check(connection, code):
    attempts, validations = [], []
    error = _busy(code)

    def observe(sql):
        if sql == "PRAGMA journal_mode = WAL":
            assert _timeout(connection) == 0
            attempts.append(sql)
            if len(attempts) == 1:
                raise error

    def validate():
        validations.append(_timeout(connection))

    connection.action = observe
    enable_wal(connection, validate_before_write=validate)
    assert validations == [137, 137]
    assert len(attempts) == 2
    assert _timeout(connection) == 137
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("after_busy", [False, True])
@pytest.mark.parametrize("failure", [_busy(), RuntimeError("invalid schema"), KeyboardInterrupt()])
def test_schema_failure_is_never_retried_or_followed_by_wal_write(connection, after_busy, failure):
    attempts, validations = [], []

    def observe(sql):
        if sql == "PRAGMA journal_mode = WAL":
            attempts.append(sql)
            raise _busy()

    def validate():
        assert _timeout(connection) == 137
        validations.append(True)
        if not after_busy or len(validations) == 2:
            raise failure

    connection.action = observe
    with pytest.raises(type(failure)) as caught:
        enable_wal(connection, validate_before_write=validate)
    assert caught.value is failure
    assert len(attempts) == int(after_busy)
    assert len(validations) == int(after_busy) + 1
    assert _timeout(connection) == 137
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_slow_schema_validation_does_not_consume_wal_wait_budget(connection, monkeypatch):
    clock = [0.0]
    attempts, validations = [], []

    def sleep(duration):
        clock[0] += duration

    monkeypatch.setattr(sqlite_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=sleep))

    def validate():
        validations.append(_timeout(connection))
        clock[0] += 100

    def observe(sql):
        if sql == "PRAGMA journal_mode = WAL":
            attempts.append(sql)
            if len(attempts) < 3:
                raise _busy()

    connection.action = observe
    enable_wal(connection, validate_before_write=validate, timeout_s=0.05)
    assert validations == [137, 137, 137]
    assert len(attempts) == 3
    assert clock[0] == pytest.approx(300.02)
    assert _timeout(connection) == 137


def test_retry_wait_budget_is_shared_across_attempts(connection, monkeypatch):
    clock = [0.0]
    attempts, sleeps = [], []
    error = _busy()

    def sleep(duration):
        sleeps.append(duration)
        clock[0] += duration

    monkeypatch.setattr(sqlite_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=sleep))

    def observe(sql):
        if sql == "PRAGMA journal_mode = WAL":
            attempts.append(sql)
            clock[0] += 0.006
            raise error

    connection.action = observe
    with pytest.raises(sqlite3.OperationalError) as caught:
        enable_wal(connection, validate_before_write=lambda: None, timeout_s=0.025)
    assert caught.value is error
    assert len(attempts) == 2
    assert sum(sleeps) == pytest.approx(0.013)
    assert clock[0] == pytest.approx(0.025)
    assert _timeout(connection) == 137


@pytest.mark.parametrize("phase", ["setup", "restore"])
def test_busy_outside_wal_phase_does_not_retry(connection, phase):
    error = _busy()
    attempts = []

    def observe(sql):
        attempts.append(sql)
        if sql == f"PRAGMA busy_timeout = {0 if phase == 'setup' else 137}":
            # Simulate an error after the local setting was applied.
            sqlite3.Connection.execute(connection, sql).close()
            raise error

    connection.action = observe
    with pytest.raises(sqlite3.OperationalError) as caught:
        enable_wal(connection, validate_before_write=lambda: None)
    assert caught.value is error
    assert attempts.count("PRAGMA journal_mode = WAL") == int(phase == "restore")
    assert _timeout(connection) == 137


@pytest.mark.parametrize("failure", [_busy(), KeyboardInterrupt()])
def test_restore_failure_preserves_original_wal_failure_without_retry(connection, failure):
    attempts = []

    def observe(sql):
        attempts.append(sql)
        if sql == "PRAGMA journal_mode = WAL":
            raise failure
        if sql == "PRAGMA busy_timeout = 137":
            sqlite3.Connection.execute(connection, sql).close()
            raise OSError("restoration failure")

    connection.action = observe
    with pytest.raises(type(failure)) as caught:
        enable_wal(connection, validate_before_write=lambda: None)
    assert caught.value is failure
    assert attempts.count("PRAGMA journal_mode = WAL") == 1
    assert any("restoration failure" in note for note in failure.__notes__)
    assert _timeout(connection) == 137


@pytest.mark.parametrize("succeeds", [False, True])
def test_zero_wait_budget_allows_one_attempt_without_sleep(connection, monkeypatch, succeeds):
    attempts = []
    error = _busy()

    def observe(sql):
        if sql == "PRAGMA journal_mode = WAL":
            attempts.append(sql)
            if not succeeds:
                raise error

    monkeypatch.setattr(sqlite_module, "time", SimpleNamespace(
        monotonic=lambda: 0.0, sleep=lambda _: pytest.fail("zero wait must not sleep")))
    connection.action = observe
    if succeeds:
        enable_wal(connection, validate_before_write=lambda: None, timeout_s=0)
    else:
        with pytest.raises(sqlite3.OperationalError) as caught:
            enable_wal(connection, validate_before_write=lambda: None, timeout_s=0)
        assert caught.value is error
    assert len(attempts) == 1
    assert _timeout(connection) == 137


@pytest.mark.parametrize("timeout", [None, True, "5", -1, float("nan"),
                                    float("inf"), 10**1000])
def test_invalid_wait_budget_is_rejected_before_accessing_connection(timeout):
    with pytest.raises(ValueError, match="finite and non-negative"):
        enable_wal(None, validate_before_write=lambda: pytest.fail("validator called"),
                   timeout_s=timeout)
