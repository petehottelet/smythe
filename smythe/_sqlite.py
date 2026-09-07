"""Internal SQLite initialization helpers shared by durable stores."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable


def _wal_attempt(connection: sqlite3.Connection, original_timeout: int) -> sqlite3.OperationalError | None:
    """Return only a WAL-phase BUSY; setup and restoration failures propagate."""

    failure: BaseException | None = None
    busy: sqlite3.OperationalError | None = None
    try:
        connection.execute("PRAGMA busy_timeout = 0").close()
        try:
            cursor = connection.execute("PRAGMA journal_mode = WAL")
            try:
                row = cursor.fetchone()
            finally:
                cursor.close()
            if row is None or row[0] != "wal":
                raise sqlite3.OperationalError("SQLite could not enable WAL journal mode")
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", None)
            if isinstance(code, int) and code & 0xff == sqlite3.SQLITE_BUSY:
                busy = error
            raise
    except BaseException as error:
        failure = error

    try:
        connection.execute(f"PRAGMA busy_timeout = {original_timeout}").close()
    except BaseException as error:
        if failure is None:
            raise
        failure.add_note(f"Restoring SQLite busy_timeout also failed: {error!r}")
        raise failure from error
    if failure is not None and busy is None:
        raise failure
    return busy


def enable_wal(
    connection: sqlite3.Connection,
    *,
    validate_before_write: Callable[[], None],
    timeout_s: float = 5.0,
) -> None:
    """Enable and verify WAL, retrying only transient WAL-phase SQLITE_BUSY.

    The finite budget covers WAL attempts and retry waiting, excluding schema
    validation. The validator runs before every attempt with the caller's
    original busy timeout; its errors propagate without retry. WAL attempts use
    zero native busy timeout, and the original value is restored on every exit.
    Schema creation/migration and locked revalidation remain caller duties.
    """

    try:
        valid = type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("WAL timeout must be finite and non-negative")
    remaining = float(timeout_s)
    cursor = connection.execute("PRAGMA busy_timeout")
    try:
        original_timeout = cursor.fetchone()[0]
    finally:
        cursor.close()

    while True:
        validate_before_write()
        # Schema validation may scan a large ledger. It is deliberately outside
        # this WAL-specific wait budget and never runs with a temporary timeout.
        deadline = time.monotonic() + remaining
        busy = _wal_attempt(connection, original_timeout)
        if busy is None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise busy
        time.sleep(min(0.01, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise busy
