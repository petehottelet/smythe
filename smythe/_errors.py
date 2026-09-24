"""Pickle and copy support for exceptions whose constructors differ from ``args``.

``BaseException`` pickles and copies as ``type(error)(*error.args)`` followed by
its instance ``__dict__``. A constructor that formats its message from other
parameters, or takes keyword-only ones, then fails on unpickle or wraps its
message twice. Exceptions are pickled whenever they cross a process boundary,
for example from a ``ProcessPoolExecutor``. This module imports nothing from
smythe, so any module can use it without an import cycle.
"""

from __future__ import annotations

from typing import Any, cast


def _rebuild(cls: type[BaseException], args: tuple[Any, ...]) -> BaseException:
    """Recreate an exception with its ``args`` but without calling ``__init__``."""
    return cls.__new__(cls, *args)


class PicklableError:
    """Mixin restoring ``args`` and instance attributes without the constructor.

    List it before the exception base, as in
    ``class QuoteError(PicklableError, ValueError)``. Pickle (every protocol),
    ``copy.copy`` and ``copy.deepcopy`` then return an exception with the same
    type, ``args``, message and attributes.
    """

    __slots__ = ()

    def __reduce__(self) -> tuple[Any, ...]:
        error = cast(BaseException, self)
        return _rebuild, (type(error), error.args), error.__dict__ or None
