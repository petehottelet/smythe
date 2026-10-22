"""Exact conversions between the units each ledger keeps money in.

The ordinary ``Sentinel`` and ``CompletionResult`` hold float USD, the durable
workflow journal and native receipts hold integer nanoUSD, and Jobs and the
Autotune ledger hold integer microUSD. Every conversion here names its units
and its rounding, uses integer arithmetic only, and so does not depend on the
caller's decimal context:

- ``EXACT`` for a confirmed amount: convert exactly or raise.
- ``CEILING`` for a reservation or exposure: never round down.
- ``FLOOR`` for a budget or remaining allowance: never round up.

A float is read by its shortest decimal form, the number its ``repr`` shows,
not by its binary expansion. Unknown money has no conversion: ``None`` raises
rather than becoming zero. This module imports nothing from smythe.
"""

from __future__ import annotations

from decimal import Decimal

USD, MICROUSD, NANOUSD = "USD", "microUSD", "nanoUSD"
EXACT, CEILING, FLOOR = "exact", "ceiling", "floor"
# Decimal places of one USD that each unit counts.
_PLACES = {USD: 0, MICROUSD: 6, NANOUSD: 9}
# Python converts at most this many digits between an int and its text by
# default, so no store can record a longer amount. Refusing one early also
# bounds the work a huge exponent would cost.
_MAX_DIGITS = 4300


class MoneyUnitError(ValueError):
    """An amount cannot be converted between money units as requested."""


def _places(unit: object) -> int:
    if type(unit) is not str or unit not in _PLACES:
        raise MoneyUnitError(f"Unknown money unit {unit!r}")
    return _PLACES[unit]


def _check_rounding(rounding: object) -> None:
    if type(rounding) is not str or rounding not in (EXACT, CEILING, FLOOR):
        raise MoneyUnitError(f"Unknown money rounding {rounding!r}")


def _count(value: object, unit: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MoneyUnitError(f"{unit} amount must be a non-negative integer")
    return int(value)


def _round(whole: int, inexact: bool, rounding: str, message: str) -> int:
    if not inexact or rounding == FLOOR:
        return whole
    if rounding == CEILING:
        return whole + 1
    raise MoneyUnitError(message)


def to_units(amount: object, unit: str, rounding: str = EXACT) -> int:
    """Convert a non-negative USD amount to a whole number of ``unit``.

    ``amount`` is a ``Decimal``, an ``int`` or a ``float``.
    """
    places = _places(unit)
    _check_rounding(rounding)
    if isinstance(amount, bool) or not isinstance(amount, (Decimal, int, float)):
        raise MoneyUnitError("USD amount must be a Decimal, int or float")
    if isinstance(amount, float):
        amount = Decimal(repr(float(amount)))
    elif isinstance(amount, int):
        amount = Decimal(int(amount))
    if not amount.is_finite() or amount < 0:
        raise MoneyUnitError("USD amount must be finite and non-negative")
    _, digits, exponent = amount.as_tuple()
    if not any(digits):
        return 0
    # Digits at or before the unit's decimal point; the rest are fractions
    # of one unit.
    whole = len(digits) + exponent + places
    if whole > _MAX_DIGITS:
        raise MoneyUnitError("USD amount is too large to convert")
    count = 0
    if whole > 0:
        count = int("".join(map(str, digits[:whole]))) * 10 ** max(0, whole - len(digits))
    inexact = any(digits[max(0, whole):])
    return _round(count, inexact, rounding, f"USD amount is finer than one {unit}")


def from_units(count: object, unit: str) -> Decimal:
    """Return a whole number of ``unit`` as an exact USD ``Decimal``.

    The result always has the unit's number of decimal places, so 1,500,000
    nanoUSD is ``Decimal("0.001500000")``.
    """
    places = _places(unit)
    digits = Decimal(_count(count, unit)).as_tuple().digits
    return Decimal((0, digits, -places))


def convert(count: object, source: str, target: str, rounding: str = EXACT) -> int:
    """Convert a whole number of ``source`` units to ``target`` units.

    Converting to a finer unit, such as microUSD to nanoUSD, is always exact.
    Converting nanoUSD to microUSD raises for a fraction of one microUSD
    unless a rounding is named: ``CEILING`` for a reservation or exposure,
    which therefore never shrinks, and ``FLOOR`` for a remaining budget.
    """
    shift = _places(target) - _places(source)
    _check_rounding(rounding)
    count = _count(count, source)
    if shift >= 0:
        return count * 10 ** shift
    whole, remainder = divmod(count, 10 ** -shift)
    return _round(whole, bool(remainder), rounding, f"{source} amount is finer than one {target}")
