"""Dedicated tests for WhiteRabbit — the task classifier/router."""

import pytest

from helpers import ClassifierMockProvider, FixedArchitect
from smythe.planner import ArchitectError
from smythe.router import WhiteRabbit
from smythe.task import Task


def test_no_autonomous_and_no_classifier_raises():
    """WhiteRabbit with no autonomous architect and no classifier must raise."""
    router = WhiteRabbit()
    task = Task(goal="Anything")

    with pytest.raises(ArchitectError, match="No autonomous architect configured"):
        router.route(task)


@pytest.mark.asyncio
async def test_aroute_no_classifier_falls_back():
    """aroute() without a classifier provider should return the autonomous architect."""
    autonomous = FixedArchitect("auto")
    router = WhiteRabbit(autonomous=autonomous)
    task = Task(goal="Test")

    result = await router.aroute(task)
    assert result is autonomous


@pytest.mark.asyncio
async def test_aroute_selects_deterministic():
    """aroute() should select the correct deterministic architect by key."""
    det = FixedArchitect("det")
    router = WhiteRabbit(
        deterministic={"my-key": det},
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider("deterministic:my-key"),
        classifier_model="test",
    )
    task = Task(goal="Deterministic")
    result = await router.aroute(task)
    assert result is det


@pytest.mark.asyncio
async def test_aroute_unknown_deterministic_key_falls_back():
    """If classifier returns a deterministic key that doesn't exist, fall back."""
    fallback = FixedArchitect("fallback")
    router = WhiteRabbit(
        deterministic={"real-key": FixedArchitect("real")},
        autonomous=fallback,
        classifier_provider=ClassifierMockProvider("deterministic:nonexistent"),
        classifier_model="test",
    )
    task = Task(goal="Bad key")
    result = await router.aroute(task)
    assert result is fallback


@pytest.mark.asyncio
async def test_aroute_constrained_when_none_falls_back():
    """'constrained' classification with no constrained architect should fall back."""
    fallback = FixedArchitect("fallback")
    router = WhiteRabbit(
        autonomous=fallback,
        classifier_provider=ClassifierMockProvider("constrained"),
        classifier_model="test",
    )
    task = Task(goal="Constrained but missing")
    result = await router.aroute(task)
    assert result is fallback


def test_build_options_includes_all_tiers():
    """_build_options() should list deterministic keys, constrained, and autonomous."""
    router = WhiteRabbit(
        deterministic={"a": FixedArchitect("a"), "b": FixedArchitect("b")},
        constrained=FixedArchitect("c"),
        autonomous=FixedArchitect("auto"),
    )
    options = router._build_options()
    assert "deterministic:a" in options
    assert "deterministic:b" in options
    assert "constrained" in options
    assert "autonomous" in options


def test_build_options_without_constrained():
    """_build_options() should omit constrained when not set."""
    router = WhiteRabbit(
        deterministic={"x": FixedArchitect("x")},
        autonomous=FixedArchitect("auto"),
    )
    options = router._build_options()
    assert "constrained" not in options
    assert "deterministic:x" in options
    assert "autonomous" in options


@pytest.mark.asyncio
async def test_classifier_case_insensitive():
    """Classification should be case-insensitive."""
    constrained = FixedArchitect("constrained")
    router = WhiteRabbit(
        constrained=constrained,
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider("  CONSTRAINED  "),
        classifier_model="test",
    )
    task = Task(goal="CasE test")
    result = await router.aroute(task)
    assert result is constrained


def test_sync_route_delegates_to_aroute():
    """Sync route() should produce the same result as aroute()."""
    det = FixedArchitect("det")
    router = WhiteRabbit(
        deterministic={"k": det},
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider("deterministic:k"),
        classifier_model="test",
    )
    task = Task(goal="Sync test")
    result = router.route(task)
    assert result is det
