"""Dedicated tests for WhiteRabbit — the task classifier/router."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from helpers import ClassifierMockProvider, FixedArchitect, MockProvider
from smythe import RoutingDecision as ExportedRoutingDecision, Swarm
from smythe.planner import ArchitectError
from smythe.provider import CompletionResult, Provider
from smythe.router import CLASSIFIER_SYSTEM_PROMPT, ClassifierUsage, RoutingDecision, WhiteRabbit
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
    provider = ClassifierMockProvider("deterministic:my-key")
    router = WhiteRabbit(
        deterministic={"my-key": det},
        autonomous=FixedArchitect("fallback"),
        classifier_provider=provider,
        classifier_model="test",
    )
    task = Task(goal="Deterministic")
    result = await router.aroute(task)
    assert result is det

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert "deterministic:my-key" in call.system
    assert "autonomous" in call.system
    assert call.prompt == "Goal: Deterministic"
    assert call.model == "test"


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


@pytest.mark.parametrize("reply", [
    "deterministic:MyShape", "deterministic:myshape", "DETERMINISTIC:MYSHAPE",
    "Deterministic: MyShape", "`deterministic:MyShape`", "deterministic:MyShape.",
    '"deterministic:MyShape"', "**deterministic:`MyShape`**",
])
def test_deterministic_keys_match_regardless_of_case_and_decoration(reply):
    """The reply used to be lowercased before a case-sensitive key lookup."""
    shape = FixedArchitect("shape")
    router = WhiteRabbit(
        deterministic={"MyShape": shape},
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider(reply),
        classifier_model="test",
    )
    assert router.route(Task(goal="Shape it")) is shape


@pytest.mark.parametrize("reply", [
    "constrained.", "`constrained`", '"constrained"', "**Constrained**", "constrained!\n",
])
def test_decorated_tier_names_are_matched(reply):
    constrained = FixedArchitect("constrained")
    router = WhiteRabbit(
        constrained=constrained,
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider(reply),
        classifier_model="test",
    )
    assert router.route(Task(goal="Constrain it")) is constrained


def test_keys_differing_only_by_case_are_rejected():
    with pytest.raises(ValueError, match="differ by more than case"):
        WhiteRabbit(
            deterministic={"MyShape": FixedArchitect("a"), "myshape": FixedArchitect("b")},
            autonomous=FixedArchitect("fallback"),
        )


@pytest.mark.parametrize("key", ["faq?", "[legacy]", "summary:", "report (short)", "report."])
def test_keys_with_surrounding_punctuation_stay_reachable(key):
    """Replies lose their surrounding punctuation, so stored keys must too."""
    tier = FixedArchitect("tier")
    router = WhiteRabbit(
        deterministic={key: tier},
        autonomous=FixedArchitect("fallback"),
        classifier_provider=ClassifierMockProvider(f"deterministic:{key}"),
        classifier_model="test",
    )
    assert router.route(Task(goal="Route it")) is tier


def test_keys_differing_only_by_surrounding_punctuation_are_rejected():
    with pytest.raises(ValueError, match="surrounding punctuation"):
        WhiteRabbit(
            deterministic={"report": FixedArchitect("a"), "report.": FixedArchitect("b")},
            autonomous=FixedArchitect("fallback"),
        )


@pytest.mark.parametrize("reply", [
    "I think this is a constrained task", "deterministic:missing", "constrained",
])
def test_unmatched_reply_falls_back_with_a_warning(reply, caplog):
    fallback = FixedArchitect("fallback")
    router = WhiteRabbit(
        deterministic={"real": FixedArchitect("real")},
        autonomous=fallback,
        classifier_provider=ClassifierMockProvider(reply),
        classifier_model="test",
    )
    with caplog.at_level("WARNING", logger="smythe.router"):
        assert router.route(Task(goal="Unclear")) is fallback
    assert "falling back to the autonomous architect" in caplog.text


def test_choosing_autonomous_does_not_warn(caplog):
    fallback = FixedArchitect("fallback")
    router = WhiteRabbit(
        autonomous=fallback,
        classifier_provider=ClassifierMockProvider("Autonomous."),
        classifier_model="test",
    )
    with caplog.at_level("WARNING", logger="smythe.router"):
        assert router.route(Task(goal="Open-ended")) is fallback
    assert caplog.text == ""


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


# --- Routing decision records ---


class UsageProvider(Provider):
    """Replies with a fixed classification and the usage it is given."""

    def __init__(self, reply: str, **usage) -> None:
        self.reply = reply
        self.usage = usage
        self.calls = 0

    async def complete(self, system, prompt, model):
        self.calls += 1
        return CompletionResult(text=self.reply, **self.usage)


def _router(reply: str | None, **tiers) -> WhiteRabbit:
    tiers.setdefault("autonomous", FixedArchitect("fallback"))
    provider = None if reply is None else ClassifierMockProvider(reply)
    return WhiteRabbit(classifier_provider=provider, classifier_model="test", **tiers)


def test_decide_without_a_classifier_records_the_fallback():
    router = _router(None, deterministic={"k": FixedArchitect("k")})

    architect, decision = router.decide(Task(goal="Anything"))

    assert architect is router._autonomous
    assert decision.to_dict() == {
        "version": 1, "tier": "autonomous", "method": "fallback_no_classifier",
        "alternatives": ["deterministic:k", "autonomous"],
        "reasons": ["no classifier is configured, so the autonomous tier plans"],
        "policy_version": 1, "matched_token": None, "usage": None, "call_ids": [],
    }


@pytest.mark.asyncio
async def test_adecide_without_a_classifier_or_autonomous_tier_raises():
    with pytest.raises(ArchitectError, match="No autonomous architect configured"):
        await WhiteRabbit().adecide(Task(goal="Anything"))


@pytest.mark.parametrize("reply,tier,token", [
    ("Deterministic: MyShape", "deterministic:MyShape", "deterministic:myshape"),
    ("**deterministic:`MyShape`**", "deterministic:MyShape", "deterministic:myshape"),
    ("`Constrained`.", "constrained", "constrained"),
    ("autonomous", "autonomous", "autonomous"),
])
def test_a_matched_reply_records_its_tier_and_normalized_token(reply, tier, token):
    router = _router(reply, deterministic={"MyShape": FixedArchitect("shape")},
                     constrained=FixedArchitect("constrained"))

    architect, decision = router.decide(Task(goal="Shape it"))

    assert architect is router.route(Task(goal="Shape it"))
    assert (decision.tier, decision.method, decision.matched_token) == (tier, "classifier", token)
    assert decision.alternatives == ("deterministic:MyShape", "constrained", "autonomous")
    assert decision.reasons == (f"the classifier chose {tier}",)
    assert decision.usage == ClassifierUsage(prompt_tokens=5, completion_tokens=2)
    assert decision.call_ids == ()


@pytest.mark.parametrize("reply", [
    "I think this is a constrained task", "deterministic:missing", "constrained",
])
def test_an_unmatched_reply_records_the_fallback_without_the_reply(reply):
    router = _router(reply, deterministic={"real": FixedArchitect("real")})

    architect, decision = router.decide(Task(goal="Unclear"))

    assert architect is router._autonomous
    assert (decision.tier, decision.method, decision.matched_token) == (
        "autonomous", "fallback_unmatched", None)
    assert decision.reasons == ("the classifier reply named no configured tier, so the "
                                "autonomous tier plans",)
    assert reply not in json.dumps(decision.to_dict())
    assert decision.usage == ClassifierUsage(prompt_tokens=5, completion_tokens=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,expected", [
    ({"prompt_tokens": 7, "completion_tokens": 3},
     {"prompt_tokens": 7, "completion_tokens": 3, "cost_usd": None,
      "cost_usd_is_estimate": False, "cost_usd_unknown": False}),
    ({"prompt_tokens": 7, "completion_tokens": 3, "cost_usd": 0.002, "cost_usd_is_estimate": True},
     {"prompt_tokens": 7, "completion_tokens": 3, "cost_usd": 0.002,
      "cost_usd_is_estimate": True, "cost_usd_unknown": False}),
    ({"prompt_tokens": 0, "completion_tokens": 9, "cost_usd": 1, "cost_usd_unknown": True},
     {"prompt_tokens": 0, "completion_tokens": 9, "cost_usd": 1.0,
      "cost_usd_is_estimate": False, "cost_usd_unknown": True}),
])
async def test_adecide_records_the_classifier_usage_the_provider_reported(usage, expected):
    provider = UsageProvider("constrained", **usage)
    constrained = FixedArchitect("constrained")
    router = WhiteRabbit(constrained=constrained, autonomous=FixedArchitect("fallback"),
                         classifier_provider=provider, classifier_model="test")

    architect, decision = await router.adecide(Task(goal="Guided"))

    assert architect is constrained and provider.calls == 1
    assert decision.to_dict()["usage"] == expected
    assert RoutingDecision.from_dict(decision.to_dict()) == decision


class ResultProvider(Provider):
    """Returns the result object it is given, as a provider with its own result type may."""

    def __init__(self, result) -> None:
        self.result = result

    async def complete(self, system, prompt, model):
        return self.result


@pytest.mark.parametrize("result,flags", [
    (CompletionResult(text="constrained", prompt_tokens=1, completion_tokens=1,
                      cost_usd_unknown=None), (False, False)),
    (CompletionResult(text="constrained", prompt_tokens=1, completion_tokens=1,
                      cost_usd_is_estimate=1, cost_usd_unknown=0), (True, False)),
    (SimpleNamespace(text="constrained", prompt_tokens=1, completion_tokens=1), (False, False)),
    (SimpleNamespace(text="constrained", prompt_tokens=1, completion_tokens=1, cost_usd=0.5,
                     cost_usd_unknown=1, native_receipt=["not", "a", "dict"]), (False, True)),
])
def test_routing_accepts_a_classifier_result_with_loose_usage_flags(result, flags):
    constrained = FixedArchitect("constrained")
    router = WhiteRabbit(constrained=constrained, autonomous=FixedArchitect("fallback"),
                         classifier_provider=ResultProvider(result), classifier_model="test")

    assert router.route(Task(goal="Guided")) is constrained
    architect, decision = router.decide(Task(goal="Guided"))

    assert architect is constrained
    recorded = (decision.usage.cost_usd_is_estimate, decision.usage.cost_usd_unknown)
    assert recorded == flags and {type(flag) for flag in recorded} == {bool}
    assert decision.usage.cost_usd == getattr(result, "cost_usd", None)
    assert RoutingDecision.from_dict(decision.to_dict()) == decision
    graph = Swarm(provider=MockProvider(), router=router, artifact_dir=None).plan(Task(goal="Guided"))
    assert graph.nodes[0].label == "constrained: Guided"
    assert graph.planning == {"routing": decision.to_dict()}


@pytest.mark.asyncio
async def test_aroute_returns_the_architect_adecide_chooses():
    det = FixedArchitect("det")
    router = _router("deterministic:k", deterministic={"k": det})
    assert await router.aroute(Task(goal="Route")) is det
    assert (await router.adecide(Task(goal="Route")))[0] is det


def test_routing_records_leave_the_classifier_prompt_unchanged():
    assert hashlib.sha256(CLASSIFIER_SYSTEM_PROMPT.encode()).hexdigest() == (
        "06c5c9280bfbb157a9017e3e4bae9cd932c038c74fe94818e248a41e09dab632")
    provider = ClassifierMockProvider("constrained")
    router = WhiteRabbit(deterministic={"MyShape": FixedArchitect("s")},
                         constrained=FixedArchitect("c"), autonomous=FixedArchitect("a"),
                         classifier_provider=provider, classifier_model="test")
    router.decide(Task(goal="Shape it"))
    assert provider.calls[0].system == (
        "You are a task classifier.  Given a user's goal, decide which planning strategy "
        "to use.  Respond with ONLY one of the following tokens:\n\n- deterministic:MyShape\n"
        "- constrained\n- autonomous\n\nDo not add any other text.\n"
    )


def _record(**changes):
    data = {
        "version": 1, "tier": "constrained", "method": "classifier",
        "alternatives": ["constrained", "autonomous"], "reasons": ["the classifier chose constrained"],
        "policy_version": 1, "matched_token": "constrained",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost_usd": None,
                  "cost_usd_is_estimate": False, "cost_usd_unknown": False},
        "call_ids": [],
    }
    data.update(changes)
    return data


def test_routing_decision_round_trips_and_is_exported():
    assert ExportedRoutingDecision is RoutingDecision
    decision = RoutingDecision.from_dict(_record())
    assert decision.to_dict() == _record()
    durable = RoutingDecision.from_dict(_record(usage=None, call_ids=["call-1"]))
    assert durable.call_ids == ("call-1",) and durable.usage is None
    assert RoutingDecision.from_dict(durable.to_dict()) == durable
    hash(decision)


_USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "cost_usd": None,
          "cost_usd_is_estimate": False, "cost_usd_unknown": False}


@pytest.mark.parametrize("data", [
    _record(version=2), _record(version=True), _record(method="rule"), _record(extra=1),
    {key: value for key, value in _record().items() if key != "call_ids"},
    _record(tier="single"), _record(tier=""), _record(alternatives="constrained"),
    _record(reasons=[""]), _record(reasons="chose"), _record(matched_token=None),
    _record(matched_token=""), _record(policy_version=0), _record(policy_version="1"),
    _record(usage={**_USAGE, "prompt_tokens": -1}),
    _record(usage={**_USAGE, "cost_usd": float("nan")}),
    _record(usage={**_USAGE, "cost_usd_is_estimate": "no"}),
    _record(usage={"prompt_tokens": 5, "completion_tokens": 2}),
    _record(usage={**_USAGE, "raw_reply": "constrained"}),
    _record(usage=[5, 2]), _record(call_ids=[""]), _record(call_ids="call-1"),
    _record(tier="autonomous", method="fallback_unmatched"),
    _record(tier="autonomous", method="fallback_no_classifier", matched_token=None),
    _record(tier="constrained", method="fallback_unmatched", matched_token=None),
    "not a record", None,
])
def test_routing_decision_from_dict_rejects_what_it_cannot_read(data):
    with pytest.raises(ValueError):
        RoutingDecision.from_dict(data)
