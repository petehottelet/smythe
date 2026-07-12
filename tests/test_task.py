"""Tests for Task construction and validation."""

import pytest

from smythe.task import Task


def test_task_requires_goal():
    with pytest.raises(ValueError, match="non-empty goal"):
        Task(goal="")


def test_task_rejects_whitespace_goal():
    with pytest.raises(ValueError, match="non-empty goal"):
        Task(goal="   ")


def test_task_rejects_non_string_goal():
    with pytest.raises(TypeError, match="goal must be a string"):
        Task(goal=None)


def test_task_minimal():
    t = Task(goal="Summarize this document")
    assert t.goal == "Summarize this document"
    assert t.constraints == []
    assert t.context == {}


def test_task_with_constraints():
    t = Task(
        goal="Analyze market trends",
        constraints=["Must cite sources", "Max 500 words"],
    )
    assert len(t.constraints) == 2


def test_task_with_context():
    t = Task(goal="Translate text", context={"language": "fr"})
    assert t.context["language"] == "fr"


def test_task_normalises_goal_and_constraints():
    t = Task(
        goal="  Analyze market trends  ",
        constraints=["  Cite sources  ", "", "   "],
    )

    assert t.goal == "Analyze market trends"
    assert t.constraints == ["Cite sources"]


def test_task_detaches_mutable_inputs():
    constraints = ["Cite sources"]
    context = {"region": "US"}

    task = Task(goal="Analyze market trends", constraints=constraints, context=context)
    constraints.append("Maximum 500 words")
    context["region"] = "EU"

    assert task.constraints == ["Cite sources"]
    assert task.context == {"region": "US"}


def test_task_rejects_string_constraints():
    with pytest.raises(TypeError, match="iterable of strings"):
        Task(goal="Analyze market trends", constraints="Cite sources")


def test_task_rejects_non_mapping_context():
    with pytest.raises(TypeError, match="context must be a mapping"):
        Task(goal="Analyze market trends", context=["region", "US"])
