"""Tests for Task construction and validation."""

import pytest

from smythe.task import Task


def test_task_requires_goal():
    with pytest.raises(ValueError, match="non-empty goal"):
        Task(goal="")


def test_task_rejects_whitespace_goal():
    with pytest.raises(ValueError, match="non-empty goal"):
        Task(goal="   ")


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
