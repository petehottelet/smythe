"""Publishing cannot bypass tag identity or exact-commit CI qualification."""

import pytest

from tools.release_gate import validate_release

SHA = "a" * 40
REPO = "petehottelet/smythe"


def good_run():
    return {"head_sha": SHA, "event": "push", "head_branch": "main",
            "head_repository": {"full_name": REPO}, "status": "completed",
            "conclusion": "success", "html_url": "https://github.com/example/run"}


def test_exact_qualified_release_is_accepted():
    assert validate_release("refs/tags/v0.8.0rc1", "0.8.0rc1", SHA, [good_run()], REPO)


@pytest.mark.parametrize("change", [
    {"head_sha": "b" * 40}, {"event": "pull_request"}, {"head_branch": "candidate"},
    {"head_repository": {"full_name": "someone/fork"}}, {"status": "in_progress"},
    {"conclusion": "failure"}, {"conclusion": "cancelled"}, {"conclusion": "skipped"},
])
def test_unqualified_runs_do_not_authorize_publishing(change):
    with pytest.raises(ValueError, match="No successful"):
        validate_release("refs/tags/v0.8.0", "0.8.0", SHA, [good_run() | change], REPO)


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/tags/v0.7.0", "refs/tags/v0.8.0rc1"])
def test_branch_dispatch_and_mismatched_tags_are_rejected(ref):
    with pytest.raises(ValueError, match="tag matching"):
        validate_release(ref, "0.8.0", SHA, [good_run()], REPO)


def test_missing_ci_and_abbreviated_sha_fail_closed():
    with pytest.raises(ValueError, match="No successful"):
        validate_release("refs/tags/v0.8.0", "0.8.0", SHA, [], REPO)
    with pytest.raises(ValueError, match="full checked-out"):
        validate_release("refs/tags/v0.8.0", "0.8.0", SHA[:7], [good_run()], REPO)
