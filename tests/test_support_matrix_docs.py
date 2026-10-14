"""Every test the support matrix page cites exists under that node id."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "support-matrix.md"
NODE_ID = re.compile(r"`(tests/test_\w+\.py::\w+(?:\[[^\]`]+\])?)`")


def cited_node_ids():
    return sorted(set(NODE_ID.findall(PAGE.read_text(encoding="utf-8"))))


def collected_node_ids(files):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *files],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


def test_page_cites_tests_for_every_execution_path():
    cited = cited_node_ids()
    for module in ("test_support_matrix.py", "test_provider.py", "test_jobs_runtime_support.py",
                   "test_workflow_provider.py"):
        assert any(node.startswith(f"tests/{module}::") for node in cited), module


def test_the_external_actions_section_cites_each_path_and_the_conformance_tests():
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("\n## External actions\n", 1)[1].split("\n## ", 1)[0]
    cited = set(NODE_ID.findall(section))
    for module in ("test_action_approvals.py", "test_action_conformance.py", "test_action_reconciliation.py",
                   "test_action_runner.py", "test_action_process_loss.py", "test_support_matrix.py"):
        assert any(node.startswith(f"tests/{module}::") for node in cited), module
    for path in ("Durable text workflows", "Ordinary runs", "Jobs"):
        assert f"| {path} |" in section, path


def test_the_lineage_and_repair_section_cites_each_path_and_its_tests():
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("\n## Lineage and repair\n", 1)[1].split("\n## ", 1)[0]
    cited = set(NODE_ID.findall(section))
    for module in ("test_workflow_lineage.py", "test_reuse_decisions.py", "test_repair_plan.py",
                   "test_workflow_repair.py", "test_workflow_reuse.py",
                   "test_workflow_repair_process_loss.py", "test_lineage_retention.py",
                   "test_archive_maintenance.py", "test_support_matrix.py", "test_jobs_runner.py",
                   "test_jobs_cli.py"):
        assert any(node.startswith(f"tests/{module}::") for node in cited), module
    for path in ("Durable text workflows", "Ordinary runs", "Jobs"):
        assert f"| {path} |" in section, path


def test_the_templates_section_cites_each_path_and_the_learning_boundaries():
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("\n## Templates and policy versions\n", 1)[1].split("\n## ", 1)[0]
    cited = set(NODE_ID.findall(section))
    for module in ("test_template_lifecycle.py", "test_template_promotion.py",
                   "test_template_records.py", "test_optimize_workflow_eval.py",
                   "test_learning_boundaries.py", "test_support_matrix.py"):
        assert any(node.startswith(f"tests/{module}::") for node in cited), module
    for path in ("Durable text workflows", "Ordinary runs", "Jobs"):
        assert f"| {path} |" in section, path


OWNERSHIP_PATHS = ["Ordinary runs", "Durable text workflows", "Jobs"]
OWNERSHIP_ROWS = [
    "Admission and scheduling", "Logical and attempt identity", "Permission to dispatch",
    "Calling again", "Settling cost and its unit", "Keeping results for recovery",
    "Stop and cancellation", "Waits and timers", "Ownership and stale writers",
    "Recovery after process loss", "Observing without controlling",
    "Upgrading with work in flight",
]


def ownership_section():
    text = PAGE.read_text(encoding="utf-8")
    assert "\n## Who owns each execution responsibility\n" in text
    section = text.split("\n## Who owns each execution responsibility\n", 1)[1].split("\n## ", 1)[0]
    # The section follows the retry-owner list and precedes the durable components.
    assert text.index("Who calls again after a failure") < text.index(section)
    assert text.index(section) < text.index("\n## Durable text workflow components\n")
    return section


def test_the_ownership_table_has_a_cell_for_each_path_and_responsibility():
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in ownership_section().splitlines() if line.startswith("|")]
    header, rule, *body = rows
    assert header == ["Responsibility", *OWNERSHIP_PATHS]
    assert len(rule) == 4 and all(set(cell) == {"-"} for cell in rule)
    assert [row[0] for row in body] == OWNERSHIP_ROWS
    # "None" is stated where nothing owns a responsibility; no cell is left blank.
    assert all(len(row) == 4 and all(row) for row in body), body


def test_every_ownership_cell_cites_a_test_for_its_path():
    section = ownership_section()
    tests = section.split("\nTests:\n", 1)[1]
    items = {}
    for block in tests.strip().split("\n- "):
        name, _, citations = block.removeprefix("- ").partition(":")
        items[name] = " ".join(citations.split())
    assert list(items) == OWNERSHIP_ROWS
    for name, citations in items.items():
        # One part per path, in the table's order, each naming at least one test.
        parts = citations.split("; ")
        assert len(parts) == len(OWNERSHIP_PATHS), name
        for part, path in zip(parts, OWNERSHIP_PATHS):
            assert part.startswith(path if path == "Jobs" else path.lower()), (name, part)
            assert NODE_ID.search(part), (name, part)


def test_every_cited_test_exists():
    cited = cited_node_ids()
    collected = collected_node_ids(sorted({node.split("::")[0] for node in cited}))
    # A citation without parameters names every case of a parametrized test.
    missing = [node for node in cited if node not in collected
               and not any(item.startswith(node + "[") for item in collected)]
    assert missing == []
