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


def test_every_cited_test_exists():
    cited = cited_node_ids()
    collected = collected_node_ids(sorted({node.split("::")[0] for node in cited}))
    # A citation without parameters names every case of a parametrized test.
    missing = [node for node in cited if node not in collected
               and not any(item.startswith(node + "[") for item in collected)]
    assert missing == []
