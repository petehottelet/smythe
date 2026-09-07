"""Frozen task/source packs; evaluator material never enters provider input."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re

from ._json import (
    CampaignPlanError, TEXT_HASH_POLICY, canonical, digest, keys, read_json, strict_json, strings,
)

DATA_DIR = Path(__file__).with_name("data")
SHAPES = ("single-step", "wide-parallel", "deep-serial", "adversarial", "fan-in-verify")
_IDENTITY = re.compile(r"[a-z][a-z0-9-]{1,63}")
_HASH = re.compile(r"[0-9a-f]{64}")
MAX_OUTPUT_BYTES = 128 * 1024


@dataclass(frozen=True)
class TaskCase:
    task_id: str
    stage: str
    shape: str
    task_json: str
    source_json: str
    rubric_json: str
    checks_json: str
    task_sha256: str
    source_sha256: str

    @property
    def rubric(self):
        return strict_json(self.rubric_json)

    @property
    def checks(self):
        return strict_json(self.checks_json)


@dataclass(frozen=True)
class TaskPack:
    tasks: tuple[TaskCase, ...]
    manifest_sha256: str
    file_hashes: tuple[tuple[str, str], ...]

    def for_stage(self, stage: str) -> tuple[TaskCase, ...]:
        if stage not in ("pilot", "main"):
            raise CampaignPlanError("Stage must be pilot or main")
        return tuple(task for task in self.tasks if task.stage == stage)


def _path(root: Path, name: str) -> Path:
    if type(name) is not str or "\\" in name or ":" in name:
        raise CampaignPlanError("Invalid pack file path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
        raise CampaignPlanError("Pack paths must name one local task or source file")
    if relative.parts[0] not in {"tasks", "sources"} or relative.suffix != ".json":
        raise CampaignPlanError("Invalid pack directory or extension")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise CampaignPlanError("Pack links are not allowed")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise CampaignPlanError("Pack path escapes its root")
    return candidate


def _validate_check(check):
    if type(check) is not dict or check.get("kind") not in {
        "equals", "number", "nonempty", "object", "length", "keys", "citations",
    }:
        raise CampaignPlanError("Unknown deterministic check")
    kind = check["kind"]
    extra = {"equals": {"expected"}, "number": {"expected", "tolerance"},
             "nonempty": {"min_chars"}, "object": {"minimum"}, "length": {"expected"}, "keys": {"expected"},
             "citations": {"allowed", "minimum"}}[kind]
    keys(check, {"id", "kind", "path"} | extra, "check")
    if type(check["id"]) is not str or not _IDENTITY.fullmatch(check["id"]):
        raise CampaignPlanError("Invalid check identity")
    if (type(check["path"]) is not list or len(check["path"]) > 16
            or any(not (type(part) is str and part or type(part) is int and part >= 0)
                   for part in check["path"])):
        raise CampaignPlanError("Invalid check path")
    if kind == "number":
        for field in ("expected", "tolerance"):
            value = check[field]
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise CampaignPlanError("Numeric check requires finite numbers")
        if check["tolerance"] < 0:
            raise CampaignPlanError("Check tolerance must be nonnegative")
    if kind in {"nonempty", "object", "length", "citations"}:
        field = {"nonempty": "min_chars", "object": "minimum", "length": "expected", "citations": "minimum"}[kind]
        if type(check[field]) is not int or not 0 <= check[field] <= 10000:
            raise CampaignPlanError("Invalid check count")
    if kind in {"keys", "citations"}:
        values = check["expected" if kind == "keys" else "allowed"]
        strings(values, "Check names")
        if len(set(values)) != len(values):
            raise CampaignPlanError("Duplicate check names")


def _case(data, raw, source, source_raw):
    keys(data, {"version", "id", "stage", "shape", "task", "source_file", "rubric", "checks"}, "task")
    if type(data["version"]) is not int or data["version"] != 1:
        raise CampaignPlanError("Unsupported task version")
    if type(data["id"]) is not str or not _IDENTITY.fullmatch(data["id"]):
        raise CampaignPlanError("Invalid task ID")
    if data["stage"] not in ("pilot", "main") or data["shape"] not in SHAPES:
        raise CampaignPlanError("Invalid task stage or shape")
    task = data["task"]
    keys(task, {"goal", "constraints", "done_when"}, "provider task")
    if type(task["goal"]) is not str or not task["goal"].strip():
        raise CampaignPlanError("Task goal must be nonempty")
    strings(task["constraints"], "Task constraints")
    strings(task["done_when"], "Task done_when", minimum=0)
    keys(source, {"version", "id", "provenance", "documents"}, "source pack")
    if type(source["version"]) is not int or source["version"] != 1 or source["id"] != data["id"]:
        raise CampaignPlanError("Source pack identity mismatch")
    if source["provenance"] != "Original synthetic benchmark material; no real-world claims.":
        raise CampaignPlanError("Unsupported source provenance")
    if type(source["documents"]) is not list or not 1 <= len(source["documents"]) <= 16:
        raise CampaignPlanError("Source pack requires 1..16 documents")
    source_ids = set()
    for document in source["documents"]:
        keys(document, {"id", "title", "content"}, "source document")
        strings(list(document.values()), "Source document fields")
        if document["id"] in source_ids:
            raise CampaignPlanError("Duplicate source document identity")
        source_ids.add(document["id"])
    rubric = data["rubric"]
    if type(rubric) is not list or not 3 <= len(rubric) <= 8:
        raise CampaignPlanError("Rubric requires 3..8 criteria")
    for criterion in rubric:
        keys(criterion, {"id", "criterion", "anchors"}, "rubric criterion")
        strings([criterion["id"], criterion["criterion"]], "Rubric labels")
        keys(criterion["anchors"], {"0", "1", "2", "3", "4"}, "ordinal rubric anchors")
        strings(list(criterion["anchors"].values()), "Rubric anchors")
    if len({item["id"] for item in rubric}) != len(rubric):
        raise CampaignPlanError("Duplicate rubric identity")
    checks = data["checks"]
    if type(checks) is not list or not 1 <= len(checks) <= 64:
        raise CampaignPlanError("Task requires 1..64 deterministic checks")
    for check in checks:
        _validate_check(check)
        if check["kind"] == "citations" and not set(check["allowed"]) <= source_ids:
            raise CampaignPlanError("Check cites unknown source documents")
    if len({check["id"] for check in checks}) != len(checks):
        raise CampaignPlanError("Duplicate check identity")
    return TaskCase(data["id"], data["stage"], data["shape"], canonical(task), canonical(source),
                    canonical(rubric), canonical(checks), digest(raw), digest(source_raw))


def load_task_pack(directory: str | Path = DATA_DIR) -> TaskPack:
    root = Path(directory)
    manifest, raw_manifest = read_json(root / "pack-manifest.json")
    keys(manifest, {"version", "status", "text_hash_policy", "task_files", "file_sha256"}, "pack manifest")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise CampaignPlanError("Unsupported pack version")
    if manifest["status"] != "frozen-preparation-candidate":
        raise CampaignPlanError("Unexpected pack status")
    if manifest["text_hash_policy"] != TEXT_HASH_POLICY:
        raise CampaignPlanError("Unsupported pack text hash policy")
    strings(manifest["task_files"], "Task files")
    if len(manifest["task_files"]) != 13 or len(set(manifest["task_files"])) != 13:
        raise CampaignPlanError("Pack requires exactly 13 distinct tasks")
    hashes = manifest["file_sha256"]
    if type(hashes) is not dict or len(hashes) != 26:
        raise CampaignPlanError("Pack requires 13 task and 13 source hashes")
    files = {}
    for name, expected in hashes.items():
        if type(expected) is not str or not _HASH.fullmatch(expected):
            raise CampaignPlanError("Invalid pack SHA-256")
        data, raw = read_json(_path(root, name))
        if digest(raw) != expected:
            raise CampaignPlanError(f"Pack hash mismatch: {name}")
        files[name] = (data, raw)
    observed = {path.relative_to(root).as_posix() for kind in ("tasks", "sources")
                for path in (root / kind).glob("*.json")}
    if observed != set(files):
        raise CampaignPlanError("Pack contains unmanifested or missing files")
    cases, used = [], set()
    for name in manifest["task_files"]:
        if name not in files or not name.startswith("tasks/"):
            raise CampaignPlanError("Manifest task file missing")
        data, raw = files[name]
        source_name = data.get("source_file") if type(data) is dict else None
        if type(source_name) is not str or not source_name.startswith("sources/") or source_name not in files:
            raise CampaignPlanError("Manifest source file missing")
        if source_name in used:
            raise CampaignPlanError("Tasks must have distinct source packs")
        used.add(source_name)
        cases.append(_case(data, raw, *files[source_name]))
    if len({case.task_id for case in cases}) != 13:
        raise CampaignPlanError("Duplicate task identity")
    if Counter(case.stage for case in cases) != {"pilot": 3, "main": 10}:
        raise CampaignPlanError("Expected three pilot and ten held-out tasks")
    if Counter(case.shape for case in cases if case.stage == "main") != dict.fromkeys(SHAPES, 2):
        raise CampaignPlanError("Held-out pack requires two tasks per shape")
    if len({case.source_json for case in cases}) != 13:
        raise CampaignPlanError("Source packs must differ")
    return TaskPack(tuple(sorted(cases, key=lambda case: case.task_id)), digest(raw_manifest),
                    tuple(sorted(hashes.items())))


def provider_task(case: TaskCase) -> dict:
    """Return only authored requirements and sources, never evaluator answers."""
    task = strict_json(case.task_json)
    task["context"] = {"source_pack": strict_json(case.source_json)}
    return task


def _at(value, path):
    for part in path:
        if type(value) is list and type(part) is int:
            value = value[part]
        elif type(value) is dict and type(part) is str:
            value = value[part]
        else:
            raise KeyError(part)
    return value


def check_output(case: TaskCase, text: str) -> dict:
    """Check format/facts only. A pass is not a rubric or task-acceptance verdict."""
    failed = []
    try:
        if type(text) is not str or len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise CampaignPlanError("Output must be text within the 128 KiB check limit")
        value = strict_json(text)
    except (CampaignPlanError, UnicodeError):
        return {"deterministic_passed": False, "failed_checks": ["json-output"],
                "quality_evaluated": False, "accepted": None}
    for check in case.checks:
        try:
            actual = _at(value, check["path"])
            kind = check["kind"]
            if kind == "equals":
                passed = canonical(actual) == canonical(check["expected"])
            elif kind == "number":
                passed = (type(actual) in (int, float) and math.isfinite(actual)
                          and abs(actual - check["expected"]) <= check["tolerance"])
            elif kind == "nonempty":
                passed = type(actual) is str and len(actual.strip()) >= check["min_chars"]
            elif kind == "object":
                passed = type(actual) is dict and len(actual) >= check["minimum"]
            elif kind == "length":
                passed = type(actual) in (list, dict) and len(actual) == check["expected"]
            elif kind == "keys":
                passed = type(actual) is dict and set(actual) == set(check["expected"])
            else:
                passed = (type(actual) is list and all(type(item) is str for item in actual)
                          and len(set(actual)) >= check["minimum"]
                          and len(actual) == len(set(actual))
                          and set(actual) <= set(check["allowed"]))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            passed = False
        if not passed:
            failed.append(check["id"])
    return {"deterministic_passed": not failed, "failed_checks": failed,
            "quality_evaluated": False, "accepted": None}
