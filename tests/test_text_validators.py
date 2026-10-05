"""Built-in text and JSON final validators: versioned code interpreting JSON specs."""

from __future__ import annotations

import inspect
import json
import time
import typing

import pytest

import smythe
import smythe.validators
from smythe import (
    AcceptancePolicy,
    JSONValidator,
    Swarm,
    TextValidator,
    ValidationSubject,
    ValidatorIdentity,
)
from smythe.acceptance import check_validator
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import CompletionResult, Provider
from smythe.validation import ValidationOutcome, ValidationReport

TEXT_DEFAULTS = {"min_chars": 1, "max_chars": None, "sections": [], "ordered": True,
                 "min_section_chars": 1}


def check(validator, output: str | bytes) -> ValidationReport:
    subject = (ValidationSubject(output) if isinstance(output, bytes)
               else ValidationSubject.from_text(output))
    report = validator.validate(subject)
    assert report.validator == validator.identity
    assert report.outcome is not ValidationOutcome.ERROR
    return report


def codes(report: ValidationReport) -> list[str]:
    return [finding.code for finding in report.findings]


# --- identity and specification -----------------------------------------


def test_identity_names_the_code_version_and_the_spec_with_its_defaults():
    validator = TextValidator()
    assert validator.identity == ValidatorIdentity.for_config("smythe.text-spec", "1", TEXT_DEFAULTS)
    assert validator.spec == TEXT_DEFAULTS
    assert TextValidator({"min_chars": 1, "ordered": True}).identity == validator.identity
    assert TextValidator({"sections": ["Summary"]}).identity != validator.identity
    renamed = TextValidator(name="summary-sections")
    assert (renamed.identity.name, renamed.identity.config_sha256) == (
        "summary-sections", validator.identity.config_sha256)
    assert JSONValidator().identity == ValidatorIdentity.for_config(
        "smythe.json-spec", "1", {"checks": [], "max_bytes": 1_048_576})
    assert (smythe.validators.TEXT_VALIDATOR_VERSION, smythe.validators.JSON_VALIDATOR_VERSION) == ("1", "1")


def test_json_check_defaults_are_part_of_the_identity():
    short = JSONValidator({"checks": [{"id": "title", "kind": "nonempty", "path": ["title"]}]})
    explicit = JSONValidator({"checks": [
        {"id": "title", "kind": "nonempty", "path": ["title"], "min_chars": 1},
    ]})
    assert short.identity == explicit.identity
    assert short.spec["checks"] == [{"id": "title", "kind": "nonempty", "path": ["title"], "min_chars": 1}]
    longer = JSONValidator({"checks": [
        {"id": "title", "kind": "nonempty", "path": ["title"], "min_chars": 2},
    ]})
    assert longer.identity != short.identity


def test_the_spec_is_copied_when_the_validator_is_built_and_when_it_is_read():
    spec = {"sections": ["Summary"]}
    validator = TextValidator(spec)
    identity = validator.identity
    spec["sections"].append("Risks")
    validator.spec["sections"].append("Risks")
    assert validator.spec["sections"] == ["Summary"] and validator.identity == identity
    assert check(validator, "# Summary\nDone.").passed


def test_built_in_validators_register_in_a_strict_policy():
    text, document = TextValidator({"sections": ["Summary"]}), JSONValidator()
    assert check_validator(text) == text.identity
    policy = AcceptancePolicy(mode="strict", final_validators=[text, document])
    assert [item["name"] for item in policy.to_dict()["final_validators"]] == [
        "smythe.json-spec", "smythe.text-spec"]
    with pytest.raises(ValueError, match="duplicate validator names"):
        AcceptancePolicy(mode="strict", final_validators=[text, TextValidator()])
    AcceptancePolicy(mode="strict", final_validators=[text, TextValidator(name="length")])


def test_public_names_and_type_hints():
    for name in ("TextValidator", "JSONValidator"):
        assert name in smythe.__all__ and name in smythe.validators.__all__
    for kind in (TextValidator, JSONValidator):
        typing.get_type_hints(kind.__init__)
        for member in vars(kind).values():
            member = member.fget if isinstance(member, property) else member
            if inspect.isfunction(member):
                typing.get_type_hints(member)


@pytest.mark.parametrize("spec,message", [
    ([], "JSON object"),
    ({"heading": "Summary"}, "unknown keys"),
    ({"min_chars": -1}, "min_chars"),
    ({"min_chars": True}, "min_chars"),
    ({"min_chars": 1.0}, "min_chars"),
    ({"min_chars": 1_000_000_001}, "min_chars"),
    ({"max_chars": 1_000_000_001}, "max_chars"),
    ({"max_chars": 0, "min_chars": 1}, "max_chars"),
    ({"max_chars": "10"}, "max_chars"),
    ({"min_section_chars": None}, "min_section_chars"),
    ({"sections": "Summary"}, "sections"),
    ({"sections": ["Summary", ""]}, "sections"),
    ({"sections": ["Summary", "Summary"]}, "sections"),
    ({"sections": ["Summary", "**summary:**"]}, "distinct"),
    ({"sections": ["***"]}, "nonempty"),
    ({"sections": ["x"] * 257}, "sections"),
    ({"ordered": 1}, "ordered"),
    ({"min_chars": float("nan")}, "NaN"),
    ({"sections": [object()]}, "JSON data"),
])
def test_text_spec_is_checked_when_the_validator_is_built(spec, message):
    with pytest.raises(ValueError, match=message):
        TextValidator(spec)


# --- text: length -------------------------------------------------------


def test_default_text_validator_requires_nonempty_output():
    validator = TextValidator()
    assert check(validator, "Done.").passed
    for empty in ("", " \n\t "):
        report = check(validator, empty)
        assert report.outcome is ValidationOutcome.REJECTED
        assert codes(report) == ["text_too_short"]
        assert (report.findings[0].expected, report.findings[0].observed) == (1, 0)
    assert check(TextValidator({"min_chars": 0}), "").passed


def test_length_bounds_count_characters_after_surrounding_whitespace():
    long_report = TextValidator({"min_chars": 2_000, "max_chars": 12_000})
    assert check(long_report, "x" * 12_000).passed
    assert codes(check(long_report, "x" * 12_001)) == ["text_too_long"]
    validator = TextValidator({"min_chars": 3, "max_chars": 5})
    assert check(validator, "  abc  ").passed
    assert check(validator, "\nébcde\n").passed  # characters, not bytes
    assert codes(check(validator, " ab ")) == ["text_too_short"]
    too_long = check(validator, "abcdef")
    assert codes(too_long) == ["text_too_long"]
    assert (too_long.findings[0].expected, too_long.findings[0].observed) == (5, 6)


def test_bytes_that_are_not_utf8_are_rejected_not_undecided():
    for validator in (TextValidator(), JSONValidator()):
        report = check(validator, b"\xff\xfe")
        assert report.outcome is ValidationOutcome.REJECTED
        assert codes(report) == ["not_utf8"]


# --- text: sections -----------------------------------------------------

ARGUMENT_AND_CRITIQUE = TextValidator({"sections": ["Argument", "Critique"]})


def test_required_sections_present_in_order_pass():
    output = (
        "# Report\n\nIntro.\n\n## Argument\nWe should ship.\n\n"
        "## Evidence\nTwo trials.\n\n## Critique\nThe trials were small.\n"
    )
    assert check(ARGUMENT_AND_CRITIQUE, output).passed


def test_an_increment_without_the_earlier_section_is_rejected():
    report = check(ARGUMENT_AND_CRITIQUE, "## Critique\nThe argument overstates the evidence.")
    assert report.outcome is ValidationOutcome.REJECTED
    assert codes(report) == ["section_missing"]
    assert report.findings[0].expected == "Argument"
    assert report.findings[0].message == "required section 'Argument' is missing"


def test_sections_out_of_order_are_rejected_unless_order_is_not_required():
    output = "## Critique\nWeak.\n\n## Argument\nShip it."
    report = check(ARGUMENT_AND_CRITIQUE, output)
    assert codes(report) == ["section_out_of_order"]
    assert report.findings[0].message == "section 'Critique' must follow 'Argument'"
    assert report.findings[0].observed == ["critique", "argument"]
    assert check(TextValidator({"sections": ["Argument", "Critique"], "ordered": False}), output).passed
    # A later repetition in the required order satisfies it.
    assert check(ARGUMENT_AND_CRITIQUE, output + "\n\n## Critique\nStill weak.").passed


def test_out_of_order_findings_name_each_required_heading_once():
    """Each such finding used to list every required heading in the output,
    so 255 findings over 100,000 repeated headings held 25 million entries."""
    sections = [f"Part {index}" for index in range(1, 257)]
    output = "".join(f"# Part {index}\nx\n" for index in range(256, 1, -1)) + "# Part 1\nx\n" * 100_000
    report = check(TextValidator({"sections": sections}), output)
    assert codes(report) == ["section_out_of_order"] * 255
    in_output_order = [section.casefold() for section in reversed(sections)]
    assert all(finding.observed == in_output_order for finding in report.findings)
    repeated = check(ARGUMENT_AND_CRITIQUE, "## Critique\nWeak.\n" * 3 + "## Argument\nShip it.\n" * 3)
    assert codes(repeated) == ["section_out_of_order"]
    assert repeated.findings[0].observed == ["critique", "argument"]


def test_changing_a_finding_does_not_change_the_validator():
    validator = TextValidator({"sections": ["Argument", "Critique"]})
    output = "## Critique\nWeak.\n\n## Argument\nShip it."
    first = check(validator, output).findings[0]
    first.expected.append("Sources")
    first.observed.append("sources")
    assert validator.spec["sections"] == ["Argument", "Critique"]
    again = check(validator, output).findings[0]
    assert (again.expected, again.observed) == (["Argument", "Critique"], ["critique", "argument"])


@pytest.mark.parametrize("heading", [
    "## Argument", "# argument", "###### ARGUMENT", "   ## Argument", "## **Argument**",
    "## Argument:", "## __Argument__:", "## **Argument**:", "## Argument ##", "##\tArgument",
    "## Argument  ", "##  Argument #", "## Argument\t###\t", "## **Argument:** ##", "## Argument ::",
])
def test_heading_matching_ignores_case_spacing_emphasis_and_a_final_colon(heading):
    assert check(ARGUMENT_AND_CRITIQUE, f"{heading}\nShip it.\n## Critique\nWeak.").passed


@pytest.mark.parametrize("section", [
    "## Sources", "# Sources", "###### Sources", "  ## Sources  ", "##\tSources", "## Sources ##",
    "## **Sources:**",
])
def test_a_section_written_as_a_heading_line_requires_that_heading(section):
    """The '#' marker used to stay in the required title, which no heading
    title ever holds, so such a section was always reported missing."""
    validator = TextValidator({"sections": ["Argument", section]})
    bare = TextValidator({"sections": ["Argument", section.strip().lstrip("#").strip(" \t#")]})
    assert validator.spec == bare.spec and validator.identity == bare.identity
    assert check(validator, "## Argument\nShip it.\n\n### Sources\nThe pilot report.").passed
    report = check(validator, "## Argument\nShip it.")
    assert codes(report) == ["section_missing"]
    assert report.findings[0].expected == bare.spec["sections"][1]


@pytest.mark.parametrize("section,title", [
    ("#1 priority", "#1 priority"), ("C#", "C#"), ("####### Sources", "####### Sources"),
])
def test_a_section_name_that_is_not_a_heading_line_keeps_its_hashes(section, title):
    validator = TextValidator({"sections": [section]})
    assert validator.spec["sections"] == [title]
    assert check(validator, f"## {title}\nText.").passed


@pytest.mark.parametrize("sections,message", [
    (["##"], "nonempty"), (["## "], "nonempty"), (["## #"], "nonempty"),
    (["Sources", "## Sources"], "distinct"),
])
def test_heading_line_sections_must_still_name_distinct_headings(sections, message):
    with pytest.raises(ValueError, match=message):
        TextValidator({"sections": sections})


@pytest.mark.parametrize("line", [
    "Argument", "**Argument**", "#Argument", "####### Argument", "    ## Argument",
    "> ## Argument", "- ## Argument", "## Arguments", "## The Argument", "## Argument#",
    "## Argument # #", "## :Argument",
])
def test_lines_that_are_not_markdown_headings_do_not_count(line):
    report = check(ARGUMENT_AND_CRITIQUE, f"{line}\nShip it.\n## Critique\nWeak.")
    assert codes(report) == ["section_missing"]


def test_headings_inside_fenced_code_do_not_count():
    fenced = "```markdown\n## Argument\nShip it.\n```\n\n## Critique\nWeak."
    assert codes(check(ARGUMENT_AND_CRITIQUE, fenced)) == ["section_missing"]
    tildes = "~~~~\n## Argument\n~~~\nstill code\n~~~~\n## Critique\nWeak."
    assert codes(check(ARGUMENT_AND_CRITIQUE, tildes)) == ["section_missing"]
    # A closed fence ends the code block; a backtick fence is not closed by tildes.
    closed = "```\ncode\n```\n## Argument\nShip it.\n## Critique\nWeak."
    assert check(ARGUMENT_AND_CRITIQUE, closed).passed
    unclosed = "```\ncode\n~~~\n## Argument\nShip it.\n## Critique\nWeak."
    assert codes(check(ARGUMENT_AND_CRITIQUE, unclosed)) == ["section_missing", "section_missing"]


def test_a_required_section_needs_body_text_before_the_next_heading_at_its_level():
    empty = check(ARGUMENT_AND_CRITIQUE, "## Argument\n\n## Critique\nWeak.")
    assert codes(empty) == ["section_empty"]
    assert (empty.findings[0].expected, empty.findings[0].observed) == (1, 0)
    # A subsection belongs to its section's body; a heading at the same level ends it.
    nested = "## Argument\n### Detail\nShip it.\n## Critique\nWeak."
    assert check(ARGUMENT_AND_CRITIQUE, nested).passed
    parent = "## Argument\n# Critique\nWeak."
    assert codes(check(ARGUMENT_AND_CRITIQUE, parent)) == ["section_empty"]
    # The last section runs to the end of the output.
    assert codes(check(ARGUMENT_AND_CRITIQUE, "## Argument\nShip it.\n## Critique\n  \n")) == [
        "section_empty"]
    longer = TextValidator({"sections": ["Argument"], "min_section_chars": 10})
    assert codes(check(longer, "## Argument\nShip it.")) == ["section_empty"]
    assert check(TextValidator({"sections": ["Argument"], "min_section_chars": 0}), "## Argument").passed


def test_every_failed_check_is_reported():
    validator = TextValidator({"sections": ["Argument", "Critique", "Sources"], "max_chars": 20})
    report = check(validator, "## Critique\n\n## Argument\nShip it, because the pilot worked.")
    assert codes(report) == ["text_too_long", "section_out_of_order", "section_missing"]


def test_line_endings_and_windows_newlines_are_lines():
    assert check(ARGUMENT_AND_CRITIQUE, "## Argument\r\nShip it.\r\n## Critique\rWeak.").passed


MEGABYTE = 1_000_000


@pytest.mark.parametrize("heading,missing", [
    pytest.param("## Argument" + " " * MEGABYTE + "x", True, id="spaces-then-text"),
    pytest.param("## Argument" + "\t" * MEGABYTE + "x", True, id="tabs-then-text"),
    pytest.param("## Argument" + " " * MEGABYTE + "#", False, id="spaces-then-closing-hashes"),
    pytest.param("## Argument" + ":" * MEGABYTE, False, id="colons"),
    pytest.param("## Argument" + " :" * (MEGABYTE // 2), False, id="spaced-colons"),
    pytest.param("## **Argument" + "*:" * (MEGABYTE // 2), False, id="emphasis-and-colons"),
])
def test_a_megabyte_heading_line_is_read_in_linear_time(heading, missing):
    """Heading lines used to be parsed in quadratic time: a run of spaces or
    tabs inside one took about 28 s at 64,000 characters, and each final
    colon or emphasis marker copied the title again, stalling acceptance."""
    output = f"{heading}\nShip it.\n## Critique\nWeak."
    started = time.perf_counter()
    report = check(ARGUMENT_AND_CRITIQUE, output)
    assert time.perf_counter() - started < 1.0
    assert codes(report) == (["section_missing"] if missing else [])


# --- JSON ---------------------------------------------------------------

DOCUMENT = {
    "title": "Quarterly plan",
    "total": 12.5,
    "count": 3,
    "items": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
    "meta": {"owner": "ops", "status": "draft"},
    "sources": ["s1", "s2"],
    "flag": True,
}

CHECKS = [
    {"id": "has-title", "kind": "present", "path": ["title"]},
    {"id": "title", "kind": "nonempty", "path": ["title"], "min_chars": 5},
    {"id": "total", "kind": "number", "path": ["total"], "expected": 12.4, "tolerance": 0.25},
    {"id": "count", "kind": "equals", "path": ["count"], "expected": 3},
    {"id": "first-name", "kind": "equals", "path": ["items", 0, "name"], "expected": "a"},
    {"id": "three-items", "kind": "length", "path": ["items"], "expected": 3},
    {"id": "meta", "kind": "object", "path": ["meta"], "minimum": 2},
    {"id": "meta-keys", "kind": "keys", "path": ["meta"], "expected": ["status", "owner"]},
    {"id": "root-keys", "kind": "required_keys", "path": [], "expected": ["title", "items"]},
    {"id": "sources", "kind": "citations", "path": ["sources"], "allowed": ["s1", "s2", "s3"],
     "minimum": 2},
]


def run_json(document, checks=CHECKS, **spec) -> ValidationReport:
    return check(JSONValidator({"checks": checks, **spec}), json.dumps(document))


def failed(report: ValidationReport) -> list[str]:
    return [finding.message.split("'")[1] for finding in report.findings]


def test_a_document_passing_every_check_kind_passes():
    assert run_json(DOCUMENT).passed
    assert check(JSONValidator(), '[1, "two", {"three": null}]').passed


@pytest.mark.parametrize("change,failing", [
    (lambda d: d.pop("title"), ["has-title", "title", "root-keys"]),
    (lambda d: d.update(title="Plan"), ["title"]),
    (lambda d: d.update(title="     x     "), ["title"]),
    (lambda d: d.update(title=12345), ["title"]),
    (lambda d: d.update(total=12.7), ["total"]),
    (lambda d: d.update(total="12.5"), ["total"]),
    (lambda d: d.update(total=True), ["total"]),
    (lambda d: d.update(count=3.0), ["count"]),
    (lambda d: d.update(count=True), ["count"]),
    (lambda d: d["items"].pop(), ["three-items"]),
    (lambda d: d.update(items={"0": {"name": "a"}, "1": 1, "2": 2}), ["first-name"]),
    (lambda d: d["items"][0].update(name="A"), ["first-name"]),
    (lambda d: d["meta"].pop("status"), ["meta", "meta-keys"]),
    (lambda d: d["meta"].update(extra=1), ["meta-keys"]),
    (lambda d: d.update(meta=["owner", "status"]), ["meta", "meta-keys"]),
    (lambda d: d.update(sources=["s1"]), ["sources"]),
    (lambda d: d.update(sources=["s1", "s1", "s2"]), ["sources"]),
    (lambda d: d.update(sources=["s1", "s4"]), ["sources"]),
    (lambda d: d.update(sources=["s1", 2]), ["sources"]),
    (lambda d: d.update(sources="s1 s2"), ["sources"]),
])
def test_each_check_kind_fails_on_the_values_it_excludes(change, failing):
    document = json.loads(json.dumps(DOCUMENT))
    change(document)
    report = run_json(document)
    assert report.outcome is ValidationOutcome.REJECTED
    assert failed(report) == failing


def test_failed_checks_name_their_path_and_what_they_expected():
    document = dict(DOCUMENT, total=20, meta={"owner": "ops"})
    report = run_json(document, CHECKS[2:3] + CHECKS[7:8])
    number, keys = report.findings
    assert (number.code, number.message) == ("json_check_failed", "check 'total' (number) failed at $['total']")
    assert number.expected == {"kind": "number", "expected": 12.4, "tolerance": 0.25}
    assert number.observed == 20
    assert keys.observed == {"keys": ["owner"], "count": 1}
    missing = run_json({"items": []}, CHECKS[4:5])
    assert codes(missing) == ["json_path_missing"]
    assert missing.findings[0].message == "check 'first-name': $['items'][0]['name'] is missing"
    assert missing.findings[0].expected == ["items", 0, "name"]
    long_text = run_json({"title": "x" * 100},
                         [{"id": "tt", "kind": "equals", "path": ["title"], "expected": ""}])
    assert long_text.findings[0].observed == "a string of 100 characters"


@pytest.mark.parametrize("value,passes", [
    (12.15, True), (12.65, True), (12.149, False), (12.651, False), (12, False),
    (10**400, False), (-(10**400), False),
])
def test_number_tolerance_is_inclusive_and_huge_integers_fail(value, passes):
    checks = [{"id": "total", "kind": "number", "path": ["total"], "expected": 12.4, "tolerance": 0.25}]
    assert run_json({"total": value}, checks).passed is passes


def test_equality_compares_canonical_json():
    checks = [{"id": "meta", "kind": "equals", "path": ["meta"],
               "expected": {"b": [1, {"c": None}], "a": "é"}}]
    assert check(JSONValidator({"checks": checks}), '{"meta": {"a": "\\u00e9", "b": [1, {"c": null}]}}').passed
    assert not run_json({"meta": {"a": "é", "b": [1.0, {"c": None}]}}, checks).passed
    assert not run_json({"meta": {"a": "é", "b": [True, {"c": None}]}}, checks).passed


def test_paths_resolve_keys_in_objects_and_indexes_in_arrays_only():
    checks = [{"id": "second", "kind": "equals", "path": [1], "expected": "b"}]
    assert run_json(["a", "b"], checks).passed
    assert codes(run_json({"1": "b"}, checks)) == ["json_path_missing"]
    assert codes(run_json(["a"], checks)) == ["json_path_missing"]
    by_key = [{"id": "first", "kind": "present", "path": ["0"]}]
    assert codes(run_json(["a"], by_key)) == ["json_path_missing"]
    assert codes(run_json("scalar", by_key)) == ["json_path_missing"]
    root = [{"id": "root", "kind": "length", "path": [], "expected": 2}]
    assert run_json(["a", "b"], root).passed and run_json({"a": 1, "b": 2}, root).passed


@pytest.mark.parametrize("text", [
    "", "not JSON", "```json\n{}\n```", '{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}',
    '{"x": -Infinity}', '{"x": 1e999}', '{"x": 1} trailing', "[" * 100_000 + "]" * 100_000,
], ids=["empty", "text", "fence", "duplicate-key", "nan", "infinity", "negative-infinity",
        "overflow", "trailing-text", "deep-nesting"])
def test_output_that_is_not_strict_json_is_rejected(text):
    report = check(JSONValidator(), text)
    assert report.outcome is ValidationOutcome.REJECTED
    assert codes(report) == ["json_invalid"]
    assert report.findings[0].message.startswith("output is not strict JSON: ")
    assert len(report.findings[0].message) <= 200


def test_output_larger_than_max_bytes_is_rejected_before_parsing():
    validator = JSONValidator({"max_bytes": 10})
    assert check(validator, '"123456"').passed
    report = check(validator, '"é123456789"')  # 13 bytes, 11 characters
    assert codes(report) == ["json_too_large"]
    assert (report.findings[0].expected, report.findings[0].observed) == (10, 13)


@pytest.mark.parametrize("spec,message", [
    ([], "JSON object"),
    ({"check": []}, "unknown keys"),
    ({"checks": {}}, "checks"),
    ({"max_bytes": 0}, "max_bytes"),
    ({"max_bytes": True}, "max_bytes"),
    ({"checks": [{"id": "a1", "kind": "regex", "path": []}]}, "unknown JSON check kind"),
    ({"checks": ["present"]}, "unknown JSON check kind"),
    ({"checks": [{"id": "a1", "kind": "equals", "path": []}]}, "missing keys"),
    ({"checks": [{"id": "a1", "kind": "present", "path": [], "expected": 1}]}, "unknown keys"),
    ({"checks": [{"kind": "present", "path": []}]}, "missing keys"),
    ({"checks": [{"id": "A1", "kind": "present", "path": []}]}, "check id"),
    ({"checks": [{"id": "a", "kind": "present", "path": []}]}, "check id"),
    ({"checks": [{"id": "1a", "kind": "present", "path": []}]}, "check id"),
    ({"checks": [{"id": "a1", "kind": "present", "path": "a"}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "present", "path": [""]}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "present", "path": [-1]}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "present", "path": [True]}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "present", "path": [1.0]}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "present", "path": ["a"] * 17}]}, "path"),
    ({"checks": [{"id": "a1", "kind": "number", "path": [], "expected": "1"}]}, "finite"),
    ({"checks": [{"id": "a1", "kind": "number", "path": [], "expected": True}]}, "finite"),
    ({"checks": [{"id": "a1", "kind": "number", "path": [], "expected": 10**400}]}, "finite"),
    ({"checks": [{"id": "a1", "kind": "number", "path": [], "expected": 1, "tolerance": -1}]},
     "tolerance"),
    ({"checks": [{"id": "a1", "kind": "nonempty", "path": [], "min_chars": -1}]}, "min_chars"),
    ({"checks": [{"id": "a1", "kind": "object", "path": [], "minimum": 10_001}]}, "minimum"),
    ({"checks": [{"id": "a1", "kind": "length", "path": [], "expected": None}]}, "length"),
    ({"checks": [{"id": "a1", "kind": "keys", "path": [], "expected": ["a", "a"]}]}, "repeat"),
    ({"checks": [{"id": "a1", "kind": "required_keys", "path": [], "expected": [1]}]}, "keys"),
    ({"checks": [{"id": "a1", "kind": "citations", "path": [], "allowed": [" "]}]}, "citations"),
    ({"checks": [{"id": "a1", "kind": "present", "path": []}] * 2}, "distinct"),
    ({"checks": [{"id": f"c{index}", "kind": "present", "path": []} for index in range(257)]},
     "checks"),
    ({"checks": [{"id": "a1", "kind": "equals", "path": [], "expected": float("inf")}]}, "infinity"),
])
def test_json_spec_is_checked_when_the_validator_is_built(spec, message):
    with pytest.raises(ValueError, match=message):
        JSONValidator(spec)


# --- strict delivery acceptance -----------------------------------------


class Steps(Provider):
    """Answers each node by its label, the prompt's first line."""

    def __init__(self, answers):
        self.answers = answers

    async def complete(self, system, prompt, model):
        return CompletionResult(text=self.answers[prompt.splitlines()[0]],
                                prompt_tokens=1, completion_tokens=1)


def argue_then_critique(critique: str):
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="argue", label="Argue"),
        Node(id="critique", label="Critique", depends_on=["argue"]),
    ])
    provider = Steps({"Argue": "## Argument\nShip the pilot.", "Critique": critique})
    policy = AcceptancePolicy(mode="strict", final_validators=[ARGUMENT_AND_CRITIQUE])
    return Swarm(provider=provider, model="test", acceptance=policy).execute(graph)


@pytest.mark.parametrize("parallel", [False, True])
def test_strict_run_rejects_a_final_step_that_returns_only_its_increment(parallel):
    increment = "## Critique\nThe pilot was too small to justify shipping."
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="argue", label="Argue"),
        Node(id="critique", label="Critique", depends_on=["argue"]),
    ])
    provider = Steps({"Argue": "## Argument\nShip the pilot.", "Critique": increment})
    policy = AcceptancePolicy(mode="strict", final_validators=[ARGUMENT_AND_CRITIQUE])
    result = Swarm(provider=provider, model="test", acceptance=policy, parallel=parallel).execute(graph)

    assert not result.accepted
    assert result.acceptance.decision == "rejected"
    assert result.output == "" and result.rejected_output == increment
    (outcome,) = result.acceptance.validators
    assert (outcome.validator, outcome.outcome) == (ARGUMENT_AND_CRITIQUE.identity, "rejected")
    assert result.acceptance.reasons == (
        "final validator 'smythe.text-spec' rejected the output: section_missing: "
        "required section 'Argument' is missing",
    )


def test_strict_run_accepts_a_final_step_that_returns_the_whole_deliverable():
    complete = "## Argument\nShip the pilot.\n\n## Critique\nThe pilot was small."
    result = argue_then_critique(complete)
    assert result.accepted and result.output == complete
    assert result.acceptance.validators[0].outcome == "passed"


PLAN_CHECKS = JSONValidator({"checks": CHECKS})


def gather_then_report(report: str, *, parallel: bool = False):
    """A strict run whose final step's JSON a JSONValidator checks."""
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="gather", label="Gather"),
        Node(id="report", label="Report", depends_on=["gather"]),
    ])
    provider = Steps({"Gather": "The plan has three items, owned by ops.", "Report": report})
    policy = AcceptancePolicy(mode="strict", final_validators=[PLAN_CHECKS])
    return Swarm(provider=provider, model="test", acceptance=policy, parallel=parallel).execute(graph)


@pytest.mark.parametrize("parallel", [False, True])
def test_strict_run_accepts_json_that_passes_every_check(parallel):
    report = json.dumps(DOCUMENT)
    result = gather_then_report(report, parallel=parallel)

    assert result.accepted and result.output == report and result.rejected_output is None
    (outcome,) = result.acceptance.validators
    assert (outcome.validator, outcome.outcome) == (PLAN_CHECKS.identity, "passed")
    assert result.acceptance.output_sha256 == ValidationSubject.from_text(report).sha256


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("report,reason", [
    (json.dumps(dict(DOCUMENT, items=[{"title": "a"}, {"name": "b"}, {"name": "c"}])),
     "json_path_missing: check 'first-name': $['items'][0]['name'] is missing"),
    (json.dumps(dict(DOCUMENT, count=4)),
     "json_check_failed: check 'count' (equals) failed at $['count']"),
    ("```json\n" + json.dumps(DOCUMENT) + "\n```", "json_invalid: output is not strict JSON: "),
], ids=["missing-path", "wrong-value", "fenced"])
def test_strict_run_rejects_json_that_fails_a_check(report, reason, parallel):
    result = gather_then_report(report, parallel=parallel)

    assert not result.accepted and result.acceptance.decision == "rejected"
    assert result.output == "" and result.rejected_output == report
    (outcome,) = result.acceptance.validators
    assert (outcome.validator, outcome.outcome) == (PLAN_CHECKS.identity, "rejected")
    (recorded,) = result.acceptance.reasons
    assert recorded.startswith(f"final validator 'smythe.json-spec' rejected the output: {reason}")
