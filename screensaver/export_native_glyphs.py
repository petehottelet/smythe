"""Export exact current SVG contours to the three native screensaver ports.

This reads published artwork; it never invokes the glyph generator. Run with
``--check`` to verify all generated outputs without writing to the workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "native-mixed-svg-v1"
REFERENCE = Path("screensaver/svg-preview/reference")
ORIGINAL = Path("benchmarks/partitions/glyph_svg_v1/catalog")
TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
ARITY = {"M": 2, "L": 2, "C": 6, "Z": 0}
CODES = {name: index for index, name in enumerate(ARITY)}
SVG_NS = "{http://www.w3.org/2000/svg}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _text(root: Path, relative: str | Path) -> str:
    return (root / relative).read_text(encoding="utf-8")


def parse_path(path: str) -> list[list[int | float]]:
    """Parse explicit absolute M/L/C/Z, preserving command and contour order.

    Coordinates are IEEE-754 doubles in every native target. No flattening,
    simplification, stroke expansion, fitting, mirroring, or winding changes.
    """
    tokens = []
    end = 0
    for match in TOKEN.finditer(path):
        if path[end:match.start()].strip(" ,\r\n\t"):
            raise ValueError("Invalid path token")
        tokens.append(match.group())
        end = match.end()
    if path[end:].strip(" ,\r\n\t") or not tokens:
        raise ValueError("Empty or invalid path")
    commands = []
    index = 0
    opened = False
    while index < len(tokens):
        name = tokens[index]
        if name not in ARITY:
            raise ValueError("Only explicit absolute M/L/C/Z paths are accepted")
        index += 1
        if name == "M":
            if opened:
                raise ValueError("Every contour must close before another starts")
            opened = True
        elif not opened:
            raise ValueError("Path command outside a contour")
        count = ARITY[name]
        values = tokens[index:index + count]
        if len(values) != count:
            raise ValueError("Truncated path command")
        try:
            numbers = [float(value) for value in values]
        except ValueError as error:
            raise ValueError("Invalid path coordinate") from error
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Nonfinite path coordinate")
        commands.append([CODES[name], *numbers])
        index += count
        if name == "Z":
            opened = False
    if opened:
        raise ValueError("Every contour must explicitly close")
    return commands


def _svg_paths(data: bytes) -> list[str]:
    root = ET.fromstring(data)
    if root.tag != SVG_NS + "svg" or root.get("viewBox") != "0 0 100 100":
        raise ValueError("Expected canonical 100x100 SVG")
    paths = []
    for child in root:
        if child.tag in {SVG_NS + "title", SVG_NS + "desc"}:
            continue
        if (child.tag != SVG_NS + "path"
                or set(child.attrib) != {"d", "fill", "fill-rule"}
                or child.get("fill") != "#000000"
                or child.get("fill-rule") != "nonzero"):
            raise ValueError("Expected untransformed black nonzero SVG paths")
        paths.append(child.attrib["d"])
    if not paths:
        raise ValueError("A visible glyph must contain paths")
    return paths


def build_payload(repo_root: Path = REPO_ROOT) -> dict:
    """Return the shared 249-slot native data and its canonical semantic hash."""
    repo_root = Path(repo_root)
    reference_bytes = (repo_root / REFERENCE / "catalog.json").read_bytes()
    reference = json.loads(reference_bytes)
    manifest_bytes = (repo_root / ORIGINAL / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (reference["count"] != 56 or reference["canvas"] != [100, 100]
            or reference["fill_rule"] != "nonzero"
            or reference["source"]["sequence_length"] != 57
            or reference["source"]["blank_sequence_indices"] != [4]):
        raise ValueError("Reference catalog identity changed")
    if manifest["count"] != 192 or len(manifest["glyphs"]) != 192:
        raise ValueError("Expected the published 192-original catalog")
    reference_by_slot = {g["source_sequence_index"]: g for g in reference["glyphs"]}
    if (len(reference_by_slot) != 56
            or set(reference_by_slot) != set(range(57)) - {4}):
        raise ValueError("Reference sequence must retain precisely its blank slot")
    commands, glyph_ids = [], []
    for slot in range(57):
        if slot == 4:
            commands.append([])
            glyph_ids.append("BASE-BLANK")
            continue
        glyph = reference_by_slot[slot]
        paths = _svg_paths((repo_root / REFERENCE / (glyph["glyph_id"] + ".svg")).read_bytes())
        if paths != glyph["paths"]:
            raise ValueError("Reference SVG disagrees with the canonical catalog")
        commands.append([command for path in paths for command in parse_path(path)])
        glyph_ids.append(glyph["glyph_id"])
    for index, glyph in enumerate(manifest["glyphs"]):
        if glyph["glyph_id"] != f"GLYPH-{index:03d}" or glyph["file"] != glyph["glyph_id"] + ".svg":
            raise ValueError("Original catalog indices must be complete and ordered")
        data = (repo_root / ORIGINAL / glyph["file"]).read_bytes()
        if _sha(data) != glyph["svg_sha256"]:
            raise ValueError("Published original SVG hash mismatch")
        commands.append([command for path in _svg_paths(data) for command in parse_path(path)])
        glyph_ids.append(glyph["glyph_id"])
    legacy_js = _text(repo_root, "screensaver/glyphs.js")
    legacy_match = re.search(r"const GLYPHS=(\{.*\});\s*$", legacy_js, re.S)
    if legacy_match is None:
        raise ValueError("Legacy motion source is missing")
    legacy = json.loads(legacy_match.group(1))
    if len(legacy["speeds"]) != 192 or len(legacy["trails"]) != 192:
        raise ValueError("Legacy motion inventory changed")
    reference_license = _text(repo_root, REFERENCE / "LICENSE")
    if _sha(reference_license.encode()) != reference["source"]["license_sha256"]:
        raise ValueError("Reference MIT notice hash mismatch")
    payload = {
        "version": VERSION, "canvas": [100, 100], "count": 249,
        "reference_count": 57, "reference_visible_count": 56,
        "original_count": 192, "original_offset": 57, "blank_index": 4,
        "original_share": 0.1, "fill_rule": "nonzero",
        "reference_sha256": _sha(reference_bytes), "original_sha256": _sha(manifest_bytes),
        "commands": commands, "glyph_ids": glyph_ids,
        "speeds": [legacy["speeds"][i % 192] for i in range(249)],
        "trails": [legacy["trails"][i % 192] for i in range(249)],
        "licenses": [
            {"scope": "Reference artwork", "source": str(REFERENCE / "LICENSE").replace("\\", "/"),
             "text": reference_license},
            {"scope": "Smythe original catalog and native exporter", "source": "LICENSE",
             "text": _text(repo_root, "LICENSE")},
        ],
    }
    payload["catalog_sha256"] = _sha(_json(payload))
    return payload


def _number(value: int | float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def _notice(data: dict) -> str:
    return "\n\n".join(item["scope"] + "\n" + item["text"].strip() for item in data["licenses"])


def _csharp(data: dict) -> bytes:
    fields = {
        "Count": "count", "ReferenceCount": "reference_count",
        "ReferenceVisibleCount": "reference_visible_count", "OriginalCount": "original_count",
        "OriginalOffset": "original_offset", "BlankIndex": "blank_index",
    }
    lines = ["// Generated by screensaver/export_native_glyphs.py; do not edit.",
             "// Fill each complete glyph as one compound path with nonzero winding.",
             "using System.Globalization;",
             "namespace SmytheGlyphRain", "{", "    internal static class GlyphData", "    {",
             "        public const float CanvasW = 100f, CanvasH = 100f;"]
    lines += [f"        public const int {name} = {data[key]};" for name, key in fields.items()]
    lines.append("        public const double OriginalShare = 0.1;")
    for name, key in [("Version", "version"), ("CatalogSha256", "catalog_sha256"),
                      ("ReferenceSha256", "reference_sha256"), ("OriginalSha256", "original_sha256"),
                      ("FillRule", "fill_rule")]:
        lines.append(f"        public const string {name} = {json.dumps(data[key])};")
    lines.append("        public const string LicenseNotice = " + json.dumps(_notice(data)) + ";")
    lines.append("        public static readonly string[] GlyphIds = new string[]{"
                 + ",".join(json.dumps(value) for value in data["glyph_ids"]) + "};")
    for name, key, kind in [("Speeds", "speeds", "double"), ("Trails", "trails", "int")]:
        lines.append(f"        public static readonly {kind}[] {name} = new {kind}[]{{"
                     + ",".join(_number(value) for value in data[key]) + "};")
    lines.append("        public static readonly double[][][] Commands = new double[][][]{"
                 + ",".join(f"Glyph{i}()" for i in range(data["count"])) + "};")
    # A literal array for every command exceeds the CLR's 65,535-field limit:
    # csc emits a backing RVA field for each numeric array. One invariant string
    # per glyph avoids those fields while reconstructing the exact doubles.
    lines += [
        "        private static double[][] ParseCommands(string encoded)", "        {",
        "            if (encoded.Length == 0) return new double[0][];",
        "            string[] rows = encoded.Split(';');",
        "            var result = new double[rows.Length][];",
        "            for (int i = 0; i < rows.Length; i++)", "            {",
        "                string[] values = rows[i].Split(',');",
        "                result[i] = new double[values.Length];",
        "                for (int j = 0; j < values.Length; j++)",
        "                    result[i][j] = double.Parse(values[j], NumberStyles.Float, CultureInfo.InvariantCulture);",
        "            }", "            return result;", "        }",
    ]
    for index, commands in enumerate(data["commands"]):
        encoded = ";".join(",".join(_number(v) for v in command) for command in commands)
        lines += [f"        private static double[][] Glyph{index}()", "        {",
                  "            return ParseCommands(" + json.dumps(encoded) + ");", "        }"]
    return ("\n".join([*lines, "    }", "}"]) + "\n").encode()


def _c_header(data: dict) -> bytes:
    lines = ["/* Generated by screensaver/export_native_glyphs.py; do not edit. */",
             "/* Fill each complete glyph with Cairo's nonzero winding rule. */",
             "#ifndef SMYTHE_GLYPH_DATA_H", "#define SMYTHE_GLYPH_DATA_H",
             "typedef struct { int kind; double values[6]; } GlyphCommand;",
             "typedef struct { int offset, count; double speed; int trail; } GlyphSpec;"]
    for key in ["count", "reference_count", "reference_visible_count", "original_count",
                "original_offset", "blank_index", "original_share"]:
        lines.append(f"#define GLYPH_{key.upper()} {_number(data[key])}")
    lines += ["#define GLYPH_CANVAS_W 100", "#define GLYPH_CANVAS_H 100"]
    for key in ["version", "fill_rule", "catalog_sha256", "reference_sha256", "original_sha256"]:
        lines.append(f"#define GLYPH_{key.upper()} {json.dumps(data[key])}")
    lines.append("static const char GLYPH_LICENSE_NOTICE[] = " + json.dumps(_notice(data)) + ";")
    lines.append("static const char *const GLYPH_IDS[] = {"
                 + ",".join(json.dumps(value) for value in data["glyph_ids"]) + "};")
    lines.append("static const GlyphCommand GLYPH_COMMANDS[] = {")
    specs, offset = [], 0
    for index, commands in enumerate(data["commands"]):
        specs.append(f"    {{{offset}, {len(commands)}, {_number(data['speeds'][index])}, "
                     f"{data['trails'][index]}}}")
        offset += len(commands)
        for command in commands:
            values = [*command[1:], *([0] * (7 - len(command)))]
            lines.append(f"    {{{command[0]}, {{" + ", ".join(_number(v) for v in values) + "}},")
    lines += ["};", "static const GlyphSpec GLYPHS[] = {", ",\n".join(specs), "};", "#endif"]
    return ("\n".join(lines) + "\n").encode()


def render_outputs(repo_root: Path = REPO_ROOT) -> dict[str, bytes]:
    """Return deterministic native files and their publication manifest."""
    repo_root = Path(repo_root)
    data = build_payload(repo_root)
    outputs = {
        "screensaver/windows/GlyphData.cs": _csharp(data),
        "screensaver/macos/glyphs.json": _json(data) + b"\n",
        "screensaver/linux/glyph_data.h": _c_header(data),
    }
    source_paths = ["screensaver/export_native_glyphs.py", "screensaver/glyphs.js", "LICENSE",
                    str(REFERENCE / "catalog.json"), str(REFERENCE / "provenance.json"),
                    str(REFERENCE / "LICENSE"), str(ORIGINAL / "manifest.json")]
    sources = {path.replace("\\", "/"): _sha(_text(repo_root, path).encode()) for path in source_paths}
    glyph_sources = []
    for index, name in enumerate(data["glyph_ids"]):
        path = None if index == data["blank_index"] else (
            REFERENCE if index < data["original_offset"] else ORIGINAL) / (name + ".svg")
        glyph_sources.append({
            "index": index, "glyph_id": name,
            "file": path.as_posix() if path is not None else None,
            "svg_sha256": _sha((repo_root / path).read_bytes()) if path is not None else None,
            "command_count": len(data["commands"][index]),
        })
    provenance = {
        "version": VERSION, "catalog_sha256": data["catalog_sha256"],
        "count": 249, "reference_count": 57, "reference_visible_count": 56,
        "original_count": 192, "original_offset": 57, "blank_index": 4,
        "original_share": 0.1, "canvas": [100, 100], "fill_rule": "nonzero",
        "command_codes": {name: code for name, code in CODES.items()},
        "coordinate_method": "Source-order absolute M/L/C/Z in double precision; no flattening, "
                             "redrawing, fitting, mirroring, or winding changes. Fill once per glyph.",
        "catalog_hash_basis": "SHA256 of UTF-8 JSON payload excluding catalog_sha256, sorted keys, "
                              "compact separators, ensure_ascii=false, no trailing newline.",
        "source_hash_basis": "Text sources use UTF-8 after CRLF-to-LF normalization. "
                             "Per-glyph svg_sha256 and output_sha256 bind exact bytes.",
        "motion": "Legacy web speeds/trails repeated by native index modulo 192; weighted "
                  "90% reference branch and 10% original branch belongs to native renderers.",
        "source_sha256": sources,
        "source_glyphs": glyph_sources,
        "output_sha256": {path: _sha(content) for path, content in outputs.items()},
        "licenses": data["licenses"],
        "reference_rights_scope": json.loads(_text(repo_root, REFERENCE / "provenance.json"))["rights_scope"],
    }
    outputs["screensaver/native-catalog.json"] = (
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode()
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated files without writing")
    args = parser.parse_args(argv)
    outputs = render_outputs()
    different = []
    for path, content in outputs.items():
        destination = REPO_ROOT / path
        if args.check:
            if not destination.exists() or destination.read_bytes() != content:
                different.append(path)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            print(f"wrote {path} ({len(content)} bytes)")
    if different:
        print("Native glyph export differs: " + ", ".join(different), file=sys.stderr)
        return 1
    if args.check:
        print("Native glyph export matches all four generated files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
