"""Cross-language parity for the published native SVG catalogs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

import pytest

from screensaver import export_glyphs, export_native_glyphs as exporter


@pytest.fixture(scope="module")
def outputs():
    return exporter.render_outputs()


@pytest.fixture(scope="module")
def payload(outputs):
    return json.loads(outputs["screensaver/macos/glyphs.json"])


def test_complete_slots_and_blank_identity(payload):
    assert payload["canvas"] == [100, 100]
    assert payload["fill_rule"] == "nonzero"
    assert payload["count"] == len(payload["commands"]) == 249
    assert payload["reference_count"] == payload["original_offset"] == 57
    assert payload["reference_visible_count"] == 56
    assert payload["original_count"] == 192
    assert payload["original_share"] == 0.1
    assert payload["blank_index"] == 4
    assert [i for i, glyph in enumerate(payload["commands"]) if not glyph] == [4]
    assert payload["glyph_ids"][4] == "BASE-BLANK"
    assert payload["glyph_ids"][5] == "BASE-004"
    assert payload["glyph_ids"][56] == "BASE-055"
    assert payload["glyph_ids"][57:] == [f"GLYPH-{i:03d}" for i in range(192)]


def test_every_source_coordinate_and_contour_order_is_preserved(payload):
    namespace = "{http://www.w3.org/2000/svg}"
    counts = {"M": 2, "L": 2, "C": 6, "Z": 0}
    code_names = "MLCZ"
    curve_count = 0
    contour_count = 0
    for index, glyph_id in enumerate(payload["glyph_ids"]):
        if index == 4:
            continue
        folder = exporter.REFERENCE if index < 57 else exporter.ORIGINAL
        root = ET.fromstring((exporter.REPO_ROOT / folder / (glyph_id + ".svg")).read_bytes())
        source = []
        for path in root.findall(namespace + "path"):
            assert path.attrib["fill-rule"] == "nonzero"
            source.extend(re.findall(r"[MLCZ]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",
                                     path.attrib["d"]))
        cursor = 0
        for command in payload["commands"][index]:
            kind = code_names[command[0]]
            assert source[cursor] == kind
            assert len(command) == counts[kind] + 1
            cursor += 1
            for value in command[1:]:
                # A decimal-to-double conversion can round at IEEE-754 precision,
                # but exporting fewer significant digits must fail this check.
                assert abs(Decimal.from_float(float(value)) - Decimal(source[cursor])) < Decimal("1e-12")
                cursor += 1
            curve_count += kind == "C"
            contour_count += kind == "M"
        assert cursor == len(source)
    assert curve_count > 100
    assert contour_count > 249


def test_all_native_serializations_have_identical_commands_and_motion(outputs, payload):
    cs = outputs["screensaver/windows/GlyphData.cs"].decode()
    methods = re.findall(r"private static double\[\]\[\] Glyph(\d+)\(\)(.*?)\n        }", cs, re.S)
    assert [int(index) for index, _ in methods] == list(range(249))
    for index, body in methods:
        encoded = json.loads(re.search(r'ParseCommands\((".*")\)', body).group(1))
        commands = ([[float(value) for value in row.split(",")] for row in encoded.split(";")]
                    if encoded else [])
        assert commands == payload["commands"][int(index)]
    for name, key in [("Speeds", "speeds"), ("Trails", "trails")]:
        values = re.search(name + r" = new \w+\[\]\{([^}]+)\}", cs).group(1)
        assert [float(v) for v in values.split(",")] == payload[key]
    header = outputs["screensaver/linux/glyph_data.h"].decode()
    commands_text = header.split("GLYPH_COMMANDS[] = {\n", 1)[1].split("\n};", 1)[0]
    rows = re.findall(r"\{(\d+), \{([^}]+)\}\}", commands_text)
    commands = [(int(kind), [float(v) for v in numbers.split(",")]) for kind, numbers in rows]
    specs_text = header.split("GLYPHS[] = {\n", 1)[1].split("\n};", 1)[0]
    specs = re.findall(r"\{(\d+), (\d+), ([\d.]+), (\d+)\}", specs_text)
    assert len(specs) == 249
    offset = 0
    for index, (start, length, speed, trail) in enumerate(specs):
        assert int(start) == offset
        assert int(length) == len(payload["commands"][index])
        assert float(speed) == payload["speeds"][index]
        assert int(trail) == payload["trails"][index]
        for actual, expected in zip(commands[offset:offset + int(length)], payload["commands"][index], strict=True):
            assert actual[0] == expected[0]
            assert actual[1][:len(expected) - 1] == expected[1:]
            assert actual[1][len(expected) - 1:] == [0] * (7 - len(expected))
        offset += int(length)
    assert offset == len(commands)


def test_nonzero_contours_keep_opposite_orientation(payload):
    # GLYPH-004 has one exterior and two counters. Their signs must remain
    # opposite; filling separate polygons would erase the two openings.
    glyph = payload["commands"][57 + 4]
    contours, points = [], []
    for command in glyph:
        if command[0] in {0, 1}:
            points.append(command[1:])
        elif command[0] == 3:
            contours.append(points)
            points = []
    areas = [sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(poly, poly[1:] + poly[:1], strict=True))
             for poly in contours]
    assert len(areas) == 3
    assert any(area > 0 for area in areas)
    assert any(area < 0 for area in areas)


def test_full_license_notices_and_hash_binding(outputs, payload):
    reference = (exporter.REPO_ROOT / exporter.REFERENCE / "LICENSE").read_text()
    assert reference.startswith("MIT License\n\nCopyright (c) 2018 Rezmason")
    assert payload["licenses"][0]["text"] == reference
    for path, field in [("screensaver/windows/GlyphData.cs", "LicenseNotice"),
                        ("screensaver/linux/glyph_data.h", "GLYPH_LICENSE_NOTICE\\[\\]")]:
        text = outputs[path].decode()
        encoded = re.search(field + r' = (".*");', text).group(1)
        assert reference.strip() in json.loads(encoded)
        assert "Copyright (c) 2026 Pete Hottelet" in json.loads(encoded)
    canonical = {key: value for key, value in payload.items() if key != "catalog_sha256"}
    content = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    assert payload["catalog_sha256"] == hashlib.sha256(content).hexdigest()
    manifest = json.loads(outputs["screensaver/native-catalog.json"])
    assert manifest["catalog_sha256"] == payload["catalog_sha256"]
    for path, digest in manifest["output_sha256"].items():
        assert hashlib.sha256(outputs[path]).hexdigest() == digest
    assert len(manifest["source_glyphs"]) == 249
    assert manifest["source_glyphs"][4]["file"] is None


@pytest.mark.parametrize("path", ["", "M0 0 L1 1", "m0 0 Z", "L0 0 Z", "M0 0 Z Z",
                                      "M0 0 M1 1 Z", "M0 0 Q1 2 3 4 Z", "M0 0 L1e999 2 Z",
                                      "M0 0 L1 Z", "M0 0 @ L1 1 Z"])
def test_invalid_or_implicit_geometry_is_rejected(path):
    with pytest.raises(ValueError):
        exporter.parse_path(path)


def test_changed_source_hash_is_rejected(tmp_path):
    for relative in [exporter.REFERENCE, exporter.ORIGINAL]:
        shutil.copytree(exporter.REPO_ROOT / relative, tmp_path / relative)
    changed = tmp_path / exporter.ORIGINAL / "GLYPH-000.svg"
    changed.write_bytes(changed.read_bytes().replace(b"37.327", b"37.328", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.build_payload(tmp_path)


def test_check_is_read_only_and_reports_stale_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "render_outputs", lambda: {"native.txt": b"new\n"})
    target = tmp_path / "native.txt"
    target.write_bytes(b"old\r\n")
    assert exporter.main(["--check"]) == 1
    assert target.read_bytes() == b"old\r\n"
    target.write_bytes(b"new\n")
    assert exporter.main(["--check"]) == 0


def test_legacy_export_cannot_overwrite_native_catalogs(monkeypatch):
    writes = []
    monkeypatch.setattr(export_glyphs, "_write", lambda path, _content: writes.append(path))
    export_glyphs.main()
    assert [path.relative_to(exporter.REPO_ROOT).as_posix() for path in writes] == ["screensaver/glyphs.js"]


def test_checked_in_exports_match_current_sources(outputs):
    for path, expected in outputs.items():
        assert (exporter.REPO_ROOT / path).read_bytes() == expected


def test_windows_compiled_catalog_initializes_without_clr_field_overflow(tmp_path, outputs, payload):
    compiler = Path(os.environ.get("WINDIR", "")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if os.name != "nt" or not compiler.is_file():
        pytest.skip("The compiled CLR regression requires Windows' bundled C# compiler")
    source = tmp_path / "GlyphData.cs"
    source.write_bytes(outputs["screensaver/windows/GlyphData.cs"])
    probe = tmp_path / "Probe.cs"
    probe.write_text('''using System;
using System.IO;
using System.Security.Cryptography;
using SmytheGlyphRain;
class Probe {
    static void Main() {
        using (var bytes = new MemoryStream()) {
            using (var writer = new BinaryWriter(bytes, System.Text.Encoding.UTF8, true)) {
                writer.Write(GlyphData.Commands.Length);
                foreach (double[][] glyph in GlyphData.Commands) {
                    writer.Write(glyph.Length);
                    foreach (double[] command in glyph) {
                        writer.Write(command.Length);
                        foreach (double value in command) writer.Write(value);
                    }
                }
            }
            using (var sha = SHA256.Create()) {
                Console.WriteLine(BitConverter.ToString(sha.ComputeHash(bytes.ToArray())).Replace("-", "").ToLowerInvariant());
            }
        }
    }
}
''', encoding="utf-8", newline="\n")
    binary = tmp_path / "Probe.exe"
    subprocess.run([str(compiler), "/nologo", "/target:exe", "/out:" + str(binary),
                    str(source), str(probe)], check=True, capture_output=True, timeout=60)
    actual = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    encoded = bytearray(struct.pack("<i", len(payload["commands"])))
    for glyph in payload["commands"]:
        encoded.extend(struct.pack("<i", len(glyph)))
        for command in glyph:
            encoded.extend(struct.pack("<i", len(command)))
            for value in command:
                encoded.extend(struct.pack("<d", value))
    assert actual.stdout.strip() == hashlib.sha256(encoded).hexdigest()
