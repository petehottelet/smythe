"""Check original SVG/MSDF geometry and committed texture provenance offline."""

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from screensaver.export_generated_sdf import (
    CELL, COUNT, PREVIEW, RANGE, SOURCE, combined_svg, command, decode_mask, verify_atlas,
)


def test_compound_input_keeps_every_path_without_refitting_or_mirroring():
    data = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
            'width="128" height="128">'
            '<path fill="#000000" fill-rule="nonzero" d="M10 20L30 20L30 40Z"/>'
            '<path fill="#000000" fill-rule="nonzero" d="M70 80L75 80L75 90Z"/></svg>')
    result = ET.fromstring(combined_svg(data))
    assert result.get("viewBox") == "0 0 100 100"
    assert result.get("width") == result.get("height") == "100"
    assert len(result) == 1
    assert result[0].get("d") == "M10 20L30 20L30 40Z M70 80L75 80L75 90Z"
    assert result[0].get("fill-rule") == "nonzero"


@pytest.mark.parametrize("content", [
    "", '<image href="external.svg"/>',
    '<path fill="red" fill-rule="nonzero" d="M0 0L1 0L1 1Z"/>',
    '<path fill="#000000" fill-rule="evenodd" d="M0 0L1 0L1 1Z"/>',
    '<path fill="#000000" fill-rule="nonzero" transform="scale(2)" d="M0 0L1 0L1 1Z"/>',
])
def test_unexpected_source_content_fails_closed(content):
    with pytest.raises(ValueError):
        combined_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                     + content + '</svg>')


def test_encoding_contract_uses_true_msdf_and_full_cell_coordinates():
    args = command(Path("msdfgen"), Path("source.svg"), Path("output.png"))
    assert args[1] == "msdf"
    assert args[args.index("-dimensions")+1:args.index("-dimensions")+3] == ["128", "128"]
    assert args[args.index("-scale")+1] == "1.28"
    assert args[args.index("-pxrange")+1] == str(RANGE) == "16"
    assert args[args.index("-fillrule")+1] == "nonzero"
    assert "-autoframe" not in args and "-yflip" not in args
    assert "-scanline" in args and "-nopreprocess" in args


def test_msdf_decode_uses_channel_median_instead_of_a_single_channel():
    pixels = np.array([[[255, 0, 0], [0, 255, 255], [127, 127, 255], [128, 128, 0]]], dtype=np.uint8)
    assert decode_mask(Image.fromarray(pixels)).tolist() == [[False, True, False, True]]


def test_complete_committed_atlas_is_bound_to_all_original_sources():
    receipt = verify_atlas()
    assert receipt["count"] == COUNT == 192
    assert receipt["channels"].startswith("RGB multi-channel")
    assert receipt["validation"]["minimum_iou"] >= .94
    assert receipt["validation"]["all_browser_paths_match_source_svgs"] is True
    assert all(not record["tool_stderr"] for record in receipt["glyphs"])
    assert all(record["source_raster_iou_at128"] >= .94 for record in receipt["glyphs"])
    # Regenerate every input SVG serialization: multiple paths cannot disappear.
    for record in receipt["glyphs"]:
        root = ET.fromstring(combined_svg((SOURCE/f'{record["glyph_id"]}.svg').read_text()))
        assert root[0].get("d")


def test_corrupted_atlas_cannot_pass_source_binding_check(tmp_path):
    for name in ("generated-sdf.png", "generated-sdf.json"):
        shutil.copyfile(PREVIEW/name, tmp_path/name)
    with Image.open(tmp_path/"generated-sdf.png") as original:
        corrupted = original.copy()
    corrupted.paste((0, 0, 0), (0, 0, CELL, CELL))
    corrupted.save(tmp_path/"generated-sdf.png")
    with pytest.raises(ValueError, match="atlas image changed"):
        verify_atlas(tmp_path)


def test_reordered_glyph_receipts_fail_even_with_unchanged_texture(tmp_path):
    shutil.copyfile(PREVIEW/"generated-sdf.png", tmp_path/"generated-sdf.png")
    receipt = json.loads((PREVIEW/"generated-sdf.json").read_text())
    receipt["glyphs"][0], receipt["glyphs"][1] = receipt["glyphs"][1], receipt["glyphs"][0]
    (tmp_path/"generated-sdf.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="order changed"):
        verify_atlas(tmp_path)
