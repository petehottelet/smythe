"""Linux native catalog parity and executable rendering checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from screensaver.export_glyphs import _c_header, _payload


ROOT = Path(__file__).resolve().parents[1]


def test_linux_catalog_is_generated_from_the_same_192_glyphs():
    data = _payload()
    header = ROOT / "screensaver/linux/glyph_data.h"
    assert len(data["strokes"]) == 192
    assert header.read_text(encoding="utf-8") == _c_header(data)
    assert len(data["speeds"]) == len(data["trails"]) == 192


@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux executable test")
def test_linux_elf_renders_and_embeds_under_xvfb(tmp_path):
    for command in ("cc", "pkg-config", "xvfb-run"):
        if shutil.which(command) is None:
            pytest.skip(f"Linux native smoke needs {command}")
    dependencies = subprocess.run(["pkg-config", "--exists", "x11", "cairo"], check=False)
    if dependencies.returncode:
        pytest.skip("Linux native smoke needs X11/Cairo development packages")
    if importlib.util.find_spec("PIL") is None:
        pytest.skip("Linux native smoke needs Pillow")
    binary = tmp_path / "smythe-glyph-rain"
    subprocess.run(["sh", str(ROOT / "screensaver/linux/build_linux.sh"), str(binary)],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", sys.executable,
                    str(ROOT / "screensaver/linux/smoke_linux.py"), str(binary),
                    "--out", str(tmp_path / "verification")],
                   check=True, capture_output=True, text=True, timeout=90)
