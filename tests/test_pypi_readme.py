"""Build-time release links do not rewrite source or import Smythe."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smythe_readme_build", ROOT / "hatch_build.py")
HOOK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOK)


def test_current_readme_has_absolute_release_docs_and_raw_images_without_source_changes():
    path = ROOT / "README.md"
    before = path.read_bytes()
    text = before.decode()
    output = HOOK.render_pypi_readme(text, "0.7.0")
    assert path.read_bytes() == before
    assert 'src="https://raw.githubusercontent.com/petehottelet/smythe/v0.7.0/assets/wordmark.svg"' in output
    assert "https://github.com/petehottelet/smythe/blob/v0.7.0/docs/index.md" in output
    assert "https://github.com/petehottelet/smythe/tree/v0.7.0/examples/acquisition_diligence/" in output
    assert "https://github.com/petehottelet/smythe/blob/v0.7.0/ROADMAP.md#coming-soon" in output
    assert 'href="#quickstart"' in output
    assert "https://developers.openai.com/api/docs/models/gpt-6-astra" in output
    assert HOOK.render_pypi_readme(output, "0.7.0") == output


def test_markdown_html_images_fragments_queries_and_encoded_paths():
    source = ('[guide](./docs/a%20b.md?plain=1#two) ![figure](assets/a.png)\n'
              "<a href='docs/read.md?x=1&amp;y=2#anchor'>read</a>\n"
              '<img width="80" src="assets/img.svg" alt="image">\n')
    value = HOOK.render_pypi_readme(source, "0.8.0rc1")
    assert "blob/v0.8.0rc1/docs/a%20b.md?plain=1#two" in value
    assert "![figure](https://raw.githubusercontent.com/petehottelet/smythe/v0.8.0rc1/assets/a.png)" in value
    assert "blob/v0.8.0rc1/docs/read.md?x=1&amp;y=2#anchor" in value
    assert 'width="80"' in value and 'alt="image"' in value


def test_only_registry_badge_uses_latest_main_and_generic_label():
    source = ('<img src="assets/badges/pypi.svg" alt="PyPI v0.6.0">\n'
              '<img src="assets/badges/python.svg" alt="Python 3.11+">\n'
              '![PyPI v0.6.0](assets/badges/pypi.svg)\n')
    rendered = HOOK.render_pypi_readme(source, "0.7.0")
    assert 'src="https://raw.githubusercontent.com/petehottelet/smythe/main/assets/badges/pypi.svg"' in rendered
    assert 'alt="Latest PyPI release"' in rendered
    assert 'v0.7.0/assets/badges/python.svg" alt="Python 3.11+"' in rendered
    assert '![Latest PyPI release](https://raw.githubusercontent.com/petehottelet/smythe/main/assets/badges/pypi.svg)' in rendered
    assert "v0.6.0" not in rendered
    assert HOOK.render_pypi_readme(rendered, "0.7.0") == rendered


def test_code_and_external_destinations_are_unchanged():
    source = ('```python\nprint("[example](docs/not-a-link.md)")\n```\n'
              '~~~text\n<img src="example.png">\n~~~\n'
              '`[example](docs/not-a-link.md)` ``<a href="example.md">x</a>``\n'
              '[API](https://example.com/x) [email](mailto:test@example.com)\n'
              '<img src="https://example.com/a.svg"> [cdn](//example.com/x)\n')
    assert HOOK.render_pypi_readme(source, "0.7.0") == source


@pytest.mark.parametrize("text", [
    "[x](../outside.md)", "[x](%2e%2e/outside.md)", "[x](/root.md)",
    "[x](<docs/a b.md>)", '[x](docs/a.md "Title")', "[x](docs/a(b).md)",
    '<img src=assets/a.svg>', '[ref]: docs/a.md', '```python\nunfinished',
])
def test_unsupported_or_escaping_relative_markup_fails_build(text):
    with pytest.raises(ValueError):
        HOOK.render_pypi_readme(text, "0.7.0")


@pytest.mark.parametrize("version", [None, True, "", "main", "0.7/branch", "0.7.0#bad"])
def test_release_version_rejects_ambiguous_url_values(version):
    with pytest.raises(ValueError):
        HOOK.render_pypi_readme("[guide](docs/read.md)", version)


def test_hook_pure_import_needs_neither_hatchling_nor_smythe():
    code = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('hook',{str(ROOT / 'hatch_build.py')!r}); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert not any(n=='smythe' or n.startswith('smythe.') or n=='hatchling' "
        "or n.startswith('hatchling.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=15)


def test_project_declares_custom_dynamic_readme_without_runtime_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "readme" in data["project"]["dynamic"] and "readme" not in data["project"]
    assert data["tool"]["hatch"]["metadata"]["hooks"]["custom"] == {}
    assert data["project"]["dependencies"] == ["pyyaml>=6.0"]


def test_sdist_rebuild_keeps_readme_hook_and_identical_wheel_metadata(tmp_path, monkeypatch):
    build = pytest.importorskip("hatchling.build", reason="Build backend required for sdist integration")
    source, first, second, archives = (tmp_path / name for name in ("source", "first", "second", "archives"))
    for path in (source, first, second, archives):
        path.mkdir()
    (source / "smythe").mkdir()
    (source / "smythe/__init__.py").write_text('__version__ = "0.7.0"\n')
    (source / "hatch_build.py").write_bytes((ROOT / "hatch_build.py").read_bytes())
    original = b'# Smythe\n[Guide](docs/index.md)\n<img src="assets/chart.svg">\n'
    (source / "README.md").write_bytes(original)
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        '[project]\nname="smythe"\nversion="0.7.0"\ndynamic=["readme"]\n'
        '[tool.hatch.metadata.hooks.custom]\n', encoding="utf-8",
    )
    monkeypatch.chdir(source)
    wheel1 = first / build.build_wheel(str(first))
    sdist = archives / build.build_sdist(str(archives))
    assert (source / "README.md").read_bytes() == original
    with tarfile.open(sdist) as archive:
        members = archive.getnames()
        assert "smythe-0.7.0/README.md" in members and "smythe-0.7.0/hatch_build.py" in members
        for member in archive.getmembers():
            assert not member.issym() and not member.islnk() and ".." not in Path(member.name).parts
        archive.extractall(tmp_path / "rebuild", filter="data")
    monkeypatch.chdir(tmp_path / "rebuild/smythe-0.7.0")
    wheel2 = second / build.build_wheel(str(second))
    with zipfile.ZipFile(wheel1) as a, zipfile.ZipFile(wheel2) as b:
        assert sorted(a.namelist()) == sorted(b.namelist())
        assert all(a.read(name) == b.read(name) for name in a.namelist())
        metadata = a.read("smythe-0.7.0.dist-info/METADATA").decode()
        assert "Description-Content-Type: text/markdown" in metadata
        assert "blob/v0.7.0/docs/index.md" in metadata and "v0.7.0/assets/chart.svg" in metadata
        assert "hatch_build.py" not in a.namelist()
