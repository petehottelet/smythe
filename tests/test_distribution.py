"""Distribution audits reject leaked files and changed rebuilds."""

import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from tools.distribution import audit, compare_wheels, source_path_allowed


@pytest.mark.parametrize("name", [
    "00_project_files/private.md", "smythe/tmp/scratch.py", "smythe/__pycache__/x.pyc",
    "smythe/.env", "smythe/credentials.json", "tests/output/data.py", "../secrets.py",
    "/smythe/x.py", "smythe\\x.py", "assets/preview.gif", "benchmarks/results.json",
])
def test_private_generated_and_repository_materials_are_not_package_sources(name):
    assert not source_path_allowed(name)


@pytest.fixture
def archives(tmp_path):
    (tmp_path / "tools").mkdir()
    source = {"smythe/__init__.py": b"", "smythe/py.typed": b""}
    (tmp_path / "tools/distribution-files.json").write_text(
        json.dumps({"source_files": list(source)}), encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.7.0"\n')
    meta = "smythe-0.7.0.dist-info/"
    metadata = (b"Name: smythe\nVersion: 0.7.0\nDescription-Content-Type: text/markdown\n\n"
                b"https://raw.githubusercontent.com/petehottelet/smythe/v0.7.0/assets/wordmark.svg\n")
    wheel = dict(source, **{meta + "METADATA": metadata, meta + "WHEEL": b"",
                          meta + "RECORD": b"", meta + "licenses/LICENSE": b"MIT",
                          meta + "entry_points.txt": b"smythe = smythe.cli:main\n"})
    source["PKG-INFO"] = metadata

    def write(*, add_source=None, add_wheel=None, missing=None):
        src, whl = dict(source), dict(wheel)
        src.update(add_source or {})
        whl.update(add_wheel or {})
        if missing:
            whl.pop(missing)
        sdist, package = tmp_path / "source.tar.gz", tmp_path / "package.whl"
        with tarfile.open(sdist, "w:gz") as archive:
            for name, data in src.items():
                info = tarfile.TarInfo("smythe-0.7.0/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        with zipfile.ZipFile(package, "w") as archive:
            for name, data in whl.items():
                archive.writestr(name, data)
        return sdist, package

    return tmp_path, write


def test_package_audit_reports_exact_inventory_and_sizes(archives):
    root, write = archives
    source, wheel = write()
    report = audit(source, wheel, root=root)
    assert report["source"]["members"] == 3
    assert report["wheel"]["members"] == 7
    assert report["source"]["bytes"] == source.stat().st_size


@pytest.mark.parametrize("name", ["smythe/untracked_scratch.py", "tests/.env", "00_project_files/plan.md"])
def test_audit_rejects_unapproved_files_even_inside_allowed_tree(archives, name):
    root, write = archives
    with pytest.raises(ValueError, match="Unapproved source member"):
        audit(*write(add_source={name: b"private"}), root=root)


def test_audit_requires_the_typing_marker_when_listed(archives):
    root, write = archives
    with pytest.raises(ValueError, match="Wheel inventory mismatch"):
        audit(*write(missing="smythe/py.typed"), root=root)


def test_audit_rejects_different_package_bytes(archives):
    root, write = archives
    with pytest.raises(ValueError, match="package bytes differ"):
        audit(*write(add_wheel={"smythe/__init__.py": b"changed"}), root=root)


def test_rebuild_comparison_checks_contents_and_metadata(tmp_path):
    paths = [tmp_path / name for name in ("first.whl", "second.whl")]
    for path in paths:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("package.py", b"same")
    compare_wheels(*paths)
    with zipfile.ZipFile(paths[1], "w") as archive:
        archive.writestr("package.py", b"changed")
    with pytest.raises(ValueError, match="Rebuilt wheel differs"):
        compare_wheels(*paths)


def test_manifest_is_a_unique_allowlist_of_existing_approved_paths():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "tools/distribution-files.json").read_text(encoding="utf-8"))
    names = manifest["source_files"]
    assert names == sorted(set(names))
    assert all(source_path_allowed(name) and (root / name).is_file() for name in names)
