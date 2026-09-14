"""Large-file review reports new binary bytes without flagging unchanged history."""

import subprocess

from tools.large_files import LIMIT, new_large_binaries


def test_large_file_review_distinguishes_text_old_binaries_and_new_binaries(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path).decode().strip()

    def commit():
        git("add", "--", "old.bin", "new.bin", "large.txt", "small.bin")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture")
        return git("rev-parse", "HEAD")

    git("init", "-q")
    for name in ("old.bin", "new.bin", "large.txt", "small.bin"):
        (tmp_path / name).write_bytes(b"\0" * (LIMIT + 1) if name == "old.bin" else b"small")
    base = commit()
    (tmp_path / "new.bin").write_bytes(b"\0" * (LIMIT + 1))
    (tmp_path / "large.txt").write_text("a" * (LIMIT + 1))
    (tmp_path / "small.bin").write_bytes(b"\0" * LIMIT)
    commit()
    findings = new_large_binaries(base, root=tmp_path)
    assert [entry["path"] for entry in findings] == ["new.bin"]
    assert findings[0]["bytes"] == LIMIT + 1
