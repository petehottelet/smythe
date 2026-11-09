use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, process::Command};

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git").current_dir(root).args(args).output()?;
    ensure!(output.status.success(), "Git provenance command failed");
    Ok(output.stdout)
}
pub fn require_clean(root: &Path) -> Result<String> {
    let status = git(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
    )?;
    let flags = git(root, &["ls-files", "-v", "-z"])?;
    validate_status(&status, &flags)?;
    Ok(String::from_utf8(git(root, &["rev-parse", "HEAD"])?)?
        .trim()
        .to_owned())
}
fn validate_status(status: &[u8], flags: &[u8]) -> Result<()> {
    ensure!(
        status.is_empty(),
        "benchmark refuses an uncommitted tree; commit the intended source changes first"
    );
    ensure!(
        flags
            .split(|b| *b == 0)
            .filter(|v| !v.is_empty())
            .all(|v| v[0] == b'H'),
        "benchmark refuses hidden assume-unchanged or skip-worktree entries"
    );
    Ok(())
}
fn source_path(path: &str) -> bool {
    path == "Cargo.toml"
        || path == "Cargo.lock"
        || path.starts_with("crates/")
        || path.starts_with("xtask/")
        || path == "benchmarks/noumenon/glyph_contours_v2.json"
        || path == "benchmarks/partitions/glyph_svg_v2_256/contours-extra.json"
}
pub fn committed_sources(root: &Path, revision: &str) -> Result<BTreeMap<String, String>> {
    let listing = git(root, &["ls-tree", "-r", "-z", revision])?;
    let mut hashes = BTreeMap::new();
    for entry in listing.split(|b| *b == 0).filter(|entry| !entry.is_empty()) {
        let entry = std::str::from_utf8(entry)?;
        let (metadata, path) = entry.split_once('\t').context("invalid tree record")?;
        if !source_path(path) {
            continue;
        }
        let fields: Vec<_> = metadata.split_whitespace().collect();
        ensure!(
            fields.len() == 3 && fields[1] == "blob",
            "source must be a committed blob"
        );
        hashes.insert(
            path.into(),
            hash(&git(root, &["cat-file", "blob", fields[2]])?),
        );
    }
    ensure!(
        hashes.contains_key("Cargo.lock") && hashes.contains_key("Cargo.toml"),
        "missing committed workspace"
    );
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_staged_unstaged_untracked_and_hidden_changes() {
        for status in [
            b" M Cargo.toml\n".as_slice(),
            b"M  Cargo.toml\n",
            b"?? crates/new.rs\n",
        ] {
            assert!(validate_status(status, b"H Cargo.toml\0").is_err());
        }
        for flags in [b"h Cargo.toml\0".as_slice(), b"S Cargo.toml\0"] {
            assert!(validate_status(b"", flags).is_err());
        }
        assert!(validate_status(b"", b"H Cargo.toml\0").is_ok());
    }
    #[test]
    fn raw_bytes_do_not_normalize_line_endings() {
        assert_ne!(hash(b"a\nb\n"), hash(b"a\r\nb\r\n"));
        assert_eq!(
            hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
    #[test]
    fn includes_runtime_and_assets_excludes_private_material() {
        for p in [
            "crates/smythe/src/store.rs",
            "xtask/src/main.rs",
            "Cargo.lock",
            "benchmarks/noumenon/glyph_contours_v2.json",
        ] {
            assert!(source_path(p));
        }
        for p in [
            "00_project_files/plan.md",
            ".env",
            "target/release/smythe.exe",
        ] {
            assert!(!source_path(p));
        }
    }
}
