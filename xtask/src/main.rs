//! Native campaign runner: committed source bytes and a clean tree are required.
#![allow(clippy::disallowed_methods)] // Report/benchmark timing only.
mod provenance;
use anyhow::{Context, Result, ensure};
use provenance::{committed_sources, hash, require_clean};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

fn execute(root: &Path, out: &Path, repeats: usize) -> Result<Value> {
    let revision = require_clean(root)?;
    let sources = committed_sources(root, &revision)?;
    ensure!(!out.exists(), "campaign output already exists");
    ensure!(
        (1..=1000).contains(&repeats),
        "repeat count must be 1 through 1000"
    );
    let target = root.join("target/native-benchmark");
    ensure!(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "build",
                "--release",
                "--locked",
                "--offline",
                "-p",
                "smythe-cli"
            ])
            .arg("--target-dir")
            .arg(&target)
            .status()?
            .success(),
        "native build failed"
    );
    ensure!(
        require_clean(root)? == revision,
        "source changed during build"
    );
    let binary = target.join("release").join(if cfg!(windows) {
        "smythe.exe"
    } else {
        "smythe"
    });
    let binary_hash = hash(&fs::read(&binary)?);
    fs::create_dir_all(out)?;
    let mut trials = vec![];
    for repeat in 0..repeats {
        let widths = if repeat % 2 == 0 {
            [1, 4, 8]
        } else {
            [8, 4, 1]
        };
        for workers in widths {
            let trial = out.join(format!("r{repeat}-w{workers}"));
            let started = Instant::now();
            let output = Command::new(&binary)
                .current_dir(root)
                .args([
                    "noumenon",
                    "--glyphs",
                    "256",
                    "--workers",
                    &workers.to_string(),
                    "--mode",
                    "full",
                    "--out",
                ])
                .arg(&trial)
                .output()?;
            let elapsed = started.elapsed().as_secs_f64();
            ensure!(output.status.success(), "native glyph trial failed");
            let report: Value = serde_json::from_slice(&output.stdout)?;
            ensure!(
                report["status"] == "passed" && report["glyphs"] == 256,
                "native trial did not pass acceptance"
            );
            trials.push(json!({"repeat":repeat,"workers":workers,"cold_cli_wall_s":elapsed,"report":report}));
        }
    }
    ensure!(
        require_clean(root)? == revision,
        "source changed during campaign; results are invalid"
    );
    ensure!(
        hash(&fs::read(&binary)?) == binary_hash,
        "binary changed during campaign"
    );
    let tool_version = |name: &str| -> Result<String> {
        let output = Command::new(name).arg("--version").output()?;
        ensure!(output.status.success(), "cannot record compiler version");
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    };
    let record = json!({"protocol":"noumenon-native-journal-v1","claimable":false,"evidence_status":"diagnostic","source_revision":revision,"source_hash_basis":"committed_git_blob_bytes","source_sha256":sources,"binary_sha256":binary_hash,"environment":{"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"logical_cpus":std::thread::available_parallelism()?.get(),"rustc":tool_version("rustc")?,"cargo":tool_version("cargo")?},"clock":"fresh CLI process through terminal report publication","trials":trials});
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}
fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned();
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("provenance") => {
            let revision = require_clean(&root)?;
            let sources = committed_sources(&root, &revision)?;
            ensure!(
                require_clean(&root)? == revision,
                "source changed while hashing"
            );
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"revision":revision,"source_hash_basis":"committed_git_blob_bytes","source_sha256":sources})
                )?
            );
        }
        Some("benchmark") => {
            let out = args
                .get(1)
                .context("usage: cargo run -p xtask -- benchmark OUTPUT [REPEATS]")?;
            let repeats = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3);
            let out = std::env::current_dir()?.join(out);
            let record = execute(&root, &out, repeats)?;
            println!(
                "{}",
                json!({"status":"recorded","trials":record["trials"].as_array().unwrap().len(),"claimable":false})
            );
        }
        _ => anyhow::bail!("usage: cargo run -p xtask -- provenance | benchmark OUTPUT [REPEATS]"),
    }
    Ok(())
}
