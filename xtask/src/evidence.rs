//! Incremental native campaign evidence, including unsuccessful trials.
#![allow(clippy::disallowed_methods)] // Measure the complete child-process invocation.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    process::Output,
    time::Instant,
};

pub struct Evidence(File);
impl Evidence {
    pub fn create(out: &Path, metadata: &Value) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("trials.jsonl"))?;
        let mut evidence = Self(file);
        evidence.append(&json!({"kind":"campaign_started","metadata":metadata}))?;
        Ok(evidence)
    }
    fn append(&mut self, record: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(record)?;
        bytes.push(b'\n');
        self.0.write_all(&bytes)?;
        self.0.sync_data()?;
        Ok(())
    }
    pub fn finish(&mut self, status: &str, failure_class: Option<&str>) -> Result<()> {
        self.append(
            &json!({"kind":"campaign_finished","status":status,"failure_class":failure_class}),
        )
    }
    pub fn trial(
        &mut self,
        repeat: usize,
        workers: usize,
        invoke: impl FnOnce() -> io::Result<Output>,
    ) -> Result<Value> {
        self.append(&json!({"kind":"trial_started","repeat":repeat,"workers":workers}))?;
        let started = Instant::now();
        let output = invoke();
        let elapsed = started.elapsed().as_secs_f64();
        let mut record = json!({"kind":"trial_completed","repeat":repeat,"workers":workers,"cold_cli_wall_s":elapsed,"status":"failed","exit_code":null});
        let failure = match output {
            Err(_) => Some("process_start_failed"),
            Ok(output) => {
                record["exit_code"] = output.status.code().into();
                record["stdout_bytes"] = output.stdout.len().into();
                record["stderr_bytes"] = output.stderr.len().into();
                let parsed = serde_json::from_slice::<Value>(&output.stdout);
                if !output.status.success() {
                    // Never retain arbitrary child output, including its error message.
                    // Only recognized decision classes survive as failure diagnostics.
                    record["cli_error_class"] = parsed
                        .as_ref()
                        .ok()
                        .and_then(|v| v["error"]["class"].as_str())
                        .and_then(known_class)
                        .into();
                    Some("process_exit_failed")
                } else {
                    match parsed {
                        Err(_) => Some("invalid_report_json"),
                        Ok(report) if super::validate_report(&report).is_err() => {
                            Some("invalid_report")
                        }
                        Ok(report) => {
                            record["status"] = "passed".into();
                            record["report"] = report;
                            None
                        }
                    }
                }
            }
        };
        record["failure_class"] = failure.into();
        self.append(&record)?;
        if failure.is_some() {
            self.finish("failed", failure)?;
        }
        ensure!(
            failure.is_none(),
            "native glyph trial failed; see trials.jsonl for sanitized evidence"
        );
        Ok(record)
    }
}

fn known_class(class: &str) -> Option<&'static str> {
    Some(match class {
        "paused" => "paused",
        "stopped" => "stopped",
        "run_exists" => "run_exists",
        "run_missing" => "run_missing",
        "integrity" => "integrity",
        "provider_failed" => "provider_failed",
        "budget_blocked" => "budget_blocked",
        "quote_blocked" => "quote_blocked",
        "invalid_command" => "invalid_command",
        "journal_changed" => "journal_changed",
        "not_admitting" => "not_admitting",
        "stale_owner" => "stale_owner",
        "unsupported_effect" => "unsupported_effect",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        process::ExitStatus,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "smythe-evidence-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn output(success: bool, stdout: Vec<u8>) -> io::Result<Output> {
        #[cfg(windows)]
        let status = {
            use std::os::windows::process::ExitStatusExt;
            ExitStatus::from_raw(if success { 0 } else { 1 })
        };
        #[cfg(unix)]
        let status = {
            use std::os::unix::process::ExitStatusExt;
            ExitStatus::from_raw(if success { 0 } else { 256 })
        };
        Ok(Output {
            status,
            stdout,
            stderr: b"sensitive child diagnostic".to_vec(),
        })
    }
    fn valid_report() -> Vec<u8> {
        serde_json::to_vec(&json!({"protocol":"noumenon-rust-journal-v3","journal_schema":3,"status":"passed","glyphs":256,"validation_provider_calls":256})).unwrap()
    }
    fn read(out: &Path) -> Vec<Value> {
        fs::read_to_string(out.join("trials.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    #[test]
    fn failure_keeps_provenance_prior_timings_and_sanitized_diagnostics() {
        for failure in 0..4 {
            let out = Directory::new();
            let metadata = json!({"source_revision":"fixture","binary_sha256":"a".repeat(64),"claimable":false});
            let mut evidence = Evidence::create(&out.0, &metadata).unwrap();
            evidence
                .trial(0, 1, || output(true, valid_report()))
                .unwrap();
            let result = evidence.trial(0, 4, || match failure {
                0 => output(false, serde_json::to_vec(&json!({"error":{"class":"integrity","message":"sensitive child diagnostic"}})).unwrap()),
                1 => output(true, b"sensitive child diagnostic".to_vec()),
                2 => output(true, br#"{"status":"sensitive child diagnostic"}"#.to_vec()),
                _ => Err(io::Error::other("sensitive child diagnostic")),
            });
            assert!(result.is_err());
            // Records must be readable before the writer is dropped or the campaign ends.
            let records = read(&out.0);
            assert_eq!(records[0]["metadata"], metadata);
            let completed: Vec<_> = records
                .iter()
                .filter(|r| r["kind"] == "trial_completed")
                .collect();
            assert_eq!(completed.len(), 2);
            assert_eq!(completed[0]["status"], "passed");
            assert_eq!(completed[1]["status"], "failed");
            assert!(
                completed
                    .iter()
                    .all(|r| r["cold_cli_wall_s"].as_f64().unwrap() >= 0.0)
            );
            assert_eq!(records.last().unwrap()["status"], "failed");
            if failure == 0 {
                assert_eq!(completed[1]["cli_error_class"], "integrity");
            }
            assert!(
                !fs::read_to_string(out.0.join("trials.jsonl"))
                    .unwrap()
                    .contains("sensitive child diagnostic")
            );
        }
    }
    #[test]
    fn successful_campaign_keeps_every_trial_and_refuses_evidence_overwrite() {
        let out = Directory::new();
        let mut evidence = Evidence::create(&out.0, &json!({"claimable":false})).unwrap();
        for workers in [1, 4, 8] {
            evidence
                .trial(0, workers, || output(true, valid_report()))
                .unwrap();
        }
        evidence.finish("passed", None).unwrap();
        assert_eq!(
            read(&out.0)
                .iter()
                .filter(|r| r["kind"] == "trial_completed")
                .count(),
            3
        );
        assert_eq!(read(&out.0).last().unwrap()["status"], "passed");
        assert!(Evidence::create(&out.0, &json!({})).is_err());
    }
    #[test]
    fn unrecognized_error_classes_are_omitted() {
        let out = Directory::new();
        let mut evidence = Evidence::create(&out.0, &json!({})).unwrap();
        assert!(
            evidence
                .trial(0, 1, || output(
                    false,
                    br#"{"error":{"class":"untrusted diagnostic"}}"#.to_vec()
                ))
                .is_err()
        );
        assert!(
            !fs::read_to_string(out.0.join("trials.jsonl"))
                .unwrap()
                .contains("untrusted diagnostic")
        );
    }
}
