use std::fs;
use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn killed_process_reuses_validation_journal_and_finishes_catalog() {
    let sources = smythe::seams::Sources::default();
    let nonce = sources.ids.next_id();
    let out = std::env::temp_dir().join(format!(
        "smythe-process-test-{}-{nonce}",
        std::process::id()
    ));
    let mut child = Command::new(env!("CARGO_BIN_EXE_smythe"))
        .args(["noumenon", "--glyphs", "32", "--workers", "1", "--out"])
        .arg(&out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = sources.clock.monotonic();
    let committed = loop {
        if let Ok(store) = smythe::store::Store::open_read_only(out.join(".smythe"))
            && let Ok(state) = store.inspect("noumenon-validate")
            && state.accepted_nodes > 0
        {
            break store.history("noumenon-validate").unwrap();
        }
        if sources.clock.monotonic().saturating_sub(started) > Duration::from_secs(30)
            || child.try_wait().unwrap().is_some()
        {
            let _ = child.kill();
            let _ = child.wait();
            panic!("process did not reach its first checkpoint");
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(!committed.is_empty());
    let result = Command::new(env!("CARGO_BIN_EXE_smythe"))
        .args([
            "noumenon",
            "--glyphs",
            "32",
            "--workers",
            "2",
            "--resume",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(result.status.success(), "resume failed");
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["reused_glyphs"], 32);
    assert!(report["reused_validation"].as_u64().unwrap() >= 1);
    assert_eq!(report["provider_calls"], 0); // All compilation receipts preceded publication.
    let accepted = report.get("saved_report").unwrap_or(&report);
    assert_eq!(accepted["compared_pairs"], 496);
    let store = smythe::store::Store::open_read_only(out.join(".smythe")).unwrap();
    let records = store.history("noumenon-validate").unwrap();
    assert_eq!(
        serde_json::to_value(&records[..committed.len()]).unwrap(),
        serde_json::to_value(committed).unwrap()
    );
    assert!(!out.join("receipts").exists());
    drop(store);
    let reference: Vec<String> = serde_json::from_str(include_str!(
        "../../smythe-noumenon/tests/fixtures/python_svg_sha256.json"
    ))
    .unwrap();
    for (i, expected) in reference.iter().enumerate().take(32) {
        assert_eq!(
            &smythe_noumenon::geometry::sha(
                &fs::read(out.join(format!("GLYPH-{i:03}.svg"))).unwrap()
            ),
            expected
        );
    }
    fs::remove_dir_all(out).unwrap();
}
