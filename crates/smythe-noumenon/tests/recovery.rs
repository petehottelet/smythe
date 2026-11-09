use std::fs;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[test]
fn killed_process_reuses_committed_batch_and_finishes_catalog() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
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
    let started = Instant::now();
    while !out.join("receipts/batch-000.json").is_file() {
        if started.elapsed() > Duration::from_secs(90) || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let _ = child.wait();
            panic!("process did not reach its first checkpoint");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let committed = fs::read(out.join("receipts/batch-000.json")).unwrap();
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
    assert_eq!(report["reused_glyphs"], 16);
    assert_eq!(report["provider_calls"], 16);
    assert_eq!(report["compared_pairs"], 496);
    assert_eq!(
        fs::read(out.join("receipts/batch-000.json")).unwrap(),
        committed
    );
    let reference: Vec<String> =
        serde_json::from_str(include_str!("fixtures/python_svg_sha256.json")).unwrap();
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
