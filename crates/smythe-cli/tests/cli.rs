use serde_json::{Value, json};
use smythe::{ExecutionGraph, OfflineProvider, Swarm, store::Store};

#[test]
fn noumenon_uses_the_same_typed_cli_failure_contract() {
    let f = Fixture::new();
    let out = f.root.join("glyphs");
    let invoke = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_smythe"))
            .args(["noumenon", "--out"])
            .arg(&out)
            .args(extra)
            .output()
            .unwrap()
    };
    check(invoke(&["--glyphs", "0"]), 2, Some("invalid_command"));
    check(
        invoke(&["--glyphs", "1", "--resume"]),
        6,
        Some("run_missing"),
    );
    check(invoke(&["--glyphs", "1", "--mode", "compile"]), 0, None);
    check(
        invoke(&["--glyphs", "1", "--mode", "compile"]),
        6,
        Some("run_exists"),
    );
    fs::write(out.join("GLYPH-000.svg"), "damaged").unwrap();
    check(
        invoke(&["--glyphs", "1", "--mode", "compile", "--resume"]),
        5,
        Some("integrity"),
    );
}
use smythe_core::{
    NanoUsd,
    journal::{Control, Effect, Event},
};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};
struct Fixture {
    root: PathBuf,
    graph: PathBuf,
    store: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "smythe-cli-{}",
            smythe::seams::Sources::default().ids.next_id()
        ));
        fs::create_dir(&root).unwrap();
        let graph = root.join("graph.json");
        let store = root.join("store");
        fs::write(
            &graph,
            serde_json::to_vec(&json!({"nodes":[{"id":"a","task":{"goal":"fixture","input":1}}]}))
                .unwrap(),
        )
        .unwrap();
        Self { root, graph, store }
    }
    fn run(&self, id: &str, extra: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_smythe"))
            .args(["run", "--graph"])
            .arg(&self.graph)
            .arg("--store")
            .arg(&self.store)
            .args(["--id", id])
            .args(extra)
            .output()
            .unwrap()
    }
    fn cmd(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_smythe"))
            .args(args)
            .arg("--store")
            .arg(&self.store)
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn check(output: Output, code: i32, class: Option<&str>) -> Value {
    assert_eq!(
        output.status.code(),
        Some(code),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    if let Some(class) = class {
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["status"], "error");
        assert_eq!(value["error"]["class"], class);
        assert!(!output.stderr.is_empty());
    }
    value
}
#[test]
fn cli_success_errors_and_read_only_observation_have_stable_contracts() {
    let f = Fixture::new();
    check(f.cmd(&["list"]), 6, Some("run_missing"));
    assert!(!f.store.exists());
    let result = check(f.run("ok", &[]), 0, None);
    assert_eq!(result["outputs"][0]["node"], "a");
    assert!(result["outputs"][0]["bytes"].as_u64().unwrap() > 0);
    assert_eq!(result["outputs"][0]["sha256"].as_str().unwrap().len(), 64);
    let state = check(f.cmd(&["inspect", "ok"]), 0, None);
    assert_eq!(state["status"], "completed");
    assert_eq!(state["shape"]["nodes"], 1);
    assert_eq!(state["health"]["owner"], "absent");
    check(f.run("ok", &[]), 6, Some("run_exists"));
    check(
        f.run("bad", &["--workers", "0"]),
        2,
        Some("invalid_command"),
    );
    check(
        f.run("failed", &["--max-output-bytes", "0"]),
        1,
        Some("provider_failed"),
    );
    check(f.run("missing", &["--resume"]), 6, Some("run_missing"));
    check(
        f.run("ok", &["--resume", "--budget-nanousd", "10"]),
        2,
        Some("invalid_command"),
    );
    let inline = check(f.run("ok", &["--resume", "--outputs", "inline"]), 0, None);
    assert_eq!(inline["outputs"][0]["input"], 1);
    let output = check(f.cmd(&["output", "ok", "a"]), 0, None);
    assert_eq!(output, inline["outputs"][0]);
    let before = fs::read(f.store.join("journal.sqlite3")).unwrap();
    check(f.cmd(&["list"]), 0, None);
    let page = check(
        f.cmd(&["inspect", "ok", "--events", "--after", "0", "--limit", "1"]),
        0,
        None,
    );
    assert_eq!(page["events"].as_array().unwrap().len(), 1);
    assert_eq!(page["next_sequence"], 1);
    assert_eq!(fs::read(f.store.join("journal.sqlite3")).unwrap(), before);
    check(
        f.cmd(&["inspect", "ok", "--after", "0"]),
        2,
        Some("invalid_command"),
    );
    let db = rusqlite::Connection::open(f.store.join("journal.sqlite3")).unwrap();
    db.execute(
        "UPDATE events SET digest='corrupt' WHERE run_id='ok' AND sequence=1",
        [],
    )
    .unwrap();
    drop(db);
    check(f.cmd(&["inspect", "ok"]), 5, Some("integrity"));
}
#[test]
fn cli_pause_stop_and_budget_errors_preserve_run_identity() {
    let f = Fixture::new();
    let store = Store::open(&f.store).unwrap();
    let g: ExecutionGraph = serde_json::from_slice(&fs::read(&f.graph).unwrap()).unwrap();
    Swarm::new(1, 1)
        .unwrap()
        .execute_in(&g, &OfflineProvider, &store, "reference", false)
        .unwrap();
    let started = store.history("reference").unwrap()[0].event.clone();
    for (id, control, class) in [
        ("paused", Control::Pause, "paused"),
        ("stopped", Control::Stop, "stopped"),
    ] {
        store.append(id, 0, 0, started.clone()).unwrap();
        store
            .append(id, 1, 1, Event::Claimed { generation: 1 })
            .unwrap();
        store.control(id, control).unwrap();
        let result = check(f.run(id, &["--resume"]), 3, Some(class));
        assert_eq!(result["run_id"], id);
    }
    store.append("budget", 0, 0, started).unwrap();
    store
        .append_batch(
            "budget",
            1,
            1,
            vec![
                Event::Claimed { generation: 1 },
                Event::Admitted {
                    node: "a".into(),
                    attempt: 1,
                    bound: NanoUsd(1),
                    effect: Effect::Pure,
                },
            ],
        )
        .unwrap();
    let result = check(
        f.cmd(&["control", "budget", "budget", "0"]),
        4,
        Some("budget_blocked"),
    );
    assert_eq!(result["run_id"], "budget");
    drop(store);
}
