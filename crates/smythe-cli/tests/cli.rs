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

fn diagnostic(output: Output, code: i32, class: &str, forbidden: &[&str]) -> String {
    for bytes in [&output.stdout, &output.stderr] {
        let text = String::from_utf8(bytes.clone()).unwrap();
        for value in forbidden {
            assert!(!text.contains(value), "private diagnostic escaped");
        }
    }
    let stderr = String::from_utf8(output.stderr.clone()).unwrap();
    let value = check(output, code, Some(class));
    let message = value["error"]["message"].as_str().unwrap();
    assert!(message.len() <= smythe_core::diagnostics::MAX_DIAGNOSTIC_BYTES);
    assert_eq!(stderr.trim_end(), format!("Smythe: {message}"));
    message.into()
}

#[test]
fn graph_diagnostics_explain_cycles_and_misspelled_fields_without_source_snippets() {
    let mut f = Fixture::new();
    fs::write(
        &f.graph,
        r#"{"nodes":[{"id":"a","task":{"goal":"fixture","input":0},"depends_on":["a"]}]}"#,
    )
    .unwrap();
    assert!(diagnostic(f.run("cycle", &[]), 2, "invalid_command", &[]).contains("cycle"));
    for (extension, text) in [
        (
            "json",
            r#"{"nodes":[{"id":"a","task":{"goal":"fixture","input":0},"depend_on":[] }]}"#,
        ),
        (
            "toml",
            "[[nodes]]\nid = 'a'\ndepend_on = []\n[nodes.task]\ngoal = 'fixture'\ninput = 0\n",
        ),
    ] {
        f.graph.set_extension(extension);
        fs::write(&f.graph, text).unwrap();
        let message = diagnostic(f.run("field", &[]), 2, "invalid_command", &["fixture"]);
        assert!(message.contains("unknown field `depend_on`"));
        assert!(message.contains("line") && message.contains("column"));
    }
}

#[test]
fn parser_and_argument_diagnostics_omit_sensitive_values_paths_and_long_unicode() {
    let mut f = Fixture::new();
    let secrets = [
        ["org", "-", "syntheticaccount"].concat(),
        ["sk", "-", "syntheticcredential"].concat(),
        ["proj", "_", "syntheticproject"].concat(),
        ["fixture", "@", "example.invalid"].concat(),
        ["/", "home", "/fixture/private"].concat(),
        ["C:", "\\", "Users", "\\Fixture\\private"].concat(),
        "雪🙂".repeat(2000),
    ];
    for secret in &secrets {
        for extension in ["json", "toml"] {
            f.graph.set_extension(extension);
            // Invalid type, unknown field, and syntax error: all parser paths
            // must omit both the sensitive value and the source excerpt.
            let quoted = serde_json::to_string(secret).unwrap();
            let cases = if extension == "json" {
                vec![
                    format!("{{\"nodes\":{quoted}}}"),
                    format!("{{{quoted}:0,\"nodes\":[]}}"),
                    format!("{{\"nodes\":[] {quoted}}}"),
                ]
            } else {
                vec![
                    format!("nodes = {quoted}"),
                    format!("{quoted} = 0\nnodes = []"),
                    format!("nodes = []\ninvalid {quoted}"),
                ]
            };
            for text in cases {
                fs::write(&f.graph, &text).unwrap();
                let message = diagnostic(
                    f.run("private", &[]),
                    2,
                    "invalid_command",
                    &[secret, &text],
                );
                assert!(message.contains("graph at line"));
            }
        }
        diagnostic(
            f.run("argument", &["--workers", secret]),
            2,
            "invalid_command",
            &[secret],
        );
        diagnostic(
            f.run("argument", &[secret]),
            2,
            "invalid_command",
            &[secret],
        );
    }
    f.graph = f.root.join("missing.json");
    let path = f.graph.to_str().unwrap();
    assert!(
        diagnostic(f.run("missing-file", &[]), 2, "invalid_command", &[path])
            .contains("file or directory not found")
    );
}

#[test]
fn ownership_diagnostics_keep_the_existing_class_and_explain_the_live_owner() {
    let f = Fixture::new();
    check(f.run("owned", &[]), 0, None);
    let store = Store::open(&f.store).unwrap();
    let _lease = store.lease("owned").unwrap();
    let message = diagnostic(f.run("owned", &["--resume"]), 1, "provider_failed", &[]);
    assert!(message.contains("another coordinator owns this run"));
}

#[test]
fn explicit_limits_require_a_cursor_and_paging_keeps_its_existing_shape() {
    let f = Fixture::new();
    check(f.run("pages", &[]), 0, None);
    let message = diagnostic(
        f.cmd(&["inspect", "pages", "--events", "--limit", "2"]),
        2,
        "invalid_command",
        &[],
    );
    assert!(message.contains("--after"));
    let full = check(f.cmd(&["inspect", "pages", "--events"]), 0, None);
    assert!(full["state"].is_object() && full["next_sequence"].is_null());
    let expected = full["events"].as_array().unwrap();
    let mut collected = vec![];
    let mut after = 0;
    loop {
        let page = check(
            f.cmd(&[
                "inspect",
                "pages",
                "--events",
                "--after",
                &after.to_string(),
                "--limit",
                "2",
            ]),
            0,
            None,
        );
        assert!(page["state"].is_null());
        let events = page["events"].as_array().unwrap();
        assert!(events.len() <= 2);
        if events.is_empty() {
            break;
        }
        collected.extend(events.iter().cloned());
        after = page["next_sequence"].as_u64().unwrap();
    }
    assert_eq!(&collected, expected);
    let default = check(
        f.cmd(&["inspect", "pages", "--events", "--after", "0"]),
        0,
        None,
    );
    assert_eq!(default["events"].as_array().unwrap(), expected);
    for limit in ["0", "10001"] {
        check(
            f.cmd(&[
                "inspect", "pages", "--events", "--after", "0", "--limit", limit,
            ]),
            2,
            Some("invalid_command"),
        );
    }
}

#[test]
fn provider_error_text_never_reaches_the_journal_or_resumed_cli_diagnostic() {
    use smythe::{CompletionResult, Dependency, Provider, Task};
    struct Broken(String);
    impl Provider for Broken {
        type Output = Value;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        // A fixture deliberately shares the offline binding so the CLI can
        // inspect the persisted failure without another provider dispatch.
        fn binding(&self) -> String {
            OfflineProvider.binding()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> anyhow::Result<u64> {
            Ok(0)
        }
        fn complete(
            &self,
            _: &Task,
            _: &[Dependency<'_, Value>],
        ) -> anyhow::Result<CompletionResult<Value>> {
            anyhow::bail!(self.0.clone())
        }
    }
    let f = Fixture::new();
    let secret = ["org", "-", "fixtureprivate"].concat();
    let store = Store::open(&f.store).unwrap();
    let graph: ExecutionGraph = serde_json::from_slice(&fs::read(&f.graph).unwrap()).unwrap();
    let error = Swarm::new(1, 0)
        .unwrap()
        .execute_in(&graph, &Broken(secret.clone()), &store, "failure", false)
        .err()
        .unwrap();
    assert!(!format!("{error:#}").contains(&secret));
    assert!(
        !serde_json::to_string(&store.history("failure").unwrap())
            .unwrap()
            .contains(&secret)
    );
    drop(store);
    diagnostic(
        f.run("failure", &["--resume"]),
        1,
        "provider_failed",
        &[&secret],
    );
}
