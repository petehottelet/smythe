use anyhow::Result;
use serde_json::json;
use smythe::{
    CompletionResult, Control, Effect, EngineError, ExecutionGraph, NanoUsd, Node, Provider, Swarm,
    Task, Topology,
    seams::{Clock, IdSource, Sources},
    store::{Store, digest},
};
use smythe_core::journal::{AttemptState, Event, Projection, Record};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

fn graph() -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![Topology::ForkJoin],
        nodes: [vec![], vec![], vec!["a"], vec!["a", "b"]]
            .into_iter()
            .enumerate()
            .map(|(i, parents)| Node {
                id: ((b'a' + i as u8) as char).to_string(),
                task: Task {
                    goal: "sum".into(),
                    input: (i as u64 + 1).into(),
                },
                depends_on: parents.into_iter().map(str::to_owned).collect(),
            })
            .collect(),
    }
}
struct Sum {
    calls: AtomicUsize,
    bound: u64,
    cost: u64,
    effect: Effect,
}
impl Sum {
    fn new(bound: u64, cost: u64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            bound,
            cost,
            effect: Effect::Pure,
        }
    }
}
impl Provider for Sum {
    type Output = u64;
    fn effect(&self) -> Effect {
        self.effect
    }
    fn binding(&self) -> String {
        "recovery-sum-v1".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(self.bound)
    }
    fn complete(&self, task: &Task, dependencies: &[&u64]) -> Result<CompletionResult<u64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResult {
            output: task.input.as_u64().unwrap() + dependencies.iter().map(|n| **n).sum::<u64>(),
            cost_nanousd: self.cost,
        })
    }
}
struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "smythe-recovery-{}",
            Sources::default().ids.next_id()
        )))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn unaffordable_second_admission_does_not_discard_the_affordable_first() {
    let mut graph = graph();
    graph.nodes.truncate(2);
    let provider = Sum::new(6, 3);
    let store = Store::memory().unwrap();
    let result = Swarm::new(2, 10)
        .unwrap()
        .execute_in(&graph, &provider, &store, "budget", false)
        .unwrap();
    assert_eq!(result.outputs, [1, 2]);
    assert_eq!(result.total_cost_nanousd, 6);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert!(store.inspect("budget").unwrap().budget.pending().is_empty());
}

#[test]
fn independent_admissions_share_a_transaction_before_dispatch() {
    let store = Store::memory().unwrap();
    Swarm::new(3, 4)
        .unwrap()
        .execute_in(&graph(), &Sum::new(1, 1), &store, "batch", false)
        .unwrap();
    let records = store.history("batch").unwrap();
    let initial: Vec<_> = records
        .iter()
        .filter(|r| matches!(&r.event, Event::Admitted {node, ..} if node == "a" || node == "b"))
        .collect();
    assert_eq!(initial.len(), 2);
    assert_eq!(initial[0].transaction_start, initial[1].transaction_start);
    let batches: BTreeSet<_> = records.iter().map(|r| r.transaction_start).collect();
    assert!(batches.len() < 2 * graph().nodes.len() + 3);
}

#[test]
fn blob_publication_failure_retains_cost_and_settles_successful_siblings() {
    let directory = Directory::new();
    let store = Store::open(&directory.0).unwrap();
    // A directory at one immutable blob name makes its publication fail.
    fs::create_dir(directory.0.join("blobs").join(digest(b"1"))).unwrap();
    let provider = Sum::new(1, 1);
    assert!(
        Swarm::new(2, 4)
            .unwrap()
            .execute_in(&graph(), &provider, &store, "io", false)
            .is_err()
    );
    let state = store.inspect("io").unwrap();
    assert_eq!(state.budget.spent, NanoUsd(2));
    assert!(state.budget.pending().is_empty());
    assert_eq!(state.attempts["a"].state, AttemptState::Rejected);
    assert_eq!(state.attempts["b"].state, AttemptState::Accepted);
    assert!(!state.attempts.contains_key("c"));
    assert!(!state.attempts.contains_key("d"));
    assert!(
        store
            .history("io")
            .unwrap()
            .iter()
            .all(|r| !matches!(r.event, Event::Failed { .. }))
    );
}

#[test]
fn typed_errors_survive_context_and_ignore_coincidental_error_text() {
    let error = anyhow::Error::new(EngineError::JournalChanged)
        .context("a completely different display message");
    assert!(EngineError::JournalChanged.matches(&error));
    assert!(!EngineError::JournalChanged.matches(&anyhow::anyhow!("journal changed")));
    let mut budget = smythe::Sentinel::new(NanoUsd(0));
    let error = budget
        .reserve("a", NanoUsd(1))
        .unwrap_err()
        .context("quote refused");
    assert!(EngineError::BudgetExhausted.matches(&error));
    assert!(!error.to_string().contains("budget exhausted"));
}

#[test]
fn pause_racing_an_admission_reloads_typed_revision_errors_without_dispatch() {
    struct PausingQuote {
        store: Store,
        calls: AtomicUsize,
    }
    impl Provider for PausingQuote {
        type Output = u64;
        fn effect(&self) -> Effect {
            Effect::Pure
        }
        fn binding(&self) -> String {
            "pausing-quote-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            self.store.control("race", Control::Pause)?;
            Ok(1)
        }
        fn complete(&self, _: &Task, _: &[&u64]) -> Result<CompletionResult<u64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            unreachable!("paused admission must not dispatch")
        }
    }
    let store = Store::memory().unwrap();
    let provider = PausingQuote {
        store: store.clone(),
        calls: AtomicUsize::new(0),
    };
    let error = Swarm::new(2, 4)
        .unwrap()
        .execute_in(&graph(), &provider, &store, "race", false)
        .err()
        .unwrap();
    assert!(EngineError::NotAdmitting.matches(&error));
    let state = store.inspect("race").unwrap();
    assert_eq!(state.control, Control::Pause);
    assert!(state.attempts.is_empty());
    assert!(state.budget.pending().is_empty());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn serialization_panics_preserve_successful_call_costs() {
    #[derive(serde::Deserialize)]
    struct BrokenOutput;
    impl serde::Serialize for BrokenOutput {
        fn serialize<S: serde::Serializer>(&self, _: S) -> std::result::Result<S::Ok, S::Error> {
            panic!("fixture serialization failure")
        }
    }
    struct SuccessfulCall;
    impl Provider for SuccessfulCall {
        type Output = BrokenOutput;
        fn effect(&self) -> Effect {
            Effect::Pure
        }
        fn binding(&self) -> String {
            "broken-serialization-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(
            &self,
            _: &Task,
            _: &[&BrokenOutput],
        ) -> Result<CompletionResult<BrokenOutput>> {
            Ok(CompletionResult {
                output: BrokenOutput,
                cost_nanousd: 1,
            })
        }
    }
    let store = Store::memory().unwrap();
    assert!(
        Swarm::new(2, 4)
            .unwrap()
            .execute_in(&graph(), &SuccessfulCall, &store, "serialization", false)
            .is_err()
    );
    let state = store.inspect("serialization").unwrap();
    assert_eq!(state.budget.spent, NanoUsd(2));
    assert!(state.budget.pending().is_empty());
    assert!(
        state
            .attempts
            .values()
            .all(|a| a.state == AttemptState::Rejected)
    );
}

#[test]
fn stop_after_inflight_admissions_preserves_terminal_control_on_recovery() {
    let reference = Store::memory().unwrap();
    Swarm::new(3, 4)
        .unwrap()
        .execute_in(&graph(), &Sum::new(1, 1), &reference, "reference", false)
        .unwrap();
    let records = reference.history("reference").unwrap();
    let store = Store::memory().unwrap();
    for record in records
        .iter()
        .take_while(|r| !matches!(r.event, Event::Received { .. }))
    {
        store
            .append(
                "stopped",
                record.sequence - 1,
                record.generation,
                record.event.clone(),
            )
            .unwrap();
    }
    store.control("stopped", Control::Stop).unwrap();
    let provider = Sum::new(1, 1);
    assert!(
        Swarm::new(3, 4)
            .unwrap()
            .execute_in(&graph(), &provider, &store, "stopped", true)
            .is_err()
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.inspect("stopped").unwrap().control, Control::Stop);
    assert!(store.control("stopped", Control::Run).is_err());
}

#[test]
fn external_providers_are_refused_before_creating_or_claiming_runs() {
    let store = Store::memory().unwrap();
    let mut provider = Sum::new(1, 1);
    provider.effect = Effect::External;
    let error = Swarm::new(2, 4)
        .unwrap()
        .execute_in(&graph(), &provider, &store, "external", false)
        .err()
        .unwrap();
    assert!(EngineError::UnsupportedEffect.matches(&error));
    assert!(store.history("external").unwrap().is_empty());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn external_interruption_keeps_exposure_and_rejected_transitions_are_atomic() {
    let store = Store::memory().unwrap();
    Swarm::new(2, 4)
        .unwrap()
        .execute_in(&graph(), &Sum::new(1, 1), &store, "reference", false)
        .unwrap();
    let started = store.history("reference").unwrap()[0].event.clone();
    store.append("external", 0, 0, started).unwrap();
    store
        .append("external", 1, 1, Event::Claimed { generation: 1 })
        .unwrap();
    let state = store
        .append_batch(
            "external",
            2,
            1,
            vec![
                Event::Admitted {
                    node: "a".into(),
                    attempt: 1,
                    bound: NanoUsd(1),
                    effect: Effect::External,
                },
                Event::Dispatched {
                    node: "a".into(),
                    attempt: 1,
                },
            ],
        )
        .unwrap();
    let before = serde_json::to_value(&state).unwrap();
    for event in [
        Event::Failed {
            node: "a".into(),
            attempt: 1,
            reason: "unknown".into(),
        },
        Event::LocalReplay {
            node: "a".into(),
            attempt: 1,
        },
    ] {
        let mut projection = state.clone();
        let first = Record::new(
            "external",
            5,
            1,
            5,
            Event::Control {
                control: Control::Pause,
            },
            10,
            10,
        );
        let second = Record::new("external", 6, 1, 5, event.clone(), 10, 10);
        assert!(
            EngineError::UnsupportedEffect
                .matches(&projection.apply_batch(&[first, second]).unwrap_err())
        );
        assert_eq!(serde_json::to_value(projection).unwrap(), before);
        assert!(
            store
                .append_batch(
                    "external",
                    4,
                    1,
                    vec![
                        Event::Control {
                            control: Control::Pause
                        },
                        event
                    ]
                )
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(store.inspect("external").unwrap()).unwrap(),
            before
        );
    }
    let provider = Sum::new(1, 1);
    let error = Swarm::new(2, 4)
        .unwrap()
        .execute_in(&graph(), &provider, &store, "external", true)
        .err()
        .unwrap();
    assert!(EngineError::UnsupportedEffect.matches(&error));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        serde_json::to_value(store.inspect("external").unwrap()).unwrap(),
        before
    );
    assert_eq!(state.budget.exposure().unwrap(), NanoUsd(1));
}

struct SimClock(AtomicU64);
impl Clock for SimClock {
    fn unix_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
    fn monotonic(&self) -> Duration {
        Duration::from_millis(self.0.load(Ordering::SeqCst))
    }
}
struct SimIds(AtomicU64);
impl IdSource for SimIds {
    fn next_id(&self) -> String {
        format!("sim-{}", self.0.fetch_add(1, Ordering::SeqCst))
    }
}
fn sources() -> (Sources, Arc<SimClock>) {
    let clock = Arc::new(SimClock(AtomicU64::new(1000)));
    (
        Sources {
            clock: clock.clone(),
            ids: Arc::new(SimIds(AtomicU64::new(0))),
            ..Sources::default()
        },
        clock,
    )
}

#[test]
fn metadata_is_replayable_with_clock_skew_and_injected_identities() {
    let (sources, clock) = sources();
    let store = Store::memory_with_sources(sources.clone()).unwrap();
    let g = ExecutionGraph {
        nodes: vec![],
        topology: vec![],
    };
    store
        .append(
            "meta",
            0,
            0,
            Event::Started {
                binding: "fixture".into(),
                graph: g.clone(),
                ceiling: NanoUsd(0),
            },
        )
        .unwrap();
    clock.0.store(500, Ordering::SeqCst);
    store
        .append_observed_batch("meta", 1, 1, vec![(Event::Claimed { generation: 1 }, 900)])
        .unwrap();
    let records = store.history("meta").unwrap();
    assert_eq!(records[1].occurred_at_unix_ms, 900);
    assert_eq!(records[1].recorded_at_unix_ms, 500);
    assert_eq!(records[1].event_id, "meta/2");
    assert!(Projection::replay(&records).is_ok());
    let mut damaged = records.clone();
    damaged[1].run_id = "another".into();
    assert!(Projection::replay(&damaged).is_err());
    let mut damaged = records;
    damaged[1].node = Some("invented".into());
    assert!(Projection::replay(&damaged).is_err());
    let result = Swarm::new(1, 0)
        .unwrap()
        .with_sources(sources)
        .execute(&g, &Sum::new(0, 0))
        .unwrap();
    assert_eq!(result.run_id, "sim-0");
}

#[test]
fn old_alpha_journals_are_refused_without_relabeling_their_schema() {
    let directory = Directory::new();
    fs::create_dir_all(&directory.0).unwrap();
    let path = directory.0.join("journal.sqlite3");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch("PRAGMA user_version=1;").unwrap();
    drop(conn);
    assert!(Store::open(&directory.0).is_err());
    let conn = rusqlite::Connection::open(&path).unwrap();
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}

// Small deterministic schedule generator; all failing seeds are explicit fixtures.
fn choose(seed: &mut u64, count: usize) -> usize {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed % count as u64) as usize
}

#[test]
fn seeded_concurrent_completions_controls_and_every_prefix_recover() {
    let g = graph();
    let reference = Store::memory().unwrap();
    let expected = Swarm::new(3, 4)
        .unwrap()
        .execute_in(&g, &Sum::new(1, 1), &reference, "run", false)
        .unwrap();
    let started = reference.history("run").unwrap()[0].event.clone();
    for seed in [1, 7, 42, 99] {
        let mut random = seed;
        let (sources, clock) = sources();
        let history = Store::memory_with_sources(sources.clone()).unwrap();
        history.append("run", 0, 0, started.clone()).unwrap();
        let mut state = history
            .append("run", 1, 1, Event::Claimed { generation: 1 })
            .unwrap();
        let mut values = BTreeMap::new();
        let mut active = Vec::new();
        while values.len() < g.nodes.len() {
            let mut admissions = Vec::new();
            for node in &g.nodes {
                if !state.attempts.contains_key(&node.id)
                    && active.len() < 3
                    && node.depends_on.iter().all(|id| values.contains_key(id))
                {
                    active.push(node);
                    admissions.extend([
                        Event::Admitted {
                            node: node.id.clone(),
                            attempt: 1,
                            effect: Effect::Pure,
                            bound: NanoUsd(1),
                        },
                        Event::Dispatched {
                            node: node.id.clone(),
                            attempt: 1,
                        },
                    ]);
                }
            }
            if !admissions.is_empty() {
                history
                    .append_batch("run", state.sequence, 1, admissions)
                    .unwrap();
            }
            // Controls race outstanding work; its completion may still settle while paused.
            if choose(&mut random, 2) == 0 {
                history.control("run", Control::Pause).unwrap();
            }
            state = history.inspect("run").unwrap();
            let node = active.remove(choose(&mut random, active.len()));
            let output: u64 = node.task.input.as_u64().unwrap()
                + node.depends_on.iter().map(|id| values[id]).sum::<u64>();
            let hash = history.put(&serde_json::to_vec(&output).unwrap()).unwrap();
            clock.0.fetch_add(1, Ordering::SeqCst);
            state = history
                .append_batch(
                    "run",
                    state.sequence,
                    1,
                    vec![
                        Event::Received {
                            node: node.id.clone(),
                            attempt: 1,
                            output: json!({"sha256":hash}),
                            output_valid: true,
                            cost: NanoUsd(1),
                            elapsed_s: 0.001,
                        },
                        Event::Settled {
                            node: node.id.clone(),
                            attempt: 1,
                        },
                        Event::Accepted {
                            node: node.id.clone(),
                            attempt: 1,
                        },
                    ],
                )
                .unwrap();
            values.insert(node.id.clone(), output);
            if state.control == Control::Pause {
                state = history.control("run", Control::Run).unwrap();
            }
        }
        history
            .append("run", state.sequence, 1, Event::Completed)
            .unwrap();
        let records = history.history("run").unwrap();
        for cut in 1..=records.len() {
            let recovered = Store::memory_with_sources(sources.clone()).unwrap();
            // Preserve complete transaction groups; the last may be a synthetic
            // event prefix, separately exercising intermediate state transitions.
            let mut offset = 0;
            while offset < cut {
                let end = (offset + 1..cut)
                    .find(|&i| records[i].transaction_start != records[offset].transaction_start)
                    .unwrap_or(cut);
                for record in &records[offset..end] {
                    if let Event::Received { output, .. } = &record.event {
                        recovered.put(&history.output(output).unwrap()).unwrap();
                    }
                }
                recovered
                    .append_batch(
                        "run",
                        offset as u64,
                        records[offset].generation,
                        records[offset..end]
                            .iter()
                            .map(|r| r.event.clone())
                            .collect(),
                    )
                    .unwrap();
                offset = end;
            }
            let before = recovered.inspect("run").unwrap();
            if before.control == Control::Pause {
                recovered.control("run", Control::Run).unwrap();
            }
            let provider = Sum::new(1, 1);
            let result = Swarm::new(3, 4)
                .unwrap()
                .with_sources(sources.clone())
                .execute_in(&g, &provider, &recovered, "run", true)
                .unwrap();
            assert_eq!(result.outputs, expected.outputs, "seed {seed}, cut {cut}");
            assert_eq!(result.total_cost_nanousd, 4, "seed {seed}, cut {cut}");
            let received = records[..cut]
                .iter()
                .filter(|r| matches!(r.event, Event::Received { .. }))
                .count();
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                4 - received,
                "seed {seed}, cut {cut}"
            );
            assert!(
                recovered
                    .inspect("run")
                    .unwrap()
                    .budget
                    .pending()
                    .is_empty()
            );
        }
    }
}

struct CrashObserver {
    directory: std::path::PathBuf,
    kind: String,
    remaining: AtomicUsize,
}
impl smythe::seams::StorageObserver for CrashObserver {
    fn observe(&self, boundary: smythe::seams::StorageBoundary) -> Result<()> {
        use smythe::seams::StorageBoundary::*;
        let (kind, first, last) = match boundary {
            BeforeCommit { first, last } => ("before", first, last),
            AfterCommit { first, last } => ("after", first, last),
            BlobWritten => ("written", 0, 0),
            BlobPublished => ("published", 0, 0),
        };
        if kind == self.kind && self.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
            fs::write(
                self.directory.join("boundary"),
                serde_json::to_vec(&json!({"first":first,"last":last}))?,
            )?;
            loop {
                std::thread::park();
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "subprocess fixture invoked by disk_crashes_preserve_atomic_batches_and_blob_ordering"]
fn disk_child() {
    let directory = std::path::PathBuf::from(std::env::var_os("SMYTHE_TEST_DIRECTORY").unwrap());
    let sources = Sources {
        storage: Arc::new(CrashObserver {
            directory: directory.clone(),
            kind: std::env::var("SMYTHE_TEST_BOUNDARY").unwrap(),
            remaining: AtomicUsize::new(std::env::var("SMYTHE_TEST_CUT").unwrap().parse().unwrap()),
        }),
        ..Sources::default()
    };
    let store = Store::open_with_sources(&directory, sources.clone()).unwrap();
    Swarm::new(3, 4)
        .unwrap()
        .with_sources(sources)
        .execute_in(&graph(), &Sum::new(1, 1), &store, "disk", false)
        .unwrap();
}

#[test]
fn disk_crashes_preserve_atomic_batches_and_blob_ordering() {
    use std::process::{Command, Stdio};
    for kind in ["before", "after", "written", "published"] {
        let mut killed = 0;
        // The scheduler can coalesce different numbers of completions. Exercise
        // every possible append ordinal, plus each individual blob publication.
        let limit = if kind == "before" || kind == "after" {
            11
        } else {
            4
        };
        for cut in 1..=limit {
            let directory = Directory::new();
            fs::create_dir_all(&directory.0).unwrap();
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "disk_child", "--ignored", "--nocapture"])
                .env("SMYTHE_TEST_DIRECTORY", &directory.0)
                .env("SMYTHE_TEST_BOUNDARY", kind)
                .env("SMYTHE_TEST_CUT", cut.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            let clock = Sources::default().clock;
            let started = clock.monotonic();
            let mut marker = None;
            loop {
                if let Ok(bytes) = fs::read(directory.0.join("boundary"))
                    && let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes)
                {
                    marker = Some(value);
                    break;
                }
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    break;
                }
                if clock.monotonic().saturating_sub(started) > Duration::from_secs(15) {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("storage boundary was not reached");
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            if marker.is_some() {
                child.kill().unwrap();
                killed += 1;
            }
            child.wait().unwrap();
            let store = Store::open(&directory.0).unwrap();
            let records = store.history("disk").unwrap();
            if let Some(marker) = marker {
                let last = records.last().map_or(0, |r| r.sequence);
                if kind == "before" {
                    assert_eq!(last + 1, marker["first"].as_u64().unwrap());
                }
                if kind == "after" {
                    assert_eq!(last, marker["last"].as_u64().unwrap());
                }
            }
            let received = records
                .iter()
                .filter(|r| matches!(r.event, Event::Received { .. }))
                .count();
            let provider = Sum::new(1, 1);
            let result = Swarm::new(3, 4)
                .unwrap()
                .execute_in(&graph(), &provider, &store, "disk", !records.is_empty())
                .unwrap();
            assert_eq!(result.outputs, [1, 2, 4, 7], "{kind} cut {cut}");
            assert_eq!(result.total_cost_nanousd, 4);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 4 - received);
            assert!(store.inspect("disk").unwrap().budget.pending().is_empty());
        }
        assert!(
            killed
                >= if kind == "before" || kind == "after" {
                    7
                } else {
                    4
                }
        );
    }
}
