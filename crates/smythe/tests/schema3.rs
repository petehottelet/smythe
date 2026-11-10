use anyhow::Result;
use smythe::{
    CompletionResult, Dependency, ExecutionGraph, Limits, NanoUsd, Node, Provider, Swarm, Task,
    store::Store,
};
use smythe_core::{
    EngineError,
    journal::{Admission, CloseReason, Control, Disposition},
};
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn deadline_is_observed_even_when_completions_keep_the_shell_busy() {
    use smythe::seams::{Clock, Sources};
    use std::{
        sync::{Arc, atomic::AtomicU64},
        time::Duration,
    };
    struct Manual(AtomicU64);
    impl Clock for Manual {
        fn unix_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
        fn monotonic(&self) -> Duration {
            Duration::from_millis(self.unix_ms())
        }
    }
    struct Advances(Arc<Manual>);
    impl Provider for Advances {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "advances".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(0)
        }
        fn complete(&self, _: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
            self.0.0.store(100, Ordering::SeqCst);
            Ok(CompletionResult {
                output: 1,
                cost_nanousd: 0,
            })
        }
    }
    let clock = Arc::new(Manual(AtomicU64::new(0)));
    let store = Store::memory().unwrap();
    let error = Swarm::new(1, 0)
        .unwrap()
        .with_limits(Limits {
            wall_deadline_ms: Some(100),
            ..Limits::default()
        })
        .with_sources(Sources {
            clock: clock.clone(),
            ..Sources::default()
        })
        .execute_in(&graph(2), &Advances(clock), &store, "busy", false)
        .err()
        .unwrap();
    assert!(EngineError::Stopped.matches(&error));
    let state = store.inspect("busy").unwrap();
    assert_eq!(state.accepted_nodes, 1);
    assert_eq!(state.attempts.len(), 1);
}

#[test]
fn deadline_closes_at_boundary_and_does_not_change_operator_intent() {
    use smythe::seams::{Clock, Sources};
    use std::{sync::Arc, time::Duration};
    struct Fixed(u64);
    impl Clock for Fixed {
        fn unix_ms(&self) -> u64 {
            self.0
        }
        fn monotonic(&self) -> Duration {
            Duration::ZERO
        }
    }
    for now in [99, 100] {
        let store = Store::memory().unwrap();
        let result = Swarm::new(1, 0)
            .unwrap()
            .with_limits(Limits {
                wall_deadline_ms: Some(100),
                ..Limits::default()
            })
            .with_sources(Sources {
                clock: Arc::new(Fixed(now)),
                ..Sources::default()
            })
            .execute_in(&graph(1), &priced(0, 0), &store, "deadline", false);
        let state = store.inspect("deadline").unwrap();
        assert_eq!(state.control, Control::Run);
        if now < 100 {
            assert!(result.is_ok());
        } else {
            assert!(EngineError::Stopped.matches(&result.err().unwrap()));
            assert!(state.attempts.is_empty());
            assert_eq!(
                state.admission,
                Admission::Closed {
                    reason: CloseReason::Deadline
                }
            );
        }
    }
}

#[test]
fn five_thousand_node_chain_bounds_resident_values_by_concurrency() {
    use std::sync::atomic::AtomicUsize;
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    #[derive(serde::Serialize)]
    struct Value(u64);
    impl Value {
        fn new(n: u64) -> Self {
            let live = LIVE.fetch_add(1, Ordering::SeqCst) + 1;
            PEAK.fetch_max(live, Ordering::SeqCst);
            Self(n)
        }
    }
    impl Drop for Value {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl<'de> serde::Deserialize<'de> for Value {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            Ok(Self::new(<u64 as serde::Deserialize>::deserialize(d)?))
        }
    }
    struct Counted;
    impl Provider for Counted {
        type Output = Value;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "counted".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(0)
        }
        fn complete(
            &self,
            _: &Task,
            deps: &[Dependency<'_, Value>],
        ) -> Result<CompletionResult<Value>> {
            let previous = deps.first().map(|d| d.load().unwrap());
            Ok(CompletionResult {
                output: Value::new(previous.as_ref().map_or(1, |p| p.0 + 1)),
                cost_nanousd: 0,
            })
        }
    }
    let mut g = graph(5000);
    for i in 1..5000 {
        g.nodes[i].depends_on = vec![format!("n{}", i - 1)];
    }
    let result = Swarm::new(4, 0).unwrap().execute(&g, &Counted).unwrap();
    assert_eq!(LIVE.load(Ordering::SeqCst), 0);
    assert!(PEAK.load(Ordering::SeqCst) <= 12);
    assert_eq!(result.load::<Value>("n4999").unwrap().0, 5000);
}

#[test]
fn topology_is_descriptive_and_shape_is_derived_from_dependencies() {
    let store = Store::memory().unwrap();
    let p = priced(0, 0);
    let mut g = graph(3);
    g.nodes[2].depends_on = vec!["n0".into(), "n1".into()];
    Swarm::new(2, 0)
        .unwrap()
        .execute_in(&g, &p, &store, "shape", false)
        .unwrap();
    g.topology = vec![smythe::Topology::Serial];
    assert_eq!(
        Swarm::new(2, 0)
            .unwrap()
            .execute_in(&g, &p, &store, "shape", true)
            .unwrap()
            .reused_nodes,
        3
    );
    let shape = g.shape().unwrap();
    assert_eq!(
        (
            shape.nodes,
            shape.edges,
            shape.maximum_width,
            shape.depth,
            shape.joins
        ),
        (3, 2, 2, 2, 1)
    );
}

#[test]
fn expansion_and_revision_replay_without_reusing_invalid_descendants() {
    use smythe_core::journal::{Event, Projection, call_key};
    let store = Store::memory().unwrap();
    let p = priced(1, 1);
    let g = graph(2);
    Swarm::new(2, 10)
        .unwrap()
        .execute_in(&g, &p, &store, "rev", false)
        .unwrap();
    let before = store.inspect("rev").unwrap();
    let mut added = graph(3).nodes.remove(2);
    added.depends_on = vec!["n0".into()];
    added.phase = Some("review".into());
    let expansion = Event::Expanded {
        parent_revision: 0,
        revision: 1,
        input_manifest_sha256: "a".repeat(64),
        nodes: vec![added],
    };
    store
        .append("rev", before.sequence, before.generation, expansion)
        .unwrap();
    // Simulated crash immediately after expansion: reconstruct from durable records.
    let replay = Projection::replay(&store.history("rev").unwrap()).unwrap();
    assert_eq!(replay.revision, 1);
    assert_eq!(replay.accepted_nodes, 2);
    let result = Swarm::new(2, 10)
        .unwrap()
        .execute_in(&g, &p, &store, "rev", true)
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), [1, 2, 4]);
    assert_eq!(result.reused_nodes, 2);
    let state = store.inspect("rev").unwrap();
    store
        .append(
            "rev",
            state.sequence,
            state.generation,
            Event::Revised {
                revision: 2,
                reason: "changed_input".into(),
                invalidated: vec!["n0".into()],
            },
        )
        .unwrap();
    assert_eq!(store.inspect("rev").unwrap().accepted_nodes, 1);
    let result = Swarm::new(2, 10)
        .unwrap()
        .execute_in(&g, &p, &store, "rev", true)
        .unwrap();
    assert_eq!(result.reused_nodes, 1);
    assert_eq!(result.total_cost_nanousd, 5);
    let state = store.inspect("rev").unwrap();
    assert_eq!(state.attempts["n0"].revision, 2);
    assert_eq!(state.attempts["n2"].number, 2);
    assert_eq!(state.attempts["n1"].revision, 0);
    assert_ne!(call_key(0, "n0", 1), call_key(2, "n0", 2));
    let records = store.history("rev").unwrap();
    assert!(
        records
            .iter()
            .filter(|r| r.node.as_deref() == Some("n2"))
            .all(|r| r.phase.as_deref() == Some("review"))
    );
    for bad in [
        Event::Expanded {
            parent_revision: 2,
            revision: 3,
            input_manifest_sha256: "a".repeat(64),
            nodes: vec![g.nodes[0].clone()],
        },
        Event::Revised {
            revision: 4,
            reason: "bad".into(),
            invalidated: vec!["n0".into()],
        },
        Event::Revised {
            revision: 3,
            reason: "bad".into(),
            invalidated: vec!["missing".into()],
        },
    ] {
        assert!(
            store
                .append("rev", state.sequence, state.generation, bad)
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(store.inspect("rev").unwrap()).unwrap(),
            serde_json::to_value(&state).unwrap()
        );
    }
}

#[test]
fn limits_apply_at_boundaries_and_attempt_exhaustion_refuses_retry() {
    let mut g = graph(2);
    g.nodes[1].depends_on = vec!["n0".into()];
    let limits = Limits {
        max_nodes: 2,
        max_edges: 1,
        max_output_bytes: 1,
        max_total_output_bytes: 2,
        ..Limits::default()
    };
    Swarm::new(1, 2)
        .unwrap()
        .with_limits(limits.clone())
        .execute(&g, &priced(1, 1))
        .unwrap();
    for limit in [
        Limits {
            max_nodes: 1,
            ..limits.clone()
        },
        Limits {
            max_edges: 0,
            ..limits.clone()
        },
    ] {
        let store = Store::memory().unwrap();
        assert!(
            Swarm::new(1, 2)
                .unwrap()
                .with_limits(limit)
                .execute_in(&g, &priced(1, 1), &store, "invalid", false)
                .is_err()
        );
        assert!(store.history("invalid").unwrap().is_empty());
    }
    let store = Store::memory().unwrap();
    assert!(
        Swarm::new(1, 2)
            .unwrap()
            .with_limits(Limits {
                max_total_output_bytes: 1,
                ..limits
            })
            .execute_in(&g, &priced(1, 1), &store, "total", false)
            .is_err()
    );
    let state = store.inspect("total").unwrap();
    assert_eq!(state.output_bytes, 1);
    assert_eq!(state.budget.spent, NanoUsd(2));
    let p = priced(0, 0);
    p.fail.store(true, Ordering::SeqCst);
    let swarm = Swarm::new(1, 0).unwrap().with_limits(Limits {
        max_attempts_per_node: 2,
        ..Limits::default()
    });
    assert!(
        swarm
            .execute_in(&graph(1), &p, &store, "attempts", false)
            .is_err()
    );
    store.retry("attempts", "n0").unwrap();
    assert!(
        swarm
            .execute_in(&graph(1), &p, &store, "attempts", true)
            .is_err()
    );
    assert!(store.retry("attempts", "n0").is_err());
}

#[test]
fn reference_only_provider_loads_no_dependency_values_and_cursors_split_transactions() {
    use smythe_core::journal::Projection;
    let mut g = graph(100);
    for i in 1..100 {
        g.nodes[i].depends_on = vec![format!("n{}", i - 1)];
    }
    let store = Store::memory().unwrap();
    let result = Swarm::new(3, 0)
        .unwrap()
        .execute_in(&g, &smythe::OfflineProvider, &store, "refs", false)
        .unwrap();
    assert_eq!(result.loaded_dependency_bytes, 0);
    assert!(result.dependency_bytes > 0);
    let history = store.history("refs").unwrap();
    let mut pages = vec![];
    let mut cursor = 0;
    loop {
        let page = store.events_after("refs", cursor, 2).unwrap();
        if page.is_empty() {
            break;
        }
        cursor = page.last().unwrap().sequence;
        pages.extend(page);
    }
    assert_eq!(
        serde_json::to_value(&history).unwrap(),
        serde_json::to_value(&pages).unwrap()
    );
    assert_eq!(
        Projection::replay(&pages).unwrap().dependency_bytes,
        result.dependency_bytes
    );
    assert_eq!(store.list().unwrap()[0].status, "completed");
}

#[test]
fn failing_commit_rolls_back_inline_blob_and_every_projection_change() {
    use smythe::seams::{Sources, StorageBoundary, StorageObserver};
    use smythe_core::journal::{Effect, Event, Projection};
    use std::sync::Arc;
    struct RejectCommit(AtomicBool);
    impl StorageObserver for RejectCommit {
        fn observe(&self, b: StorageBoundary) -> Result<()> {
            anyhow::ensure!(
                !(matches!(b, StorageBoundary::BeforeCommit { .. })
                    && self.0.load(Ordering::SeqCst)),
                "fixture commit failure"
            );
            Ok(())
        }
    }
    let observer = Arc::new(RejectCommit(AtomicBool::new(false)));
    let store = Store::memory_with_sources(Sources {
        storage: observer.clone(),
        ..Sources::default()
    })
    .unwrap();
    store
        .append(
            "atomic",
            0,
            0,
            Event::Started {
                binding: "fixture".into(),
                graph: graph(1),
                ceiling: NanoUsd(2),
                limits: Limits::default(),
            },
        )
        .unwrap();
    store
        .append_batch(
            "atomic",
            1,
            1,
            vec![
                Event::Claimed { generation: 1 },
                Event::Admitted {
                    node: "n0".into(),
                    attempt: 1,
                    bound: NanoUsd(2),
                    effect: Effect::Pure,
                },
                Event::Dispatched {
                    node: "n0".into(),
                    attempt: 1,
                    dependency_bytes: 0,
                },
            ],
        )
        .unwrap();
    let before = store.inspect("atomic").unwrap();
    let (reference, blob) = store.prepare(b"1".to_vec()).unwrap();
    assert!(store.get(&reference.sha256).is_err());
    let event = Event::Received {
        node: "n0".into(),
        attempt: 1,
        output: Some(reference.clone()),
        output_valid: true,
        cost: NanoUsd(2),
        elapsed_s: 0.0,
    };
    observer.0.store(true, Ordering::SeqCst);
    assert!(
        store
            .append_observed_batch(
                "atomic",
                4,
                1,
                vec![(event.clone(), 0)],
                vec![blob.clone().unwrap()]
            )
            .is_err()
    );
    assert!(store.get(&reference.sha256).is_err());
    assert_eq!(
        serde_json::to_value(store.inspect("atomic").unwrap()).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
    let mut direct = Projection::replay(&store.history("atomic").unwrap()).unwrap();
    let records =
        smythe::store::make_records("atomic", 4, 1, Some(&before), vec![(event.clone(), 0)], 0)
            .unwrap();
    assert!(
        direct
            .transact::<()>(&records, || anyhow::bail!("fixture SQL failure"))
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(direct).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
    observer.0.store(false, Ordering::SeqCst);
    store
        .append_observed_batch("atomic", 4, 1, vec![(event, 0)], vec![blob.unwrap()])
        .unwrap();
    assert_eq!(store.output(&reference).unwrap(), b"1");
    assert!(store.change_ceiling("atomic", NanoUsd(1)).is_err());
}

#[test]
fn external_rejected_attempt_is_never_eligible_for_retry() {
    use smythe_core::journal::{Effect, Event};
    let store = Store::memory().unwrap();
    store
        .append(
            "external",
            0,
            0,
            Event::Started {
                binding: "fixture".into(),
                graph: graph(1),
                ceiling: NanoUsd(1),
                limits: Limits::default(),
            },
        )
        .unwrap();
    store
        .append_batch(
            "external",
            1,
            1,
            vec![
                Event::Claimed { generation: 1 },
                Event::Admitted {
                    node: "n0".into(),
                    attempt: 1,
                    bound: NanoUsd(1),
                    effect: Effect::External,
                },
                Event::Dispatched {
                    node: "n0".into(),
                    attempt: 1,
                    dependency_bytes: 0,
                },
                Event::Received {
                    node: "n0".into(),
                    attempt: 1,
                    output: None,
                    output_valid: false,
                    cost: NanoUsd(1),
                    elapsed_s: 0.0,
                },
                Event::Settled {
                    node: "n0".into(),
                    attempt: 1,
                },
                Event::Rejected {
                    node: "n0".into(),
                    attempt: 1,
                    reason: "fixture".into(),
                },
            ],
        )
        .unwrap();
    assert!(EngineError::UnsupportedEffect.matches(&store.retry("external", "n0").unwrap_err()));
    assert_eq!(store.inspect("external").unwrap().budget.spent, NanoUsd(1));
}

#[test]
fn deterministic_watch_pauses_blocked_provider_and_reports_owner_health() {
    use smythe::seams::{ControlWatch, Sources};
    use std::sync::{Arc, Mutex, mpsc};
    struct Watch {
        released: Mutex<mpsc::Sender<()>>,
        called: AtomicBool,
    }
    impl ControlWatch for Watch {
        fn poll(
            &self,
            store: &Store,
            run: &str,
            after: u64,
            _: &mpsc::Receiver<()>,
        ) -> Result<Vec<smythe_core::journal::Record>> {
            if !self.called.swap(true, Ordering::SeqCst) {
                let health = store.health(&store.inspect(run)?)?;
                assert_eq!(health.owner, "live");
                assert_eq!(health.unresolved_reservations, 1);
                store.control(run, Control::Pause)?;
                self.released.lock().unwrap().send(())?;
            }
            store.changes_since(run, after)
        }
    }
    struct Blocked(Mutex<mpsc::Receiver<()>>);
    impl Provider for Blocked {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "blocked".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(&self, _: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
            self.0.lock().unwrap().recv()?;
            Ok(CompletionResult {
                output: 1,
                cost_nanousd: 1,
            })
        }
    }
    let (tx, rx) = mpsc::channel();
    let watch = Arc::new(Watch {
        released: Mutex::new(tx),
        called: AtomicBool::new(false),
    });
    let store = Store::memory().unwrap();
    let error = Swarm::new(1, 2)
        .unwrap()
        .with_sources(Sources {
            controls: watch.clone(),
            ..Sources::default()
        })
        .execute_in(&graph(2), &Blocked(Mutex::new(rx)), &store, "watch", false)
        .err()
        .unwrap();
    assert!(EngineError::Paused.matches(&error));
    assert!(watch.called.load(Ordering::SeqCst));
    let state = store.inspect("watch").unwrap();
    let health = store.health(&state).unwrap();
    assert_eq!(health.owner, "absent");
    assert_eq!(health.accepted_nodes, 1);
    assert_eq!(health.unresolved_reservations, 0);
}

fn graph(n: usize) -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: (0..n)
            .map(|i| Node {
                id: format!("n{i}"),
                task: Task {
                    goal: "sum".into(),
                    input: (i as u64 + 1).into(),
                },
                depends_on: vec![],
                phase: Some("execution".into()),
            })
            .collect(),
    }
}
struct Priced {
    fail: AtomicBool,
    bound: u64,
    cost: u64,
}
impl Provider for Priced {
    type Output = u64;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        "priced-fixture".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(self.bound)
    }
    fn complete(&self, t: &Task, deps: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
        anyhow::ensure!(!self.fail.load(Ordering::SeqCst), "fixture failure");
        Ok(CompletionResult {
            output: t.input.as_u64().unwrap() + deps.iter().map(|d| d.load().unwrap()).sum::<u64>(),
            cost_nanousd: self.cost,
        })
    }
}
fn priced(bound: u64, cost: u64) -> Priced {
    Priced {
        fail: AtomicBool::new(false),
        bound,
        cost,
    }
}

#[test]
fn final_overrun_completes_without_changing_operator_intent() {
    let store = Store::memory().unwrap();
    let result = Swarm::new(1, 1)
        .unwrap()
        .execute_in(&graph(1), &priced(1, 2), &store, "overrun", false)
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), [1]);
    assert!(result.overrun);
    let state = store.inspect("overrun").unwrap();
    assert_eq!(state.disposition, Some(Disposition::Completed));
    assert_eq!(state.control, Control::Run);
    assert_eq!(
        state.admission,
        Admission::Closed {
            reason: CloseReason::Overrun
        }
    );
    assert_eq!(state.budget.spent, NanoUsd(2));
}

#[test]
fn budget_can_be_raised_without_changing_binding_or_prior_settlements() {
    let store = Store::memory().unwrap();
    let p = priced(1, 1);
    let g = graph(2);
    let error = Swarm::new(1, 1)
        .unwrap()
        .execute_in(&g, &p, &store, "budget", false)
        .err()
        .unwrap();
    assert!(EngineError::BudgetExhausted.matches(&error));
    let before = store.inspect("budget").unwrap();
    assert!(before.budget_blocked);
    assert_eq!(before.budget.spent, NanoUsd(1));
    assert_eq!(before.disposition, None);
    store.change_ceiling("budget", NanoUsd(2)).unwrap();
    let result = Swarm::new(1, 2)
        .unwrap()
        .execute_in(&g, &p, &store, "budget", true)
        .unwrap();
    assert_eq!(result.reused_nodes, 1);
    assert_eq!(result.total_cost_nanousd, 2);
    assert_eq!(store.inspect("budget").unwrap().binding, before.binding);
}

#[test]
fn pure_failure_requires_an_explicit_bounded_retry() {
    let store = Store::memory().unwrap();
    let p = priced(1, 1);
    let g = graph(1);
    p.fail.store(true, Ordering::SeqCst);
    assert!(
        Swarm::new(1, 3)
            .unwrap()
            .execute_in(&g, &p, &store, "retry", false)
            .is_err()
    );
    assert_eq!(
        store.inspect("retry").unwrap().disposition,
        Some(Disposition::Failed)
    );
    p.fail.store(false, Ordering::SeqCst);
    assert!(
        Swarm::new(1, 3)
            .unwrap()
            .execute_in(&g, &p, &store, "retry", true)
            .is_err()
    );
    store.retry("retry", "n0").unwrap();
    let result = Swarm::new(1, 3)
        .unwrap()
        .execute_in(&g, &p, &store, "retry", true)
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), [1]);
    assert_eq!(store.inspect("retry").unwrap().attempts["n0"].number, 2);
}

#[test]
fn handles_measure_dependency_reads_and_output_limits_retain_cost() {
    let mut g = graph(2);
    g.nodes[1].depends_on = vec!["n0".into()];
    let result = Swarm::new(2, 2)
        .unwrap()
        .execute(&g, &priced(1, 1))
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), [1, 3]);
    assert_eq!(result.dependency_bytes, 1);
    assert_eq!(result.loaded_dependency_bytes, 1);
    let store = Store::memory().unwrap();
    assert!(
        Swarm::new(1, 1)
            .unwrap()
            .with_limits(Limits {
                max_output_bytes: 0,
                ..Limits::default()
            })
            .execute_in(&graph(1), &priced(1, 1), &store, "limit", false)
            .is_err()
    );
    let state = store.inspect("limit").unwrap();
    assert_eq!(state.budget.spent, NanoUsd(1));
    assert_eq!(
        state.attempts["n0"].reason.as_deref(),
        Some("output_too_large")
    );
    assert!(state.budget.pending().is_empty());
}
