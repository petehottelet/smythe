#![allow(clippy::disallowed_methods)]
use anyhow::Result;
use smythe::{
    CompletionResult, Dependency, ExecutionGraph, Node, Provider, Swarm, Task, Topology,
    store::Store,
};
use smythe_core::{
    NanoUsd,
    journal::{AttemptState, Control, Event},
};
use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

fn graph() -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![Topology::ForkJoin],
        nodes: vec![
            Node {
                phase: None,
                id: "a".into(),
                task: Task {
                    goal: "a".into(),
                    input: 1.into(),
                },
                depends_on: vec![],
            },
            Node {
                phase: None,
                id: "b".into(),
                task: Task {
                    goal: "b".into(),
                    input: 2.into(),
                },
                depends_on: vec![],
            },
            Node {
                phase: None,
                id: "c".into(),
                task: Task {
                    goal: "c".into(),
                    input: 3.into(),
                },
                depends_on: vec!["a".into()],
            },
        ],
    }
}
struct Sum {
    calls: AtomicUsize,
}
impl Provider for Sum {
    type Output = u64;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        "test-sum-v1".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(1)
    }
    fn complete(&self, task: &Task, deps: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResult {
            output: task.input.as_u64().unwrap()
                + deps.iter().map(|v| v.load().unwrap()).sum::<u64>(),
            cost_nanousd: 1,
        })
    }
}
fn provider() -> Sum {
    Sum {
        calls: AtomicUsize::new(0),
    }
}

#[test]
fn ephemeral_lease_excludes_a_second_coordinator() {
    let store = Store::memory().unwrap();
    let owner = store.lease("shared").unwrap();
    assert!(store.clone().lease("shared").is_err());
    drop(owner);
    assert!(store.lease("shared").is_ok());
}

#[test]
fn rejected_atomic_admission_leaves_no_partial_reservation() {
    let store = Store::memory().unwrap();
    store
        .append(
            "atomic",
            0,
            0,
            Event::Started {
                limits: Default::default(),
                binding: "fixture".into(),
                graph: graph(),
                ceiling: NanoUsd(1),
            },
        )
        .unwrap();
    let state = store
        .append("atomic", 1, 1, Event::Claimed { generation: 1 })
        .unwrap();
    assert!(
        store
            .append_batch(
                "atomic",
                state.last,
                1,
                vec![
                    Event::Admitted {
                        effect: smythe::Effect::Pure,
                        node: "a".into(),
                        attempt: 1,
                        bound: NanoUsd(1)
                    },
                    Event::Admitted {
                        effect: smythe::Effect::Pure,
                        node: "b".into(),
                        attempt: 1,
                        bound: NanoUsd(1)
                    },
                ]
            )
            .is_err()
    );
    let restored = store.inspect("atomic").unwrap();
    assert_eq!(restored.sequence, state.last);
    assert!(restored.budget.pending().is_empty());
    assert!(restored.attempts.is_empty());
    let _valid = store
        .append_batch(
            "atomic",
            state.last,
            1,
            vec![
                Event::Admitted {
                    effect: smythe::Effect::Pure,
                    node: "a".into(),
                    attempt: 1,
                    bound: NanoUsd(1),
                },
                Event::Dispatched {
                    dependency_bytes: 0,
                    node: "a".into(),
                    attempt: 1,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        store.inspect("atomic").unwrap().attempts["a"].state,
        AttemptState::Dispatched
    );
}

#[test]
fn pause_drains_current_work_and_resume_only_dispatches_remaining_nodes() {
    struct Pausing {
        store: Store,
        calls: AtomicUsize,
    }
    impl Provider for Pausing {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "pausing-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(0)
        }
        fn complete(&self, t: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.store.control("paused", Control::Pause)?;
            }
            Ok(CompletionResult {
                output: t.input.as_u64().unwrap(),
                cost_nanousd: 0,
            })
        }
    }
    let store = Store::memory().unwrap();
    let p = Pausing {
        store: store.clone(),
        calls: AtomicUsize::new(0),
    };
    let swarm = Swarm::new(1, 0).unwrap();
    assert!(
        swarm
            .execute_in(&graph(), &p, &store, "paused", false)
            .is_err()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    let state = store.inspect("paused").unwrap();
    assert_eq!(state.control, Control::Pause);
    assert_eq!(state.attempts["a"].state, AttemptState::Accepted);
    store.control("paused", Control::Run).unwrap();
    let result = swarm
        .execute_in(&graph(), &p, &store, "paused", true)
        .unwrap();
    assert_eq!(result.reused_nodes, 1);
    assert_eq!(p.calls.load(Ordering::SeqCst), 3);
}

#[test]
fn all_local_runs_share_journal_and_resume_without_dispatch() {
    let store = Store::memory().unwrap();
    let p = provider();
    let swarm = Swarm::new(1, 3).unwrap();
    let result = swarm
        .execute_in(&graph(), &p, &store, "test", false)
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), vec![1, 2, 4]);
    assert_eq!(result.total_cost_nanousd, 3);
    let replay = swarm
        .execute_in(&graph(), &p, &store, "test", true)
        .unwrap();
    assert_eq!(replay.outputs, result.outputs);
    assert_eq!(replay.reused_nodes, 3);
    assert_eq!(p.calls.load(Ordering::SeqCst), 3);
    assert!(replay.trace.spans.is_empty());
    assert!(store.inspect("test").unwrap().completed());
    assert!(
        swarm
            .execute_in(&graph(), &p, &store, "test", false)
            .is_err()
    );
    assert!(
        Swarm::new(1, 4)
            .unwrap()
            .execute_in(&graph(), &p, &store, "test", true)
            .is_err()
    );
}

#[test]
fn ready_child_runs_while_an_unrelated_parent_is_still_running() {
    struct Straggler {
        tx: mpsc::SyncSender<()>,
        rx: Mutex<mpsc::Receiver<()>>,
    }
    impl Provider for Straggler {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "straggler-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(0)
        }
        fn complete(
            &self,
            task: &Task,
            _: &[Dependency<'_, u64>],
        ) -> Result<CompletionResult<u64>> {
            match task.goal.as_str() {
                "b" => self
                    .rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))?,
                "c" => self.tx.send(())?,
                _ => {}
            }
            Ok(CompletionResult {
                output: 0,
                cost_nanousd: 0,
            })
        }
    }
    let (tx, rx) = mpsc::sync_channel(1);
    Swarm::new(2, 0)
        .unwrap()
        .execute(
            &graph(),
            &Straggler {
                tx,
                rx: Mutex::new(rx),
            },
        )
        .unwrap();
}

#[test]
fn invalid_graphs_and_exhausted_budgets_never_dispatch() {
    for mode in 0..4 {
        let mut g = graph();
        match mode {
            0 => g.nodes[0].depends_on.push("c".into()),
            1 => g.nodes[0].depends_on.push("absent".into()),
            2 => g.nodes[1].id = "a".into(),
            _ => g.nodes[2].depends_on.push("a".into()),
        }
        let p = provider();
        assert!(Swarm::new(2, 3).unwrap().execute(&g, &p).is_err());
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }
    let p = provider();
    assert!(Swarm::new(2, 0).unwrap().execute(&graph(), &p).is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn canonical_binding_preserves_the_provider_visible_number_contract() {
    let mut g = graph();
    g.nodes.truncate(1);
    g.nodes[0].task.input = serde_json::json!(1.0);
    let p = provider();
    let store = Store::memory().unwrap();
    let swarm = Swarm::new(1, 1).unwrap();
    assert_eq!(
        swarm
            .execute_in(&g, &p, &store, "numbers", false)
            .unwrap()
            .load_all::<u64>()
            .unwrap(),
        vec![1]
    );
    g.nodes[0].task.input = serde_json::json!(1);
    assert_eq!(
        swarm
            .execute_in(&g, &p, &store, "numbers", true)
            .unwrap()
            .reused_nodes,
        1
    );
    g.nodes[0].task.input = serde_json::json!(u64::MAX);
    assert!(g.validate().is_err());
    g.nodes[0].task.input = serde_json::json!(u64::MAX.to_string());
    assert!(g.validate().is_ok());
}

#[test]
fn invalid_node_identities_are_refused_before_dispatch() {
    for id in [
        "".to_owned(),
        "__synthesis__".into(),
        "line\nbreak".into(),
        "x".repeat(257),
    ] {
        let mut g = graph();
        g.nodes.truncate(1);
        g.nodes[0].id = id;
        let p = provider();
        assert!(Swarm::new(1, 1).unwrap().execute(&g, &p).is_err());
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    }
    assert!(Swarm::new(0, 0).is_err());
    assert!(Swarm::new(65, 0).is_err());
}

#[test]
fn deep_graph_validation_is_iterative() {
    let nodes = (0..2000)
        .map(|i| Node {
            phase: None,
            id: i.to_string(),
            task: Task {
                goal: "local".into(),
                input: 0.into(),
            },
            depends_on: if i == 0 {
                vec![]
            } else {
                vec![(i - 1).to_string()]
            },
        })
        .collect();
    let mut g = ExecutionGraph {
        topology: vec![Topology::Serial],
        nodes,
    };
    assert!(g.validate().is_ok());
    g.nodes[0].depends_on.push("1999".into());
    assert!(g.validate().is_err());
}

#[test]
fn offline_outputs_reference_dependencies_without_repeating_their_histories() {
    let graph = ExecutionGraph {
        topology: vec![Topology::Serial],
        nodes: (0..40)
            .map(|i| Node {
                phase: None,
                id: i.to_string(),
                task: Task {
                    goal: "local".into(),
                    input: 0.into(),
                },
                depends_on: if i == 0 {
                    vec![]
                } else {
                    vec![(i - 1).to_string()]
                },
            })
            .collect(),
    };
    let outputs = Swarm::new(4, 0)
        .unwrap()
        .execute(&graph, &smythe::OfflineProvider)
        .unwrap()
        .load_all::<serde_json::Value>()
        .unwrap();
    assert!(
        outputs
            .iter()
            .all(|output| serde_json::to_vec(output).unwrap().len() < 200)
    );
    let expected = smythe::store::digest(&serde_json::to_vec(&outputs[38]).unwrap());
    assert_eq!(outputs[39]["dependencies"][0]["sha256"], expected);
}

#[test]
fn every_local_crash_boundary_reconstructs_the_same_outputs_and_money() {
    let mut g = graph();
    g.nodes.truncate(1);
    let complete = Store::memory().unwrap();
    Swarm::new(1, 1)
        .unwrap()
        .execute_in(&g, &provider(), &complete, "reference", false)
        .unwrap();
    let records = complete.history("reference").unwrap();
    for boundary in 1..=records.len() {
        let store = Store::memory().unwrap();
        for record in &records[..boundary] {
            if let Event::Received { output, .. } = &record.event {
                let bytes = complete.output(output.as_ref().unwrap()).unwrap();
                store.put(&bytes).unwrap();
            }
            store
                .append(
                    "replay",
                    record.sequence - 1,
                    record.generation,
                    record.event.clone(),
                )
                .unwrap();
        }
        let p = provider();
        let result = Swarm::new(1, 1)
            .unwrap()
            .execute_in(&g, &p, &store, "replay", true)
            .unwrap();
        assert_eq!(
            result.load_all::<u64>().unwrap(),
            vec![1],
            "boundary {boundary}"
        );
        assert_eq!(result.total_cost_nanousd, 1);
        let received = records[..boundary]
            .iter()
            .any(|r| matches!(r.event, Event::Received { .. }));
        assert_eq!(
            p.calls.load(Ordering::SeqCst),
            usize::from(!received),
            "boundary {boundary}"
        );
    }
}

#[test]
fn stop_is_durable_and_stale_owners_cannot_admit() {
    let store = Store::memory().unwrap();
    let mut state = store
        .append(
            "control",
            0,
            0,
            Event::Started {
                limits: Default::default(),
                binding: "fixture".into(),
                graph: graph(),
                ceiling: NanoUsd(3),
            },
        )
        .unwrap();
    store
        .append("control", state.last, 1, Event::Claimed { generation: 1 })
        .unwrap();
    state = store.control("control", Control::Stop).unwrap();
    assert!(
        store
            .append(
                "control",
                state.last,
                1,
                Event::Admitted {
                    effect: smythe::Effect::Pure,
                    node: "a".into(),
                    attempt: 1,
                    bound: NanoUsd(1)
                }
            )
            .is_err()
    );
    assert!(store.control("control", Control::Run).is_err());
    assert!(
        store
            .append(
                "control",
                state.last,
                0,
                Event::Dispatched {
                    dependency_bytes: 0,
                    node: "a".into(),
                    attempt: 1
                }
            )
            .is_err()
    );
    assert_eq!(store.inspect("control").unwrap().sequence, state.last);
}

#[test]
fn failure_keeps_successful_siblings_settled() {
    struct Broken;
    impl Provider for Broken {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "broken-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(&self, t: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
            anyhow::ensure!(t.goal != "a", "{}", ["org", "-", "fixtureprivate"].concat());
            Ok(CompletionResult {
                output: 2,
                cost_nanousd: 1,
            })
        }
    }
    let store = Store::memory().unwrap();
    let error = Swarm::new(2, 3)
        .unwrap()
        .execute_in(&graph(), &Broken, &store, "failure", false)
        .err()
        .unwrap();
    let secret = ["org", "-", "fixtureprivate"].concat();
    assert!(!format!("{error:#}").contains(&secret));
    assert!(!smythe_core::diagnostics::message(&error).contains(&secret));
    assert!(
        !serde_json::to_string(&store.history("failure").unwrap())
            .unwrap()
            .contains(&secret)
    );
    let state = store.inspect("failure").unwrap();
    assert_eq!(state.budget.spent, NanoUsd(1));
    assert!(state.budget.pending().is_empty());
    assert_eq!(state.attempts["a"].state, AttemptState::Failed);
    assert_eq!(state.attempts["b"].state, AttemptState::Accepted);
    assert!(!state.attempts.contains_key("c"));
}

#[test]
fn invalid_output_preserves_known_cost_and_never_runs_children() {
    struct Invalid;
    impl Provider for Invalid {
        type Output = f64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "invalid-output-v1".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(&self, _: &Task, _: &[Dependency<'_, f64>]) -> Result<CompletionResult<f64>> {
            Ok(CompletionResult {
                output: f64::NAN,
                cost_nanousd: 1,
            })
        }
    }
    let store = Store::memory().unwrap();
    let swarm = Swarm::new(1, 3).unwrap();
    assert!(
        swarm
            .execute_in(&graph(), &Invalid, &store, "invalid", false)
            .is_err()
    );
    let state = store.inspect("invalid").unwrap();
    assert_eq!(state.budget.spent, NanoUsd(1));
    assert!(state.budget.pending().is_empty());
    assert_eq!(state.attempts["a"].state, AttemptState::Rejected);
    assert_eq!(state.attempts.len(), 1);
    assert!(
        swarm
            .execute_in(&graph(), &Invalid, &store, "invalid", true)
            .is_err()
    );
    assert_eq!(store.inspect("invalid").unwrap().budget.spent, NanoUsd(1));
}

#[test]
fn a_later_quote_failure_drains_and_settles_an_already_dispatched_sibling() {
    struct QuoteFailure;
    impl Provider for QuoteFailure {
        type Output = u64;
        fn effect(&self) -> smythe::Effect {
            smythe::Effect::Pure
        }
        fn binding(&self) -> String {
            "quote-failure-v1".into()
        }
        fn cost_bound_nanousd(&self, t: &Task) -> Result<u64> {
            anyhow::ensure!(t.goal != "b", "invalid quote");
            Ok(1)
        }
        fn complete(&self, _: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
            Ok(CompletionResult {
                output: 1,
                cost_nanousd: 1,
            })
        }
    }
    let store = Store::memory().unwrap();
    assert!(
        Swarm::new(2, 3)
            .unwrap()
            .execute_in(&graph(), &QuoteFailure, &store, "quote", false)
            .is_err()
    );
    let state = store.inspect("quote").unwrap();
    assert_eq!(state.status(), "quote_blocked");
    assert!(state.disposition.is_none());
    assert!(state.failure_reason.is_none());
    assert_eq!(state.budget.spent, NanoUsd(1));
    assert!(state.budget.pending().is_empty());
    assert_eq!(state.attempts["a"].state, AttemptState::Accepted);
    assert!(!state.attempts.contains_key("b"));
    assert!(!state.attempts.contains_key("c"));
}
