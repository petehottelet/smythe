use anyhow::Result;
use smythe::{
    CompletionResult, Dependency, ExecutionGraph, NanoUsd, Node, Provider, Swarm, Task,
    store::Store,
};
use smythe_core::{
    EngineError,
    journal::{AttemptState, Control, Event, Projection, call_key},
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn graph(nodes: usize) -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: (0..nodes)
            .map(|i| Node {
                id: format!("n{i}"),
                phase: Some("test".into()),
                task: Task {
                    goal: format!("n{i}"),
                    input: i.into(),
                },
                depends_on: vec![],
            })
            .collect(),
    }
}
struct Local {
    bad_quote: AtomicBool,
    fail: AtomicBool,
    calls: AtomicUsize,
}
impl Local {
    fn new() -> Self {
        Self {
            bad_quote: AtomicBool::new(false),
            fail: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        }
    }
}
impl Provider for Local {
    type Output = u64;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        "amendment-fixture".into()
    }
    fn cost_bound_nanousd(&self, task: &Task) -> Result<u64> {
        anyhow::ensure!(
            task.goal != "n1" || !self.bad_quote.load(Ordering::SeqCst),
            "fixture private diagnostic"
        );
        Ok(2)
    }
    fn complete(&self, _: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        anyhow::ensure!(!self.fail.load(Ordering::SeqCst), "fixture failure");
        Ok(CompletionResult {
            output: 7,
            cost_nanousd: 1,
        })
    }
}
fn append(store: &Store, id: &str, event: Event) {
    let state = store.inspect(id).unwrap();
    store
        .append(id, state.sequence, state.generation, event)
        .unwrap();
}
fn revised(revision: u32) -> Event {
    Event::Revised {
        revision,
        reason: "fixture".into(),
        invalidated: vec!["n0".into()],
    }
}

#[test]
fn quote_block_resumes_without_redispatching_or_changing_prior_settlements() {
    for siblings in [false, true] {
        let mut g = graph(2);
        if !siblings {
            g.nodes.remove(0);
        }
        let store = Store::memory().unwrap();
        let p = Local::new();
        p.bad_quote.store(true, Ordering::SeqCst);
        let swarm = Swarm::new(2, 10).unwrap();
        let error = swarm
            .execute_in(&g, &p, &store, "quote", false)
            .err()
            .unwrap();
        assert!(EngineError::QuoteBlocked.matches(&error));
        assert_eq!(EngineError::QuoteBlocked.class(), "quote_blocked");
        assert_eq!(EngineError::QuoteBlocked.exit_code(), 1);
        assert!(
            error
                .to_string()
                .contains("correct the provider and resume")
        );
        let before = store.inspect("quote").unwrap();
        assert_eq!(before.status(), "quote_blocked");
        assert!(before.disposition.is_none() && before.failure_reason.is_none());
        assert_eq!(before.budget.spent, NanoUsd(u64::from(siblings)));
        assert!(before.budget.pending().is_empty());
        let spent = serde_json::to_value(&before.budget).unwrap();
        for _ in 0..2 {
            let sequence = store.inspect("quote").unwrap().sequence;
            let error = swarm
                .execute_in(&g, &p, &store, "quote", true)
                .err()
                .unwrap();
            assert!(EngineError::QuoteBlocked.matches(&error));
            let events = store.events_after("quote", sequence, 100).unwrap();
            assert_eq!(events.len(), 1);
            assert!(matches!(events[0].event, Event::Claimed { .. }));
            assert_eq!(
                serde_json::to_value(store.inspect("quote").unwrap().budget).unwrap(),
                spent
            );
        }
        p.bad_quote.store(false, Ordering::SeqCst);
        let result = swarm.execute_in(&g, &p, &store, "quote", true).unwrap();
        assert_eq!(result.run_id, "quote");
        assert_eq!(result.reused_nodes, usize::from(siblings));
        assert_eq!(result.trace.spans.len(), 1);
        assert_eq!(p.calls.load(Ordering::SeqCst), g.nodes.len());
        let state = store.inspect("quote").unwrap();
        assert!(state.completed() && !state.quote_blocked);
        if siblings {
            assert_eq!(
                serde_json::to_value(&state.attempts["n0"]).unwrap(),
                serde_json::to_value(&before.attempts["n0"]).unwrap()
            );
        }
        let history = store.history("quote").unwrap();
        assert_eq!(
            history
                .iter()
                .filter(|r| matches!(r.event, Event::QuoteBlocked { .. }))
                .count(),
            1
        );
        assert!(
            !serde_json::to_string(&history)
                .unwrap()
                .contains("private diagnostic")
        );
        assert!(
            !history
                .iter()
                .any(|r| matches!(r.event, Event::RunFailed { .. }))
        );
    }
}

#[test]
fn five_successful_revisions_keep_monotonic_identities_and_fresh_retry_limits() {
    let store = Store::memory().unwrap();
    let p = Local::new();
    let g = graph(1);
    let swarm = Swarm::new(1, 20).unwrap();
    swarm
        .execute_in(&g, &p, &store, "revisions", false)
        .unwrap();
    for revision in 1..=5 {
        append(&store, "revisions", revised(revision));
        let invalidated = store.inspect("revisions").unwrap();
        assert_eq!(
            invalidated.attempts["n0"].revision_base,
            u64::from(revision)
        );
        let result = swarm.execute_in(&g, &p, &store, "revisions", true).unwrap();
        assert_eq!(result.trace.spans.len(), 1);
        let state = store.inspect("revisions").unwrap();
        let a = &state.attempts["n0"];
        assert_eq!(a.number, u64::from(revision) + 1);
        assert_eq!(a.number - a.revision_base, 1);
        assert_eq!(a.revision, revision);
        assert_eq!(state.budget.spent, NanoUsd(a.number));
        assert!(state.failure_reason.is_none());
        let replay = Projection::replay(&store.history("revisions").unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            serde_json::to_value(replay).unwrap()
        );
    }
    let history = store.history("revisions").unwrap();
    let keys: std::collections::HashSet<_> = history
        .iter()
        .filter_map(|r| match &r.event {
            Event::Admitted { node, attempt, .. } => Some(call_key(r.revision, node, *attempt)),
            _ => None,
        })
        .collect();
    assert_eq!(keys.len(), 6);
}

#[test]
fn retries_after_revision_still_stop_at_three_failures() {
    let store = Store::memory().unwrap();
    let p = Local::new();
    let g = graph(1);
    let swarm = Swarm::new(1, 10).unwrap();
    swarm.execute_in(&g, &p, &store, "retry", false).unwrap();
    append(&store, "retry", revised(1));
    p.fail.store(true, Ordering::SeqCst);
    for used in 1..=3 {
        let error = swarm
            .execute_in(&g, &p, &store, "retry", true)
            .err()
            .unwrap();
        assert!(EngineError::ProviderFailed.matches(&error));
        let state = store.inspect("retry").unwrap();
        assert_eq!(state.attempts["n0"].number, used + 1);
        assert_eq!(state.attempts["n0"].revision_base, 1);
        assert!(state.failure_reason.is_none());
        let retry = store.retry("retry", "n0");
        assert_eq!(retry.is_ok(), used < 3);
    }
    assert_eq!(store.inspect("retry").unwrap().budget.spent, NanoUsd(1));
}

#[test]
fn quote_blocks_validate_admission_and_rollback_all_projection_changes() {
    let store = Store::memory().unwrap();
    let p = Local::new();
    p.bad_quote.store(true, Ordering::SeqCst);
    let g = graph(2);
    assert!(
        Swarm::new(1, 10)
            .unwrap()
            .execute_in(&g, &p, &store, "atomic", false)
            .is_err()
    );
    let blocked = store.inspect("atomic").unwrap();
    assert!(blocked.quote_blocked);
    // Every transition that clears a stale block must restore it if commit fails.
    for event in [
        Event::BudgetBlocked {
            node: "n1".into(),
            bound: NanoUsd(20),
        },
        Event::Admitted {
            node: "n1".into(),
            attempt: 1,
            bound: NanoUsd(2),
            effect: smythe::Effect::Pure,
        },
        Event::Expanded {
            parent_revision: 0,
            revision: 1,
            input_manifest_sha256: "0".repeat(64),
            nodes: vec![Node {
                id: "n2".into(),
                phase: None,
                task: Task {
                    goal: "fixture".into(),
                    input: 2.into(),
                },
                depends_on: vec!["n0".into()],
            }],
        },
        revised(1),
    ] {
        let records = smythe::store::make_records(
            "atomic",
            blocked.sequence,
            blocked.generation,
            Some(&blocked),
            vec![(event, 1000)],
            1000,
        )
        .unwrap();
        let mut projection = blocked.clone();
        assert!(
            projection
                .transact::<()>(&records, || anyhow::bail!("fixture commit failure"))
                .is_err()
        );
        assert_eq!(
            serde_json::to_value(&projection).unwrap(),
            serde_json::to_value(&blocked).unwrap()
        );
        projection.apply_batch(&records).unwrap();
        assert!(!projection.quote_blocked);
        assert_eq!(projection.budget.spent, blocked.budget.spent);
    }
    let records = smythe::store::make_records(
        "atomic",
        blocked.sequence,
        blocked.generation,
        Some(&blocked),
        vec![(
            Event::CeilingChanged {
                ceiling: NanoUsd(20),
            },
            1000,
        )],
        1000,
    )
    .unwrap();
    let mut projection = blocked.clone();
    assert!(
        projection
            .transact::<()>(&records, || anyhow::bail!("fixture commit failure"))
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(&projection).unwrap(),
        serde_json::to_value(&blocked).unwrap()
    );
    let changes = smythe::store::make_records(
        "atomic",
        blocked.sequence,
        blocked.generation,
        Some(&blocked),
        vec![(revised(1), 1000)],
        1000,
    )
    .unwrap();
    assert!(
        projection
            .transact::<()>(&changes, || anyhow::bail!("fixture commit failure"))
            .is_err()
    );
    assert_eq!(projection.attempts["n0"].state, AttemptState::Accepted);
    assert_eq!(projection.attempts["n0"].revision_base, 0);
    assert_eq!(
        serde_json::to_value(&projection).unwrap(),
        serde_json::to_value(&blocked).unwrap()
    );
    store.change_ceiling("atomic", NanoUsd(20)).unwrap();
    assert!(!store.inspect("atomic").unwrap().quote_blocked);
    let state = store.inspect("atomic").unwrap();
    assert!(
        store
            .append_batch(
                "atomic",
                state.sequence,
                state.generation,
                vec![
                    Event::QuoteBlocked { node: "n1".into() },
                    Event::QuoteBlocked {
                        node: "missing".into()
                    },
                ]
            )
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(store.inspect("atomic").unwrap()).unwrap(),
        serde_json::to_value(&state).unwrap()
    );
    store.control("atomic", Control::Pause).unwrap();
    let paused = store.inspect("atomic").unwrap();
    assert!(
        store
            .append(
                "atomic",
                paused.sequence,
                paused.generation,
                Event::QuoteBlocked { node: "n1".into() }
            )
            .is_err()
    );
    store.control("atomic", Control::Run).unwrap();
    append(
        &store,
        "atomic",
        Event::Admitted {
            node: "n1".into(),
            attempt: 1,
            bound: NanoUsd(2),
            effect: smythe::Effect::Pure,
        },
    );
    let active = store.inspect("atomic").unwrap();
    assert!(
        store
            .append(
                "atomic",
                active.sequence,
                active.generation,
                Event::QuoteBlocked { node: "n1".into() }
            )
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(store.inspect("atomic").unwrap()).unwrap(),
        serde_json::to_value(active).unwrap()
    );
}
