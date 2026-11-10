//! Pure scheduler model: no threads, clocks, filesystem, provider or SQLite.
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use smythe_core::{
    ExecutionGraph, Limits, NanoUsd, Node, OutputRef, Task,
    coordinator::{Command, Completion, Coordinator, Finish, Input, Outcome},
    journal::{AttemptState, Control, Event, Projection, Record, call_key},
};
use std::collections::{HashSet, VecDeque};

fn graph() -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: (0..6)
            .map(|i| Node {
                id: i.to_string(),
                phase: Some("simulation".into()),
                task: Task {
                    goal: "fixture".into(),
                    input: i.into(),
                },
                depends_on: match i {
                    2 => vec!["0".into()],
                    3 => vec!["0".into(), "1".into()],
                    5 => vec!["2".into(), "3".into(), "4".into()],
                    _ => vec![],
                },
            })
            .collect(),
    }
}
fn records(state: Option<&Projection>, events: Vec<(Event, u64)>) -> Vec<Record> {
    let first = state.map_or(1, |s| s.sequence + 1);
    events
        .into_iter()
        .enumerate()
        .map(|(i, (event, time))| {
            let phase = event.identity().0.map(|_| "simulation".into());
            Record::new(
                "sim",
                first + i as u64,
                state.map_or(0, |s| s.generation),
                first,
                event,
                time,
                time.saturating_sub(5),
            )
            .in_revision(0, phase)
        })
        .collect()
}
fn assert_invariants(state: &Projection, journal: &[Record]) {
    let replay = Projection::replay(journal).unwrap();
    assert_eq!(
        serde_json::to_value(state).unwrap(),
        serde_json::to_value(replay).unwrap()
    );
    assert_eq!(
        state.accepted_nodes,
        state
            .attempts
            .values()
            .filter(|a| a.state == AttemptState::Accepted)
            .count()
    );
    let mut expected = 0;
    for r in journal {
        if let Event::Settled { node, attempt } = &r.event {
            expected += journal
                .iter()
                .find_map(|prior| match &prior.event {
                    Event::Received {
                        node: n,
                        attempt: a,
                        cost,
                        ..
                    } if n == node && a == attempt => Some(cost.0),
                    _ => None,
                })
                .unwrap();
        }
    }
    assert_eq!(state.budget.spent.0, expected);
    for (node, a) in &state.attempts {
        if matches!(
            a.state,
            AttemptState::Admitted | AttemptState::Dispatched | AttemptState::Received
        ) {
            assert!(
                state
                    .budget
                    .pending()
                    .contains_key(&call_key(a.revision, node, a.number))
            );
        }
    }
    if !state.overrun {
        assert!(state.budget.exposure().unwrap() <= state.budget.ceiling);
    }
    if state.disposition.is_some() {
        assert!(state.budget.pending().is_empty());
    }
}
fn dispatch_is_committed(state: &Projection, node: usize, attempt: u64) -> bool {
    state
        .attempts
        .get(&state.graph.nodes[node].id)
        .is_some_and(|a| {
            a.number == attempt
                && a.state == AttemptState::Dispatched
                && state.budget.pending().contains_key(&call_key(
                    a.revision,
                    &state.graph.nodes[node].id,
                    attempt,
                ))
                && state.graph.nodes[node]
                    .depends_on
                    .iter()
                    .all(|p| state.attempts[p].state == AttemptState::Accepted)
        })
}
fn simulate(seed: u64) -> Result<(), String> {
    let result = std::panic::catch_unwind(|| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let uninterrupted_equivalent = seed.is_multiple_of(5);
        let mut journal = records(
            None,
            vec![(
                Event::Started {
                    binding: "sim".into(),
                    graph: graph(),
                    ceiling: NanoUsd(12),
                    limits: Limits {
                        max_attempts_per_node: 16,
                        wall_deadline_ms: if !uninterrupted_equivalent && seed.is_multiple_of(7) {
                            Some(1200)
                        } else {
                            None
                        },
                        ..Limits::default()
                    },
                },
                1000,
            )],
        );
        journal.push(Record::new(
            "sim",
            2,
            1,
            2,
            Event::Claimed { generation: 1 },
            1000,
            1000,
        ));
        let mut committed = Projection::replay(&journal).unwrap();
        let mut coordinator =
            Coordinator::new(committed.clone(), vec![Some(NanoUsd(2)); 6], 3, 1000).unwrap();
        let mut commands: VecDeque<_> = coordinator
            .step(Input::Recovered {
                invalid_outputs: HashSet::new(),
            })
            .unwrap()
            .into();
        let mut active = Vec::new();
        let mut crashes = 0;
        let mut time = 1000;
        for step in 0..500 {
            if let Some(command) = commands.pop_front() {
                match command {
                    Command::Append { events, .. } => {
                        // Sometimes a control wins the compare-and-append race.
                        if !uninterrupted_equivalent
                            && rng.next_u32() % 17 == 0
                            && committed.disposition.is_none()
                            && committed.control != Control::Stop
                        {
                            let control = if rng.next_u32() % 3 == 0 {
                                Control::Stop
                            } else {
                                Control::Pause
                            };
                            let changes =
                                records(Some(&committed), vec![(Event::Control { control }, time)]);
                            committed.apply_batch(&changes).unwrap();
                            journal.extend(changes.clone());
                            commands.extend(coordinator.step(Input::Foreign(changes)).unwrap());
                            if coordinator.needs_recovery() {
                                commands.extend(
                                    coordinator
                                        .step(Input::Recovered {
                                            invalid_outputs: HashSet::new(),
                                        })
                                        .unwrap(),
                                );
                            }
                        } else {
                            let changes = records(Some(&committed), events);
                            committed.apply_batch(&changes).unwrap();
                            journal.extend(changes.clone());
                            if rng.next_u32() % 13 == 0
                                && crashes < 3
                                && committed.disposition.is_none()
                            {
                                // Commit survived; its acknowledgement and all in-flight calls did not.
                                crashes += 1;
                                active.clear();
                                commands.clear();
                                let generation = committed.generation + 1;
                                let claim = Record::new(
                                    "sim",
                                    committed.sequence + 1,
                                    generation,
                                    committed.sequence + 1,
                                    Event::Claimed { generation },
                                    time,
                                    time,
                                );
                                committed.apply(&claim).unwrap();
                                journal.push(claim);
                                coordinator = Coordinator::new(
                                    Projection::replay(&journal).unwrap(),
                                    vec![Some(NanoUsd(2)); 6],
                                    3,
                                    time,
                                )
                                .unwrap();
                                commands.extend(
                                    coordinator
                                        .step(Input::Recovered {
                                            invalid_outputs: HashSet::new(),
                                        })
                                        .unwrap(),
                                );
                            } else {
                                commands
                                    .extend(coordinator.step(Input::Committed(changes)).unwrap());
                            }
                        }
                        assert_invariants(&committed, &journal);
                    }
                    Command::Dispatch { node, attempt, .. } => {
                        assert!(dispatch_is_committed(&committed, node, attempt));
                        assert!(!active.iter().any(|&(n, _)| n == node));
                        active.push((node, attempt));
                    }
                    Command::Finish(Finish::Paused) => {
                        let changes = records(
                            Some(&committed),
                            vec![(
                                Event::Control {
                                    control: Control::Run,
                                },
                                time,
                            )],
                        );
                        committed.apply_batch(&changes).unwrap();
                        journal.extend(changes.clone());
                        commands.extend(coordinator.step(Input::Foreign(changes)).unwrap());
                    }
                    Command::Finish(Finish::BudgetBlocked) => {
                        let changes = records(
                            Some(&committed),
                            vec![(
                                Event::CeilingChanged {
                                    ceiling: NanoUsd(100),
                                },
                                time,
                            )],
                        );
                        committed.apply_batch(&changes).unwrap();
                        journal.extend(changes.clone());
                        commands.extend(coordinator.step(Input::Foreign(changes)).unwrap());
                    }
                    Command::Finish(_) => {
                        assert_invariants(&committed, &journal);
                        if uninterrupted_equivalent {
                            assert!(committed.completed());
                            assert_eq!(committed.budget.spent, NanoUsd(6));
                            for node in 0..6 {
                                assert_eq!(
                                    committed.attempts[&node.to_string()]
                                        .output
                                        .as_ref()
                                        .unwrap()
                                        .sha256,
                                    format!("{node:064x}")
                                );
                            }
                        }
                        return;
                    }
                }
            } else if !active.is_empty() && rng.next_u32() % 3 != 0 {
                let at = rng.next_u32() as usize % active.len();
                let (node, attempt) = active.swap_remove(at);
                let outcome = match if uninterrupted_equivalent {
                    3
                } else {
                    rng.next_u32() % 23
                } {
                    0 => Outcome::Failed {
                        reason: "fixture".into(),
                    },
                    1 => Outcome::Rejected {
                        cost: NanoUsd(1),
                        elapsed_s: 0.0,
                        reason: "fixture".into(),
                    },
                    n => Outcome::Success {
                        output: OutputRef {
                            sha256: format!("{node:064x}"),
                            bytes: 1,
                        },
                        inline: None,
                        cost: NanoUsd(if n == 2 { 3 } else { 1 }),
                        elapsed_s: 0.0,
                    },
                };
                commands.extend(
                    coordinator
                        .step(Input::Completions(vec![Completion {
                            node,
                            attempt,
                            outcome,
                            occurred_at: time,
                        }]))
                        .unwrap(),
                );
            } else {
                // Both clock regressions and advancement occur; admission cannot reopen.
                time = if rng.next_u32() % 4 == 0 {
                    time.saturating_sub(20)
                } else {
                    time + 10
                };
                commands.extend(coordinator.step(Input::Tick { now_ms: time }).unwrap());
            }
            assert!(step < 499, "scheduler failed to terminate");
        }
    });
    result.map_err(|_| format!("ChaCha8 seed {seed}; replay this exact seed"))
}
#[test]
fn five_thousand_seeded_schedules_replay_committed_prefixes() {
    for seed in 0..5000 {
        simulate(seed).unwrap();
    }
}
#[test]
fn recorded_counterexample_detects_dispatch_before_commit() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dispatch-before-commit.json")).unwrap();
    let seed = fixture["seed"].as_u64().unwrap();
    simulate(seed).unwrap();
    let state = Projection::replay(&records(
        None,
        vec![(
            Event::Started {
                binding: "sim".into(),
                graph: graph(),
                ceiling: NanoUsd(12),
                limits: Limits::default(),
            },
            0,
        )],
    ))
    .unwrap();
    // Intentional mutant: emitting Dispatch with no durable admission must fail the oracle.
    assert!(!dispatch_is_committed(&state, 0, 1));
}
