//! Pure scheduler model: no threads, clocks, filesystem, provider or SQLite.
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use smythe_core::{
    ExecutionGraph, Limits, NanoUsd, Node, OutputRef, Task,
    coordinator::{Command, Completion, Coordinator, Finish, Input, Outcome},
    journal::{Admission, AttemptState, Control, Event, Projection, Record, call_key},
};
use std::collections::{HashMap, HashSet, VecDeque};

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
    let mut revision = state.map_or(0, |s| s.revision);
    let mut generation = state.map_or(0, |s| s.generation);
    events
        .into_iter()
        .enumerate()
        .map(|(i, (event, time))| {
            revision = event.revision(revision);
            if let Event::Claimed { generation: next } = &event {
                generation = *next;
            }
            let phase = event
                .identity()
                .0
                .and_then(|n| state.unwrap().node(n).unwrap().phase.clone());
            Record::new(
                "sim",
                first + i as u64,
                generation,
                first,
                event,
                time,
                time.saturating_sub(5),
            )
            .in_revision(revision, phase)
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
    let mut received = HashMap::new();
    let mut settled = HashSet::new();
    let mut reservations = HashSet::new();
    let mut accepted = HashSet::new();
    let mut expected_nodes = graph().nodes;
    let mut numbers = HashMap::new();
    let mut bases = HashMap::new();
    for r in journal {
        match &r.event {
            Event::Received {
                node,
                attempt,
                cost,
                ..
            } => {
                assert!(received.insert((node.clone(), *attempt), cost.0).is_none());
            }
            Event::Settled { node, attempt } => {
                assert!(settled.insert((node.clone(), *attempt)));
                expected += received[&(node.clone(), *attempt)];
            }
            Event::Admitted { node, attempt, .. } => {
                assert!(reservations.insert(call_key(r.revision, node, *attempt)));
                assert_eq!(*attempt, numbers.get(node).copied().unwrap_or(0) + 1);
                numbers.insert(node.clone(), *attempt);
            }
            Event::Accepted { node, .. } => {
                accepted.insert(node.clone());
            }
            Event::Expanded { nodes, .. } => {
                expected_nodes.extend(nodes.clone());
            }
            Event::Revised { invalidated, .. } => {
                // Independent lineage oracle: do not ask Projection which nodes it invalidated.
                let mut invalid: HashSet<_> = invalidated.iter().cloned().collect();
                loop {
                    let len = invalid.len();
                    for n in &expected_nodes {
                        if n.depends_on.iter().any(|d| invalid.contains(d)) {
                            invalid.insert(n.id.clone());
                        }
                    }
                    if invalid.len() == len {
                        break;
                    }
                }
                for node in invalid {
                    accepted.remove(&node);
                    bases.insert(node.clone(), numbers.get(&node).copied().unwrap_or(0));
                }
            }
            Event::RunFailed { .. } => panic!("reachable local run failure"),
            _ => {}
        }
    }
    assert_eq!(
        state.graph.nodes, expected_nodes,
        "expansion must restore exactly its recorded children"
    );
    assert_eq!(
        accepted,
        state
            .attempts
            .iter()
            .filter(|(_, a)| a.state == AttemptState::Accepted)
            .map(|(n, _)| n.clone())
            .collect()
    );
    assert_eq!(state.budget.spent.0, expected);
    for (node, a) in &state.attempts {
        assert_eq!(a.revision_base, bases.get(node).copied().unwrap_or(0));
        assert!(a.number - a.revision_base <= state.limits.max_attempts_per_node);
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
#[derive(Default, Debug)]
struct Coverage {
    quotes: usize,
    retries: usize,
    expansions: usize,
    revisions: usize,
    expansion_crashes: usize,
    revision_crashes: usize,
    beyond_lifetime: usize,
}
fn restart(
    state: &mut Projection,
    journal: &mut Vec<Record>,
    quotes: &[Option<NanoUsd>],
    time: u64,
) -> (Coordinator, VecDeque<Command>) {
    assert!(state.disposition.is_none());
    let claim = records(
        Some(state),
        vec![(
            Event::Claimed {
                generation: state.generation + 1,
            },
            time,
        )],
    );
    state.apply_batch(&claim).unwrap();
    journal.extend(claim);
    let mut coordinator = Coordinator::new(
        Projection::replay(journal).unwrap(),
        quotes.to_vec(),
        3,
        time,
    )
    .unwrap();
    let commands = coordinator
        .step(Input::Recovered {
            invalid_outputs: HashSet::new(),
        })
        .unwrap()
        .into();
    (coordinator, commands)
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
fn simulate(seed: u64) -> Result<Coverage, String> {
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
        let mut quotes = vec![Some(NanoUsd(2)); 6];
        if !uninterrupted_equivalent && seed % 3 == 1 {
            quotes[1] = None;
        }
        let mut coordinator = Coordinator::new(committed.clone(), quotes.clone(), 3, 1000).unwrap();
        let mut commands: VecDeque<_> = coordinator
            .step(Input::Recovered {
                invalid_outputs: HashSet::new(),
            })
            .unwrap()
            .into();
        let mut active = Vec::new();
        let mut crashes = 0;
        let mut graph_changes = 0;
        let mut coverage = Coverage::default();
        let mut time = 1000;
        for step in 0..2000 {
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
                            if !uninterrupted_equivalent
                                && rng.next_u32() % 13 == 0
                                && crashes < 3
                                && committed.disposition.is_none()
                            {
                                // Commit survived; its acknowledgement and all in-flight calls did not.
                                crashes += 1;
                                active.clear();
                                commands.clear();
                                (coordinator, commands) =
                                    restart(&mut committed, &mut journal, &quotes, time);
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
                        if attempt > committed.limits.max_attempts_per_node {
                            coverage.beyond_lifetime += 1;
                        }
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
                    Command::Finish(Finish::QuoteBlocked) => {
                        assert!(
                            active.is_empty()
                                && committed.quote_blocked
                                && committed.disposition.is_none()
                        );
                        coverage.quotes += 1;
                        quotes.fill(Some(NanoUsd(2)));
                        (coordinator, commands) =
                            restart(&mut committed, &mut journal, &quotes, time);
                    }
                    Command::Finish(finish) => {
                        assert_invariants(&committed, &journal);
                        assert!(active.is_empty());
                        if !uninterrupted_equivalent && committed.admission == Admission::Open {
                            let retry = committed
                                .graph
                                .nodes
                                .iter()
                                .filter_map(|n| {
                                    let a = committed.attempts.get(&n.id)?;
                                    (matches!(
                                        a.state,
                                        AttemptState::Failed | AttemptState::Rejected
                                    ) && a.number - a.revision_base
                                        < committed.limits.max_attempts_per_node)
                                        .then(|| Event::Retried {
                                            node: n.id.clone(),
                                            attempt: a.number,
                                        })
                                })
                                .collect::<Vec<_>>();
                            if finish == Finish::Failed
                                && !retry.is_empty()
                                && rng.next_u32() % 2 == 0
                            {
                                let event = retry[rng.next_u32() as usize % retry.len()].clone();
                                let changes = records(Some(&committed), vec![(event, time)]);
                                committed.apply_batch(&changes).unwrap();
                                journal.extend(changes);
                                coverage.retries += 1;
                                (coordinator, commands) =
                                    restart(&mut committed, &mut journal, &quotes, time);
                                continue;
                            }
                            if graph_changes < 3 && rng.next_u32() % 4 != 0 {
                                graph_changes += 1;
                                let parent = rng.next_u32() as usize % committed.graph.nodes.len();
                                let expand = rng.next_u32() % 2 == 0;
                                let event = if expand {
                                    let i = committed.graph.nodes.len();
                                    coverage.expansions += 1;
                                    Event::Expanded {
                                        parent_revision: committed.revision,
                                        revision: committed.revision + 1,
                                        input_manifest_sha256: format!("{seed:064x}"),
                                        nodes: vec![Node {
                                            id: i.to_string(),
                                            phase: Some("simulation".into()),
                                            task: Task {
                                                goal: "expanded".into(),
                                                input: i.into(),
                                            },
                                            depends_on: vec![
                                                committed.graph.nodes[parent].id.clone(),
                                            ],
                                        }],
                                    }
                                } else {
                                    coverage.revisions += 1;
                                    Event::Revised {
                                        revision: committed.revision + 1,
                                        reason: "fixture".into(),
                                        invalidated: vec![committed.graph.nodes[parent].id.clone()],
                                    }
                                };
                                let changes = records(Some(&committed), vec![(event, time)]);
                                committed.apply_batch(&changes).unwrap();
                                journal.extend(changes.clone());
                                quotes.resize(committed.graph.nodes.len(), Some(NanoUsd(2)));
                                assert_invariants(&committed, &journal);
                                if rng.next_u32() % 2 == 0 {
                                    // Crash after the graph commit, before the old coordinator observes it.
                                    if expand {
                                        coverage.expansion_crashes += 1;
                                    } else {
                                        coverage.revision_crashes += 1;
                                    }
                                    (coordinator, commands) =
                                        restart(&mut committed, &mut journal, &quotes, time);
                                } else {
                                    commands
                                        .extend(coordinator.step(Input::Foreign(changes)).unwrap());
                                    coordinator.set_quotes(quotes.clone()).unwrap();
                                    commands.extend(
                                        coordinator.step(Input::Tick { now_ms: time }).unwrap(),
                                    );
                                }
                                continue;
                            }
                        }
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
                            assert_eq!(graph_changes, 0);
                            assert_eq!(crashes, 0);
                        }
                        return coverage;
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
            assert!(step < 1999, "scheduler failed to terminate");
        }
        unreachable!()
    });
    result.map_err(|_| format!("ChaCha8 seed {seed}; replay this exact seed"))
}
#[test]
fn five_thousand_seeded_schedules_replay_committed_prefixes() {
    let mut total = Coverage::default();
    for seed in 0..5000 {
        let c = simulate(seed).unwrap();
        total.quotes += c.quotes;
        total.retries += c.retries;
        total.expansions += c.expansions;
        total.revisions += c.revisions;
        total.expansion_crashes += c.expansion_crashes;
        total.revision_crashes += c.revision_crashes;
        total.beyond_lifetime += c.beyond_lifetime;
    }
    assert!(
        total.quotes > 0
            && total.retries > 0
            && total.expansions > 0
            && total.revisions > 0
            && total.expansion_crashes > 0
            && total.revision_crashes > 0
            && total.beyond_lifetime > 0,
        "{total:?}"
    );
    println!("{total:?}");
}
#[test]
fn recorded_counterexample_detects_lifetime_attempt_limit() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/revision-attempt-limit.json")).unwrap();
    let coverage = simulate(fixture["seed"].as_u64().unwrap()).unwrap();
    assert!(coverage.revisions > 0 && coverage.beyond_lifetime > 0);
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
