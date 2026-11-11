use serde::Deserialize;
use serde_json::Value;
use smythe_core::{
    NanoUsd,
    coordinator::{Command, Coordinator, Finish, Input},
    journal::{AttemptState, Event, JOURNAL_SCHEMA, Projection, Record},
};
use std::collections::{HashSet, VecDeque};

#[derive(Deserialize)]
struct Fixture {
    producer_revision: String,
    records: Vec<Record>,
    projections: Vec<Value>,
}
fn fixture(text: &str) -> Fixture {
    serde_json::from_str(text).unwrap()
}
const CRASH: &str = include_str!("fixtures/schema3-revision-crash-limit.json");

#[test]
fn original_schema3_journals_preserve_every_prefix_projection() {
    assert_eq!(JOURNAL_SCHEMA, 3);
    for text in [
        include_str!("fixtures/schema3-retry-revision.json"),
        include_str!("fixtures/schema3-raised-budget.json"),
        include_str!("fixtures/schema3-failed-quote.json"),
        CRASH,
    ] {
        let f = fixture(text);
        assert_eq!(
            f.producer_revision,
            "a6f4ef31307425410cf5aa382163e1ab6ce60363"
        );
        assert_eq!(f.records.len(), f.projections.len());
        for (i, expected) in f.projections.iter().enumerate() {
            let mut actual =
                serde_json::to_value(Projection::replay(&f.records[..=i]).unwrap()).unwrap();
            assert_eq!(
                actual.as_object_mut().unwrap().remove("quote_blocked"),
                Some(false.into())
            );
            for attempt in actual["attempts"].as_object_mut().unwrap().values_mut() {
                assert!(
                    attempt
                        .as_object_mut()
                        .unwrap()
                        .remove("revision_base")
                        .is_some()
                );
            }
            assert_eq!(
                &actual,
                expected,
                "{} prefix {}",
                f.records[0].run_id,
                i + 1
            );
        }
    }
}

fn append(
    state: &mut Projection,
    journal: &mut Vec<Record>,
    events: Vec<(Event, u64)>,
) -> Vec<Record> {
    let first = state.sequence + 1;
    let mut revision = state.revision;
    let mut generation = state.generation;
    let records: Vec<_> = events
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
                .and_then(|n| state.node(n).unwrap().phase.clone());
            Record::new(
                &state.run_id,
                first + i as u64,
                generation,
                first,
                event,
                time,
                time,
            )
            .in_revision(revision, phase)
        })
        .collect();
    state.apply_batch(&records).unwrap();
    journal.extend(records.clone());
    records
}

#[test]
fn revised_crash_loop_uses_new_replay_marker_and_exhausts_its_own_limit() {
    let mut journal = fixture(CRASH).records;
    let crash = journal
        .iter()
        .position(|r| matches!(r.event, Event::LocalReplay { .. }))
        .unwrap();
    journal.truncate(crash);
    let mut state = Projection::replay(&journal).unwrap();
    assert_eq!(state.attempts["n0"].revision_base, 2);
    for expected_attempt in [3, 4, 5] {
        let generation = state.generation + 1;
        append(
            &mut state,
            &mut journal,
            vec![(Event::Claimed { generation }, 1000)],
        );
        let mut coordinator =
            Coordinator::new(state.clone(), vec![Some(NanoUsd(2))], 1, 1000).unwrap();
        let mut commands: VecDeque<_> = coordinator
            .step(Input::Recovered {
                invalid_outputs: HashSet::new(),
            })
            .unwrap()
            .into();
        let mut replayed = false;
        while let Some(command) = commands.pop_front() {
            match command {
                Command::Append { events, .. } => {
                    for (event, _) in &events {
                        if let Event::LocalReplay {
                            attempt,
                            revision_base,
                            ..
                        } = event
                        {
                            assert_eq!(*attempt, expected_attempt);
                            assert_eq!(*revision_base, Some(2));
                            replayed = true;
                        }
                        assert!(!matches!(event, Event::RunFailed { .. }));
                    }
                    let ack = append(&mut state, &mut journal, events);
                    commands.extend(coordinator.step(Input::Committed(ack)).unwrap());
                }
                Command::Dispatch { attempt, .. } => {
                    assert!(expected_attempt < 5);
                    assert_eq!(attempt, expected_attempt + 1);
                    // Crash before a completion: keep exactly the committed intent.
                    break;
                }
                Command::Finish(finish) => {
                    assert_eq!(expected_attempt, 5);
                    assert_eq!(finish, Finish::Failed);
                }
            }
        }
        assert!(replayed);
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            serde_json::to_value(Projection::replay(&journal).unwrap()).unwrap()
        );
    }
    assert_eq!(state.attempts["n0"].state, AttemptState::Failed);
    assert_eq!(
        state.attempts["n0"].reason.as_deref(),
        Some("attempt_limit")
    );
    assert!(state.budget.pending().is_empty());
}

#[test]
fn replay_marker_cannot_forge_a_fresh_attempt_allowance() {
    let f = fixture(CRASH);
    let at = f
        .records
        .iter()
        .position(|r| matches!(r.event, Event::LocalReplay { .. }))
        .unwrap();
    let mut state = Projection::replay(&f.records[..at]).unwrap();
    let before = serde_json::to_value(&state).unwrap();
    let mut record = f.records[at].clone();
    if let Event::LocalReplay { revision_base, .. } = &mut record.event {
        *revision_base = Some(3);
    }
    assert!(state.apply(&record).is_err());
    assert_eq!(serde_json::to_value(state).unwrap(), before);
}
