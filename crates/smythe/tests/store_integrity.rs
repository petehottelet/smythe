//! A damaged store is refused with a specific reason, and repaired only by an
//! explicit operator decision.
use anyhow::Result;
use smythe::{
    CompletionResult, Control, Dependency, Effect, EngineError, ExecutionGraph, Node, Provider,
    Swarm, SwarmResult, Task,
    seams::{ControlWatch, Sources},
    store::Store,
};
use smythe_core::{
    diagnostics::{DamagedOutputs, message},
    journal::{AttemptState, Event, Record},
};
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::Receiver,
    },
    time::Duration,
};

const NODES: [&str; 3] = ["a", "b", "c"];

/// `b` depends on `a`; `c` is independent.
fn graph() -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: NODES
            .iter()
            .enumerate()
            .map(|(i, id)| Node {
                id: (*id).into(),
                task: Task {
                    goal: "describe".into(),
                    input: i.into(),
                },
                depends_on: if *id == "b" { vec!["a".into()] } else { vec![] },
                phase: None,
            })
            .collect(),
    }
}

/// Distinct output per node, so no two nodes share a stored blob.
struct Words {
    calls: [AtomicUsize; 3],
    padding: usize,
}
impl Words {
    fn new(padding: usize) -> Self {
        Self {
            calls: std::array::from_fn(|_| AtomicUsize::new(0)),
            padding,
        }
    }
    fn calls(&self) -> [usize; 3] {
        self.calls.each_ref().map(|n| n.load(Ordering::SeqCst))
    }
}
impl Provider for Words {
    type Output = String;
    fn effect(&self) -> Effect {
        Effect::Pure
    }
    fn binding(&self) -> String {
        "store-integrity-v1".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(1)
    }
    fn complete(
        &self,
        task: &Task,
        dependencies: &[Dependency<'_, String>],
    ) -> Result<CompletionResult<String>> {
        let index = task.input.as_u64().unwrap() as usize;
        self.calls[index].fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResult {
            output: format!(
                "{}<{}>{}",
                NODES[index],
                dependencies.len(),
                "x".repeat(self.padding)
            ),
            cost_nanousd: 1,
        })
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "smythe-integrity-{}",
            Sources::default().ids.next_id()
        )))
    }
    fn database(&self) -> rusqlite::Connection {
        rusqlite::Connection::open(self.0.join("journal.sqlite3")).unwrap()
    }
    fn events(&self, run: &str) -> i64 {
        self.database()
            .query_row("SELECT COUNT(*) FROM events WHERE run_id=?", [run], |r| {
                r.get(0)
            })
            .unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(store: &Store, provider: &Words, id: &str, resume: bool) -> Result<SwarmResult> {
    Swarm::new(2, 100)?.execute_in(&graph(), provider, store, id, resume)
}
fn refused(error: anyhow::Error, reason: &str) {
    assert!(EngineError::Integrity.matches(&error), "{error:#}");
    assert!(message(&error).contains(reason), "{}", message(&error));
}

#[test]
fn a_journal_that_lost_its_tail_is_refused_before_any_decision() {
    const SHORTER: &str = "journal is shorter than its recorded summary";
    let directory = Directory::new();
    let provider = Words::new(0);
    {
        let store = Store::open(&directory.0).unwrap();
        run(&store, &provider, "run", false).unwrap();
        run(&store, &provider, "other", false).unwrap();
    }
    let total = directory.events("run");
    directory
        .database()
        .execute(
            "DELETE FROM events WHERE run_id='run' AND sequence>?",
            [total - 4],
        )
        .unwrap();
    let store = Store::open_existing(&directory.0).unwrap();
    refused(store.inspect("run").unwrap_err(), SHORTER);
    refused(store.history("run").unwrap_err(), SHORTER);
    refused(store.events_after("run", 0, 10).unwrap_err(), SHORTER);
    refused(store.list().unwrap_err(), SHORTER);
    refused(store.control("run", Control::Pause).unwrap_err(), SHORTER);
    refused(store.retry("run", "a").unwrap_err(), SHORTER);
    refused(store.invalidate("run", "a").unwrap_err(), SHORTER);
    refused(run(&store, &provider, "run", true).err().unwrap(), SHORTER);
    // A fresh run cannot adopt the damaged identifier either.
    refused(run(&store, &provider, "run", false).err().unwrap(), SHORTER);
    // The undamaged run beside it is still readable.
    assert!(store.inspect("other").unwrap().completed());

    // Losing every event is still not an empty, reusable run.
    directory
        .database()
        .execute("DELETE FROM events WHERE run_id='run'", [])
        .unwrap();
    refused(store.inspect("run").unwrap_err(), SHORTER);
    refused(run(&store, &provider, "run", false).err().unwrap(), SHORTER);

    // Events without their summary are refused as well.
    directory
        .database()
        .execute("DELETE FROM runs WHERE run_id='other'", [])
        .unwrap();
    refused(
        store.inspect("other").unwrap_err(),
        "journal differs from its recorded summary",
    );

    // No refusal dispatched work or wrote an event.
    assert_eq!(provider.calls(), [2, 2, 2]);
    assert_eq!(directory.events("run"), 0);
    assert_eq!(directory.events("other"), total);
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    Removed,
    Altered,
}

#[test]
fn damaged_accepted_outputs_are_named_and_recomputed_only_after_invalidation() {
    // Small outputs are rows beside the journal; large ones are files.
    for padding in [0, 70_000] {
        for damage in [Damage::Removed, Damage::Altered] {
            let directory = Directory::new();
            let provider = Words::new(padding);
            let reference = {
                let store = Store::open(&directory.0).unwrap();
                let result = run(&store, &provider, "run", false).unwrap();
                assert_eq!(result.total_cost_nanousd, 3);
                store.inspect("run").unwrap().attempts["a"]
                    .output
                    .clone()
                    .unwrap()
            };
            let file = directory
                .0
                .join("blobs")
                .join(&reference.sha256[..2])
                .join(&reference.sha256);
            assert_eq!(file.is_file(), padding > 0, "{padding} {damage:?}");
            match (file.is_file(), damage) {
                (true, Damage::Removed) => fs::remove_file(&file).unwrap(),
                (true, Damage::Altered) => fs::write(&file, b"damaged").unwrap(),
                (false, Damage::Removed) => {
                    directory
                        .database()
                        .execute("DELETE FROM blobs WHERE digest=?", [&reference.sha256])
                        .unwrap();
                }
                (false, Damage::Altered) => {
                    directory
                        .database()
                        .execute(
                            "UPDATE blobs SET bytes=? WHERE digest=?",
                            rusqlite::params![b"damaged".as_slice(), reference.sha256],
                        )
                        .unwrap();
                }
            }

            let store = Store::open_existing(&directory.0).unwrap();
            // The journal alone still describes a completed run.
            let state = store.inspect("run").unwrap();
            assert!(state.completed());
            assert_eq!(store.damaged_outputs(&state).unwrap(), ["a"]);
            let error = run(&store, &provider, "run", true).err().unwrap();
            assert!(EngineError::Integrity.matches(&error));
            let named = error.downcast_ref::<DamagedOutputs>().unwrap();
            assert_eq!(named.nodes, ["a"]);
            assert_eq!(named.total, 1);
            assert!(message(&error).contains("accepted output is missing or damaged"));
            assert_eq!(provider.calls(), [1, 1, 1]);

            let unknown = store.invalidate("run", "missing").unwrap_err();
            assert!(EngineError::InvalidCommand.matches(&unknown));
            assert!(message(&unknown).contains("unknown node"));
            store.invalidate("run", "a").unwrap();
            let repeated = store.invalidate("run", "a").unwrap_err();
            assert!(EngineError::InvalidCommand.matches(&repeated));
            assert!(message(&repeated).contains("node has no accepted output"));
            let state = store.inspect("run").unwrap();
            assert!(!state.completed());
            assert_eq!(
                NODES.map(|n| state.attempts[n].state),
                [
                    AttemptState::Invalidated,
                    AttemptState::Invalidated,
                    AttemptState::Accepted
                ]
            );

            // Only the invalidated node and its descendant run and are charged again.
            let resumed = run(&store, &provider, "run", true).unwrap();
            assert_eq!(provider.calls(), [2, 2, 1], "{padding} {damage:?}");
            assert_eq!(resumed.reused_nodes, 1);
            assert_eq!(resumed.total_cost_nanousd, 5);
            assert!(resumed.load::<String>("a").unwrap().starts_with("a<0>"));
            let state = store.inspect("run").unwrap();
            assert!(state.completed());
            assert!(store.damaged_outputs(&state).unwrap().is_empty());
        }
    }
}

#[derive(Default)]
struct Gate(Mutex<bool>, Condvar);
impl Gate {
    fn release(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
    fn wait(&self) {
        let (open, _) = self
            .1
            .wait_timeout_while(self.0.lock().unwrap(), Duration::from_secs(5), |open| {
                !*open
            })
            .unwrap();
        assert!(*open, "gated node was never released");
    }
}
struct Gated {
    gate: Arc<Gate>,
}
impl Provider for Gated {
    type Output = u64;
    fn effect(&self) -> Effect {
        Effect::Pure
    }
    fn binding(&self) -> String {
        "store-integrity-gated-v1".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(1)
    }
    fn complete(&self, task: &Task, _: &[Dependency<'_, u64>]) -> Result<CompletionResult<u64>> {
        let index = task.input.as_u64().unwrap();
        if index == 2 {
            self.gate.wait();
        }
        Ok(CompletionResult {
            output: index,
            cost_nanousd: 1,
        })
    }
}
/// Attempts the invalidation while `c` is still running, then releases it.
struct InvalidateWhileActive {
    gate: Arc<Gate>,
    checked: AtomicBool,
}
impl ControlWatch for InvalidateWhileActive {
    #[allow(clippy::disallowed_methods)] // This fixture implements the control-wait seam.
    fn poll(
        &self,
        store: &Store,
        run: &str,
        after: u64,
        wake: &Receiver<()>,
    ) -> Result<Vec<Record>> {
        let state = store.inspect(run)?;
        let accepted = |node: &str| {
            state
                .attempts
                .get(node)
                .is_some_and(|a| a.state == AttemptState::Accepted)
        };
        if accepted("a") && accepted("b") && !self.checked.load(Ordering::SeqCst) {
            assert_eq!(state.attempts["c"].state, AttemptState::Dispatched);
            let error = store.invalidate(run, "a").unwrap_err();
            assert!(EngineError::InvalidCommand.matches(&error));
            assert!(message(&error).contains("invalidation requires a quiescent run"));
            self.checked.store(true, Ordering::SeqCst);
            self.gate.release();
        } else {
            let _ = wake.recv_timeout(Duration::from_millis(10));
        }
        store.changes_since(run, after)
    }
}

#[test]
fn invalidation_is_refused_while_work_is_active_or_after_a_stop() {
    let gate = Arc::new(Gate::default());
    let watch = Arc::new(InvalidateWhileActive {
        gate: gate.clone(),
        checked: AtomicBool::new(false),
    });
    let store = Store::memory().unwrap();
    let result = Swarm::new(2, 100)
        .unwrap()
        .with_sources(Sources {
            controls: watch.clone(),
            ..Sources::default()
        })
        .execute_in(&graph(), &Gated { gate }, &store, "live", false)
        .unwrap();
    assert!(watch.checked.load(Ordering::SeqCst));
    // The refusal recorded nothing: no revision, no second attempt.
    assert_eq!(result.total_cost_nanousd, 3);
    let history = store.history("live").unwrap();
    assert!(
        !history
            .iter()
            .any(|r| matches!(r.event, Event::Revised { .. }))
    );

    store
        .append("stopped", 0, 0, history[0].event.clone())
        .unwrap();
    store
        .append("stopped", 1, 1, Event::Claimed { generation: 1 })
        .unwrap();
    store.control("stopped", Control::Stop).unwrap();
    assert!(EngineError::Stopped.matches(&store.invalidate("stopped", "a").unwrap_err()));
}
