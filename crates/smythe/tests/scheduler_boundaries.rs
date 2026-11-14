use anyhow::{Result, bail};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smythe::{
    CompletionResult, Dependency, Effect, EngineError, ExecutionGraph, Limits, Node, Provider,
    Swarm, Task,
    seams::{Clock, ControlWatch, Sources, StorageBoundary, StorageObserver},
    store::Store,
};
use smythe_core::journal::{AttemptState, Event, Projection, Record};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::Receiver,
    },
    time::Duration,
};

fn graph(dependencies: &[&[&str]]) -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: dependencies
            .iter()
            .enumerate()
            .map(|(i, parents)| Node {
                id: i.to_string(),
                task: Task {
                    goal: i.to_string(),
                    input: i.into(),
                },
                depends_on: parents.iter().map(|p| (*p).into()).collect(),
                phase: None,
            })
            .collect(),
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
        let (ready, _) = self
            .1
            .wait_timeout_while(self.0.lock().unwrap(), Duration::from_secs(5), |ready| {
                !*ready
            })
            .unwrap();
        assert!(*ready, "sibling was never released");
    }
}

struct RetryWatch(Arc<Gate>);
impl ControlWatch for RetryWatch {
    #[allow(clippy::disallowed_methods)] // This fixture implements the control-wait seam.
    fn poll(
        &self,
        store: &Store,
        run: &str,
        after: u64,
        wake: &Receiver<()>,
    ) -> Result<Vec<Record>> {
        let state = store.inspect(run)?;
        if state
            .attempts
            .get("0")
            .is_some_and(|a| a.state == AttemptState::Failed)
        {
            assert_eq!(state.attempts["1"].state, AttemptState::Dispatched);
            store.retry(run, "0")?;
            self.0.release();
        } else {
            let _ = wake.recv_timeout(Duration::from_millis(10));
        }
        store.changes_since(run, after)
    }
}

#[test]
fn retry_during_a_running_sibling_preserves_completions_costs_and_children() {
    struct Fake {
        gate: Arc<Gate>,
        calls: [AtomicUsize; 4],
    }
    impl Provider for Fake {
        type Output = u64;
        fn effect(&self) -> Effect {
            Effect::Pure
        }
        fn binding(&self) -> String {
            "live-retry".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(
            &self,
            task: &Task,
            deps: &[Dependency<'_, u64>],
        ) -> Result<CompletionResult<u64>> {
            let index = task.input.as_u64().unwrap() as usize;
            let previous = self.calls[index].fetch_add(1, Ordering::SeqCst);
            if index == 0 && previous == 0 {
                bail!("fixture failure");
            }
            if index == 1 {
                self.gate.wait();
            }
            Ok(CompletionResult {
                output: 1 + deps.iter().map(|d| d.load().unwrap()).sum::<u64>(),
                cost_nanousd: 1,
            })
        }
    }
    let gate = Arc::new(Gate::default());
    let provider = Fake {
        gate: gate.clone(),
        calls: std::array::from_fn(|_| AtomicUsize::new(0)),
    };
    let store = Store::memory().unwrap();
    let graph = graph(&[&[], &[], &["1"], &["0", "2"]]);
    let result = Swarm::new(2, 10)
        .unwrap()
        .with_sources(Sources {
            controls: Arc::new(RetryWatch(gate)),
            ..Sources::default()
        })
        .execute_in(&graph, &provider, &store, "retry", false)
        .unwrap();
    assert_eq!(result.load_all::<u64>().unwrap(), [1, 1, 2, 4]);
    assert_eq!(result.total_cost_nanousd, 4);
    assert_eq!(
        provider.calls.each_ref().map(|n| n.load(Ordering::SeqCst)),
        [2, 1, 1, 1]
    );
    let state = store.inspect("retry").unwrap();
    assert!(state.budget.pending().is_empty());
    assert_eq!(state.attempts["1"].number, 1);
    let resumed = Swarm::new(2, 10)
        .unwrap()
        .execute_in(&graph, &provider, &store, "retry", true)
        .unwrap();
    assert_eq!(resumed.reused_nodes, 4);
    assert!(resumed.trace.spans.is_empty());
    assert_eq!(resumed.total_cost_nanousd, 4);
    for prefix in 1..=store.history("retry").unwrap().len() {
        Projection::replay(&store.history("retry").unwrap()[..prefix]).unwrap();
    }
}

#[derive(Clone, Copy)]
enum Delay {
    Serialization,
    Publication,
    Commit,
}
struct Timer {
    now: AtomicU64,
    delay: Delay,
}
impl Clock for Timer {
    fn unix_ms(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
    fn monotonic(&self) -> Duration {
        Duration::from_millis(self.unix_ms())
    }
}
impl StorageObserver for Timer {
    fn observe(&self, boundary: StorageBoundary) -> Result<()> {
        if matches!(
            (self.delay, boundary),
            (Delay::Publication, StorageBoundary::BlobPublished)
                | (
                    Delay::Commit,
                    StorageBoundary::AfterCommit { first: 5, last: 7 }
                )
        ) {
            self.now.store(100, Ordering::SeqCst);
        }
        Ok(())
    }
}
struct TimedOutput {
    value: String,
    timer: Option<Arc<Timer>>,
}
impl Serialize for TimedOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(timer) = &self.timer {
            timer.now.store(100, Ordering::SeqCst);
        }
        serializer.serialize_str(&self.value)
    }
}
impl<'de> Deserialize<'de> for TimedOutput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            value: String::deserialize(deserializer)?,
            timer: None,
        })
    }
}

#[test]
fn deadlines_include_serialization_publication_and_commit_time() {
    struct Fake {
        timer: Arc<Timer>,
        calls: AtomicUsize,
    }
    impl Provider for Fake {
        type Output = TimedOutput;
        fn effect(&self) -> Effect {
            Effect::Pure
        }
        fn binding(&self) -> String {
            "deadline-latency".into()
        }
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(3)
        }
        fn complete(
            &self,
            _: &Task,
            _: &[Dependency<'_, TimedOutput>],
        ) -> Result<CompletionResult<TimedOutput>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResult {
                output: TimedOutput {
                    value: "x".repeat(if matches!(self.timer.delay, Delay::Publication) {
                        70_000
                    } else {
                        1
                    }),
                    timer: matches!(self.timer.delay, Delay::Serialization)
                        .then(|| self.timer.clone()),
                },
                cost_nanousd: 3,
            })
        }
    }
    for delay in [Delay::Serialization, Delay::Publication, Delay::Commit] {
        let timer = Arc::new(Timer {
            now: AtomicU64::new(0),
            delay,
        });
        let sources = Sources {
            clock: timer.clone(),
            storage: timer.clone(),
            ..Sources::default()
        };
        let directory =
            std::env::temp_dir().join(format!("smythe-deadline-{}", sources.ids.next_id()));
        {
            let store = Store::open_with_sources(&directory, sources.clone()).unwrap();
            let provider = Fake {
                timer: timer.clone(),
                calls: AtomicUsize::new(0),
            };
            let graph = graph(&[&[], &["0"]]);
            let error = Swarm::new(1, 10)
                .unwrap()
                .with_sources(sources)
                .with_limits(Limits {
                    wall_deadline_ms: Some(10),
                    ..Limits::default()
                })
                .execute_in(&graph, &provider, &store, "deadline", false)
                .err()
                .unwrap();
            assert!(EngineError::Stopped.matches(&error));
            let state = store.inspect("deadline").unwrap();
            assert!(state.deadline_closed);
            assert_eq!(state.accepted_nodes, 1);
            assert_eq!(state.budget.spent.0, 3);
            assert!(state.budget.pending().is_empty());
            assert_eq!(state.attempts.len(), 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let history = store.history("deadline").unwrap();
            assert!(
                history
                    .iter()
                    .any(|r| matches!(r.event, Event::AdmissionClosed { .. })
                        && r.occurred_at_unix_ms == 100)
            );
            timer.now.store(0, Ordering::SeqCst);
            assert!(
                Swarm::new(1, 10)
                    .unwrap()
                    .with_sources(Sources {
                        clock: timer.clone(),
                        ..Sources::default()
                    })
                    .execute_in(&graph, &provider, &store, "deadline", true)
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
