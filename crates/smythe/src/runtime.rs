use crate::store::Store;
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use smythe_core::{
    ExecutionGraph, NanoUsd, Task,
    journal::{AttemptState, Control, Event, Projection},
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

pub struct CompletionResult<T> {
    pub output: T,
    pub cost_nanousd: u64,
}
/// Pure local work only. Failure must entail no charge or external effect.
/// Remote adapters require the separate uncertain-outcome contract before admission.
pub trait Provider: Sync {
    type Output: Serialize + DeserializeOwned + Send + Sync;
    fn binding(&self) -> String;
    fn cost_bound_nanousd(&self, task: &Task) -> Result<u64>;
    fn complete(
        &self,
        task: &Task,
        dependencies: &[&Self::Output],
    ) -> Result<CompletionResult<Self::Output>>;
}
#[derive(Debug, Serialize)]
pub struct Span {
    pub node_id: String,
    pub elapsed_s: f64,
    pub cost_nanousd: NanoUsd,
}
#[derive(Debug, Default, Serialize)]
pub struct Tracer {
    pub spans: Vec<Span>,
}
pub struct SwarmResult<T> {
    pub outputs: Vec<T>,
    pub trace: Tracer,
    pub total_cost_nanousd: u64,
    pub reused_nodes: usize,
    pub run_id: String,
}
pub struct Swarm {
    pool: rayon::ThreadPool,
    max_concurrency: usize,
    max_budget_nanousd: u64,
}

fn record(store: &Store, id: &str, state: &mut Projection, event: Event) -> Result<()> {
    record_many(store, id, state, vec![event])
}
fn record_many(store: &Store, id: &str, state: &mut Projection, events: Vec<Event>) -> Result<()> {
    // Controls can be appended by another process. Refresh their authority before retrying.
    for _ in 0..16 {
        match store.append_batch(id, state.sequence, state.generation, events.clone()) {
            Ok(next) => {
                *state = next;
                return Ok(());
            }
            Err(e) if e.to_string().contains("journal changed") => *state = store.current(id)?,
            Err(e) => return Err(e),
        }
    }
    anyhow::bail!("journal is busy")
}

impl Swarm {
    pub fn new(max_concurrency: usize, max_budget_nanousd: u64) -> Result<Self> {
        ensure!(
            (1..=64).contains(&max_concurrency),
            "concurrency must be 1 through 64"
        );
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(max_concurrency)
                .build()?,
            max_concurrency,
            max_budget_nanousd,
        })
    }
    pub fn execute<P: Provider>(
        &self,
        graph: &ExecutionGraph,
        provider: &P,
    ) -> Result<SwarmResult<P::Output>> {
        self.execute_in(
            graph,
            provider,
            &Store::memory()?,
            &uuid::Uuid::now_v7().to_string(),
            false,
        )
    }
    pub fn execute_in<P: Provider>(
        &self,
        graph: &ExecutionGraph,
        provider: &P,
        store: &Store,
        id: &str,
        resume: bool,
    ) -> Result<SwarmResult<P::Output>> {
        graph.validate()?;
        // JCS-equivalent graphs must also have the same provider-visible inputs.
        let canonical: ExecutionGraph = serde_json::from_slice(&serde_jcs::to_vec(graph)?)?;
        let graph = &canonical;
        let _lease = store.lease(id)?;
        let binding = crate::store::digest(&serde_jcs::to_vec(
            &json!({"engine":"local-v1","graph":graph,"provider":provider.binding(),"ceiling":NanoUsd(self.max_budget_nanousd)}),
        )?);
        let history = store.history(id)?;
        ensure!(
            resume || history.is_empty(),
            "run already exists; use resume"
        );
        ensure!(!resume || !history.is_empty(), "run does not exist");
        let mut state = if history.is_empty() {
            store.append(
                id,
                0,
                0,
                Event::Started {
                    binding: binding.clone(),
                    graph: graph.clone(),
                    ceiling: NanoUsd(self.max_budget_nanousd),
                },
            )?
        } else {
            Projection::replay(&history)?
        };
        ensure!(state.binding == binding, "run binding differs");
        if !state.completed {
            let generation = state
                .generation
                .checked_add(1)
                .context("generation overflow")?;
            state = store.append(
                id,
                state.sequence,
                generation,
                Event::Claimed { generation },
            )?;
            // Pure work may replay. Received evidence is completed without another call.
            for (node, attempt) in state.attempts.clone() {
                match attempt.state {
                    AttemptState::Admitted | AttemptState::Dispatched => record(
                        store,
                        id,
                        &mut state,
                        Event::LocalReplay {
                            node,
                            attempt: attempt.number,
                        },
                    )?,
                    AttemptState::Received | AttemptState::Settled => {
                        if attempt.state == AttemptState::Received {
                            record(
                                store,
                                id,
                                &mut state,
                                Event::Settled {
                                    node: node.clone(),
                                    attempt: attempt.number,
                                },
                            )?;
                        }
                        let decoded = attempt
                            .output
                            .as_ref()
                            .context("missing output")
                            .and_then(|value| store.output(value))
                            .and_then(|bytes| Ok(serde_json::from_slice::<P::Output>(&bytes)?));
                        if !attempt.output_valid || decoded.is_err() {
                            record(
                                store,
                                id,
                                &mut state,
                                Event::Rejected {
                                    node,
                                    attempt: attempt.number,
                                    reason: "invalid_local_output".into(),
                                },
                            )?;
                            anyhow::bail!("saved local output is invalid; known spend is retained");
                        }
                        record(
                            store,
                            id,
                            &mut state,
                            Event::Accepted {
                                node,
                                attempt: attempt.number,
                            },
                        )?;
                    }
                    AttemptState::Failed | AttemptState::Rejected => {
                        anyhow::bail!(
                            "run has a failed or rejected local attempt; inspect its journal"
                        )
                    }
                    _ => {}
                }
            }
        }
        let mut values: Vec<Option<Arc<P::Output>>> =
            (0..graph.nodes.len()).map(|_| None).collect();
        let ids: HashMap<_, _> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        for (node, attempt) in &state.attempts {
            if attempt.state == AttemptState::Accepted {
                values[ids[node.as_str()]] = Some(Arc::new(serde_json::from_slice(
                    &store.output(attempt.output.as_ref().context("missing output")?)?,
                )?));
            }
        }
        let reused_nodes = values.iter().filter(|v| v.is_some()).count();
        let mut ready: VecDeque<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| {
                values[*i].is_none()
                    && n.depends_on
                        .iter()
                        .all(|p| values[ids[p.as_str()]].is_some())
            })
            .map(|(i, _)| i)
            .collect();
        let mut enqueued: HashSet<usize> = ready.iter().copied().collect();
        let schedule = graph.schedule()?;
        let (sender, receiver) = mpsc::sync_channel(self.max_concurrency);
        let mut active = 0usize;
        let mut trace = Tracer::default();
        let mut failure: Option<anyhow::Error> = None;
        self.pool.in_place_scope(|scope| -> Result<()> {
            loop {
                if !state.completed {
                    state = store.current(id)?;
                }
                while failure.is_none()
                    && state.control == Control::Run
                    && active < self.max_concurrency
                {
                    let Some(index) = ready.pop_front() else {
                        break;
                    };
                    let node = &graph.nodes[index];
                    let bound = match provider.cost_bound_nanousd(&node.task) {
                        Ok(bound) => bound,
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    };
                    let attempt = state
                        .attempts
                        .get(&node.id)
                        .map_or(Some(1), |a| a.number.checked_add(1))
                        .context("attempt overflow")?;
                    if let Err(error) = record_many(
                        store,
                        id,
                        &mut state,
                        vec![
                            Event::Admitted {
                                node: node.id.clone(),
                                attempt,
                                bound: NanoUsd(bound),
                            },
                            Event::Dispatched {
                                node: node.id.clone(),
                                attempt,
                            },
                        ],
                    ) {
                        if active > 0 && error.to_string().contains("budget exhausted") {
                            ready.push_front(index);
                            break;
                        }
                        failure = Some(error);
                        break;
                    }
                    let dependencies: Vec<Arc<P::Output>> = node
                        .depends_on
                        .iter()
                        .map(|p| Arc::clone(values[ids[p.as_str()]].as_ref().unwrap()))
                        .collect();
                    let sender = sender.clone();
                    active += 1;
                    scope.spawn(move |_| {
                        let started = Instant::now();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            provider.complete(
                                &node.task,
                                &dependencies.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                            )
                        }))
                        .unwrap_or_else(|_| Err(anyhow!("local provider panicked")));
                        let _ =
                            sender.send((index, attempt, started.elapsed().as_secs_f64(), result));
                    });
                }
                if active == 0 {
                    break;
                }
                let (index, attempt, elapsed_s, result) =
                    match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(result) => result,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(_) => return Err(anyhow!("worker completion channel closed")),
                    };
                active -= 1;
                let node = &graph.nodes[index];
                match result {
                    Ok(result) => {
                        let (bytes, decoded) = match serde_json::to_vec(&result.output) {
                            Ok(bytes) => {
                                let decoded = serde_json::from_slice::<P::Output>(&bytes).ok();
                                (bytes, decoded)
                            }
                            Err(_) => (
                                b"{\"error\":\"unserializable_local_output\"}".to_vec(),
                                None,
                            ),
                        };
                        let output_valid = decoded.is_some();
                        let hash = store.put(&bytes)?;
                        // Pure local output is already durable in the blob store. Commit its
                        // receipt, settlement and acceptance atomically, retaining all events.
                        record_many(
                            store,
                            id,
                            &mut state,
                            vec![
                                Event::Received {
                                    node: node.id.clone(),
                                    attempt,
                                    output: json!({"sha256":hash}),
                                    output_valid,
                                    cost: NanoUsd(result.cost_nanousd),
                                    elapsed_s,
                                },
                                Event::Settled {
                                    node: node.id.clone(),
                                    attempt,
                                },
                                if output_valid {
                                    Event::Accepted {
                                        node: node.id.clone(),
                                        attempt,
                                    }
                                } else {
                                    Event::Rejected {
                                        node: node.id.clone(),
                                        attempt,
                                        reason: "invalid_local_output".into(),
                                    }
                                },
                            ],
                        )?;
                        let Some(output) = decoded else {
                            failure.get_or_insert(anyhow!(
                                "local output cannot be restored; known spend is retained"
                            ));
                            continue;
                        };
                        values[index] = Some(Arc::new(output));
                        trace.spans.push(Span {
                            node_id: node.id.clone(),
                            elapsed_s,
                            cost_nanousd: NanoUsd(result.cost_nanousd),
                        });
                        for &child in &schedule.children[index] {
                            if !enqueued.contains(&child)
                                && values[child].is_none()
                                && graph.nodes[child]
                                    .depends_on
                                    .iter()
                                    .all(|p| values[ids[p.as_str()]].is_some())
                            {
                                ready.push_back(child);
                                enqueued.insert(child);
                            }
                        }
                    }
                    Err(error) => {
                        // Do not persist arbitrary provider error text: adapters may include secrets.
                        record(
                            store,
                            id,
                            &mut state,
                            Event::Failed {
                                node: node.id.clone(),
                                attempt,
                                reason: "local_provider_failed".into(),
                            },
                        )?;
                        failure.get_or_insert(error);
                    }
                }
            }
            Ok(())
        })?;
        if let Some(error) = failure {
            return Err(error);
        }
        ensure!(
            state.control == Control::Run,
            "run paused or stopped; inspect retained progress"
        );
        if !state.completed {
            record(store, id, &mut state, Event::Completed)?;
        }
        let outputs = values
            .into_iter()
            .map(|value| {
                Arc::try_unwrap(value.context("node output missing")?)
                    .map_err(|_| anyhow!("worker retained output ownership"))
            })
            .collect::<Result<Vec<_>>>()?;
        trace.spans.sort_by_key(|span| ids[span.node_id.as_str()]);
        Ok(SwarmResult {
            outputs,
            trace,
            total_cost_nanousd: state.budget.spent.0,
            reused_nodes,
            run_id: id.into(),
        })
    }
}

pub struct OfflineProvider;
impl Provider for OfflineProvider {
    type Output = serde_json::Value;
    fn binding(&self) -> String {
        "offline-reference-v1".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(0)
    }
    fn complete(
        &self,
        task: &Task,
        dependencies: &[&Self::Output],
    ) -> Result<CompletionResult<Self::Output>> {
        Ok(CompletionResult {
            output: json!({"goal":task.goal,"input":task.input,"dependencies":dependencies.iter().map(|value|Ok(json!({"sha256":crate::store::digest(&serde_json::to_vec(value)?)}))).collect::<Result<Vec<_>>>()?}),
            cost_nanousd: 0,
        })
    }
}
