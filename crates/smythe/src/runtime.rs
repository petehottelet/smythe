use crate::seams::Sources;
use crate::store::Store;
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use smythe_core::{
    EngineError, ExecutionGraph, NanoUsd, Task,
    journal::{AttemptState, Control, Effect, Event, Projection, call_key},
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, mpsc},
    time::Duration,
};

pub struct CompletionResult<T> {
    pub output: T,
    pub cost_nanousd: u64,
}
/// Pure local work only. Failure must entail no charge or external effect.
/// Remote adapters require the separate uncertain-outcome contract before admission.
pub trait Provider: Sync {
    type Output: Serialize + DeserializeOwned + Send + Sync;
    fn effect(&self) -> Effect;
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
    sources: Sources,
}

struct Prepared<T> {
    output: Option<T>,
    hash: Option<String>,
    cost: u64,
    error: Option<anyhow::Error>,
}

/// Publication errors carry known cost; they are never zero-charge provider failures.
fn prepare<T: Serialize + DeserializeOwned>(
    result: CompletionResult<T>,
    store: &Store,
) -> Prepared<T> {
    let cost = result.cost_nanousd;
    let saved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let bytes = serde_json::to_vec(&result.output)?;
        let decoded = serde_json::from_slice::<T>(&bytes).ok();
        let hash = store.put(&bytes)?;
        Ok((hash, decoded))
    }))
    .unwrap_or_else(|_| Err(anyhow!("local output publication panicked")));
    match saved {
        Ok((hash, output)) => Prepared {
            hash: Some(hash),
            cost,
            error: output
                .is_none()
                .then(|| anyhow!("local output cannot be restored; known spend is retained")),
            output,
        },
        Err(error) => Prepared {
            output: None,
            hash: None,
            cost,
            error: Some(error),
        },
    }
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
            Err(e) if EngineError::JournalChanged.matches(&e) => {
                let next = store.current(id)?;
                ensure!(next.generation == state.generation, EngineError::StaleOwner);
                *state = next;
            }
            Err(e) => return Err(e),
        }
    }
    anyhow::bail!("journal is busy")
}

fn record_observations(
    store: &Store,
    id: &str,
    state: &mut Projection,
    events: Vec<(Event, u64)>,
) -> Result<()> {
    for _ in 0..16 {
        match store.append_observed_batch(id, state.sequence, state.generation, events.clone()) {
            Ok(next) => {
                *state = next;
                return Ok(());
            }
            Err(e) if EngineError::JournalChanged.matches(&e) => {
                let next = store.current(id)?;
                ensure!(next.generation == state.generation, EngineError::StaleOwner);
                *state = next;
            }
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
            sources: Sources::default(),
        })
    }
    pub fn with_sources(mut self, sources: Sources) -> Self {
        self.sources = sources;
        self
    }
    pub fn execute<P: Provider>(
        &self,
        graph: &ExecutionGraph,
        provider: &P,
    ) -> Result<SwarmResult<P::Output>> {
        self.execute_in(
            graph,
            provider,
            &Store::memory_with_sources(self.sources.clone())?,
            &self.sources.ids.next_id(),
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
        ensure!(
            provider.effect() == Effect::Pure,
            EngineError::UnsupportedEffect
        );
        graph.validate()?;
        // JCS-equivalent graphs must also have the same provider-visible inputs.
        let canonical: ExecutionGraph = serde_json::from_slice(&serde_jcs::to_vec(graph)?)?;
        let graph = &canonical;
        let _lease = store.lease(id)?;
        let binding = crate::store::digest(&serde_jcs::to_vec(
            &json!({"engine":"local-v2","graph":graph,"provider":provider.binding(),"effect":provider.effect(),"ceiling":NanoUsd(self.max_budget_nanousd)}),
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
        ensure!(
            state.attempts.values().all(|a| a.effect == Effect::Pure),
            EngineError::UnsupportedEffect
        );
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
                // Select a prefix against cumulative reservations, then commit all intents
                // before spawning any worker. A later unaffordable quote cannot discard it.
                let mut admissions = Vec::new();
                let mut admission_events = Vec::new();
                let mut budget = state.budget.clone();
                while failure.is_none()
                    && state.control == Control::Run
                    && active + admissions.len() < self.max_concurrency
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
                    if let Err(error) = budget.reserve(&call_key(&node.id, attempt), NanoUsd(bound))
                    {
                        if EngineError::BudgetExhausted.matches(&error)
                            && (active > 0 || !admissions.is_empty())
                        {
                            ready.push_front(index);
                        } else {
                            failure = Some(error);
                        }
                        break;
                    }
                    admissions.push((index, attempt));
                    admission_events.extend([
                        Event::Admitted {
                            node: node.id.clone(),
                            attempt,
                            bound: NanoUsd(bound),
                            effect: provider.effect(),
                        },
                        Event::Dispatched {
                            node: node.id.clone(),
                            attempt,
                        },
                    ]);
                }
                if !admissions.is_empty() {
                    match record_many(store, id, &mut state, admission_events) {
                        Ok(()) => {
                            for (index, attempt) in admissions {
                                let node = &graph.nodes[index];
                                let dependencies: Vec<Arc<P::Output>> = node
                                    .depends_on
                                    .iter()
                                    .map(|p| Arc::clone(values[ids[p.as_str()]].as_ref().unwrap()))
                                    .collect();
                                let sender = sender.clone();
                                let clock = Arc::clone(&self.sources.clock);
                                active += 1;
                                scope.spawn(move |_| {
                                    let started = clock.monotonic();
                                    let result = std::panic::catch_unwind(
                                        std::panic::AssertUnwindSafe(|| {
                                            provider.complete(
                                                &node.task,
                                                &dependencies
                                                    .iter()
                                                    .map(AsRef::as_ref)
                                                    .collect::<Vec<_>>(),
                                            )
                                        }),
                                    )
                                    .unwrap_or_else(|_| Err(anyhow!("local provider panicked")));
                                    let elapsed_s =
                                        clock.monotonic().saturating_sub(started).as_secs_f64();
                                    let occurred_at = clock.unix_ms();
                                    // Workers publish immutable blobs, but never journal transitions.
                                    let result = result.map(|result| prepare(result, store));
                                    let _ = sender.send((
                                        index,
                                        attempt,
                                        elapsed_s,
                                        occurred_at,
                                        result,
                                    ));
                                });
                            }
                        }
                        Err(error) => {
                            failure.get_or_insert(error);
                        }
                    }
                }
                if active == 0 {
                    break;
                }
                let first = match receiver.recv_timeout(Duration::from_millis(50)) {
                    Ok(result) => result,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return Err(anyhow!("worker completion channel closed")),
                };
                // Drain only completions already available: never wait for a sibling.
                let mut completions = vec![first];
                completions.extend(receiver.try_iter());
                active -= completions.len();
                let mut events = Vec::new();
                for (index, attempt, elapsed_s, occurred_at, result) in &completions {
                    let node = &graph.nodes[*index];
                    let batch = match result {
                        Ok(saved) => vec![
                            Event::Received {
                                node: node.id.clone(),
                                attempt: *attempt,
                                output: saved
                                    .hash
                                    .as_ref()
                                    .map_or(serde_json::Value::Null, |hash| json!({"sha256":hash})),
                                output_valid: saved.output.is_some(),
                                cost: NanoUsd(saved.cost),
                                elapsed_s: *elapsed_s,
                            },
                            Event::Settled {
                                node: node.id.clone(),
                                attempt: *attempt,
                            },
                            if saved.output.is_some() {
                                Event::Accepted {
                                    node: node.id.clone(),
                                    attempt: *attempt,
                                }
                            } else {
                                Event::Rejected {
                                    node: node.id.clone(),
                                    attempt: *attempt,
                                    reason: "invalid_local_output".into(),
                                }
                            },
                        ],
                        Err(_) => vec![Event::Failed {
                            node: node.id.clone(),
                            attempt: *attempt,
                            reason: "local_provider_failed".into(),
                        }],
                    };
                    events.extend(batch.into_iter().map(|event| (event, *occurred_at)));
                }
                record_observations(store, id, &mut state, events)?;
                for (index, _, elapsed_s, _, result) in completions {
                    let node = &graph.nodes[index];
                    match result {
                        Ok(saved) => {
                            let Some(output) = saved.output else {
                                failure.get_or_insert_with(|| {
                                    saved
                                        .error
                                        .unwrap_or_else(|| anyhow!("invalid local output"))
                                });
                                continue;
                            };
                            values[index] = Some(Arc::new(output));
                            trace.spans.push(Span {
                                node_id: node.id.clone(),
                                elapsed_s,
                                cost_nanousd: NanoUsd(saved.cost),
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
                            failure.get_or_insert(error);
                        }
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
    fn effect(&self) -> Effect {
        Effect::Pure
    }
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
