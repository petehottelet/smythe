use crate::{seams::Sources, store::Store};
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use smythe_core::{
    EngineError, ExecutionGraph, Limits, NanoUsd, OutputRef, Task,
    coordinator::{Command, Completion, Coordinator, Finish, Input, Outcome},
    journal::{AttemptState, Effect, Event, Projection},
};
use std::{
    collections::{HashSet, VecDeque},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

pub struct CompletionResult<T> {
    pub output: T,
    pub cost_nanousd: u64,
}
/// A dependency is a checked reference. Reading its value is explicit and measured.
pub struct Dependency<'a, T> {
    reference: OutputRef,
    store: &'a Store,
    loaded: &'a AtomicU64,
    marker: PhantomData<T>,
}
impl<T: DeserializeOwned> Dependency<'_, T> {
    pub fn reference(&self) -> &OutputRef {
        &self.reference
    }
    pub fn load(&self) -> Result<T> {
        let bytes = self.store.output(&self.reference)?;
        self.loaded
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                n.checked_add(bytes.len() as u64)
            })
            .map_err(|_| anyhow!("dependency byte counter overflow"))?;
        serde_json::from_slice(&bytes).context(EngineError::Integrity)
    }
}
/// Pure local work only. Errors must entail no charge or external effect.
pub trait Provider: Sync {
    type Output: Serialize + DeserializeOwned + Send + Sync;
    fn effect(&self) -> Effect;
    fn binding(&self) -> String;
    fn cost_bound_nanousd(&self, task: &Task) -> Result<u64>;
    fn complete(
        &self,
        task: &Task,
        dependencies: &[Dependency<'_, Self::Output>],
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeOutput {
    pub node: String,
    #[serde(flatten)]
    pub reference: OutputRef,
}
pub struct SwarmResult {
    pub outputs: Vec<NodeOutput>,
    pub trace: Tracer,
    pub total_cost_nanousd: u64,
    pub reused_nodes: usize,
    pub run_id: String,
    pub loaded_dependency_bytes: u64,
    pub dependency_bytes: u64,
    pub output_bytes: u64,
    pub overrun: bool,
    store: Store,
}
impl SwarmResult {
    pub fn load<T: DeserializeOwned>(&self, node: &str) -> Result<T> {
        let output = self
            .outputs
            .iter()
            .find(|o| o.node == node)
            .context("node output missing")?;
        serde_json::from_slice(&self.store.output(&output.reference)?)
            .context(EngineError::Integrity)
    }
    /// Explicitly materialize all values; the scheduler itself retains only references.
    pub fn load_all<T: DeserializeOwned>(&self) -> Result<Vec<T>> {
        self.outputs
            .iter()
            .map(|o| {
                serde_json::from_slice(&self.store.output(&o.reference)?)
                    .context(EngineError::Integrity)
            })
            .collect()
    }
}
pub struct Swarm {
    pool: rayon::ThreadPool,
    max_concurrency: usize,
    max_budget_nanousd: u64,
    limits: Option<Limits>,
    sources: Sources,
}
fn prepare<T: Serialize + DeserializeOwned>(
    result: CompletionResult<T>,
    store: &Store,
    limit: u64,
    elapsed_s: f64,
) -> Outcome {
    let cost = NanoUsd(result.cost_nanousd);
    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let bytes = serde_json::to_vec(&result.output)?;
        if bytes.len() as u64 > limit {
            return Ok(None);
        }
        let _: T = serde_json::from_slice(&bytes)?;
        Ok(Some(store.prepare(bytes)?))
    }))
    .unwrap_or_else(|_| Err(anyhow!("local output publication panicked")));
    match prepared {
        Ok(Some((output, inline))) => Outcome::Success {
            output,
            inline,
            cost,
            elapsed_s,
        },
        Ok(None) => Outcome::Rejected {
            cost,
            elapsed_s,
            reason: "output_too_large".into(),
        },
        Err(_) => Outcome::Rejected {
            cost,
            elapsed_s,
            reason: "invalid_local_output".into(),
        },
    }
}
fn verify_outputs<T: DeserializeOwned>(
    state: &Projection,
    store: &Store,
) -> Result<HashSet<String>> {
    let mut invalid = HashSet::new();
    for (node, a) in &state.attempts {
        if matches!(
            a.state,
            AttemptState::Accepted | AttemptState::Received | AttemptState::Settled
        ) {
            let checked = a
                .output
                .as_ref()
                .context(EngineError::Integrity)
                .and_then(|r| store.output(r))
                .and_then(|b| serde_json::from_slice::<T>(&b).context(EngineError::Integrity));
            if checked.is_err() || !a.output_valid {
                ensure!(a.state != AttemptState::Accepted, EngineError::Integrity);
                invalid.insert(node.clone());
            }
        }
    }
    Ok(invalid)
}
fn quotes<P: Provider>(state: &Projection, provider: &P) -> Vec<Option<NanoUsd>> {
    state
        .graph
        .nodes
        .iter()
        .map(|n| {
            if state
                .attempts
                .get(&n.id)
                .is_some_and(|a| a.state == AttemptState::Accepted)
            {
                Some(NanoUsd(0))
            } else {
                provider.cost_bound_nanousd(&n.task).ok().map(NanoUsd)
            }
        })
        .collect()
}
fn refresh<P: Provider>(
    coordinator: &mut Coordinator,
    records: Vec<smythe_core::journal::Record>,
    provider: &P,
    store: &Store,
) -> Result<Vec<Command>> {
    let revised = records.iter().any(|r| {
        matches!(
            r.event,
            Event::Expanded { .. } | Event::Revised { .. } | Event::Retried { .. }
        )
    });
    let mut commands = coordinator.step(Input::Foreign(records))?;
    if revised {
        coordinator.set_quotes(quotes(&coordinator.state, provider))?;
    }
    if coordinator.needs_recovery() {
        commands.extend(coordinator.step(Input::Recovered {
            invalid_outputs: verify_outputs::<P::Output>(&coordinator.state, store)?,
        })?);
    }
    Ok(commands)
}
impl Swarm {
    pub fn new(max_concurrency: usize, max_budget_nanousd: u64) -> Result<Self> {
        ensure!(
            (1..=64).contains(&max_concurrency),
            EngineError::InvalidCommand
        );
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(max_concurrency)
                .build()?,
            max_concurrency,
            max_budget_nanousd,
            limits: None,
            sources: Sources::default(),
        })
    }
    pub fn with_sources(mut self, sources: Sources) -> Self {
        self.sources = sources;
        self
    }
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = Some(limits);
        self
    }
    pub fn execute<P: Provider>(
        &self,
        graph: &ExecutionGraph,
        provider: &P,
    ) -> Result<SwarmResult> {
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
    ) -> Result<SwarmResult> {
        ensure!(
            provider.effect() == Effect::Pure,
            EngineError::UnsupportedEffect
        );
        let limits = self.limits.clone().unwrap_or_default();
        if !resume {
            graph
                .schedule_with(&limits)
                .context(EngineError::InvalidCommand)?;
        }
        let canonical: ExecutionGraph = serde_json::from_slice(&serde_jcs::to_vec(graph)?)?;
        let binding = crate::store::digest(&serde_jcs::to_vec(
            &json!({"engine":"local-v3","nodes":canonical.nodes,"provider":provider.binding(),"effect":provider.effect()}),
        )?);
        let _lease = store.lease(id)?;
        let history = store.history(id)?;
        ensure!(resume || history.is_empty(), EngineError::RunExists);
        ensure!(!resume || !history.is_empty(), EngineError::RunMissing);
        let mut state = if history.is_empty() {
            let ack = store.append(
                id,
                0,
                0,
                Event::Started {
                    binding: binding.clone(),
                    graph: canonical,
                    ceiling: NanoUsd(self.max_budget_nanousd),
                    limits,
                },
            )?;
            Projection::replay(&ack.records)?
        } else {
            Projection::replay(&history).context(EngineError::Integrity)?
        };
        graph
            .schedule_with(&state.limits)
            .context(EngineError::InvalidCommand)?;
        ensure!(
            state.binding == binding,
            anyhow!(EngineError::InvalidCommand).context("run binding differs")
        );
        if resume {
            ensure!(state.budget.ceiling==NanoUsd(self.max_budget_nanousd),anyhow!(EngineError::InvalidCommand).context("resume uses the journaled budget; change it with smythe control <id> budget <nanousd>"));
            ensure!(
                self.limits
                    .as_ref()
                    .is_none_or(|limits| limits == &state.limits),
                anyhow!(EngineError::InvalidCommand).context("resume uses the journaled limits")
            );
        }
        ensure!(
            state.attempts.values().all(|a| a.effect == Effect::Pure),
            EngineError::UnsupportedEffect
        );
        let invalid = verify_outputs::<P::Output>(&state, store)?;
        let reused_nodes = state
            .attempts
            .iter()
            .filter(|(n, a)| {
                !invalid.contains(*n)
                    && matches!(
                        a.state,
                        AttemptState::Accepted | AttemptState::Received | AttemptState::Settled
                    )
            })
            .count();
        if state.disposition.is_none() {
            let generation = state
                .generation
                .checked_add(1)
                .context("generation overflow")?;
            let ack = store.append(
                id,
                state.sequence,
                generation,
                Event::Claimed { generation },
            )?;
            state.apply_batch(&ack.records)?;
        }
        let quoted = quotes(&state, provider);
        let mut coordinator = Coordinator::new(
            state,
            quoted,
            self.max_concurrency,
            self.sources.clock.unix_ms(),
        )?;
        // Quote callbacks may have issued an operator control. Refresh before admission.
        let foreign = store.changes_since(id, coordinator.state.sequence)?;
        if !foreign.is_empty() {
            let revised = foreign.iter().any(|r| {
                matches!(
                    r.event,
                    Event::Expanded { .. } | Event::Revised { .. } | Event::Retried { .. }
                )
            });
            coordinator.step(Input::Foreign(foreign))?;
            if revised {
                coordinator.set_quotes(quotes(&coordinator.state, provider))?;
            }
        }
        let mut commands: VecDeque<_> = coordinator
            .step(Input::Recovered {
                invalid_outputs: invalid,
            })?
            .into();
        let (sender, receiver) = mpsc::sync_channel::<Completion>(self.max_concurrency);
        let (wake_tx, wake_rx) = mpsc::channel();
        let loaded = AtomicU64::new(0);
        let mut trace = Tracer::default();
        let finish = self.pool.in_place_scope(|scope| -> Result<Finish> {
            loop {
                while let Some(command) = commands.pop_front() {
                    match command {
                        Command::Append { events, blobs } => {
                            match store.append_observed_batch(
                                id,
                                coordinator.state.sequence,
                                coordinator.state.generation,
                                events,
                                blobs,
                            ) {
                                Ok(ack) => commands
                                    .extend(coordinator.step(Input::Committed(ack.records))?),
                                Err(error) if EngineError::JournalChanged.matches(&error) => {
                                    let foreign =
                                        store.changes_since(id, coordinator.state.sequence)?;
                                    commands.extend(refresh(
                                        &mut coordinator,
                                        foreign,
                                        provider,
                                        store,
                                    )?);
                                }
                                Err(error) => return Err(error),
                            }
                        }
                        Command::Dispatch {
                            node,
                            attempt,
                            dependencies,
                        } => {
                            let graph = Arc::clone(&coordinator.state.graph);
                            let sender = sender.clone();
                            let wake = wake_tx.clone();
                            let clock = Arc::clone(&self.sources.clock);
                            let loaded = &loaded;
                            let limit = coordinator.state.limits.max_output_bytes;
                            scope.spawn(move |_| {
                                let started = clock.monotonic();
                                let handles: Vec<_> = dependencies
                                    .into_iter()
                                    .map(|reference| Dependency {
                                        reference,
                                        store,
                                        loaded,
                                        marker: PhantomData,
                                    })
                                    .collect();
                                let result =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        provider.complete(&graph.nodes[node].task, &handles)
                                    }))
                                    .unwrap_or_else(|_| Err(anyhow!("local provider panicked")));
                                let elapsed_s =
                                    clock.monotonic().saturating_sub(started).as_secs_f64();
                                let occurred_at = clock.unix_ms();
                                let outcome = match result {
                                    Ok(result) => prepare(result, store, limit, elapsed_s),
                                    Err(_) => Outcome::Failed {
                                        reason: "local_provider_failed".into(),
                                    },
                                };
                                let _ = sender.send(Completion {
                                    node,
                                    attempt,
                                    outcome,
                                    occurred_at,
                                });
                                let _ = wake.send(());
                            });
                        }
                        Command::Finish(finish) => return Ok(finish),
                    }
                }
                let completed: Vec<_> = receiver.try_iter().collect();
                if !completed.is_empty() {
                    for c in &completed {
                        if let Outcome::Success {
                            cost, elapsed_s, ..
                        }
                        | Outcome::Rejected {
                            cost, elapsed_s, ..
                        } = &c.outcome
                        {
                            trace.spans.push(Span {
                                node_id: coordinator.state.graph.nodes[c.node].id.clone(),
                                elapsed_s: *elapsed_s,
                                cost_nanousd: *cost,
                            });
                        }
                    }
                    commands.extend(coordinator.step(Input::Completions(completed))?);
                    continue;
                }
                let foreign = store.changes_since(id, coordinator.state.sequence)?;
                if !foreign.is_empty() {
                    commands.extend(refresh(&mut coordinator, foreign, provider, store)?);
                }
                if commands.is_empty() {
                    commands.extend(coordinator.step(Input::Tick {
                        now_ms: self.sources.clock.unix_ms(),
                    })?);
                }
                if commands.is_empty() {
                    ensure!(
                        coordinator.active() > 0,
                        "coordinator has no work or disposition"
                    );
                    let foreign = self.sources.controls.poll(
                        store,
                        id,
                        coordinator.state.sequence,
                        &wake_rx,
                    )?;
                    if !foreign.is_empty() {
                        commands.extend(refresh(&mut coordinator, foreign, provider, store)?);
                    }
                    if commands.is_empty() {
                        commands.extend(coordinator.step(Input::Tick {
                            now_ms: self.sources.clock.unix_ms(),
                        })?);
                    }
                }
            }
        })?;
        match finish {
            Finish::Completed => {}
            Finish::Paused => anyhow::bail!(EngineError::Paused),
            Finish::Stopped => anyhow::bail!(EngineError::Stopped),
            Finish::Failed => anyhow::bail!(EngineError::ProviderFailed),
            Finish::BudgetBlocked => anyhow::bail!(EngineError::BudgetExhausted),
            Finish::QuoteBlocked => anyhow::bail!(EngineError::QuoteBlocked),
        }
        let state = coordinator.state;
        let outputs = state
            .graph
            .nodes
            .iter()
            .map(|n| {
                Ok(NodeOutput {
                    node: n.id.clone(),
                    reference: state
                        .attempts
                        .get(&n.id)
                        .and_then(|a| a.output.clone())
                        .context(EngineError::Integrity)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        trace
            .spans
            .sort_by_key(|span| state.node_index(&span.node_id).unwrap());
        Ok(SwarmResult {
            outputs,
            trace,
            total_cost_nanousd: state.budget.spent.0,
            reused_nodes,
            run_id: id.into(),
            loaded_dependency_bytes: loaded.load(Ordering::Relaxed),
            dependency_bytes: state.dependency_bytes,
            output_bytes: state.output_bytes,
            overrun: state.overrun,
            store: store.clone(),
        })
    }
}
pub struct OfflineProvider;
impl Provider for OfflineProvider {
    type Output = serde_json::Value;
    fn effect(&self) -> Effect {
        Effect::Pure
    }
    fn binding(&self) -> String {
        "offline-reference-v2".into()
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(0)
    }
    fn complete(
        &self,
        task: &Task,
        dependencies: &[Dependency<'_, Self::Output>],
    ) -> Result<CompletionResult<Self::Output>> {
        Ok(CompletionResult {
            output: json!({"goal":task.goal,"input":task.input,"dependencies":dependencies.iter().map(Dependency::reference).collect::<Vec<_>>()}),
            cost_nanousd: 0,
        })
    }
}
