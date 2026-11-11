//! Deterministic scheduling decisions. The shell owns clocks, threads and storage.
use crate::journal::{
    Admission, AttemptState, CloseReason, Disposition, Effect, Event, Projection, Record,
};
use crate::{EngineError, NanoUsd, OutputRef};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct Blob {
    pub reference: OutputRef,
    pub bytes: Arc<[u8]>,
}
#[derive(Clone, Debug)]
pub enum Outcome {
    Success {
        output: OutputRef,
        inline: Option<Blob>,
        cost: NanoUsd,
        elapsed_s: f64,
    },
    Rejected {
        cost: NanoUsd,
        elapsed_s: f64,
        reason: String,
    },
    Failed {
        reason: String,
    },
}
#[derive(Clone, Debug)]
pub struct Completion {
    pub node: usize,
    pub attempt: u64,
    pub outcome: Outcome,
    pub occurred_at: u64,
}
pub enum Input {
    Recovered {
        invalid_outputs: HashSet<String>,
    },
    Completions(Vec<Completion>),
    /// A rejected append refreshes the same owner's journal before a fresh decision.
    Foreign(Vec<Record>),
    Committed(Vec<Record>),
    Tick {
        now_ms: u64,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Completed,
    Stopped,
    Failed,
    Paused,
    BudgetBlocked,
    QuoteBlocked,
}
#[derive(Clone, Debug)]
pub enum Command {
    Append {
        events: Vec<(Event, u64)>,
        blobs: Vec<Blob>,
    },
    Dispatch {
        node: usize,
        attempt: u64,
        dependencies: Vec<OutputRef>,
    },
    Finish(Finish),
}
enum Purpose {
    Recovery,
    Admissions(Vec<(usize, u64)>),
    Observations(Vec<Completion>),
    Other,
}
struct Pending {
    events: Vec<(Event, u64)>,
    purpose: Purpose,
}

pub struct Coordinator {
    pub state: Projection,
    quotes: Vec<Option<NanoUsd>>,
    workers: usize,
    now_ms: u64,
    ready: VecDeque<usize>,
    remaining: Vec<usize>,
    children: Vec<Vec<usize>>,
    enqueued: Vec<bool>,
    active: HashMap<usize, u64>,
    completions: Vec<Completion>,
    pending: Option<Pending>,
    recovered: bool,
    failed: bool,
    fault: Option<String>,
}
impl Coordinator {
    pub fn new(
        state: Projection,
        quotes: Vec<Option<NanoUsd>>,
        workers: usize,
        now_ms: u64,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&workers), "invalid concurrency");
        ensure!(
            quotes.len() == state.graph.nodes.len(),
            "quote count differs"
        );
        let mut this = Self {
            state,
            quotes,
            workers,
            now_ms,
            ready: VecDeque::new(),
            remaining: vec![],
            children: vec![],
            enqueued: vec![],
            active: HashMap::new(),
            completions: vec![],
            pending: None,
            recovered: false,
            failed: false,
            fault: None,
        };
        this.rebuild_schedule()?;
        Ok(this)
    }
    /// A shell can quote newly recorded expansion nodes without placing I/O here.
    pub fn set_quotes(&mut self, quotes: Vec<Option<NanoUsd>>) -> Result<()> {
        ensure!(
            quotes.len() == self.state.graph.nodes.len(),
            "quote count differs"
        );
        self.quotes = quotes;
        Ok(())
    }
    pub fn active(&self) -> usize {
        self.active.len()
    }
    pub fn needs_recovery(&self) -> bool {
        !self.recovered
    }
    pub fn step(&mut self, input: Input) -> Result<Vec<Command>> {
        match input {
            Input::Tick { now_ms } => {
                self.now_ms = self.now_ms.max(now_ms);
            }
            Input::Recovered { invalid_outputs } => {
                ensure!(!self.recovered, "coordinator already recovered");
                self.recovered = true;
                if self.state.disposition.is_none() {
                    let mut events = vec![];
                    for (node, a) in &self.state.attempts {
                        match a.state {
                            AttemptState::Admitted | AttemptState::Dispatched => {
                                events.push(Event::LocalReplay {
                                    node: node.clone(),
                                    attempt: a.number,
                                    revision_base: Some(a.revision_base),
                                })
                            }
                            AttemptState::Received | AttemptState::Settled => {
                                if a.state == AttemptState::Received {
                                    events.push(Event::Settled {
                                        node: node.clone(),
                                        attempt: a.number,
                                    });
                                }
                                events.push(if a.output_valid && !invalid_outputs.contains(node) {
                                    Event::Accepted {
                                        node: node.clone(),
                                        attempt: a.number,
                                    }
                                } else {
                                    Event::Rejected {
                                        node: node.clone(),
                                        attempt: a.number,
                                        reason: "invalid_local_output".into(),
                                    }
                                });
                            }
                            _ => {}
                        }
                    }
                    if !events.is_empty() {
                        return Ok(self.append(
                            events.into_iter().map(|e| (e, self.now_ms)).collect(),
                            vec![],
                            Purpose::Recovery,
                        ));
                    }
                }
            }
            Input::Completions(results) => {
                for result in results {
                    self.now_ms = self.now_ms.max(result.occurred_at);
                    ensure!(
                        self.active.remove(&result.node) == Some(result.attempt),
                        "unexpected or duplicate completion"
                    );
                    self.completions.push(result);
                }
            }
            Input::Committed(records) => {
                let pending = self
                    .pending
                    .take()
                    .context("unexpected commit acknowledgement")?;
                ensure!(
                    records.len() == pending.events.len()
                        && records
                            .iter()
                            .zip(&pending.events)
                            .all(|(r, (e, t))| &r.event == e && r.occurred_at_unix_ms == *t),
                    "commit acknowledgement differs"
                );
                self.state.apply_batch(&records)?;
                if let Purpose::Recovery = pending.purpose {
                    self.rebuild_schedule()?;
                } else {
                    self.observe_progress(&records)?;
                }
                if let Purpose::Admissions(admissions) = pending.purpose {
                    let mut commands = vec![];
                    for (node, attempt) in admissions {
                        let dependencies = self.state.graph.nodes[node]
                            .depends_on
                            .iter()
                            .map(|id| {
                                self.state
                                    .attempts
                                    .get(id)
                                    .and_then(|a| a.output.clone())
                                    .context("dependency output missing")
                            })
                            .collect::<Result<Vec<_>>>()?;
                        self.active.insert(node, attempt);
                        commands.push(Command::Dispatch {
                            node,
                            attempt,
                            dependencies,
                        });
                    }
                    // Dispatch the acknowledged intents before asking the shell to append again.
                    return Ok(commands);
                }
            }
            Input::Foreign(records) => {
                if let Some(pending) = self.pending.take() {
                    match pending.purpose {
                        Purpose::Admissions(nodes) => {
                            for (i, _) in nodes.into_iter().rev() {
                                self.ready.push_front(i);
                            }
                        }
                        Purpose::Observations(mut results) => {
                            results.append(&mut self.completions);
                            self.completions = results;
                        }
                        Purpose::Recovery => {
                            self.recovered = false;
                        }
                        Purpose::Other => {}
                    }
                }
                let owner = self.state.generation;
                self.state.apply_batch(&records)?;
                ensure!(self.state.generation == owner, EngineError::StaleOwner);
                if records.iter().any(|r| {
                    matches!(
                        r.event,
                        Event::Retried { .. } | Event::Expanded { .. } | Event::Revised { .. }
                    )
                }) {
                    ensure!(self.active.is_empty(), "revision while workers are active");
                    self.rebuild_schedule()?;
                    return Ok(vec![]);
                } else {
                    self.observe_progress(&records)?;
                }
                if !self.recovered {
                    return Ok(vec![]);
                }
            }
        }
        self.drive()
    }
    fn append(
        &mut self,
        events: Vec<(Event, u64)>,
        blobs: Vec<Blob>,
        purpose: Purpose,
    ) -> Vec<Command> {
        self.pending = Some(Pending {
            events: events.clone(),
            purpose,
        });
        vec![Command::Append { events, blobs }]
    }
    fn rebuild_schedule(&mut self) -> Result<()> {
        let schedule = self.state.graph.schedule_with(&self.state.limits)?;
        self.children = schedule.children;
        self.remaining = schedule.remaining;
        self.enqueued = vec![false; self.state.graph.nodes.len()];
        self.ready.clear();
        for (i, n) in self.state.graph.nodes.iter().enumerate() {
            if self
                .state
                .attempts
                .get(&n.id)
                .is_some_and(|a| a.state == AttemptState::Accepted)
            {
                self.enqueued[i] = true;
                for &child in &self.children[i] {
                    self.remaining[child] -= 1;
                }
            }
        }
        for (i, n) in self.state.graph.nodes.iter().enumerate() {
            if self.remaining[i] == 0
                && self.state.attempts.get(&n.id).is_none_or(|a| {
                    matches!(a.state, AttemptState::Replayed | AttemptState::Invalidated)
                })
            {
                self.ready.push_back(i);
                self.enqueued[i] = true;
            }
        }
        self.failed = self.state.failure_reason.is_some()
            || self
                .state
                .attempts
                .values()
                .any(|a| matches!(a.state, AttemptState::Failed | AttemptState::Rejected));
        Ok(())
    }
    fn observe_progress(&mut self, records: &[Record]) -> Result<()> {
        for record in records {
            match &record.event {
                Event::Accepted { node, .. } => {
                    let index = self.state.node_index(node)?;
                    for &child in &self.children[index] {
                        self.remaining[child] = self.remaining[child]
                            .checked_sub(1)
                            .context("dependency count underflow")?;
                        if self.remaining[child] == 0 && !self.enqueued[child] {
                            self.enqueued[child] = true;
                            self.ready.push_back(child);
                        }
                    }
                }
                Event::Failed { .. } | Event::Rejected { .. } | Event::RunFailed { .. } => {
                    self.failed = true;
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn drive(&mut self) -> Result<Vec<Command>> {
        if self.pending.is_some() || !self.recovered {
            return Ok(vec![]);
        }
        if let Some(disposition) = self.state.disposition {
            return Ok(vec![Command::Finish(match disposition {
                Disposition::Completed => Finish::Completed,
                Disposition::Stopped => Finish::Stopped,
                Disposition::Failed => Finish::Failed,
            })]);
        }
        if !self.state.deadline_closed
            && self
                .state
                .limits
                .wall_deadline_ms
                .is_some_and(|d| self.now_ms >= d)
        {
            return Ok(self.append(
                vec![(
                    Event::AdmissionClosed {
                        reason: CloseReason::Deadline,
                    },
                    self.now_ms,
                )],
                vec![],
                Purpose::Other,
            ));
        }
        if !self.completions.is_empty() {
            let results = std::mem::take(&mut self.completions);
            let mut events = vec![];
            let mut blobs = vec![];
            let mut total = self.state.output_bytes;
            for result in &results {
                let node = &self.state.graph.nodes[result.node].id;
                let mut outcome = result.outcome.clone();
                if let Outcome::Success {
                    output,
                    cost,
                    elapsed_s,
                    ..
                } = &outcome
                {
                    if output.bytes > self.state.limits.max_output_bytes
                        || total
                            .checked_add(output.bytes)
                            .is_none_or(|n| n > self.state.limits.max_total_output_bytes)
                    {
                        outcome = Outcome::Rejected {
                            cost: *cost,
                            elapsed_s: *elapsed_s,
                            reason: "output_too_large".into(),
                        };
                    } else {
                        total += output.bytes;
                    }
                }
                let batch = match outcome {
                    Outcome::Failed { reason } => vec![Event::Failed {
                        node: node.clone(),
                        attempt: result.attempt,
                        reason,
                    }],
                    other => {
                        let (output, valid, cost, elapsed_s, reason) = match other {
                            Outcome::Success {
                                output,
                                inline,
                                cost,
                                elapsed_s,
                            } => {
                                if let Some(blob) = inline {
                                    blobs.push(blob);
                                }
                                (Some(output), true, cost, elapsed_s, String::new())
                            }
                            Outcome::Rejected {
                                cost,
                                elapsed_s,
                                reason,
                            } => (None, false, cost, elapsed_s, reason),
                            _ => unreachable!(),
                        };
                        vec![
                            Event::Received {
                                node: node.clone(),
                                attempt: result.attempt,
                                output,
                                output_valid: valid,
                                cost,
                                elapsed_s,
                            },
                            Event::Settled {
                                node: node.clone(),
                                attempt: result.attempt,
                            },
                            if valid {
                                Event::Accepted {
                                    node: node.clone(),
                                    attempt: result.attempt,
                                }
                            } else {
                                Event::Rejected {
                                    node: node.clone(),
                                    attempt: result.attempt,
                                    reason,
                                }
                            },
                        ]
                    }
                };
                events.extend(batch.into_iter().map(|event| (event, result.occurred_at)));
            }
            return Ok(self.append(events, blobs, Purpose::Observations(results)));
        }
        if let Some(reason) = &self.fault
            && self.state.failure_reason.is_none()
        {
            return Ok(self.append(
                vec![(
                    Event::RunFailed {
                        reason: reason.clone(),
                    },
                    self.now_ms,
                )],
                vec![],
                Purpose::Other,
            ));
        }
        let mut admissions = vec![];
        let mut events = vec![];
        let mut extra = NanoUsd(0);
        while !self.failed
            && self.fault.is_none()
            && self.state.admission == Admission::Open
            && self.active.len() + admissions.len() < self.workers
        {
            let Some(&index) = self.ready.front() else {
                break;
            };
            let node = &self.state.graph.nodes[index];
            let Some(bound) = *self
                .quotes
                .get(index)
                .context("missing quote for recorded expansion")?
            else {
                if self.active.is_empty() && admissions.is_empty() {
                    if self.state.quote_blocked {
                        return Ok(vec![Command::Finish(Finish::QuoteBlocked)]);
                    }
                    return Ok(self.append(
                        vec![(
                            Event::QuoteBlocked {
                                node: node.id.clone(),
                            },
                            self.now_ms,
                        )],
                        vec![],
                        Purpose::Other,
                    ));
                }
                break;
            };
            let attempt = self
                .state
                .attempts
                .get(&node.id)
                .map_or(Some(1), |a| a.number.checked_add(1))
                .context("attempt overflow")?;
            let revision_base = self
                .state
                .attempts
                .get(&node.id)
                .map_or(0, |a| a.revision_base);
            if attempt
                .checked_sub(revision_base)
                .context("invalid revision attempt base")?
                > self.state.limits.max_attempts_per_node
            {
                self.fault = Some("attempt_limit".into());
                break;
            }
            if !self.state.budget.fits(extra, bound) {
                if self.active.is_empty() && admissions.is_empty() {
                    if self.state.budget_blocked {
                        return Ok(vec![Command::Finish(Finish::BudgetBlocked)]);
                    }
                    return Ok(self.append(
                        vec![(
                            Event::BudgetBlocked {
                                node: node.id.clone(),
                                bound,
                            },
                            self.now_ms,
                        )],
                        vec![],
                        Purpose::Other,
                    ));
                }
                break;
            }
            self.ready.pop_front();
            let dependency_bytes = node.depends_on.iter().try_fold(0u64, |sum, id| {
                sum.checked_add(
                    self.state
                        .attempts
                        .get(id)
                        .and_then(|a| a.output.as_ref())
                        .context("missing dependency output")?
                        .bytes,
                )
                .context("dependency bytes overflow")
            })?;
            events.extend([
                (
                    Event::Admitted {
                        node: node.id.clone(),
                        attempt,
                        bound,
                        effect: Effect::Pure,
                    },
                    self.now_ms,
                ),
                (
                    Event::Dispatched {
                        node: node.id.clone(),
                        attempt,
                        dependency_bytes,
                    },
                    self.now_ms,
                ),
            ]);
            extra = extra.checked_add(bound)?;
            admissions.push((index, attempt));
        }
        if !admissions.is_empty() {
            return Ok(self.append(events, vec![], Purpose::Admissions(admissions)));
        }
        if self.fault.is_some() && self.state.failure_reason.is_none() {
            return self.drive();
        }
        if !self.active.is_empty() {
            return Ok(vec![]);
        }
        let disposition = if self.state.accepted_nodes == self.state.graph.nodes.len() {
            Disposition::Completed
        } else if self.failed {
            Disposition::Failed
        } else {
            match self.state.admission {
                Admission::Paused => return Ok(vec![Command::Finish(Finish::Paused)]),
                Admission::Closed { .. } => Disposition::Stopped,
                Admission::Open => anyhow::bail!(EngineError::Integrity),
            }
        };
        Ok(self.append(
            vec![(Event::Ended { disposition }, self.now_ms)],
            vec![],
            Purpose::Other,
        ))
    }
}
