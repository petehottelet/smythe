//! Schema-3 transitions. Atomic batches mutate in place and undo only touched entries.
use crate::money::LedgerUndo;
use crate::{EngineError, ExecutionGraph, Limits, NanoUsd, Node, OutputRef, Sentinel};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

pub const JOURNAL_SCHEMA: u32 = 3;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Pure,
    External,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Run,
    Execution,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    Run,
    Pause,
    Stop,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Completed,
    Stopped,
    Failed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloseReason {
    OperatorStop,
    Overrun,
    Deadline,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Admission {
    Open,
    Paused,
    Closed { reason: CloseReason },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Started {
        binding: String,
        graph: ExecutionGraph,
        ceiling: NanoUsd,
        limits: Limits,
    },
    Claimed {
        generation: u64,
    },
    Control {
        control: Control,
    },
    CeilingChanged {
        ceiling: NanoUsd,
    },
    AdmissionClosed {
        reason: CloseReason,
    },
    BudgetBlocked {
        node: String,
        bound: NanoUsd,
    },
    RunFailed {
        reason: String,
    },
    Admitted {
        node: String,
        attempt: u64,
        bound: NanoUsd,
        effect: Effect,
    },
    Dispatched {
        node: String,
        attempt: u64,
        dependency_bytes: u64,
    },
    Received {
        node: String,
        attempt: u64,
        output: Option<OutputRef>,
        output_valid: bool,
        cost: NanoUsd,
        elapsed_s: f64,
    },
    Settled {
        node: String,
        attempt: u64,
    },
    Accepted {
        node: String,
        attempt: u64,
    },
    Rejected {
        node: String,
        attempt: u64,
        reason: String,
    },
    Failed {
        node: String,
        attempt: u64,
        reason: String,
    },
    LocalReplay {
        node: String,
        attempt: u64,
    },
    Retried {
        node: String,
        attempt: u64,
    },
    Expanded {
        parent_revision: u32,
        revision: u32,
        input_manifest_sha256: String,
        nodes: Vec<Node>,
    },
    Revised {
        revision: u32,
        reason: String,
        invalidated: Vec<String>,
    },
    Ended {
        disposition: Disposition,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub sequence: u64,
    pub generation: u64,
    pub revision: u32,
    pub run_id: String,
    pub event_id: String,
    pub node: Option<String>,
    pub attempt: Option<u64>,
    pub scope: Scope,
    pub phase: Option<String>,
    pub transaction_start: u64,
    pub occurred_at_unix_ms: u64,
    pub recorded_at_unix_ms: u64,
    pub event: Event,
}
impl Event {
    pub fn identity(&self) -> (Option<&str>, Option<u64>, Scope) {
        match self {
            Self::Admitted { node, attempt, .. }
            | Self::Dispatched { node, attempt, .. }
            | Self::Received { node, attempt, .. }
            | Self::Settled { node, attempt }
            | Self::Accepted { node, attempt }
            | Self::Rejected { node, attempt, .. }
            | Self::Failed { node, attempt, .. }
            | Self::LocalReplay { node, attempt }
            | Self::Retried { node, attempt } => (Some(node), Some(*attempt), Scope::Execution),
            _ => (None, None, Scope::Run),
        }
    }
    pub fn revision(&self, current: u32) -> u32 {
        match self {
            Self::Expanded { revision, .. } | Self::Revised { revision, .. } => *revision,
            _ => current,
        }
    }
}
impl Record {
    pub fn new(
        run_id: &str,
        sequence: u64,
        generation: u64,
        transaction_start: u64,
        event: Event,
        occurred_at_unix_ms: u64,
        recorded_at_unix_ms: u64,
    ) -> Self {
        let (node, attempt, scope) = event.identity();
        Self {
            schema: JOURNAL_SCHEMA,
            sequence,
            generation,
            revision: 0,
            run_id: run_id.into(),
            event_id: format!("{run_id}/{sequence}"),
            node: node.map(str::to_owned),
            attempt,
            scope,
            phase: None,
            transaction_start,
            occurred_at_unix_ms,
            recorded_at_unix_ms,
            event,
        }
    }
    pub fn in_revision(mut self, revision: u32, phase: Option<String>) -> Self {
        self.revision = revision;
        self.phase = phase;
        self
    }
    fn validate_header(&self) -> Result<()> {
        let (node, attempt, scope) = self.event.identity();
        ensure!(
            !self.run_id.is_empty()
                && self.event_id == format!("{}/{}", self.run_id, self.sequence)
                && self.node.as_deref() == node
                && self.attempt == attempt
                && self.scope == scope
                && self.transaction_start > 0
                && self.transaction_start <= self.sequence,
            "journal event identity differs"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Admitted,
    Dispatched,
    Received,
    Settled,
    Accepted,
    Rejected,
    Failed,
    Replayed,
    Invalidated,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub revision: u32,
    pub number: u64,
    pub effect: Effect,
    pub state: AttemptState,
    pub output: Option<OutputRef>,
    pub output_valid: bool,
    pub cost: Option<NanoUsd>,
    pub elapsed_s: f64,
    pub dependency_bytes: u64,
    pub reason: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Projection {
    pub run_id: String,
    pub sequence: u64,
    pub generation: u64,
    pub revision: u32,
    pub binding: String,
    pub graph: Arc<ExecutionGraph>,
    #[serde(skip)]
    ids: Arc<HashMap<String, usize>>,
    pub limits: Limits,
    pub control: Control,
    pub admission: Admission,
    pub disposition: Option<Disposition>,
    pub overrun: bool,
    pub deadline_closed: bool,
    pub budget_blocked: bool,
    pub budget: Sentinel,
    pub attempts: BTreeMap<String, Attempt>,
    pub accepted_nodes: usize,
    pub output_bytes: u64,
    pub dependency_bytes: u64,
    pub last_progress_unix_ms: Option<u64>,
    pub failure_reason: Option<String>,
}
type SharedGraph = (Arc<ExecutionGraph>, Arc<HashMap<String, usize>>);
struct Undo {
    sequence: u64,
    generation: u64,
    revision: u32,
    control: Control,
    admission: Admission,
    disposition: Option<Disposition>,
    overrun: bool,
    deadline_closed: bool,
    budget_blocked: bool,
    accepted_nodes: usize,
    output_bytes: u64,
    dependency_bytes: u64,
    last_progress: Option<u64>,
    ledger: LedgerUndo,
    attempts: Vec<(String, Option<Attempt>)>,
    graph: Option<SharedGraph>,
    failure_reason: Option<String>,
}
pub fn call_key(revision: u32, node: &str, attempt: u64) -> String {
    format!("{revision}/{node}/{attempt}")
}

impl Projection {
    pub fn node_index(&self, id: &str) -> Result<usize> {
        self.ids.get(id).copied().context("unknown node")
    }
    pub fn node(&self, id: &str) -> Result<&Node> {
        Ok(&self.graph.nodes[self.node_index(id)?])
    }
    pub fn completed(&self) -> bool {
        self.disposition == Some(Disposition::Completed)
    }
    pub fn status(&self) -> &'static str {
        match self.disposition {
            Some(Disposition::Completed) => "completed",
            Some(Disposition::Failed) => "failed",
            Some(Disposition::Stopped) => "stopped",
            None => match self.admission {
                Admission::Paused => "paused",
                Admission::Closed { .. } => "draining",
                Admission::Open if self.budget_blocked => "budget_blocked",
                _ => "running",
            },
        }
    }
    pub fn replay(records: &[Record]) -> Result<Self> {
        let first = records.first().ok_or(EngineError::RunMissing)?;
        ensure!(
            first.schema == JOURNAL_SCHEMA
                && first.sequence == 1
                && first.generation == 0
                && first.revision == 0
                && first.phase.is_none(),
            "invalid initial journal record"
        );
        let Event::Started {
            binding,
            graph,
            ceiling,
            limits,
        } = &first.event
        else {
            anyhow::bail!("journal must start with a run")
        };
        first.validate_header()?;
        graph.schedule_with(limits)?;
        let mut state = Self {
            run_id: first.run_id.clone(),
            sequence: 1,
            generation: 0,
            revision: 0,
            binding: binding.clone(),
            graph: Arc::new(graph.clone()),
            ids: Arc::new(Self::index(graph)),
            limits: limits.clone(),
            control: Control::Run,
            admission: Admission::Open,
            disposition: None,
            overrun: false,
            deadline_closed: false,
            budget_blocked: false,
            budget: Sentinel::new(*ceiling),
            attempts: BTreeMap::new(),
            accepted_nodes: 0,
            output_bytes: 0,
            dependency_bytes: 0,
            last_progress_unix_ms: None,
            failure_reason: None,
        };
        for record in &records[1..] {
            state.transition(record)?;
        }
        Ok(state)
    }
    fn index(graph: &ExecutionGraph) -> HashMap<String, usize> {
        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect()
    }
    pub fn apply(&mut self, record: &Record) -> Result<()> {
        self.apply_batch(std::slice::from_ref(record))
    }
    pub fn apply_batch(&mut self, records: &[Record]) -> Result<()> {
        self.transact(records, || Ok(()))
    }
    /// The commit closure runs only after validation. Failure restores touched entries.
    pub fn transact<T>(
        &mut self,
        records: &[Record],
        commit: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut undo = Vec::with_capacity(records.len());
        for record in records {
            undo.push(self.checkpoint(record));
            if let Err(error) = self.transition(record) {
                for old in undo.into_iter().rev() {
                    self.restore(old);
                }
                return Err(error);
            }
        }
        match commit() {
            Ok(value) => Ok(value),
            Err(error) => {
                for old in undo.into_iter().rev() {
                    self.restore(old);
                }
                Err(error)
            }
        }
    }
    fn checkpoint(&self, record: &Record) -> Undo {
        let targets = match &record.event {
            Event::Revised { invalidated, .. } => self.descendants(invalidated).unwrap_or_default(),
            _ => record.node.iter().cloned().collect(),
        };
        let key = record
            .node
            .as_ref()
            .zip(record.attempt)
            .map(|(node, attempt)| {
                call_key(
                    self.attempts
                        .get(node)
                        .map_or(self.revision, |a| a.revision),
                    node,
                    attempt,
                )
            })
            .unwrap_or_default();
        let key = if let Event::Admitted { node, attempt, .. } = &record.event {
            call_key(self.revision, node, *attempt)
        } else {
            key
        };
        Undo {
            sequence: self.sequence,
            generation: self.generation,
            revision: self.revision,
            control: self.control,
            admission: self.admission,
            disposition: self.disposition,
            overrun: self.overrun,
            deadline_closed: self.deadline_closed,
            budget_blocked: self.budget_blocked,
            accepted_nodes: self.accepted_nodes,
            output_bytes: self.output_bytes,
            dependency_bytes: self.dependency_bytes,
            last_progress: self.last_progress_unix_ms,
            ledger: self.budget.checkpoint(key),
            attempts: targets
                .into_iter()
                .map(|node| {
                    let old = self.attempts.get(&node).cloned();
                    (node, old)
                })
                .collect(),
            graph: matches!(record.event, Event::Expanded { .. })
                .then(|| (Arc::clone(&self.graph), Arc::clone(&self.ids))),
            failure_reason: self.failure_reason.clone(),
        }
    }
    fn restore(&mut self, old: Undo) {
        self.sequence = old.sequence;
        self.generation = old.generation;
        self.revision = old.revision;
        self.control = old.control;
        self.admission = old.admission;
        self.disposition = old.disposition;
        self.overrun = old.overrun;
        self.deadline_closed = old.deadline_closed;
        self.budget_blocked = old.budget_blocked;
        self.accepted_nodes = old.accepted_nodes;
        self.output_bytes = old.output_bytes;
        self.dependency_bytes = old.dependency_bytes;
        self.last_progress_unix_ms = old.last_progress;
        self.budget.undo(old.ledger);
        self.failure_reason = old.failure_reason;
        for (node, value) in old.attempts {
            match value {
                Some(a) => {
                    self.attempts.insert(node, a);
                }
                None => {
                    self.attempts.remove(&node);
                }
            }
        }
        if let Some((graph, ids)) = old.graph {
            self.graph = graph;
            self.ids = ids;
        }
    }
    fn descendants(&self, roots: &[String]) -> Result<Vec<String>> {
        let schedule = self.graph.schedule_with(&self.limits)?;
        let mut stack = roots
            .iter()
            .map(|n| self.node_index(n))
            .collect::<Result<Vec<_>>>()?;
        let mut seen = HashSet::new();
        while let Some(i) = stack.pop() {
            if seen.insert(i) {
                stack.extend(&schedule.children[i]);
            }
        }
        let mut result: Vec<_> = seen
            .into_iter()
            .map(|i| self.graph.nodes[i].id.clone())
            .collect();
        result.sort();
        Ok(result)
    }
    fn transition(&mut self, record: &Record) -> Result<()> {
        record.validate_header()?;
        ensure!(
            record.run_id == self.run_id
                && record.schema == JOURNAL_SCHEMA
                && record.sequence == self.sequence.checked_add(1).context("sequence overflow")?,
            "journal schema or sequence differs"
        );
        ensure!(
            record.revision == record.event.revision(self.revision),
            "graph revision identity differs"
        );
        let phase = record
            .node
            .as_ref()
            .map(|n| self.node(n).map(|n| n.phase.as_ref()))
            .transpose()?
            .flatten();
        ensure!(
            record.phase.as_ref() == phase,
            "node phase identity differs"
        );
        ensure!(
            self.disposition.is_none()
                || matches!(
                    record.event,
                    Event::Retried { .. }
                        | Event::Expanded { .. }
                        | Event::Revised { .. }
                        | Event::CeilingChanged { .. }
                ),
            "run already ended"
        );
        if let Event::Claimed { generation } = &record.event {
            ensure!(
                *generation
                    == self
                        .generation
                        .checked_add(1)
                        .context("generation overflow")?
                    && record.generation == *generation,
                "invalid ownership generation"
            );
            self.generation = *generation;
        } else {
            ensure!(
                record.generation == self.generation && self.generation > 0,
                EngineError::StaleOwner
            );
            match &record.event {
                Event::Control { control } => {
                    ensure!(
                        self.control != Control::Stop || *control == Control::Stop,
                        EngineError::Stopped
                    );
                    self.control = *control;
                }
                Event::CeilingChanged { ceiling } => {
                    self.budget.change_ceiling(*ceiling)?;
                    self.budget_blocked = false;
                }
                Event::RunFailed { reason } => {
                    ensure!(!reason.is_empty(), "failure needs a reason");
                    self.failure_reason = Some(reason.clone());
                }
                Event::AdmissionClosed { reason } => {
                    ensure!(
                        *reason == CloseReason::Deadline
                            && self
                                .limits
                                .wall_deadline_ms
                                .is_some_and(|d| record.occurred_at_unix_ms >= d),
                        "deadline has not elapsed"
                    );
                    self.deadline_closed = true;
                }
                Event::BudgetBlocked { node, bound } => {
                    self.node(node)?;
                    ensure!(
                        self.admission == Admission::Open
                            && self.budget.pending().is_empty()
                            && !self.budget.fits(NanoUsd(0), *bound),
                        "run is not budget blocked"
                    );
                    self.budget_blocked = true;
                }
                Event::Admitted {
                    node,
                    attempt,
                    bound,
                    effect,
                } => {
                    ensure!(self.admission == Admission::Open, EngineError::NotAdmitting);
                    let task = self.node(node)?;
                    ensure!(
                        task.depends_on.iter().all(|p| self
                            .attempts
                            .get(p)
                            .is_some_and(|a| a.state == AttemptState::Accepted)),
                        "unaccepted dependency"
                    );
                    let previous = self.attempts.get(node);
                    ensure!(
                        previous.is_none_or(|a| matches!(
                            a.state,
                            AttemptState::Replayed | AttemptState::Invalidated
                        )),
                        "node already attempted"
                    );
                    ensure!(
                        *attempt
                            == previous
                                .map_or(Some(1), |a| a.number.checked_add(1))
                                .context("attempt overflow")?
                            && *attempt <= self.limits.max_attempts_per_node,
                        "attempt limit or identity differs"
                    );
                    self.budget
                        .reserve(&call_key(self.revision, node, *attempt), *bound)?;
                    self.attempts.insert(
                        node.clone(),
                        Attempt {
                            revision: self.revision,
                            number: *attempt,
                            effect: *effect,
                            state: AttemptState::Admitted,
                            output: None,
                            output_valid: false,
                            cost: None,
                            elapsed_s: 0.0,
                            dependency_bytes: 0,
                            reason: None,
                        },
                    );
                    self.budget_blocked = false;
                }
                Event::Dispatched {
                    node,
                    attempt,
                    dependency_bytes,
                } => {
                    ensure!(self.admission == Admission::Open, EngineError::NotAdmitting);
                    let expected =
                        self.node(node)?
                            .depends_on
                            .iter()
                            .try_fold(0u64, |sum, n| {
                                sum.checked_add(
                                    self.attempts
                                        .get(n)
                                        .and_then(|a| a.output.as_ref())
                                        .context("missing dependency output")?
                                        .bytes,
                                )
                                .context("dependency bytes overflow")
                            })?;
                    ensure!(
                        *dependency_bytes == expected,
                        "dependency byte count differs"
                    );
                    let total = self
                        .dependency_bytes
                        .checked_add(expected)
                        .context("dependency bytes overflow")?;
                    let a = self.attempt(node, *attempt, AttemptState::Admitted)?;
                    a.state = AttemptState::Dispatched;
                    a.dependency_bytes = expected;
                    self.dependency_bytes = total;
                }
                Event::Received {
                    node,
                    attempt,
                    output,
                    output_valid,
                    cost,
                    elapsed_s,
                } => {
                    ensure!(
                        elapsed_s.is_finite() && *elapsed_s >= 0.0,
                        "invalid elapsed time"
                    );
                    ensure!(
                        !output_valid || output.is_some(),
                        "valid output requires a reference"
                    );
                    let mut total = self.output_bytes;
                    if let Some(output) = output {
                        output.validate()?;
                        total = total
                            .checked_add(output.bytes)
                            .context("output bytes overflow")?;
                        ensure!(
                            output.bytes <= self.limits.max_output_bytes
                                && total <= self.limits.max_total_output_bytes,
                            "output limit exceeded"
                        );
                    }
                    let a = self.attempt(node, *attempt, AttemptState::Dispatched)?;
                    a.output = output.clone();
                    a.output_valid = *output_valid;
                    a.cost = Some(*cost);
                    a.elapsed_s = *elapsed_s;
                    a.state = AttemptState::Received;
                    self.output_bytes = total;
                    self.last_progress_unix_ms = Some(record.recorded_at_unix_ms);
                }
                Event::Settled { node, attempt } => {
                    let a = self.attempt(node, *attempt, AttemptState::Received)?;
                    let key = call_key(a.revision, node, *attempt);
                    let cost = a.cost.context("missing cost evidence")?;
                    self.overrun |= self.budget.settle(&key, cost)?;
                    self.attempts.get_mut(node).unwrap().state = AttemptState::Settled;
                }
                Event::Accepted { node, attempt } => {
                    let a = self.attempt(node, *attempt, AttemptState::Settled)?;
                    ensure!(
                        a.output_valid && a.output.is_some(),
                        "invalid output cannot be accepted"
                    );
                    a.state = AttemptState::Accepted;
                    self.accepted_nodes += 1;
                    self.last_progress_unix_ms = Some(record.recorded_at_unix_ms);
                }
                Event::Rejected {
                    node,
                    attempt,
                    reason,
                } => {
                    let a = self.attempt(node, *attempt, AttemptState::Settled)?;
                    a.state = AttemptState::Rejected;
                    a.reason = Some(reason.clone());
                }
                Event::Failed {
                    node,
                    attempt,
                    reason,
                } => {
                    let a = self.attempt(node, *attempt, AttemptState::Dispatched)?;
                    ensure!(a.effect == Effect::Pure, EngineError::UnsupportedEffect);
                    let key = call_key(a.revision, node, *attempt);
                    self.budget.settle(&key, NanoUsd(0))?;
                    let a = self.attempts.get_mut(node).unwrap();
                    a.state = AttemptState::Failed;
                    a.reason = Some(reason.clone());
                }
                Event::LocalReplay { node, attempt } => {
                    let a = self.attempts.get(node).context("missing attempt")?;
                    ensure!(a.effect == Effect::Pure, EngineError::UnsupportedEffect);
                    ensure!(
                        a.number == *attempt
                            && matches!(a.state, AttemptState::Admitted | AttemptState::Dispatched),
                        "only interrupted local work can replay"
                    );
                    self.budget
                        .settle(&call_key(a.revision, node, *attempt), NanoUsd(0))?;
                    let a = self.attempts.get_mut(node).unwrap();
                    a.state = if *attempt < self.limits.max_attempts_per_node {
                        AttemptState::Replayed
                    } else {
                        a.reason = Some("attempt_limit".into());
                        AttemptState::Failed
                    };
                }
                Event::Retried { node, attempt } => {
                    ensure!(
                        self.control != Control::Stop
                            && !matches!(
                                self.disposition,
                                Some(Disposition::Completed | Disposition::Stopped)
                            ),
                        EngineError::Stopped
                    );
                    let limit = self.limits.max_attempts_per_node;
                    let a = self.attempts.get_mut(node).context("missing attempt")?;
                    ensure!(a.effect == Effect::Pure, EngineError::UnsupportedEffect);
                    ensure!(
                        a.number == *attempt
                            && matches!(a.state, AttemptState::Failed | AttemptState::Rejected)
                            && a.number < limit,
                        "attempt cannot be retried"
                    );
                    a.state = AttemptState::Replayed;
                    self.disposition = None;
                }
                Event::Expanded {
                    parent_revision,
                    revision,
                    input_manifest_sha256,
                    nodes,
                } => {
                    self.require_revision(*revision)?;
                    ensure!(
                        *parent_revision == self.revision && !nodes.is_empty(),
                        "expansion parent or nodes differ"
                    );
                    OutputRef {
                        sha256: input_manifest_sha256.clone(),
                        bytes: 0,
                    }
                    .validate()?;
                    let mut graph = (*self.graph).clone();
                    graph.nodes.extend(nodes.iter().cloned());
                    graph.schedule_with(&self.limits)?;
                    self.ids = Arc::new(Self::index(&graph));
                    self.graph = Arc::new(graph);
                    self.revision = *revision;
                    self.disposition = None;
                    self.budget_blocked = false;
                }
                Event::Revised {
                    revision,
                    reason,
                    invalidated,
                } => {
                    self.require_revision(*revision)?;
                    ensure!(
                        !reason.is_empty() && !invalidated.is_empty(),
                        "revision needs a reason and nodes"
                    );
                    for node in self.descendants(invalidated)? {
                        if let Some(a) = self.attempts.get_mut(&node) {
                            if a.state == AttemptState::Accepted {
                                self.accepted_nodes -= 1;
                            }
                            a.state = AttemptState::Invalidated;
                        }
                    }
                    self.revision = *revision;
                    self.disposition = None;
                    self.budget_blocked = false;
                }
                Event::Ended { disposition } => {
                    ensure!(self.budget.pending().is_empty(), "unresolved reservations");
                    ensure!(
                        !self.attempts.values().any(|a| matches!(
                            a.state,
                            AttemptState::Admitted
                                | AttemptState::Dispatched
                                | AttemptState::Received
                                | AttemptState::Settled
                        )),
                        "unresolved attempts"
                    );
                    let covered = self.accepted_nodes == self.graph.nodes.len();
                    let failed = self.failure_reason.is_some()
                        || self.attempts.values().any(|a| {
                            matches!(a.state, AttemptState::Failed | AttemptState::Rejected)
                        });
                    ensure!(
                        match disposition {
                            Disposition::Completed => covered,
                            Disposition::Failed => !covered && failed,
                            Disposition::Stopped =>
                                !covered
                                    && !failed
                                    && matches!(self.admission, Admission::Closed { .. }),
                        },
                        "run disposition differs from coverage and admission"
                    );
                    self.disposition = Some(*disposition);
                }
                _ => anyhow::bail!("invalid run transition"),
            }
        }
        self.admission = if self.control == Control::Stop {
            Admission::Closed {
                reason: CloseReason::OperatorStop,
            }
        } else if self.overrun {
            Admission::Closed {
                reason: CloseReason::Overrun,
            }
        } else if self.deadline_closed {
            Admission::Closed {
                reason: CloseReason::Deadline,
            }
        } else if self.control == Control::Pause {
            Admission::Paused
        } else {
            Admission::Open
        };
        self.sequence = record.sequence;
        Ok(())
    }
    fn require_revision(&self, revision: u32) -> Result<()> {
        ensure!(
            revision == self.revision.checked_add(1).context("revision overflow")?
                && self.budget.pending().is_empty()
                && self.control != Control::Stop,
            "revision requires a quiescent run and its next identity"
        );
        ensure!(
            self.attempts.values().all(|a| !matches!(
                a.state,
                AttemptState::Admitted
                    | AttemptState::Dispatched
                    | AttemptState::Received
                    | AttemptState::Settled
            )),
            "revision has unsettled work"
        );
        Ok(())
    }
    fn attempt(&mut self, node: &str, number: u64, state: AttemptState) -> Result<&mut Attempt> {
        let a = self.attempts.get_mut(node).context("unknown attempt")?;
        ensure!(
            a.number == number && a.state == state,
            "invalid attempt transition"
        );
        Ok(a)
    }
}
