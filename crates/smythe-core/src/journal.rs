//! Replayable local execution transitions. No I/O or runtime dependencies.
use crate::{ExecutionGraph, NanoUsd, Sentinel};
use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const JOURNAL_SCHEMA: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    Run,
    Pause,
    Stop,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Started {
        binding: String,
        graph: ExecutionGraph,
        ceiling: NanoUsd,
    },
    Claimed {
        generation: u64,
    },
    Control {
        control: Control,
    },
    Admitted {
        node: String,
        attempt: u64,
        bound: NanoUsd,
    },
    Dispatched {
        node: String,
        attempt: u64,
    },
    Received {
        node: String,
        attempt: u64,
        output: Value,
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
    Completed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub sequence: u64,
    pub generation: u64,
    pub event: Event,
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub number: u64,
    pub state: AttemptState,
    pub output: Option<Value>,
    pub output_valid: bool,
    pub cost: Option<NanoUsd>,
    pub elapsed_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projection {
    pub sequence: u64,
    pub generation: u64,
    pub binding: String,
    pub graph: ExecutionGraph,
    pub control: Control,
    pub completed: bool,
    pub budget: Sentinel,
    pub attempts: BTreeMap<String, Attempt>,
}
pub fn call_key(node: &str, attempt: u64) -> String {
    format!("{node}/{attempt}")
}

impl Projection {
    pub fn replay(records: &[Record]) -> Result<Self> {
        let first = records
            .first()
            .ok_or_else(|| anyhow!("missing run journal"))?;
        ensure!(
            first.schema == JOURNAL_SCHEMA && first.sequence == 1 && first.generation == 0,
            "invalid initial journal record"
        );
        let Event::Started {
            binding,
            graph,
            ceiling,
        } = &first.event
        else {
            anyhow::bail!("journal must start with a run");
        };
        graph.validate()?;
        let mut state = Self {
            sequence: 1,
            generation: 0,
            binding: binding.clone(),
            graph: graph.clone(),
            control: Control::Run,
            completed: false,
            budget: Sentinel::new(*ceiling),
            attempts: BTreeMap::new(),
        };
        for record in &records[1..] {
            state.apply(record)?;
        }
        Ok(state)
    }
    /// Validate against a copy so rejected transitions cannot partially mutate state.
    pub fn apply(&mut self, record: &Record) -> Result<()> {
        let mut next = self.clone();
        next.transition(record)?;
        *self = next;
        Ok(())
    }
    fn transition(&mut self, record: &Record) -> Result<()> {
        ensure!(
            record.schema == JOURNAL_SCHEMA
                && record.sequence
                    == self
                        .sequence
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("sequence overflow"))?,
            "journal schema or sequence differs"
        );
        ensure!(!self.completed, "run already completed");
        match &record.event {
            Event::Claimed { generation } => {
                ensure!(
                    *generation
                        == self
                            .generation
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("generation overflow"))?
                        && record.generation == *generation,
                    "invalid ownership generation"
                );
                self.generation = *generation;
            }
            Event::Control { control } => {
                ensure!(
                    record.generation == self.generation,
                    "stale control generation"
                );
                ensure!(
                    self.control != Control::Stop || *control == Control::Stop,
                    "a stopped run cannot restart"
                );
                self.control = *control;
            }
            event => {
                ensure!(
                    record.generation == self.generation && self.generation > 0,
                    "stale owner"
                );
                match event {
                    Event::Admitted {
                        node,
                        attempt,
                        bound,
                    } => {
                        ensure!(self.control == Control::Run, "run is not admitting work");
                        let task = self
                            .graph
                            .nodes
                            .iter()
                            .find(|n| n.id == *node)
                            .ok_or_else(|| anyhow!("unknown node"))?;
                        ensure!(
                            task.depends_on.iter().all(|p| self
                                .attempts
                                .get(p)
                                .is_some_and(|a| a.state == AttemptState::Accepted)),
                            "unaccepted dependency"
                        );
                        let previous = self.attempts.get(node);
                        ensure!(
                            previous.is_none_or(|a| a.state == AttemptState::Replayed),
                            "node already attempted"
                        );
                        ensure!(
                            *attempt
                                == previous
                                    .map_or(Some(1), |a| a.number.checked_add(1))
                                    .ok_or_else(|| anyhow!("attempt overflow"))?,
                            "invalid attempt number"
                        );
                        self.budget.reserve(&call_key(node, *attempt), *bound)?;
                        self.attempts.insert(
                            node.clone(),
                            Attempt {
                                number: *attempt,
                                state: AttemptState::Admitted,
                                output: None,
                                output_valid: false,
                                cost: None,
                                elapsed_s: 0.0,
                            },
                        );
                    }
                    Event::Dispatched { node, attempt } => {
                        ensure!(self.control == Control::Run, "run stopped before dispatch");
                        self.attempt(node, *attempt, AttemptState::Admitted)?.state =
                            AttemptState::Dispatched;
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
                        let a = self.attempt(node, *attempt, AttemptState::Dispatched)?;
                        a.output = Some(output.clone());
                        a.output_valid = *output_valid;
                        a.cost = Some(*cost);
                        a.elapsed_s = *elapsed_s;
                        a.state = AttemptState::Received;
                    }
                    Event::Settled { node, attempt } => {
                        let cost = self
                            .attempt(node, *attempt, AttemptState::Received)?
                            .cost
                            .ok_or_else(|| anyhow!("missing cost evidence"))?;
                        let overrun = self.budget.settle(&call_key(node, *attempt), cost)?;
                        self.attempts.get_mut(node).unwrap().state = AttemptState::Settled;
                        if overrun {
                            self.control = Control::Stop;
                        }
                    }
                    Event::Accepted { node, attempt } => {
                        let saved = self.attempt(node, *attempt, AttemptState::Settled)?;
                        ensure!(saved.output_valid, "invalid output cannot be accepted");
                        saved.state = AttemptState::Accepted;
                    }
                    Event::Rejected { node, attempt, .. } => {
                        self.attempt(node, *attempt, AttemptState::Settled)?.state =
                            AttemptState::Rejected;
                    }
                    Event::Failed { node, attempt, .. } => {
                        self.attempt(node, *attempt, AttemptState::Dispatched)?;
                        // Local-only provider contract: errors carry no external effects or charges.
                        self.budget.settle(&call_key(node, *attempt), NanoUsd(0))?;
                        self.attempts.get_mut(node).unwrap().state = AttemptState::Failed;
                    }
                    Event::LocalReplay { node, attempt } => {
                        let a = self
                            .attempts
                            .get(node)
                            .ok_or_else(|| anyhow!("missing attempt"))?;
                        ensure!(
                            a.number == *attempt
                                && matches!(
                                    a.state,
                                    AttemptState::Admitted | AttemptState::Dispatched
                                ),
                            "only interrupted local work can replay"
                        );
                        self.budget.settle(&call_key(node, *attempt), NanoUsd(0))?;
                        self.attempts.get_mut(node).unwrap().state = AttemptState::Replayed;
                    }
                    Event::Completed => {
                        ensure!(
                            self.control == Control::Run
                                && self.graph.nodes.iter().all(|n| self
                                    .attempts
                                    .get(&n.id)
                                    .is_some_and(|a| a.state == AttemptState::Accepted)),
                            "run is not accepted"
                        );
                        ensure!(self.budget.pending().is_empty(), "unresolved reservations");
                        self.completed = true;
                    }
                    _ => anyhow::bail!("invalid run transition"),
                }
            }
        }
        self.sequence = record.sequence;
        Ok(())
    }
    fn attempt(&mut self, node: &str, number: u64, state: AttemptState) -> Result<&mut Attempt> {
        let a = self
            .attempts
            .get_mut(node)
            .ok_or_else(|| anyhow!("unknown attempt"))?;
        ensure!(
            a.number == number && a.state == state,
            "invalid attempt transition"
        );
        Ok(a)
    }
}
