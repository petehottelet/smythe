//! Runtime-independent graph contracts and exact money.

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub mod coordinator;
pub mod errors;
pub mod journal;
pub mod money;
pub use errors::EngineError;
pub use money::{NanoUsd, Sentinel};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub goal: String,
    pub input: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub task: Task,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    Serial,
    ForkJoin,
    BroadcastReduce,
    Adversarial,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionGraph {
    /// A planner's descriptive label; execution is derived from dependencies.
    #[serde(default)]
    pub topology: Vec<Topology>,
    pub nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub remaining: Vec<usize>,
    pub children: Vec<Vec<usize>>,
    pub ready: VecDeque<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_attempts_per_node: u64,
    pub max_output_bytes: u64,
    pub max_total_output_bytes: u64,
    /// Absolute UTC Unix milliseconds. A backward clock cannot reopen admission.
    pub wall_deadline_ms: Option<u64>,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_edges: 100_000,
            max_attempts_per_node: 3,
            max_output_bytes: 64 * 1024 * 1024,
            max_total_output_bytes: 1024 * 1024 * 1024,
            wall_deadline_ms: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRef {
    pub sha256: String,
    pub bytes: u64,
}
impl OutputRef {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            EngineError::Integrity
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Shape {
    pub nodes: usize,
    pub edges: usize,
    pub maximum_width: usize,
    pub depth: usize,
    pub joins: usize,
}

impl ExecutionGraph {
    pub fn validate(&self) -> Result<()> {
        self.schedule().map(|_| ())
    }

    pub fn schedule(&self) -> Result<Schedule> {
        self.schedule_with(&Limits::default())
    }
    pub fn shape(&self) -> Result<Shape> {
        let schedule = self.schedule_with(&Limits {
            max_nodes: self.nodes.len(),
            max_edges: usize::MAX,
            ..Limits::default()
        })?;
        let mut remaining = schedule.remaining;
        let mut queue = schedule.ready;
        let mut depths = vec![0; self.nodes.len()];
        let mut widths = vec![0; self.nodes.len()];
        while let Some(i) = queue.pop_front() {
            widths[depths[i]] += 1;
            for &child in &schedule.children[i] {
                depths[child] = depths[child].max(depths[i] + 1);
                remaining[child] -= 1;
                if remaining[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        Ok(Shape {
            nodes: self.nodes.len(),
            edges: self.nodes.iter().map(|n| n.depends_on.len()).sum(),
            maximum_width: widths.into_iter().max().unwrap_or(0),
            depth: depths.into_iter().max().map_or(0, |n| n + 1),
            joins: self.nodes.iter().filter(|n| n.depends_on.len() > 1).count(),
        })
    }
    pub fn schedule_with(&self, limits: &Limits) -> Result<Schedule> {
        ensure!(
            self.nodes.len() <= limits.max_nodes,
            "graph exceeds the native node limit"
        );
        ensure!(
            limits.max_attempts_per_node > 0,
            "attempt limit must be positive"
        );
        let edges = self.nodes.iter().try_fold(0usize, |n, node| {
            n.checked_add(node.depends_on.len())
                .ok_or_else(|| anyhow!("edge count overflow"))
        })?;
        ensure!(edges <= limits.max_edges, "graph exceeds the edge limit");
        let mut ids = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            ensure!(
                node.phase.as_ref().is_none_or(|p| !p.is_empty()
                    && p.len() <= 256
                    && !p.chars().any(char::is_control)),
                "invalid phase label"
            );
            ensure!(
                !node.id.is_empty()
                    && node.id.len() <= 256
                    && !node.id.chars().any(char::is_control)
                    && node.id != "__synthesis__",
                "invalid node ID"
            );
            let mut values = vec![&node.task.input];
            while let Some(value) = values.pop() {
                match value {
                    serde_json::Value::Number(n) => {
                        if n.is_u64() {
                            ensure!(
                                n.as_u64().unwrap() <= 9_007_199_254_740_991,
                                "large integers require decimal strings for canonical JSON"
                            );
                        } else if n.is_i64() {
                            ensure!(
                                n.as_i64().unwrap() >= -9_007_199_254_740_991,
                                "large integers require decimal strings for canonical JSON"
                            );
                        }
                    }
                    serde_json::Value::Array(a) => values.extend(a),
                    serde_json::Value::Object(o) => values.extend(o.values()),
                    _ => {}
                }
            }
            ensure!(ids.insert(&node.id, i).is_none(), "duplicate node ID");
        }
        let mut remaining = vec![0; self.nodes.len()];
        let mut children = vec![vec![]; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            let mut seen = HashSet::new();
            for dependency in &node.depends_on {
                ensure!(seen.insert(dependency), "duplicate dependency");
                let parent = *ids
                    .get(dependency)
                    .ok_or_else(|| anyhow!("missing dependency"))?;
                children[parent].push(i);
                remaining[i] += 1;
            }
        }
        let ready: VecDeque<_> = remaining
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| (n == 0).then_some(i))
            .collect();
        let mut pending = remaining.clone();
        let mut queue = ready.clone();
        let mut visited = 0;
        while let Some(i) = queue.pop_front() {
            visited += 1;
            for &child in &children[i] {
                pending[child] -= 1;
                if pending[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        ensure!(visited == self.nodes.len(), "graph contains a cycle");
        Ok(Schedule {
            remaining,
            children,
            ready,
        })
    }
}
