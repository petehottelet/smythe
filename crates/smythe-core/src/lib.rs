//! Runtime-independent graph contracts and exact money.

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub mod journal;
pub mod money;
pub use money::{NanoUsd, Sentinel};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub goal: String,
    pub input: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub task: Task,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    Serial,
    ForkJoin,
    BroadcastReduce,
    Adversarial,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionGraph {
    pub topology: Vec<Topology>,
    pub nodes: Vec<Node>,
}

pub struct Schedule {
    pub remaining: Vec<usize>,
    pub children: Vec<Vec<usize>>,
    pub ready: VecDeque<usize>,
}

impl ExecutionGraph {
    pub fn validate(&self) -> Result<()> {
        self.schedule().map(|_| ())
    }

    pub fn schedule(&self) -> Result<Schedule> {
        ensure!(
            self.nodes.len() <= 10_000,
            "graph exceeds the native node limit"
        );
        let mut ids = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
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
