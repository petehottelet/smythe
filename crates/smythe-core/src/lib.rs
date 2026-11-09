//! Smythe's first native execution path: validated graphs and bounded CPU work.
//!
//! This local executor does not yet implement Python's durable provider journal,
//! tool loop, supervision or ambiguous remote-call accounting.

use anyhow::{Result, anyhow, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

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

struct Schedule {
    remaining: Vec<usize>,
    children: Vec<Vec<usize>>,
    ready: VecDeque<usize>,
}

impl ExecutionGraph {
    pub fn validate(&self) -> Result<()> {
        self.schedule().map(|_| ())
    }

    fn schedule(&self) -> Result<Schedule> {
        let mut ids = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            ensure!(
                !node.id.is_empty() && node.id != "__synthesis__",
                "invalid node ID"
            );
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

/// Integer money; reservations and confirmed charges use the same nanoUSD unit.
#[derive(Debug, Default)]
pub struct Sentinel {
    ceiling: u64,
    reserved: u64,
    spent: u64,
}

impl Sentinel {
    pub fn new(ceiling_nanousd: u64) -> Self {
        Self {
            ceiling: ceiling_nanousd,
            ..Self::default()
        }
    }
    pub fn reserve(&mut self, amount: u64) -> Result<()> {
        let exposure = self
            .spent
            .checked_add(self.reserved)
            .and_then(|x| x.checked_add(amount))
            .ok_or_else(|| anyhow!("cost overflow"))?;
        ensure!(exposure <= self.ceiling, "budget exhausted before dispatch");
        self.reserved += amount;
        Ok(())
    }
    pub fn settle(&mut self, reserved: u64, actual: u64) -> Result<()> {
        ensure!(reserved <= self.reserved, "unrecognized reservation");
        self.reserved -= reserved;
        self.spent = self
            .spent
            .checked_add(actual)
            .ok_or_else(|| anyhow!("cost overflow"))?;
        ensure!(
            actual <= reserved && self.spent <= self.ceiling,
            "provider exceeded its cost bound"
        );
        Ok(())
    }
    pub fn spent_nanousd(&self) -> u64 {
        self.spent
    }
}

pub struct CompletionResult<T> {
    pub output: T,
    pub cost_nanousd: u64,
}

/// Local calls only: errors must entail neither charges nor uncertain external effects.
pub trait Provider: Sync {
    type Output: Send + Sync;
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
    pub cost_nanousd: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct Tracer {
    pub spans: Vec<Span>,
}

pub struct SwarmResult<T> {
    /// Original graph order, independent of completion order.
    pub outputs: Vec<T>,
    pub trace: Tracer,
    pub total_cost_nanousd: u64,
}

pub struct Swarm {
    pool: rayon::ThreadPool,
    max_concurrency: usize,
    max_budget_nanousd: u64,
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
        let mut schedule = graph.schedule()?;
        let mut values: Vec<Option<P::Output>> = (0..graph.nodes.len()).map(|_| None).collect();
        let ids: HashMap<_, _> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (&n.id, i))
            .collect();
        let mut budget = Sentinel::new(self.max_budget_nanousd);
        let mut trace = Tracer::default();
        // Bounded waves keep the runnable set small and never mutate graph state
        // on worker threads. All siblings settle before an error is returned.
        while !schedule.ready.is_empty() {
            let mut batch = vec![];
            for _ in 0..self.max_concurrency {
                let Some(index) = schedule.ready.pop_front() else {
                    break;
                };
                let bound = provider.cost_bound_nanousd(&graph.nodes[index].task)?;
                budget.reserve(bound)?;
                batch.push((index, bound));
            }
            let finished: Vec<_> = self.pool.install(|| {
                batch
                    .par_iter()
                    .map(|&(index, bound)| {
                        let node = &graph.nodes[index];
                        let dependencies: Vec<_> = node
                            .depends_on
                            .iter()
                            .map(|id| {
                                values[ids[id]]
                                    .as_ref()
                                    .expect("validated dependency completed")
                            })
                            .collect();
                        let started = Instant::now();
                        let result = provider.complete(&node.task, &dependencies);
                        (index, bound, started.elapsed().as_secs_f64(), result)
                    })
                    .collect()
            });
            let mut failure = None;
            for (index, bound, elapsed_s, result) in finished {
                match result {
                    Ok(result) => {
                        if let Err(error) = budget.settle(bound, result.cost_nanousd) {
                            failure.get_or_insert(error);
                        }
                        trace.spans.push(Span {
                            node_id: graph.nodes[index].id.clone(),
                            elapsed_s,
                            cost_nanousd: result.cost_nanousd,
                        });
                        values[index] = Some(result.output);
                        for &child in &schedule.children[index] {
                            schedule.remaining[child] -= 1;
                            if schedule.remaining[child] == 0 {
                                schedule.ready.push_back(child);
                            }
                        }
                    }
                    Err(error) => {
                        budget.settle(bound, 0)?;
                        failure.get_or_insert(error);
                    }
                }
            }
            if let Some(error) = failure {
                bail!("local execution failed: {error}");
            }
        }
        Ok(SwarmResult {
            outputs: values
                .into_iter()
                .map(|v| v.expect("validated graph completed"))
                .collect(),
            trace,
            total_cost_nanousd: budget.spent_nanousd(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Sum {
        calls: AtomicUsize,
        active: AtomicUsize,
        peak: AtomicUsize,
    }
    impl Provider for Sum {
        type Output = u64;
        fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
            Ok(1)
        }
        fn complete(&self, task: &Task, deps: &[&u64]) -> Result<CompletionResult<u64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(2));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(CompletionResult {
                output: task.input.as_u64().unwrap() + deps.iter().map(|v| **v).sum::<u64>(),
                cost_nanousd: 1,
            })
        }
    }
    fn provider() -> Sum {
        Sum {
            calls: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
    fn graph() -> ExecutionGraph {
        ExecutionGraph {
            topology: vec![Topology::ForkJoin],
            nodes: (0..5)
                .map(|i| Node {
                    id: i.to_string(),
                    task: Task {
                        goal: "sum".into(),
                        input: 1.into(),
                    },
                    depends_on: if i == 4 {
                        (0..4).map(|j| j.to_string()).collect()
                    } else {
                        vec![]
                    },
                })
                .collect(),
        }
    }
    #[test]
    fn fork_join_is_bounded_and_ordered() {
        let p = provider();
        let result = Swarm::new(2, 5).unwrap().execute(&graph(), &p).unwrap();
        assert_eq!(result.outputs, [1, 1, 1, 1, 5]);
        assert_eq!(result.total_cost_nanousd, 5);
        assert!(p.peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(p.calls.load(Ordering::SeqCst), 5);
    }
    #[test]
    fn invalid_graphs_never_dispatch() {
        for mode in 0..4 {
            let mut g = graph();
            match mode {
                0 => g.nodes[0].depends_on.push("4".into()),
                1 => g.nodes[0].depends_on.push("absent".into()),
                2 => g.nodes[0].id = "1".into(),
                _ => g.nodes[4].depends_on.push("0".into()),
            }
            let p = provider();
            assert!(Swarm::new(2, 5).unwrap().execute(&g, &p).is_err());
            assert_eq!(p.calls.load(Ordering::SeqCst), 0);
        }
    }
    #[test]
    fn budget_admits_before_dispatch_and_checks_overflow() {
        let p = provider();
        assert!(Swarm::new(2, 0).unwrap().execute(&graph(), &p).is_err());
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
        let mut s = Sentinel::new(u64::MAX);
        s.reserve(u64::MAX).unwrap();
        assert!(s.reserve(1).is_err());
        s.settle(u64::MAX, 10).unwrap();
        assert_eq!(s.spent_nanousd(), 10);
        assert!(s.settle(1, 0).is_err());
    }
}
