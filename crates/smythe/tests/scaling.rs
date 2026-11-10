use anyhow::Result;
use smythe::{
    CompletionResult, Dependency, ExecutionGraph, Node, Provider, Swarm, Task, seams::Sources,
    store::Store,
};
fn graph(n: usize, chain: bool) -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![],
        nodes: (0..n)
            .map(|i| Node {
                id: format!("n{i}"),
                phase: None,
                task: Task {
                    goal: "local".into(),
                    input: i.into(),
                },
                depends_on: if chain && i > 0 {
                    vec![format!("n{}", i - 1)]
                } else {
                    vec![]
                },
            })
            .collect(),
    }
}
struct Bytes(usize);
impl Provider for Bytes {
    type Output = String;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        format!("bytes-{}", self.0)
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(0)
    }
    fn complete(&self, t: &Task, _: &[Dependency<'_, String>]) -> Result<CompletionResult<String>> {
        Ok(CompletionResult {
            output: format!("{}{}", t.input, "x".repeat(self.0)),
            cost_nanousd: 0,
        })
    }
}
fn trial(n: usize, chain: bool, disk: bool, size: usize) -> f64 {
    let sources = Sources::default();
    let root = std::env::temp_dir().join(format!("smythe-scale-{}", sources.ids.next_id()));
    let store = if disk {
        Store::open(&root).unwrap()
    } else {
        Store::memory().unwrap()
    };
    let graph = graph(n, chain);
    let swarm = Swarm::new(8, 0).unwrap();
    let start = sources.clock.monotonic();
    let result = swarm
        .execute_in(&graph, &Bytes(size), &store, "scale", false)
        .unwrap();
    let seconds = sources
        .clock
        .monotonic()
        .saturating_sub(start)
        .as_secs_f64();
    assert_eq!(result.outputs.len(), n);
    assert_eq!(result.loaded_dependency_bytes, 0);
    drop(result);
    drop(store);
    if disk {
        std::fs::remove_dir_all(root).unwrap();
    }
    seconds
}
fn median(mut f: impl FnMut() -> f64) -> f64 {
    let mut samples = [f(), f(), f()];
    samples.sort_by(f64::total_cmp);
    samples[1]
}
#[test]
#[ignore = "release-mode scaling gate; explicitly run in native CI"]
fn release_scheduler_and_inline_storage_scale() {
    assert!(!cfg!(debug_assertions), "run with --release");
    trial(256, false, false, 32);
    let wide_2000 = median(|| trial(2000, false, false, 32));
    let wide_8000 = median(|| trial(8000, false, false, 32));
    let ratio = wide_8000 / wide_2000 / 4.0;
    let chain_5000 = trial(5000, true, false, 32);
    let memory = median(|| trial(256, false, false, 1024));
    let disk = median(|| trial(256, false, true, 1024));
    let file_disk = trial(256, false, true, 70_000);
    println!(
        "{}",
        serde_json::json!({"wide_2000_s":wide_2000,"wide_8000_s":wide_8000,"per_node_ratio":ratio,"chain_5000_s":chain_5000,"inline_memory_s":memory,"inline_disk_s":disk,"disk_memory_ratio":disk/memory,"large_file_disk_s":file_disk,"claimable":false})
    );
    assert!(ratio <= 1.5, "per-node scaling ratio {ratio} exceeds 1.5");
    assert!(
        disk / memory <= 3.0,
        "inline disk/memory ratio {} exceeds 3",
        disk / memory
    );
}
