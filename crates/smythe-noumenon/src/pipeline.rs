#![allow(clippy::disallowed_methods)] // Report timing only.
use crate::geometry::{self, Glyph, Recipe, sha};
use crate::validation::{self, Measurement, Verifier};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use smythe::{CompletionResult, ExecutionGraph, Node, Provider, Swarm, Task, Topology};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const PROTOCOL: &str = "noumenon-rust-journal-v2";
/// Bump when compilation/output semantics change, independently of source provenance.
pub const COMPILER_VERSION: &str = "noumenon-contours-v1";
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Compile,
    Full,
}

pub struct Options {
    pub glyphs: usize,
    pub workers: usize,
    pub out: PathBuf,
    pub resume: bool,
    pub mode: Mode,
}

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Binding {
    protocol: String,
    source_sha256: String,
    compiler_version: String,
    glyphs: usize,
    mode: Mode,
    raster_version: String,
    gate_version: String,
}

impl Binding {
    fn for_options(options: &Options) -> Self {
        Self {
            protocol: PROTOCOL.into(),
            source_sha256: geometry::source_sha(),
            compiler_version: COMPILER_VERSION.into(),
            glyphs: options.glyphs,
            mode: options.mode,
            raster_version: validation::RASTER_VERSION.into(),
            gate_version: validation::GATE_VERSION.into(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    glyph: Glyph,
    measurements: Vec<Measurement>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Batch {
    indices: Vec<usize>,
    receipts: Vec<Receipt>,
}
const CHECKPOINT_BATCH: usize = 16;

/// Publish immutable bytes by hard link after fsync. Existing bytes must match;
/// this also recovers a crash between artifact publication and receipt writing.
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    publish_bytes(path, bytes, true)
}

fn publish_bytes(path: &Path, bytes: &[u8], durable: bool) -> Result<()> {
    if path.exists() {
        ensure!(fs::read(path)? == bytes, "existing artifact differs");
        return Ok(());
    }
    let tmp = path.with_extension(format!(
        "pending-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        file.write_all(bytes)?;
        if durable {
            file.sync_all()?;
        }
        drop(file);
        fs::hard_link(&tmp, path).context("artifact publication requires hard-link support")?;
        Ok(())
    })();
    if tmp.exists() {
        fs::remove_file(tmp)?;
    }
    result
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct ContourProvider<'a> {
    recipes: &'a [Recipe],
}
impl Provider for ContourProvider<'_> {
    type Output = Glyph;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        format!("{COMPILER_VERSION}:{}", geometry::source_sha())
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(0)
    }
    fn complete(&self, task: &Task, _: &[&Glyph]) -> Result<CompletionResult<Glyph>> {
        let i = task
            .input
            .as_u64()
            .context("glyph task requires an integer index")? as usize;
        let recipe = self.recipes.get(i).context("glyph index outside catalog")?;
        Ok(CompletionResult {
            output: geometry::compile(i, recipe)?,
            cost_nanousd: 0,
        })
    }
}

fn graph(indices: &[usize]) -> ExecutionGraph {
    ExecutionGraph {
        topology: vec![Topology::BroadcastReduce],
        nodes: indices
            .iter()
            .map(|&i| Node {
                id: format!("glyph-{i:03}"),
                task: Task {
                    goal: format!("Compile Noumenon glyph {i}"),
                    input: i.into(),
                },
                depends_on: vec![],
            })
            .collect(),
    }
}

pub fn run(options: &Options) -> Result<Value> {
    ensure!(
        (1..=256).contains(&options.glyphs),
        "glyph count must be 1 through 256"
    );
    ensure!(
        (1..=64).contains(&options.workers),
        "workers must be 1 through 64"
    );
    let all = geometry::recipes()?;
    let binding = Binding::for_options(options);
    let started = Instant::now();
    let out = &options.out;
    if options.resume {
        ensure!(out.is_dir(), "resume directory does not exist");
    } else {
        fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
        fs::create_dir(out).context("output directory already exists or cannot be created")?;
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(out.join(".lock"))?;
    lock.try_lock().context("another process owns this run")?;
    let binding_path = out.join("run.json");
    if options.resume {
        let saved: Binding = serde_json::from_slice(&fs::read(&binding_path)?)?;
        ensure!(saved == binding, "resume recipe differs from the saved run");
    } else {
        publish(&binding_path, &json_bytes(&binding)?)?;
    }
    fs::create_dir_all(out.join("receipts"))?;
    let mut records: Vec<Option<Receipt>> = (0..options.glyphs).map(|_| None).collect();
    for first in 0..options.glyphs {
        let path = out.join("receipts").join(format!("batch-{first:03}.json"));
        if options.resume && path.is_file() {
            let batch: Batch = serde_json::from_slice(&fs::read(path)?)?;
            ensure!(
                batch.indices.first() == Some(&first)
                    && batch.indices.len() <= CHECKPOINT_BATCH
                    && batch.indices.len() == batch.receipts.len(),
                "invalid checkpoint batch"
            );
            for (i, r) in batch.indices.into_iter().zip(batch.receipts) {
                ensure!(
                    i < options.glyphs && records[i].is_none(),
                    "duplicate or out-of-range receipt"
                );
                ensure!(
                    r.glyph.glyph_id == format!("GLYPH-{i:03}"),
                    "receipt identity differs"
                );
                ensure!(
                    r.glyph.svg_sha256 == sha(r.glyph.svg.as_bytes()),
                    "receipt checksum differs"
                );
                // Paths and SVG must describe the same trusted authored recipe.
                let expected = geometry::compile(i, &all[i])?;
                ensure!(
                    expected == r.glyph,
                    "receipt differs from authored contours"
                );
                if options.mode == Mode::Full {
                    let (measurements, _) = Verifier.validate(&r.glyph)?;
                    ensure!(
                        measurements == r.measurements,
                        "saved raster receipt differs"
                    );
                } else {
                    ensure!(r.measurements.is_empty(), "unexpected raster receipt");
                }
                // The durable batch owns the bytes. A missing derived SVG can be
                // restored after a crash; an existing, changed SVG is never replaced.
                publish_bytes(
                    &out.join(format!("GLYPH-{i:03}.svg")),
                    r.glyph.svg.as_bytes(),
                    false,
                )?;
                records[i] = Some(r);
            }
        }
    }
    let pending: Vec<_> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.is_none().then_some(i))
        .collect();
    let reused = options.glyphs - pending.len();
    let compilation_started = Instant::now();
    let swarm = Swarm::new(options.workers, 0)?;
    let journal = smythe::store::Store::open(out.join(".smythe"))?;
    let complete_graph = graph(&(0..options.glyphs).collect::<Vec<_>>());
    let engine_resume = !journal.history("noumenon-compile")?.is_empty();
    ensure!(
        engine_resume || pending.len() == options.glyphs,
        "published receipts require their compilation journal"
    );
    let mut execution = swarm.execute_in(
        &complete_graph,
        &ContourProvider { recipes: &all },
        &journal,
        "noumenon-compile",
        engine_resume,
    )?;
    // Accepted compilation blobs may exist before any validation batch is published.
    // Check them too, before exporting or accepting saved validation evidence.
    for (i, glyph) in execution.outputs.iter().enumerate() {
        ensure!(
            *glyph == geometry::compile(i, &all[i])?,
            "compilation output differs from authored contours"
        );
        if let Some(receipt) = &records[i] {
            ensure!(
                receipt.glyph == *glyph,
                "compilation and validation evidence disagree"
            );
        }
    }
    if options.resume && out.join("report.json").exists() {
        ensure!(pending.is_empty(), "completed run has missing receipts");
        let saved: Value = serde_json::from_slice(&fs::read(out.join("report.json"))?)?;
        let artifacts = saved["artifacts"]
            .as_object()
            .context("missing artifact receipts")?;
        // Only the fixed output names are read; a journal cannot redirect I/O.
        let names: &[&str] = if options.mode == Mode::Full {
            &["catalog.json", "validation.json", "atlas.png"]
        } else {
            &["catalog.json"]
        };
        ensure!(
            artifacts.len() == names.len(),
            "invalid artifact receipt set"
        );
        for &name in names {
            ensure!(
                artifacts[name].as_str() == Some(&sha(&fs::read(out.join(name))?)),
                "export checksum differs"
            );
        }
        return Ok(
            json!({"status":"passed","replayed":true,"provider_calls":0,"reused_glyphs":options.glyphs,"saved_report":saved}),
        );
    }
    execution.outputs = execution
        .outputs
        .into_iter()
        .enumerate()
        .filter_map(|(i, g)| pending.binary_search(&i).is_ok().then_some(g))
        .collect();
    let compilation_wall_s = compilation_started.elapsed().as_secs_f64();
    drop(swarm);
    let validation_started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let fresh: Vec<Batch> = pool.install(|| {
        execution
            .outputs
            .par_chunks(CHECKPOINT_BATCH)
            .enumerate()
            .map(|(batch_index, glyphs)| {
                let receipts = glyphs
                    .iter()
                    .map(|glyph| {
                        let measurements = if options.mode == Mode::Full {
                            Verifier.validate(glyph)?.0
                        } else {
                            vec![]
                        };
                        let record = Receipt {
                            glyph: glyph.clone(),
                            measurements,
                        };
                        Ok(record)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let batch = Batch {
                    indices: pending[batch_index * CHECKPOINT_BATCH
                        ..(batch_index * CHECKPOINT_BATCH + receipts.len())]
                        .to_vec(),
                    receipts,
                };
                // One fsync commits a self-contained batch, including every SVG.
                // Derived files need no independent fsync and are reconstructible.
                publish(
                    &out.join("receipts")
                        .join(format!("batch-{:03}.json", batch.indices[0])),
                    &json_bytes(&batch)?,
                )?;
                for record in &batch.receipts {
                    publish_bytes(
                        &out.join(format!("{}.svg", record.glyph.glyph_id)),
                        record.glyph.svg.as_bytes(),
                        false,
                    )?;
                }
                Ok(batch)
            })
            .collect::<Result<_>>()
    })?;
    for batch in fresh {
        for (i, record) in batch.indices.into_iter().zip(batch.receipts) {
            records[i] = Some(record);
        }
    }
    let records: Vec<Receipt> = records.into_iter().map(Option::unwrap).collect();
    let pixels = if options.mode == Mode::Full {
        pool.install(|| {
            records
                .par_iter()
                .map(|r| validation::render(&r.glyph, 128))
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        vec![]
    };
    let distinctness_started = Instant::now();
    let distinctness = if options.mode == Mode::Full {
        let result = validation::distinctness(&pixels, &pool)?;
        ensure!(
            result.near_matches.is_empty(),
            "catalog contains near-duplicate silhouettes"
        );
        Some(result)
    } else {
        None
    };
    let distinctness_wall_s = distinctness_started.elapsed().as_secs_f64();
    let validation_wall_s = validation_started.elapsed().as_secs_f64();
    let export_started = Instant::now();
    let glyphs: Vec<_> = records
        .iter()
        .map(|r| {
            json!({"glyph_id":r.glyph.glyph_id,"family":r.glyph.recipe,
        "file":format!("{}.svg",r.glyph.glyph_id),"svg_sha256":r.glyph.svg_sha256,
        "pixel_sha256":r.measurements.last().map(|m| &m.pixel_sha256)})
        })
        .collect();
    let catalog = json!({"version":PROTOCOL,"count":records.len(),"glyph_count":records.len(),"canvas":[100,100],"glyphs":glyphs});
    let mut artifacts = serde_json::Map::new();
    let bytes = json_bytes(&catalog)?;
    publish(&out.join("catalog.json"), &bytes)?;
    artifacts.insert("catalog.json".into(), sha(&bytes).into());
    if options.mode == Mode::Full {
        let validation = json!({"gate_version":validation::GATE_VERSION,"raster_version":validation::RASTER_VERSION,
            "distinctness":distinctness,"glyphs":records.iter().map(|r| json!({"glyph_id":r.glyph.glyph_id,"measurements":r.measurements})).collect::<Vec<_>>()});
        let bytes = json_bytes(&validation)?;
        publish(&out.join("validation.json"), &bytes)?;
        artifacts.insert("validation.json".into(), sha(&bytes).into());
        let columns = records.len().min(16);
        let mut atlas = tiny_skia::Pixmap::new(
            (columns * 128) as u32,
            (records.len().div_ceil(columns) * 128) as u32,
        )
        .context("atlas allocation failed")?;
        atlas.fill(tiny_skia::Color::WHITE);
        for (i, pixels) in pixels.iter().enumerate() {
            for y in 0..128 {
                let start = ((i / columns * 128 + y) * columns * 128 + i % columns * 128) * 4;
                atlas.data_mut()[start..start + 128 * 4]
                    .copy_from_slice(&pixels.data()[y * 128 * 4..(y + 1) * 128 * 4]);
            }
        }
        let bytes = atlas.encode_png()?;
        publish(&out.join("atlas.png"), &bytes)?;
        artifacts.insert("atlas.png".into(), sha(&bytes).into());
    }
    let export_wall_s = export_started.elapsed().as_secs_f64();
    drop(pool);
    let report = json!({"protocol":PROTOCOL,"status":"passed","mode":options.mode,"glyphs":records.len(),
        "workers":options.workers,"provider_calls":execution.trace.spans.len(),"reused_glyphs":reused,
        "trace":execution.trace,
        "api_calls":0,"cost_nanousd":smythe::NanoUsd(execution.total_cost_nanousd),"compilation_wall_s":compilation_wall_s,
        "validation_and_receipts_wall_s":validation_wall_s,"export_wall_s":export_wall_s,
        "end_to_end_wall_s":started.elapsed().as_secs_f64(),"artifacts":artifacts,
        "source_sha256":binding.source_sha256,"compiler_version":binding.compiler_version,"raster_version":binding.raster_version,"gate_version":binding.gate_version,
        "compared_pairs":distinctness.map(|d|d.compared_pairs),"claimable":false,"checkpoint_batch_size":CHECKPOINT_BATCH,
        "distinctness_wall_s":distinctness_wall_s,
        "implementation_sources":geometry::implementation_sources(),"optimized":!cfg!(debug_assertions)});
    publish(&out.join("report.json"), &json_bytes(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> Options {
        Options {
            glyphs: 4,
            workers: 2,
            out: std::env::temp_dir().join(format!(
                "smythe-native-test-{}-{}",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            )),
            resume: false,
            mode: Mode::Full,
        }
    }
    fn compilation_checkpoint(options: &Options, changed: bool) {
        struct SavedCompiler<'a> {
            inner: ContourProvider<'a>,
            changed: bool,
        }
        impl Provider for SavedCompiler<'_> {
            type Output = Glyph;
            fn effect(&self) -> smythe::Effect {
                smythe::Effect::Pure
            }
            fn binding(&self) -> String {
                self.inner.binding()
            }
            fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
                Ok(0)
            }
            fn complete(&self, task: &Task, deps: &[&Glyph]) -> Result<CompletionResult<Glyph>> {
                let mut result = self.inner.complete(task, deps)?;
                if self.changed {
                    result.output.recipe = "old-compiler-result".into();
                }
                Ok(result)
            }
        }
        fs::create_dir(&options.out).unwrap();
        publish(
            &options.out.join("run.json"),
            &json_bytes(&Binding::for_options(options)).unwrap(),
        )
        .unwrap();
        let recipes = geometry::recipes().unwrap();
        let provider = SavedCompiler {
            inner: ContourProvider { recipes: &recipes },
            changed,
        };
        let store = smythe::store::Store::open(options.out.join(".smythe")).unwrap();
        Swarm::new(2, 0)
            .unwrap()
            .execute_in(
                &graph(&(0..options.glyphs).collect::<Vec<_>>()),
                &provider,
                &store,
                "noumenon-compile",
                false,
            )
            .unwrap();
        assert!(!options.out.join("receipts").exists());
    }

    #[test]
    fn recovery_checks_compilation_outputs_before_any_batch_exists() {
        for changed in [false, true] {
            let mut o = options();
            o.mode = Mode::Compile;
            compilation_checkpoint(&o, changed);
            o.resume = true;
            let result = run(&o);
            if changed {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("compilation output differs")
                );
                assert!(!o.out.join("catalog.json").exists());
            } else {
                assert_eq!(result.unwrap()["provider_calls"], 0);
                assert!(o.out.join("catalog.json").exists());
            }
            fs::remove_dir_all(o.out).unwrap();
        }
    }

    #[test]
    fn resume_binds_semantics_and_keeps_source_provenance_informational() {
        let mut o = options();
        o.mode = Mode::Compile;
        run(&o).unwrap();
        let binding: Value =
            serde_json::from_slice(&fs::read(o.out.join("run.json")).unwrap()).unwrap();
        assert!(binding.get("implementation_sha256").is_none());
        // Simulate a report written by a build with unrelated CLI source bytes.
        let path = o.out.join("report.json");
        let mut report: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        report["implementation_sources"]["crates/smythe-cli/src/main.rs"] =
            "unrelated-source-digest".into();
        fs::write(&path, json_bytes(&report).unwrap()).unwrap();
        o.resume = true;
        assert_eq!(run(&o).unwrap()["provider_calls"], 0);
        let mut changed = binding;
        changed["compiler_version"] = "different-semantics".into();
        fs::write(o.out.join("run.json"), json_bytes(&changed).unwrap()).unwrap();
        assert!(run(&o).unwrap_err().to_string().contains("recipe differs"));
        fs::remove_dir_all(o.out).unwrap();
    }

    #[test]
    #[ignore = "release-mode performance gate; explicitly run in native CI"]
    fn release_compile_256_has_bounded_journal_cost() {
        assert!(!cfg!(debug_assertions), "run this gate with --release");
        let mut o = options();
        o.glyphs = 256;
        o.workers = 8;
        o.mode = Mode::Compile;
        let report = run(&o).unwrap();
        let store = smythe::store::Store::open(o.out.join(".smythe")).unwrap();
        let records = store.history("noumenon-compile").unwrap();
        let transactions = records
            .iter()
            .map(|r| r.transaction_start)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert_eq!(records.len(), 1283);
        // Even an adversarial completion schedule must group the initial eight admissions.
        assert!(transactions <= 508, "admissions were not grouped");
        let seconds = report["compilation_wall_s"].as_f64().unwrap();
        let limit = if cfg!(windows) { 30.0 } else { 15.0 };
        assert!(
            seconds < limit,
            "256-glyph disk compilation took {seconds:.3}s (limit {limit}s)"
        );
        println!(
            "{}",
            json!({"compile_seconds":seconds,"events":records.len(),"transactions":transactions,"claimable":false})
        );
        drop(store);
        fs::remove_dir_all(o.out).unwrap();
    }
    #[test]
    fn completed_resume_checks_hashes_without_dispatch() {
        let mut o = options();
        let r = run(&o).unwrap();
        assert_eq!(r["provider_calls"], 4);
        assert_eq!(r["compared_pairs"], 6);
        assert!(run(&o).is_err());
        o.resume = true;
        assert_eq!(run(&o).unwrap()["provider_calls"], 0);
        fs::write(o.out.join("GLYPH-000.svg"), "broken").unwrap();
        assert!(run(&o).is_err());
        fs::remove_dir_all(o.out).unwrap();
    }
    #[test]
    fn missing_terminal_export_resumes_and_mismatched_recipe_is_refused() {
        let mut o = options();
        run(&o).unwrap();
        fs::remove_file(o.out.join("report.json")).unwrap();
        fs::remove_file(o.out.join("GLYPH-000.svg")).unwrap();
        o.resume = true;
        let r = run(&o).unwrap();
        assert_eq!(r["reused_glyphs"], 4);
        assert_eq!(r["provider_calls"], 0);
        assert!(o.out.join("GLYPH-000.svg").is_file());
        o.glyphs = 5;
        assert!(run(&o).is_err());
        fs::remove_dir_all(o.out).unwrap();
    }

    #[test]
    fn receipt_metadata_is_bound_and_concurrent_resume_is_fenced() {
        let mut o = options();
        run(&o).unwrap();
        o.resume = true;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(o.out.join(".lock"))
            .unwrap();
        lock.try_lock().unwrap();
        assert!(run(&o).is_err());
        drop(lock);
        let path = o.out.join("receipts/batch-000.json");
        let mut batch: Batch = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        batch.receipts[0].glyph.recipe = "changed".into();
        fs::write(path, json_bytes(&batch).unwrap()).unwrap();
        assert!(run(&o).is_err());
        fs::remove_dir_all(o.out).unwrap();
    }
    #[test]
    fn completed_resume_refuses_missing_or_corrupt_compilation_evidence() {
        let mut o = options();
        o.mode = Mode::Compile;
        run(&o).unwrap();
        o.resume = true;
        let blob = fs::read_dir(o.out.join(".smythe/blobs"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let original = fs::read(&blob).unwrap();
        fs::write(&blob, b"corrupt").unwrap();
        assert!(run(&o).unwrap_err().to_string().contains("blob checksum"));
        fs::write(&blob, original).unwrap();
        assert_eq!(run(&o).unwrap()["provider_calls"], 0);
        fs::remove_file(o.out.join(".smythe/journal.sqlite3")).unwrap();
        assert!(
            run(&o)
                .unwrap_err()
                .to_string()
                .contains("compilation journal")
        );
        fs::remove_dir_all(o.out).unwrap();
    }
}
