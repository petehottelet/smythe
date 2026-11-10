#![allow(clippy::disallowed_methods)] // Report timing only.
use crate::geometry::{self, Glyph, Recipe, sha};
use crate::validation::{self, Measurement, Verifier};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use smythe::{
    CompletionResult, Dependency, EngineError, ExecutionGraph, Node, Provider, Swarm, Task,
    Topology,
};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const PROTOCOL: &str = "noumenon-rust-journal-v3";
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
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    glyph: Glyph,
    measurements: Vec<Measurement>,
}
/// Immutable publication for process-crash recovery. OS/power-loss durability
/// is not claimed; the journal is the receipt authority for both engine phases.
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        ensure!(fs::read(path)? == bytes, EngineError::Integrity);
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
    fn complete(
        &self,
        task: &Task,
        _: &[Dependency<'_, Glyph>],
    ) -> Result<CompletionResult<Glyph>> {
        let i = task
            .input
            .as_u64()
            .context("glyph task requires an integer index")? as usize;
        Ok(CompletionResult {
            output: geometry::compile(
                i,
                self.recipes.get(i).context("glyph index outside catalog")?,
            )?,
            cost_nanousd: 0,
        })
    }
}
struct ValidationProvider<'a> {
    glyphs: &'a [Glyph],
}
impl Provider for ValidationProvider<'_> {
    type Output = Receipt;
    fn effect(&self) -> smythe::Effect {
        smythe::Effect::Pure
    }
    fn binding(&self) -> String {
        format!(
            "{}:{}:{}",
            validation::RASTER_VERSION,
            validation::GATE_VERSION,
            sha(self
                .glyphs
                .iter()
                .flat_map(|g| g.svg_sha256.bytes())
                .collect::<Vec<_>>()
                .as_slice())
        )
    }
    fn cost_bound_nanousd(&self, _: &Task) -> Result<u64> {
        Ok(0)
    }
    fn complete(
        &self,
        task: &Task,
        _: &[Dependency<'_, Receipt>],
    ) -> Result<CompletionResult<Receipt>> {
        let i = task
            .input
            .as_u64()
            .context("validation requires a glyph index")? as usize;
        let glyph = self
            .glyphs
            .get(i)
            .context("validation index outside catalog")?;
        Ok(CompletionResult {
            output: Receipt {
                glyph: glyph.clone(),
                measurements: Verifier.validate(glyph)?.0,
            },
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
                phase: Some("compile".into()),
            })
            .collect(),
    }
}
fn validation_graph(count: usize) -> ExecutionGraph {
    let mut result = graph(&(0..count).collect::<Vec<_>>());
    for (i, n) in result.nodes.iter_mut().enumerate() {
        n.id = format!("validate-{i:03}");
        n.phase = Some("validation".into());
        n.task.goal = format!("Validate Noumenon glyph {i}");
    }
    result
}
pub fn run(options: &Options) -> Result<Value> {
    ensure!(
        (1..=256).contains(&options.glyphs),
        EngineError::InvalidCommand
    );
    ensure!(
        (1..=64).contains(&options.workers),
        EngineError::InvalidCommand
    );
    let all = geometry::recipes()?;
    let binding = Binding::for_options(options);
    let started = Instant::now();
    let out = &options.out;
    if options.resume {
        ensure!(out.is_dir(), EngineError::RunMissing);
    } else {
        fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
        fs::create_dir(out).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                anyhow::Error::new(EngineError::RunExists)
            } else {
                e.into()
            }
        })?;
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(out.join(".lock"))?;
    lock.try_lock().context("another process owns this run")?;
    if options.resume {
        let saved: Binding = serde_json::from_slice(
            &fs::read(out.join("run.json")).context(EngineError::Integrity)?,
        )
        .context(EngineError::Integrity)?;
        ensure!(
            saved == binding,
            anyhow::Error::new(EngineError::InvalidCommand)
                .context("resume recipe differs from the saved run")
        );
        ensure!(
            out.join(".smythe/journal.sqlite3").exists()
                || (!out.join("catalog.json").exists() && !out.join("GLYPH-000.svg").exists()),
            "published outputs require their compilation journal"
        );
    } else {
        publish(&out.join("run.json"), &json_bytes(&binding)?)?;
    }
    let journal = smythe::store::Store::open(out.join(".smythe"))?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let compilation_started = Instant::now();
    let swarm = Swarm::new(options.workers, 0)?;
    let engine_resume = !journal.history("noumenon-compile")?.is_empty();
    let execution = swarm.execute_in(
        &graph(&(0..options.glyphs).collect::<Vec<_>>()),
        &ContourProvider { recipes: &all },
        &journal,
        "noumenon-compile",
        engine_resume,
    )?;
    let compiled = execution.load_all::<Glyph>()?;
    pool.install(|| {
        compiled
            .par_iter()
            .enumerate()
            .try_for_each(|(i, glyph)| -> Result<()> {
                ensure!(
                    *glyph == geometry::compile(i, &all[i])?,
                    "compilation output differs from authored contours"
                );
                Ok(())
            })
    })?;
    let reused = execution.reused_nodes;
    let compilation_wall_s = compilation_started.elapsed().as_secs_f64();
    let validation_started = Instant::now();
    let mut validation_calls = 0;
    let mut reused_validation = 0;
    let records: Vec<Receipt> = if options.mode == Mode::Full {
        let validation_resume = !journal.history("noumenon-validate")?.is_empty();
        let validation = swarm.execute_in(
            &validation_graph(options.glyphs),
            &ValidationProvider { glyphs: &compiled },
            &journal,
            "noumenon-validate",
            validation_resume,
        )?;
        validation_calls = validation.trace.spans.len();
        reused_validation = validation.reused_nodes;
        let receipts = validation.load_all::<Receipt>()?;
        pool.install(|| {
            receipts
                .par_iter()
                .zip(&compiled)
                .try_for_each(|(r, g)| -> Result<()> {
                    ensure!(
                        r.glyph == *g,
                        "compilation and validation evidence disagree"
                    );
                    if validation_resume {
                        ensure!(
                            Verifier.validate(g)?.0 == r.measurements,
                            "saved raster receipt differs"
                        );
                    }
                    Ok(())
                })
        })?;
        receipts
    } else {
        compiled
            .into_iter()
            .map(|glyph| Receipt {
                glyph,
                measurements: vec![],
            })
            .collect()
    };
    // Derived exports can be recreated from the engine's accepted immutable outputs.
    pool.install(|| {
        records.par_iter().try_for_each(|r| {
            publish(
                &out.join(format!("{}.svg", r.glyph.glyph_id)),
                r.glyph.svg.as_bytes(),
            )
        })
    })?;
    if options.resume && out.join("report.json").exists() {
        let saved: Value = serde_json::from_slice(&fs::read(out.join("report.json"))?)?;
        let artifacts = saved["artifacts"]
            .as_object()
            .context("missing artifact receipts")?;
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
            json!({"status":"passed","replayed":true,"provider_calls":execution.trace.spans.len(),"validation_provider_calls":validation_calls,"reused_glyphs":reused,"reused_validation":reused_validation,"saved_report":saved}),
        );
    }
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
        "compared_pairs":distinctness.map(|d|d.compared_pairs),"claimable":false,"journal_schema":3, "validation_provider_calls":validation_calls, "reused_validation":reused_validation,
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
            fn complete(
                &self,
                task: &Task,
                deps: &[smythe::Dependency<'_, Glyph>],
            ) -> Result<CompletionResult<Glyph>> {
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
        let replay = run(&o).unwrap();
        assert_eq!(replay["provider_calls"], 0);
        assert_eq!(replay["validation_provider_calls"], 0);
        assert_eq!(replay["reused_validation"], 4);
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
        assert!(!o.out.join("receipts").exists());
        let store = smythe::store::Store::open_read_only(o.out.join(".smythe")).unwrap();
        let state = store.inspect("noumenon-validate").unwrap();
        assert!(state.completed());
        let reference = state.attempts["validate-000"].output.as_ref().unwrap();
        let conn = rusqlite::Connection::open(o.out.join(".smythe/journal.sqlite3")).unwrap();
        conn.execute(
            "UPDATE blobs SET bytes=? WHERE digest=?",
            rusqlite::params![b"corrupt".as_slice(), reference.sha256],
        )
        .unwrap();
        drop(conn);
        drop(store);
        assert!(run(&o).is_err());
        fs::remove_dir_all(o.out).unwrap();
    }
    #[test]
    fn completed_resume_refuses_missing_or_corrupt_compilation_evidence() {
        let mut o = options();
        o.mode = Mode::Compile;
        run(&o).unwrap();
        o.resume = true;
        let conn = rusqlite::Connection::open(o.out.join(".smythe/journal.sqlite3")).unwrap();
        let (hash, original): (String, Vec<u8>) = conn
            .query_row("SELECT digest,bytes FROM blobs LIMIT 1", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        conn.execute(
            "UPDATE blobs SET bytes=? WHERE digest=?",
            rusqlite::params![b"corrupt".as_slice(), hash],
        )
        .unwrap();
        assert!(run(&o).is_err());
        conn.execute(
            "UPDATE blobs SET bytes=? WHERE digest=?",
            rusqlite::params![original, hash],
        )
        .unwrap();
        drop(conn);
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
