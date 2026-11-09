use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use smythe::{ExecutionGraph, OfflineProvider, Swarm, store::Store};
use smythe_noumenon::pipeline::{self, Mode, Options};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "smythe",
    version,
    about = "Smythe: execution graphs with journaled recovery"
)]
struct Cli {
    /// Emit machine-readable output (currently the default).
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute a JSON or TOML graph with the local offline provider.
    Run {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long, default_value_t = 0)]
        budget_nanousd: u64,
    },
    /// Rebuild a run's state from its durable journal.
    Inspect {
        id: String,
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
        #[arg(long)]
        events: bool,
    },
    /// Record pause, stop or continue; the acknowledgement does not mean draining ended.
    Control {
        id: String,
        #[arg(value_enum)]
        action: Action,
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
    },
    /// Run the deterministic authored-contour workflow without network calls.
    Noumenon {
        #[arg(long, default_value_t = 192)]
        glyphs: usize,
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long,value_enum,default_value_t=Mode::Full)]
        mode: Mode,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Action {
    Pause,
    Stop,
    Continue,
}

fn execute(command: Command) -> Result<serde_json::Value> {
    match command {
        Command::Run {
            graph,
            store,
            id,
            resume,
            workers,
            budget_nanousd,
        } => {
            let bytes = std::fs::read_to_string(&graph)?;
            let graph: ExecutionGraph = if graph.extension().is_some_and(|e| e == "toml") {
                toml::from_str(&bytes)?
            } else {
                serde_json::from_str(&bytes)?
            };
            anyhow::ensure!(!resume || id.is_some(), "resume requires --id");
            let id = id.unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
            let result = Swarm::new(workers, budget_nanousd)?.execute_in(
                &graph,
                &OfflineProvider,
                &Store::open(store)?,
                &id,
                resume,
            )?;
            Ok(
                json!({"schema_version":1,"status":"passed","run_id":result.run_id,"outputs":result.outputs,"cost_nanousd":smythe::NanoUsd(result.total_cost_nanousd),"reused_nodes":result.reused_nodes,"provider_calls":result.trace.spans.len()}),
            )
        }
        Command::Inspect { id, store, events } => {
            let store = Store::open(store)?;
            let state = store.inspect(&id).context("cannot inspect run")?;
            Ok(
                json!({"schema_version":1,"run_id":id,"state":state,"events":if events {Some(store.history(&id)?)}else{None}}),
            )
        }
        Command::Control { id, store, action } => {
            let control = match action {
                Action::Pause => smythe::Control::Pause,
                Action::Stop => smythe::Control::Stop,
                Action::Continue => smythe::Control::Run,
            };
            let state = Store::open(store)?.control(&id, control)?;
            Ok(
                json!({"schema_version":1,"run_id":id,"acknowledgement":"durably_recorded","sequence":state.sequence,"control":state.control}),
            )
        }
        Command::Noumenon {
            glyphs,
            workers,
            out,
            resume,
            mode,
        } => pipeline::run(&Options {
            glyphs,
            workers,
            out,
            resume,
            mode,
        }),
    }
}

fn main() {
    let Cli { command, .. } = Cli::parse();
    match execute(command) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => {
            eprintln!("Smythe: {error:#}");
            std::process::exit(1);
        }
    }
}
