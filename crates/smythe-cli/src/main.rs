use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use smythe::{EngineError, ExecutionGraph, Limits, NanoUsd, OfflineProvider, Swarm, store::Store};
use smythe_noumenon::pipeline::{self, Mode, Options};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "smythe",
    version,
    about = "Smythe: execution graphs with journaled recovery"
)]
struct Cli {
    /// Emit machine-readable output (also the default).
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Args, Default)]
struct LimitsArgs {
    #[arg(long)]
    max_nodes: Option<usize>,
    #[arg(long)]
    max_edges: Option<usize>,
    #[arg(long)]
    max_attempts_per_node: Option<u64>,
    #[arg(long)]
    max_output_bytes: Option<u64>,
    #[arg(long)]
    max_total_output_bytes: Option<u64>,
    /// Absolute UTC deadline, in milliseconds since Unix epoch.
    #[arg(long)]
    wall_deadline_ms: Option<u64>,
}
impl LimitsArgs {
    fn specified(&self) -> bool {
        self.max_nodes.is_some()
            || self.max_edges.is_some()
            || self.max_attempts_per_node.is_some()
            || self.max_output_bytes.is_some()
            || self.max_total_output_bytes.is_some()
            || self.wall_deadline_ms.is_some()
    }
    fn apply(&self, mut limits: Limits) -> Limits {
        if let Some(n) = self.max_nodes {
            limits.max_nodes = n;
        }
        if let Some(n) = self.max_edges {
            limits.max_edges = n;
        }
        if let Some(n) = self.max_attempts_per_node {
            limits.max_attempts_per_node = n;
        }
        if let Some(n) = self.max_output_bytes {
            limits.max_output_bytes = n;
        }
        if let Some(n) = self.max_total_output_bytes {
            limits.max_total_output_bytes = n;
        }
        if let Some(n) = self.wall_deadline_ms {
            limits.wall_deadline_ms = Some(n);
        }
        limits
    }
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
        /// Initial allowance. On resume, omit this to use the journaled ceiling.
        #[arg(long)]
        budget_nanousd: Option<u64>,
        #[arg(long,value_enum,default_value_t=Outputs::References)]
        outputs: Outputs,
        #[command(flatten)]
        limits: LimitsArgs,
    },
    /// Verify the complete journal, or read a bounded page of events.
    Inspect {
        id: String,
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
        #[arg(long)]
        events: bool,
        #[arg(long, requires = "events")]
        after: Option<u64>,
        #[arg(long, default_value_t = 1000, requires = "events")]
        limit: usize,
    },
    /// List existing runs without replaying their journals.
    List {
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
    },
    /// Load one accepted output and verify its hash and length.
    Output {
        id: String,
        node: String,
        #[arg(long, default_value = ".smythe")]
        store: PathBuf,
    },
    /// Record an explicit operator decision.
    Control {
        id: String,
        #[arg(long, global = true, default_value = ".smythe")]
        store: PathBuf,
        #[arg(value_enum)]
        action: Action,
        /// Node ID for retry, or nanoUSD ceiling for budget.
        value: Option<String>,
    },
    /// Generate the authored glyph catalog without network calls.
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
enum Outputs {
    References,
    Inline,
}
#[derive(Clone, Copy, ValueEnum)]
enum Action {
    Pause,
    Stop,
    Continue,
    Retry,
    Budget,
}
fn execute(command: Command) -> Result<Value> {
    match command {
        Command::Run {
            graph,
            store,
            id,
            resume,
            workers,
            budget_nanousd,
            outputs,
            limits,
        } => {
            let text = std::fs::read_to_string(&graph).context(EngineError::InvalidCommand)?;
            let graph: ExecutionGraph = if graph.extension().is_some_and(|e| e == "toml") {
                toml::from_str(&text).context(EngineError::InvalidCommand)?
            } else {
                serde_json::from_str(&text).context(EngineError::InvalidCommand)?
            };
            ensure!(!resume || id.is_some(), EngineError::InvalidCommand);
            let id = id.context(EngineError::InvalidCommand)?;
            smythe::store::validate_id(&id).context(EngineError::InvalidCommand)?;
            let store = if resume {
                Store::open_existing(store)?
            } else {
                Store::open(store)?
            };
            let saved = if resume {
                Some(store.inspect(&id)?)
            } else {
                None
            };
            let budget =
                budget_nanousd.unwrap_or_else(|| saved.as_ref().map_or(0, |s| s.budget.ceiling.0));
            let mut swarm = Swarm::new(workers, budget)?;
            if limits.specified() {
                swarm = swarm.with_limits(
                    limits.apply(
                        saved
                            .as_ref()
                            .map_or_else(Limits::default, |s| s.limits.clone()),
                    ),
                );
            }
            let result = swarm.execute_in(&graph, &OfflineProvider, &store, &id, resume)?;
            let output = match outputs {
                Outputs::References => serde_json::to_value(&result.outputs)?,
                Outputs::Inline => serde_json::to_value(result.load_all::<Value>()?)?,
            };
            Ok(
                json!({"schema_version":1,"status":"passed","run_id":result.run_id,"outputs":output,"cost_nanousd":NanoUsd(result.total_cost_nanousd),"overrun":result.overrun,"reused_nodes":result.reused_nodes,"provider_calls":result.trace.spans.len(),"output_bytes":result.output_bytes,"dependency_bytes":result.dependency_bytes,"loaded_dependency_bytes":result.loaded_dependency_bytes}),
            )
        }
        Command::Inspect {
            id,
            store,
            events,
            after,
            limit,
        } => {
            smythe::store::validate_id(&id).context(EngineError::InvalidCommand)?;
            let store = Store::open_read_only(store)?;
            if let Some(after) = after {
                ensure!((1..=10000).contains(&limit), EngineError::InvalidCommand);
                let records = store.events_after(&id, after, limit)?;
                return Ok(
                    json!({"schema_version":1,"run_id":id,"next_sequence":records.last().map_or(after,|r|r.sequence),"events":records}),
                );
            }
            let state = store.inspect(&id)?;
            Ok(
                json!({"schema_version":1,"run_id":id,"status":state.status(),"health":store.health(&state)?,"shape":state.graph.shape()?,"state":state,"events":if events {Some(store.history(&id)?)}else{None}}),
            )
        }
        Command::List { store } => {
            Ok(json!({"schema_version":1,"runs":Store::open_read_only(store)?.list()?}))
        }
        Command::Output { id, node, store } => {
            smythe::store::validate_id(&id).context(EngineError::InvalidCommand)?;
            let store = Store::open_read_only(store)?;
            let state = store.inspect(&id)?;
            let a = state.attempts.get(&node).context(EngineError::RunMissing)?;
            ensure!(
                a.state == smythe_core::journal::AttemptState::Accepted,
                EngineError::Integrity
            );
            serde_json::from_slice(
                &store.output(a.output.as_ref().context(EngineError::Integrity)?)?,
            )
            .context(EngineError::Integrity)
        }
        Command::Control {
            id,
            store,
            action,
            value,
        } => {
            ensure!(
                matches!(action, Action::Retry | Action::Budget) == value.is_some(),
                EngineError::InvalidCommand
            );
            smythe::store::validate_id(&id).context(EngineError::InvalidCommand)?;
            let store = Store::open_existing(store)?;
            let ack = match action {
                Action::Pause => store.control(&id, smythe::Control::Pause)?,
                Action::Stop => store.control(&id, smythe::Control::Stop)?,
                Action::Continue => store.control(&id, smythe::Control::Run)?,
                Action::Retry => store.retry(&id, value.as_deref().unwrap())?,
                Action::Budget => store.change_ceiling(
                    &id,
                    NanoUsd(
                        value
                            .as_deref()
                            .unwrap()
                            .parse()
                            .context(EngineError::InvalidCommand)?,
                    ),
                )?,
            };
            Ok(
                json!({"schema_version":1,"run_id":id,"acknowledgement":"durably_recorded","sequence":ack.last}),
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
fn failure(run_id: Option<&str>, class: EngineError, message: &str) -> ! {
    println!(
        "{}",
        json!({"schema_version":1,"status":"error","run_id":run_id,"error":{"class":class.class(),"message":message}})
    );
    eprintln!("Smythe: {message}");
    std::process::exit(class.exit_code());
}
fn main() {
    let mut cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            let _ = error.print();
            return;
        }
        Err(error) => failure(None, EngineError::InvalidCommand, &error.to_string()),
    };
    if let Command::Run {
        id, resume: false, ..
    } = &mut cli.command
        && id.is_none()
    {
        *id = Some(smythe::seams::Sources::default().ids.next_id());
    }
    let run_id = match &cli.command {
        Command::Run { id, .. } => id.clone(),
        Command::Inspect { id, .. } | Command::Output { id, .. } | Command::Control { id, .. } => {
            Some(id.clone())
        }
        _ => None,
    };
    match execute(cli.command) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => {
            let class = error
                .downcast_ref::<EngineError>()
                .copied()
                .unwrap_or(EngineError::ProviderFailed);
            let message = if error.downcast_ref::<EngineError>().is_some() {
                error.to_string()
            } else {
                class.to_string()
            };
            failure(run_id.as_deref(), class, &message)
        }
    }
}
