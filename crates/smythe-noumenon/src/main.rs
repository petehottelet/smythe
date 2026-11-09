use clap::{Parser, Subcommand};
use smythe_noumenon::pipeline::{self, Mode, Options};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "smythe",
    version,
    about = "Smythe native preview: Noumenon glyph compilation, validation and export"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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

fn main() {
    let Cli {
        command:
            Command::Noumenon {
                glyphs,
                workers,
                out,
                resume,
                mode,
            },
    } = Cli::parse();
    match pipeline::run(&Options {
        glyphs,
        workers,
        out,
        resume,
        mode,
    }) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => {
            eprintln!("Smythe: {error:#}");
            std::process::exit(1);
        }
    }
}
