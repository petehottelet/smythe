use clap::error::{ContextKind, ContextValue, ErrorKind};
use smythe_core::diagnostics::bounded;

pub fn arguments(error: &clap::Error) -> String {
    let reason = match error.kind() {
        ErrorKind::MissingRequiredArgument => "missing required argument",
        ErrorKind::UnknownArgument => "unrecognized argument",
        ErrorKind::InvalidSubcommand | ErrorKind::MissingSubcommand => {
            "invalid or missing subcommand"
        }
        ErrorKind::ArgumentConflict => "conflicting arguments",
        _ => "invalid argument value",
    };
    let candidates: Vec<&str> = match error.get(ContextKind::InvalidArg) {
        Some(ContextValue::String(s)) => vec![s],
        Some(ContextValue::Strings(ss)) => ss.iter().map(String::as_str).collect(),
        _ => vec![],
    };
    let names: Vec<_> = candidates
        .into_iter()
        .filter_map(|s| {
            let name = s.split_whitespace().next()?;
            [
                "--graph",
                "--store",
                "--id",
                "--resume",
                "--workers",
                "--budget-nanousd",
                "--outputs",
                "--max-nodes",
                "--max-edges",
                "--max-attempts-per-node",
                "--max-output-bytes",
                "--max-total-output-bytes",
                "--wall-deadline-ms",
                "--events",
                "--after",
                "--limit",
                "--glyphs",
                "--out",
                "--mode",
                "<ID>",
                "<NODE>",
                "<ACTION>",
            ]
            .contains(&name)
            .then_some(name)
        })
        .collect();
    bounded(&format!(
        "{reason}{}; see --help",
        if names.is_empty() {
            String::new()
        } else {
            format!(": {}", names.join(", "))
        }
    ))
}
