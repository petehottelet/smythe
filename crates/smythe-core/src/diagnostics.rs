//! Public diagnostics retain recognized reasons, never arbitrary cause text.
use crate::EngineError;
use std::{error::Error, fmt};

pub const MAX_DIAGNOSTIC_BYTES: usize = 1024;
const TRUNCATED: &str = " [truncated]";

/// Bound already-safe text by UTF-8 bytes. This is not a redactor.
pub fn bounded(text: &str) -> String {
    if text.len() <= MAX_DIAGNOSTIC_BYTES {
        return text.into();
    }
    let mut end = MAX_DIAGNOSTIC_BYTES - TRUNCATED.len();
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{TRUNCATED}", &text[..end])
}

#[derive(Clone, Copy, Debug)]
pub enum GraphFormat {
    Json,
    Toml,
}

/// Contains only a parser location and a sanitized schema diagnostic.
#[derive(Debug)]
pub struct GraphDiagnostic(String);
impl GraphDiagnostic {
    pub fn new(format: GraphFormat, parser_message: &str, line: usize, column: usize) -> Self {
        let format = match format {
            GraphFormat::Json => "JSON",
            GraphFormat::Toml => "TOML",
        };
        let mut detail = "invalid graph syntax or value".to_owned();
        for kind in ["unknown field", "missing field", "duplicate field"] {
            if let Some(rest) = parser_message.strip_prefix(kind)
                && let Some(rest) = rest.strip_prefix(" `")
                && let Some((field, _)) = rest.split_once('`')
            {
                detail = format!("{kind} `{}`", field_name(field));
                break;
            }
        }
        Self(bounded(&format!(
            "{format} graph at line {line}, column {column}: {detail}"
        )))
    }
}
impl fmt::Display for GraphDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for GraphDiagnostic {}

// Only schema names and one-edit spelling mistakes are useful here. Unknown
// names may themselves be credentials, paths or account identifiers.
fn field_name(name: &str) -> &str {
    let fields = [
        "nodes",
        "topology",
        "id",
        "task",
        "goal",
        "input",
        "depends_on",
        "phase",
    ];
    if name.len() > 16 || !name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') {
        return "redacted field";
    }
    for field in fields {
        let mut row: Vec<usize> = (0..=field.len()).collect();
        for (i, a) in name.bytes().enumerate() {
            let mut previous = row[0];
            row[0] = i + 1;
            for (j, b) in field.bytes().enumerate() {
                let old = row[j + 1];
                row[j + 1] = (row[j] + 1)
                    .min(old + 1)
                    .min(previous + usize::from(a != b));
                previous = old;
            }
        }
        if row[field.len()] <= 1 {
            return name;
        }
    }
    "redacted field"
}

/// Preserve the typed class; inspect a bounded cause chain for safe details.
/// Unsupported text is omitted wholesale, including paths and identifiers.
pub fn message(error: &anyhow::Error) -> String {
    let class = error
        .downcast_ref::<EngineError>()
        .copied()
        .unwrap_or(EngineError::ProviderFailed);
    if let Some(parser) = error.downcast_ref::<GraphDiagnostic>() {
        return bounded(&format!("{class}: {parser}"));
    }
    let mut result = class.to_string();
    for cause in error.chain().take(16) {
        let reason = if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            match io.kind() {
                std::io::ErrorKind::NotFound => "file or directory not found",
                std::io::ErrorKind::PermissionDenied => "filesystem permission denied",
                std::io::ErrorKind::AlreadyExists => "file or directory already exists",
                _ => "filesystem operation failed",
            }
        } else {
            // Match complete literals only. Never publish a suffix, a value,
            // an arbitrary context string, or a parser's source excerpt.
            match cause.to_string().as_str() {
                "graph contains a cycle" => "graph contains a cycle",
                "missing dependency" => "missing dependency",
                "duplicate dependency" => "duplicate dependency",
                "duplicate node ID" => "duplicate node ID",
                "invalid node ID" => "invalid node ID",
                "invalid phase label" => "invalid phase label",
                "graph exceeds the native node limit" => "graph exceeds the native node limit",
                "graph exceeds the edge limit" => "graph exceeds the edge limit",
                "attempt limit must be positive" => "attempt limit must be positive",
                "large integers require decimal strings for canonical JSON" => {
                    "large integers require decimal strings for canonical JSON"
                }
                "another coordinator owns this run" => "another coordinator owns this run",
                "run is busy; retry control command" => "run is busy; retry control command",
                "invalid run identifier" => "invalid run identifier",
                "run binding differs" => "run binding differs",
                "resume uses the journaled limits" => "resume uses the journaled limits",
                "resume uses the journaled budget; change it with smythe control <id> budget <nanousd>" => {
                    "resume uses the journaled budget; change it with smythe control <id> budget <nanousd>"
                }
                _ => continue,
            }
        };
        if !result.contains(reason) {
            result.push_str(": ");
            result.push_str(reason);
        }
    }
    bounded(&result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_untrusted_causes_are_omitted_without_losing_safe_reasons() {
        let secret = ["org", "-", "synthetic", "identifier"].concat();
        let error = anyhow::anyhow!("{secret}")
            .context("graph contains a cycle")
            .context(["/", "home", "/", "fixture/private"].concat())
            .context(EngineError::InvalidCommand);
        let text = message(&error);
        assert!(text.contains("cycle"));
        assert!(!text.contains(&secret) && !text.contains("fixture/private"));
        assert_eq!(
            message(&anyhow::anyhow!("unrecognized sensitive context")),
            EngineError::ProviderFailed.to_string()
        );
    }
    #[test]
    fn unicode_truncation_respects_the_byte_bound_and_marks_truncation() {
        let text = bounded(&"雪🙂".repeat(300));
        assert!(text.len() <= MAX_DIAGNOSTIC_BYTES);
        assert!(text.ends_with(TRUNCATED));
        assert_eq!(
            bounded(&"x".repeat(MAX_DIAGNOSTIC_BYTES)).len(),
            MAX_DIAGNOSTIC_BYTES
        );
    }
}
