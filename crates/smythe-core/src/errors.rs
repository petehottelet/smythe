//! Stable classifications for decisions; display text is only diagnostic.
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineError {
    JournalChanged,
    BudgetExhausted,
    NotAdmitting,
    StaleOwner,
    UnsupportedEffect,
    Paused,
    Stopped,
    RunExists,
    RunMissing,
    Integrity,
    ProviderFailed,
    InvalidCommand,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::JournalChanged => "journal revision differs; reload before admission",
            Self::BudgetExhausted => "insufficient remaining allowance before dispatch",
            Self::NotAdmitting => "run is not admitting work",
            Self::StaleOwner => "stale owner generation",
            Self::UnsupportedEffect => "external effects require an uncertain-outcome runtime",
            Self::Paused => "run paused with retained progress",
            Self::Stopped => "run stopped with retained progress",
            Self::RunExists => "run already exists; use --resume",
            Self::RunMissing => "run or store does not exist",
            Self::Integrity => "saved evidence failed integrity validation",
            Self::ProviderFailed => {
                "execution failed; inspect the journal and explicitly retry eligible pure attempts"
            }
            Self::InvalidCommand => "invalid command line or run configuration",
        })
    }
}
impl Error for EngineError {}

impl EngineError {
    pub fn class(self) -> &'static str {
        match self {
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::RunExists => "run_exists",
            Self::RunMissing => "run_missing",
            Self::Integrity => "integrity",
            Self::ProviderFailed => "provider_failed",
            Self::BudgetExhausted => "budget_blocked",
            Self::InvalidCommand => "invalid_command",
            Self::JournalChanged => "journal_changed",
            Self::NotAdmitting => "not_admitting",
            Self::StaleOwner => "stale_owner",
            Self::UnsupportedEffect => "unsupported_effect",
        }
    }
    pub fn exit_code(self) -> i32 {
        match self {
            Self::InvalidCommand => 2,
            Self::Paused | Self::Stopped => 3,
            Self::BudgetExhausted => 4,
            Self::Integrity => 5,
            Self::RunExists | Self::RunMissing => 6,
            _ => 1,
        }
    }
    pub fn matches(self, error: &anyhow::Error) -> bool {
        error.downcast_ref::<Self>() == Some(&self)
    }
}
