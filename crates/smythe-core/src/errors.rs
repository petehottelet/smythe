//! Stable classifications for decisions; display text is only diagnostic.
use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineError {
    JournalChanged,
    BudgetExhausted,
    NotAdmitting,
    StaleOwner,
    UnsupportedEffect,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::JournalChanged => "journal revision differs; reload before admission",
            Self::BudgetExhausted => "insufficient remaining allowance before dispatch",
            Self::NotAdmitting => "run is not admitting work",
            Self::StaleOwner => "stale owner generation",
            Self::UnsupportedEffect => "external effects require an uncertain-outcome runtime",
        })
    }
}
impl Error for EngineError {}

impl EngineError {
    pub fn matches(self, error: &anyhow::Error) -> bool {
        error.downcast_ref::<Self>() == Some(&self)
    }
}
