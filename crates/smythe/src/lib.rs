//! Smythe's shared native local execution engine.
//! Remote effects are not admitted until their uncertain-outcome contract is implemented.
pub use smythe_core::EngineError;
pub use smythe_core::journal::Control;
pub use smythe_core::journal::Effect;
pub use smythe_core::{ExecutionGraph, NanoUsd, Node, Sentinel, Task, Topology};
mod runtime;
pub mod seams;
pub mod store;
pub use runtime::{CompletionResult, OfflineProvider, Provider, Span, Swarm, SwarmResult, Tracer};
