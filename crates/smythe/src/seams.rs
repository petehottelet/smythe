//! Injectable clocks and identities. Wall clocks never determine event order.
#![allow(clippy::disallowed_methods)]
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageBoundary {
    BlobWritten,
    BlobPublished,
    BeforeCommit { first: u64, last: u64 },
    AfterCommit { first: u64, last: u64 },
}
/// Optional fault/scheduling seam. Production observes without changing storage.
pub trait StorageObserver: Send + Sync {
    fn observe(&self, boundary: StorageBoundary) -> anyhow::Result<()>;
}
struct NoopObserver;
impl StorageObserver for NoopObserver {
    fn observe(&self, _: StorageBoundary) -> anyhow::Result<()> {
        Ok(())
    }
}

pub trait Clock: Send + Sync {
    fn unix_ms(&self) -> u64;
    fn monotonic(&self) -> Duration;
}
pub trait IdSource: Send + Sync {
    /// Return a fresh, filesystem-safe identity, including across restarts.
    fn next_id(&self) -> String;
}
struct SystemClock {
    origin: Instant,
}
impl Clock for SystemClock {
    fn unix_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
    fn monotonic(&self) -> Duration {
        self.origin.elapsed()
    }
}
struct UniqueIds;
impl IdSource for UniqueIds {
    fn next_id(&self) -> String {
        uuid::Uuid::now_v7().to_string()
    }
}
#[derive(Clone)]
pub struct Sources {
    pub clock: Arc<dyn Clock>,
    pub ids: Arc<dyn IdSource>,
    pub storage: Arc<dyn StorageObserver>,
}
impl Default for Sources {
    fn default() -> Self {
        Self {
            clock: Arc::new(SystemClock {
                origin: Instant::now(),
            }),
            ids: Arc::new(UniqueIds),
            storage: Arc::new(NoopObserver),
        }
    }
}
