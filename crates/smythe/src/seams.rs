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
/// Owns the production wait and external-control polling; deterministic tests can
/// deliver a control at a chosen poll without sleeping or consulting a wall clock.
pub trait ControlWatch: Send + Sync {
    fn poll(
        &self,
        store: &crate::store::Store,
        run: &str,
        after: u64,
        wake: &std::sync::mpsc::Receiver<()>,
    ) -> anyhow::Result<Vec<smythe_core::journal::Record>>;
}
struct PollControls;
impl ControlWatch for PollControls {
    fn poll(
        &self,
        store: &crate::store::Store,
        run: &str,
        after: u64,
        wake: &std::sync::mpsc::Receiver<()>,
    ) -> anyhow::Result<Vec<smythe_core::journal::Record>> {
        let _ = wake.recv_timeout(Duration::from_millis(50));
        store.changes_since(run, after)
    }
}
pub(crate) fn lock_run(file: &std::fs::File) -> anyhow::Result<()> {
    for _ in 0..20 {
        if file.try_lock().is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    anyhow::bail!("another coordinator owns this run")
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
    pub controls: Arc<dyn ControlWatch>,
}
impl Default for Sources {
    fn default() -> Self {
        Self {
            clock: Arc::new(SystemClock {
                origin: Instant::now(),
            }),
            ids: Arc::new(UniqueIds),
            storage: Arc::new(NoopObserver),
            controls: Arc::new(PollControls),
        }
    }
}
