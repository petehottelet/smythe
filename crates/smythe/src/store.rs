//! Single-worker SQLite journal and immutable, content-addressed output storage.
use crate::seams::{Sources, StorageBoundary};
use anyhow::{Context, Result, anyhow, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use sha2::{Digest, Sha256};
use smythe_core::{
    EngineError,
    journal::{Control, Event, JOURNAL_SCHEMA, Projection, Record},
};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
};

type Cache = BTreeMap<String, Projection>;
#[cfg(test)]
thread_local! {static HISTORY_READS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};}
type Operation = Box<dyn FnOnce(&mut Connection, &mut Cache) + Send>;
struct Worker {
    sender: Option<mpsc::SyncSender<Operation>>,
    join: Option<thread::JoinHandle<()>>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
#[derive(Clone)]
pub struct Store {
    worker: Arc<Worker>,
    root: Option<PathBuf>,
    owners: Arc<Mutex<HashSet<String>>>,
    sources: Sources,
}
pub struct Lease {
    _file: Option<File>,
    owners: Arc<Mutex<HashSet<String>>>,
    id: String,
}
impl Drop for Lease {
    fn drop(&mut self) {
        if let Ok(mut owners) = self.owners.lock() {
            owners.remove(&self.id);
        }
    }
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= 96
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "invalid run identifier"
    );
    Ok(())
}
fn records(conn: &Connection, id: &str) -> Result<Vec<Record>> {
    #[cfg(test)]
    HISTORY_READS.with(|count| count.set(count.get() + 1));
    let mut statement = conn
        .prepare("SELECT sequence,payload,digest FROM events WHERE run_id=? ORDER BY sequence")?;
    let rows = statement.query_map([id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Vec<u8>>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut result = vec![];
    for row in rows {
        let (sequence, payload, hash) = row?;
        ensure!(digest(&payload) == hash, "journal checksum differs");
        let record: Record = serde_json::from_slice(&payload)?;
        ensure!(
            i64::try_from(record.sequence)? == sequence && record.run_id == id,
            "journal sequence differs"
        );
        result.push(record);
    }
    Ok(result)
}

fn append_many(
    conn: &mut Connection,
    cache: &mut Cache,
    id: &str,
    expected: Option<u64>,
    generation: u64,
    events: Vec<(Event, u64)>,
    sources: &Sources,
) -> Result<Projection> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let latest: i64 = tx.query_row(
        "SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?",
        [id],
        |r| r.get(0),
    )?;
    let sequence = u64::try_from(latest)?;
    let mut state = match cache.get(id) {
        Some(state) if state.sequence == sequence => Some(state.clone()),
        _ if sequence > 0 => Some(Projection::replay(&records(&tx, id)?)?),
        _ => None,
    };
    ensure!(
        expected.is_none_or(|n| n == sequence),
        EngineError::JournalChanged
    );
    ensure!(!events.is_empty(), "empty journal transaction");
    let transaction_start = sequence.checked_add(1).context("sequence overflow")?;
    let recorded_at = sources.clock.unix_ms();
    let batch = events
        .into_iter()
        .enumerate()
        .map(|(offset, (event, occurred_at))| {
            Ok(Record::new(
                id,
                transaction_start
                    .checked_add(u64::try_from(offset)?)
                    .context("sequence overflow")?,
                generation,
                transaction_start,
                event,
                occurred_at,
                recorded_at,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(current) = state.as_mut() {
        current.apply_batch(&batch)?;
    } else {
        state = Some(Projection::replay(&batch)?);
    }
    for record in batch {
        let payload = serde_json::to_vec(&record)?;
        tx.execute(
            "INSERT INTO events(run_id,sequence,payload,digest) VALUES(?,?,?,?)",
            params![
                id,
                i64::try_from(record.sequence)?,
                payload,
                digest(&payload)
            ],
        )?;
    }
    let state = state.context("missing journal projection")?;
    sources.storage.observe(StorageBoundary::BeforeCommit {
        first: transaction_start,
        last: state.sequence,
    })?;
    tx.commit()?;
    sources.storage.observe(StorageBoundary::AfterCommit {
        first: transaction_start,
        last: state.sequence,
    })?;
    cache.insert(id.to_owned(), state.clone());
    Ok(state)
}

impl Store {
    pub fn memory() -> Result<Self> {
        Self::memory_with_sources(Sources::default())
    }
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_sources(root, Sources::default())
    }
    pub fn memory_with_sources(sources: Sources) -> Result<Self> {
        Self::start(None, sources)
    }
    pub fn open_with_sources(root: impl AsRef<Path>, sources: Sources) -> Result<Self> {
        Self::start(Some(root.as_ref().to_owned()), sources)
    }
    fn start(root: Option<PathBuf>, sources: Sources) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Operation>(32);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let directory = root.clone();
        let join=thread::Builder::new().name("smythe-store".into()).spawn(move|| {
            let connection=(||->Result<Connection> {
                ensure!(rusqlite::version_number()>=3_051_003,"SQLite WAL-reset fix is required");
                let c=if let Some(root)=directory {
                    fs::create_dir_all(root.join("blobs"))?; fs::create_dir_all(root.join("locks"))?;
                    Connection::open(root.join("journal.sqlite3"))?
                } else {Connection::open_in_memory()?};
                c.busy_timeout(std::time::Duration::from_secs(5))?;
                let version:i64=c.pragma_query_value(None,"user_version",|r|r.get(0))?;
                ensure!(version == 0 || version == i64::from(JOURNAL_SCHEMA),
                    "unsupported journal schema; alpha schema 1 requires a new run");
                c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA trusted_schema=OFF; CREATE TABLE IF NOT EXISTS events(run_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload BLOB NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(run_id,sequence)); CREATE TABLE IF NOT EXISTS blobs(digest TEXT PRIMARY KEY,bytes BLOB NOT NULL);")?;
                c.pragma_update(None, "user_version", JOURNAL_SCHEMA)?;
                Ok(c)
            })();
            match connection {
                Ok(mut conn)=> {let _=ready_tx.send(Ok(())); let mut cache=Cache::new(); for operation in receiver {operation(&mut conn,&mut cache);}},
                Err(e)=> {let _=ready_tx.send(Err(e));}
            }
        })?;
        ready_rx.recv().context("store worker failed to start")??;
        Ok(Self {
            worker: Arc::new(Worker {
                sender: Some(sender),
                join: Some(join),
            }),
            root,
            owners: Arc::new(Mutex::new(HashSet::new())),
            sources,
        })
    }
    fn call<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Connection, &mut Cache) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.worker
            .sender
            .as_ref()
            .expect("live store worker")
            .send(Box::new(move |c, cache| {
                let _ = tx.send(operation(c, cache));
            }))
            .map_err(|_| anyhow!("store worker stopped"))?;
        rx.recv()
            .context("store worker stopped before acknowledgement")?
    }
    pub fn history(&self, id: &str) -> Result<Vec<Record>> {
        validate_id(id)?;
        let id = id.to_owned();
        self.call(move |c, _| records(c, &id))
    }
    pub fn inspect(&self, id: &str) -> Result<Projection> {
        Projection::replay(&self.history(id)?)
    }
    /// Coordinator projection cache; explicit inspection still verifies every record.
    pub(crate) fn current(&self, id: &str) -> Result<Projection> {
        validate_id(id)?;
        let id = id.to_owned();
        self.call(move |conn, cache| {
            let latest: i64 = conn.query_row(
                "SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?",
                [&id],
                |r| r.get(0),
            )?;
            if let Some(state) = cache.get(&id)
                && i64::try_from(state.sequence)? == latest
            {
                return Ok(state.clone());
            }
            let state = Projection::replay(&records(conn, &id)?)?;
            cache.insert(id, state.clone());
            Ok(state)
        })
    }
    pub fn append(
        &self,
        id: &str,
        expected: u64,
        generation: u64,
        event: Event,
    ) -> Result<Projection> {
        self.append_batch(id, expected, generation, vec![event])
    }
    pub fn append_batch(
        &self,
        id: &str,
        expected: u64,
        generation: u64,
        events: Vec<Event>,
    ) -> Result<Projection> {
        let now = self.sources.clock.unix_ms();
        self.append_observed_batch(
            id,
            expected,
            generation,
            events.into_iter().map(|event| (event, now)).collect(),
        )
    }
    /// Occurrence times come from the producer; observation is stamped by the store worker.
    pub fn append_observed_batch(
        &self,
        id: &str,
        expected: u64,
        generation: u64,
        events: Vec<(Event, u64)>,
    ) -> Result<Projection> {
        validate_id(id)?;
        let id = id.to_owned();
        let sources = self.sources.clone();
        self.call(move |conn, cache| {
            append_many(
                conn,
                cache,
                &id,
                Some(expected),
                generation,
                events,
                &sources,
            )
        })
    }
    pub fn control(&self, id: &str, control: Control) -> Result<Projection> {
        validate_id(id)?;
        let id = id.to_owned();
        let sources = self.sources.clone();
        let occurred_at = sources.clock.unix_ms();
        self.call(move |c, cache| {
            // Retry a racing coordinator append, never overwrite its revision.
            for _ in 0..16 {
                let p = Projection::replay(&records(c, &id)?)?;
                match append_many(
                    c,
                    cache,
                    &id,
                    Some(p.sequence),
                    p.generation,
                    vec![(Event::Control { control }, occurred_at)],
                    &sources,
                ) {
                    Ok(next) => return Ok(next),
                    Err(e) if EngineError::JournalChanged.matches(&e) => continue,
                    Err(e) => return Err(e),
                }
            }
            anyhow::bail!("run is busy; retry control command")
        })
    }
    pub fn lease(&self, id: &str) -> Result<Lease> {
        validate_id(id)?;
        let mut owners = self
            .owners
            .lock()
            .map_err(|_| anyhow!("ownership state unavailable"))?;
        ensure!(!owners.contains(id), "another coordinator owns this run");
        let file = if let Some(root) = &self.root {
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(root.join("locks").join(format!("{id}.lock")))?;
            file.try_lock()
                .context("another coordinator owns this run")?;
            Some(file)
        } else {
            None
        };
        owners.insert(id.to_owned());
        Ok(Lease {
            _file: file,
            owners: Arc::clone(&self.owners),
            id: id.to_owned(),
        })
    }
    pub fn put(&self, bytes: &[u8]) -> Result<String> {
        let hash = digest(bytes);
        if let Some(root) = &self.root {
            let path = root.join("blobs").join(&hash);
            if path.exists() {
                ensure!(fs::read(&path)? == bytes, "existing blob differs");
                return Ok(hash);
            }
            let temporary_id = self.sources.ids.next_id();
            validate_id(&temporary_id)?;
            let temporary = root.join("blobs").join(format!("{temporary_id}.pending"));
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            // write_all and immutable publication precede the SQL reference.
            // The process-crash contract does not promise survival of OS/power loss.
            drop(file);
            self.sources.storage.observe(StorageBoundary::BlobWritten)?;
            let result = fs::hard_link(&temporary, &path);
            fs::remove_file(&temporary)?;
            if let Err(error) = result {
                if !path.exists() {
                    return Err(error.into());
                }
                ensure!(fs::read(&path)? == bytes, "existing blob differs");
            }
            self.sources
                .storage
                .observe(StorageBoundary::BlobPublished)?;
        } else {
            let key = hash.clone();
            let bytes = bytes.to_vec();
            self.call(move |c, _| {
                c.execute(
                    "INSERT OR IGNORE INTO blobs(digest,bytes) VALUES(?,?)",
                    params![key, bytes],
                )?;
                Ok(())
            })?;
        }
        Ok(hash)
    }
    pub fn get(&self, hash: &str) -> Result<Vec<u8>> {
        ensure!(
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid blob reference"
        );
        let bytes = if let Some(root) = &self.root {
            fs::read(root.join("blobs").join(hash))?
        } else {
            let key = hash.to_owned();
            self.call(move |c, _| {
                c.query_row("SELECT bytes FROM blobs WHERE digest=?", [key], |r| {
                    r.get(0)
                })
                .optional()?
                .ok_or_else(|| anyhow!("missing blob"))
            })?
        };
        ensure!(digest(&bytes) == hash, "blob checksum differs");
        Ok(bytes)
    }
    pub fn output(&self, value: &Value) -> Result<Vec<u8>> {
        self.get(
            value
                .get("sha256")
                .and_then(Value::as_str)
                .context("missing output reference")?,
        )
    }
    pub fn list(&self) -> Result<BTreeMap<String, bool>> {
        self.call(|c, _| {
            let ids = c
                .prepare("SELECT DISTINCT run_id FROM events ORDER BY run_id")?
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            ids.into_iter()
                .map(|id| {
                    let p = Projection::replay(&records(c, &id)?)?;
                    Ok((id, p.completed))
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smythe_core::{ExecutionGraph, NanoUsd};
    #[test]
    fn appending_does_not_reaudit_the_growing_journal() {
        let store = Store::memory().unwrap();
        store
            .append(
                "growth",
                0,
                0,
                Event::Started {
                    binding: "fixture".into(),
                    graph: ExecutionGraph {
                        nodes: vec![],
                        topology: vec![],
                    },
                    ceiling: NanoUsd(0),
                },
            )
            .unwrap();
        let mut state = store
            .append("growth", 1, 1, Event::Claimed { generation: 1 })
            .unwrap();
        for i in 0..200 {
            state = store
                .append(
                    "growth",
                    state.sequence,
                    1,
                    Event::Control {
                        control: if i % 2 == 0 {
                            Control::Pause
                        } else {
                            Control::Run
                        },
                    },
                )
                .unwrap();
        }
        let count = store
            .call(|_, _| Ok(HISTORY_READS.with(|n| n.get())))
            .unwrap();
        assert_eq!(
            count, 0,
            "each append must use the validated projection, not rescan history"
        );
        assert_eq!(store.inspect("growth").unwrap().sequence, 202);
        assert_eq!(
            store
                .call(|_, _| Ok(HISTORY_READS.with(|n| n.get())))
                .unwrap(),
            1,
            "explicit inspection must still verify the full journal"
        );
    }
}
