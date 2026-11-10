//! Single-owner SQLite writer, atomic inline receipts, and immutable large blobs.

#[cfg(test)]
mod tests {
    use super::*;
    use smythe_core::{ExecutionGraph, Limits};
    #[test]
    fn appending_and_listing_do_not_reaudit_the_growing_journal() {
        let store = Store::memory().unwrap();
        let ack = store
            .append(
                "cache",
                0,
                0,
                Event::Started {
                    binding: "fixture".into(),
                    graph: ExecutionGraph {
                        topology: vec![],
                        nodes: vec![],
                    },
                    ceiling: NanoUsd(0),
                    limits: Limits::default(),
                },
            )
            .unwrap();
        let mut sequence = ack.last;
        sequence = store
            .append("cache", sequence, 1, Event::Claimed { generation: 1 })
            .unwrap()
            .last;
        let before = store
            .call(|_, _| Ok(HISTORY_READS.with(|n| n.get())))
            .unwrap();
        for _ in 0..200 {
            sequence = store
                .append(
                    "cache",
                    sequence,
                    1,
                    Event::Control {
                        control: Control::Pause,
                    },
                )
                .unwrap()
                .last;
            assert_eq!(store.list().unwrap()[0].sequence, sequence);
            assert!(store.changes_since("cache", sequence).unwrap().is_empty());
        }
        assert_eq!(
            store
                .call(|_, _| Ok(HISTORY_READS.with(|n| n.get())))
                .unwrap(),
            before
        );
    }
}
use crate::seams::{Sources, StorageBoundary};
use anyhow::{Context, Result, anyhow, ensure};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use smythe_core::{
    EngineError, NanoUsd, OutputRef,
    coordinator::Blob,
    journal::{Control, Disposition, Event, JOURNAL_SCHEMA, Projection, Record},
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions as FileOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
};

pub const INLINE_BYTES: usize = 64 * 1024;
type Cache = BTreeMap<String, Projection>;
type Operation = Box<dyn FnOnce(&mut Connection, &mut Cache) + Send>;
#[cfg(test)]
thread_local! {static HISTORY_READS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};}
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
struct Staging {
    path: PathBuf,
    marker: Option<File>,
}
impl Drop for Staging {
    fn drop(&mut self) {
        self.marker.take();
        let _ = fs::remove_dir_all(&self.path);
    }
}
#[derive(Clone)]
pub struct Store {
    worker: Arc<Worker>,
    root: Option<PathBuf>,
    owners: Arc<Mutex<HashSet<String>>>,
    sources: Sources,
    readonly: bool,
    staging: Option<Arc<Staging>>,
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
#[derive(Clone, Debug)]
pub struct Ack {
    pub first: u64,
    pub last: u64,
    pub records: Vec<Record>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub sequence: u64,
    pub status: String,
    pub disposition: Option<Disposition>,
}
#[derive(Debug, Serialize)]
pub struct Health {
    pub owner: &'static str,
    pub last_progress_unix_ms: Option<u64>,
    pub accepted_nodes: usize,
    pub total_nodes: usize,
    pub unresolved_reservations: usize,
    pub unresolved_nanousd: NanoUsd,
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
fn latest(c: &Connection, id: &str) -> Result<u64> {
    Ok(u64::try_from(c.query_row(
        "SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?",
        [id],
        |r| r.get::<_, i64>(0),
    )?)?)
}
fn records_after(c: &Connection, id: &str, after: u64, limit: usize) -> Result<Vec<Record>> {
    let mut statement=c.prepare("SELECT sequence,payload,digest FROM events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?")?;
    let rows = statement.query_map(
        params![
            id,
            i64::try_from(after)?,
            i64::try_from(limit).unwrap_or(i64::MAX)
        ],
        |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Vec<u8>>(1)?,
                r.get::<_, String>(2)?,
            ))
        },
    )?;
    let mut result = vec![];
    let mut expected = after;
    for row in rows {
        let (sequence, payload, hash) = row?;
        ensure!(digest(&payload) == hash, EngineError::Integrity);
        let record: Record = serde_json::from_slice(&payload).context(EngineError::Integrity)?;
        expected = expected.checked_add(1).context(EngineError::Integrity)?;
        ensure!(
            record.sequence == expected
                && i64::try_from(record.sequence)? == sequence
                && record.run_id == id
                && record.schema == JOURNAL_SCHEMA,
            EngineError::Integrity
        );
        result.push(record);
    }
    Ok(result)
}
fn records(c: &Connection, id: &str) -> Result<Vec<Record>> {
    #[cfg(test)]
    HISTORY_READS.with(|n| n.set(n.get() + 1));
    records_after(c, id, 0, usize::MAX)
}
fn blob_path(root: &Path, hash: &str) -> PathBuf {
    root.join("blobs").join(&hash[..2]).join(hash)
}
fn read_blob(c: &Connection, root: Option<&Path>, reference: &OutputRef) -> Result<Vec<u8>> {
    reference.validate()?;
    let bytes: Option<Vec<u8>> = c
        .query_row(
            "SELECT bytes FROM blobs WHERE digest=?",
            [&reference.sha256],
            |r| r.get(0),
        )
        .optional()?;
    let bytes = match bytes {
        Some(bytes) => bytes,
        None => fs::read(blob_path(
            root.context(EngineError::Integrity)?,
            &reference.sha256,
        ))
        .context(EngineError::Integrity)?,
    };
    ensure!(
        digest(&bytes) == reference.sha256 && bytes.len() as u64 == reference.bytes,
        EngineError::Integrity
    );
    Ok(bytes)
}
fn cached(
    c: &Connection,
    cache: &mut Cache,
    id: &str,
    sequence: u64,
) -> Result<Option<Projection>> {
    match cache.remove(id) {
        Some(mut state) => {
            ensure!(state.sequence <= sequence, EngineError::Integrity);
            if state.sequence < sequence {
                state
                    .apply_batch(&records_after(c, id, state.sequence, usize::MAX)?)
                    .context(EngineError::Integrity)?;
            }
            Ok(Some(state))
        }
        None if sequence > 0 => Ok(Some(
            Projection::replay(&records(c, id)?).context(EngineError::Integrity)?,
        )),
        None => Ok(None),
    }
}
/// Construct headers incrementally, including expansion and named phase identities.
pub fn make_records(
    id: &str,
    sequence: u64,
    generation: u64,
    state: Option<&Projection>,
    events: Vec<(Event, u64)>,
    recorded_at: u64,
) -> Result<Vec<Record>> {
    let first = sequence.checked_add(1).context("sequence overflow")?;
    let mut revision = state.map_or(0, |s| s.revision);
    let mut phases: HashMap<String, Option<String>> = HashMap::new();
    let mut batch = Vec::with_capacity(events.len());
    for (offset, (event, occurred_at)) in events.into_iter().enumerate() {
        match &event {
            Event::Started { graph, .. } => {
                phases.extend(graph.nodes.iter().map(|n| (n.id.clone(), n.phase.clone())))
            }
            Event::Expanded { nodes, .. } => {
                phases.extend(nodes.iter().map(|n| (n.id.clone(), n.phase.clone())))
            }
            _ => {}
        }
        revision = event.revision(revision);
        let phase = event
            .identity()
            .0
            .map(|node| {
                phases
                    .get(node)
                    .cloned()
                    .or_else(|| {
                        state
                            .and_then(|s| s.node(node).ok())
                            .map(|n| n.phase.clone())
                    })
                    .context("unknown node")
            })
            .transpose()?
            .flatten();
        batch.push(
            Record::new(
                id,
                first
                    .checked_add(offset as u64)
                    .context("sequence overflow")?,
                generation,
                first,
                event,
                occurred_at,
                recorded_at,
            )
            .in_revision(revision, phase),
        );
    }
    Ok(batch)
}
#[allow(clippy::too_many_arguments)]
fn append_many(
    c: &mut Connection,
    cache: &mut Cache,
    id: &str,
    expected: u64,
    generation: u64,
    events: Vec<(Event, u64)>,
    blobs: Vec<Blob>,
    sources: &Sources,
    root: Option<&Path>,
) -> Result<Ack> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let sequence = latest(&tx, id)?;
    ensure!(expected == sequence, EngineError::JournalChanged);
    ensure!(!events.is_empty(), "empty journal transaction");
    let mut state = cached(&tx, cache, id, sequence)?;
    let batch = make_records(
        id,
        sequence,
        generation,
        state.as_ref(),
        events,
        sources.clock.unix_ms(),
    )?;
    let first = batch.first().unwrap().sequence;
    let last = batch.last().unwrap().sequence;
    let save = |projection: &Projection| -> Result<()> {
        for blob in &blobs {
            ensure!(
                blob.reference.bytes == blob.bytes.len() as u64
                    && digest(&blob.bytes) == blob.reference.sha256,
                EngineError::Integrity
            );
            ensure!(batch.iter().any(|r| matches!(&r.event,Event::Received { output:Some(reference),.. } if reference==&blob.reference)),"unreferenced inline blob");
            tx.execute(
                "INSERT OR IGNORE INTO blobs(digest,bytes) VALUES(?,?)",
                params![blob.reference.sha256, blob.bytes.as_ref()],
            )?;
        }
        for record in &batch {
            if let Event::Received {
                output: Some(reference),
                ..
            } = &record.event
            {
                read_blob(&tx, root, reference)?;
            }
            let payload = serde_json::to_vec(record)?;
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
        let summary = serde_json::to_vec(&RunSummary {
            run_id: id.into(),
            sequence: last,
            status: projection.status().into(),
            disposition: projection.disposition,
        })?;
        tx.execute("INSERT INTO runs(run_id,summary,digest) VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary,digest=excluded.digest",params![id,summary,digest(&summary)])?;
        sources
            .storage
            .observe(StorageBoundary::BeforeCommit { first, last })?;
        Ok(())
    };
    // SQL errors discard the cache entry. Successful appends never clone a projection.
    if let Some(current) = state.as_mut() {
        current.apply_batch(&batch)?;
        save(current)?;
    } else {
        state = Some(Projection::replay(&batch)?);
        save(state.as_ref().unwrap())?;
    }
    tx.commit()?;
    cache.insert(id.into(), state.unwrap());
    sources
        .storage
        .observe(StorageBoundary::AfterCommit { first, last })?;
    Ok(Ack {
        first,
        last,
        records: batch,
    })
}

fn staging(root: &Path, sources: &Sources) -> Result<Arc<Staging>> {
    let directory = root.join("staging");
    fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    // Serialize marker creation with sweeping. A writer locks its marker before a
    // sweeper can see it; unmarked or linked entries are never deleted.
    let registry = FileOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join(".registry.lock"))?;
    registry.lock()?;
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if !kind.is_dir() || kind.is_symlink() {
            continue;
        }
        let path = entry.path().canonicalize()?;
        if path.parent() != Some(directory.as_path()) {
            continue;
        }
        let marker = path.join("owner.lock");
        if !fs::symlink_metadata(&marker)
            .is_ok_and(|m| m.file_type().is_file() && !m.file_type().is_symlink())
        {
            continue;
        }
        if let Ok(file) = FileOptions::new().read(true).write(true).open(&marker)
            && file.try_lock().is_ok()
        {
            // The registry excludes new writers; this identity has no live owner.
            drop(file);
            fs::remove_dir_all(&path)?;
        }
    }
    let id = sources.ids.next_id();
    validate_id(&id)?;
    let path = directory.join(format!("{}-{id}", std::process::id()));
    fs::create_dir(&path)?;
    let marker = FileOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path.join("owner.lock"))?;
    marker.try_lock()?;
    drop(registry);
    Ok(Arc::new(Staging {
        path,
        marker: Some(marker),
    }))
}
impl Store {
    pub fn memory() -> Result<Self> {
        Self::memory_with_sources(Sources::default())
    }
    pub fn memory_with_sources(sources: Sources) -> Result<Self> {
        Self::start(None, sources, false, true)
    }
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_sources(root, Sources::default())
    }
    pub fn open_with_sources(root: impl AsRef<Path>, sources: Sources) -> Result<Self> {
        Self::start(Some(root.as_ref().into()), sources, false, true)
    }
    pub fn open_read_only(root: impl AsRef<Path>) -> Result<Self> {
        Self::start(Some(root.as_ref().into()), Sources::default(), true, false)
    }
    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self> {
        Self::start(Some(root.as_ref().into()), Sources::default(), false, false)
    }
    pub fn is_memory(&self) -> bool {
        self.root.is_none()
    }
    fn start(
        root: Option<PathBuf>,
        sources: Sources,
        readonly: bool,
        create: bool,
    ) -> Result<Self> {
        if let Some(root) = &root {
            ensure!(
                create || root.join("journal.sqlite3").is_file(),
                EngineError::RunMissing
            );
            if create {
                fs::create_dir_all(root)?;
            }
        }
        let (sender, receiver) = mpsc::sync_channel::<Operation>(32);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let directory = root.clone();
        let join=thread::Builder::new().name("smythe-store".into()).spawn(move|| {
            let connection=(||->Result<Connection>{
                ensure!(rusqlite::version_number()>=3_051_003,"SQLite WAL-reset fix is required");
                let c=match &directory {
                    Some(root)=>Connection::open_with_flags(root.join("journal.sqlite3"),if readonly {OpenFlags::SQLITE_OPEN_READ_ONLY} else if create {OpenFlags::SQLITE_OPEN_READ_WRITE|OpenFlags::SQLITE_OPEN_CREATE} else {OpenFlags::SQLITE_OPEN_READ_WRITE})?,
                    None=>Connection::open_in_memory()?
                };
                c.busy_timeout(std::time::Duration::from_secs(5))?;
                let version:i64=c.pragma_query_value(None,"user_version",|r|r.get(0))?;
                ensure!(version==i64::from(JOURNAL_SCHEMA) || (version==0 && !readonly),EngineError::Integrity);
                c.execute_batch("PRAGMA trusted_schema=OFF;")?;
                if !readonly {
                    if let Some(root)=&directory { fs::create_dir_all(root.join("blobs"))?; fs::create_dir_all(root.join("locks"))?; }
                    c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA wal_autocheckpoint=16384; PRAGMA journal_size_limit=67108864; CREATE TABLE IF NOT EXISTS events(run_id TEXT NOT NULL,sequence INTEGER NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(run_id,sequence)); CREATE TABLE IF NOT EXISTS blobs(digest TEXT PRIMARY KEY,bytes BLOB NOT NULL); CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,summary BLOB NOT NULL,digest TEXT NOT NULL);")?;
                    if version==0 { c.pragma_update(None,"user_version",JOURNAL_SCHEMA)?; }
                }
                Ok(c)
            })();
            match connection {
                Ok(mut c)=> { let _=ready_tx.send(Ok(())); let mut cache=Cache::new(); for op in receiver { op(&mut c,&mut cache); } }
                Err(error)=> { let _=ready_tx.send(Err(error)); }
            }
        })?;
        let worker = Arc::new(Worker {
            sender: Some(sender),
            join: Some(join),
        });
        ready_rx.recv().context("store worker failed to start")??;
        let staging = if readonly {
            None
        } else {
            root.as_ref()
                .map(|root| staging(root, &sources))
                .transpose()?
        };
        Ok(Self {
            worker,
            root,
            owners: Arc::new(Mutex::new(HashSet::new())),
            sources,
            readonly,
            staging,
        })
    }
    fn call<T: Send + 'static>(
        &self,
        op: impl FnOnce(&mut Connection, &mut Cache) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.worker
            .sender
            .as_ref()
            .context("store worker unavailable")?
            .send(Box::new(move |c, cache| {
                let _ = tx.send(op(c, cache));
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
        let records = self.history(id)?;
        ensure!(!records.is_empty(), EngineError::RunMissing);
        Projection::replay(&records).context(EngineError::Integrity)
    }
    pub fn events_after(&self, id: &str, sequence: u64, limit: usize) -> Result<Vec<Record>> {
        validate_id(id)?;
        ensure!(limit > 0, "page limit must be positive");
        let id = id.to_owned();
        self.call(move |c, _| {
            let end = latest(c, &id)?;
            ensure!(end > 0, EngineError::RunMissing);
            ensure!(sequence <= end, EngineError::Integrity);
            records_after(c, &id, sequence, limit)
        })
    }
    pub fn changes_since(&self, id: &str, sequence: u64) -> Result<Vec<Record>> {
        validate_id(id)?;
        let id = id.to_owned();
        self.call(move |c, _| {
            let end = latest(c, &id)?;
            ensure!(sequence <= end, EngineError::Integrity);
            if end == sequence {
                Ok(vec![])
            } else {
                records_after(c, &id, sequence, usize::MAX)
            }
        })
    }
    pub fn append(&self, id: &str, expected: u64, generation: u64, event: Event) -> Result<Ack> {
        self.append_batch(id, expected, generation, vec![event])
    }
    pub fn append_batch(
        &self,
        id: &str,
        expected: u64,
        generation: u64,
        events: Vec<Event>,
    ) -> Result<Ack> {
        let now = self.sources.clock.unix_ms();
        self.append_observed_batch(
            id,
            expected,
            generation,
            events.into_iter().map(|e| (e, now)).collect(),
            vec![],
        )
    }
    pub fn append_observed_batch(
        &self,
        id: &str,
        expected: u64,
        generation: u64,
        events: Vec<(Event, u64)>,
        blobs: Vec<Blob>,
    ) -> Result<Ack> {
        ensure!(!self.readonly, "store is read-only");
        validate_id(id)?;
        let id = id.to_owned();
        let sources = self.sources.clone();
        let root = self.root.clone();
        self.call(move |c, cache| {
            append_many(
                c,
                cache,
                &id,
                expected,
                generation,
                events,
                blobs,
                &sources,
                root.as_deref(),
            )
        })
    }
    fn operator(&self, id: &str, event: impl Fn(&Projection) -> Result<Event>) -> Result<Ack> {
        let mut state = self.inspect(id)?;
        for _ in 0..16 {
            match self.append(id, state.sequence, state.generation, event(&state)?) {
                Ok(ack) => return Ok(ack),
                Err(error) if EngineError::JournalChanged.matches(&error) => {
                    state.apply_batch(&self.changes_since(id, state.sequence)?)?
                }
                Err(error) => return Err(error),
            }
        }
        anyhow::bail!("run is busy; retry control command")
    }
    pub fn control(&self, id: &str, control: Control) -> Result<Ack> {
        self.operator(id, |_| Ok(Event::Control { control }))
    }
    pub fn change_ceiling(&self, id: &str, ceiling: NanoUsd) -> Result<Ack> {
        self.operator(id, |_| Ok(Event::CeilingChanged { ceiling }))
    }
    pub fn retry(&self, id: &str, node: &str) -> Result<Ack> {
        self.operator(id, |s| {
            Ok(Event::Retried {
                node: node.into(),
                attempt: s.attempts.get(node).context("node has no attempt")?.number,
            })
        })
    }
    pub fn lease(&self, id: &str) -> Result<Lease> {
        ensure!(!self.readonly, "store is read-only");
        validate_id(id)?;
        let mut owners = self
            .owners
            .lock()
            .map_err(|_| anyhow!("ownership state unavailable"))?;
        ensure!(!owners.contains(id), "another coordinator owns this run");
        let file = if let Some(root) = &self.root {
            let file = FileOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(root.join("locks").join(format!("{id}.lock")))?;
            crate::seams::lock_run(&file)?;
            Some(file)
        } else {
            None
        };
        owners.insert(id.into());
        Ok(Lease {
            _file: file,
            owners: Arc::clone(&self.owners),
            id: id.into(),
        })
    }
    pub fn health(&self, state: &Projection) -> Result<Health> {
        validate_id(&state.run_id)?;
        let live = if let Some(root) = &self.root {
            match File::open(root.join("locks").join(format!("{}.lock", state.run_id))) {
                Ok(file) => file.try_lock_shared().is_err(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => return Err(error.into()),
            }
        } else {
            self.owners
                .lock()
                .map_err(|_| anyhow!("ownership state unavailable"))?
                .contains(&state.run_id)
        };
        Ok(Health {
            owner: if live { "live" } else { "absent" },
            last_progress_unix_ms: state.last_progress_unix_ms,
            accepted_nodes: state.accepted_nodes,
            total_nodes: state.graph.nodes.len(),
            unresolved_reservations: state.budget.pending().len(),
            unresolved_nanousd: state
                .budget
                .pending()
                .values()
                .try_fold(NanoUsd(0), |sum, &n| sum.checked_add(n))?,
        })
    }
    pub fn prepare(&self, bytes: Vec<u8>) -> Result<(OutputRef, Option<Blob>)> {
        ensure!(!self.readonly, "store is read-only");
        let reference = OutputRef {
            sha256: digest(&bytes),
            bytes: bytes.len() as u64,
        };
        if bytes.len() <= INLINE_BYTES || self.is_memory() {
            return Ok((
                reference.clone(),
                Some(Blob {
                    reference,
                    bytes: bytes.into(),
                }),
            ));
        }
        self.put_file(&reference, &bytes)?;
        Ok((reference, None))
    }
    fn put_file(&self, reference: &OutputRef, bytes: &[u8]) -> Result<()> {
        let root = self.root.as_ref().context("file store unavailable")?;
        let path = blob_path(root, &reference.sha256);
        if path.exists() {
            ensure!(fs::read(&path)? == bytes, EngineError::Integrity);
            return Ok(());
        }
        fs::create_dir_all(path.parent().unwrap())?;
        let id = self.sources.ids.next_id();
        validate_id(&id)?;
        let temporary = self
            .staging
            .as_ref()
            .context("staging unavailable")?
            .path
            .join(format!("{id}.pending"));
        let result = (|| {
            let mut file = FileOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            drop(file);
            self.sources.storage.observe(StorageBoundary::BlobWritten)?;
            if let Err(error) = fs::hard_link(&temporary, &path) {
                if !path.exists() {
                    return Err(error.into());
                }
                ensure!(fs::read(&path)? == bytes, EngineError::Integrity);
            }
            self.sources
                .storage
                .observe(StorageBoundary::BlobPublished)?;
            Ok(())
        })();
        if temporary.exists() {
            fs::remove_file(&temporary)?;
        }
        result
    }
    /// Explicit blob import. Runtime small outputs use prepare plus atomic receipt append.
    pub fn put(&self, bytes: &[u8]) -> Result<String> {
        ensure!(!self.readonly, "store is read-only");
        let reference = OutputRef {
            sha256: digest(bytes),
            bytes: bytes.len() as u64,
        };
        if self.is_memory() || bytes.len() <= INLINE_BYTES {
            let key = reference.sha256.clone();
            let bytes = bytes.to_vec();
            self.call(move |c, _| {
                c.execute(
                    "INSERT OR IGNORE INTO blobs(digest,bytes) VALUES(?,?)",
                    params![key, bytes],
                )?;
                let saved: Vec<u8> =
                    c.query_row("SELECT bytes FROM blobs WHERE digest=?", [key], |r| {
                        r.get(0)
                    })?;
                ensure!(saved == bytes, EngineError::Integrity);
                Ok(())
            })?;
        } else {
            self.put_file(&reference, bytes)?;
        }
        Ok(reference.sha256)
    }
    pub fn output(&self, reference: &OutputRef) -> Result<Vec<u8>> {
        let reference = reference.clone();
        let root = self.root.clone();
        self.call(move |c, _| read_blob(c, root.as_deref(), &reference))
    }
    pub fn get(&self, hash: &str) -> Result<Vec<u8>> {
        let check = OutputRef {
            sha256: hash.into(),
            bytes: 0,
        };
        check.validate()?;
        let hash = hash.to_owned();
        let root = self.root.clone();
        self.call(move |c, _| {
            let bytes: Option<Vec<u8>> = c
                .query_row("SELECT bytes FROM blobs WHERE digest=?", [&hash], |r| {
                    r.get(0)
                })
                .optional()?;
            let bytes = match bytes {
                Some(bytes) => bytes,
                None => fs::read(blob_path(
                    root.as_deref().context(EngineError::Integrity)?,
                    &hash,
                ))
                .context(EngineError::Integrity)?,
            };
            ensure!(digest(&bytes) == hash, EngineError::Integrity);
            Ok(bytes)
        })
    }
    pub fn list(&self) -> Result<Vec<RunSummary>> {
        self.call(|c, _| {
            let mut statement =
                c.prepare("SELECT run_id,summary,digest FROM runs ORDER BY run_id")?;
            let rows = statement.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            rows.map(|row| {
                let (id, bytes, hash) = row?;
                ensure!(digest(&bytes) == hash, EngineError::Integrity);
                let summary: RunSummary =
                    serde_json::from_slice(&bytes).context(EngineError::Integrity)?;
                ensure!(id == summary.run_id, EngineError::Integrity);
                Ok(summary)
            })
            .collect()
        })
    }
}
