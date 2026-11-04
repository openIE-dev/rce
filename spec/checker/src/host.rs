//! Appendix E host runtime, version 0: the admitter, the durable log D,
//! verify on open, bindings and replay with every adapter disconnected.
//!
//! The rules this file implements, and only these:
//! - HC-1 one admitter: [`Host::admit`] is the only path from an input to δ,
//!   and an input reaches δ only after its entry is appended to D.
//! - HC-2 clocks: wall time enters only as `host.WallTime` (and, when the
//!   module declares `event WallTime(ns: u64)`, as that bound event).
//! - HC-8 / T37 replay: [`replay`] reads D, reads no clock, file, network or
//!   environment, and re-derives every `state_hash`.
//! - HS-2 durable append, group confirm (D-2 of the RCEdb design): one
//!   barrier per admitted batch; on Apple platforms `File::sync_data` is
//!   `fcntl(F_FULLFSYNC)` in Rust std, on Linux it is `fdatasync`.
//! - HS-3 / T39 fatal barrier: after a failed barrier the host appends nothing.
//! - HS-4 / T38 verify on open: D is cut after its longest valid prefix and
//!   `host.Truncated` records the number of entries dropped.
//! - HC-7 fuel: each admitted input (one application of δ) gets the full
//!   fuel of 6.7, so a long session never exhausts a budget shared across
//!   inputs; replay does the same, so it is a function of D.
//! - E.2a canonical admission: [`Host::close_window`] orders a window by
//!   (priority, source, seq), dedupes by (source, seq), acks after the barrier.
//! - E.2b admitter role: [`Host::take_role`] logs the term (fence token) and
//!   [`Host::resume`] opens a log handed over by the replicated log.
//! - HS-6 bindings: `host.Bind` at start, then `host.Synced` after each
//!   barrier and the bound program event `Synced(upto)` when declared.
//! - HC-10b typed inputs: an input from the input stream that names an
//!   event the host delivers is admitted after `host.Typed`.
//! - E.4c delivery by declaration (HD-1 to HD-4): every delivery is
//!   built into the module's declaration, matched by field name
//!   ([`Host::delivery_notes`] says how at start), and `host.Kept` records
//!   in D what was not delivered and why.
//!
//! The entry format is exactly the 6.1 chain of `form_json`: `state_hash` is
//! SHA-256 of the canonical state JSON, `hash` is SHA-256 of
//! `{"event":..,"prev_hash":..,"seq":..,"state_hash":..}`. Each line of the
//! file is `{"event":E,"hash":H,"prev_hash":P,"seq":N,"state_hash":S}`.
use crate::ast::{Arg, Expr, Item, Module, Ty};
use crate::interp::{Cause, D_GENESIS, Interp, Region, V, d_sha, event_json, event_pos_json, state_pos_json, why};
use crate::parse::parse;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// A snapshot (HS-5): a cache of the state after entry `seq`, never truth.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub seq: u64,
    pub state_hash: String,
    /// The hashed state form (`state_pos_json`, T7); its SHA-256 is `state_hash`.
    pub bytes: String,
}

/// One entry of D (6.1).
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub seq: u64,
    /// Canonical JSON of the event, `{"payload":..,"type":..}`, kept as bytes.
    pub event: String,
    pub prev_hash: String,
    pub state_hash: String,
    pub hash: String,
}

fn body(event: &str, prev: &str, seq: u64, sh: &str) -> String {
    format!("{{\"event\":{event},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}")
}

/// The hashed form of a stored event (6.1, T7): a program event is rebuilt
/// from its stored JSON and embedded positionally (`event_pos_json`); a
/// `host.*` entry, which names no declaration, is hashed as stored. `None`
/// when the stored event is not an event of the module.
pub fn hashed_event(m: &Module, event: &str) -> Option<String> {
    let v: Value = serde_json::from_str(event).ok()?;
    let name = v["type"].as_str()?;
    if name.starts_with("host.") || !declares_event(m, name) {
        return Some(event.to_string());
    }
    let (name, args) = decode_event(event, m).ok()?;
    let mut it = Interp::new(m);
    it.fuel = crate::interp::fuel_limit();
    match it.construct(&name, &args) {
        Ok(Some(ev)) => Some(event_pos_json(&name, &ev, m)),
        _ => None,
    }
}

impl Entry {
    /// An entry whose `hash` covers the hashed form of `event` (T7).
    pub fn new(m: &Module, seq: u64, event: String, prev_hash: &str, state_hash: &str) -> Entry {
        let hashed = hashed_event(m, &event).unwrap_or_else(|| event.clone());
        let hash = d_sha(&body(&hashed, prev_hash, seq, state_hash));
        Entry { seq, event, prev_hash: prev_hash.into(), state_hash: state_hash.into(), hash }
    }
    /// The stored line (no newline).
    pub fn line(&self) -> String {
        format!(
            "{{\"event\":{},\"hash\":\"{}\",\"prev_hash\":\"{}\",\"seq\":{},\"state_hash\":\"{}\"}}",
            self.event, self.hash, self.prev_hash, self.seq, self.state_hash
        )
    }
    /// Parse a stored line by its fixed layout, keeping the event bytes
    /// exactly (no JSON re-serialisation, so float text is never rewritten).
    pub fn parse_line(l: &str) -> Option<Entry> {
        let rest = l.strip_prefix("{\"event\":")?.strip_suffix("\"}")?;
        let (rest, sh) = rest.rsplit_once(",\"state_hash\":\"")?;
        let (rest, seq) = rest.rsplit_once(",\"seq\":")?;
        let (rest, prev) = rest.rsplit_once(",\"prev_hash\":\"")?;
        let prev = prev.strip_suffix('"')?;
        let (event, hash) = rest.rsplit_once(",\"hash\":\"")?;
        let hash = hash.strip_suffix('"')?;
        let seq: u64 = seq.parse().ok()?;
        let hex = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
        if !hex(sh) || !hex(prev) || !hex(hash) {
            return None;
        }
        serde_json::from_str::<Value>(event).ok()?;
        Some(Entry { seq, event: event.into(), prev_hash: prev.into(), state_hash: sh.into(), hash: hash.into() })
    }
    pub fn kind(&self) -> String {
        serde_json::from_str::<Value>(&self.event)
            .ok()
            .and_then(|v| v["type"].as_str().map(String::from))
            .unwrap_or_default()
    }
    pub fn is_host(&self) -> bool {
        self.kind().starts_with("host.")
    }
}

/// The event type of a checkpoint (HC-37).
pub const CHECKPOINT: &str = "host.Checkpoint";

/// HC-37: the `head` of a checkpoint entry, the hash of the last entry it
/// stands for in the log it was proven from; `None` for any other entry.
pub fn checkpoint_head(e: &Entry) -> Option<String> {
    let v: Value = serde_json::from_str(&e.event).ok()?;
    if v["type"].as_str()? != CHECKPOINT {
        return None;
    }
    let h = v["payload"]["head"].as_str()?;
    (h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())).then(|| h.to_string())
}

/// HS-4: the number of lines in the longest prefix that links from the
/// genesis hash, each entry's `hash` recomputed over the hashed form of its
/// event (T7). A line that does not parse (a torn write) ends the prefix.
pub fn verify_prefix(lines: &[&str], m: &Module) -> usize {
    let mut prev = D_GENESIS.to_string();
    let mut base = 0u64;
    for (i, l) in lines.iter().enumerate() {
        let Some(e) = Entry::parse_line(l) else { return i };
        // HC-37: a checkpoint begins D; it is hashed over its stored event
        // from the genesis hash, and the entry after it links to its head
        if i == 0
            && let Some(head) = checkpoint_head(&e)
        {
            if e.prev_hash != D_GENESIS || e.hash != d_sha(&body(&e.event, &e.prev_hash, e.seq, &e.state_hash)) {
                return 0;
            }
            base = e.seq;
            prev = head;
            continue;
        }
        if e.seq != base + i as u64 || e.prev_hash != prev {
            return i;
        }
        let Some(hashed) = hashed_event(m, &e.event) else { return i };
        if e.hash != d_sha(&body(&hashed, &e.prev_hash, e.seq, &e.state_hash)) {
            return i;
        }
        prev = e.hash;
    }
    lines.len()
}

/// The number of lines in the longest prefix whose entries parse and link
/// (`seq` and `prev_hash`) from the genesis hash, without the module: the
/// chain's shape, checked by a store that holds no program. The hashes
/// themselves are checked by `verify_prefix` when the log is opened.
pub fn verify_links(lines: &[&str]) -> usize {
    let mut prev = D_GENESIS.to_string();
    let mut base = 0u64;
    for (i, l) in lines.iter().enumerate() {
        let Some(e) = Entry::parse_line(l) else { return i };
        if i == 0
            && let Some(head) = checkpoint_head(&e)
        {
            if e.prev_hash != D_GENESIS {
                return 0;
            }
            base = e.seq;
            prev = head;
            continue;
        }
        if e.seq != base + i as u64 || e.prev_hash != prev {
            return i;
        }
        prev = e.hash;
    }
    lines.len()
}

/// The durable append primitive the host needs from storage (E.3).
pub trait Store {
    fn append(&mut self, line: &str) -> io::Result<()>;
    /// The barrier of E.3; success means every appended line is durable.
    fn barrier(&mut self) -> io::Result<()>;
    /// The E.3 row name, or `volatile`.
    fn caps(&self) -> &'static str;
}

impl<T: Store + ?Sized> Store for &mut T {
    fn append(&mut self, line: &str) -> io::Result<()> {
        (**self).append(line)
    }
    fn barrier(&mut self) -> io::Result<()> {
        (**self).barrier()
    }
    fn caps(&self) -> &'static str {
        (**self).caps()
    }
}

/// E.3: a new file needs the directory barrier too. POSIX makes the new
/// directory entry durable by fsyncing the directory. Windows cannot open a
/// directory as a file (`Access is denied`) and NTFS journals the entry with
/// the file's own flush; WASI preview 1 has no directory sync (`Bad file
/// descriptor`). On those targets the file's own `sync_all` is the barrier.
#[cfg(unix)]
fn sync_dir(path: &Path) -> io::Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    File::open(dir)?.sync_all()
}
#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// D in a file: one entry per line, append only.
pub struct FileStore {
    f: File,
    path: std::path::PathBuf,
}

impl FileStore {
    /// Open (creating if needed) and return the store and the existing bytes.
    pub fn open(path: &Path) -> io::Result<(FileStore, String)> {
        let created = !path.exists();
        let mut f = OpenOptions::new().read(true).append(true).create(true).open(path)?;
        if created {
            sync_dir(path)?;
        }
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok((FileStore { f, path: path.to_path_buf() }, s))
    }
    /// Cut the file to `len` bytes (HS-4) and make the cut durable.
    /// The cut goes through its own write handle: on Windows an append-mode
    /// handle has FILE_APPEND_DATA but not FILE_WRITE_DATA, and SetEndOfFile
    /// on it is "Access is denied" (found by rce-conformance t38 on
    /// windows-x86_64). Appends continue on `f` at the new end.
    pub fn cut(&mut self, len: u64) -> io::Result<()> {
        let w = OpenOptions::new().write(true).open(&self.path)?;
        w.set_len(len)?;
        w.sync_all()
    }
}

impl Store for FileStore {
    fn append(&mut self, line: &str) -> io::Result<()> {
        let mut b = Vec::with_capacity(line.len() + 1);
        b.extend_from_slice(line.as_bytes());
        b.push(b'\n');
        self.f.write_all(&b)
    }
    fn barrier(&mut self) -> io::Result<()> {
        // Rust std: F_FULLFSYNC on Apple platforms, fdatasync on Linux.
        self.f.sync_data()
    }
    fn caps(&self) -> &'static str {
        if cfg!(target_vendor = "apple") {
            "macOS, iOS"
        } else if cfg!(target_os = "linux") {
            "Linux"
        } else if cfg!(windows) {
            "Windows"
        } else {
            "FreeBSD, other POSIX"
        }
    }
}

/// A volatile in-memory store (WASI or browser without a file system, and
/// tests). `fail_barrier_at` makes barrier number k fail (T39 tests).
#[derive(Default)]
pub struct MemStore {
    pub lines: Vec<String>,
    pub barriers: usize,
    pub fail_barrier_at: Option<usize>,
}

impl Store for MemStore {
    fn append(&mut self, line: &str) -> io::Result<()> {
        self.lines.push(line.into());
        Ok(())
    }
    fn barrier(&mut self) -> io::Result<()> {
        self.barriers += 1;
        if self.fail_barrier_at == Some(self.barriers) {
            return Err(io::Error::other("injected barrier failure"));
        }
        Ok(())
    }
    fn caps(&self) -> &'static str {
        "volatile"
    }
}

/// What one admitted input did.
#[derive(Debug, PartialEq)]
pub struct Admitted {
    pub seq: u64,
    pub event: String,
    /// `Ok(outputs)` (each output's display text) or `Err(reason)` (T8: the step keeps the state).
    pub outcome: Result<Vec<String>, String>,
    /// Display sink (λ output only): the bytes of the `data` of each
    /// emitted `Display(data)`, text or a byte list `List[i64]` (HC-30).
    pub display: Vec<Vec<u8>>,
    /// The step emitted `Exit`: the host loop ends after this input.
    pub exit: bool,
    /// HC-29: the frames the step emitted (`Frame(view)`), in emit order,
    /// or why one is not a frame; a function of D, as replay re-derives it.
    pub frames: Vec<Result<crate::frame::Frame, String>>,
    /// HC-21: why the step kept the state (`fuel`, `size`, `value`, `depth`,
    /// `fail` or `trap`), `None` when the input was applied.
    pub cause: Option<&'static str>,
}

/// How an input reached δ. It decides the descent of HC-21 and which
/// request a result answers. The live host knows it from the path that
/// appended the entry; replay derives it from the entries around it, so
/// both agree (HC-10b, HC-21).
#[derive(Clone, Debug, PartialEq)]
enum How {
    /// an input with no mark: a result event `KDone` with a `req` is
    /// matched to the request (K, req) it answers
    Input,
    /// an input read from the input stream (HC-10b); when its name is one
    /// the host delivers, `host.Typed` precedes it. It answers no request
    /// and starts no descent.
    Typed,
    /// an input of a window, announced by `host.Input` (E.2a); as `Typed`
    Sourced,
    /// the `StepKept` the host delivers after `host.StepKept` (HC-21)
    Root,
    /// the result the host delivers for the request (K, req) (HC-3)
    Answers(String, u64),
    /// another delivery of the host (`Restarted`, `Synced`, `WallTime`,
    /// an HTTP input)
    Delivery,
    /// replay only: a host entry D owed (`host.Kept` after `host.StepKept`
    /// or `host.Restarted`), already checked
    Owed,
}

/// An entry D owes after the one just folded (HC-10b, HC-21, HC-22):
/// replay requires it next, and a D that ends without it is completed on
/// open, except a typed input, which the host had not appended.
#[derive(Clone, Debug)]
enum Next {
    /// `host.StepKept` {cause, seq} after the entry of the kept step at seq;
    /// `bind`: the module declares `StepKept` and the kept step does not descend
    /// from a delivered `StepKept`, so the delivery of `StepKept` follows
    HostKept { seq: u64, cause: Cause, bind: bool },
    /// `host.Kept` with exactly this event JSON (E.4c)
    Kept(String),
    /// the program event `name` the host delivers, with these arguments
    Deliver { name: String, args: Vec<Arg>, how: How },
    /// the typed input `host.Typed` announced at seq
    Typed { seq: u64 },
}

/// The `req` field of a request or of a result event.
fn req_of(fs: &[(String, V)]) -> Option<u64> {
    fs.iter().find_map(|(f, v)| match (f.as_str(), v) {
        ("req", V::Int(i, _)) => u64::try_from(*i).ok(),
        _ => None,
    })
}

/// The request (K, req) the input `name` (value `ev`) answers: the one the
/// host delivered it for, or, for an input with no mark, `KDone` by its
/// `req`. A typed input answers none (HC-10b).
fn answered(how: &How, name: &str, ev: &V) -> Option<(String, u64)> {
    match how {
        How::Answers(k, r) => Some((k.clone(), *r)),
        How::Input => {
            let k = name.strip_suffix("Done")?;
            let V::Rec(_, fs) = ev else { return None };
            Some((k.to_string(), req_of(fs)?))
        }
        _ => None,
    }
}

fn kept_payload(cause: Cause, seq: u64) -> String {
    format!("{{\"cause\":\"{}\",\"seq\":{seq}}}", cause.as_str())
}

fn kept_map(cause: Cause, seq: u64) -> serde_json::Map<String, Value> {
    let mut p = serde_json::Map::new();
    p.insert("cause".into(), Value::String(cause.as_str().into()));
    p.insert("seq".into(), Value::from(seq));
    p
}

fn restarted_map(below: u64) -> serde_json::Map<String, Value> {
    one_field("below", below)
}

fn one_field(k: &str, n: u64) -> serde_json::Map<String, Value> {
    let mut p = serde_json::Map::new();
    p.insert(k.into(), Value::from(n));
    p
}

/// The program events the host delivers, each with the host kind it is
/// bound to (E.5 `host.Bind`), in the order the bindings are appended at
/// start: `Synced`, `WallTime`, `StepKept`, `Restarted`, the HTTP inputs
/// under `http`, then the result `KDone` of each kind K performed. Only
/// events the module declares are bound.
fn binds(m: &Module, bound: &[&'static str], http: bool) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = vec![];
    for n in ["Synced", "WallTime", "StepKept", "Restarted", "Presented"] {
        v.push((n.to_string(), format!("host.{n}")));
    }
    if http {
        for n in ["HttpRequest", "HttpMalformed"] {
            v.push((n.to_string(), format!("host.{n}")));
        }
    }
    for k in bound {
        v.push((format!("{k}Done"), format!("host.{k}")));
    }
    v.retain(|(e, _)| declares_event(m, e));
    v
}

/// HC-10b: the program events a typed input is marked for: every event
/// some host delivers to this module, whatever adapters this run was
/// granted, so the mark is a function of the module and D alone. They are
/// the host inputs `Synced`, `WallTime`, `StepKept` and `Restarted`, the
/// HTTP inputs `HttpRequest` and `HttpMalformed`, and the result `KDone`
/// of every struct K the module declares, each when the module declares it.
fn host_named(m: &Module) -> Vec<String> {
    let mut v: Vec<String> = ["Synced", "WallTime", "StepKept", "Restarted", "Presented", "HttpRequest", "HttpMalformed"].iter().map(|n| n.to_string()).collect();
    for i in &m.items {
        if let Item::Struct { name, .. } = i {
            v.push(format!("{name}Done"));
        }
    }
    v.retain(|e| declares_event(m, e));
    v
}

/// What replay re-derived (HC-8, HC-21).
#[derive(Debug)]
pub struct Replayed {
    pub entries: u64,
    pub state_hash: String,
    pub state: Vec<Region>,
    /// Every kept step: (seq, cause, the text of why it was kept, re-derived by replay).
    pub kept_outcomes: Vec<(u64, &'static str, String)>,
}

/// The fabric record of E.7 (HC-11) as canonical JSON (keys sorted): the CPU
/// ISA features detected at runtime from a fixed list, in list order. GPUs,
/// NPUs, memory and counters are not queried by this host yet, so they are
/// `null` and placement treats those fabrics as absent.
pub fn fabric_caps() -> String {
    #[allow(unused_mut)]
    let mut cpu: Vec<&str> = vec![];
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as d;
        for (n, on) in [("neon", d!("neon")), ("aes", d!("aes")), ("sha2", d!("sha2")), ("sha3", d!("sha3")), ("crc", d!("crc")), ("lse", d!("lse")), ("dotprod", d!("dotprod")), ("fp16", d!("fp16")), ("sve", d!("sve")), ("sve2", d!("sve2"))] {
            if on {
                cpu.push(n);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as d;
        for (n, on) in [("sse4.2", d!("sse4.2")), ("avx", d!("avx")), ("avx2", d!("avx2")), ("fma", d!("fma")), ("avx512f", d!("avx512f")), ("aes", d!("aes")), ("sha", d!("sha"))] {
            if on {
                cpu.push(n);
            }
        }
    }
    let cpu: Vec<String> = cpu.iter().map(|c| format!("\"{c}\"")).collect();
    format!("{{\"counters\":null,\"cpu\":{{\"arch\":\"{}\",\"features\":[{}]}},\"gpus\":null,\"memory\":null,\"npus\":null}}", std::env::consts::ARCH, cpu.join(","))
}

/// HC-38: the members of a `host.Energy` payload, in byte order.
pub const ENERGY_FIELDS: [&str; 6] = ["amortization", "basis", "entries", "joules", "provenance", "source"];

/// HC-38: check a `host.Energy` payload at `seq` by D.11: only the six
/// members; a basis that says something was read (`reported` or
/// `measured`); joules finite and >= 0; a provenance valid by rule 1, with
/// no unknown field; an amortization of rule 3; and entries `[first, last]`
/// with first <= last < seq, the entries the reading covers.
pub fn energy_check(p: &Value, seq: u64) -> Result<(), String> {
    let bad = |why: String| format!("host.Energy at seq {seq} is not a reading (D.11): {why}");
    let o = p.as_object().ok_or_else(|| bad("the payload is not an object".into()))?;
    let mut keys: Vec<&str> = o.keys().map(|k| k.as_str()).collect();
    keys.sort();
    if keys != ENERGY_FIELDS {
        return Err(bad(format!("its members are {keys:?}, not {ENERGY_FIELDS:?}")));
    }
    match p["basis"].as_str() {
        Some("reported") | Some("measured") => {}
        b => return Err(bad(format!("basis {b:?}: a reading is `reported` or `measured`"))),
    }
    match p["joules"].as_f64() {
        Some(j) if j.is_finite() && j >= 0.0 => {}
        _ => return Err(bad(format!("joules {} is not a finite figure >= 0", p["joules"]))),
    }
    if p["source"].as_str().is_none_or(|s| s.is_empty()) {
        return Err(bad("source is empty".into()));
    }
    rce_energy::reading::Provenance::parse(&p["provenance"].to_string()).map_err(bad)?;
    let a = &p["amortization"];
    let ok = match a["kind"].as_str() {
        Some("per_call") | Some("per_solved_task") => a.as_object().is_some_and(|m| m.len() == 1),
        Some("batch_mean") => a.as_object().is_some_and(|m| m.len() == 2) && a["n"].as_u64().is_some_and(|n| n >= 1),
        _ => false,
    };
    if !ok {
        return Err(bad(format!("amortization {a} is not one of D.11 rule 3")));
    }
    match (p["entries"][0].as_u64(), p["entries"][1].as_u64(), p["entries"].as_array().map(|x| x.len())) {
        (Some(f), Some(l), Some(2)) if f <= l && l < seq => Ok(()),
        _ => Err(bad(format!("entries {} are not [first, last] with first <= last < {seq}", p["entries"]))),
    }
}

fn host_event(kind: &str, payload: &str) -> String {
    format!("{{\"payload\":{payload},\"type\":\"host.{kind}\"}}")
}

/// Rebuild a value from its stored Form JSON (6.1), guided by its declared
/// type, so replay uses no parser of program text. Records become
/// constructor calls of their struct (or of the kind variant named by `$v`),
/// maps come back from their canonical list of [key, value] pairs.
pub(crate) fn json_expr(v: &Value, ty: Option<&Ty>, m: &Module) -> Result<Expr, String> {
    let ty = match ty {
        Some(Ty::Opt(t)) => {
            if v.is_null() {
                return Ok(Expr::None(0));
            }
            let inner = json_expr(v, Some(&**t), m)?;
            return Ok(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e: inner }], 0, false));
        }
        Some(Ty::Region(t, _)) | Some(Ty::Ref(_, t)) => Some(&**t),
        t => t,
    };
    let (tname, targs): (Option<&str>, &[Ty]) = match ty {
        Some(Ty::Path(p, g)) => (p.last().map(String::as_str), g.as_slice()),
        _ => (None, &[]),
    };
    Ok(match v {
        Value::Null => Expr::None(0),
        Value::Bool(b) => Expr::Bool(*b, 0),
        Value::Number(n) => {
            if matches!(tname, Some("f64" | "f32")) || n.is_f64() {
                Expr::Float(n.as_f64().ok_or("number")?, 0)
            } else if let Some(i) = n.as_i64() {
                Expr::Int(i as i128, 0)
            } else {
                Expr::Int(n.as_u64().ok_or("number")? as i128, 0)
            }
        }
        Value::String(s) => Expr::Str(s.clone(), 0),
        Value::Array(xs) => match (tname, ty) {
            (Some("Map"), _) => {
                let mut ps = vec![];
                for x in xs {
                    let kv = x.as_array().filter(|a| a.len() == 2).ok_or("map entry is not a pair")?;
                    ps.push((json_expr(&kv[0], targs.first(), m)?, json_expr(&kv[1], targs.get(1), m)?));
                }
                Expr::Map(ps, 0)
            }
            (_, Some(Ty::Tuple(ts))) => {
                Expr::Tuple(xs.iter().zip(ts).map(|(x, t)| json_expr(x, Some(t), m)).collect::<Result<_, _>>()?, 0)
            }
            (_, Some(Ty::Slice(t))) | (_, Some(Ty::Array(t, _))) => {
                Expr::List(xs.iter().map(|x| json_expr(x, Some(t), m)).collect::<Result<_, _>>()?, 0)
            }
            _ => Expr::List(xs.iter().map(|x| json_expr(x, targs.first(), m)).collect::<Result<_, _>>()?, 0),
        },
        Value::Object(o) => {
            let (ctor, fields): (String, Vec<(String, Ty)>) = if let Some(vn) = o.get("$v").and_then(Value::as_str) {
                let fs = m
                    .items
                    .iter()
                    .find_map(|i| match i {
                        Item::Kind { variants, .. } => variants.iter().find(|(n, _)| n == vn).map(|(_, f)| f.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| format!("unknown variant `{vn}`"))?;
                (vn.to_string(), fs)
            } else {
                let n = tname.ok_or("record payload without a declared type")?;
                let fs = m
                    .items
                    .iter()
                    .find_map(|i| match i {
                        Item::Struct { name, fields, .. } if name == n => {
                            Some(fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| format!("`{n}` is not a struct"))?;
                (n.to_string(), fs)
            };
            let mut args = vec![];
            for (k, t) in &fields {
                if let Some(x) = o.get(k) {
                    args.push(Arg { label: Some(k.clone()), e: json_expr(x, Some(t), m)? });
                }
            }
            Expr::Call(Box::new(Expr::Name(ctor, 0)), args, 0, false)
        }
    })
}

fn decode_event(event: &str, m: &Module) -> Result<(String, Vec<Arg>), String> {
    let v: Value = serde_json::from_str(event).map_err(|e| e.to_string())?;
    let name = v["type"].as_str().ok_or("event without type")?.to_string();
    let decl: Vec<(String, Ty)> = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Event { name: n, payload, .. } if *n == name => Some(payload.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let mut args = vec![];
    if let Some(o) = v["payload"].as_object() {
        for (k, x) in o {
            if k.starts_with('$') {
                continue;
            }
            let t = decl.iter().find(|(n, _)| n == k).map(|(_, t)| t);
            args.push(Arg { label: Some(k.clone()), e: json_expr(x, t, m)? });
        }
    }
    Ok((name, args))
}

fn declares_event(m: &Module, name: &str) -> bool {
    m.items.iter().any(|i| matches!(i, Item::Event { name: n, .. } if n == name))
}

/// The outcome of one effect (HC-3): exactly one per request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Delivered,
    Failed,
    TimedOut,
    Unreachable,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Delivered => "delivered",
            Status::Failed => "failed",
            Status::TimedOut => "timed_out",
            Status::Unreachable => "unreachable",
        }
    }
}

/// What an adapter returns for one request: a status, the result value as
/// Form JSON (typed by the `data` field of the result event) and an error text.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectResult {
    pub status: Status,
    pub data: Option<Value>,
    pub error: String,
}

impl EffectResult {
    pub fn ok(data: Value) -> EffectResult {
        EffectResult { status: Status::Delivered, data: Some(data), error: String::new() }
    }
    pub fn failed(error: impl Into<String>) -> EffectResult {
        EffectResult { status: Status::Failed, data: None, error: error.into() }
    }
}

/// An adapter (E.1): turns one kind of output request into an effect and
/// one result. A program requests the effect by emitting a value of the
/// struct named `kind()` with a field `req: u64`; the admitter answers with
/// the program's event `<kind>Done(req: u64, status: text, data: T?,
/// err: text)`. Adapters are never called on replay (HC-8).
pub trait Adapter {
    fn kind(&self) -> &'static str;
    fn perform(&mut self, request: &Value) -> EffectResult;
    /// HC-22: the host restarted at entry `floor` (the seq of its
    /// `host.Restarted`). An adapter that issues resource ids issues them
    /// from `floor` upward, so no id issued before is reused within D.
    fn restart(&mut self, _floor: u64) {}
    /// HD-3: the module takes this kind's received bytes as `text`. An
    /// adapter that receives bytes then keeps a UTF-8 character that a read
    /// ends inside for the next read, so no character is split; the host
    /// delivers the rest as text when it is UTF-8.
    fn text(&mut self) {}
}

fn declares_struct(m: &Module, name: &str) -> bool {
    m.items.iter().any(|i| matches!(i, Item::Struct { name: n, .. } if n == name))
}

/// The declared payload of `event name`, if the module declares it.
fn event_payload<'m>(m: &'m Module, name: &str) -> Option<&'m [(String, Ty)]> {
    m.items.iter().find_map(|i| match i {
        Item::Event { name: n, payload, .. } if n == name => Some(payload.as_slice()),
        _ => None,
    })
}

/// The fields of host values that hold bytes received from outside
/// (HC-10a): delivered exactly as `List[i64]`, or as `text` only when they
/// are UTF-8 (HD-3). A line of the input stream held as `Input(data)`
/// (HC-24) is such bytes too, and so are a link's target and an account
/// name (HC-31, HC-33).
const BYTES_FIELDS: [(&str, &str); 6] = [("HttpRequest", "body"), ("NetRecvDone", "data"), ("Input", "data"), ("FileReadlinkDone", "data"), ("UserNameDone", "data"), ("GroupNameDone", "data")];

/// The fields that hold a list whose every element is such bytes: the
/// names of a directory (HC-30).
const BYTES_LIST_FIELDS: [(&str, &str); 1] = [("FileListDone", "data")];

/// Whether a host value holds received bytes (HC-10a): itself, or each
/// element of a list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bytes {
    No,
    Here,
    Each,
}

fn bytes_field(event: &str, field: &str) -> Bytes {
    if BYTES_FIELDS.contains(&(event, field)) {
        Bytes::Here
    } else if BYTES_LIST_FIELDS.contains(&(event, field)) {
        Bytes::Each
    } else {
        Bytes::No
    }
}

/// A declared type with every `type` alias it names followed.
fn unalias<'m>(mut t: &'m Ty, m: &'m Module) -> &'m Ty {
    for _ in 0..32 {
        let Ty::Path(p, g) = t else { return t };
        let next = (p.len() == 1 && g.is_empty())
            .then(|| m.items.iter().find_map(|i| match i {
                Item::TypeAlias { name, ty, generics, .. } if *name == p[0] && generics.is_empty() => Some(ty),
                _ => None,
            }))
            .flatten();
        match next {
            Some(n) => t = n,
            None => return t,
        }
    }
    t
}

/// A declared type with its aliases followed and one `?` removed: the type
/// of the value a field holds when it holds one.
fn held_type<'m>(t: &'m Ty, m: &'m Module) -> &'m Ty {
    match unalias(t, m) {
        Ty::Opt(x) => unalias(x, m),
        x => x,
    }
}

/// Whether a declared type holds text (HD-3), through aliases and `?`.
fn takes_text(t: &Ty, m: &Module) -> bool {
    matches!(held_type(t, m), Ty::Path(p, g) if p.len() == 1 && g.is_empty() && p[0] == "text")
}

/// Whether a declared type holds an integer, through aliases and `?`.
fn takes_int(t: &Ty, m: &Module) -> bool {
    matches!(held_type(t, m), Ty::Path(p, g) if p.len() == 1 && g.is_empty() && crate::types::INTS.contains(&p[0].as_str()))
}

/// What a JSON value is, for a reason text.
fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "no value",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "text",
        Value::Array(_) => "a list",
        Value::Object(_) => "a record",
    }
}

/// What a JSON value is, for a reason text (`rap.rs`).
pub(crate) fn json_kind_of(v: &Value) -> &'static str {
    json_kind(v)
}

/// A declared type with its aliases followed (`rap.rs`).
pub(crate) fn unalias_ty<'m>(t: &'m Ty, m: &'m Module) -> &'m Ty {
    unalias(t, m)
}

/// A declared type as written (`rap.rs`).
pub(crate) fn ty_name(t: &Ty) -> String {
    ty_text(t)
}

/// A JSON value as an argument of a declared type, when it converts
/// exactly (E.4c), for a value that is not received bytes (`rap.rs`).
pub(crate) fn convert_value(v: &Value, ty: &Ty, m: &Module) -> Result<Expr, String> {
    convert(v, ty, Bytes::No, m)
}

/// The program events the host delivers itself (HC-10b, `rap.rs`).
pub(crate) fn host_named_events(m: &Module) -> Vec<String> {
    host_named(m)
}

/// Received bytes as text, when they are UTF-8 (HD-3); otherwise why not,
/// with the offset of the first byte that is not.
pub(crate) fn utf8_text(b: &[u8]) -> Result<String, String> {
    match std::str::from_utf8(b) {
        Ok(s) => Ok(s.to_string()),
        Err(e) if e.error_len().is_none() => Err(format!("incomplete UTF-8 character at offset {}", e.valid_up_to())),
        Err(e) => Err(format!("not UTF-8 at offset {}", e.valid_up_to())),
    }
}

/// The host's value `v` of one field as an argument of the declared type
/// `ty`, when it converts exactly (E.4c); otherwise why not. `bytes`: the
/// value is bytes received from outside, which become `text` only when
/// they are UTF-8 (HD-3).
fn convert(v: &Value, ty: &Ty, bytes: Bytes, m: &Module) -> Result<Expr, String> {
    let ty = unalias(ty, m);
    let mismatch = || format!("{} where {} is declared", json_kind(v), ty_text(ty));
    match ty {
        Ty::Opt(t) => {
            if v.is_null() {
                return Ok(Expr::None(0));
            }
            let inner = convert(v, t, bytes, m)?;
            Ok(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e: inner }], 0, false))
        }
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() && crate::types::INTS.contains(&p[0].as_str()) => {
            let n = v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from)).ok_or_else(mismatch)?;
            let (lo, hi) = crate::types::int_range(&p[0]);
            if n < lo || n > hi {
                return Err(format!("{n} is outside {}", p[0]));
            }
            Ok(Expr::Int(n, 0))
        }
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() && (p[0] == "f64" || p[0] == "f32") => Ok(Expr::Float(v.as_f64().ok_or_else(mismatch)?, 0)),
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() && p[0] == "bool" => Ok(Expr::Bool(v.as_bool().ok_or_else(mismatch)?, 0)),
        Ty::Path(p, g) if p.len() == 1 && g.is_empty() && p[0] == "text" => match v {
            Value::String(s) => Ok(Expr::Str(s.clone(), 0)),
            Value::Array(xs) if bytes == Bytes::Here => {
                let b: Vec<u8> = xs.iter().map(|x| x.as_u64().filter(|b| *b <= 255).map(|b| b as u8)).collect::<Option<_>>().ok_or("a byte outside 0..255")?;
                utf8_text(&b).map(|s| Expr::Str(s, 0))
            }
            _ => Err(mismatch()),
        },
        Ty::Path(p, g) if p.len() == 1 && p[0] == "List" && g.len() == 1 => match v {
            Value::Array(xs) => {
                let each = if bytes == Bytes::Each { Bytes::Here } else { Bytes::No };
                let mut out = Vec::with_capacity(xs.len());
                for (i, x) in xs.iter().enumerate() {
                    out.push(convert(x, &g[0], each, m).map_err(|w| if each == Bytes::Here { format!("element {i}: {w}") } else { w })?);
                }
                Ok(Expr::List(out, 0))
            }
            // received bytes the host holds as text (a UTF-8 line, HC-24)
            // are exactly the bytes of that text
            Value::String(t) if bytes == Bytes::Here => Ok(Expr::List(t.bytes().map(|b| convert(&Value::from(b), &g[0], Bytes::No, m)).collect::<Result<_, _>>()?, 0)),
            _ => Err(mismatch()),
        },
        _ => json_expr(v, Some(ty), m),
    }
}

/// What the module's declaration of its event makes of a host value (E.4c).
enum Build {
    /// the arguments, and the host fields with a value the event does not declare
    Built(Vec<Arg>, Vec<String>),
    /// the event is declared and this value cannot be built: why, and the
    /// host field whose value has no exact conversion, if that is the reason
    Unbuildable(String, Option<String>),
    /// the module does not declare the event
    Undeclared,
}

/// What the admitter appends for one delivery (E.4c): `host.Kept` with
/// the reason, when the delivery is not the host's whole value and no
/// host entry before it holds that value, then the program event, when it
/// can be built.
#[derive(Clone, Debug, Default)]
struct Plan {
    kept: Option<String>,
    deliver: Option<(String, Vec<Arg>)>,
}

/// Whether a host value is the result of a request (HC-3): `KDone` with
/// the four fields every result has.
fn is_result(name: &str, p: &serde_json::Map<String, Value>) -> bool {
    name.ends_with("Done") && ["req", "status", "data", "err"].iter().all(|k| p.contains_key(*k))
}

/// The entry `host.Kept` {event, why}: the host's whole value of a
/// delivery, as an event JSON, and why the delivery is not exactly it.
fn kept_event(name: &str, p: &serde_json::Map<String, Value>, why: &str) -> String {
    let v = serde_json::json!({"event": {"payload": p, "type": name}, "why": why});
    host_event("Kept", &v.to_string())
}

/// The value of a result as the host makes it (HC-3).
fn result_payload(req: u64, r: &EffectResult) -> serde_json::Map<String, Value> {
    let mut p = serde_json::Map::new();
    p.insert("data".into(), r.data.clone().unwrap_or(Value::Null));
    p.insert("err".into(), Value::String(r.error.clone()));
    p.insert("req".into(), Value::from(req));
    p.insert("status".into(), Value::String(r.status.as_str().into()));
    p
}

/// The host's fields of each event it delivers, with their types, for the
/// lines printed at start (E.4c). The data of a result has the type of
/// its kind; only received bytes (HC-10a) are known here.
fn host_fields(event: &str) -> Vec<(&'static str, &'static str)> {
    match event {
        "StepKept" => vec![("seq", "u64"), ("cause", "text")],
        "Restarted" => vec![("below", "u64")],
        "Synced" => vec![("upto", "u64")],
        "WallTime" => vec![("ns", "u64")],
        "Presented" => vec![("seq", "u64"), ("digest", "text")],
        "HttpRequest" => vec![("id", "u64"), ("method", "text"), ("target", "text"), ("version", "text"), ("names", "List[text]"), ("values", "List[text]"), ("body", "List[i64]")],
        "HttpMalformed" => vec![("id", "u64"), ("status", "u64"), ("why", "text")],
        "NetRecvDone" => vec![("req", "u64"), ("status", "text"), ("data", "List[i64]?"), ("err", "text")],
        _ => vec![("req", "u64"), ("status", "text"), ("data", "?"), ("err", "text")],
    }
}

/// The line printed at start for one event the host delivers, when the
/// module's declaration is not exactly the host's value (E.4c): what the
/// host will deliver and record. `held`: a host entry before each delivery
/// holds the whole value. `None` when the declaration is exactly the host's.
fn delivery_note(m: &Module, event: &str, held: bool) -> Option<String> {
    let decl = event_payload(m, event)?;
    let host = host_fields(event);
    let mut says: Vec<String> = vec![];
    let mut never = false;
    for (i, (k, ty)) in decl.iter().enumerate() {
        let t = unalias(ty, m);
        let (opt, inner) = match t {
            Ty::Opt(x) => (true, ty_text(unalias(x, m))),
            x => (false, ty_text(x)),
        };
        match host.iter().find(|(h, _)| h == k) {
            None if opt => says.push(format!("{k} is always none (the host has no value for it)")),
            None => {
                says.push(format!("field {} ({k}) has no value from the host", i + 1));
                never = true;
            }
            Some((_, h)) => {
                // "?": the data of a result, whose type is its kind's
                let (hopt, hinner) = match h.strip_suffix('?') {
                    Some(x) if !x.is_empty() => (true, x),
                    _ => (false, *h),
                };
                if hinner == "?" {
                    if !opt {
                        says.push(format!("{k} is not optional, so a result without data (failed, timed out, unreachable) is not delivered"));
                    }
                } else if inner == "text" && hinner == "List[i64]" && bytes_field(event, k) == Bytes::Here {
                    let otherwise = if event == "HttpRequest" { "the request is delivered as HttpMalformed with status 415, with the offset" } else { "the result is delivered as failed, with the offset" };
                    says.push(format!("{k} is text: received bytes are delivered as text when they are UTF-8, and otherwise {otherwise}"));
                } else if inner != hinner {
                    if *k == "data" && is_result_name(event) {
                        says.push(format!("{k} is {inner}, the host's is {hinner}: each result is delivered as failed"));
                    } else {
                        says.push(format!("{k} is {}, the host's is {hinner}: no delivery can be built", ty_text(t)));
                        never = true;
                    }
                } else if hopt && !opt {
                    says.push(format!("{k} is not optional, so a value without it is not delivered"));
                }
            }
        }
    }
    let lost: Vec<&str> = host.iter().map(|(h, _)| *h).filter(|h| !decl.iter().any(|(k, _)| k == h)).collect();
    if !lost.is_empty() && !held && !never {
        says.push(format!("the host's {} {} not delivered, and host.Kept keeps {} in D before each delivery", lost.join(", "), if lost.len() == 1 { "is" } else { "are" }, if lost.len() == 1 { "it" } else { "them" }));
    }
    if says.is_empty() {
        return None;
    }
    let tail = match (never, event) {
        (false, _) => "",
        (true, "HttpRequest") => "; so no request is delivered as HttpRequest: each is recorded in D as host.Kept and delivered as HttpMalformed with status 501, or answered 501 by the host",
        (true, _) => "; so it is never delivered, and each value is recorded in D as host.Kept with the reason",
    };
    Some(format!("event {event}: {}{tail}", says.join("; ")))
}

fn is_result_name(event: &str) -> bool {
    !matches!(event, "StepKept" | "Restarted" | "Synced" | "WallTime" | "Presented" | "HttpRequest" | "HttpMalformed") && event.ends_with("Done")
}

/// The admitter and the program state it folds (one per program instance, HC-1).
/// HC-35, HC-36: one invocation of D, from its `host.RapInvoke` or
/// `host.RapDuplicate` to its `host.RapSent`. Every field is folded from D.
#[derive(Clone, Debug, Default)]
struct RapCall {
    /// the host's whole value of the invocation (for a duplicate: its
    /// call_id, callback_url, group_id and id)
    inv: serde_json::Map<String, Value>,
    /// the event and its arguments, or why the invocation does not conform;
    /// `None` for a duplicate
    op: Option<Result<(String, Vec<Arg>), String>>,
    /// a duplicate: the seq of the first invocation of its (group_id, id)
    of: Option<u64>,
    /// `host.RapStep` names it
    stepped: bool,
    /// how its step ended, once replay or the host applied it
    outcome: Option<crate::rap::Outcome>,
    /// the bytes of its result, once `host.RapResult` holds their digest
    body: Option<String>,
    /// the status that settled its delivery (`host.RapSent`)
    sent: Option<u16>,
    /// HC-37: for a kept step, the canonical named form of the state it was
    /// kept on, from which a checkpoint re-derives the kept text (HC-21: the text
    /// is never in D)
    kept_state: Option<String>,
}

/// HC-35, HC-36: the invocations of D by seq, the first seq of each
/// (group_id, id), and the invocation whose step the next program entry is.
#[derive(Debug, Default)]
struct Rap {
    calls: std::collections::BTreeMap<u64, RapCall>,
    keys: HashMap<(String, String), u64>,
    stepping: Option<u64>,
    /// the calls without a result, and those whose result has not settled,
    /// so a round's work is bounded by what is open, not by D
    open: std::collections::BTreeSet<u64>,
    unsent: std::collections::BTreeSet<u64>,
}

impl Rap {
    fn add(&mut self, seq: u64, c: RapCall) {
        self.calls.insert(seq, c);
        self.open.insert(seq);
    }
    fn answered(&mut self, c: u64, body: String) {
        if let Some(x) = self.calls.get_mut(&c) {
            x.body = Some(body);
        }
        self.open.remove(&c);
        self.unsent.insert(c);
    }
    fn settled(&mut self, c: u64, status: u16) {
        if let Some(x) = self.calls.get_mut(&c) {
            x.sent = Some(status);
        }
        self.unsent.remove(&c);
    }
}

/// How the host admitted one RAP invocation (HC-35).
#[derive(Clone, Debug, PartialEq)]
pub enum RapAdmit {
    /// a first invocation of its (group_id, id), at this seq
    Invoked(u64),
    /// a duplicate at seq of the invocation at `of`: not stepped again
    Duplicate { seq: u64, of: u64 },
}

/// One result to deliver (HC-36): the seq of the invocation (or duplicate)
/// it answers, where it goes, its bytes, and the status that settled it.
#[derive(Clone, Debug, PartialEq)]
pub struct RapOut {
    pub call: u64,
    pub callback_url: String,
    pub body: String,
    pub sent: Option<u16>,
}

fn rap_key(p: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    Some((p.get("group_id")?.as_str()?.to_string(), p.get("id")?.as_str()?.to_string()))
}

pub struct Host<'a> {
    m: &'a Module,
    it: Interp<'a>,
    state: Vec<Region>,
    state_hash: String,
    prev: String,
    seq: u64,
    store: Box<dyn Store + 'a>,
    stopped: bool,
    /// Entries appended since the last successful barrier.
    pending: usize,
    /// Of those, the ones that are not the confirmation of a barrier
    /// (`host.Synced` and its bound `Synced`): a barrier is taken for these.
    pending_real: usize,
    /// While the entries confirming a barrier are appended.
    confirming: bool,
    /// The last seq made durable.
    pub durable: Option<u64>,
    /// The seq of the snapshot this host was opened from, if one was used (HS-5).
    pub from_snapshot: Option<u64>,
    adapters: Vec<Box<dyn Adapter + 'a>>,
    /// Effect kinds bound at start (HS-6), in adapter order.
    bound: Vec<&'static str>,
    /// Requests emitted and not yet answered, in emit order (E-x).
    queue: std::collections::VecDeque<Queued>,
    /// HC-37: every request emitted and not yet answered, whatever adapters
    /// this host has: each output of a declared struct with an integer
    /// `req`. `queue` is the part of it this host's adapters perform, so a
    /// checkpoint, which no adapter set is known for, carries this.
    unanswered: std::collections::VecDeque<Queued>,
    /// HC-21: the requests (`K` with a `req`, `KDone` declared) emitted by
    /// steps that descend from a `StepKept` entry and not yet answered by a
    /// `KDone` with their req: (K, req, the `StepKept` entry's seq), in emit
    /// order. Derived from D alone, so live and replay agree.
    descends: std::collections::VecDeque<(String, u64, u64)>,
    /// The most queued requests one call of [`Host::drain`] performs
    /// (unbounded unless the host loop bounds it, HC-24).
    drain_max: usize,
    /// The last seq covered by a successful barrier (on every store; on a
    /// volatile one nothing is reported durable, E.3).
    synced: Option<u64>,
    /// E.2a: per source, the highest seq whose input is in D (dedupe key).
    admitted: std::collections::BTreeMap<String, u64>,
    /// E.2a: the id of the next admission window, and the tick the last one closed at.
    window: u64,
    closed_tick: u64,
    /// E.2a: offered inputs not yet in a closed window (volatile; sources retransmit until acked).
    offered: Vec<Sourced>,
    /// E.2b: the term of the admitter role this host holds (0: unreplicated).
    pub term: u64,
    /// HC-6: a `fn main` run being folded from D (start payload, step entries).
    /// HC-6: the live `fn main` task, part of the state D folds to.
    task: Option<MainTask>,
    /// HC-12: fabric executors registered by the platform (wgpu, NPU, ...).
    fabrics: Vec<Box<dyn FabricExec + 'a>>,
    /// HC-12: calibration (fabric, class) -> (overhead, throughput), and the
    /// (fabric, class) pairs admitted by a matching probe, folded from D.
    calib: std::collections::BTreeMap<(String, String), (f64, f64)>,
    exacts: std::collections::BTreeSet<(String, String)>,
    /// HC-10b, HC-21, HC-22: the entries the last folded entry requires next.
    owe: std::collections::VecDeque<Next>,
    /// HC-10b: the program events a typed input is marked for ([`host_named`]).
    delivered: Vec<String>,
    /// E.4c: one line for each declaration that is not exactly what the
    /// host delivers, saying what the host will deliver and record.
    notes: Vec<String>,
    /// HC-21: the kept steps replay re-derived: (seq, cause, text).
    kept_outcomes: Vec<(u64, &'static str, String)>,
    /// HC-22: the n of the last `Tick(n)` in D, so the host loop continues it.
    last_tick: u64,
    /// When set, replay records each program entry's outputs (Form JSON).
    outputs: Option<Vec<(u64, Vec<Value>)>>,
    /// HC-29: what frames derive from the entries folded so far.
    clock: crate::frame::Clock,
    /// HC-29: (seq, frame digest) of every frame derived from D, in order;
    /// a confirmation names one of them.
    shown: Vec<(u64, String)>,
    /// HC-29: the view of the last frame derived from D.
    last_view: Option<crate::frame::ViewTree>,
    /// When set, replay records every frame it re-derives (HC-29).
    frames: Option<Vec<Result<crate::frame::Frame, String>>>,
    /// HC-35, HC-36: the RAP invocations of D.
    rap: Rap,
}

/// A request emitted and not yet answered.
#[derive(Clone, Debug)]
struct Queued {
    kind: String,
    req: u64,
    /// the request as Form JSON
    json: Value,
    /// the seq of the entry that requested it
    at: u64,
}

/// A declared type as written (`u64`, `List[text]`, `u64?`).
fn ty_text(t: &Ty) -> String {
    match t {
        Ty::Path(p, g) if g.is_empty() => p.join("."),
        Ty::Path(p, g) => format!("{}[{}]", p.join("."), g.iter().map(ty_text).collect::<Vec<_>>().join(", ")),
        Ty::Opt(x) => format!("{}?", ty_text(x)),
        other => format!("{other:?}"),
    }
}

/// HD-2: the status the host gives a request it cannot deliver as
/// `HttpRequest`, from the reason: 415 Unsupported Media Type when the
/// body is not UTF-8 and the module takes it as text (RFC 9110 15.5.16, the
/// content inspected directly), 501 Not Implemented otherwise (RFC 9110
/// 15.6.2: the program does not support what the request needs).
fn http_fallback_status(why: &str) -> u64 {
    if why.starts_with("body: ") && why.contains("UTF-8") { 415 } else { 501 }
}

/// How the host admitted one HTTP input (HC-24, HD-2).
#[derive(Debug)]
pub struct HttpAdmit {
    /// the id: the seq of the entry that delivers the input, or, when
    /// `host.Kept` records it first, of the entry after that one
    pub id: u64,
    pub admitted: Vec<Admitted>,
    /// the response's framing (HC-26): a HEAD request, and whether the
    /// connection closes after the response
    pub head: bool,
    pub close: bool,
    /// the status the host answers with itself, after the batch barrier,
    /// when the program cannot answer: the input was not delivered, it
    /// was delivered without its id, or the module declares no
    /// `HttpRespond` with `req` and `status` (HD-2)
    pub answer: Option<u64>,
}

/// HD-2: whether the program can answer an HTTP input: the module
/// declares `struct HttpRespond` with `req` and `status` of integer types
/// (any other field is read by name, HC-26).
fn responds(m: &Module) -> bool {
    m.items.iter().any(|i| match i {
        Item::Struct { name, fields, .. } if name == "HttpRespond" => ["req", "status"].iter().all(|k| fields.iter().any(|f| f.name == *k && takes_int(&f.ty, m))),
        _ => false,
    })
}

/// The integer argument `k` of a delivery, also when it is held in a `?`.
fn int_arg(args: &[Arg], k: &str) -> Option<u64> {
    args.iter().find_map(|a| match (&a.label, &a.e) {
        (Some(l), Expr::Int(n, _)) if l == k => u64::try_from(*n).ok(),
        (Some(l), Expr::Call(f, xs, _, _)) if l == k && matches!(&**f, Expr::Name(s, _) if s == "some") => match xs.first().map(|x| &x.e) {
            Some(Expr::Int(n, _)) => u64::try_from(*n).ok(),
            _ => None,
        },
        _ => None,
    })
}

/// HC-26, HD-2: the response to one HTTP input, from the host's whole
/// value `p` of the input `name` (its id included), the plan of its
/// delivery and `at`, the seq of the entry that delivers it: whether the
/// request is HEAD, whether the connection closes after the response, and
/// the status the host answers with itself, `None` when the program
/// answers. The program answers only an input delivered with its id, the
/// seq of that entry, to a module that can answer (`responds`); every
/// other input is answered by the host with the host's status of it: 501
/// for a request delivered as `HttpRequest`, the status of
/// [`http_fallback_status`] for one that is not, and the status of a
/// malformed request.
fn http_terms(m: &Module, name: &str, p: &serde_json::Map<String, Value>, plan: &Plan, at: u64) -> (bool, bool, Option<u64>) {
    let answers = responds(m) && plan.deliver.as_ref().is_some_and(|(_, args)| int_arg(args, "id") == Some(at));
    let delivered = plan.deliver.as_ref().map(|(n, _)| n.as_str());
    if !answers {
        // the host's own response has no body and closes the connection
        let status = match (name, delivered) {
            ("HttpRequest", Some("HttpRequest")) => 501,
            ("HttpRequest", _) => http_fallback_status(plan.kept.as_deref().unwrap_or("")),
            _ => p.get("status").and_then(Value::as_u64).unwrap_or(400),
        };
        return (false, true, Some(status));
    }
    if name == "HttpRequest" && delivered == Some("HttpRequest") {
        let texts = |k: &str| -> Vec<String> { p.get(k).and_then(Value::as_array).into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect() };
        let close = crate::adapters::http::closes(p.get("version").and_then(Value::as_str).unwrap_or(""), &texts("names"), &texts("values"));
        return (p.get("method").and_then(Value::as_str) == Some("HEAD"), close, None);
    }
    // a malformed request, delivered as HttpMalformed: the connection closes
    (false, true, None)
}

/// One HTTP input of D, and how its response is made (HC-26).
#[derive(Clone, Debug, PartialEq)]
pub struct HttpOfD {
    /// the id: the seq of the entry that delivers the input (or would)
    pub id: u64,
    /// a HEAD request: the response carries no body
    pub head: bool,
    /// the connection closes after the response
    pub close: bool,
    /// the status of the response the host makes itself (HD-2), with no
    /// fields and no body; `None` when the program answers with the
    /// `HttpRespond` of the step at `id`
    pub answer: Option<u64>,
}

/// HC-26, HD-2: every HTTP input the host admitted into D, in order, with
/// the framing of its response and who makes it, read from the host's
/// whole value of the input (its entry, or the `host.Kept` that records it
/// first) and the module, the way the live host decided them
/// ([`Host::http_input`]). A typed input (HC-10b) is not an HTTP input. With
/// the `HttpRespond` requests replay re-derives ([`replay_outputs`]) and
/// `http::response_bytes`, every byte the host writes for a request, and
/// whether the connection closes after it, is a function of D.
pub fn http_inputs(m: &Module, text: &str) -> Result<Vec<HttpOfD>, String> {
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    let mut out = vec![];
    // the entry that is no HTTP input of its own: the typed input after
    // `host.Typed`, or the delivery after the `host.Kept` that holds it
    let mut skip: Option<u64> = None;
    for l in text.split_terminator('\n') {
        let e = Entry::parse_line(l).ok_or("D: a line that is not an entry")?;
        if skip == Some(e.seq) {
            continue;
        }
        // HC-37: the HTTP inputs still open where the checkpoint stands
        if e.kind() == CHECKPOINT {
            let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
            for x in v["payload"]["http"].as_array().into_iter().flatten() {
                let (Some(id), Some(head), Some(close)) = (x[0].as_u64(), x[1].as_bool(), x[2].as_bool()) else { return Err("D: an open HTTP input of the checkpoint is not [id, head, close]".into()) };
                out.push(HttpOfD { id, head, close, answer: None });
            }
            continue;
        }
        let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
        let (ev, at) = match e.kind().as_str() {
            "host.Typed" => {
                skip = Some(e.seq + 1);
                continue;
            }
            "host.Kept" => (v["payload"]["event"].clone(), e.seq + 1),
            "HttpRequest" | "HttpMalformed" => (v, e.seq),
            _ => continue,
        };
        let (Some(name), Some(p)) = (ev["type"].as_str(), ev["payload"].as_object()) else { continue };
        if name != "HttpRequest" && name != "HttpMalformed" {
            continue;
        }
        let plan = h.plan(name, p, false);
        if at > e.seq && plan.deliver.is_some() {
            skip = Some(at);
        }
        let (head, close, answer) = http_terms(m, name, p, &plan, at);
        out.push(HttpOfD { id: p.get("id").and_then(Value::as_u64).unwrap_or(at), head, close, answer });
    }
    Ok(out)
}

/// HC-37: the HTTP inputs of `text` still open at its end: inputs the
/// program answers ([`http_inputs`]) with no `HttpRespondDone` delivered for
/// their id, and admitted after the last `host.Restarted`, whose process
/// owned the connections of every earlier request (HC-22). A checkpoint
/// carries them, so the bytes of a response made after it are a function of
/// D' as they are of D (HC-26).
pub fn http_open(m: &Module, text: &str) -> Result<Vec<HttpOfD>, String> {
    let mut closed = std::collections::BTreeSet::new();
    let mut restart = 0u64;
    for l in text.split_terminator('\n') {
        let e = Entry::parse_line(l).ok_or("D: a line that is not an entry")?;
        let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
        let ev = match e.kind().as_str() {
            "host.Restarted" => {
                restart = e.seq;
                continue;
            }
            "host.Kept" => v["payload"]["event"].clone(),
            _ => v,
        };
        if ev["type"].as_str() == Some("HttpRespondDone")
            && ev["payload"]["status"].as_str() == Some("delivered")
            && let Some(r) = ev["payload"]["req"].as_u64()
        {
            closed.insert(r);
        }
    }
    Ok(http_inputs(m, text)?.into_iter().filter(|x| x.answer.is_none() && !closed.contains(&x.id) && x.id > restart).collect())
}

/// HC-6: a live `fn main` task. Its state is a function of D: the start
/// entry, the inputs logged so far and the steps logged so far. `run` is a
/// cache of the scheduler re-run on those inputs and is not part of it.
#[derive(Debug)]
struct MainTask {
    start: Value,
    inputs: Vec<(u64, bool, String, String)>,
    /// program events fed to the task (HC-6): (tick, canonical event JSON)
    events: Vec<(u64, String)>,
    steps: usize,
    digest: String,
    last_tick: u64,
    run: Option<crate::conc::RunResult>,
}

impl MainTask {
    fn json(&self) -> String {
        let ins: Vec<Value> = self.inputs.iter().map(|(t, p, a, b)| serde_json::json!([t, p, a, b])).collect();
        let mut j = serde_json::json!({"digest": self.digest, "inputs": ins, "start": self.start, "steps": self.steps});
        if !self.events.is_empty() {
            j["events"] = Value::Array(self.events.iter().map(|(t, e)| serde_json::json!([t, e])).collect());
        }
        j.to_string()
    }
}

/// HC-6: what one scheduled `fn main` run did.
#[derive(Debug)]
pub struct MainRun {
    pub outputs: Vec<String>,
    pub outcome: String,
    pub steps: usize,
    pub choices: Vec<usize>,
}

/// HC-12: the CPU's cost line (overhead, throughput) in the units of C.6.4.
pub const CPU_COST: (f64, f64) = (16.0, 1.0);

/// HC-12: choose the fabric for an operation of `class` and size `work`:
/// the least overhead + work / throughput among the CPU and every fabric
/// with a `host.Calibrated` row for the class and a `host.Exact` probe
/// equal to the reference. Ties go to the lexically smaller fabric id, the
/// CPU first. A function of the host entries only.
pub fn place_from(calib: &std::collections::BTreeMap<(String, String), (f64, f64)>, exacts: &std::collections::BTreeSet<(String, String)>, class: &str, work: f64) -> String {
    let mut best = ("cpu".to_string(), CPU_COST.0 + work / CPU_COST.1);
    for ((f, c), (o, th)) in calib {
        if c != class || !exacts.contains(&(f.clone(), c.clone())) || *th <= 0.0 {
            continue;
        }
        let t = o + work / th;
        if t < best.1 {
            best = (f.clone(), t);
        }
    }
    best.0
}

/// T18's fixed reduction tree for f64 addition: split at the largest power
/// of two below the length, add left and right. The tree is fixed by the
/// length alone, so any executor that evaluates it gives the same bits.
/// The split of the fixed reduction tree (C.8) for m >= 2 leaves: the
/// left part is the largest power of two below m, so every left subtree
/// is a complete binary tree. One definition for the reference evaluator
/// (`conc.rs` tree and pscan), the placement executor and, by the same
/// formula, every runtime.
pub use crate::conc::tree_split;

pub fn fixed_tree_sum(xs: &[f64]) -> f64 {
    match xs.len() {
        0 => 0.0,
        1 => xs[0],
        n => {
            let h = tree_split(n);
            fixed_tree_sum(&xs[..h]) + fixed_tree_sum(&xs[h..])
        }
    }
}

/// The `cpu.threads` fabric: the same tree, its top `depth` levels run on
/// scoped threads. Bit-identical to [`fixed_tree_sum`] by construction.
pub fn fixed_tree_sum_threads(xs: &[f64], depth: u32) -> f64 {
    if depth == 0 || xs.len() < 2 {
        return fixed_tree_sum(xs);
    }
    let h = tree_split(xs.len());
    let (l, r) = xs.split_at(h);
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| fixed_tree_sum_threads(l, depth - 1));
        let b = fixed_tree_sum_threads(r, depth - 1);
        (a.join().unwrap(), b)
    });
    a + b
}

fn lane_op(class: &str) -> Option<LaneOp> {
    [LaneOp::Add, LaneOp::Sub, LaneOp::Mul, LaneOp::Div, LaneOp::Neg, LaneOp::Sqrt].into_iter().find(|o| o.class() == class)
}

/// HC-12: the fixed tree of [`fixed_tree_sum`] for any combine `f`. Its
/// shape is fixed by the length alone.
pub fn fixed_tree<T: Clone>(xs: &[T], zero: &T, f: &impl Fn(&T, &T) -> T) -> T {
    match xs.len() {
        0 => zero.clone(),
        1 => xs[0].clone(),
        n => {
            let h = tree_split(n);
            f(&fixed_tree(&xs[..h], zero, f), &fixed_tree(&xs[h..], zero, f))
        }
    }
}

/// [`fixed_tree`] with its top `depth` levels on scoped threads. The same
/// tree, so the same bits for every depth.
pub fn fixed_tree_threads<T: Clone + Send + Sync>(xs: &[T], zero: &T, f: &(impl Fn(&T, &T) -> T + Sync), depth: u32) -> T {
    if depth == 0 || xs.len() < 2 {
        return fixed_tree(xs, zero, f);
    }
    let h = tree_split(xs.len());
    let (l, r) = xs.split_at(h);
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| fixed_tree_threads(l, zero, f, depth - 1));
        let b = fixed_tree_threads(r, zero, f, depth - 1);
        (a.join().unwrap(), b)
    });
    f(&a, &b)
}

/// Lane-wise evaluation `out[i] = f(i)` on `threads` scoped threads, each
/// taking a contiguous chunk. Lanes do not read each other, so the result
/// is the sequential one for every thread count.
pub fn lanes_threads<T: Send>(n: usize, threads: usize, f: &(impl Fn(usize) -> T + Sync)) -> Vec<T> {
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| s.spawn(move || (t * chunk..((t + 1) * chunk).min(n)).map(f).collect::<Vec<T>>()))
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// HC-12: the lane-wise f64 operations a fabric can be asked to run. Each
/// is a single IEEE 754 binary64 operation per lane, correctly rounded, so
/// its result has one right answer on every conforming fabric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sqrt,
}

impl LaneOp {
    pub fn name(self) -> &'static str {
        match self {
            LaneOp::Add => "add",
            LaneOp::Sub => "sub",
            LaneOp::Mul => "mul",
            LaneOp::Div => "div",
            LaneOp::Neg => "neg",
            LaneOp::Sqrt => "sqrt",
        }
    }
    /// The placement class, e.g. `lanes.f64.add`.
    pub fn class(self) -> String {
        format!("lanes.f64.{}", self.name())
    }
    /// The CPU reference for one lane.
    pub fn eval(self, a: f64, b: f64) -> f64 {
        match self {
            LaneOp::Add => a + b,
            LaneOp::Sub => a - b,
            LaneOp::Mul => a * b,
            LaneOp::Div => a / b,
            LaneOp::Neg => -a,
            LaneOp::Sqrt => a.sqrt(),
        }
    }
}

/// HC-12: the interface a platform fabric (a wgpu device, an NPU) offers
/// the placement executor. Contract: `lanes` returns, for each lane, the
/// correctly rounded binary64 result of `op` (unary ops ignore `b`), and
/// `sum` evaluates exactly the tree of [`fixed_tree_sum`]. A fabric is used
/// for a class only after `Host::probe_fabric` found its output equal to
/// the CPU reference on the probe vector and logged `host.Exact`; until
/// then, and whenever it reports an error, the CPU result is the one that
/// counts and nothing about the fabric is trusted.
pub trait FabricExec {
    fn id(&self) -> &str;
    fn lanes(&self, op: LaneOp, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String>;
    fn sum(&self, xs: &[f64]) -> Result<f64, String>;
}

/// The probe vector for HC-13: signed zeros, subnormals, infinities, NaN,
/// and values whose sums cancel, so a fabric that reassociates, flushes
/// subnormals or contracts operations differs from the reference.
pub fn probe_vector() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0, -0.0, f64::MIN_POSITIVE / 4.0, 1e308, -1e308, f64::INFINITY, f64::NAN, 1.0, 3.0, 1e16, -1e16, 0.1];
    let mut b = vec![-0.0, 0.0, f64::MIN_POSITIVE / 8.0, 1e308, 1e-308, 1.0, 2.0, 3.0, 7.0, 1.0, 1.0, 0.2];
    // Lanes with subnormal results, so a fabric that flushes them to zero
    // differs from the reference.
    a.extend([f64::MIN_POSITIVE, f64::MIN_POSITIVE / 4.0, f64::MIN_POSITIVE * 0.75]);
    b.extend([0.5, 1.0, -f64::MIN_POSITIVE / 2.0]);
    for i in 0..1000 {
        a.push(((i * 7919) % 1009) as f64 * if i % 3 == 0 { -1e8 } else { 1.0 / 3.0 });
        b.push(((i * 104_729) % 1013) as f64 + 0.5);
    }
    (a, b)
}

fn bits_digest(xs: &[f64]) -> String {
    let mut s = String::with_capacity(xs.len() * 17);
    for x in xs {
        // NaN payloads are not portable across fabrics; every NaN is one class.
        let b = if x.is_nan() { 0x7ff8_0000_0000_0000u64 } else { x.to_bits() };
        s.push_str(&format!("{b:016x}"));
    }
    d_sha(&s)
}

const CPU_THREADS: usize = 8;

/// An input offered to the admitter (E.2a): its source, the source's
/// sequence number (1, 2, ... per source), and the logical tick the source
/// stamped on it. Arrival time is not part of it.
#[derive(Clone, Debug)]
pub struct Sourced {
    pub source: String,
    pub seq: u64,
    pub tick: u64,
    pub name: String,
    pub args: Vec<Arg>,
}

impl Sourced {
    pub fn new(source: &str, seq: u64, tick: u64, name: &str, args: Vec<Arg>) -> Sourced {
        Sourced { source: source.into(), seq, tick, name: name.into(), args }
    }
}

/// The result of closing one admission window (E.2a). `acks` are given only
/// after the barrier covering the window succeeded (exactly once: a source
/// retransmits an unacked input, and a retransmission of an input already
/// in D is a duplicate).
#[derive(Debug, Default)]
pub struct WindowAck {
    pub window: u64,
    pub admitted: Vec<Admitted>,
    pub acks: Vec<(String, u64)>,
    pub dups: Vec<(String, u64)>,
    /// Inputs after a gap in their source's sequence: kept for a later window.
    pub held: Vec<(String, u64)>,
    /// Inputs that are not events of the module: dropped, never in D.
    pub rejected: Vec<(String, u64, String)>,
}

impl<'a> Host<'a> {
    /// Start on an empty store: s₀, then `host.StorageCaps` and the bindings.
    pub fn start(m: &'a Module, store: Box<dyn Store + 'a>) -> Result<Host<'a>, String> {
        Host::start_with(m, store, vec![])
    }

    /// Start with adapters: each one whose request struct the module
    /// declares is bound (`host.Bind`) and must have its result event.
    pub fn start_with(m: &'a Module, store: Box<dyn Store + 'a>, adapters: Vec<Box<dyn Adapter + 'a>>) -> Result<Host<'a>, String> {
        let mut h = Host::fresh(m, store, adapters)?;
        h.start_entries()?;
        Ok(h)
    }

    fn fresh(m: &'a Module, store: Box<dyn Store + 'a>, adapters: Vec<Box<dyn Adapter + 'a>>) -> Result<Host<'a>, String> {
        let mut bound = vec![];
        let mut adapters = adapters;
        // E.4c: every request struct of a kind the host performs is bound;
        // its result is delivered as declared, or recorded in D
        for a in adapters.iter_mut() {
            if declares_struct(m, a.kind()) {
                bound.push(a.kind());
                // HD-3: received bytes taken as text carry a character that
                // a read ends inside to the next read
                let ev = format!("{}Done", a.kind());
                let text = event_payload(m, &ev).and_then(|p| p.iter().find(|(k, _)| k == "data")).is_some_and(|(_, t)| bytes_field(&ev, "data") == Bytes::Here && takes_text(t, m));
                if text {
                    a.text();
                }
            }
        }
        let http = adapters.iter().any(|a| a.kind() == "HttpRespond");
        let delivered = host_named(m);
        let mut notes = vec![];
        for (e, held) in [("Synced", true), ("WallTime", true), ("StepKept", true), ("Restarted", true), ("Presented", true)] {
            notes.extend(delivery_note(m, e, held));
        }
        if http {
            match event_payload(m, "HttpRequest") {
                None => notes.push(format!("http: event HttpRequest is not declared, so each request is recorded in D as host.Kept and {}", if declares_event(m, "HttpMalformed") { "delivered as HttpMalformed with status 501" } else { "answered 501 Not Implemented by the host" })),
                Some(_) => notes.extend(delivery_note(m, "HttpRequest", false)),
            }
            match event_payload(m, "HttpMalformed") {
                None => notes.push("http: event HttpMalformed is not declared, so a request that breaks the syntax, a limit or a deadline is recorded in D as host.Kept and answered by the host with its status".to_string()),
                Some(_) => notes.extend(delivery_note(m, "HttpMalformed", false)),
            }
            if !declares_struct(m, "HttpRespond") {
                notes.push("http: struct HttpRespond is not declared, so the program cannot answer, and the host answers each request it delivers with 501 Not Implemented".to_string());
            } else if !responds(m) {
                notes.push("http: struct HttpRespond does not declare req and status as integers, so the program cannot answer, and the host answers each request it delivers with 501 Not Implemented".to_string());
            }
            for e in ["HttpRequest", "HttpMalformed"] {
                if responds(m) && event_payload(m, e).is_some_and(|p| !p.iter().any(|(k, t)| k == "id" && takes_int(t, m))) {
                    notes.push(format!("http: event {e} does not take id as an integer, so the program cannot answer it, and the host answers each with its status"));
                }
            }
        }
        for k in &bound {
            let e = format!("{k}Done");
            if declares_event(m, &e) {
                notes.extend(delivery_note(m, &e, false));
            } else {
                notes.push(format!("struct {k}: event {e} is not declared, so each request is performed and its result is recorded in D as host.Kept, not delivered"));
            }
        }
        let mut it = Interp::new(m);
        let state = it.initial_state().map_err(why)?;
        let state_hash = d_sha(&state_pos_json(&state, m));
        Ok(Host { m, it, state, state_hash, prev: D_GENESIS.into(), seq: 0, store, stopped: false, pending: 0, pending_real: 0, confirming: false, durable: None, from_snapshot: None, adapters, bound, queue: Default::default(), unanswered: Default::default(), descends: Default::default(), drain_max: usize::MAX, admitted: Default::default(), window: 0, closed_tick: 0, offered: vec![], term: 0, task: None, fabrics: vec![], calib: Default::default(), exacts: Default::default(), owe: Default::default(), delivered, notes, kept_outcomes: vec![], last_tick: 0, synced: None, outputs: None, clock: Default::default(), shown: vec![], last_view: None, frames: None, rap: Rap::default() })
    }

    /// GT-9: the host loads every blob the program's graph terms name from
    /// the store, each checked against its digest, and records in D the
    /// digests and lengths it loaded (`host.Blobs`), before any input. A
    /// blob no store holds stops the start: the program cannot run.
    fn blobs_entry(&mut self) -> Result<(), String> {
        let ds = crate::graph::blob_digests(self.m);
        if ds.is_empty() {
            return Ok(());
        }
        let mut items = vec![];
        for d in &ds {
            let b = crate::graph::blob_bytes(d)?;
            items.push(format!("{{\"bytes\":{},\"digest\":\"{d}\"}}", b.len()));
        }
        self.append(host_event("Blobs", &format!("{{\"blobs\":[{}]}}", items.join(","))))?;
        Ok(())
    }

    fn start_entries(&mut self) -> Result<(), String> {
        let caps = self.store.caps();
        self.append(host_event("StorageCaps", &format!("{{\"barrier\":\"{caps}\"}}")))?;
        self.append(host_event("FabricCaps", &fabric_caps()))?;
        self.blobs_entry()?;
        let http = self.adapters.iter().any(|a| a.kind() == "HttpRespond");
        for (e, k) in binds(self.m, &self.bound, http) {
            self.append(host_event("Bind", &format!("{{\"event\":\"{e}\",\"kind\":\"{k}\"}}")))?;
        }
        self.confirm()
    }

    /// Open D from a file: verify (HS-4), cut a bad suffix, replay the kept
    /// prefix (HC-8), then log `host.Truncated` and `host.StorageCaps`.
    pub fn open_file(m: &'a Module, path: &Path) -> Result<Host<'a>, String> {
        Host::open_file_with(m, path, vec![])
    }

    /// `open_file` with adapters. A request that was emitted before a crash
    /// and has no result in D gets exactly one result now: `failed`, with
    /// the error `interrupted` (HC-3: the program sees one outcome).
    pub fn open_file_with(m: &'a Module, path: &Path, adapters: Vec<Box<dyn Adapter + 'a>>) -> Result<Host<'a>, String> {
        Host::open_file_snap(m, path, adapters, None)
    }

    /// `open_file_with` from a snapshot (HS-5). The whole chain is still
    /// verified (HS-4); the snapshot is used only when it is valid for D
    /// (see [`Host::load_snapshot`]), else it is discarded and D is refolded
    /// from genesis. Either way the meaning is the same.
    pub fn open_file_snap(m: &'a Module, path: &Path, adapters: Vec<Box<dyn Adapter + 'a>>, snap: Option<&Snapshot>) -> Result<Host<'a>, String> {
        Host::open_inner(m, path, adapters, snap, usize::MAX)
    }

    /// `open_file_with` for the host loop: every drain, the one at open
    /// among them, performs at most `drain_max` requests (HC-24), so the
    /// requests the restart's results make never keep the loop from
    /// starting and reading its stop.
    pub fn open_file_bounded(m: &'a Module, path: &Path, adapters: Vec<Box<dyn Adapter + 'a>>, drain_max: usize) -> Result<Host<'a>, String> {
        Host::open_inner(m, path, adapters, None, drain_max)
    }

    fn open_inner(m: &'a Module, path: &Path, adapters: Vec<Box<dyn Adapter + 'a>>, snap: Option<&Snapshot>, drain_max: usize) -> Result<Host<'a>, String> {
        let (mut fs, text) = FileStore::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if text.is_empty() {
            let mut h = Host::start_with(m, Box::new(fs), adapters)?;
            h.bound_drains(drain_max);
            return Ok(h);
        }
        let lines: Vec<&str> = text.split_terminator('\n').collect();
        let keep = verify_prefix(&lines, m);
        // HC-37: a checkpoint that does not verify, or that the entry after
        // it does not link to, fails closed: D is not cut and not opened
        if let Some(first) = lines.first().and_then(|l| Entry::parse_line(l))
            && first.kind() == CHECKPOINT
            && (keep == 0 || (keep == 1 && lines.get(1).is_some_and(|l| Entry::parse_line(l).is_some())))
        {
            return Err(format!("host fault: {}: the checkpoint that begins D does not verify, or the entry after it does not link to its head (HC-37); D is left unchanged", path.display()));
        }
        let dropped = lines.len() - keep;
        if dropped > 0 || !text.ends_with('\n') {
            let len: usize = lines[..keep].iter().map(|l| l.len() + 1).sum();
            fs.cut(len as u64).map_err(|e| format!("host.StorageFault: cut: {e}"))?;
        }
        let entries: Vec<Entry> = lines[..keep].iter().map(|l| Entry::parse_line(l).unwrap()).collect();
        let mut h = Host::fresh(m, Box::new(fs), adapters)?;
        h.bound_drains(drain_max);
        let from = match snap {
            Some(sn) if h.load_snapshot(sn, &entries) => entries.iter().position(|e| e.seq == sn.seq).map_or(0, |i| i + 1),
            _ => 0,
        };
        h.refold(&entries[from..])?;
        // HC-21: a D that ends after the entry of a kept step, or after its
        // `host.StepKept`, is completed before anything else is appended
        h.complete_owed()?;
        let caps = h.store.caps();
        if dropped > 0 {
            h.append(host_event("Truncated", &format!("{{\"dropped\":{dropped}}}")))?;
        }
        h.append(host_event("StorageCaps", &format!("{{\"barrier\":\"{caps}\"}}")))?;
        h.append(host_event("FabricCaps", &fabric_caps()))?;
        h.blobs_entry()?;
        h.restarted()?;
        h.drain()?;
        h.confirm()?;
        Ok(h)
    }

    /// HC-22: the adapter resources of the process that wrote D are gone.
    /// Append `host.Restarted` {below}, below being that entry's own seq;
    /// let every adapter issue ids from below upward; admit the bound
    /// `Restarted(below)` when declared; then give every request that was
    /// outstanding in D one result, `failed` with `interrupted` (HC-3). The
    /// requests the `Restarted` step makes are not among them.
    fn restarted(&mut self) -> Result<(), String> {
        let lost = std::mem::take(&mut self.queue);
        let below = self.seq;
        self.append(host_event("Restarted", &format!("{{\"below\":{below}}}")))?;
        for a in self.adapters.iter_mut() {
            a.restart(below);
        }
        let p = restarted_map(below);
        let plan = self.plan("Restarted", &p, true);
        self.deliver_plan("Restarted", &p, plan, How::Delivery)?;
        for q in lost {
            self.answer(&q.kind, q.req, EffectResult::failed("interrupted"))?;
        }
        Ok(())
    }

    /// The n of the last `Tick(n)` in D (0 when none), so a host loop
    /// started on an existing D continues the count (HC-22).
    pub fn last_tick(&self) -> u64 {
        self.last_tick
    }

    fn note_tick(&mut self, name: &str, ev: &V) {
        if name == "Tick"
            && let V::Rec(_, fs) = ev
            && let Some((_, V::Int(n, _))) = fs.iter().find(|(f, _)| f == "n")
        {
            self.last_tick = (*n).max(0) as u64;
        }
    }

    /// A snapshot of the state after the last entry (HS-5). Taken only
    /// between batches, when no effect request is outstanding, so the
    /// entries after it rebuild every request that matters.
    pub fn snapshot(&self) -> Option<Snapshot> {
        if !self.queue.is_empty() || !self.descends.is_empty() || self.seq == 0 || self.task.is_some() || !self.owe.is_empty() {
            return None;
        }
        // HC-35: the dedupe keys and results of RAP invocations fold from
        // every entry, so a log that holds any is folded from its start
        if !self.rap.calls.is_empty() {
            return None;
        }
        // HC-29: frames derive from every entry (the first is `start`, the
        // time counts ticks, a confirmation names any earlier frame), so a
        // program that emits them is always folded from the start of D
        if declares_struct(self.m, crate::frame::FRAME) {
            return None;
        }
        Some(Snapshot { seq: self.seq - 1, state_hash: self.state_hash.clone(), bytes: state_pos_json(&self.state, self.m) })
    }

    /// Use a snapshot only when SHA-256 of its bytes is its state_hash, that
    /// equals D's `state_hash` at its seq, and the bytes decode (guided by
    /// the declared field types) to a state that re-encodes to the same
    /// bytes. Anything else: discarded, nothing changes (HS-5).
    fn load_snapshot(&mut self, sn: &Snapshot, entries: &[Entry]) -> bool {
        if declares_struct(self.m, crate::frame::FRAME) {
            return false; // HC-29: folded from the start of D
        }
        let Some(e) = entries.iter().find(|e| e.seq == sn.seq) else { return false };
        if d_sha(&sn.bytes) != sn.state_hash || e.state_hash != sn.state_hash {
            return false;
        }
        let Some(st) = self.decode_state(&sn.bytes) else { return false };
        self.state = st;
        self.state_hash = sn.state_hash.clone();
        self.prev = e.hash.clone();
        self.seq = sn.seq + 1;
        self.from_snapshot = Some(sn.seq);
        true
    }

    /// The state whose positional form (`state_pos_json`) is `bytes`,
    /// decoded guided by the declared field types in declaration order; `None`
    /// unless it re-encodes to the same bytes (HS-5, HC-37).
    fn decode_state(&mut self, bytes: &str) -> Option<Vec<Region>> {
        let v = serde_json::from_str::<Value>(bytes).ok()?;
        let mut st = self.it.initial_state().ok()?;
        if v.as_array().map(Vec::len) != Some(st.len()) {
            return None;
        }
        for (ri, (_, fs)) in st.iter_mut().enumerate() {
            if v[ri].as_array().map(Vec::len) != Some(fs.len()) {
                return None;
            }
            for (fi, (_, val, ty)) in fs.iter_mut().enumerate() {
                let x = json_expr(&v[ri][fi], Some(ty), self.m).ok()?;
                self.it.scopes = vec![HashMap::new()];
                self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
                // typed as the initial state's fields are, so a decoded
                // value is the value δ left, not an untyped literal
                let v = self.it.eval(&x).ok()?;
                *val = self.it.fit(v, ty).ok()?;
            }
        }
        (state_pos_json(&st, self.m) == bytes).then_some(st)
    }

    /// HC-37: everything the fold of D has reached that replay of the
    /// entries after this point depends on, as canonical JSON values, keyed
    /// as a checkpoint carries them. Each item and its reason:
    /// - `state`: the canonical named form of the state (6.1); the SHA-256
    ///   of its positional form is the state hash unless a task is live
    ///   (`task`);
    /// - `head`: the hash of the last entry folded, which the next entry's
    ///   `prev_hash` names (HS-4);
    /// - `queue`: every request emitted and not yet answered (an output of a
    ///   declared struct with an integer `req`), with the seq that requested
    ///   it: a later result answers one of them, and a restart gives each
    ///   one a host performs one result, `interrupted` (HC-3, HC-22);
    /// - `descends`: the requests whose results descend from a delivered
    ///   `StepKept`, so a later kept step is not delivered again (HC-21);
    /// - `admitted`, `window`, `closed_tick`: per source the highest seq in
    ///   D, the next window id and the last closed tick, the dedupe and
    ///   window state of HC-15 to HC-17;
    /// - `term`: the admitter's term, which a later `host.Term` must exceed
    ///   (HC-19);
    /// - `task`: a live `fn main` task, part of the state D folds to (HC-6);
    /// - `calib`, `exacts`: the calibrations and exact probes placement
    ///   reads (HC-12, HC-13);
    /// - `last_tick`: the n of the last `Tick(n)`, which a host loop
    ///   continues (HC-22);
    /// - `rap`: every RAP invocation: its key, which dedupes every later
    ///   invocation (HC-35), whether it was stepped, how its step ended,
    ///   the length and SHA-256 of its result and the status that settled
    ///   it (HC-36). A kept step's text is never carried (HC-21): the state
    ///   it was kept on is, and the text is derived again from it.
    fn carried(&self) -> serde_json::Map<String, Value> {
        use serde_json::json;
        let mut p = serde_json::Map::new();
        p.insert("admitted".into(), Value::Object(self.admitted.iter().map(|(k, v)| (k.clone(), json!(v))).collect()));
        let fin = |x: f64| if x.is_finite() { json!(x) } else { Value::Null };
        p.insert("calib".into(), Value::Array(self.calib.iter().map(|((f, c), (o, t))| json!([f, c, fin(*o), fin(*t)])).collect()));
        p.insert("closed_tick".into(), json!(self.closed_tick));
        p.insert("descends".into(), Value::Array(self.descends.iter().map(|(k, r, q)| json!([k, r, q])).collect()));
        p.insert("exacts".into(), Value::Array(self.exacts.iter().map(|(f, c)| json!([f, c])).collect()));
        p.insert("head".into(), json!(self.prev));
        p.insert("last_tick".into(), json!(self.last_tick));
        p.insert("queue".into(), Value::Array(self.unanswered.iter().map(|q| json!([q.kind, q.req, q.at, q.json])).collect()));
        let rap: Vec<Value> = self
            .rap
            .calls
            .iter()
            .map(|(c, x)| {
                let outcome = match &x.outcome {
                    Some(crate::rap::Outcome::Applied { event, seq, outputs, state_hash }) => json!({"applied": {"event": event, "outputs": outputs, "seq": seq, "state_hash": state_hash}}),
                    Some(crate::rap::Outcome::Kept { event, seq, cause, .. }) => json!({"kept": {"cause": cause, "event": event, "seq": seq, "state": x.kept_state}}),
                    _ => Value::Null,
                };
                let answered = x.body.as_ref().map_or(Value::Null, |b| json!({"length": b.len(), "sha256": d_sha(b)}));
                json!({"answered": answered, "call": c, "inv": x.inv, "of": x.of, "outcome": outcome, "sent": x.sent, "stepped": x.stepped})
            })
            .collect();
        p.insert("rap".into(), Value::Array(rap));
        p.insert("state".into(), json!(crate::interp::state_json(&self.state, self.m)));
        let task = self.task.as_ref().map_or(Value::Null, |t| {
            let ins: Vec<Value> = t.inputs.iter().map(|(a, b, c, d)| json!([a, b, c, d])).collect();
            let evs: Vec<Value> = t.events.iter().map(|(a, b)| json!([a, b])).collect();
            json!({"digest": t.digest, "events": evs, "inputs": ins, "last_tick": t.last_tick, "start": t.start, "steps": t.steps})
        });
        p.insert("task".into(), task);
        p.insert("term".into(), json!(self.term));
        p.insert("window".into(), json!(self.window));
        p
    }

    /// HC-37: what a checkpoint after the entries folded so far carries
    /// ([`Host::carried`]), or why no checkpoint may stand here: D owes
    /// entries after the last one folded (HC-10b, HC-21, HC-22, HC-35), the
    /// last one announces an input of a window that has not followed
    /// (HC-17), or the module declares `Frame`, whose frames derive from
    /// every entry of D (HC-29).
    pub fn checkpoint_value(&self) -> Result<serde_json::Map<String, Value>, String> {
        if self.seq == 0 {
            return Err("no entry to stand for".into());
        }
        if !self.owe.is_empty() || self.rap.stepping.is_some() {
            return Err(format!("D owes entries after seq {} (HC-10b, HC-21, HC-22, HC-35): a checkpoint stands only where every input's entries are complete", self.seq - 1));
        }
        if declares_struct(self.m, crate::frame::FRAME) {
            return Err("the module declares Frame, whose frames derive from every entry of D (HC-29)".into());
        }
        Ok(self.carried())
    }

    /// HC-37: the digest of everything the fold has reached, owed entries
    /// included, so two folds can be compared wherever they end.
    pub(crate) fn fold_digest(&self) -> String {
        let mut p = self.carried();
        p.insert("owe".into(), Value::String(format!("{:?}", self.owe)));
        p.insert("state_hash".into(), Value::String(self.state_hash.clone()));
        p.insert("seq".into(), Value::from(self.seq));
        d_sha(&crate::interp::canon_json(&Value::Object(p)))
    }

    /// HC-37: take the place of the entries a checkpoint stands for. Every
    /// item is read back, the state is decoded and the state hash
    /// recomputed, each kept RAP step's text is derived again from its
    /// state, and each result's bytes are derived again and compared with
    /// the length and SHA-256 carried. The values read back must give the
    /// carried payload byte for byte. Anything else fails closed.
    fn load_checkpoint(&mut self, e: &Entry) -> Result<(), String> {
        let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
        let p = v["payload"].as_object().ok_or("no payload")?;
        let head = checkpoint_head(e).ok_or("no head")?;
        let u = |k: &str| p.get(k).and_then(Value::as_u64).ok_or_else(|| format!("no {k}"));
        let arr = |k: &str| p.get(k).and_then(Value::as_array).cloned().ok_or_else(|| format!("no {k}"));
        let text = |x: &Value| x.as_str().map(String::from).ok_or_else(|| format!("{x} is not text"));
        let num = |x: &Value| x.as_u64().ok_or_else(|| format!("{x} is not a count"));
        let sb = p.get("state").and_then(Value::as_str).ok_or("no state")?;
        self.state = self.decode_named(sb).ok_or("the state does not decode to the module's state")?;
        self.admitted = p.get("admitted").and_then(Value::as_object).ok_or("no admitted")?.iter().map(|(k, x)| Ok((k.clone(), num(x)?))).collect::<Result<_, String>>()?;
        self.calib = arr("calib")?.iter().map(|x| Ok(((text(&x[0])?, text(&x[1])?), (x[2].as_f64().unwrap_or(f64::INFINITY), x[3].as_f64().unwrap_or(f64::INFINITY))))).collect::<Result<_, String>>()?;
        self.closed_tick = u("closed_tick")?;
        self.descends = arr("descends")?.iter().map(|x| Ok((text(&x[0])?, num(&x[1])?, num(&x[2])?))).collect::<Result<_, String>>()?;
        self.exacts = arr("exacts")?.iter().map(|x| Ok((text(&x[0])?, text(&x[1])?))).collect::<Result<_, String>>()?;
        self.last_tick = u("last_tick")?;
        self.unanswered = arr("queue")?.iter().map(|x| Ok(Queued { kind: text(&x[0])?, req: num(&x[1])?, at: num(&x[2])?, json: x[3].clone() })).collect::<Result<_, String>>()?;
        // the part this host's adapters perform, as the fold from the start gives it
        self.queue = self.unanswered.iter().filter(|q| self.bound.iter().any(|k| *k == q.kind)).cloned().collect();
        self.term = u("term")?;
        self.window = u("window")?;
        self.task = match p.get("task") {
            Some(Value::Null) => None,
            Some(t) => Some(MainTask {
                start: t["start"].clone(),
                inputs: t["inputs"].as_array().ok_or("task inputs")?.iter().map(|x| Ok((num(&x[0])?, x[1].as_bool().ok_or("task input")?, text(&x[2])?, text(&x[3])?))).collect::<Result<_, String>>()?,
                events: t["events"].as_array().ok_or("task events")?.iter().map(|x| Ok((num(&x[0])?, text(&x[1])?))).collect::<Result<_, String>>()?,
                steps: num(&t["steps"])? as usize,
                digest: text(&t["digest"])?,
                last_tick: num(&t["last_tick"])?,
                run: None,
            }),
            None => return Err("no task".into()),
        };
        for x in arr("rap")? {
            let c = num(&x["call"])?;
            let inv = x["inv"].as_object().cloned().ok_or("rap: no inv")?;
            let key = rap_key(&inv).ok_or("rap: an invocation without a string group_id and id")?;
            let of = match &x["of"] {
                Value::Null => None,
                o => Some(num(o)?),
            };
            let mut call = RapCall { inv, of, stepped: x["stepped"].as_bool().ok_or("rap: stepped")?, ..Default::default() };
            match of {
                None => {
                    if self.rap.keys.insert(key, c).is_some() {
                        return Err(format!("rap: call {c} repeats a key"));
                    }
                    call.op = Some(self.rap_validate(&call.inv));
                }
                Some(o) => {
                    if self.rap.keys.get(&key) != Some(&o) {
                        return Err(format!("rap: call {c} is a duplicate of {o}, which is not the first of its key"));
                    }
                }
            }
            if call.stepped && !matches!(call.op, Some(Ok(_))) {
                return Err(format!("rap: call {c} is stepped and is no invocation that conforms"));
            }
            let o = &x["outcome"];
            if !o.is_null() {
                let Some(Ok((name, args))) = call.op.clone() else { return Err(format!("rap: call {c} has an outcome and no step")) };
                if let Some(a) = o.get("applied") {
                    let outputs = a["outputs"].as_array().ok_or("rap: outputs")?.iter().map(text).collect::<Result<_, String>>()?;
                    call.outcome = Some(crate::rap::Outcome::Applied { event: text(&a["event"])?, seq: num(&a["seq"])?, outputs, state_hash: text(&a["state_hash"])? });
                } else if let Some(k) = o.get("kept") {
                    // HC-21: the kept text is derived again from the state the
                    // step was kept on, never read from D
                    let ks = text(&k["state"])?;
                    let st = self.decode_named(&ks).ok_or_else(|| format!("rap: the state call {c} was kept on does not decode"))?;
                    self.it.scopes = vec![HashMap::new()];
                    self.it.fuel = crate::interp::fuel_limit(); // HC-7a
                    let ev = match self.it.construct(&name, &args) {
                        Ok(Some(v)) => v,
                        _ => return Err(format!("rap: the event of call {c} cannot be built")),
                    };
                    let r = match self.it.delta_r(&st, &ev) {
                        Err(r) if r.cause.as_str() == text(&k["cause"])? => r,
                        _ => return Err(format!("rap: the step of call {c} is not kept with cause {} on the state carried", k["cause"])),
                    };
                    call.outcome = Some(crate::rap::Outcome::Kept { event: text(&k["event"])?, seq: num(&k["seq"])?, cause: r.cause.as_str().to_string(), text: r.text });
                    call.kept_state = Some(ks);
                } else {
                    return Err(format!("rap: call {c} has an outcome of no known kind"));
                }
            }
            let answered = x["answered"].clone();
            let sent = match &x["sent"] {
                Value::Null => None,
                s => Some(u16::try_from(num(s)?).map_err(|_| "rap: sent")?),
            };
            self.rap.add(c, call);
            if !answered.is_null() {
                let body = self.rap_derive(c).ok_or_else(|| format!("rap: call {c} is answered and its result cannot be derived"))?;
                if answered["length"].as_u64() != Some(body.len() as u64) || answered["sha256"].as_str() != Some(d_sha(&body).as_str()) {
                    return Err(format!("rap: the result of call {c} differs from the one derived ({} bytes, {})", body.len(), d_sha(&body)));
                }
                self.rap.answered(c, body);
            }
            if let Some(st) = sent {
                if answered.is_null() {
                    return Err(format!("rap: call {c} is settled without a result"));
                }
                self.rap.settled(c, st);
            }
        }
        self.prev = head;
        self.seq = e.seq + 1;
        self.rehash();
        if self.state_hash != e.state_hash {
            return Err(format!("the state carried hashes to {}, the checkpoint says {}", self.state_hash, e.state_hash));
        }
        // what was read back gives the carried payload byte for byte, so
        // nothing is carried that replay does not use, and nothing is lost
        let mut again = self.carried();
        for k in ["archive", "http"] {
            if let Some(x) = p.get(k) {
                again.insert(k.into(), x.clone());
            }
        }
        if host_event("Checkpoint", &crate::interp::canon_json(&Value::Object(again))) != e.event {
            return Err("the values read back do not give the payload byte for byte".into());
        }
        Ok(())
    }

    /// HC-37: the state whose canonical named form (`state_json`, 6.1) is
    /// `bytes`, each field decoded by its declared type as an event's
    /// fields are; `None` unless it re-encodes to the same bytes. The named
    /// form, not the positional one, is carried, because the positional form
    /// leaves out what only the types say (a variant is its index, an
    /// optional value is the value), and the decoding must give back the
    /// value δ left exactly.
    fn decode_named(&mut self, bytes: &str) -> Option<Vec<Region>> {
        let v = serde_json::from_str::<Value>(bytes).ok()?;
        let mut st = self.it.initial_state().ok()?;
        if v.as_object().map(serde_json::Map::len) != Some(st.len()) {
            return None;
        }
        for (rn, fs) in st.iter_mut() {
            let r = v.get(rn.as_str())?.as_object()?;
            if r.len() != fs.len() {
                return None;
            }
            for (fname, val, ty) in fs.iter_mut() {
                let x = json_expr(r.get(fname.as_str())?, Some(ty), self.m).ok()?;
                self.it.scopes = vec![HashMap::new()];
                self.it.fuel = crate::interp::fuel_limit(); // HC-7a
                let e = self.it.eval(&x).ok()?;
                *val = self.it.fit(e, ty).ok()?;
            }
        }
        (crate::interp::state_json(&st, self.m) == bytes).then_some(st)
    }

    /// Fold verified entries into this host without appending (HC-8).
    fn refold(&mut self, entries: &[Entry]) -> Result<(), String> {
        // E.2a: a `host.Input` marks its source and seq admitted only when the
        // program entry it announces follows it (a cut between them admits nothing).
        let mut announced: Option<(String, u64)> = None;
        for e in entries {
            // HC-37: a checkpoint stands for the entries before it, and only
            // at the start of D, on a host that has folded nothing
            if e.kind() == CHECKPOINT {
                if self.seq != 0 || self.prev != D_GENESIS {
                    return Err(format!("host fault: seq {}: a checkpoint stands only at the start of D", e.seq));
                }
                self.load_checkpoint(e).map_err(|x| format!("host fault: seq {}: the checkpoint does not hold: {x}", e.seq))?;
                continue;
            }
            // HC-10b, HC-21, HC-22: the entry the previous one requires
            let mut how = self.meet(e)?;
            if e.is_host() {
                announced = None;
                self.fold_admission(e, &mut announced);
                self.fold_sched(e).map_err(|x| format!("host fault: seq {}: {x}", e.seq))?;
                let at = |k: &str| serde_json::from_str::<Value>(&e.event).ok().and_then(|v| v["payload"][k].as_u64());
                match e.kind().as_str() {
                    "host.Restarted" => {
                        // HC-22: below is the entry's own seq, and the bound
                        // Restarted with the same payload follows when declared
                        if at("below") != Some(e.seq) {
                            return Err(format!("host fault: seq {}: host.Restarted must have below {}", e.seq, e.seq));
                        }
                        let p = restarted_map(e.seq);
                        let plan = self.plan("Restarted", &p, true);
                        let nexts = self.nexts("Restarted", &p, plan, How::Delivery);
                        self.owe.extend(nexts);
                    }
                    // E.4c: a host.Kept no entry before it owed records a
                    // delivery replay plans again from the value it holds
                    "host.Kept" if how != Some(How::Owed) => self.fold_kept(e)?,
                    "host.Presented" => {
                        // HC-29: a confirmation names a frame D derived before it
                        let d = serde_json::from_str::<Value>(&e.event).ok().and_then(|v| v["payload"]["digest"].as_str().map(String::from));
                        if !matches!((at("seq"), &d), (Some(q), Some(d)) if self.shown.iter().any(|(s, x)| *s == q && x == d)) {
                            return Err(format!("host fault: seq {}: host.Presented names no frame derived from the entries before it", e.seq));
                        }
                    }
                    "host.Typed" => {
                        // HC-10b: seq is the typed input's, the entry after this one
                        if at("seq") != Some(e.seq + 1) {
                            return Err(format!("host fault: seq {}: host.Typed must have seq {}", e.seq, e.seq + 1));
                        }
                        self.owe.push_back(Next::Typed { seq: e.seq + 1 });
                    }
                    k if k.starts_with("host.Rap") => self.rap_fold(e).map_err(|x| format!("host fault: seq {}: {x}", e.seq))?,
                    _ => {}
                }
                self.rehash();
            } else if let Some((src, q)) = announced.take() {
                self.admitted.insert(src, q);
                how = Some(How::Sourced);
            }
            if !e.is_host() {
                let (name, args) = decode_event(&e.event, self.m)?;
                let how = how.unwrap_or(How::Input);
                // HC-21, HC-22: the host delivers `StepKept` and `Restarted`
                // only after their host entries; one that is neither that
                // delivery, typed nor announced is not a fact of the run
                if how == How::Input && (name == "StepKept" || name == "Restarted") {
                    return Err(format!("host fault: seq {}: {name} is delivered only after host.{name}, and this one is neither that delivery nor typed (host.Typed)", e.seq));
                }
                self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
                let ev = match self.it.construct(&name, &args) {
                    Ok(Some(v)) => v,
                    Ok(None) => return Err(format!("seq {}: `{name}` is not an event of this module", e.seq)),
                    Err(f) => return Err(format!("seq {}: {}", e.seq, why(f))),
                };
                let ej = event_json(&name, &ev, self.m);
                if ej != e.event {
                    return Err(format!("seq {}: event re-encodes as {ej}, log has {}", e.seq, e.event));
                }
                // HC-3: the request this result answers is no longer outstanding
                let ans = answered(&how, &name, &ev);
                if let Some((k, r)) = &ans {
                    if let Some(i) = self.queue.iter().position(|q| q.kind == *k && q.req == *r) {
                        self.queue.remove(i);
                    }
                    self.unqueue(k, *r);
                }
                let under = self.descent(&how, e.seq, ans);
                self.note_tick(&name, &ev);
                match self.it.delta_r(&self.state, &ev) {
                    Ok((s2, outs)) => {
                        let fs = self.note_frames(e.seq, &name, Some(&outs));
                        if let Some(f) = &mut self.frames {
                            f.extend(fs);
                        }
                        self.state = s2;
                        self.rehash();
                        self.rap_capture(e.seq, &name, Ok(&outs));
                        self.enqueue(&outs, e.seq);
                        self.note_descent(&outs, under);
                        if let Some(o) = &mut self.outputs {
                            o.push((e.seq, outs.iter().map(|x| serde_json::from_str(&crate::interp::form_value(x, self.m).json()).unwrap_or(Value::Null)).collect()));
                        }
                    }
                    Err(r) => {
                        self.note_frames(e.seq, &name, None);
                        self.rap_capture(e.seq, &name, Err((r.cause, &r.text)));
                        // HC-21: replay re-derives the kept step and its cause
                        let bind = under.is_none() && declares_event(self.m, "StepKept");
                        self.owe.push_back(Next::HostKept { seq: e.seq, cause: r.cause, bind });
                        self.kept_outcomes.push((e.seq, r.cause.as_str(), r.text));
                    }
                }
            }
            if e.state_hash != self.state_hash {
                return Err(format!("host fault: seq {} state_hash {} but replay gives {}", e.seq, e.state_hash, self.state_hash));
            }
            self.prev = e.hash.clone();
            self.seq = e.seq + 1;
        }
        self.durable = self.seq.checked_sub(1);
        self.synced = self.durable;
        Ok(())
    }

    fn append(&mut self, event: String) -> Result<Entry, String> {
        if self.stopped {
            return Err("host.StorageFault: admitter stopped after a failed barrier".into());
        }
        let e = Entry::new(self.m, self.seq, event, &self.prev, &self.state_hash);
        if let Err(x) = self.store.append(&e.line()) {
            self.stopped = true;
            return Err(format!("host.StorageFault: seq {}: {x}", e.seq));
        }
        self.prev = e.hash.clone();
        self.seq += 1;
        self.pending += 1;
        if !self.confirming {
            self.pending_real += 1;
        }
        Ok(e)
    }

    /// The barrier for everything pending, when something other than the
    /// confirmation of the last barrier is pending (so a batch that admitted
    /// nothing appends nothing); on success log `host.Synced` and the bound
    /// `Synced` event (HS-6). A failure stops the admitter (HS-3).
    fn confirm(&mut self) -> Result<(), String> {
        if self.pending_real == 0 {
            return Ok(());
        }
        self.barrier()
    }

    /// The barrier for every pending entry (HS-2 before an effect).
    fn barrier(&mut self) -> Result<(), String> {
        if self.pending == 0 {
            return Ok(());
        }
        if let Err(x) = self.store.barrier() {
            self.stopped = true;
            return Err(format!("host.StorageFault: seq {}: {x}", self.seq - 1));
        }
        self.pending = 0;
        self.pending_real = 0;
        let upto = self.seq - 1;
        self.synced = Some(upto);
        if self.store.caps() == "volatile" {
            return Ok(()); // E.3: no entry of a volatile store is reported durable
        }
        self.durable = Some(upto);
        self.confirming = true;
        let r = self.append(host_event("Synced", &format!("{{\"upto\":{upto}}}"))).and_then(|_| {
            if declares_event(self.m, "Synced") {
                let p = one_field("upto", upto);
                let plan = self.plan("Synced", &p, true);
                self.deliver_plan("Synced", &p, plan, How::Delivery)?;
            }
            Ok(())
        });
        self.confirming = false;
        r
    }

    /// HC-21: the delivered `StepKept` entry the input at `seq` descends
    /// from: itself when it is that delivery (`Root`); for the result of a
    /// request (K, req), the one the earliest request K with that req in
    /// `descends` descends from, which it answers (taken off); else none.
    /// A typed input descends from nothing.
    fn descent(&mut self, how: &How, seq: u64, ans: Option<(String, u64)>) -> Option<u64> {
        match how {
            How::Root => Some(seq),
            How::Typed | How::Sourced | How::Delivery | How::Owed => None,
            How::Input | How::Answers(..) => {
                let (k, r) = ans?;
                let i = self.descends.iter().position(|(qk, qr, _)| *qk == k && *qr == r)?;
                self.descends.remove(i).map(|x| x.2)
            }
        }
    }

    /// HC-21: note the requests among the outputs of a step that descends
    /// from the `StepKept` entry `under`.
    fn note_descent(&mut self, outs: &[V], under: Option<u64>) {
        let Some(r) = under else { return };
        for o in outs {
            if let V::Rec(n, fs) = o
                && let Some(req) = req_of(fs)
                && declares_event(self.m, &format!("{n}Done"))
            {
                self.descends.push_back((n.clone(), req, r));
            }
        }
    }

    /// Queue the effect requests among a step's outputs, in emit order;
    /// `at` is the seq of the entry that requested them.
    fn enqueue(&mut self, outs: &[V], at: u64) {
        for o in outs {
            if let V::Rec(n, fs) = o
                && let Some(k) = self.bound.iter().find(|k| **k == n.as_str())
            {
                let req = fs.iter().find_map(|(f, v)| match (f.as_str(), v) {
                    ("req", V::Int(i, _)) => Some(*i as u64),
                    _ => None,
                });
                let j: Value = serde_json::from_str(&crate::interp::form_value(o, self.m).json()).unwrap_or(Value::Null);
                self.queue.push_back(Queued { kind: k.to_string(), req: req.unwrap_or(0), json: j, at });
            }
            if let V::Rec(n, fs) = o
                && declares_struct(self.m, n)
                && let Some(req) = req_of(fs)
            {
                let j: Value = serde_json::from_str(&crate::interp::form_value(o, self.m).json()).unwrap_or(Value::Null);
                self.unanswered.push_back(Queued { kind: n.clone(), req, json: j, at });
            }
        }
    }

    /// HC-37: the request (K, req) has its result.
    fn unqueue(&mut self, k: &str, r: u64) {
        if let Some(i) = self.unanswered.iter().position(|q| q.kind == k && q.req == r) {
            self.unanswered.remove(i);
        }
    }

    /// Admit the result of one request (HC-3), as the module declares
    /// `KDone` (HD-1): delivered as declared, or recorded in
    /// D as `host.Kept` when it is not delivered whole.
    fn answer(&mut self, kind: &str, req: u64, r: EffectResult) -> Result<Vec<Admitted>, String> {
        let ev = format!("{kind}Done");
        let p = result_payload(req, &r);
        let plan = self.plan(&ev, &p, false);
        self.deliver_plan(&ev, &p, plan, How::Answers(kind.to_string(), req))
    }

    /// Build the host value `p` of the event `name` into the module's
    /// declaration of it (E.4c).
    fn build(&mut self, name: &str, p: &serde_json::Map<String, Value>) -> Build {
        let Some(decl) = event_payload(self.m, name) else { return Build::Undeclared };
        let mut args = vec![];
        for (i, (k, ty)) in decl.iter().enumerate() {
            let lab = |e: Expr| Arg { label: Some(k.clone()), e };
            match p.get(k).filter(|v| !v.is_null()) {
                Some(v) => match convert(v, ty, bytes_field(name, k), self.m) {
                    Ok(e) => args.push(lab(e)),
                    Err(w) => return Build::Unbuildable(format!("{k}: {w}"), Some(k.clone())),
                },
                None if matches!(unalias(ty, self.m), Ty::Opt(_)) => args.push(lab(Expr::None(0))),
                None if p.contains_key(k) => return Build::Unbuildable(format!("{k} has no value and is not optional"), Some(k.clone())),
                None => return Build::Unbuildable(format!("field {} has no value from the host", i + 1), None),
            }
        }
        // a refined type (5.1) decides here
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit();
        if let Err(f) = self.it.construct(name, &args) {
            return Build::Unbuildable(why(f), None);
        }
        let lost = p.iter().filter(|(k, v)| !v.is_null() && !decl.iter().any(|(d, _)| d == *k)).map(|(k, _)| k.clone()).collect();
        Build::Built(args, lost)
    }

    /// The plan of one delivery (E.4c). `held`: the host entry before it
    /// (`host.StepKept`, `host.Restarted`, `host.Synced`, `host.WallTime`)
    /// holds the whole value, so a field the event does not take is not
    /// recorded again and an event the module does not declare is not
    /// recorded at all. A result whose data has no exact conversion is
    /// delivered as `failed` with the reason (HC-3), and `host.Kept` keeps
    /// the data. A result whose req is not delivered as an integer is
    /// recorded too: replay matches an unrecorded result to its request by
    /// that integer (HC-3, HC-21), and the `host.Kept` before it by the
    /// req it holds.
    fn plan(&mut self, name: &str, p: &serde_json::Map<String, Value>, held: bool) -> Plan {
        match self.build(name, p) {
            Build::Built(args, lost) => {
                let mut why: Vec<String> = vec![];
                if !lost.is_empty() && !held {
                    why.push(format!("not declared: {}", lost.join(", ")));
                }
                let plain = args.iter().any(|a| a.label.as_deref() == Some("req") && matches!(a.e, Expr::Int(r, _) if Some(r) == p.get("req").and_then(Value::as_u64).map(i128::from)));
                if is_result(name, p) && !held && !plain && !lost.iter().any(|k| k == "req") {
                    why.push("req: not delivered as an integer".to_string());
                }
                Plan { kept: (!why.is_empty()).then(|| why.join("; ")), deliver: Some((name.to_string(), args)) }
            }
            Build::Undeclared if name == "HttpRequest" => self.http_fallback(name, p, held, "not declared".to_string()),
            Build::Undeclared => Plan { kept: (!held).then(|| "not declared".to_string()), deliver: None },
            Build::Unbuildable(why, field) => {
                if is_result(name, p) && field.as_deref() == Some("data") && p.get("data").is_some_and(|d| !d.is_null()) {
                    let w = why.strip_prefix("data: ").unwrap_or(&why).to_string();
                    let mut failed = p.clone();
                    failed.insert("status".into(), Value::String("failed".into()));
                    failed.insert("data".into(), Value::Null);
                    failed.insert("err".into(), Value::String(format!("result cannot be built into the declared type: {w}")));
                    let inner = self.plan(name, &failed, held);
                    let kept = Some(match inner.kept {
                        Some(u) => format!("{why}; {u}"),
                        None => why,
                    });
                    return Plan { kept, deliver: inner.deliver };
                }
                self.http_fallback(name, p, held, why)
            }
        }
    }

    /// HD-2: an `HttpRequest` that is not delivered is delivered as
    /// `HttpMalformed` {id, status, why} when that can be built, with the
    /// status of [`http_fallback_status`]; `host.Kept` keeps the request.
    fn http_fallback(&mut self, name: &str, p: &serde_json::Map<String, Value>, held: bool, why: String) -> Plan {
        if name != "HttpRequest" {
            return Plan { kept: Some(why), deliver: None };
        }
        let mut mal = serde_json::Map::new();
        mal.insert("id".into(), p.get("id").cloned().unwrap_or(Value::Null));
        mal.insert("status".into(), Value::from(http_fallback_status(&why)));
        mal.insert("why".into(), Value::String(format!("HttpRequest: {why}")));
        let deliver = match self.build("HttpMalformed", &mal) {
            Build::Built(args, _) => Some(("HttpMalformed".to_string(), args)),
            _ => None,
        };
        let _ = held;
        Plan { kept: Some(why), deliver }
    }

    /// The entries a plan appends, as owed entries for replay.
    fn nexts(&mut self, name: &str, p: &serde_json::Map<String, Value>, plan: Plan, how: How) -> Vec<Next> {
        let mut v = vec![];
        if let Some(why) = &plan.kept {
            v.push(Next::Kept(kept_event(name, p, why)));
        }
        if let Some((n, args)) = plan.deliver {
            v.push(Next::Deliver { name: n, args, how });
        }
        v
    }

    /// Append what a plan says: `host.Kept`, then the delivery. A result
    /// that is not delivered still answers its request (HC-3, HC-21).
    fn deliver_plan(&mut self, name: &str, p: &serde_json::Map<String, Value>, plan: Plan, how: How) -> Result<Vec<Admitted>, String> {
        if let Some(why) = &plan.kept {
            self.append(kept_event(name, p, why))?;
        }
        match plan.deliver {
            Some((n, args)) => self.apply(&n, args, how),
            None => {
                if let How::Answers(k, r) = &how {
                    self.descent(&how, 0, Some((k.clone(), *r)));
                }
                Ok(vec![])
            }
        }
    }

    /// Replay of a `host.Kept` that no entry before it owed (a result, E.4c):
    /// plan the delivery again from the value it holds, require the same
    /// reason, and owe the delivery when there is one. A result that is not
    /// delivered answers its request here.
    fn fold_kept(&mut self, e: &Entry) -> Result<(), String> {
        let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
        let ev = &v["payload"]["event"];
        let (Some(name), Some(p), Some(why)) = (ev["type"].as_str(), ev["payload"].as_object(), v["payload"]["why"].as_str()) else {
            return Err(format!("host fault: seq {}: host.Kept without event and why", e.seq));
        };
        let plan = self.plan(name, p, false);
        if plan.kept.as_deref() != Some(why) {
            return Err(format!("host fault: seq {}: host.Kept says `{why}`, and the module gives {:?}", e.seq, plan.kept));
        }
        let how = match (is_result(name, p), p.get("req").and_then(Value::as_u64)) {
            (true, Some(r)) => How::Answers(name.trim_end_matches("Done").to_string(), r),
            _ => How::Delivery,
        };
        if plan.deliver.is_none()
            && let How::Answers(k, r) = &how
        {
            if let Some(i) = self.queue.iter().position(|q| q.kind == *k && q.req == *r) {
                self.queue.remove(i);
            }
            self.unqueue(k, *r);
            self.descent(&how, 0, Some((k.clone(), *r)));
        }
        let nexts = self.nexts(name, p, Plan { kept: None, deliver: plan.deliver }, how);
        self.owe.extend(nexts);
        Ok(())
    }

    /// HC-24: a line of the input stream that the host could not admit as
    /// events of the module (it is not UTF-8, does not parse as events,
    /// names no event of the module, or an event cannot be built) is held:
    /// it is delivered as `Input(data)`, data being the bytes of the line
    /// with its line end, built into the declaration like any delivery
    /// (E.4c): as `List[i64]`, or as `text` when they are UTF-8 (HD-3). When the
    /// module declares no `Input`, or the line cannot be built into it,
    /// `host.Kept` records the line in D, as text when it is UTF-8 and as
    /// its bytes otherwise. Part of the input batch; no barrier.
    pub fn line_input(&mut self, line: &[u8]) -> Result<Vec<Admitted>, String> {
        let mut p = serde_json::Map::new();
        let data = match std::str::from_utf8(line) {
            Ok(t) => Value::String(t.to_string()),
            Err(_) => Value::from(line.to_vec()),
        };
        p.insert("data".into(), data);
        let plan = self.plan("Input", &p, false);
        self.deliver_plan("Input", &p, plan, How::Typed)
    }

    /// E.4c: the lines that say, for each declaration that is not exactly
    /// what the host delivers, what the host will deliver and record. The
    /// host loop prints them at start.
    pub fn delivery_notes(&self) -> &[String] {
        &self.notes
    }

    /// Perform queued requests with their adapters and admit their results;
    /// a result's step may request more effects, which are performed in
    /// turn, up to `drain_max` requests in all (HC-24: the host loop then
    /// reads its stop and its inputs before it performs the rest).
    fn drain(&mut self) -> Result<Vec<Admitted>, String> {
        let mut out = vec![];
        let mut done = 0;
        while done < self.drain_max
            && let Some(q) = self.queue.pop_front()
        {
            done += 1;
            // HS-2: an effect is an output, so the entry that requested it
            // is made durable before the effect is performed. One barrier
            // covers every effect requested by the entries it made durable.
            if self.pending > 0 && self.synced.is_none_or(|s| s < q.at) {
                self.barrier()?;
            }
            let r = match self.adapters.iter_mut().find(|a| a.kind() == q.kind) {
                Some(a) => a.perform(&q.json),
                None => EffectResult { status: Status::Unreachable, data: None, error: "no adapter".into() },
            };
            out.extend(self.answer(&q.kind, q.req, r)?);
        }
        Ok(out)
    }

    /// Bound every later [`Host::drain`] to `n` requests (HC-24): the host
    /// loop performs the rest in its next round, after it has read a stop.
    pub fn bound_drains(&mut self, n: usize) {
        self.drain_max = n.max(1);
    }

    /// The number of requests emitted and not yet performed.
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Append one input and apply δ to it (HC-1). The first element is the
    /// input; a kept step adds the entries of HC-21 after it. A typed input
    /// whose name is one the host delivers is preceded by `host.Typed`
    /// {seq} (HC-10b), appended only once the input is known to be an
    /// event of the module.
    fn apply(&mut self, name: &str, args: Vec<Arg>, how: How) -> Result<Vec<Admitted>, String> {
        if self.stopped {
            return Err("host.StorageFault: admitter stopped after a failed barrier".into());
        }
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
        let ev: V = match self.it.construct(name, &args) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("`{name}` is not an event")),
            Err(f) => return Err(why(f)),
        };
        let ej = event_json(name, &ev, self.m);
        // δ runs on the input, but the entry is appended before any output
        // is released (HC-1); a kept step leaves the state unchanged (T8).
        let (outcome, next) = match self.it.delta_r(&self.state, &ev) {
            Ok((s2, outs)) => (Ok(outs), Some(s2)),
            Err(r) => (Err(r), None),
        };
        // HC-10b: only a typed input of a name the host delivers is marked,
        // and replay sees any other input as unmarked, so treat it so here
        let how = match how {
            How::Typed if !self.delivered.iter().any(|d| d == name) => How::Input,
            how => how,
        };
        if how == How::Typed {
            self.append(host_event("Typed", &format!("{{\"seq\":{}}}", self.seq + 1)))?;
        }
        let (old_state, old_hash) = (self.state.clone(), self.state_hash.clone());
        if let Some(s2) = next {
            self.state = s2;
            self.rehash();
        }
        let e = match self.append(ej) {
            Ok(e) => e,
            Err(x) => {
                self.state = old_state;
                self.state_hash = old_hash;
                return Err(x);
            }
        };
        self.note_tick(name, &ev);
        let ans = answered(&how, name, &ev);
        if let Some((k, r)) = &ans {
            self.unqueue(k, *r);
        }
        let under = self.descent(&how, e.seq, ans);
        let (mut display, mut exit) = (vec![], false);
        let frames = self.note_frames(e.seq, name, outcome.as_ref().ok().map(Vec::as_slice));
        if let Ok(outs) = &outcome {
            self.enqueue(outs, e.seq);
            self.note_descent(outs, under);
            for o in outs {
                match o {
                    V::Rec(n, fs) if n == "Display" => {
                        match fs.iter().find(|(f, _)| f == "data") {
                            Some((_, V::Text(t))) => display.push(t.as_bytes().to_vec()),
                            // HC-30: bytes, written exactly; a list with an
                            // element outside 0..255 is not bytes
                            Some((_, V::List(xs))) => {
                                let b: Option<Vec<u8>> = xs.iter().map(|x| match x {
                                    V::Int(n, _) if (0..=255).contains(n) => Some(*n as u8),
                                    _ => None,
                                }).collect();
                                display.extend(b);
                            }
                            _ => {}
                        }
                    }
                    V::Rec(n, _) if n == "Exit" => exit = true,
                    _ => {}
                }
            }
        }
        match &outcome {
            Ok(outs) => self.rap_capture(e.seq, name, Ok(outs)),
            Err(r) => self.rap_capture(e.seq, name, Err((r.cause, &r.text))),
        }
        let kept = outcome.as_ref().err().map(|r| r.cause);
        let outcome = outcome.map(|os| os.iter().map(|o| o.show()).collect()).map_err(|r| r.text);
        let mut out = vec![Admitted { seq: e.seq, event: ev.show(), outcome, display, exit, frames, cause: kept.map(Cause::as_str) }];
        if let Some(cause) = kept {
            let bind = under.is_none() && declares_event(self.m, "StepKept");
            out.extend(self.kept(e.seq, cause, bind)?);
        }
        Ok(out)
    }

    /// HC-21: after the entry of the kept step at `seq`, `host.StepKept` {cause, seq},
    /// then (`bind`) the delivery of `StepKept`, as the module declares it
    /// (HD-4). Delivery is bounded: a kept step that descends from a
    /// delivered `StepKept` ([`Host::descent`]) is recorded as `host.StepKept`
    /// only and not redelivered, so no `StepKept` leads to another.
    fn kept(&mut self, seq: u64, cause: Cause, bind: bool) -> Result<Vec<Admitted>, String> {
        self.append(host_event("StepKept", &kept_payload(cause, seq)))?;
        if !bind {
            return Ok(vec![]);
        }
        let p = kept_map(cause, seq);
        let plan = self.plan("StepKept", &p, true);
        self.deliver_plan("StepKept", &p, plan, How::Root)
    }

    /// The canonical event JSON of a host-delivered input, as the live path appends it.
    fn bound_json(&mut self, name: &str, args: &[Arg]) -> Result<String, String> {
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit();
        match self.it.construct(name, args) {
            Ok(Some(v)) => Ok(event_json(name, &v, self.m)),
            Ok(None) => Err(format!("`{name}` is not an event")),
            Err(f) => Err(why(f)),
        }
    }

    /// Replay (HC-10b, HC-21, HC-22): `e` must be the entry the previous
    /// entries require, if any. Returns how the input `e` reached δ when
    /// that is decided by what precedes it: a delivery or a typed input.
    fn meet(&mut self, e: &Entry) -> Result<Option<How>, String> {
        let Some(o) = self.owe.pop_front() else {
            if e.kind() == "host.StepKept" {
                return Err(format!("host fault: seq {}: host.StepKept follows an entry that replay applies", e.seq));
            }
            return Ok(None);
        };
        let (want, how) = match o {
            Next::HostKept { seq, cause, bind } => {
                if bind {
                    let p = kept_map(cause, seq);
                    let plan = self.plan("StepKept", &p, true);
                    for n in self.nexts("StepKept", &p, plan, How::Root).into_iter().rev() {
                        self.owe.push_front(n);
                    }
                }
                (host_event("StepKept", &kept_payload(cause, seq)), None)
            }
            Next::Kept(want) => (want, Some(How::Owed)),
            Next::Deliver { name, args, how } => (self.bound_json(&name, &args)?, Some(how)),
            // the host stopped after `host.Typed` and before the typed
            // input: nothing was admitted, and nothing is owed
            Next::Typed { seq } if e.is_host() || e.seq != seq => return self.meet(e),
            Next::Typed { .. } => return Ok(Some(How::Typed)),
        };
        if e.event != want {
            return Err(format!("host fault: seq {}: D has {} where {want} must follow", e.seq, e.event));
        }
        Ok(how)
    }

    /// Complete the entries D owes at its end (HC-21, HC-22): a host
    /// stopped after appending the entry of a kept step, or a `host.StepKept` or
    /// `host.Restarted` without the delivery that follows it. A typed input
    /// announced by `host.Typed` and not appended is not completed: it was
    /// never admitted (HC-10b).
    fn complete_owed(&mut self) -> Result<Vec<Admitted>, String> {
        let mut out = vec![];
        while let Some(n) = self.owe.pop_front() {
            match n {
                Next::HostKept { seq, cause, bind } => out.extend(self.kept(seq, cause, bind)?),
                Next::Kept(event) => {
                    self.append(event)?;
                }
                Next::Deliver { name, args, how } => {
                    // HC-3: the result completed here answers its request,
                    // which a restart must not interrupt again (the live
                    // path took it off the queue before performing it)
                    if let How::Answers(k, r) = &how
                        && let Some(i) = self.queue.iter().position(|q| q.kind == *k && q.req == *r)
                    {
                        self.queue.remove(i);
                    }
                    out.extend(self.apply(&name, args, how)?)
                }
                Next::Typed { .. } => {}
            }
        }
        Ok(out)
    }

    /// Admit one batch of inputs written in example syntax (`Put(key: "a",
    /// val: "1"), Scan(start: "")`), then one barrier for the batch (D-2).
    /// These are typed inputs (HC-10b): any event of the module is admitted,
    /// one the host delivers (`StepKept`, a result `KDone`, ...) included,
    /// and D marks the latter with `host.Typed`. When one of the events
    /// cannot be built, nothing is appended and the error is returned, so
    /// the caller holds the whole line ([`Host::line_input`]).
    pub fn admit(&mut self, src: &str, events: &str) -> Result<Vec<Admitted>, String> {
        let probe = parse(&format!("{src}\nexample {events} -> 0\n")).map_err(|e| format!("events: {e}"))?;
        let Some(Item::Example(x)) = probe.items.last() else {
            return Err("events: not an event list".into());
        };
        // HC-24: a line is admitted whole or held whole, so every event is
        // built before any is appended; building reads only the module
        for (name, args) in &x.events {
            self.bound_json(name, args)?;
        }
        let mut out = vec![];
        for (name, args) in x.events.clone() {
            out.extend(self.apply(&name, args, How::Typed)?);
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// Admit one input given as an event name and arguments (the host
    /// loop's `Tick` and `Input`, and embedders), its effects, then the
    /// batch barrier. It is a typed input (HC-10b).
    pub fn admit_event(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
        let mut out = self.apply(name, args, How::Typed)?;
        out.extend(self.drain()?);
        self.confirm()?;
        Ok(out)
    }

    /// Admit one input of a batch (group commit, D-2): append its entry, run
    /// δ and queue its effects, without a barrier. [`Host::settle`] ends the
    /// batch.
    pub fn apply_input(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
        self.apply(name, args, How::Typed)
    }

    /// HC-24, HD-2: admit one input the HTTP adapter framed (`HttpRequest`
    /// or `HttpMalformed`, its fields in `p` without the id) as one input of
    /// the batch, without a barrier, as the module declares it. The id is the
    /// seq of the entry that delivers it: its own entry when it is
    /// delivered exactly, else the entry after the `host.Kept` that
    /// records it first. The framing and who answers are those
    /// [`http_inputs`] reads from D.
    pub fn http_input(&mut self, name: &str, p: serde_json::Map<String, Value>) -> Result<HttpAdmit, String> {
        let mut q = p;
        q.insert("id".into(), Value::from(self.seq));
        let mut plan = self.plan(name, &q, false);
        if plan.kept.is_some() || plan.deliver.is_none() {
            q.insert("id".into(), Value::from(self.seq + 1));
            plan = self.plan(name, &q, false);
        }
        let id = q["id"].as_u64().unwrap_or(0);
        // the entry that delivers it; a delivery whose id is not that
        // entry's seq (a refined id type, 5.1) is one the program cannot
        // answer, so the host answers it and no two open requests share an id
        let at = self.seq + u64::from(plan.kept.is_some());
        let (head, close, answer) = http_terms(self.m, name, &q, &plan, at);
        let admitted = self.deliver_plan(name, &q, plan, How::Delivery)?;
        Ok(HttpAdmit { id, admitted, head, close, answer })
    }

    /// End a batch: perform the queued effects (each after a barrier that
    /// covers the entry that requested it, HS-2) and take the batch barrier.
    pub fn settle(&mut self) -> Result<Vec<Admitted>, String> {
        let out = self.drain()?;
        self.confirm()?;
        Ok(out)
    }

    /// HC-24: `host.HttpListen` {addr and the limits of HC-25, or addr and
    /// the bind error}.
    pub fn http_listen(&mut self, payload: &str) -> Result<(), String> {
        self.append(host_event("HttpListen", payload))?;
        self.confirm()
    }

    /// HC-34: `host.RapListen` {addr, endpoint, toolset}, toolset being the
    /// SHA-256 of the toolset this module publishes at that endpoint, which
    /// replay re-derives; returns the toolset's bytes.
    pub fn rap_listen(&mut self, addr: &str, endpoint: &str) -> Result<String, String> {
        let ts = crate::rap::toolset(self.m, endpoint)?;
        let p = serde_json::json!({"addr": addr, "endpoint": endpoint, "toolset": d_sha(&ts)});
        self.append(host_event("RapListen", &crate::interp::canon_json(&p)))?;
        self.confirm()?;
        Ok(ts)
    }

    /// HC-35: the event an invocation names, built for the module, or why
    /// it does not conform (the operation, the arguments, or a refined type).
    fn rap_validate(&mut self, inv: &serde_json::Map<String, Value>) -> Result<(String, Vec<Arg>), String> {
        let (name, args) = crate::rap::arguments(self.m, inv)?;
        self.bound_json(&name, &args).map_err(|w| format!("{name}: {w}"))?;
        Ok((name, args))
    }

    /// HC-35: append one invocation, the host's whole value from
    /// `rap::envelope`, as part of a batch (no barrier): `host.RapInvoke`
    /// with that value, or, when D already holds an invocation of the same
    /// (group_id, id), `host.RapDuplicate` {call_id, callback_url, group_id,
    /// id, of}, which is answered from the log and never stepped. The
    /// caller acknowledges it once [`Host::rap_commit`] made it durable.
    pub fn rap_invoke(&mut self, inv: serde_json::Map<String, Value>) -> Result<RapAdmit, String> {
        let key = rap_key(&inv).ok_or("rap: an invocation without a string group_id and id")?;
        let seq = self.seq;
        if let Some(&of) = self.rap.keys.get(&key) {
            let mut d = serde_json::Map::new();
            for k in ["call_id", "callback_url", "group_id", "id"] {
                d.insert(k.into(), inv.get(k).cloned().unwrap_or(Value::Null));
            }
            let mut p = d.clone();
            p.insert("of".into(), Value::from(of));
            self.append(host_event("RapDuplicate", &crate::interp::canon_json(&Value::Object(p))))?;
            self.rap.add(seq, RapCall { inv: d, of: Some(of), ..Default::default() });
            return Ok(RapAdmit::Duplicate { seq, of });
        }
        let op = self.rap_validate(&inv);
        self.append(host_event("RapInvoke", &crate::interp::canon_json(&Value::Object(inv.clone()))))?;
        self.rap.keys.insert(key, seq);
        self.rap.add(seq, RapCall { inv, op: Some(op), ..Default::default() });
        Ok(RapAdmit::Invoked(seq))
    }

    /// HC-35: the barrier that makes the invocations of the batch durable,
    /// after which each is acknowledged with 200 (HS-2).
    pub fn rap_commit(&mut self) -> Result<(), String> {
        self.confirm()
    }

    /// HC-35, HC-36: step every invocation of D that conforms and has not been
    /// stepped, in D order (`host.RapStep` {call}, then its event, as the
    /// host delivers it), then append `host.RapResult` {call, length,
    /// sha256} for every invocation and duplicate whose result D does not
    /// hold yet. Returns the steps and the results to deliver, which the
    /// caller POSTs once the batch is durable (HS-2). Called after the
    /// acknowledgements of a batch, and on open for what a stopped host left.
    pub fn rap_steps(&mut self) -> Result<(Vec<Admitted>, Vec<RapOut>), String> {
        let mut adm = vec![];
        let todo: Vec<(u64, String, Vec<Arg>)> = self
            .rap
            .open
            .iter()
            .filter_map(|c| match self.rap.calls.get(c) {
                Some(RapCall { op: Some(Ok((n, a))), stepped: false, .. }) => Some((*c, n.clone(), a.clone())),
                _ => None,
            })
            .collect();
        for (c, name, args) in todo {
            self.append(host_event("RapStep", &format!("{{\"call\":{c}}}")))?;
            if let Some(x) = self.rap.calls.get_mut(&c) {
                x.stepped = true;
            }
            self.rap.stepping = Some(c);
            let r = self.apply(&name, args, How::Delivery);
            self.rap.stepping = None;
            adm.extend(r?);
        }
        let mut outs = vec![];
        let need: Vec<u64> = self.rap.open.iter().copied().collect();
        for c in need {
            let Some(body) = self.rap_derive(c) else { continue };
            self.append(host_event("RapResult", &format!("{{\"call\":{c},\"length\":{},\"sha256\":\"{}\"}}", body.len(), d_sha(&body))))?;
            self.rap.answered(c, body);
            outs.extend(self.rap_out(c));
        }
        Ok((adm, outs))
    }

    /// HC-36: the delivery of the result of `call` settled with `status`
    /// (2xx delivered, 4xx and others rejected): `host.RapSent` {call,
    /// status}. A call with no result, or one already settled, appends
    /// nothing and gives false. Part of a batch (no barrier).
    pub fn rap_sent(&mut self, call: u64, status: u16) -> Result<bool, String> {
        match self.rap.calls.get(&call) {
            Some(x) if x.body.is_some() && x.sent.is_none() => {}
            _ => return Ok(false),
        }
        self.append(host_event("RapSent", &format!("{{\"call\":{call},\"status\":{status}}}")))?;
        self.rap.settled(call, status);
        Ok(true)
    }

    /// HC-36: the results D holds whose delivery has not settled: a host
    /// opened on D sends each one again (at least once; the runtime keeps
    /// one result per id).
    pub fn rap_unsent(&self) -> Vec<RapOut> {
        self.rap.unsent.iter().filter_map(|c| self.rap_out(*c)).collect()
    }

    fn rap_out(&self, c: u64) -> Option<RapOut> {
        let x = self.rap.calls.get(&c)?;
        Some(RapOut { call: c, callback_url: x.inv.get("callback_url")?.as_str()?.to_string(), body: x.body.clone()?, sent: x.sent })
    }

    /// HC-36: the bytes of the result of `call`, a function of D: the
    /// outcome of its invocation's step (of the first invocation, for a
    /// duplicate), or why it does not conform; `None` while it is not stepped.
    fn rap_derive(&self, c: u64) -> Option<String> {
        let x = self.rap.calls.get(&c)?;
        let base = self.rap.calls.get(&x.of.unwrap_or(c))?;
        let o = match &base.op {
            Some(Err(w)) => crate::rap::Outcome::Invalid(w.clone()),
            Some(Ok(_)) => base.outcome.clone()?,
            None => return None,
        };
        Some(crate::rap::result_body(&x.inv, &o))
    }

    /// HC-36: the outcome of the step of the invocation `host.RapStep` named
    /// last, when the entry at seq is that step.
    fn rap_capture(&mut self, seq: u64, name: &str, r: Result<&[V], (Cause, &str)>) {
        let Some(c) = self.rap.stepping.take() else { return };
        let o = match r {
            // each output as D writes an event: {"payload":..,"type":..}
            Ok(outs) => crate::rap::Outcome::Applied {
                event: name.to_string(),
                seq,
                outputs: outs
                    .iter()
                    .map(|x| match x {
                        V::Rec(n, _) => event_json(n, x, self.m),
                        _ => crate::interp::form_value(x, self.m).json(),
                    })
                    .collect(),
                state_hash: self.state_hash.clone(),
            },
            Err((cause, text)) => crate::rap::Outcome::Kept { event: name.to_string(), seq, cause: cause.as_str().to_string(), text: text.to_string() },
        };
        // a kept step leaves the state unchanged, so this is the state it was kept on
        let ks = matches!(o, crate::rap::Outcome::Kept { .. }).then(|| crate::interp::state_json(&self.state, self.m));
        if let Some(x) = self.rap.calls.get_mut(&c) {
            x.outcome = Some(o);
            x.kept_state = ks;
        }
    }

    /// Replay of the RAP entries (HC-34 to HC-36): each is checked against
    /// what the entries before it and the module give, and a difference is
    /// a host fault.
    fn rap_fold(&mut self, e: &Entry) -> Result<(), String> {
        let v: Value = serde_json::from_str(&e.event).map_err(|x| x.to_string())?;
        let p = v["payload"].as_object().cloned().unwrap_or_default();
        let call = p.get("call").and_then(Value::as_u64);
        match e.kind().as_str() {
            "host.RapListen" => {
                let ep = p.get("endpoint").and_then(Value::as_str).unwrap_or("");
                let want = crate::rap::toolset_digest(self.m, ep)?;
                if p.get("toolset").and_then(Value::as_str) != Some(want.as_str()) {
                    return Err(format!("host.RapListen names a toolset other than the one this module publishes at {ep} ({want})"));
                }
            }
            "host.RapInvoke" => {
                let key = rap_key(&p).ok_or("host.RapInvoke without a string group_id and id")?;
                if !p.get("callback_url").is_some_and(Value::is_string) {
                    return Err("host.RapInvoke without a callback_url".into());
                }
                if self.rap.keys.contains_key(&key) {
                    return Err(format!("host.RapInvoke repeats ({}, {}), which only host.RapDuplicate may", key.0, key.1));
                }
                let op = self.rap_validate(&p);
                self.rap.keys.insert(key, e.seq);
                self.rap.add(e.seq, RapCall { inv: p, op: Some(op), ..Default::default() });
            }
            "host.RapDuplicate" => {
                let key = rap_key(&p).ok_or("host.RapDuplicate without a string group_id and id")?;
                let of = p.get("of").and_then(Value::as_u64);
                if of.is_none() || self.rap.keys.get(&key).copied() != of {
                    return Err(format!("host.RapDuplicate of {of:?} names no earlier host.RapInvoke of ({}, {})", key.0, key.1));
                }
                let mut d = p.clone();
                d.remove("of");
                self.rap.add(e.seq, RapCall { inv: d, of, ..Default::default() });
            }
            "host.RapStep" => {
                let c = call.ok_or("host.RapStep without a call")?;
                let (name, args) = match self.rap.calls.get(&c) {
                    Some(RapCall { op: Some(Ok((n, a))), stepped: false, .. }) => (n.clone(), a.clone()),
                    _ => return Err(format!("host.RapStep names {c}, which is no invocation that conforms and is not stepped")),
                };
                if let Some(x) = self.rap.calls.get_mut(&c) {
                    x.stepped = true;
                }
                self.rap.stepping = Some(c);
                self.owe.push_back(Next::Deliver { name, args, how: How::Delivery });
            }
            "host.RapResult" => {
                let c = call.ok_or("host.RapResult without a call")?;
                if self.rap.calls.get(&c).is_none_or(|x| x.body.is_some()) {
                    return Err(format!("host.RapResult names {c}, which is no invocation without a result"));
                }
                let body = self.rap_derive(c).ok_or_else(|| format!("host.RapResult for {c} before its step"))?;
                let sha = d_sha(&body);
                if p.get("sha256").and_then(Value::as_str) != Some(sha.as_str()) || p.get("length").and_then(Value::as_u64) != Some(body.len() as u64) {
                    return Err(format!("host.RapResult for {c} differs from the result replay derives ({} bytes, {sha})", body.len()));
                }
                self.rap.answered(c, body);
            }
            "host.RapSent" => {
                let c = call.ok_or("host.RapSent without a call")?;
                let st = p.get("status").and_then(Value::as_u64).and_then(|s| u16::try_from(s).ok()).ok_or("host.RapSent without a status")?;
                match self.rap.calls.get(&c) {
                    Some(x) if x.body.is_some() && x.sent.is_none() => self.rap.settled(c, st),
                    _ => return Err(format!("host.RapSent names {c}, which has no result, or one already settled")),
                }
            }
            k => return Err(format!("{k} is not a RAP entry of HC-34 to HC-36")),
        }
        Ok(())
    }

    /// Whether the store reported its last barrier failed (HS-3).
    pub fn is_storage_fault(e: &str) -> bool {
        e.starts_with("host.StorageFault")
    }

    /// The priority of an input (HC-9, T25): `interrupt event E priority p`
    /// gives p (an integer literal in 0..=255); every other event has 0.
    pub fn priority(&self, name: &str) -> u8 {
        self.m.items.iter().find_map(|i| match i {
            Item::Event { name: n, interrupt: Some(Expr::Int(p, _)), .. } if n == name => Some((*p).clamp(0, 255) as u8),
            _ => None,
        }).unwrap_or(0)
    }

    /// Admit inputs that arrived simultaneously (HC-9), with no source
    /// identity: the admitter orders them by priority, highest first, then by
    /// the bytes of their canonical event JSON, so the order is a function of
    /// the set of inputs and not of the order the caller passed them in. The
    /// order is the order of their entries in D. One barrier (D-2). Inputs
    /// that carry a source and sequence number use [`Host::close_window`].
    pub fn admit_simultaneous(&mut self, inputs: Vec<(String, Vec<Arg>)>) -> Result<Vec<Admitted>, String> {
        let mut ord: Vec<(u8, String, String, Vec<Arg>)> =
            inputs.into_iter().map(|(n, a)| (self.priority(&n), self.event_key(&n, &a), n, a)).collect();
        ord.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut out = vec![];
        for (_, _, name, args) in ord {
            out.extend(self.apply(&name, args, How::Typed)?);
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// The canonical event JSON of an input, or `!` and the reason it is not one.
    fn event_key(&mut self, name: &str, args: &[Arg]) -> String {
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7a
        match self.it.construct(name, args) {
            Ok(Some(v)) => event_json(name, &v, self.m),
            Ok(None) => format!("!`{name}` is not an event"),
            Err(f) => format!("!{}", why(f)),
        }
    }

    /// E.2a / E.2b: fold one host entry into the admission state.
    fn fold_admission(&mut self, e: &Entry, announced: &mut Option<(String, u64)>) {
        let k = e.kind();
        if k != "host.Window" && k != "host.Input" && k != "host.Term" {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(&e.event) else { return };
        let p = &v["payload"];
        match k.as_str() {
            "host.Window" => {
                self.window = p["id"].as_u64().unwrap_or(0) + 1;
                self.closed_tick = p["upto"].as_u64().unwrap_or(0);
            }
            "host.Input" => {
                if let (Some(s), Some(q)) = (p["source"].as_str(), p["seq"].as_u64()) {
                    *announced = Some((s.to_string(), q));
                }
            }
            _ => self.term = p["term"].as_u64().unwrap_or(0),
        }
    }

    /// Offer one input to the admitter (E.2a). Nothing is appended until the
    /// window holding it closes; an offer is never acked by itself.
    ///
    /// HC-15: when the module declares `event Tick(n: u64)`, an offered
    /// `Tick(n)` not yet in D closes the window at `upto = n`, and is itself
    /// the last input of that window. So the window boundary is the logged
    /// tick, and an input stamped with a tick at or below the last closed
    /// `upto` (a fact of D) is late: it goes into the next window and its
    /// `host.Input` says `"late":true`.
    pub fn offer(&mut self, mut x: Sourced) -> Result<Option<WindowAck>, String> {
        let n = self.tick_of(&x);
        if let Some(n) = n {
            x.tick = n;
        }
        let fresh = x.seq > self.admitted_upto(&x.source);
        self.offered.push(x);
        match n {
            Some(n) if fresh && (n > self.closed_tick || self.window == 0) => self.close_window(n).map(Some),
            _ => Ok(None),
        }
    }

    /// The n of an offered `Tick(n)`, when the module declares `Tick`.
    fn tick_of(&self, x: &Sourced) -> Option<u64> {
        if x.name != "Tick" || !declares_event(self.m, "Tick") {
            return None;
        }
        x.args.iter().find_map(|a| match (&a.label, &a.e) {
            (Some(l), Expr::Int(n, _)) if l == "n" && *n >= 0 => Some(*n as u64),
            _ => None,
        })
    }

    /// HC-6 / HC-12: fold one host entry of a `fn main` run or of placement,
    /// and check what replay re-derives against what D says.
    fn fold_sched(&mut self, e: &Entry) -> Result<(), String> {
        let k = e.kind();
        let v: Value = match k.as_str() {
            "host.MainStart" | "host.MainInput" | "host.Step" | "host.MainEnd" | "host.Calibrated" | "host.Exact" | "host.Placed" | "host.Energy" => serde_json::from_str(&e.event).map_err(|x| x.to_string())?,
            _ => return Ok(()),
        };
        let p = v["payload"].clone();
        match k.as_str() {
            // HC-38: a reading is checked where a record is read (D.11)
            "host.Energy" => energy_check(&p, e.seq)?,
            "host.MainStart" | "host.MainInput" | "host.Step" | "host.MainEnd" => {
                self.apply_main(k.as_str(), &p)?;
            }
            "host.Calibrated" => {
                let key = (p["fabric"].as_str().unwrap_or("").to_string(), p["class"].as_str().unwrap_or("").to_string());
                self.calib.insert(key, (p["overhead"].as_f64().unwrap_or(f64::INFINITY), p["throughput"].as_f64().unwrap_or(0.0)));
            }
            "host.Exact" => {
                if p.get("probe").is_some() && p["probe"] == p["reference"] {
                    self.exacts.insert((p["fabric"].as_str().unwrap_or("").to_string(), p["class"].as_str().unwrap_or("").to_string()));
                }
            }
            _ => {
                let got = place_from(&self.calib, &self.exacts, p["class"].as_str().unwrap_or(""), p["work"].as_f64().unwrap_or(0.0));
                if p["fabric"].as_str() != Some(got.as_str()) {
                    return Err(format!("placement re-derives {got}, D has {}", p["fabric"]));
                }
            }
        }
        Ok(())
    }

    fn exec_main(&self, start: &Value, choices: Option<Vec<usize>>, events: Vec<(u64, V)>) -> Result<crate::conc::RunResult, String> {
        let links = start["links"].as_array().cloned().unwrap_or_default().iter().filter_map(|l| {
            let a = l.as_array()?;
            Some((a.first()?.as_u64()?, a.get(1)?.as_bool()?, a.get(2)?.as_str()?.to_string(), a.get(3)?.as_str()?.to_string()))
        }).collect();
        let cfg = crate::conc::RunCfg { seed: start["seed"].as_u64().unwrap_or(0), choices, links, max_ticks: start["max_ticks"].as_u64().unwrap_or(100_000), par_mode: crate::conc::ParMode::Seq, restarts: None, restart_tree: None, events, tasks: crate::conc::Tasks::default() };
        crate::conc::run_main(self.m, &cfg)
    }

    fn step_payloads(r: &crate::conc::RunResult) -> Vec<Value> {
        r.log.iter().enumerate().map(|(i, e)| {
            serde_json::json!({"cause": e.cause, "i": i, "node": r.nodes.get(e.node).cloned().unwrap_or_default(), "tick": e.tick, "what": e.what})
        }).collect()
    }

    fn outcome_text(o: &crate::conc::Outcome) -> String {
        match o {
            crate::conc::Outcome::Done => "done".into(),
            crate::conc::Outcome::Stopped(r) => format!("stopped: {r}"),
            crate::conc::Outcome::Deadlock => "deadlock".into(),
        }
    }

    /// The state hash of δ's state and, while a `fn main` task is live,
    /// the task's state: the state after each entry is the fold of D.
    fn rehash(&mut self) {
        let sj = state_pos_json(&self.state, self.m);
        self.state_hash = match &self.task {
            None => d_sha(&sj),
            Some(t) => d_sha(&format!("{sj}\n{}", t.json())),
        };
    }

    fn task_run(&mut self) -> Result<&crate::conc::RunResult, String> {
        let t = self.task.as_ref().ok_or("no fn main task is live")?;
        if t.run.is_none() {
            let mut start = t.start.clone();
            let mut links: Vec<Value> = start["links"].as_array().cloned().unwrap_or_default();
            links.extend(t.inputs.iter().map(|(a, b, c, d)| serde_json::json!([a, b, c, d])));
            start["links"] = Value::Array(links);
            let evj = t.events.clone();
            let mut evs = vec![];
            for (tick, ej) in evj {
                let (name, args) = decode_event(&ej, self.m)?;
                self.it.scopes = vec![HashMap::new()];
                self.it.fuel = crate::interp::fuel_limit();
                match self.it.construct(&name, &args) {
                    Ok(Some(v)) => evs.push((tick, v)),
                    Ok(None) => return Err(format!("`{name}` is not an event")),
                    Err(f) => return Err(why(f)),
                }
            }
            let r = self.exec_main(&start, None, evs)?;
            self.task.as_mut().unwrap().run = Some(r);
        }
        Ok(self.task.as_ref().unwrap().run.as_ref().unwrap())
    }

    /// HC-6: the transition of the live task on one host entry, shared by
    /// the live path (before the entry is appended) and by replay. Anything
    /// the scheduler re-derives that differs from D is reported, and the entry
    /// is not applied.
    fn apply_main(&mut self, kind: &str, p: &Value) -> Result<(), String> {
        match kind {
            "host.MainStart" => {
                if self.task.is_some() {
                    return Err("host.MainStart while a fn main task is live".into());
                }
                self.task = Some(MainTask { start: p.clone(), inputs: vec![], events: vec![], steps: 0, digest: String::new(), last_tick: 0, run: None });
            }
            "host.MainInput" => {
                let t = self.task.as_ref().ok_or("host.MainInput outside a run")?;
                let tick = p["tick"].as_u64().ok_or("tick")?;
                if t.steps > 0 && tick <= t.last_tick {
                    return Err(format!("late input: tick {tick} is not after the last logged step (tick {})", t.last_tick));
                }
                let (old_steps, old_digest) = (t.steps, t.digest.clone());
                let is_event = p.get("event").is_some();
                if is_event {
                    let ej = p["event"].as_str().ok_or("event")?.to_string();
                    let t = self.task.as_mut().unwrap();
                    t.events.push((tick, ej));
                } else {
                    let inp = (tick, p["partition"].as_bool().ok_or("partition")?, p["a"].as_str().ok_or("a")?.to_string(), p["b"].as_str().ok_or("b")?.to_string());
                    let t = self.task.as_mut().unwrap();
                    t.inputs.push(inp);
                }
                let t = self.task.as_mut().unwrap();
                t.run = None;
                // Causality: the steps already in D do not change.
                let want = Self::step_payloads(self.task_run()?);
                let mut d = String::new();
                for s in want.iter().take(old_steps) {
                    d = d_sha(&format!("{d}{s}"));
                }
                if want.len() < old_steps || d != old_digest {
                    let t = self.task.as_mut().unwrap();
                    if is_event {
                        t.events.pop();
                    } else {
                        t.inputs.pop();
                    }
                    t.run = None;
                    return Err("input would change steps already in D".into());
                }
            }
            "host.Step" => {
                let k = self.task.as_ref().ok_or("host.Step outside a run")?.steps;
                let want = Self::step_payloads(self.task_run()?);
                if want.get(k) != Some(p) {
                    return Err(format!("fn main step {k} differs from the scheduler's"));
                }
                let t = self.task.as_mut().unwrap();
                t.digest = d_sha(&format!("{}{p}", t.digest));
                t.steps += 1;
                t.last_tick = p["tick"].as_u64().unwrap_or(0);
            }
            _ => {
                let k = self.task.as_ref().ok_or("host.MainEnd outside a run")?.steps;
                let r = self.task_run()?;
                let choices: Vec<Value> = r.choices.iter().map(|c| serde_json::json!(c)).collect();
                if r.log.len() != k || Self::outcome_text(&r.outcome) != p["outcome"].as_str().unwrap_or("") || p["choices"].as_array() != Some(&choices) {
                    return Err("fn main replay differs from the logged steps".into());
                }
                self.task = None;
            }
        }
        Ok(())
    }

    fn main_entry(&mut self, kind: &str, p: Value) -> Result<(), String> {
        if self.stopped {
            return Err("host.StorageFault: admitter stopped after a failed barrier".into());
        }
        self.apply_main(&format!("host.{kind}"), &p)?;
        self.rehash();
        self.append(host_event(kind, &p.to_string()))?;
        self.confirm()
    }

    /// HC-6: start `fn main` as a live task: log `host.MainStart` {links,
    /// max_ticks, seed}. From here the task advances one logged step at a
    /// time ([`Host::main_step`]), and inputs to it ([`Host::main_input`])
    /// and to δ interleave with its steps in D.
    pub fn main_begin(&mut self, seed: u64, links: Vec<(u64, bool, String, String)>, max_ticks: u64) -> Result<(), String> {
        let lj: Vec<Value> = links.iter().map(|(t, p, a, b)| serde_json::json!([t, p, a, b])).collect();
        self.main_entry("MainStart", serde_json::json!({"links": lj, "max_ticks": max_ticks, "seed": seed}))
    }

    /// HC-6: log a link change for the live task (`host.MainInput`). It is
    /// not logged unless its tick is after the last logged step, so the input
    /// cannot rewrite steps D already holds.
    pub fn main_input(&mut self, tick: u64, partition: bool, a: &str, b: &str) -> Result<(), String> {
        self.main_entry("MainInput", serde_json::json!({"a": a, "b": b, "partition": partition, "tick": tick}))
    }

    /// HC-6: feed a program event to the live task (`host.MainInput`
    /// {event, tick}): at its tick the scheduler logs it as `Input(..)` and
    /// delivers it to every actor that handles it (`conc::RunCfg.events`).
    /// StepKept like a link input when late or when it would change a step
    /// already in D.
    pub fn main_event(&mut self, tick: u64, name: &str, args: Vec<Arg>) -> Result<(), String> {
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit();
        let ev = match self.it.construct(name, &args) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("`{name}` is not an event")),
            Err(f) => return Err(why(f)),
        };
        let ej = event_json(name, &ev, self.m);
        self.main_entry("MainInput", serde_json::json!({"event": ej, "tick": tick}))
    }

    /// HC-6: log the task's next scheduler step (`host.Step`); `None` when
    /// the scheduler has no further step on the inputs logged so far.
    pub fn main_step(&mut self) -> Result<Option<Value>, String> {
        let k = self.task.as_ref().ok_or("no fn main task is live")?.steps;
        let Some(p) = Self::step_payloads(self.task_run()?).get(k).cloned() else { return Ok(None) };
        self.main_entry("Step", p.clone())?;
        Ok(Some(p))
    }

    /// HC-6: end the live task after its last step: log `host.MainEnd`
    /// {choices, outcome, ticks}.
    pub fn main_end(&mut self) -> Result<MainRun, String> {
        while self.main_step()?.is_some() {}
        let r = self.task_run()?;
        let outputs = r.outputs.iter().map(|o| o.show()).collect();
        let outcome = Self::outcome_text(&r.outcome);
        let (choices, steps, ticks) = (r.choices.clone(), r.log.len(), r.ticks);
        self.main_entry("MainEnd", serde_json::json!({"choices": choices, "outcome": outcome, "ticks": ticks}))?;
        Ok(MainRun { outputs, outcome, steps, choices })
    }

    /// HC-6: `fn main` with all its inputs known up front: begin, every
    /// step, end.
    pub fn run_main(&mut self, seed: u64, links: Vec<(u64, bool, String, String)>, max_ticks: u64) -> Result<MainRun, String> {
        self.main_begin(seed, links, max_ticks)?;
        self.main_end()
    }

    /// The live task's step count, if a task is live.
    pub fn main_steps(&self) -> Option<usize> {
        self.task.as_ref().map(|t| t.steps)
    }

    /// HC-12: log a calibration row (`host.Calibrated`).
    pub fn calibrate(&mut self, fabric: &str, class: &str, overhead: f64, throughput: f64) -> Result<(), String> {
        let p = serde_json::json!({"class": class, "fabric": fabric, "overhead": overhead, "throughput": throughput});
        self.append(host_event("Calibrated", &p.to_string()))?;
        self.calib.insert((fabric.into(), class.into()), (overhead, throughput));
        self.confirm()
    }

    /// HC-13: probe a fabric on `class` against the CPU reference and log
    /// `host.Exact` with both digests; the fabric is admitted for the class
    /// only when they are equal.
    pub fn probe(&mut self, fabric: &str, class: &str, probe_digest: &str, reference_digest: &str) -> Result<bool, String> {
        let p = serde_json::json!({"class": class, "fabric": fabric, "probe": probe_digest, "reference": reference_digest});
        self.append(host_event("Exact", &p.to_string()))?;
        let ok = probe_digest == reference_digest;
        if ok {
            self.exacts.insert((fabric.into(), class.into()));
        }
        self.confirm()?;
        Ok(ok)
    }

    /// HC-12: register a platform fabric executor (see [`FabricExec`]).
    /// Registration changes nothing in D; the fabric becomes a placement
    /// candidate only through `host.Calibrated` and `host.Exact` entries.
    pub fn register_fabric(&mut self, f: Box<dyn FabricExec + 'a>) {
        self.fabrics.retain(|g| g.id() != f.id());
        self.fabrics.push(f);
    }

    fn fabric(&self, id: &str) -> Option<&(dyn FabricExec + 'a)> {
        self.fabrics.iter().find(|f| f.id() == id).map(|b| b.as_ref())
    }

    /// HC-13: probe a fabric on a class against the CPU reference using
    /// [`probe_vector`], and log `host.Exact` with both digests. `class`
    /// is `sum.f64` or a lane class `lanes.f64.<op>`. `cpu.threads` is
    /// probed like any other fabric.
    pub fn probe_fabric(&mut self, fabric: &str, class: &str) -> Result<bool, String> {
        let (a, b) = probe_vector();
        let (reference, got): (Vec<f64>, Result<Vec<f64>, String>) = if class == "sum.f64" {
            // Sums of every length 0..=64 and of the whole finite vector,
            // so each tree shape up to 64 and cancellation are covered;
            // the special values are checked by a separate sum each.
            let fin: Vec<f64> = a.iter().copied().filter(|x| x.is_finite()).collect();
            let mut cases: Vec<Vec<f64>> = (0..=64).map(|k| fin[..k].to_vec()).collect();
            cases.push(fin.clone());
            cases.extend(a.iter().filter(|x| !x.is_finite() || **x == 0.0).map(|x| vec![1.0, *x, -1.0]));
            let r: Vec<f64> = cases.iter().map(|c| fixed_tree_sum(c)).collect();
            let g: Result<Vec<f64>, String> = match fabric {
                "cpu.threads" => Ok(cases.iter().map(|c| fixed_tree_sum_threads(c, 3)).collect()),
                f => {
                    let fx = self.fabric(f).ok_or(format!("no executor for fabric `{f}`"))?;
                    cases.iter().map(|c| fx.sum(c)).collect()
                }
            };
            (r, g)
        } else {
            let op = lane_op(class).ok_or(format!("unknown class `{class}`"))?;
            let r: Vec<f64> = (0..a.len()).map(|i| op.eval(a[i], b[i])).collect();
            let g = match fabric {
                "cpu.threads" => Ok(lanes_threads(a.len(), CPU_THREADS, &|i| op.eval(a[i], b[i]))),
                f => self.fabric(f).ok_or(format!("no executor for fabric `{f}`"))?.lanes(op, &a, &b),
            };
            (r, g)
        };
        let probe = match got {
            Ok(g) if g.len() == reference.len() => bits_digest(&g),
            Ok(_) => "wrong-length".into(),
            Err(e) => format!("error: {e}"),
        };
        self.probe(fabric, class, &probe, &bits_digest(&reference))
    }

    fn placed(&mut self, class: &str, op: &str, work: f64, fabric: &str) -> Result<(), String> {
        let p = serde_json::json!({"class": class, "fabric": fabric, "op": op, "work": work});
        self.append(host_event("Placed", &p.to_string()))?;
        self.confirm()
    }

    fn can_run(&self, fabric: &str) -> bool {
        fabric == "cpu" || fabric == "cpu.threads" || self.fabric(fabric).is_some()
    }

    /// HC-12 placement executor for the reduction class `sum.f64`: choose
    /// the fabric from D's host entries, log `host.Placed`, and run the
    /// fixed tree of T18 on it. A placement on a fabric this host cannot
    /// execute is not performed, and nothing is logged. A registered fabric
    /// that errors at run time is replaced by the CPU reference, which by
    /// HC-13 gives the same bits.
    pub fn place_sum(&mut self, op: &str, xs: &[f64]) -> Result<(String, f64), String> {
        let class = "sum.f64";
        let work = xs.len() as f64;
        let fabric = place_from(&self.calib, &self.exacts, class, work);
        if !self.can_run(&fabric) {
            return Err(format!("no executor for fabric `{fabric}`"));
        }
        self.placed(class, op, work, &fabric)?;
        let r = match fabric.as_str() {
            "cpu" => fixed_tree_sum(xs),
            "cpu.threads" => fixed_tree_sum_threads(xs, 3),
            f => self.fabric(f).unwrap().sum(xs).unwrap_or_else(|_| fixed_tree_sum(xs)),
        };
        Ok((fabric, r))
    }

    /// HC-12 placement executor for a lane-wise array operation
    /// `out[i] = op(a[i], b[i])` (unary ops ignore `b`, which may be empty).
    pub fn place_lanes(&mut self, op_id: &str, op: LaneOp, a: &[f64], b: &[f64]) -> Result<(String, Vec<f64>), String> {
        let unary = matches!(op, LaneOp::Neg | LaneOp::Sqrt);
        if !unary && a.len() != b.len() {
            return Err("lane-wise operands differ in length".into());
        }
        let class = op.class();
        let work = a.len() as f64;
        let fabric = place_from(&self.calib, &self.exacts, &class, work);
        if !self.can_run(&fabric) {
            return Err(format!("no executor for fabric `{fabric}`"));
        }
        self.placed(&class, op_id, work, &fabric)?;
        let bb = |i: usize| if unary { 0.0 } else { b[i] };
        let cpu = || (0..a.len()).map(|i| op.eval(a[i], bb(i))).collect::<Vec<f64>>();
        let r = match fabric.as_str() {
            "cpu" => cpu(),
            "cpu.threads" => lanes_threads(a.len(), CPU_THREADS, &|i| op.eval(a[i], bb(i))),
            f => {
                let bv: Vec<f64> = if unary { vec![0.0; a.len()] } else { b.to_vec() };
                match self.fabric(f).unwrap().lanes(op, a, &bv) {
                    Ok(v) if v.len() == a.len() => v,
                    _ => cpu(),
                }
            }
        };
        Ok((fabric, r))
    }

    /// HC-12 placement executor for a `for … par` body over `0..n` whose
    /// results are combined by `combine` (identity `zero`): the bodies run
    /// lane-wise (they share no mutable state, 5.13), and the combination
    /// is the fixed tree, so the result is the same on `cpu` and
    /// `cpu.threads` for any combine, including non-associative ones. The
    /// class is `par.<class>`. Bodies are host closures, so only CPU
    /// fabrics can execute them; a placement elsewhere is not performed and
    /// not logged.
    pub fn place_par<T: Clone + Send + Sync>(
        &mut self,
        op_id: &str,
        class: &str,
        n: usize,
        body: &(impl Fn(usize) -> T + Sync),
        zero: &T,
        combine: &(impl Fn(&T, &T) -> T + Sync),
    ) -> Result<(String, T), String> {
        let class = format!("par.{class}");
        let work = n as f64;
        let fabric = place_from(&self.calib, &self.exacts, &class, work);
        if fabric != "cpu" && fabric != "cpu.threads" {
            return Err(format!("no executor for fabric `{fabric}` on a `for … par` body"));
        }
        self.placed(&class, op_id, work, &fabric)?;
        let r = if fabric == "cpu" {
            let xs: Vec<T> = (0..n).map(body).collect();
            fixed_tree(&xs, zero, combine)
        } else {
            let xs = lanes_threads(n, CPU_THREADS, body);
            fixed_tree_threads(&xs, zero, combine, 3)
        };
        Ok((fabric, r))
    }

    /// The `upto` of the last closed window (0 before any), as folded from D.
    pub fn closed_tick(&self) -> u64 {
        self.closed_tick
    }

    /// The highest seq of `source` that is in D (0 when none).
    pub fn admitted_upto(&self, source: &str) -> u64 {
        self.admitted.get(source).copied().unwrap_or(0)
    }

    /// The id the next window will get.
    pub fn next_window(&self) -> u64 {
        self.window
    }

    /// Close the admission window at logical tick `upto` (E.2a). The window
    /// holds every offered input stamped with a tick at or below `upto`.
    /// Inside it the order is a total key on the inputs, not on their
    /// arrival: priority (highest first), then source id, then source seq.
    /// A (source, seq) already in D, or offered twice, is a duplicate; an
    /// input after a gap in its source's sequence is held for a later
    /// window, so each source is admitted in its own order with no loss.
    /// The admitter appends `host.Window`, then `host.Input` before each
    /// admitted input, and acks the window only after the barrier.
    pub fn close_window(&mut self, upto: u64) -> Result<WindowAck, String> {
        if self.stopped {
            return Err("host.StorageFault: admitter stopped after a failed barrier".into());
        }
        let upto = upto.max(self.closed_tick);
        let (take, keep): (Vec<Sourced>, Vec<Sourced>) = std::mem::take(&mut self.offered).into_iter().partition(|x| x.tick <= upto);
        self.offered = keep;
        let mut ack = WindowAck { window: self.window, ..Default::default() };
        // Canonical candidate order: (source, seq, event bytes), so which of
        // two conflicting copies of one (source, seq) wins is not arrival.
        let mut cand: Vec<(String, Sourced)> = vec![];
        for x in take {
            let key = self.event_key(&x.name, &x.args);
            if let Some(r) = key.strip_prefix('!') {
                ack.rejected.push((x.source.clone(), x.seq, r.to_string()));
            } else {
                cand.push((key, x));
            }
        }
        cand.sort_by(|a, b| (&a.1.source, a.1.seq, &a.0).cmp(&(&b.1.source, b.1.seq, &b.0)));
        let mut next: std::collections::BTreeMap<String, u64> = Default::default();
        let mut chosen: Vec<(u8, Sourced)> = vec![];
        for (_, x) in cand {
            let n = next.entry(x.source.clone()).or_insert_with(|| self.admitted.get(&x.source).copied().unwrap_or(0) + 1);
            if x.seq < *n {
                ack.dups.push((x.source.clone(), x.seq));
            } else if x.seq > *n {
                ack.held.push((x.source.clone(), x.seq));
                self.offered.push(x);
            } else {
                *n += 1;
                chosen.push((self.priority(&x.name), x));
            }
        }
        // Held inputs may become contiguous only through inputs of this
        // window; re-scan them once in seq order so the result is the same
        // whatever order they were offered in.
        loop {
            let mut moved = false;
            self.offered.sort_by(|a, b| (&a.source, a.seq).cmp(&(&b.source, b.seq)));
            let mut rest = vec![];
            for x in std::mem::take(&mut self.offered) {
                let n = next.get(&x.source).copied();
                if x.tick <= upto && n == Some(x.seq) {
                    next.insert(x.source.clone(), x.seq + 1);
                    ack.held.retain(|h| !(h.0 == x.source && h.1 == x.seq));
                    chosen.push((self.priority(&x.name), x));
                    moved = true;
                } else if x.tick <= upto && n.is_some_and(|n| x.seq < n) {
                    ack.held.retain(|h| !(h.0 == x.source && h.1 == x.seq));
                    ack.dups.push((x.source.clone(), x.seq));
                    moved = true;
                } else {
                    rest.push(x);
                }
            }
            self.offered = rest;
            if !moved {
                break;
            }
        }
        // The closing Tick goes last; the rest by (priority desc, source, seq).
        let is_tick = |h: &Host, x: &Sourced| h.tick_of(x).is_some();
        let mut keyed: Vec<(bool, u8, Sourced)> = chosen.into_iter().map(|(p, x)| (is_tick(self, &x), p, x)).collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then_with(|| (&a.2.source, a.2.seq).cmp(&(&b.2.source, b.2.seq))));
        let chosen: Vec<(u8, Sourced)> = keyed.into_iter().map(|(_, p, x)| (p, x)).collect();
        let prev_upto = (self.window > 0).then_some(self.closed_tick);
        let id = self.window;
        self.append(host_event("Window", &format!("{{\"count\":{},\"id\":{id},\"term\":{},\"upto\":{upto}}}", chosen.len(), self.term)))?;
        self.window += 1;
        self.closed_tick = upto;
        let mut acks = vec![];
        for (_, x) in chosen {
            let src = serde_json::to_string(&x.source).unwrap();
            let late = if prev_upto.is_some_and(|u| x.tick <= u) { "\"late\":true," } else { "" };
            self.append(host_event("Input", &format!("{{{late}\"seq\":{},\"source\":{src},\"window\":{id}}}", x.seq)))?;
            let a = self.apply(&x.name, x.args, How::Sourced)?;
            self.admitted.insert(x.source.clone(), x.seq);
            ack.admitted.extend(a);
            ack.admitted.extend(self.drain()?);
            acks.push((x.source, x.seq));
        }
        self.confirm()?;
        ack.acks = acks; // only now: every entry of the window is behind a successful barrier
        Ok(ack)
    }

    /// E.2b: take the admitter role for `term`. The term is a fence token:
    /// `host.Term` is appended (and made durable through the replicated log)
    /// before any input of the new term, every later `host.Window` carries
    /// it, and a lower or equal term is not taken. Requests emitted under the
    /// previous term with no result in D get `failed`/`interrupted` (HC-3).
    pub fn take_role(&mut self, term: u64, leader: &str) -> Result<(), String> {
        if term <= self.term {
            return Err(format!("host.Term: term {term} is not above the current term {}", self.term));
        }
        // HC-21: a kept step the previous admitter logged without its
        // `host.StepKept` (or bound `StepKept`) is completed first; it belongs
        // to the input of the previous term
        self.complete_owed()?;
        self.term = term;
        self.offered.clear();
        let l = serde_json::to_string(leader).unwrap();
        self.append(host_event("Term", &format!("{{\"leader\":{l},\"term\":{term}}}")))?;
        // HC-22: the new admitter's adapters are fresh
        self.restarted()?;
        self.drain()?;
        self.confirm()
    }

    /// E.2b: open a host on a log received from the replicated log (the
    /// longest log among a quorum at election). The whole chain must verify;
    /// the host appends nothing until [`Host::take_role`].
    pub fn resume(m: &'a Module, text: &str, store: Box<dyn Store + 'a>) -> Result<Host<'a>, String> {
        let lines: Vec<&str> = text.split_terminator('\n').collect();
        let keep = verify_prefix(&lines, m);
        if keep < lines.len() {
            return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
        }
        let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
        let mut h = Host::fresh(m, store, vec![])?;
        h.refold(&entries)?;
        Ok(h)
    }

    /// Evaluate a query (HS-8): a step that reads state and emits its result
    /// rows, run on the current state without appending to D. A step that
    /// would change state is not a query and is not performed.
    pub fn query(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<V>, String> {
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
        let ev = match self.it.construct(name, &args) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("`{name}` is not an event")),
            Err(f) => return Err(why(f)),
        };
        let (s2, outs) = self.it.delta(&self.state, &ev)?;
        if d_sha(&state_pos_json(&s2, self.m)) != self.state_hash {
            return Err(format!("`{name}` changes state; a query only reads"));
        }
        Ok(outs)
    }

    /// HC-2, HD-4: the host's value of frame tick `n`: {n}, matched by field
    /// name like any delivery; a `Tick` declared with no field carries no
    /// count.
    fn tick_value(&self, n: u64) -> serde_json::Map<String, Value> {
        let mut p = serde_json::Map::new();
        if self.event_fields("Tick").is_some_and(|f| !f.is_empty()) {
            p.insert("n".into(), Value::from(n));
        }
        p
    }

    /// HD-4: frame tick `n` built into the module's `Tick`, or why it
    /// cannot be built (the host loop checks tick 1 at start).
    pub fn tick_args(&mut self, n: u64) -> Result<Vec<Arg>, String> {
        let p = self.tick_value(n);
        match self.build("Tick", &p) {
            Build::Built(args, _) => Ok(args),
            Build::Unbuildable(why, _) => Err(why),
            Build::Undeclared => Err("not declared".into()),
        }
    }

    /// HC-2, HD-4: deliver frame tick `n` (the host loop's `Tick`) as the
    /// module declares it, perform its effects and take the batch barrier.
    /// `Ok(None)` when tick `n` cannot be built: `host.Kept` records it
    /// with the reason, and the host loop delivers no later tick, since a
    /// count that no longer converts (`Tick(n: u8)` past 255) does not
    /// convert again.
    pub fn tick(&mut self, n: u64) -> Result<Option<Vec<Admitted>>, String> {
        let p = self.tick_value(n);
        let plan = self.plan("Tick", &p, false);
        let built = plan.deliver.is_some();
        let mut out = self.deliver_plan("Tick", &p, plan, How::Delivery)?;
        if built {
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(built.then_some(out))
    }

    /// Whether the module declares this event, and its payload field names.
    pub fn event_fields(&self, name: &str) -> Option<Vec<String>> {
        self.m.items.iter().find_map(|i| match i {
            Item::Event { name: n, payload, .. } if n == name => Some(payload.iter().map(|(f, _)| f.clone()).collect()),
            _ => None,
        })
    }

    /// The wall-clock adapter (HC-2): the admitter reads the clock, logs
    /// `host.WallTime`, and delivers the bound `WallTime(ns)` when declared.
    /// `host.TlsIdentity`: the Ed25519 public key this run presents to its
    /// peers over TLS (the seed stays outside D). Changes no state.
    pub fn tls_identity(&mut self, public_hex: &str) -> Result<(), String> {
        self.append(host_event("TlsIdentity", &format!("{{\"ed25519\":\"{public_hex}\"}}")))?;
        self.confirm()
    }

    /// HC-32: `host.Zone` {file, sha256, tz}: the time zone rules the
    /// LocalOffset adapter answers from, appended at start when the module
    /// binds that request. Changes no state; replay needs it for nothing.
    pub fn zone(&mut self, payload_json: &str) -> Result<(), String> {
        self.append(host_event("Zone", payload_json))?;
        self.confirm()
    }

    /// Whether the module binds this request kind (a struct of that name,
    /// with an adapter that performs it, E.4c).
    pub fn binds_kind(&self, kind: &str) -> bool {
        self.bound.contains(&kind)
    }

    /// HC-13: the operations a fabric may compute for state, with the probe
    /// digests that admitted them (`host.Exact`).
    pub fn exact(&mut self, payload_json: &str) -> Result<(), String> {
        self.append(host_event("Exact", payload_json))?;
        self.confirm()
    }

    /// HC-38: `host.Energy`, a package counter reading over the entries
    /// `first..=last` of this run, in the D.11 form of a `reported` term
    /// with its provenance and amortization. Changes no state; δ never
    /// reads it.
    pub fn energy(&mut self, source: &str, term: &rce_energy::reading::Term, first: u64) -> Result<(), String> {
        let seq = self.len();
        if first >= seq {
            return Err(format!("host.Energy: no entry of D from seq {first} to cover"));
        }
        let last = seq - 1;
        let t: Value = serde_json::from_str(&term.json()).map_err(|e| e.to_string())?;
        let p = serde_json::json!({
            "amortization": t["amortization"],
            "basis": t["basis"],
            "entries": [first, last],
            "joules": t["j"],
            "provenance": t["provenance"],
            "source": source,
        });
        energy_check(&p, seq)?;
        self.append(host_event("Energy", &crate::interp::canon_json(&p)))?;
        self.confirm()
    }

    pub fn wall_time(&mut self, ns: u64) -> Result<(), String> {
        self.append(host_event("WallTime", &format!("{{\"ns\":{ns}}}")))?;
        if declares_event(self.m, "WallTime") {
            let p = one_field("ns", ns);
            let plan = self.plan("WallTime", &p, true);
            self.deliver_plan("WallTime", &p, plan, How::Delivery)?;
            self.drain()?;
        }
        self.confirm()
    }

    /// HC-29: the frames of one program entry, from its step's outputs
    /// (`None`: the step was kept, and emitted nothing).
    fn note_frames(&mut self, seq: u64, name: &str, outs: Option<&[V]>) -> Vec<Result<crate::frame::Frame, String>> {
        self.clock.entry(name);
        let fs = outs.map(|o| self.clock.frames(seq, name, o)).unwrap_or_default();
        for f in fs.iter().flatten() {
            self.shown.push((f.seq, f.frame_digest.clone()));
            self.last_view = Some(f.view.clone());
        }
        fs
    }

    /// HC-29: a renderer's confirmation that it presented the frame of the
    /// entry at `seq` whose frame digest is `digest`. The admitter appends
    /// `host.Presented` {digest, seq}, then delivers the bound
    /// `Presented(seq, digest)` when declared. A confirmation that names no
    /// frame derived from D is refused: nothing is appended.
    pub fn presented(&mut self, seq: u64, digest: &str) -> Result<Vec<Admitted>, String> {
        if !self.shown.iter().any(|(s, d)| *s == seq && d == digest) {
            return Err(format!("presented {seq} {digest}: no frame of D has this seq and frame digest"));
        }
        let mut p = serde_json::Map::new();
        p.insert("digest".into(), Value::from(digest));
        p.insert("seq".into(), Value::from(seq));
        self.append(host_event("Presented", &Value::Object(p.clone()).to_string()))?;
        let mut out = vec![];
        if declares_event(self.m, "Presented") {
            let plan = self.plan("Presented", &p, true);
            out = self.deliver_plan("Presented", &p, plan, How::Delivery)?;
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// HC-29: the view of the last frame derived from D.
    pub fn last_frame(&self) -> Option<&crate::frame::ViewTree> {
        self.last_view.as_ref()
    }

    /// HC-29: (seq, frame digest) of every frame derived from D so far.
    pub fn shown(&self) -> &[(u64, String)] {
        &self.shown
    }

    pub fn state(&self) -> &[Region] {
        &self.state
    }
    pub fn state_hash(&self) -> &str {
        &self.state_hash
    }
    pub fn len(&self) -> u64 {
        self.seq
    }
    pub fn is_empty(&self) -> bool {
        self.seq == 0
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
}

/// Replay (HC-8, T37): verify D and re-derive every state hash with every
/// adapter disconnected. Returns (entries, final state hash, final state).
pub fn replay(m: &Module, text: &str) -> Result<(u64, String, Vec<Region>), String> {
    replay_report(m, text).map(|r| (r.entries, r.state_hash, r.state))
}

/// [`replay`] with the kept steps it re-derived (HC-21): each kept step's
/// seq, cause and text. The text is not in D; replay derives it.
pub fn replay_report(m: &Module, text: &str) -> Result<Replayed, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if keep < lines.len() {
        return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
    }
    let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries)?;
    Ok(Replayed { entries: h.seq, state_hash: h.state_hash.clone(), state: h.state.clone(), kept_outcomes: std::mem::take(&mut h.kept_outcomes) })
}

/// The outputs δ gives on replay, for each program entry it applies: (seq,
/// each output as Form JSON). Every effect request, and so every byte an
/// adapter writes (HC-26), is a function of D: this is that function.
pub fn replay_outputs(m: &Module, text: &str) -> Result<Vec<(u64, Vec<Value>)>, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if keep < lines.len() {
        return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
    }
    let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.outputs = Some(vec![]);
    h.refold(&entries)?;
    Ok(h.outputs.take().unwrap_or_default())
}

/// HC-36: replay D with every adapter disconnected and re-derive the bytes
/// of every RAP result D holds, in D order, each with the status that
/// settled its delivery. `rce host replay rap FILE D` prints them.
pub fn replay_rap(m: &Module, text: &str) -> Result<Vec<RapOut>, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if keep < lines.len() {
        return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
    }
    let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries)?;
    let calls: Vec<u64> = h.rap.calls.keys().copied().collect();
    Ok(calls.into_iter().filter_map(|c| h.rap_out(c)).collect())
}

/// HC-29: replay D with every adapter disconnected and re-derive every
/// frame, in D order, with the report of [`replay_report`].
pub fn replay_frames(m: &Module, text: &str) -> Result<(Replayed, Vec<Result<crate::frame::Frame, String>>), String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if keep < lines.len() {
        return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
    }
    let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.frames = Some(vec![]);
    h.refold(&entries)?;
    let fs = h.frames.take().unwrap_or_default();
    Ok((Replayed { entries: h.seq, state_hash: h.state_hash.clone(), state: h.state.clone(), kept_outcomes: std::mem::take(&mut h.kept_outcomes) }, fs))
}

/// The host as of entry k (HS-9): the fold of the first k entries of a
/// verified D, on a volatile store, with every adapter disconnected.
pub fn as_of<'a>(m: &'a Module, text: &str, k: usize) -> Result<Host<'a>, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if k > keep {
        return Err(format!("D verifies only up to entry {keep}, asked for {k}"));
    }
    let entries: Vec<Entry> = lines[..k].iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries)?;
    Ok(h)
}

/// HC-37: a compaction of D at N, proven: the text of D' (the checkpoint,
/// then the entries of D from seq N on, unchanged), the archive (the lines
/// of D the checkpoint stands for, byte for byte) and its SHA-256.
#[derive(Debug)]
pub struct Compaction {
    pub text: String,
    pub archive: String,
    pub sha: String,
    /// the checkpoint's seq (N - 1) and state hash
    pub seq: u64,
    pub state_hash: String,
    /// the entries D' keeps after the checkpoint
    pub kept: usize,
}

fn lines_text(lines: &[&str]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

/// The verified entries of `text`, or why it is not a whole log: a line
/// that does not verify, or a last line without its line end (a torn tail).
fn whole_log(m: &Module, text: &str) -> Result<Vec<Entry>, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines, m);
    if keep < lines.len() || (!text.is_empty() && !text.ends_with('\n')) {
        return Err(format!("D does not verify after entry {keep} of {}: a torn or corrupted tail is cut by a host on open (HS-4), and is not compacted", lines.len()));
    }
    Ok(lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect())
}

/// HC-37: the checkpoint payload of a fold of the archive text: what the
/// host carries, the archive's SHA-256 and the HTTP inputs open at its end.
fn checkpoint_payload(h: &Host, archive: &str, m: &Module) -> Result<String, String> {
    let mut p = h.checkpoint_value()?;
    p.insert("archive".into(), Value::String(d_sha(archive)));
    let open: Vec<Value> = http_open(m, archive)?.into_iter().map(|x| serde_json::json!([x.id, x.head, x.close])).collect();
    p.insert("http".into(), Value::Array(open));
    Ok(crate::interp::canon_json(&Value::Object(p)))
}

/// HC-37: compact the whole, verified log `text` at N: the checkpoint
/// stands for the entries with seq below N and carries the fold of D at N.
/// The checkpoint is admitted only as that fold: D is replayed in full, D'
/// is replayed from the checkpoint, and the two must agree at every entry
/// after N on the state hash, the outputs, the kept steps, the RAP results,
/// the open HTTP inputs and everything else the fold reaches. `Err` says
/// why no checkpoint may stand at N.
pub fn compact_text(m: &Module, text: &str, n: u64) -> Result<Compaction, String> {
    let entries = whole_log(m, text)?;
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let (Some(first), Some(last)) = (entries.first(), entries.last()) else { return Err("D is empty".into()) };
    // the lowest N: above the entries an earlier checkpoint stands for
    let low = if first.kind() == CHECKPOINT { first.seq + 2 } else { 1 };
    if n < low || n > last.seq + 1 {
        return Err(format!("N must be from {low} to {} (the entries of D have seq {} to {})", last.seq + 1, first.seq, last.seq));
    }
    let count = (n - first.seq) as usize;
    let at = &entries[count - 1];
    if at.kind() == "host.Input" {
        return Err(format!("entry {} announces an input of a window that follows it (HC-17): a checkpoint stands only where every input's entries are complete", at.seq));
    }
    // the full replay of D, with the fold at N taken on the way
    let mut full = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    full.outputs = Some(vec![]);
    full.refold(&entries[..count]).map_err(|e| format!("replay of D fails: {e}"))?;
    let archive = lines_text(&lines[..count]);
    let payload = checkpoint_payload(&full, &archive, m)?;
    let at_n = format!("{:?}", full.state);
    let before_kept = full.kept_outcomes.len();
    full.outputs = Some(vec![]);
    full.refold(&entries[count..]).map_err(|e| format!("replay of D fails: {e}"))?;
    let cp = Entry::new(m, n - 1, host_event("Checkpoint", &payload), D_GENESIS, &at.state_hash);
    let new = format!("{}\n{}", cp.line(), lines_text(&lines[count..]));
    // the replay of D' from the checkpoint
    let ne = whole_log(m, &new).map_err(|e| format!("D' does not verify: {e}"))?;
    // the state read back from the checkpoint is the value δ left, every
    // part of it with its type, not only one with the same bytes: the
    // entries a host appends later are folded from it
    let mut back = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    back.refold(&ne[..1]).map_err(|e| format!("the checkpoint does not load: {e}"))?;
    if format!("{:?}", back.state) != at_n {
        return Err(format!("the checkpoint at N {n} does not replay equal to D: the state read back differs from the state at N"));
    }
    let mut from = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    from.outputs = Some(vec![]);
    from.refold(&ne).map_err(|e| format!("D' does not replay: {e}"))?;
    let differ = |what: &str| Err(format!("the checkpoint at N {n} does not replay equal to D: {what} differ"));
    if from.state_hash != full.state_hash || from.seq != full.seq {
        return differ("the final state hashes");
    }
    if from.outputs != full.outputs {
        return differ("the outputs of the entries after N");
    }
    if from.kept_outcomes != full.kept_outcomes[before_kept..] {
        return differ("the kept steps after N");
    }
    let rap = |h: &Host| -> Vec<RapOut> { h.rap.calls.keys().filter_map(|c| h.rap_out(*c)).collect() };
    if rap(&from) != rap(&full) {
        return differ("the RAP results");
    }
    if from.fold_digest() != full.fold_digest() {
        return differ("the host states after the last entry");
    }
    let carried = http_open(m, &archive)?;
    let want: Vec<HttpOfD> = http_inputs(m, text)?.into_iter().filter(|x| x.id >= n || carried.contains(x)).collect();
    if http_inputs(m, &new)? != want {
        return differ("the HTTP inputs");
    }
    Ok(Compaction { text: new, sha: d_sha(&archive), archive, seq: n - 1, state_hash: at.state_hash.clone(), kept: entries.len() - count })
}

/// HC-37: prove the checkpoint that begins `text` from the archive text it
/// names: the archive's SHA-256 is the one the checkpoint carries, the
/// archive is a whole log that replays, and its fold gives the checkpoint
/// byte for byte. An archive that itself begins with a checkpoint is
/// proven in turn by [`prove_file`].
pub fn prove_checkpoint(m: &Module, text: &str, archive: &str) -> Result<(), String> {
    let first = text.split_terminator('\n').next().and_then(Entry::parse_line).ok_or("D is empty")?;
    let v: Value = serde_json::from_str(&first.event).map_err(|x| x.to_string())?;
    if first.kind() != CHECKPOINT {
        return Err("D does not begin with a checkpoint".into());
    }
    if v["payload"]["archive"].as_str() != Some(d_sha(archive).as_str()) {
        return Err(format!("the archive's SHA-256 is {}, the checkpoint names {}", d_sha(archive), v["payload"]["archive"]));
    }
    let entries = whole_log(m, archive).map_err(|e| format!("the archive: {e}"))?;
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries).map_err(|e| format!("the archive does not replay: {e}"))?;
    let want = Entry::new(m, h.seq.saturating_sub(1), host_event("Checkpoint", &checkpoint_payload(&h, archive, m)?), D_GENESIS, &h.state_hash);
    if want.line() != text.split_terminator('\n').next().unwrap_or("") {
        return Err("the fold of the archive does not give the checkpoint".into());
    }
    Ok(())
}

/// HC-37: prove the checkpoint that begins the log `text` of the file `d`
/// from its archive beside D, then the archive's own checkpoint, and so on.
/// Returns one line per checkpoint: proven, or its archive not present (the
/// checkpoint was then checked only by the entries after it). An archive
/// that is present and does not prove its checkpoint is an `Err`.
pub fn prove_file(m: &Module, d: &Path, text: &str) -> Result<Vec<String>, String> {
    let mut notes = vec![];
    let mut cur = text.to_string();
    while let Some((seq, sha)) = checkpoint_archive(&cur) {
        let ap = archive_path(d, &sha);
        let Ok(a) = std::fs::read_to_string(&ap) else {
            notes.push(format!("checkpoint at seq {seq}: archive {} is not beside D; the checkpoint is checked by the entries after it", ap.display()));
            break;
        };
        prove_checkpoint(m, &cur, &a).map_err(|e| format!("checkpoint at seq {seq}: {}: {e}", ap.display()))?;
        notes.push(format!("checkpoint at seq {seq}: proven from archive {}", ap.display()));
        cur = a;
    }
    Ok(notes)
}

/// HC-37: the archive a checkpoint names, beside D: `D.archive.SHA256`.
pub fn archive_path(d: &Path, sha: &str) -> PathBuf {
    let mut s = d.as_os_str().to_os_string();
    s.push(format!(".archive.{sha}"));
    PathBuf::from(s)
}

/// HC-37: the archive SHA-256 the checkpoint that begins `text` names.
pub fn checkpoint_archive(text: &str) -> Option<(u64, String)> {
    let e = text.split_terminator('\n').next().and_then(Entry::parse_line)?;
    checkpoint_head(&e)?;
    let v: Value = serde_json::from_str(&e.event).ok()?;
    Some((e.seq, v["payload"]["archive"].as_str()?.to_string()))
}

/// HS-2, E.3: write `bytes` to `path` through a file beside it, made
/// durable, then renamed into place with the rename barrier of E.3.
pub fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(&tmp)?;
        f.write_all(bytes)?;
        // Rust std: F_FULLFSYNC on Apple platforms, fsync elsewhere
        f.sync_all()?;
    }
    sync_dir(&tmp)?;
    rename_durable(&tmp, path)?;
    sync_dir(path)
}

#[cfg(not(windows))]
fn rename_durable(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::rename(from, to)
}

/// E.3, Windows: `MoveFileEx` with `MOVEFILE_WRITE_THROUGH`.
#[cfg(windows)]
fn rename_durable(from: &Path, to: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let w = |p: &Path| -> Vec<u16> { p.as_os_str().encode_wide().chain(std::iter::once(0)).collect() };
    let (a, b) = (w(from), w(to));
    // MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    // SAFETY: two NUL-terminated wide strings that outlive the call.
    if unsafe { MoveFileExW(a.as_ptr(), b.as_ptr(), 0x1 | 0x8) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// HC-37: compact the log file `d` at N. The archive is written first and
/// made durable (`D.archive.SHA256`, its bytes the lines replaced), then D'
/// is written beside D and made durable, D is read again and must be
/// unchanged, and D' is renamed over D with the rename barrier (E.3), so a
/// crash at any point leaves D or D' whole, each with every entry it names.
/// `Err` is a refusal (exit status 2): why no checkpoint may stand at N.
pub fn compact_file(m: &Module, d: &Path, n: u64) -> Result<Compaction, String> {
    let text = std::fs::read_to_string(d).map_err(|e| format!("{}: {e}", d.display()))?;
    let c = compact_text(m, &text, n)?;
    let ap = archive_path(d, &c.sha);
    let have = std::fs::read(&ap).ok();
    if have.as_deref() != Some(c.archive.as_bytes()) {
        write_durable(&ap, c.archive.as_bytes()).map_err(|e| format!("host.StorageFault: {}: {e}", ap.display()))?;
    }
    if std::fs::read_to_string(d).ok().as_deref() != Some(text.as_str()) {
        return Err(format!("{} changed while it was compacted: compact a log no host has open", d.display()));
    }
    write_durable(d, c.text.as_bytes()).map_err(|e| format!("host.StorageFault: {}: {e}", d.display()))?;
    Ok(c)
}

/// Path helper for the CLI.
pub fn log_path(p: &str) -> PathBuf {
    PathBuf::from(p)
}
