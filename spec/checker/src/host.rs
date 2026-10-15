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
//! - HS-6 bindings: `host.Bind` at start, then `host.Synced` after each
//!   barrier and the bound program event `Synced(upto)` when declared.
//!
//! The entry format is exactly the 6.1 chain of `form_json`: `state_hash` is
//! SHA-256 of the canonical state JSON, `hash` is SHA-256 of
//! `{"event":..,"prev_hash":..,"seq":..,"state_hash":..}`. Each line of the
//! file is `{"event":E,"hash":H,"prev_hash":P,"seq":N,"state_hash":S}`.
use crate::ast::{Arg, Expr, Item, Module, Ty};
use crate::interp::{D_GENESIS, Interp, Region, V, d_sha, event_json, state_json, why};
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
    /// Canonical state JSON (`state_json`); its SHA-256 is `state_hash`.
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

impl Entry {
    pub fn new(seq: u64, event: String, prev_hash: &str, state_hash: &str) -> Entry {
        let hash = d_sha(&body(&event, prev_hash, seq, state_hash));
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

/// HS-4: the number of lines in the longest prefix that links from the
/// genesis hash. A line that does not parse (a torn write) ends the prefix.
pub fn verify_prefix(lines: &[&str]) -> usize {
    let mut prev = D_GENESIS.to_string();
    for (i, l) in lines.iter().enumerate() {
        let Some(e) = Entry::parse_line(l) else { return i };
        if e.seq != i as u64 || e.prev_hash != prev || e.hash != d_sha(&body(&e.event, &e.prev_hash, e.seq, &e.state_hash)) {
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

/// D in a file: one entry per line, append only.
pub struct FileStore {
    f: File,
}

impl FileStore {
    /// Open (creating if needed) and return the store and the existing bytes.
    pub fn open(path: &Path) -> io::Result<(FileStore, String)> {
        let created = !path.exists();
        let mut f = OpenOptions::new().read(true).append(true).create(true).open(path)?;
        if created {
            // E.3: a new file needs the directory barrier too
            let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
            File::open(dir)?.sync_all()?;
        }
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok((FileStore { f }, s))
    }
    /// Cut the file to `len` bytes (HS-4) and make the cut durable.
    pub fn cut(&mut self, len: u64) -> io::Result<()> {
        self.f.set_len(len)?;
        self.f.sync_all()
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
    /// `Ok(outputs)` (each output's display text) or `Err(refusal)` (T8: state unchanged).
    pub outcome: Result<Vec<String>, String>,
    /// Display sink (λ output only): the `data` of each emitted `Display(data: text)`.
    pub display: Vec<String>,
    /// The step emitted `Exit`: the host loop ends after this input.
    pub exit: bool,
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

fn host_event(kind: &str, payload: &str) -> String {
    format!("{{\"payload\":{payload},\"type\":\"host.{kind}\"}}")
}

/// Rebuild a value from its stored Form JSON (6.1), guided by its declared
/// type, so replay uses no parser of program text. Records become
/// constructor calls of their struct (or of the kind variant named by `$v`),
/// maps come back from their canonical list of [key, value] pairs.
fn json_expr(v: &Value, ty: Option<&Ty>, m: &Module) -> Result<Expr, String> {
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
}

/// The result event's required shape; `Err` names what does not fit.
fn result_event_fits(m: &Module, kind: &str) -> Result<(), String> {
    let ev = format!("{kind}Done");
    let Some(p) = m.items.iter().find_map(|i| match i {
        Item::Event { name, payload, .. } if *name == ev => Some(payload),
        _ => None,
    }) else {
        return Err(format!("W-HOST-1 effect `{kind}` is emitted but `event {ev}(req: u64, status: text, data: T?, err: text)` is not declared"));
    };
    let ty = |n: &str| p.iter().find(|(k, _)| k == n).map(|(_, t)| t.clone());
    let path = |t: Option<Ty>, want: &str| matches!(t, Some(Ty::Path(ref x, _)) if x.len() == 1 && x[0] == want);
    if !path(ty("req"), "u64") || !path(ty("status"), "text") || !path(ty("err"), "text") || !matches!(ty("data"), Some(Ty::Opt(_))) || p.len() != 4 {
        return Err(format!("W-HOST-1 `event {ev}` must be ({ev}(req: u64, status: text, data: T?, err: text))"));
    }
    Ok(())
}

fn declares_struct(m: &Module, name: &str) -> bool {
    m.items.iter().any(|i| matches!(i, Item::Struct { name: n, .. } if n == name))
}

/// The admitter and the program state it folds (one per program instance, HC-1).
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
    /// The last seq made durable.
    pub durable: Option<u64>,
    /// The seq of the snapshot this host was opened from, if one was used (HS-5).
    pub from_snapshot: Option<u64>,
    adapters: Vec<Box<dyn Adapter + 'a>>,
    /// Effect kinds bound at start (HS-6), in adapter order.
    bound: Vec<&'static str>,
    /// Requests emitted and not yet answered: (kind, req, request JSON), in emit order (E-x).
    queue: std::collections::VecDeque<(String, u64, Value)>,
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
        for a in &adapters {
            if declares_struct(m, a.kind()) {
                result_event_fits(m, a.kind())?;
                bound.push(a.kind());
            }
        }
        let mut it = Interp::new(m);
        let state = it.initial_state().map_err(why)?;
        let state_hash = d_sha(&state_json(&state, m));
        Ok(Host { m, it, state, state_hash, prev: D_GENESIS.into(), seq: 0, store, stopped: false, pending: 0, durable: None, from_snapshot: None, adapters, bound, queue: Default::default() })
    }

    fn start_entries(&mut self) -> Result<(), String> {
        let caps = self.store.caps();
        self.append(host_event("StorageCaps", &format!("{{\"barrier\":\"{caps}\"}}")))?;
        self.append(host_event("FabricCaps", &fabric_caps()))?;
        if declares_event(self.m, "Synced") {
            self.append(host_event("Bind", "{\"event\":\"Synced\",\"kind\":\"host.Synced\"}"))?;
        }
        if declares_event(self.m, "WallTime") {
            self.append(host_event("Bind", "{\"event\":\"WallTime\",\"kind\":\"host.WallTime\"}"))?;
        }
        for k in self.bound.clone() {
            self.append(host_event("Bind", &format!("{{\"event\":\"{k}Done\",\"kind\":\"host.{k}\"}}")))?;
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
        let (mut fs, text) = FileStore::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if text.is_empty() {
            return Host::start_with(m, Box::new(fs), adapters);
        }
        let lines: Vec<&str> = text.split_terminator('\n').collect();
        let keep = verify_prefix(&lines);
        let dropped = lines.len() - keep;
        if dropped > 0 || !text.ends_with('\n') {
            let len: usize = lines[..keep].iter().map(|l| l.len() + 1).sum();
            fs.cut(len as u64).map_err(|e| format!("host.StorageFault: cut: {e}"))?;
        }
        let entries: Vec<Entry> = lines[..keep].iter().map(|l| Entry::parse_line(l).unwrap()).collect();
        let mut h = Host::fresh(m, Box::new(fs), adapters)?;
        let from = match snap {
            Some(sn) if h.load_snapshot(sn, &entries) => sn.seq as usize + 1,
            _ => 0,
        };
        h.refold(&entries[from..])?;
        let caps = h.store.caps();
        if dropped > 0 {
            h.append(host_event("Truncated", &format!("{{\"dropped\":{dropped}}}")))?;
        }
        h.append(host_event("StorageCaps", &format!("{{\"barrier\":\"{caps}\"}}")))?;
        h.append(host_event("FabricCaps", &fabric_caps()))?;
        while let Some((k, req, _)) = h.queue.pop_front() {
            h.answer(&k, req, EffectResult::failed("interrupted"))?;
        }
        h.drain()?;
        h.confirm()?;
        Ok(h)
    }

    /// A snapshot of the state after the last entry (HS-5). Taken only
    /// between batches, when no effect request is outstanding, so the
    /// entries after it rebuild every request that matters.
    pub fn snapshot(&self) -> Option<Snapshot> {
        if !self.queue.is_empty() || self.seq == 0 {
            return None;
        }
        Some(Snapshot { seq: self.seq - 1, state_hash: self.state_hash.clone(), bytes: state_json(&self.state, self.m) })
    }

    /// Use a snapshot only when SHA-256 of its bytes is its state_hash, that
    /// equals D's `state_hash` at its seq, and the bytes decode (guided by
    /// the declared field types) to a state that re-encodes to the same
    /// bytes. Anything else: discarded, nothing changes (HS-5).
    fn load_snapshot(&mut self, sn: &Snapshot, entries: &[Entry]) -> bool {
        let Some(e) = entries.get(sn.seq as usize) else { return false };
        if d_sha(&sn.bytes) != sn.state_hash || e.state_hash != sn.state_hash {
            return false;
        }
        let Ok(v) = serde_json::from_str::<Value>(&sn.bytes) else { return false };
        let mut st = self.state.clone();
        for (rn, fs) in st.iter_mut() {
            for (fname, val, ty) in fs.iter_mut() {
                let Some(j) = v.get(rn.as_str()).and_then(|r| r.get(fname.as_str())) else { return false };
                let Ok(x) = json_expr(j, Some(ty), self.m) else { return false };
                self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
                match self.it.eval(&x) {
                    Ok(nv) => *val = nv,
                    Err(_) => return false,
                }
            }
        }
        if state_json(&st, self.m) != sn.bytes {
            return false;
        }
        self.state = st;
        self.state_hash = sn.state_hash.clone();
        self.prev = e.hash.clone();
        self.seq = sn.seq + 1;
        self.from_snapshot = Some(sn.seq);
        true
    }

    /// Fold verified entries into this host without appending (HC-8).
    fn refold(&mut self, entries: &[Entry]) -> Result<(), String> {
        for e in entries {
            if !e.is_host() {
                let (name, args) = decode_event(&e.event, self.m)?;
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
                if let Some(k) = name.strip_suffix("Done").filter(|k| self.bound.contains(k)) {
                    let req = serde_json::from_str::<Value>(&e.event).ok().and_then(|v| v["payload"]["req"].as_u64());
                    self.queue.retain(|(qk, qr, _)| !(qk == k && Some(*qr) == req));
                }
                if let Ok((s2, outs)) = self.it.delta(&self.state, &ev) {
                    self.state = s2;
                    self.state_hash = d_sha(&state_json(&self.state, self.m));
                    self.enqueue(&outs);
                }
            }
            if e.state_hash != self.state_hash {
                return Err(format!("host fault: seq {} state_hash {} but replay gives {}", e.seq, e.state_hash, self.state_hash));
            }
            self.prev = e.hash.clone();
            self.seq = e.seq + 1;
        }
        self.durable = self.seq.checked_sub(1);
        Ok(())
    }

    fn append(&mut self, event: String) -> Result<Entry, String> {
        if self.stopped {
            return Err("host.StorageFault: admitter stopped after a failed barrier".into());
        }
        let e = Entry::new(self.seq, event, &self.prev, &self.state_hash);
        if let Err(x) = self.store.append(&e.line()) {
            self.stopped = true;
            return Err(format!("host.StorageFault: seq {}: {x}", e.seq));
        }
        self.prev = e.hash.clone();
        self.seq += 1;
        self.pending += 1;
        Ok(e)
    }

    /// The barrier for everything pending; on success log `host.Synced` and
    /// the bound `Synced` event (HS-6). A failure stops the admitter (HS-3).
    fn confirm(&mut self) -> Result<(), String> {
        if self.pending == 0 {
            return Ok(());
        }
        if let Err(x) = self.store.barrier() {
            self.stopped = true;
            return Err(format!("host.StorageFault: seq {}: {x}", self.seq - 1));
        }
        self.pending = 0;
        let upto = self.seq - 1;
        if self.store.caps() == "volatile" {
            return Ok(()); // E.3: no entry of a volatile store is reported durable
        }
        self.durable = Some(upto);
        self.append(host_event("Synced", &format!("{{\"upto\":{upto}}}")))?;
        if declares_event(self.m, "Synced") {
            self.apply_named("Synced", vec![Arg { label: Some("upto".into()), e: Expr::Int(upto as i128, 0) }])?;
        }
        Ok(())
    }

    /// Queue the effect requests among a step's outputs, in emit order.
    fn enqueue(&mut self, outs: &[V]) {
        for o in outs {
            if let V::Rec(n, fs) = o
                && let Some(k) = self.bound.iter().find(|k| **k == n.as_str())
            {
                let req = fs.iter().find_map(|(f, v)| match (f.as_str(), v) {
                    ("req", V::Int(i, _)) => Some(*i as u64),
                    _ => None,
                });
                let j: Value = serde_json::from_str(&crate::interp::form_value(o, self.m).json()).unwrap_or(Value::Null);
                self.queue.push_back((k.to_string(), req.unwrap_or(0), j));
            }
        }
    }

    /// Admit the result event of one request (HC-3).
    fn answer(&mut self, kind: &str, req: u64, r: EffectResult) -> Result<Admitted, String> {
        let ev = format!("{kind}Done");
        let dty = self.m.items.iter().find_map(|i| match i {
            Item::Event { name, payload, .. } if *name == ev => payload.iter().find(|(k, _)| k == "data").map(|(_, t)| t.clone()),
            _ => None,
        });
        let (status, data, error) = match r.data.as_ref().map(|d| json_expr(d, dty.as_ref(), self.m)) {
            Some(Err(e)) => ("failed", Expr::None(0), format!("result does not fit the declared type: {e}")),
            Some(Ok(x)) => (r.status.as_str(), x, r.error.clone()),
            None => (r.status.as_str(), Expr::None(0), r.error.clone()),
        };
        let lab = |l: &str, e: Expr| Arg { label: Some(l.into()), e };
        let args = vec![lab("req", Expr::Int(req as i128, 0)), lab("status", Expr::Str(status.into(), 0)), lab("data", data), lab("err", Expr::Str(error, 0))];
        self.apply_named(&ev, args)
    }

    /// Perform every queued request with its adapter and admit its result;
    /// a result's step may request more effects, which are performed in turn.
    fn drain(&mut self) -> Result<Vec<Admitted>, String> {
        let mut out = vec![];
        while let Some((k, req, j)) = self.queue.pop_front() {
            // HS-2: an effect is an output, so the entry that requested it
            // is made durable before the effect is performed.
            if self.pending > 0 {
                self.confirm()?;
            }
            let r = match self.adapters.iter_mut().find(|a| a.kind() == k) {
                Some(a) => a.perform(&j),
                None => EffectResult { status: Status::Unreachable, data: None, error: "no adapter".into() },
            };
            out.push(self.answer(&k, req, r)?);
        }
        Ok(out)
    }

    fn apply_named(&mut self, name: &str, args: Vec<Arg>) -> Result<Admitted, String> {
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
        // is released (HC-1); a refusal leaves the state unchanged (T8).
        let (outcome, next) = match self.it.delta(&self.state, &ev) {
            Ok((s2, outs)) => (Ok(outs), Some(s2)),
            Err(r) => (Err(r), None),
        };
        let (old_state, old_hash) = (self.state.clone(), self.state_hash.clone());
        if let Some(s2) = next {
            self.state = s2;
            self.state_hash = d_sha(&state_json(&self.state, self.m));
        }
        match self.append(ej.clone()) {
            Ok(e) => {
                let (mut display, mut exit) = (vec![], false);
                if let Ok(outs) = &outcome {
                    self.enqueue(outs);
                    for o in outs {
                        match o {
                            V::Rec(n, fs) if n == "Display" => {
                                if let Some((_, V::Text(t))) = fs.iter().find(|(f, _)| f == "data") {
                                    display.push(t.clone());
                                }
                            }
                            V::Rec(n, _) if n == "Exit" => exit = true,
                            _ => {}
                        }
                    }
                }
                let outcome = outcome.map(|os| os.iter().map(|o| o.show()).collect());
                Ok(Admitted { seq: e.seq, event: ev.show(), outcome, display, exit })
            }
            Err(x) => {
                self.state = old_state;
                self.state_hash = old_hash;
                Err(x)
            }
        }
    }

    /// Admit one batch of inputs written in example syntax (`Put(key: "a",
    /// val: "1"), Scan(start: "")`), then one barrier for the batch (D-2).
    pub fn admit(&mut self, src: &str, events: &str) -> Result<Vec<Admitted>, String> {
        let probe = parse(&format!("{src}\nexample {events} -> 0\n")).map_err(|e| format!("events: {e}"))?;
        let Some(Item::Example(x)) = probe.items.last() else {
            return Err("events: not an event list".into());
        };
        let mut out = vec![];
        for (name, args) in x.events.clone() {
            out.push(self.apply_named(&name, args)?);
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// Admit one input given as an event name and arguments (adapters and
    /// the host loop), its effects, then the batch barrier.
    pub fn admit_event(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
        let mut out = vec![self.apply_named(name, args)?];
        out.extend(self.drain()?);
        self.confirm()?;
        Ok(out)
    }

    /// The priority of an input (HC-9, T25): `interrupt event E priority p`
    /// gives p (an integer literal in 0..=255); every other event has 0.
    pub fn priority(&self, name: &str) -> u8 {
        self.m.items.iter().find_map(|i| match i {
            Item::Event { name: n, interrupt: Some(Expr::Int(p, _)), .. } if n == name => Some((*p).clamp(0, 255) as u8),
            _ => None,
        }).unwrap_or(0)
    }

    /// Admit inputs that arrived simultaneously (HC-9): the admitter orders
    /// them by (priority, arrival), highest priority first and arrival order
    /// among equals, and the order it chose is the order of their entries in
    /// D, so replay needs no priority. One barrier for the batch (D-2).
    pub fn admit_simultaneous(&mut self, inputs: Vec<(String, Vec<Arg>)>) -> Result<Vec<Admitted>, String> {
        let mut ord: Vec<(u8, usize, String, Vec<Arg>)> =
            inputs.into_iter().enumerate().map(|(i, (n, a))| (self.priority(&n), i, n, a)).collect();
        ord.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut out = vec![];
        for (_, _, name, args) in ord {
            out.push(self.apply_named(&name, args)?);
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// Evaluate a query (HS-8): a step that reads state and emits its result
    /// rows, run on the current state without appending to D. A step that
    /// would change state is not a query and is refused.
    pub fn query(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<V>, String> {
        self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
        let ev = match self.it.construct(name, &args) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("`{name}` is not an event")),
            Err(f) => return Err(why(f)),
        };
        let (s2, outs) = self.it.delta(&self.state, &ev)?;
        if d_sha(&state_json(&s2, self.m)) != self.state_hash {
            return Err(format!("`{name}` changes state; a query only reads"));
        }
        Ok(outs)
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

    /// HC-13: the operations a fabric may compute for state, with the probe
    /// digests that admitted them (`host.Exact`).
    pub fn exact(&mut self, payload_json: &str) -> Result<(), String> {
        self.append(host_event("Exact", payload_json))?;
        self.confirm()
    }

    pub fn wall_time(&mut self, ns: u64) -> Result<(), String> {
        self.append(host_event("WallTime", &format!("{{\"ns\":{ns}}}")))?;
        if declares_event(self.m, "WallTime") {
            self.apply_named("WallTime", vec![Arg { label: Some("ns".into()), e: Expr::Int(ns as i128, 0) }])?;
            self.drain()?;
        }
        self.confirm()
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
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines);
    if keep < lines.len() {
        return Err(format!("D does not verify after entry {keep} of {}", lines.len()));
    }
    let entries: Vec<Entry> = lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries)?;
    Ok((h.seq, h.state_hash.clone(), h.state.clone()))
}

/// The host as of entry k (HS-9): the fold of the first k entries of a
/// verified D, on a volatile store, with every adapter disconnected.
pub fn as_of<'a>(m: &'a Module, text: &str, k: usize) -> Result<Host<'a>, String> {
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    let keep = verify_prefix(&lines);
    if k > keep {
        return Err(format!("D verifies only up to entry {keep}, asked for {k}"));
    }
    let entries: Vec<Entry> = lines[..k].iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let mut h = Host::fresh(m, Box::new(MemStore::default()), vec![])?;
    h.refold(&entries)?;
    Ok(h)
}

/// Path helper for the CLI.
pub fn log_path(p: &str) -> PathBuf {
    PathBuf::from(p)
}
