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

/// HS-4: the number of lines in the longest prefix that links from the
/// genesis hash, each entry's `hash` recomputed over the hashed form of its
/// event (T7). A line that does not parse (a torn write) ends the prefix.
pub fn verify_prefix(lines: &[&str], m: &Module) -> usize {
    let mut prev = D_GENESIS.to_string();
    for (i, l) in lines.iter().enumerate() {
        let Some(e) = Entry::parse_line(l) else { return i };
        if e.seq != i as u64 || e.prev_hash != prev {
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
    for (i, l) in lines.iter().enumerate() {
        let Some(e) = Entry::parse_line(l) else { return i };
        if e.seq != i as u64 || e.prev_hash != prev {
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
    /// `Ok(outputs)` (each output's display text) or `Err(refusal)` (T8: state unchanged).
    pub outcome: Result<Vec<String>, String>,
    /// Display sink (λ output only): the `data` of each emitted `Display(data: text)`.
    pub display: Vec<String>,
    /// The step emitted `Exit`: the host loop ends after this input.
    pub exit: bool,
    /// HC-21: the cause of a refusal (`fuel`, `size`, `value`, `depth`,
    /// `fail` or `trap`), `None` when the input was applied.
    pub cause: Option<&'static str>,
}

/// An entry D owes after the one just folded (HC-21, HC-22): replay
/// requires it next, and a D that ends without it is completed on open.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Owe {
    /// `host.Refused` {cause, seq} after the refused entry at seq; `bind`:
    /// the bound `Refused` input follows it.
    HostRefused { seq: u64, cause: Cause, bind: bool },
    /// the bound input `Refused(seq, cause)`
    BoundRefused { seq: u64, cause: Cause },
    /// the bound input `Restarted(below)` after `host.Restarted`
    BoundRestarted { below: u64 },
}

/// The `req` field of a request or of a result event.
fn req_of(fs: &[(String, V)]) -> Option<u64> {
    fs.iter().find_map(|(f, v)| match (f.as_str(), v) {
        ("req", V::Int(i, _)) => u64::try_from(*i).ok(),
        _ => None,
    })
}

fn refused_payload(cause: Cause, seq: u64) -> String {
    format!("{{\"cause\":\"{}\",\"seq\":{seq}}}", cause.as_str())
}

fn refused_args(cause: Cause, seq: u64) -> Vec<Arg> {
    vec![Arg { label: Some("seq".into()), e: Expr::Int(seq as i128, 0) }, Arg { label: Some("cause".into()), e: Expr::Str(cause.as_str().into(), 0) }]
}

/// What replay re-derived (HC-8, HC-21).
#[derive(Debug)]
pub struct Replayed {
    pub entries: u64,
    pub state_hash: String,
    pub state: Vec<Region>,
    /// Every refused entry: (seq, cause, the refusal text re-derived by replay).
    pub refusals: Vec<(u64, &'static str, String)>,
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
    /// HC-22: the host restarted at entry `floor` (the seq of its
    /// `host.Restarted`). An adapter that issues resource ids issues them
    /// from `floor` upward, so no id issued before is reused within D.
    fn restart(&mut self, _floor: u64) {}
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

/// The declared payload of `event name`, if the module declares it.
fn event_payload<'m>(m: &'m Module, name: &str) -> Option<&'m [(String, Ty)]> {
    m.items.iter().find_map(|i| match i {
        Item::Event { name: n, payload, .. } if n == name => Some(payload.as_slice()),
        _ => None,
    })
}

/// `List[i64]`, the byte type of `core.bytes` (HC-10a).
pub(crate) fn is_bytes(t: &Ty) -> bool {
    matches!(t, Ty::Path(p, a) if p.len() == 1 && p[0] == "List" && a.len() == 1 && matches!(&a[0], Ty::Path(q, b) if q.len() == 1 && q[0] == "i64" && b.is_empty()))
}

/// W-HOST-3 (HC-10a): a result that carries bytes received from outside
/// is declared `List[i64]`, because no `text` holds every byte sequence.
fn bytes_result_fits(m: &Module, kind: &str) -> Result<(), String> {
    let ev = format!("{kind}Done");
    let data = event_payload(m, &ev).and_then(|p| p.iter().find(|(k, _)| k == "data")).map(|(_, t)| t);
    match data {
        Some(Ty::Opt(t)) if is_bytes(t) => Ok(()),
        _ => Err(format!("W-HOST-3 {ev} delivers the bytes received: declare data: List[i64]? and decode text in the program (text.utf8)")),
    }
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
    /// HC-21: the requests (`K` with a `req`, `KDone` declared) emitted by
    /// steps that descend from a `Refused` entry and not yet answered by a
    /// `KDone` with their req: (K, req, the `Refused` entry's seq), in emit
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
    /// HC-21, HC-22: the entry the last folded entry requires next.
    owe: Option<Owe>,
    /// HC-21: the refusals replay re-derived: (seq, cause, text).
    refusals: Vec<(u64, &'static str, String)>,
    /// HC-22: the n of the last `Tick(n)` in D, so the host loop continues it.
    last_tick: u64,
    /// When set, replay records each program entry's outputs (Form JSON).
    outputs: Option<Vec<(u64, Vec<Value>)>>,
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

fn shape_is(p: &[(String, Ty)], want: &[(&str, &str)]) -> bool {
    p.len() == want.len() && p.iter().zip(want).all(|((k, t), (wk, wt))| k == wk && ty_text(t) == *wt)
}

/// W-HOST-2 (HC-24): a module served with `--http` declares the HTTP
/// inputs, the response effect and its result with exactly these shapes;
/// W-HOST-3 (HC-10a): the body is the bytes received.
pub const HTTP_SHAPES: &str = "event HttpRequest(id: u64, method: text, target: text, version: text, names: List[text], values: List[text], body: List[i64]), event HttpMalformed(id: u64, status: u64, why: text), struct HttpRespond {req: u64, status: u64, names: List[text], values: List[text], body: text or List[i64]} and event HttpRespondDone(req: u64, status: text, data: u64?, err: text)";

fn http_fits(m: &Module) -> Result<(), String> {
    let req = event_payload(m, "HttpRequest");
    if let Some(p) = req
        && p.iter().any(|(k, t)| k == "body" && !is_bytes(t))
    {
        return Err("W-HOST-3 HttpRequest delivers the bytes received: declare body: List[i64] and decode text in the program (text.utf8)".into());
    }
    let request = [("id", "u64"), ("method", "text"), ("target", "text"), ("version", "text"), ("names", "List[text]"), ("values", "List[text]"), ("body", "List[i64]")];
    let malformed = [("id", "u64"), ("status", "u64"), ("why", "text")];
    let done = [("req", "u64"), ("status", "text"), ("data", "u64?"), ("err", "text")];
    let respond = m.items.iter().find_map(|i| match i {
        Item::Struct { name, fields, .. } if name == "HttpRespond" => Some(fields.iter().map(|f| (f.name.clone(), ty_text(&f.ty))).collect::<Vec<_>>()),
        _ => None,
    });
    let respond_ok = respond.is_some_and(|f| {
        let want = [("req", "u64"), ("status", "u64"), ("names", "List[text]"), ("values", "List[text]")];
        f.len() == 5 && f[..4].iter().zip(want).all(|((k, t), (wk, wt))| k == wk && t == wt) && f[4].0 == "body" && (f[4].1 == "text" || f[4].1 == "List[i64]")
    });
    if req.is_some_and(|p| shape_is(p, &request)) && event_payload(m, "HttpMalformed").is_some_and(|p| shape_is(p, &malformed)) && event_payload(m, "HttpRespondDone").is_some_and(|p| shape_is(p, &done)) && respond_ok {
        return Ok(());
    }
    Err(format!("W-HOST-2 --http needs {HTTP_SHAPES}"))
}

/// W-HOST-4: a host-delivered event the module declares must have exactly
/// the payload the host delivers (HC-21 `Refused`, HC-22 `Restarted`).
fn host_event_fits(m: &Module, name: &str, want: &[(&str, &str)]) -> Result<(), String> {
    let Some(p) = event_payload(m, name) else { return Ok(()) };
    let fits = p.len() == want.len() && p.iter().zip(want).all(|((k, t), (wk, wt))| k == wk && matches!(t, Ty::Path(x, g) if x.len() == 1 && x[0] == *wt && g.is_empty()));
    if fits {
        return Ok(());
    }
    let shape: Vec<String> = want.iter().map(|(k, t)| format!("{k}: {t}")).collect();
    Err(format!("W-HOST-4 `event {name}` is delivered by the host and must be ({name}({}))", shape.join(", ")))
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
        if adapters.iter().any(|a| a.kind() == "HttpRespond") {
            http_fits(m)?;
        }
        for a in &adapters {
            if declares_struct(m, a.kind()) {
                result_event_fits(m, a.kind())?;
                if a.kind() == "NetRecv" {
                    bytes_result_fits(m, a.kind())?;
                }
                bound.push(a.kind());
            }
        }
        host_event_fits(m, "Refused", &[("seq", "u64"), ("cause", "text")])?;
        host_event_fits(m, "Restarted", &[("below", "u64")])?;
        let mut it = Interp::new(m);
        let state = it.initial_state().map_err(why)?;
        let state_hash = d_sha(&state_pos_json(&state, m));
        Ok(Host { m, it, state, state_hash, prev: D_GENESIS.into(), seq: 0, store, stopped: false, pending: 0, pending_real: 0, confirming: false, durable: None, from_snapshot: None, adapters, bound, queue: Default::default(), descends: Default::default(), drain_max: usize::MAX, admitted: Default::default(), window: 0, closed_tick: 0, offered: vec![], term: 0, task: None, fabrics: vec![], calib: Default::default(), exacts: Default::default(), owe: None, refusals: vec![], last_tick: 0, synced: None, outputs: None })
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
        if declares_event(self.m, "Refused") {
            self.append(host_event("Bind", "{\"event\":\"Refused\",\"kind\":\"host.Refused\"}"))?;
        }
        if declares_event(self.m, "Restarted") {
            self.append(host_event("Bind", "{\"event\":\"Restarted\",\"kind\":\"host.Restarted\"}"))?;
        }
        if self.bound.contains(&"HttpRespond") {
            self.append(host_event("Bind", "{\"event\":\"HttpRequest\",\"kind\":\"host.HttpRequest\"}"))?;
            self.append(host_event("Bind", "{\"event\":\"HttpMalformed\",\"kind\":\"host.HttpMalformed\"}"))?;
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
        let dropped = lines.len() - keep;
        if dropped > 0 || !text.ends_with('\n') {
            let len: usize = lines[..keep].iter().map(|l| l.len() + 1).sum();
            fs.cut(len as u64).map_err(|e| format!("host.StorageFault: cut: {e}"))?;
        }
        let entries: Vec<Entry> = lines[..keep].iter().map(|l| Entry::parse_line(l).unwrap()).collect();
        let mut h = Host::fresh(m, Box::new(fs), adapters)?;
        h.bound_drains(drain_max);
        let from = match snap {
            Some(sn) if h.load_snapshot(sn, &entries) => sn.seq as usize + 1,
            _ => 0,
        };
        h.refold(&entries[from..])?;
        // HC-21: a D that ends after a refused entry, or after its
        // `host.Refused`, is completed before anything else is appended
        h.complete_owed()?;
        let caps = h.store.caps();
        if dropped > 0 {
            h.append(host_event("Truncated", &format!("{{\"dropped\":{dropped}}}")))?;
        }
        h.append(host_event("StorageCaps", &format!("{{\"barrier\":\"{caps}\"}}")))?;
        h.append(host_event("FabricCaps", &fabric_caps()))?;
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
        if declares_event(self.m, "Restarted") {
            self.apply_named("Restarted", vec![Arg { label: Some("below".into()), e: Expr::Int(below as i128, 0) }])?;
        }
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
        if !self.queue.is_empty() || !self.descends.is_empty() || self.seq == 0 || self.task.is_some() || self.owe.is_some() {
            return None;
        }
        Some(Snapshot { seq: self.seq - 1, state_hash: self.state_hash.clone(), bytes: state_pos_json(&self.state, self.m) })
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
        for (ri, (_, fs)) in st.iter_mut().enumerate() {
            for (fi, (_, val, ty)) in fs.iter_mut().enumerate() {
                let Some(j) = v.get(ri).and_then(|r| r.get(fi)) else { return false };
                let Ok(x) = json_expr(j, Some(ty), self.m) else { return false };
                self.it.scopes = vec![HashMap::new()];
        self.it.fuel = crate::interp::fuel_limit(); // HC-7: each input is one run of δ
                match self.it.eval(&x) {
                    Ok(nv) => *val = nv,
                    Err(_) => return false,
                }
            }
        }
        if state_pos_json(&st, self.m) != sn.bytes {
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
        // E.2a: a `host.Input` marks its source and seq admitted only when the
        // program entry it announces follows it (a cut between them admits nothing).
        let mut announced: Option<(String, u64)> = None;
        for e in entries {
            // HC-21, HC-22: the entry the previous one requires
            match self.owe.take() {
                Some(o) => self.meet(o, e)?,
                None if e.kind() == "host.Refused" => return Err(format!("host fault: seq {}: host.Refused follows an entry that replay applies", e.seq)),
                None => {}
            }
            if e.is_host() {
                announced = None;
                self.fold_admission(e, &mut announced);
                self.fold_sched(e).map_err(|x| format!("host fault: seq {}: {x}", e.seq))?;
                if e.kind() == "host.Restarted" {
                    // HC-22: below is the entry's own seq, and the bound
                    // Restarted with the same payload follows when declared
                    let below = serde_json::from_str::<Value>(&e.event).ok().and_then(|v| v["payload"]["below"].as_u64());
                    if below != Some(e.seq) {
                        return Err(format!("host fault: seq {}: host.Restarted must have below {}", e.seq, e.seq));
                    }
                    if declares_event(self.m, "Restarted") {
                        self.owe = Some(Owe::BoundRestarted { below: e.seq });
                    }
                }
                self.rehash();
            } else if let Some((src, q)) = announced.take() {
                self.admitted.insert(src, q);
            }
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
                    self.queue.retain(|q| !(q.kind == k && Some(q.req) == req));
                }
                let under = self.descent(&name, e.seq, &ev);
                self.note_tick(&name, &ev);
                match self.it.delta_r(&self.state, &ev) {
                    Ok((s2, outs)) => {
                        self.state = s2;
                        self.rehash();
                        self.enqueue(&outs, e.seq);
                        self.note_descent(&outs, under);
                        if let Some(o) = &mut self.outputs {
                            o.push((e.seq, outs.iter().map(|x| serde_json::from_str(&crate::interp::form_value(x, self.m).json()).unwrap_or(Value::Null)).collect()));
                        }
                    }
                    Err(r) => {
                        // HC-21: replay re-derives the refusal and its cause
                        let bind = under.is_none() && declares_event(self.m, "Refused");
                        self.owe = Some(Owe::HostRefused { seq: e.seq, cause: r.cause, bind });
                        self.refusals.push((e.seq, r.cause.as_str(), r.text));
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
                self.apply_named("Synced", vec![Arg { label: Some("upto".into()), e: Expr::Int(upto as i128, 0) }])?;
            }
            Ok(())
        });
        self.confirming = false;
        r
    }

    /// HC-21: the `Refused` entry the input `name` (value `ev`) at `seq`
    /// descends from: itself when it is `Refused`; for a result `KDone`
    /// with req r, the one the earliest request K with req r in
    /// `descends` descends from, which it answers (taken off); else none.
    fn descent(&mut self, name: &str, seq: u64, ev: &V) -> Option<u64> {
        if name == "Refused" {
            return Some(seq);
        }
        let k = name.strip_suffix("Done")?;
        let V::Rec(_, fs) = ev else { return None };
        let req = req_of(fs)?;
        let i = self.descends.iter().position(|(qk, qr, _)| qk == k && *qr == req)?;
        self.descends.remove(i).map(|x| x.2)
    }

    /// HC-21: note the requests among the outputs of a step that descends
    /// from the `Refused` entry `under`.
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
        }
    }

    /// Admit the result event of one request (HC-3).
    fn answer(&mut self, kind: &str, req: u64, r: EffectResult) -> Result<Vec<Admitted>, String> {
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
    /// input; a refusal adds the entries of HC-21 after it.
    fn apply_named(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
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
        let (outcome, next) = match self.it.delta_r(&self.state, &ev) {
            Ok((s2, outs)) => (Ok(outs), Some(s2)),
            Err(r) => (Err(r), None),
        };
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
        let under = self.descent(name, e.seq, &ev);
        let (mut display, mut exit) = (vec![], false);
        if let Ok(outs) = &outcome {
            self.enqueue(outs, e.seq);
            self.note_descent(outs, under);
            for o in outs {
                match o {
                    V::Rec(n, fs) if n == "Display" => {
                        if let Some((_, V::Text(t))) = fs.iter().find(|(f, _)| f == "data") {
                            display.push(t.to_string());
                        }
                    }
                    V::Rec(n, _) if n == "Exit" => exit = true,
                    _ => {}
                }
            }
        }
        let refused = outcome.as_ref().err().map(|r| r.cause);
        let outcome = outcome.map(|os| os.iter().map(|o| o.show()).collect()).map_err(|r| r.text);
        let mut out = vec![Admitted { seq: e.seq, event: ev.show(), outcome, display, exit, cause: refused.map(Cause::as_str) }];
        if let Some(cause) = refused {
            let bind = under.is_none() && declares_event(self.m, "Refused");
            out.extend(self.refused(e.seq, cause, bind)?);
        }
        Ok(out)
    }

    /// HC-21: after the refused entry at `seq`, `host.Refused` {cause, seq},
    /// then (`bind`) the bound input `Refused(seq, cause)`. A refused step
    /// that is a `Refused` or descends from one ([`Host::descent`]) gets
    /// `host.Refused` only, so no `Refused` leads to another.
    fn refused(&mut self, seq: u64, cause: Cause, bind: bool) -> Result<Vec<Admitted>, String> {
        self.append(host_event("Refused", &refused_payload(cause, seq)))?;
        if bind { self.apply_named("Refused", refused_args(cause, seq)) } else { Ok(vec![]) }
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

    /// Replay (HC-21, HC-22): `e` must be the entry `o` the previous entry requires.
    fn meet(&mut self, o: Owe, e: &Entry) -> Result<(), String> {
        let want = match o {
            Owe::HostRefused { seq, cause, bind } => {
                if bind {
                    self.owe = Some(Owe::BoundRefused { seq, cause });
                }
                host_event("Refused", &refused_payload(cause, seq))
            }
            Owe::BoundRefused { seq, cause } => self.bound_json("Refused", &refused_args(cause, seq))?,
            Owe::BoundRestarted { below } => self.bound_json("Restarted", &[Arg { label: Some("below".into()), e: Expr::Int(below as i128, 0) }])?,
        };
        if e.event != want {
            return Err(format!("host fault: seq {}: D has {} where {want} must follow", e.seq, e.event));
        }
        Ok(())
    }

    /// Complete an entry D owes at its end (HC-21, HC-22): a host stopped
    /// after appending a refused entry, or a `host.Refused` or
    /// `host.Restarted` without its bound input.
    fn complete_owed(&mut self) -> Result<Vec<Admitted>, String> {
        match self.owe.take() {
            None => Ok(vec![]),
            Some(Owe::HostRefused { seq, cause, bind }) => self.refused(seq, cause, bind),
            Some(Owe::BoundRefused { seq, cause }) => self.apply_named("Refused", refused_args(cause, seq)),
            Some(Owe::BoundRestarted { below }) => self.apply_named("Restarted", vec![Arg { label: Some("below".into()), e: Expr::Int(below as i128, 0) }]),
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
            out.extend(self.apply_named(&name, args)?);
            out.extend(self.drain()?);
        }
        self.confirm()?;
        Ok(out)
    }

    /// Admit one input given as an event name and arguments (adapters and
    /// the host loop), its effects, then the batch barrier.
    pub fn admit_event(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
        let mut out = self.apply_named(name, args)?;
        out.extend(self.drain()?);
        self.confirm()?;
        Ok(out)
    }

    /// Admit one input of a batch (group commit, D-2): append its entry, run
    /// δ and queue its effects, without a barrier. [`Host::settle`] ends the
    /// batch.
    pub fn apply_input(&mut self, name: &str, args: Vec<Arg>) -> Result<Vec<Admitted>, String> {
        self.apply_named(name, args)
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
            out.extend(self.apply_named(&name, args)?);
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
            "host.MainStart" | "host.MainInput" | "host.Step" | "host.MainEnd" | "host.Calibrated" | "host.Exact" | "host.Placed" => serde_json::from_str(&e.event).map_err(|x| x.to_string())?,
            _ => return Ok(()),
        };
        let p = v["payload"].clone();
        match k.as_str() {
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
        let cfg = crate::conc::RunCfg { seed: start["seed"].as_u64().unwrap_or(0), choices, links, max_ticks: start["max_ticks"].as_u64().unwrap_or(100_000), par_mode: crate::conc::ParMode::Seq, restarts: None, restart_tree: None, events };
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
            crate::conc::Outcome::Refused(r) => format!("refused: {r}"),
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
    /// the scheduler re-derives that differs from D refuses the entry.
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
    /// refused unless its tick is after the last logged step, so the input
    /// cannot rewrite steps D already holds.
    pub fn main_input(&mut self, tick: u64, partition: bool, a: &str, b: &str) -> Result<(), String> {
        self.main_entry("MainInput", serde_json::json!({"a": a, "b": b, "partition": partition, "tick": tick}))
    }

    /// HC-6: feed a program event to the live task (`host.MainInput`
    /// {event, tick}): at its tick the scheduler logs it as `Input(..)` and
    /// delivers it to every actor that handles it (`conc::RunCfg.events`).
    /// Refused like a link input when late or when it would change a step
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
    /// execute is refused before anything is logged. A registered fabric
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
    /// fabrics can execute them; a placement elsewhere is refused before
    /// it is logged.
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
            let a = self.apply_named(&x.name, x.args)?;
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
    /// it, and a lower or equal term is refused. Requests emitted under the
    /// previous term with no result in D get `failed`/`interrupted` (HC-3).
    pub fn take_role(&mut self, term: u64, leader: &str) -> Result<(), String> {
        if term <= self.term {
            return Err(format!("host.Term: term {term} is not above the current term {}", self.term));
        }
        // HC-21: a refusal the previous admitter logged without its
        // `host.Refused` (or bound `Refused`) is completed first; it belongs
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
        if d_sha(&state_pos_json(&s2, self.m)) != self.state_hash {
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
    replay_report(m, text).map(|r| (r.entries, r.state_hash, r.state))
}

/// [`replay`] with the refusals it re-derived (HC-21): each refused entry's
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
    Ok(Replayed { entries: h.seq, state_hash: h.state_hash.clone(), state: h.state.clone(), refusals: std::mem::take(&mut h.refusals) })
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

/// Path helper for the CLI.
pub fn log_path(p: &str) -> PathBuf {
    PathBuf::from(p)
}
