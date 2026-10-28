//! The `rce` command line and language server (spec section 10, book/12-tooling.md).
//! Everything here is deterministic: the same files and arguments give the
//! same output bytes. No network access and no model is consulted.
use crate::ast::{Item, Module};
use crate::interp::{Interp, Region, V, form_json, why};
use crate::parse::{Tok, lex, parse};
use crate::{check, interp};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};

/// Exit codes of spec 10.1.
pub const OK: i32 = 0;
pub const REFUSED: i32 = 1;
pub const USAGE: i32 = 2;

/// Diagnostics of one text as (1-based line, message). Empty means admitted.
/// `known` is the set of module paths `use` may resolve to.
pub fn diagnostics(src: &str, known: &HashSet<String>) -> Vec<(usize, String)> {
    let errs = match parse(src) {
        Err(e) => vec![e],
        Ok(m) => {
            // the program as the tools run it: `use` imports linked in from
            // the built-in library (4.1)
            let lk = crate::link::link(&m, &[]);
            let mut c = check::Checker::new();
            c.imported_fns = lk.imported_fns;
            c.imported_types = lk.imported_types;
            for e in lk.errs {
                c.e(e);
            }
            c.check_module(&lk.module, known);
            if c.errs.is_empty() {
                interp::admit(&lk.module)
            } else {
                c.errs
            }
        }
    };
    errs.into_iter().map(|e| (line_of(&e), e)).collect()
}

/// The line number a diagnostic starts with ("line N: ..."), or 1.
pub fn line_of(e: &str) -> usize {
    e.strip_prefix("line ")
        .and_then(|r| r.split(':').next())
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or(1)
        .max(1)
}

fn show_state(s: &[Region]) -> String {
    show_state_redacted(s, &[])
}

/// Like `show_state`, but each field named in `redact` prints as a short
/// SHA-256 digest of its value instead of the value. Used by `rce host replay
/// --redact` so private keys and ratchet state stay in the device log and off
/// the terminal; the digest still lets two replays be compared.
fn show_state_redacted(s: &[Region], redact: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    s.iter()
        .map(|(r, fs)| {
            let f: Vec<String> = fs
                .iter()
                .map(|(n, v, _)| {
                    if redact.contains(&n.as_str()) {
                        let h: String = Sha256::digest(v.show().as_bytes()).iter().take(6).map(|b| format!("{b:02x}")).collect();
                        format!("{n} = <redacted sha256:{h}>")
                    } else {
                        format!("{n} = {}", v.show())
                    }
                })
                .collect();
            format!("{r} {{ {} }}", f.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Outcome of `rce run`: the printed lines and whether the run was refused.
#[derive(Debug, PartialEq)]
pub struct RunOut {
    pub lines: Vec<String>,
    pub refused: bool,
}

/// `rce run FILE EVENTS`: apply δ to the initial state once per event, in
/// order, printing the state and outputs after each. Stops at the first
/// refused event. Without events, runs `fn main()` and prints its outputs.
/// The module must be admitted first (as by `rce check`).
pub fn run(src: &str, events: Option<&str>) -> Result<RunOut, String> {
    let d = diagnostics(src, &HashSet::new());
    if !d.is_empty() {
        return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
    }
    let m = crate::program(src)?;
    let mut it = Interp::new(&m);
    let mut lines = vec![];
    let Some(evs) = events else {
        // fn main() runs on the tick runtime of chapter C (tasks, channels,
        // actors, par, budgets, deadlines)
        let r = crate::conc::run_main(&m, &crate::conc::RunCfg::default())
            .map_err(|e| format!("{e}: give events, or declare `fn main()`"))?;
        for o in &r.outputs {
            lines.push(format!("emit {}", o.show()));
        }
        let refused = match r.outcome {
            crate::conc::Outcome::Done => false,
            crate::conc::Outcome::Deadlock => {
                lines.push("deadlock".into());
                true
            }
            crate::conc::Outcome::Refused(e) => {
                lines.push(format!("refused: {e}"));
                true
            }
        };
        return Ok(RunOut { lines, refused });
    };
    // The event list uses the syntax of an `example` line (3.x): parse it as one.
    let probe =
        parse(&format!("{src}\nexample {evs} -> 0\n")).map_err(|e| format!("events: {e}"))?;
    let Some(Item::Example(x)) = probe.items.last() else {
        return Err("events: not an event list".into());
    };
    let mut s = it.initial_state().map_err(why)?;
    lines.push(format!("0 (initial) {}", show_state(&s)));
    for (k, (evn, args)) in x.events.iter().enumerate() {
        it.scopes = vec![HashMap::new()];
        let ev: V = match it.construct(evn, args) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(format!("events: `{evn}` is not an event")),
            Err(f) => return Err(format!("events: {}", why(f))),
        };
        match it.delta(&s, &ev) {
            Ok((s2, outs)) => {
                s = s2;
                lines.push(format!("{} {} {}", k + 1, ev.show(), show_state(&s)));
                for o in outs {
                    lines.push(format!("  emit {}", o.show()));
                }
            }
            Err(r) => {
                lines.push(format!("{} {} refused: {r}", k + 1, ev.show()));
                return Ok(RunOut {
                    lines,
                    refused: true,
                });
            }
        }
    }
    Ok(RunOut {
        lines,
        refused: false,
    })
}

/// Number of `example` lines and `check` items in a module.
pub fn count_tests(m: &Module) -> (usize, usize) {
    let (mut e, mut c) = (0, 0);
    for i in &m.items {
        match i {
            Item::Example(_) => e += 1,
            Item::Check(_) => c += 1,
            Item::Intent(_, xs, cs) => {
                e += xs.len();
                c += cs.len();
            }
            _ => {}
        }
    }
    (e, c)
}

/// `rce emit TARGET FILE`: `form` prints the Form JSON; `rust`, `python`
/// and `js` print a self-contained program that prints the module's
/// observations (10.5).
pub fn emit(target: &str, src: &str) -> Result<String, String> {
    match target {
        "form" => {
            let d = diagnostics(src, &HashSet::new());
            if !d.is_empty() {
                return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
            }
            Ok(form_full(&crate::program(src)?)? + "\n")
        }
        "rust" | "python" | "js" | "scala" => {
            let d = diagnostics(src, &HashSet::new());
            if !d.is_empty() {
                return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
            }
            let lang = crate::emit::Lang::parse(target).expect("target");
            crate::emit::emit(&crate::program(src)?, lang)
        }
        t => Err(format!(
            "unknown emit target `{t}` (form, rust, python, js, scala)"
        )),
    }
}

fn toks(src: &str) -> Result<Vec<Tok>, String> {
    Ok(lex(src)?.into_iter().map(|(t, _)| t).collect())
}

/// Bracket depth change of one line, skipping strings and the comment.
fn depth_delta(line: &str) -> i64 {
    let (mut d, mut in_str, mut esc) = (0i64, false, false);
    for c in line.chars() {
        if in_str {
            if esc {
                esc = false
            } else if c == '\\' {
                esc = true
            } else if c == '"' {
                in_str = false
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '#' => break,
            '(' | '[' | '{' => d += 1,
            ')' | ']' | '}' => d -= 1,
            _ => {}
        }
    }
    d
}

/// `rce fmt`: the canonical layout of 10.6. Indentation becomes 4 spaces
/// per block level, trailing spaces go, runs of blank lines become one, the
/// file ends in exactly one line end. The token sequence is unchanged; if it
/// would change, the text is refused and left as it was.
pub fn fmt_text(src: &str) -> Result<String, String> {
    let before = toks(src)?;
    let mut out: Vec<String> = vec![];
    let mut pending: Vec<&str> = vec![]; // comment-only lines awaiting a level
    let mut stack = vec![0usize];
    let mut depth = 0i64;
    let mut shift = 0i64;
    let mut blank = false;
    let lead = |l: &str| l.len() - l.trim_start_matches(' ').len();
    for raw in src.lines() {
        let l = raw.trim_end();
        let t = l.trim_start();
        if depth > 0 {
            let w = (lead(l) as i64 + shift).max(0) as usize;
            out.push(format!("{}{}", " ".repeat(w), t));
            depth += depth_delta(l);
            continue;
        }
        if t.is_empty() {
            blank = !out.is_empty() || !pending.is_empty();
            continue;
        }
        if t.starts_with('#') {
            if blank {
                pending.push("");
                blank = false;
            }
            pending.push(t);
            continue;
        }
        let w = lead(l);
        if w > *stack.last().unwrap() {
            stack.push(w);
        } else {
            while w < *stack.last().unwrap() {
                stack.pop();
            }
            if w != *stack.last().unwrap() {
                return Err(format!("W-LEX-2 inconsistent dedent: {raw:?}"));
            }
        }
        let ind = 4 * (stack.len() - 1);
        shift = ind as i64 - w as i64;
        if blank && !out.is_empty() && pending.is_empty() {
            out.push(String::new());
        }
        blank = false;
        for p in pending.drain(..) {
            if p.is_empty() {
                if !out.is_empty() {
                    out.push(String::new())
                }
            } else {
                out.push(format!("{}{}", " ".repeat(ind), p));
            }
        }
        out.push(format!("{}{}", " ".repeat(ind), t));
        depth = depth_delta(l);
    }
    for p in pending.drain(..) {
        if p.is_empty() {
            out.push(String::new())
        } else {
            out.push(p.to_string())
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    let text = if out.is_empty() {
        String::new()
    } else {
        out.join("\n") + "\n"
    };
    if toks(&text)? != before {
        return Err("formatting would change the token sequence; left unchanged".into());
    }
    Ok(text)
}

// ---------------- language server (10.7) ----------------

/// Read one LSP message (Content-Length framing). None at end of input.
pub fn read_msg(r: &mut impl BufRead) -> Option<Value> {
    let mut len = None;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 {
            return None;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.strip_prefix("Content-Length:") {
            len = v.trim().parse::<usize>().ok();
        }
    }
    let mut buf = vec![0; len?];
    r.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
}

pub fn write_msg(w: &mut impl Write, v: &Value) {
    let s = v.to_string();
    let _ = write!(w, "Content-Length: {}\r\n\r\n{s}", s.len());
    let _ = w.flush();
}

/// State of the language server: the open documents.
#[derive(Default)]
pub struct Server {
    pub docs: HashMap<String, String>,
    pub shutdown: bool,
}

impl Server {
    fn known(&self) -> HashSet<String> {
        self.docs
            .values()
            .filter_map(|t| parse(t).ok()?.path.map(|p| p.join(".")))
            .collect()
    }

    fn publish(&self, uri: &str) -> Value {
        let text = self.docs.get(uri).cloned().unwrap_or_default();
        let diags: Vec<Value> = if self.docs.contains_key(uri) {
            diagnostics(&text, &self.known())
                .into_iter()
                .map(|(l, m)| {
                    json!({"range": {"start": {"line": l - 1, "character": 0},
                                     "end": {"line": l, "character": 0}},
                           "severity": 1, "source": "rce", "message": m})
                })
                .collect()
        } else {
            vec![]
        };
        json!({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
               "params": {"uri": uri, "diagnostics": diags}})
    }

    /// Handle one message; returns the messages to send back, and Some(code)
    /// when the server must exit.
    pub fn handle(&mut self, m: &Value) -> (Vec<Value>, Option<i32>) {
        let method = m["method"].as_str().unwrap_or("");
        let id = m.get("id").cloned();
        let p = &m["params"];
        let uri = p["textDocument"]["uri"].as_str().unwrap_or("").to_string();
        let reply = |r: Value| json!({"jsonrpc": "2.0", "id": id, "result": r});
        match method {
            "initialize" => (
                vec![reply(json!({
                    "capabilities": {"textDocumentSync": 1, "documentFormattingProvider": true,
                                     "hoverProvider": true, "definitionProvider": true,
                                     "documentSymbolProvider": true,
                                     "completionProvider": {"triggerCharacters": ["."]}},
                    "serverInfo": {"name": "rce", "version": env!("CARGO_PKG_VERSION")}}))],
                None,
            ),
            "textDocument/didOpen" => {
                let t = p["textDocument"]["text"].as_str().unwrap_or("").to_string();
                self.docs.insert(uri.clone(), t);
                (vec![self.publish(&uri)], None)
            }
            "textDocument/didChange" => {
                if let Some(t) = p["contentChanges"]
                    .as_array()
                    .and_then(|a| a.last())
                    .and_then(|c| c["text"].as_str())
                {
                    self.docs.insert(uri.clone(), t.to_string());
                }
                (vec![self.publish(&uri)], None)
            }
            "textDocument/didClose" => {
                self.docs.remove(&uri);
                (vec![self.publish(&uri)], None)
            }
            "textDocument/formatting" => {
                let t = self.docs.get(&uri).cloned().unwrap_or_default();
                let edits = match fmt_text(&t) {
                    Ok(n) if n != t => {
                        let lines = t.split('\n').count();
                        json!([{"range": {"start": {"line": 0, "character": 0},
                                          "end": {"line": lines, "character": 0}},
                                "newText": n}])
                    }
                    _ => json!([]),
                };
                (vec![reply(edits)], None)
            }
            "textDocument/hover"
            | "textDocument/definition"
            | "textDocument/completion"
            | "textDocument/documentSymbol" => {
                let ix =
                    crate::lsp::Index::new(self.docs.get(&uri).map(|s| s.as_str()).unwrap_or(""));
                let line = p["position"]["line"].as_u64().unwrap_or(0) as usize;
                let col = p["position"]["character"].as_u64().unwrap_or(0) as usize;
                let r = match method {
                    "textDocument/hover" => ix.hover(line, col),
                    "textDocument/definition" => ix.definition(&uri, line, col),
                    "textDocument/completion" => ix.completion(line, col),
                    _ => ix.symbols(),
                };
                (vec![reply(r)], None)
            }
            "shutdown" => {
                self.shutdown = true;
                (vec![reply(Value::Null)], None)
            }
            "exit" => (vec![], Some(if self.shutdown { 0 } else { 1 })),
            _ if id.is_some() => (
                vec![json!({"jsonrpc": "2.0", "id": id,
                            "error": {"code": -32601, "message": format!("method not found: {method}")}})],
                None,
            ),
            _ => (vec![], None),
        }
    }
}

/// Serve LSP over the given streams until `exit` or end of input.
pub fn serve(r: &mut impl BufRead, w: &mut impl Write) -> i32 {
    let mut s = Server::default();
    while let Some(m) = read_msg(r) {
        let (outs, code) = s.handle(&m);
        for o in outs {
            write_msg(w, &o);
        }
        if let Some(c) = code {
            return c;
        }
    }
    if s.shutdown { 0 } else { 1 }
}

/// Version of the reference trace format (10.9, informative).
pub const TRACE_FORMAT: &str = "rce-trace/2";

/// `rce trace --json FILE [EVENTS]`: the reference trace of a Form (10.9).
/// Without EVENTS, one trace per `example` of the module (its log D, 6.1);
/// with EVENTS, one trace of that event list. Each step carries the
/// canonical D fields `seq`, `event`, `state_hash`, `prev_hash`, `hash`,
/// the derived `outcome`, and `hashed`, the positional forms of the event
/// and the state that the two hashes are over (6.1, T7); a trace stops at
/// the first refused event.
/// Keys are sorted and the JSON is compact, so equal traces are equal bytes.
pub fn trace(file: &str, src: &str, events: Option<&str>) -> Result<String, String> {
    match events {
        None => trace_streams(file, src, &[]),
        Some(evs) => trace_streams(file, src, &[evs]),
    }
}

/// `rce trace --json FILE --streams PATH`: one trace per event list of
/// `streams` (one list per line, example syntax), traced in one run of the
/// checker; the module's own examples are not traced. With no stream, the
/// module's examples are traced (the plain form). A stream's expectation is
/// never checked: these are event lists, not examples (used by
/// `scripts/xarch-streams.py` for the determinism streams of T37).
pub fn trace_streams(file: &str, src: &str, streams: &[&str]) -> Result<String, String> {
    let d = diagnostics(src, &HashSet::new());
    if !d.is_empty() {
        return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
    }
    let events: Option<&str> = streams.first().copied();
    let (m, only) = if streams.is_empty() {
        (crate::program(src)?, None)
    } else {
        let mut text = src.to_string();
        for evs in streams {
            text.push_str(&format!("\nexample {evs} -> 0"));
        }
        text.push('\n');
        let probe = parse(&text).map_err(|e| format!("events: {e}"))?;
        let n = probe.items.iter().filter(|i| matches!(i, Item::Example(_))).count();
        if n < streams.len() {
            return Err("events: not an event list".into());
        }
        // the first appended example: everything from its line on is a stream
        let first = probe
            .items
            .iter()
            .filter_map(|i| if let Item::Example(x) = i { Some(x.line) } else { None })
            .rev()
            .nth(streams.len() - 1)
            .ok_or("events: not an event list")?;
        (probe, Some(first))
    };
    let form: serde_json::Value =
        serde_json::from_str(&crate::interp::form_json(&m)).map_err(|e| format!("form json: {e}"))?;
    let mut traces = vec![];
    for l in form["logs"].as_array().into_iter().flatten() {
        let line = l["example_line"].as_u64().unwrap_or(0) as usize;
        if only.is_some_and(|o| line < o) {
            continue;
        }
        let steps: Vec<serde_json::Value> = l["D"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|e| {
                serde_json::json!({
                    "seq": e["seq"], "event": e["event"], "state_hash": e["state_hash"],
                    "prev_hash": e["prev_hash"], "hash": e["hash"], "outcome": e["outcome"],
                    "hashed": e["hashed"],
                })
            })
            .collect();
        let source = match only {
            Some(o) if streams.len() > 1 => format!("stream@{}", line - o),
            Some(_) => "events".to_string(),
            None => format!("example@{line}"),
        };
        traces.push(serde_json::json!({ "source": source, "steps": steps }));
    }
    if let (Some(evs), true) = (events, traces.is_empty()) {
        return Err(format!("events: `{evs}` gives no trace (no state, or an unknown event)"));
    }
    let name = std::path::Path::new(file).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let mut v = serde_json::json!({
        "format": TRACE_FORMAT,
        "file": name,
        "events": if streams.len() > 1 { None } else { events },
        "genesis": crate::interp::D_GENESIS,
        "traces": traces,
    });
    if streams.len() > 1 {
        v["streams"] = serde_json::json!(streams.len());
    }
    Ok(crate::interp::canon_json(&v))
}

/// Usage of the language commands of `rce` (10.2 to 10.7).
pub const USAGE_TEXT: &str = "usage:
  rce check FILE|DIR...          parse, type check and admit; one verdict per file (a DIR is a package: its .rce files)
  rce run FILE [EVENTS]          apply EVENTS (example syntax, e.g. \"Inc, Inc, Dec\"), or run fn main()
  rce test FILE|DIR...           run every example and check; report per file
  rce trace --json FILE [EVENTS] reference trace (seq, event, state_hash per step) of the examples, or of EVENTS
  rce trace --json FILE --streams PATH  one trace per event list in PATH (one per line), in one run
  rce emit TARGET FILE           TARGET: form (Form JSON), rust, python, js or scala (a program printing its observations)
  rce fmt [--check] FILE...      rewrite files in canonical layout (--check: only report)
  rce fmt --nameless FILE        print the name-free projection: every declared identifier replaced by its positional name (10.6)
  rce dim [--fuel N] [--control] [--json] FILE  behaviours a program can be in and the redundancy of its state (6.10, 10.11); --control adds dim_c and shape_ctl
  rce equiv [--fuel N] [--json] A B      whether two programs have the same behaviour on the explored alphabet (6.10, T58), with a distinguishing event sequence when not
  rce change [--fuel N] OLD NEW          classify a change: api-diff rows (F.4.3) and the behaviour check; erratum when every behaviour is kept, meaning change otherwise (C.15, CC-2)
  rce lib [MODULE]               the built-in standard library: every module `use` can name (4.1) and the hash that pins it, or the source of one module
  rce hash FILE|DIR...           the definition hash (F.4.3) of every named item of each module: HASH, kind, name, and `pub` for an export
  rce manifest DIR               write the hash-pinned manifest of a directory (F.5.3): every .rce file as a module pinned by its hash, canonical JSON
  rce manifest-check MANIFEST [--advisories FILE] [--key ISSUER=HEXPUBKEY]...  verify a hash-pinned manifest (F.5.3, W-PIN-1) and list pinned hashes that carry a signed advisory
  rce api-diff OLD NEW           compare the exports of two module sets (FILE or DIR each); one row per change, compatible or breaking; status 1 if any is breaking
  rce v09-status                 the Appendix F rules this checker implements and the ones it refuses by version
  rce lsp                        language server on stdin/stdout
  rce index [--vectors N] FILE|DIR...  brick catalog (rce-catalog/1 JSON): every exported fn, type key, SHA-256, verdict, D.7 Form IR term
  rce search CATALOG QUERY       bricks of type QUERY (e.g. (A, B) -> R; a, b, ... are variables), best first
  rce unform FORM.json           the canonical source of a Form's terms (inverse of `rce emit form`)
  rce host run FILE --log D [--fs DIR] [--blob DIR] [--media DIR] [--wasm DIR] [--net HOST:PORT]... [--chan HOST:PORT [--tls] [--peer HOST:PORT[=KEY]]...] [--pty] [EVENTS]  run on the host runtime (Appendix E): admit EVENTS, or one batch per stdin line, into the durable log D
  rce host serve FILE --log D [--fs DIR] [--blob DIR] [--media DIR] [--wasm DIR] [--net HOST:PORT]... [--chan HOST:PORT [--tls] [--peer HOST:PORT[=KEY]]...] [--http HOST:PORT] [--pty]  the host loop (W1d): stdin lines are inputs (event syntax, else Input(data) when declared), Tick(n) every frame when declared, each HTTP request is HttpRequest or HttpMalformed answered by HttpRespond (--http, Appendix E HC-24), Display(data) is written to stdout, Exit or SIGTERM ends the loop
  rce host tls-id FILE --log D   the Ed25519 key of this host's TLS identity (kept in D.tlskey; peers pin it with --peer ADDR=KEY under --tls)
  rce host replay FILE --log D [--redact F,..]   replay D with every adapter disconnected (HC-8) and re-derive every state hash
  rce db D CMD [ARGS]            RCEdb (W3), serverless and in-process: the database is the log D. CMD: insert T K ROW, delete T K,
                                 link S L D, unlink S L D, schema T COLS (name:text|i64|bool,...), index T COL, scan T [START],
                                 lookup T COL VALUE, edges S, describe, dump, emit sqlite|postgres|duckdb|cypher, as-of N scan T [START]";

/// The language commands of `rce`, given the arguments after the program
/// name; returns the exit status (10.1). Output goes to `out` and `err`;
/// `rce lsp` reads `input`.
pub fn main(a: &[String], input: &mut dyn BufRead, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match run_main(a, input, out, err) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            USAGE
        }
    }
}

fn read_file(p: &str) -> Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))
}

/// The `.rce` files of a package directory (4.1), recursively, sorted.
/// A directory argument names exactly these files, so the program is
/// still the set of modules given together; nothing outside it is read.
pub fn package_files(dir: &std::path::Path) -> Result<Vec<String>, String> {
    let mut v = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rce") {
                v.push(p.to_string_lossy().into_owned());
            }
        }
    }
    v.sort();
    Ok(v)
}

/// FILE and DIR arguments as the list of module files (a DIR is a package).
fn expand(ps: &[String]) -> Result<Vec<String>, String> {
    let mut v = vec![];
    for p in ps {
        if std::path::Path::new(p).is_dir() {
            v.extend(package_files(std::path::Path::new(p))?);
        } else {
            v.push(p.clone());
        }
    }
    Ok(v)
}

fn run_main(a: &[String], input: &mut dyn BufRead, out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, String> {
    let rest = if a.is_empty() { &[][..] } else { &a[1..] };
    let load = |ps: &[String]| -> Result<Vec<(String, String)>, String> {
        expand(ps)?.iter().map(|p| Ok((p.clone(), read_file(p)?))).collect()
    };
    let w = |e: std::io::Error| e.to_string();
    // `rce SUB --help` / `-h` anywhere prints the usage (10.1: exit 0)
    if a.len() > 1 && rest.iter().any(|x| x == "--help" || x == "-h") {
        writeln!(out, "{USAGE_TEXT}").map_err(w)?;
        return Ok(OK);
    }
    Ok(match a.first().map(String::as_str) {
        Some("check") if !rest.is_empty() => {
            let mut bad = 0;
            for (p, v) in crate::check_program(&load(rest)?, false) {
                match v {
                    Ok(()) => writeln!(out, "ok {p}").map_err(w)?,
                    Err(e) => {
                        bad += 1;
                        writeln!(out, "REFUSED {p}: {e}").map_err(w)?
                    }
                }
            }
            if bad > 0 { REFUSED } else { OK }
        }
        Some("run") if rest.len() == 1 || rest.len() == 2 => {
            match run(&read_file(&rest[0])?, rest.get(1).map(String::as_str)) {
                Ok(r) => {
                    for l in &r.lines {
                        writeln!(out, "{l}").map_err(w)?;
                    }
                    if r.refused { REFUSED } else { OK }
                }
                Err(e) => {
                    writeln!(out, "REFUSED {}: {e}", rest[0]).map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("test") if !rest.is_empty() => {
            let files = load(rest)?;
            let (mut pass, mut fail) = (0, 0);
            for ((p, v), (_, src)) in crate::check_program(&files, false).into_iter().zip(&files) {
                let (e, c) = parse(src).map(|m| count_tests(&m)).unwrap_or((0, 0));
                match v {
                    Ok(()) => {
                        pass += 1;
                        writeln!(out, "ok {p} ({e} examples, {c} checks)").map_err(w)?
                    }
                    Err(x) => {
                        fail += 1;
                        writeln!(out, "FAIL {p}: {x}").map_err(w)?
                    }
                }
            }
            writeln!(out, "{pass} passed, {fail} failed").map_err(w)?;
            if fail > 0 { REFUSED } else { OK }
        }
        Some("unform") if rest.len() == 1 => match unform(&read_file(&rest[0])?) {
            Ok(t) => {
                write!(out, "{t}").map_err(w)?;
                OK
            }
            Err(e) => {
                writeln!(out, "REFUSED {}: {e}", rest[0]).map_err(w)?;
                REFUSED
            }
        },
        Some("db") if rest.len() >= 2 => db_cmd(rest, out)?,
        Some("host") if rest.len() >= 4 && rest[2] == "--log" && rest[0] != "serve" || rest.len() >= 4 && rest[0] == "serve" && rest[2] == "--log" => {
            host_cmd(rest, input, out, err).map_err(|e| e.to_string())?
        }
        Some("trace") if rest.len() == 4 && rest[0] == "--json" && rest[2] == "--streams" => {
            let text = read_file(&rest[3])?;
            let lists: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).collect();
            match trace_streams(&rest[1], &read_file(&rest[1])?, &lists) {
                Ok(t) => {
                    writeln!(out, "{t}").map_err(w)?;
                    OK
                }
                Err(e) => {
                    writeln!(out, "REFUSED {}: {e}", rest[1]).map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("trace") if (rest.len() == 2 || rest.len() == 3) && rest[0] == "--json" => {
            match trace(&rest[1], &read_file(&rest[1])?, rest.get(2).map(String::as_str)) {
                Ok(t) => {
                    writeln!(out, "{t}").map_err(w)?;
                    OK
                }
                Err(e) => {
                    writeln!(out, "REFUSED {}: {e}", rest[1]).map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("change") if rest.len() >= 2 => {
            // rce change [--fuel N] OLD NEW   (10.11, CC-2): the C.15 classification of a change
            let (mut fuel, mut files) = (100_000usize, Vec::new());
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--fuel" if i + 1 < rest.len() => {
                        fuel = rest[i + 1].parse().map_err(|_| "--fuel N: N is a count of states".to_string())?;
                        i += 2;
                    }
                    f if !f.starts_with("--") => {
                        files.push(f.to_string());
                        i += 1;
                    }
                    other => return Err(format!("change: unexpected argument `{other}`")),
                }
            }
            if files.len() != 2 {
                return Err("change: OLD and NEW are needed".into());
            }
            let a = parse(&read_file(&files[0])?).map_err(|e| format!("{}: {e}", files[0]))?;
            let b = parse(&read_file(&files[1])?).map_err(|e| format!("{}: {e}", files[1]))?;
            let ha = crate::apidiff::program_hash(std::slice::from_ref(&a));
            let hb = crate::apidiff::program_hash(std::slice::from_ref(&b));
            writeln!(out, "program             {ha}").map_err(w)?;
            writeln!(out, "                    {hb}").map_err(w)?;
            if ha == hb {
                writeln!(out, "verdict             same program  (equal program hash: a renaming or a reformatting)").map_err(w)?;
                return Ok(OK);
            }
            let rows = crate::apidiff::diff(std::slice::from_ref(&a), std::slice::from_ref(&b));
            let breaking = rows.iter().any(|r| r.breaking);
            for r in &rows {
                writeln!(out, "api                 {}", r.show()).map_err(w)?;
            }
            if rows.is_empty() {
                writeln!(out, "api                 no export changed").map_err(w)?;
            }
            let beh = match crate::dim::equiv(&a, &b, fuel) {
                Ok(e) => {
                    write!(out, "{}", crate::dim::equiv_report(&e).lines().map(|l| format!("behaviour {l}\n")).collect::<String>()).map_err(w)?;
                    Some(e)
                }
                Err(e) => {
                    writeln!(out, "behaviour           not compared ({e})").map_err(w)?;
                    None
                }
            };
            let (verdict, status) = match (&beh, breaking) {
                (Some(e), _) if !e.equivalent => ("meaning change  the behaviour of the program changed", REFUSED),
                (Some(e), false) if e.exact => ("erratum  every export compatible and every behaviour kept on the whole explored graph", OK),
                (Some(_), false) => ("erratum to the explored bound  every export compatible and every behaviour kept as far as explored", OK),
                (Some(_), true) => ("interface change  behaviour kept, an export broken (F.4.3): callers must change", REFUSED),
                (None, false) => ("compatible  every export compatible; no Form to compare behaviour on", OK),
                (None, true) => ("breaking  an export broken (F.4.3); no Form to compare behaviour on", REFUSED),
            };
            writeln!(out, "verdict             {verdict}").map_err(w)?;
            status
        }
        Some("equiv") if rest.len() >= 2 => {
            // rce equiv [--fuel N] [--json] A B   (10.11)
            let (mut fuel, mut json, mut files) = (100_000usize, false, Vec::new());
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--fuel" if i + 1 < rest.len() => {
                        fuel = rest[i + 1].parse().map_err(|_| "--fuel N: N is a count of states".to_string())?;
                        i += 2;
                    }
                    "--json" => {
                        json = true;
                        i += 1;
                    }
                    f if !f.starts_with("--") => {
                        files.push(f.to_string());
                        i += 1;
                    }
                    other => return Err(format!("equiv: unexpected argument `{other}`")),
                }
            }
            if files.len() != 2 {
                return Err("equiv: two files are needed".into());
            }
            let a = parse(&read_file(&files[0])?).map_err(|e| format!("{}: {e}", files[0]))?;
            let b = parse(&read_file(&files[1])?).map_err(|e| format!("{}: {e}", files[1]))?;
            let e = crate::dim::equiv(&a, &b, fuel)?;
            if json {
                writeln!(out, "{}", crate::dim::equiv_json(&e)).map_err(w)?;
            } else {
                write!(out, "{}", crate::dim::equiv_report(&e)).map_err(w)?;
            }
            if e.equivalent { OK } else { REFUSED }
        }
        Some("dim") if !rest.is_empty() => {
            // rce dim [--fuel N] [--control] [--json] FILE   (10.11)
            let (mut fuel, mut control, mut json, mut file) = (100_000usize, false, false, None);
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--fuel" if i + 1 < rest.len() => {
                        fuel = rest[i + 1].parse().map_err(|_| "--fuel N: N is a count of states".to_string())?;
                        i += 2;
                    }
                    "--control" => {
                        control = true;
                        i += 1;
                    }
                    "--json" => {
                        json = true;
                        i += 1;
                    }
                    f if file.is_none() && !f.starts_with("--") => {
                        file = Some(f.to_string());
                        i += 1;
                    }
                    other => return Err(format!("dim: unexpected argument `{other}`")),
                }
            }
            let file = file.ok_or("dim: a FILE is required")?;
            let src = read_file(&file)?;
            let d = diagnostics(&src, &HashSet::new());
            if !d.is_empty() {
                writeln!(out, "REFUSED {file}: {}", d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; ")).map_err(w)?;
                return Ok(REFUSED);
            }
            let m = crate::program(&src)?;
            match crate::dim::dim_with(&m, fuel, control) {
                Ok(r) => {
                    if json {
                        writeln!(out, "{}", crate::dim::report_json(&r)).map_err(w)?;
                    } else {
                        write!(out, "{}", crate::dim::report(&r)).map_err(w)?;
                    }
                    OK
                }
                Err(e) => {
                    writeln!(out, "REFUSED {file}: {e}").map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("hash") if !rest.is_empty() => {
            let mut ms = Vec::new();
            for p in expand(rest)? {
                let m = parse(&read_file(&p)?).map_err(|e| format!("{p}: {e}"))?;
                for d in crate::apidiff::defs(&m) {
                    writeln!(out, "{}  {} {}{}", d.hash, d.kind, d.name, if d.exported { "  pub" } else { "" }).map_err(w)?;
                }
                ms.push(m);
            }
            writeln!(out, "{}  program", crate::apidiff::program_hash(&ms)).map_err(w)?;
            OK
        }
        Some("api-diff") if rest.len() == 2 => {
            let load = |p: &String| -> Result<Vec<Module>, String> {
                expand(std::slice::from_ref(p))?.iter().map(|f| parse(&read_file(f)?).map_err(|e| format!("{f}: {e}"))).collect()
            };
            let rows = crate::apidiff::diff(&load(&rest[0])?, &load(&rest[1])?);
            for r in &rows {
                writeln!(out, "{}", r.show()).map_err(w)?;
            }
            if rows.iter().any(|r| r.breaking) { REFUSED } else { OK }
        }
        Some("lib") if rest.is_empty() => {
            for n in crate::stdlib::names() {
                let src = crate::stdlib::source(n).unwrap_or("");
                let doc = src.lines().find(|l| l.starts_with("# ") && !l.starts_with("# rce-version")).map(|l| l.trim_start_matches("# ").to_string()).unwrap_or_default();
                writeln!(out, "{n:<16} {doc}").map_err(w)?;
            }
            // the library is named by the hash of its manifest (F.5.3), and a
            // module that differs from its pin is reported, never used quietly
            let bad = crate::manifest::stdlib_unpinned();
            writeln!(out, "\n{} modules, pinned by stdlib/lib/MANIFEST.json; a manifest names this library as the dependency stdlib with sha256 {}", crate::stdlib::names().len(), crate::manifest::stdlib_hash()).map_err(w)?;
            for e in &bad {
                writeln!(out, "{e}").map_err(w)?;
            }
            if bad.is_empty() { OK } else { REFUSED }
        }
        Some("lib") if rest.len() == 1 => match crate::stdlib::source(&rest[0]) {
            Some(src) => {
                write!(out, "{src}").map_err(w)?;
                OK
            }
            None => return Err(format!("lib: no built-in module `{}` (rce lib lists them)", rest[0])),
        },
        Some("manifest") if rest.len() == 1 => {
            // rce manifest DIR: the manifest of a directory (F.5.3), canonical JSON, to stdout
            let text = crate::manifest::generate(std::path::Path::new(&rest[0]))?;
            writeln!(out, "{text}").map_err(w)?;
            OK
        }
        Some("manifest-check") if !rest.is_empty() => {
            // rce manifest-check MANIFEST [--advisories FILE] [--key ISSUER=HEXPUBKEY]...
            let mpath = std::path::Path::new(&rest[0]);
            let text = read_file(&rest[0])?;
            let root = mpath.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
            let (mut adv_file, mut keys) = (None, std::collections::BTreeMap::new());
            let mut i = 1;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--advisories" if i + 1 < rest.len() => {
                        adv_file = Some(rest[i + 1].clone());
                        i += 2;
                    }
                    "--key" if i + 1 < rest.len() => {
                        let (n, h) = rest[i + 1].split_once('=').ok_or("--key ISSUER=HEXPUBKEY")?;
                        let b: Vec<u8> = (0..h.len() / 2).filter_map(|j| u8::from_str_radix(h.get(2 * j..2 * j + 2)?, 16).ok()).collect();
                        let k: [u8; 32] = b.try_into().map_err(|_| "a public key is 32 bytes (64 hex digits)".to_string())?;
                        keys.insert(n.to_string(), k);
                        i += 2;
                    }
                    other => return Err(format!("manifest-check: unexpected argument `{other}`")),
                }
            }
            match crate::manifest::check(text.trim_end_matches('\n').trim_end_matches('\r').to_string().as_str(), root)
                .map_err(|e| e.join("; "))
                .and_then(|pinned| {
                    let adv = match &adv_file {
                        Some(f) => crate::manifest::verify_advisories(&read_file(f)?, &keys).map_err(|e| e.join("; "))?,
                        None => vec![],
                    };
                    Ok((pinned, adv))
                }) {
                Ok((pinned, adv)) => {
                    let unsafe_ = crate::manifest::unsafe_list(text.trim_end_matches('\n'), &pinned, &adv);
                    writeln!(out, "ok {}: {} modules pinned, {} unsafe", rest[0], pinned.len(), unsafe_.len()).map_err(w)?;
                    for u in unsafe_ {
                        writeln!(out, "UNSAFE {u}").map_err(w)?;
                    }
                    OK
                }
                Err(e) => {
                    writeln!(out, "REFUSED {}: {e}", rest[0]).map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("v09-status") if rest.is_empty() => {
            writeln!(out, "implemented (version 0.9 subset):").map_err(w)?;
            for (r, d) in crate::v09::IMPLEMENTED {
                writeln!(out, "  {r}: {d}").map_err(w)?;
            }
            writeln!(out, "not implemented (reserved, not implemented, or not enforced where stated):").map_err(w)?;
            for (r, d) in crate::v09::UNIMPLEMENTED {
                writeln!(out, "  {r}: {d}").map_err(w)?;
            }
            OK
        }
        Some("emit") if rest.len() == 2 => match emit(&rest[0], &read_file(&rest[1])?) {
            Ok(s) => {
                write!(out, "{s}").map_err(w)?;
                OK
            }
            Err(e) => {
                writeln!(err, "{e}").map_err(w)?;
                // A program the checker refuses is status 1; an unknown target
                // or a construct an emitter does not translate is status 2.
                if ["form", "rust", "python", "js", "scala"].contains(&rest[0].as_str())
                    && !e.starts_with("not supported by the emitter")
                {
                    REFUSED
                } else {
                    USAGE
                }
            }
        },
        Some("form") => {
            writeln!(
                err,
                "rce form: the Form command line is not part of this build (spec 10.1); \
                 this `rce` has the language commands only"
            )
            .map_err(w)?;
            USAGE
        }
        Some("fmt") if rest.len() == 2 && rest[0] == "--nameless" => {
            // rce fmt --nameless FILE: the name-free projection (10.6), to stdout
            let src = read_file(&rest[1])?;
            match crate::nameless::nameless(&src).and_then(|t| fmt_text(&t)) {
                Ok(t) => {
                    write!(out, "{t}").map_err(w)?;
                    OK
                }
                Err(e) => {
                    writeln!(out, "REFUSED {}: {e}", rest[1]).map_err(w)?;
                    REFUSED
                }
            }
        }
        Some("fmt") if !rest.is_empty() => {
            let check = rest[0] == "--check";
            let files = if check { &rest[1..] } else { rest };
            let mut code = OK;
            for p in files {
                let src = read_file(p)?;
                match fmt_text(&src) {
                    Ok(n) if n == src => {}
                    Ok(n) => {
                        if check {
                            writeln!(out, "would reformat {p}").map_err(w)?;
                            code = REFUSED;
                        } else {
                            std::fs::write(p, n).map_err(|e| format!("{p}: {e}"))?;
                            writeln!(out, "formatted {p}").map_err(w)?;
                        }
                    }
                    Err(e) => {
                        writeln!(out, "REFUSED {p}: {e}").map_err(w)?;
                        code = REFUSED;
                    }
                }
            }
            code
        }
        Some("index") if !rest.is_empty() => {
            let mut o = crate::brick::Opts::default();
            let mut paths = vec![];
            let mut k = 0;
            while k < rest.len() {
                if rest[k] == "--vectors" && k + 1 < rest.len() {
                    o.vectors = rest[k + 1].parse().map_err(|_| format!("--vectors {}: not a count", rest[k + 1]))?;
                    k += 2;
                } else {
                    paths.push(rest[k].clone());
                    k += 1;
                }
            }
            let c = crate::brick::index_with(&load(&paths)?, &o);
            writeln!(out, "{}", crate::brick::canon(&c)).map_err(w)?;
            OK
        }
        Some("search") if rest.len() == 2 => {
            let c = crate::brick::read_catalog(&read_file(&rest[0])?).map_err(|e| format!("{}: {e}", rest[0]))?;
            for (r, m, n, k) in crate::brick::search(&c, &rest[1])? {
                writeln!(out, "{r} {m}.{n} : {k}").map_err(w)?;
            }
            OK
        }
        Some("lsp") if rest.is_empty() => serve_dyn(input, out),
        Some("help") | Some("--help") | Some("-h") => {
            writeln!(out, "{USAGE_TEXT}").map_err(w)?;
            OK
        }
        _ => {
            writeln!(err, "{USAGE_TEXT}").map_err(w)?;
            USAGE
        }
    })
}

/// `serve` over trait objects (it is generic over sized streams).
fn serve_dyn(r: &mut dyn BufRead, w: &mut dyn Write) -> i32 {
    struct R<'a>(&'a mut dyn BufRead);
    impl std::io::Read for R<'_> {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(b)
        }
    }
    impl BufRead for R<'_> {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            self.0.fill_buf()
        }
        fn consume(&mut self, n: usize) {
            self.0.consume(n)
        }
    }
    struct W<'a>(&'a mut dyn Write);
    impl Write for W<'_> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.write(b)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.flush()
        }
    }
    serve(&mut R(r), &mut W(w))
}

/// `rce host run|replay FILE --log D [EVENTS]` (Appendix E host runtime v0).
fn host_cmd(a: &[String], input: &mut dyn BufRead, out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, String> {
    use crate::host::Host;
    let w = |e: std::io::Error| e.to_string();
    let (sub, file, log) = (a[0].as_str(), &a[1], a.get(3).ok_or("--log D")?);
    let src = read_file(file)?;
    let d = diagnostics(&src, &HashSet::new());
    if !d.is_empty() {
        writeln!(out, "REFUSED {file}: {}", d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; ")).map_err(w)?;
        return Ok(REFUSED);
    }
    let m = crate::program(&src)?;
    match sub {
        "replay" => {
            let text = std::fs::read_to_string(log).map_err(|e| format!("{log}: {e}"))?;
            match crate::host::replay_report(&m, &text) {
                Ok(r) => {
                    let (n, h, s) = (r.entries, r.state_hash, r.state);
                    writeln!(out, "replay ok: {n} entries, {} refused, state_hash {h}", r.refusals.len()).map_err(w)?;
                    // HC-21: the refusal texts are not in D; replay re-derives them
                    for (q, c, t) in &r.refusals {
                        writeln!(out, "  seq {q} refused ({c}): {t}").map_err(w)?;
                    }
                    // `--redact f,g`: print those fields as digests (secrets stay in D)
                    let red: Vec<&str> = match a.get(4).map(|x| x.as_str()) {
                        Some("--redact") => a.get(5).map(|x| x.split(',').collect()).unwrap_or_default(),
                        _ => vec![],
                    };
                    writeln!(out, "{}", show_state_redacted(&s, &red)).map_err(w)?;
                    Ok(OK)
                }
                Err(e) => {
                    writeln!(out, "REPLAY FAILED {log}: {e}").map_err(w)?;
                    Ok(REFUSED)
                }
            }
        }
        "tls-id" => {
            // the Ed25519 key peers pin for this host (`--peer ADDR=KEY`)
            let id = crate::tls::Identity::load_or_create(std::path::Path::new(log))?;
            writeln!(out, "{}", crate::tls::hex(&id.public)).map_err(w)?;
            Ok(OK)
        }
        "run" | "serve" => {
            // SIGTERM and SIGINT reach the host loop as an input (HC-24):
            // blocked here, before any thread starts, and waited for by one
            #[cfg(unix)]
            if sub == "serve" {
                block_stop_signals();
            }
            // Authority is granted explicitly (HC-5): `--fs DIR` gives the file
            // adapters that root, `--pty` gives the PTY adapters.
            let mut rest: Vec<&String> = a[4..].iter().collect();
            let mut adapters: Vec<Box<dyn crate::host::Adapter>> = vec![];
            let (mut fs_root, mut pty) = (None::<std::path::PathBuf>, false);
            let mut net: Vec<String> = vec![];
            let (mut chan_me, mut peers) = (None::<String>, vec![]);
            let (mut tls, mut pins) = (false, vec![]);
            let mut tls_id = None;
            let mut exact: Option<String> = None;
            let mut blob_root = None::<std::path::PathBuf>;
            let mut media_root = None::<std::path::PathBuf>;
            let mut http_addr = None::<String>;
            // random bytes are an input, not authority: always available (HC-10)
            adapters.push(Box::new(crate::adapters::RandomAdapter));
            while let Some(f) = rest.first().map(|s| s.as_str()) {
                match f {
                    "--fs" if rest.len() >= 2 => {
                        fs_root = Some(std::path::PathBuf::from(rest[1].as_str()));
                        for x in crate::adapters::file_adapters(std::path::Path::new(rest[1].as_str())) {
                            adapters.push(Box::new(x));
                        }
                        rest.drain(..2);
                    }
                    "--net" if rest.len() >= 2 => {
                        net.push(rest[1].to_string());
                        rest.drain(..2);
                    }
                    "--chan" if rest.len() >= 2 => {
                        chan_me = Some(rest[1].to_string());
                        rest.drain(..2);
                    }
                    "--http" if rest.len() >= 2 => {
                        if http_addr.is_some() {
                            return Err("--http: one listening address per host".into());
                        }
                        http_addr = Some(rest[1].to_string());
                        rest.drain(..2);
                    }
                    "--peer" if rest.len() >= 2 => {
                        // with --tls: ADDR=ED25519HEX, the key pinned for that peer
                        match rest[1].split_once('=') {
                            Some((a, k)) => {
                                pins.push((a.to_string(), crate::tls::unhex32(k).ok_or(format!("--peer {}: not a 32-byte hex key", rest[1]))?));
                                peers.push(a.to_string());
                            }
                            None => peers.push(rest[1].to_string()),
                        }
                        rest.drain(..2);
                    }
                    "--tls" => {
                        tls = true;
                        rest.remove(0);
                    }
                    "--wasm" if rest.len() >= 2 => {
                        let wa = crate::adapters::wasm_adapter(std::path::Path::new(rest[1].as_str()));
                        exact = Some(wa.exact.clone());
                        adapters.push(Box::new(wa));
                        rest.drain(..2);
                    }
                    "--media" if rest.len() >= 2 => {
                        // capture and playback (W1i media): frames are blobs under DIR
                        media_root = Some(std::path::PathBuf::from(rest[1].as_str()));
                        for x in crate::adapters::media::media_adapters(std::path::Path::new(rest[1].as_str())) {
                            adapters.push(Box::new(x));
                        }
                        rest.drain(..2);
                    }
                    "--blob" if rest.len() >= 2 => {
                        blob_root = Some(std::path::PathBuf::from(rest[1].as_str()));
                        for x in crate::adapters::blob_adapters(std::path::Path::new(rest[1].as_str())) {
                            adapters.push(Box::new(x));
                        }
                        rest.drain(..2);
                    }
                    #[cfg(unix)]
                    "--pty" => {
                        pty = true;
                        for x in crate::adapters::pty::pty_adapters() {
                            adapters.push(Box::new(x));
                        }
                        rest.remove(0);
                    }
                    _ => break,
                }
            }
            if tls && pins.len() != peers.len() {
                return Err("--tls: every --peer needs a pinned key (ADDR=ED25519HEX)".into());
            }
            // the channel listener is bound now and its threads start after
            // the sandbox is entered, so they run under it (HC-5a)
            let mut chan_start: Option<crate::adapters::chan::Start> = None;
            if let Some(me) = &chan_me {
                let (xs, start) = if tls {
                    let id = crate::tls::Identity::load_or_create(std::path::Path::new(log))?;
                    let l = std::net::TcpListener::bind(me).map_err(|e| format!("--chan {me}: {e}"))?;
                    let xs = crate::adapters::chan::chan_adapters_tls_deferred(l, me, &pins, &id).map_err(|e| format!("--chan {me}: {e}"))?;
                    tls_id = Some(id);
                    xs
                } else {
                    crate::adapters::chan::chan_adapters_deferred(me, &peers).map_err(|e| format!("--chan {me}: {e}"))?
                };
                for x in xs {
                    adapters.push(Box::new(x));
                }
                chan_start = Some(start);
                net.push(me.clone());
            }
            if !net.is_empty() {
                for x in crate::adapters::net::net_adapters(&net) {
                    adapters.push(Box::new(x));
                }
            }
            // HC-24: the HTTP listener is bound before the sandbox is entered
            let mut http = match &http_addr {
                Some(_) if sub != "serve" => return Err("--http needs `rce host serve`: requests arrive while the host loop runs".into()),
                Some(addr) => {
                    let s = crate::adapters::http::Server::bind(addr);
                    adapters.push(Box::new(s.responder()));
                    Some(s)
                }
                None => None,
            };
            let mut h = if sub == "serve" { Host::open_file_bounded(&m, std::path::Path::new(log), adapters, DRAIN_MAX)? } else { Host::open_file_with(&m, std::path::Path::new(log), adapters)? };
            // E.4c: the host starts on any module; it says how it delivers to
            // each declaration that is not exactly what it has
            for n in h.delivery_notes() {
                eprintln!("rce host: {n}");
            }
            if let Some(s) = &http {
                h.http_listen(&s.listen_payload())?;
                match s.local_addr() {
                    Some(a) => eprintln!("rce host: http listening on {a}"),
                    None => eprintln!("rce host: http cannot listen on {} yet; retrying every second", s.addr),
                }
            }
            if let Some(id) = &tls_id {
                h.tls_identity(&crate::tls::hex(&id.public))?;
            }
            if let Some(x) = &exact {
                h.exact(x)?;
            }
            if crate::sandbox::enabled() {
                use crate::sandbox::{GrantKind, NetGrant};
                let lp = std::path::Path::new(log);
                let dir = lp.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
                let writable: Vec<&std::path::Path> = fs_root.iter().chain(blob_root.iter()).chain(media_root.iter()).map(|p| p.as_path()).collect();
                // every address the adapters may use, and how (HC-5a): the net
                // adapter connects to and binds its grants (the channel address
                // among them), a peer is connected to, the channel listener and
                // the HTTP listener were bound before entering; an HTTP listener
                // that could not be bound yet is bound later, inside the sandbox
                let mut grants: Vec<NetGrant> = vec![];
                for a in &net {
                    grants.extend(NetGrant::parse(a, GrantKind::Connect));
                    grants.extend(NetGrant::parse(a, GrantKind::Listen));
                }
                for a in &peers {
                    grants.extend(NetGrant::parse(a, GrantKind::Connect));
                }
                if let Some(me) = &chan_me {
                    grants.extend(NetGrant::parse(me, GrantKind::Prebound));
                }
                for s in http.iter() {
                    grants.extend(NetGrant::parse(&s.addr, if s.bound() { GrantKind::Prebound } else { GrantKind::Listen }));
                }
                grants.sort();
                grants.dedup();
                match crate::sandbox::enter(dir, &writable, pty, &grants) {
                    Ok(d) => eprintln!("rce host: {d}"),
                    Err(r) => eprintln!("rce host: sandbox OS target not entered: {r}"),
                }
            }
            if let Some(start) = chan_start.take() {
                start();
            }
            if sub == "serve" {
                let code = serve_loop(&mut h, &src, out, http.as_mut())?;
                let live = h.state_hash().to_string();
                drop(h);
                return validate_run(&m, log, &live, code, err);
            }
            // HC-24: the lines of the input, as bytes, so a line that is not
            // UTF-8 is held like any line that is not events of the module
            let batches: Vec<Vec<u8>> = match rest.first() {
                Some(e) => vec![e.as_bytes().to_vec()],
                None => {
                    let mut all = vec![];
                    std::io::Read::read_to_end(input, &mut all).map_err(|e| format!("stdin: {e}"))?;
                    all.split(|b| *b == b'\n').map(|l| l.strip_suffix(b"\r").unwrap_or(l).to_vec()).filter(|l| !l.iter().all(u8::is_ascii_whitespace)).collect()
                }
            };
            let mut refused = false;
            for b in batches {
                let before = h.len();
                let admitted = match std::str::from_utf8(&b).map_err(|e| format!("the line is not UTF-8 at offset {}", e.valid_up_to())).and_then(|t| h.admit(&src, t)) {
                    Ok(a) => a,
                    // HC-24: a line that is not events of the module is held
                    Err(e) if !Host::is_storage_fault(&e) && h.len() == before => {
                        let a = h.line_input(&[b.as_slice(), b"\n"].concat())?;
                        h.settle()?;
                        if a.is_empty() {
                            refused = true;
                            writeln!(out, "{before} host.Kept: the line is not events of the module ({e}); recorded in D, not delivered").map_err(w)?;
                        }
                        a
                    }
                    Err(e) => return Err(e),
                };
                for x in admitted {
                    match x.outcome {
                        Ok(outs) => {
                            writeln!(out, "{} {}", x.seq, x.event).map_err(w)?;
                            for o in outs {
                                writeln!(out, "  emit {o}").map_err(w)?;
                            }
                        }
                        Err(r) => {
                            refused = true;
                            writeln!(out, "{} {} refused: {r}", x.seq, x.event).map_err(w)?
                        }
                    }
                }
            }
            writeln!(out, "durable {} entries={} state_hash {}", h.durable.map_or("none".into(), |d| d.to_string()), h.len(), h.state_hash()).map_err(w)?;
            writeln!(out, "{}", show_state(h.state())).map_err(w)?;
            let live = h.state_hash().to_string();
            drop(h);
            validate_run(&m, log, &live, if refused { REFUSED } else { OK }, err)
        }
        _ => Err(format!("rce host: unknown subcommand `{sub}`")),
    }
}

/// The complete Form of a module (W2b): `form_json` (S, E, errors, the δ
/// index and the D logs of the examples) plus
/// - `terms`: the whole module as canonical AST terms (every δ handler and
///   function body, `fn main`, tasks, actors), which `rce unform` turns back
///   into source that parses to the same terms (tests/form_roundtrip.rs);
/// - `source`: that canonical source text;
/// - `main`: for a program with `fn main`, the outputs and outcome of its
///   run on the tick runtime (its D body), else null.
pub fn form_full(m: &Module) -> Result<String, String> {
    let base = form_json(m);
    let terms = serde_json::to_string(m).map_err(|e| format!("terms: {e}"))?;
    let source = serde_json::to_string(&crate::unparse::unparse(m)).map_err(|e| e.to_string())?;
    let has_main = m.items.iter().any(|i| matches!(i, Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty()));
    let main = if has_main {
        match crate::conc::run_main(m, &crate::conc::RunCfg::default()) {
            Ok(r) => {
                let outs: Vec<String> = r.outputs.iter().map(|o| interp::form_value(o, m).json()).collect();
                let outcome = match r.outcome {
                    crate::conc::Outcome::Done => "\"done\"".to_string(),
                    crate::conc::Outcome::Deadlock => "\"deadlock\"".to_string(),
                    crate::conc::Outcome::Refused(e) => format!("{{\"refused\":{}}}", serde_json::to_string(&e).unwrap_or_default()),
                };
                format!("{{\"outcome\":{outcome},\"outputs\":[{}]}}", outs.join(","))
            }
            Err(e) => format!("{{\"error\":{}}}", serde_json::to_string(&e).unwrap_or_default()),
        }
    } else {
        "null".into()
    };
    let body = base.strip_suffix('}').ok_or("form json")?;
    let names = concept_record(m);
    Ok(format!("{body},\"main\":{main},\"source\":{source},\"terms\":{terms},\"names\":{names}}}"))
}

/// The concept record (6.6): from each name-free key (the declaration
/// index of 6.6, as `#i`) to the name as written and, for an `intent`, its
/// text. Printed after the hashed fields; never hashed, never read by δ.
/// Two modules that differ only here have the same outcomes, outputs and
/// ids (`tests/nameless.rs`).
pub fn concept_record(m: &Module) -> String {
    let mut rows: Vec<String> = vec![];
    let mut intents: Vec<String> = vec![];
    for (i, it) in m.items.iter().enumerate() {
        if let Item::Intent(text, _, _) = it {
            intents.push(serde_json::to_string(text).unwrap_or_default());
        }
        if let Some(n) = crate::link::item_name(it) {
            let kind = match it {
                Item::State { .. } => "state",
                Item::Actor { .. } => "actor",
                Item::Event { .. } => "event",
                Item::Error { .. } => "error",
                Item::Fn(_) => "fn",
                Item::Struct { .. } => "struct",
                Item::Kind { .. } => "kind",
                Item::Trait { .. } => "trait",
                Item::TypeAlias { .. } => "type",
                Item::Const { .. } => "const",
                _ => "item",
            };
            rows.push(format!("{{\"key\":\"#{i}\",\"kind\":\"{kind}\",\"name\":{}}}", serde_json::to_string(&n).unwrap_or_default()));
        }
    }
    format!("{{\"intent\":[{}],\"items\":[{}]}}", intents.join(","), rows.join(","))
}

/// `rce unform FORM.json`: the canonical source of a Form's `terms`. The
/// terms are the linked program (4.1): the items of every module it uses
/// are already among them, under their own names, so the source leaves out
/// the `use` lines, which would import those items a second time (W-MOD-3).
pub fn unform(json: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("form json: {e}"))?;
    let t = v.get("terms").ok_or("the Form has no `terms` (emit it with `rce emit form`)")?;
    let mut m: Module = serde_json::from_value(t.clone()).map_err(|e| format!("terms: {e}"))?;
    m.items.retain(|i| !matches!(i, Item::Use(..)));
    let m = crate::v09::restore_world(m)?;
    Ok(crate::unparse::unparse(&m))
}

/// The engine's frame period for `Tick` (W1d). It is chosen by the engine,
/// not by a setting (HC-14), and it never changes meaning: each Tick is a
/// logged input with its logical number n (HC-2).
pub const FRAME_MS: u64 = 16;

/// `rce host serve`: the host loop. Inputs, in the order the admitter
/// receives them, are admitted in rounds: each round takes one input and
/// whatever is already waiting (at most 64), applies them, then performs
/// their effects and takes one barrier for the round (group commit, D-2).
/// An input is a stdin line in event syntax. A line that is not events of
/// the module (it is not UTF-8, does not parse, names no event, or one of
/// its events cannot be built) is held whole as `Input(data: line)`, and
/// recorded in D as `host.Kept` when the module declares no `Input`
/// (HC-24, E.4c); with `--http`, each request is delivered as the module
/// declares `HttpRequest` or `HttpMalformed` (HC-24, HD-2); between inputs,
/// `Tick` (or `Tick(n)`) every frame when declared, like any delivery (HD-4). Only a
/// storage fault (HS-3) stops the loop. EOF on stdin ends the
/// loop unless `--http` is granted; an emitted `Exit` ends it; SIGTERM or
/// SIGINT stop accepting, wait up to STOP_MS for open requests, and end it.
/// Each drain performs at most DRAIN_MAX requests and the next round
/// starts at once while requests are queued, so a stop is read between
/// drains even when results keep requesting effects.
fn serve_loop(h: &mut crate::host::Host, src: &str, out: &mut dyn Write, mut http: Option<&mut crate::adapters::http::Server>) -> Result<i32, String> {
    use crate::adapters::http::{HttpIn, Pending};
    use crate::host::Host;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    let w = |e: std::io::Error| e.to_string();
    let (tx, rx) = mpsc::channel::<Inbound>();
    let stdin_tx = tx.clone();
    std::thread::spawn(move || {
        // HC-24: lines are read as bytes, so a line that is not UTF-8 is
        // held like any other, and never read as the end of the input
        let mut stdin = std::io::stdin().lock();
        let mut l: Vec<u8> = vec![];
        loop {
            l.clear();
            match stdin.read_until(b'\n', &mut l) {
                Ok(0) | Err(_) => {
                    let _ = stdin_tx.send(Inbound::Eof);
                    return;
                }
                Ok(_) => {
                    if stdin_tx.send(Inbound::Line(l.clone())).is_err() {
                        return;
                    }
                }
            }
        }
    });
    #[cfg(unix)]
    spawn_stop_thread(tx.clone());
    let open = http.as_ref().map(|s| s.open());
    if let Some(s) = http.as_deref_mut() {
        let htx = std::sync::Mutex::new(tx.clone());
        s.start(std::sync::Arc::new(move |x| htx.lock().map(|t| t.send(Inbound::Http(x)).is_ok()).unwrap_or(false)));
    }
    // HD-4: Tick is delivered like any delivery; a declaration that no frame
    // count can be built into is not delivered, and the host says so once
    let mut tick = h.event_fields("Tick");
    if tick.is_some()
        && let Err(why) = h.tick_args(1)
    {
        eprintln!("rce host: event Tick: {why}, so no frame count can be built into it and Tick is not delivered");
        tick = None;
    }
    // HC-22: on an existing D the count continues from the last Tick(n)
    let mut n: u64 = h.last_tick();
    let frame = Duration::from_millis(FRAME_MS);
    let mut next_tick = Instant::now() + frame;
    let mut stopping: Option<Instant> = None;
    h.bound_drains(DRAIN_MAX);
    // an input whose admission failed although it was held: reported, and
    // the loop goes on; a storage fault (HS-3) ends the loop
    let report = |e: String, out: &mut dyn Write| -> Result<(), String> {
        if Host::is_storage_fault(&e) {
            return Err(e);
        }
        writeln!(out, "input not admitted: {e}").map_err(|x| x.to_string())
    };
    'serve: loop {
        let wait = if h.queued() > 0 {
            Duration::ZERO
        } else if stopping.is_some() {
            Duration::from_millis(50)
        } else if tick.is_some() {
            next_tick.saturating_duration_since(Instant::now())
        } else {
            Duration::from_secs(3600)
        };
        let mut batch = vec![];
        match rx.recv_timeout(wait) {
            Ok(x) => batch.push(x),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        while batch.len() < 64 {
            match rx.try_recv() {
                Ok(x) => batch.push(x),
                Err(_) => break,
            }
        }
        let mut adm = vec![];
        let mut answers: Vec<(mpsc::Sender<crate::adapters::http::Reply>, u64)> = vec![];
        let mut end = false;
        if tick.is_some() && stopping.is_none() && Instant::now() >= next_tick {
            next_tick += frame;
            n += 1;
            match h.tick(n) {
                Ok(Some(a)) => adm.extend(a),
                // HD-4: a count that no longer converts is recorded once
                // (host.Kept), and no later count converts either
                Ok(None) => {
                    eprintln!("rce host: Tick({n}) cannot be built into the declaration of Tick; recorded in D (host.Kept), and no later Tick is delivered");
                    tick = None;
                }
                Err(e) => report(e, out)?,
            }
        }
        for x in batch {
            match x {
                Inbound::Line(line) => {
                    let mut t: &[u8] = &line;
                    while let Some(x) = t.strip_suffix(b"\n").or_else(|| t.strip_suffix(b"\r")) {
                        t = x;
                    }
                    let before = h.len();
                    match std::str::from_utf8(t).map_err(|e| format!("the line is not UTF-8 at offset {}", e.valid_up_to())).and_then(|x| h.admit(src, x)) {
                        Ok(a) => adm.extend(a),
                        // HC-24: a line that is not events of the module is
                        // held whole, as Input(data) or as host.Kept in D
                        Err(e) if !Host::is_storage_fault(&e) && h.len() == before => match h.line_input(&[t, b"\n"].concat()) {
                            Ok(a) => {
                                if a.is_empty() {
                                    eprintln!("rce host: line recorded in D (host.Kept), not events of the module: {e}");
                                }
                                adm.extend(a);
                            }
                            Err(e) => report(e, out)?,
                        },
                        Err(e) => report(e, out)?,
                    }
                }
                Inbound::Eof => {
                    if http.is_none() {
                        end = true;
                    }
                }
                Inbound::Stop => {
                    if http.is_none() {
                        end = true;
                    } else if stopping.is_none() {
                        stopping = Some(Instant::now());
                        if let Some(s) = http.as_deref() {
                            s.stop();
                        }
                    }
                }
                Inbound::Http(HttpIn::Listening) => {
                    if let Some(s) = http.as_deref_mut() {
                        s.listening();
                        h.http_listen(&s.listen_payload())?;
                    }
                }
                Inbound::Http(x @ (HttpIn::Request { .. } | HttpIn::Malformed { .. })) => {
                    // HC-24, HD-2: the host delivers the input as the module declares it; the
                    // id is the seq of the entry that delivers it, and the
                    // framing (HC-26) is read from the host's whole value
                    let (name, p, reply) = match x {
                        HttpIn::Request { req, reply } => {
                            let mut p = serde_json::Map::new();
                            p.insert("body".into(), serde_json::Value::from(req.body.clone()));
                            p.insert("method".into(), req.method.clone().into());
                            p.insert("names".into(), req.names.clone().into());
                            p.insert("target".into(), req.target.clone().into());
                            p.insert("values".into(), req.values.clone().into());
                            p.insert("version".into(), req.version().into());
                            ("HttpRequest", p, reply)
                        }
                        HttpIn::Malformed { status, why, reply } => {
                            let mut p = serde_json::Map::new();
                            p.insert("status".into(), serde_json::Value::from(status));
                            p.insert("why".into(), why.into());
                            ("HttpMalformed", p, reply)
                        }
                        HttpIn::Listening => unreachable!(),
                    };
                    match h.http_input(name, p) {
                        Ok(a) => {
                            match a.answer {
                                // the host answers once the batch is durable (HS-2)
                                Some(status) => answers.push((reply, status)),
                                None => {
                                    if let Some(o) = &open {
                                        o.borrow_mut().insert(a.id, Pending { reply, close: a.close, head: a.head });
                                    }
                                }
                            }
                            adm.extend(a.admitted);
                        }
                        Err(e) => report(e, out)?,
                    }
                }
            }
        }
        adm.extend(h.settle()?);
        // HD-2: a request the program cannot answer gets the host's own
        // response, with no body, and the connection closes after it
        for (reply, status) in answers.drain(..) {
            let r = serde_json::json!({"status": status, "names": [], "values": [], "body": ""});
            if let Ok((bytes, close)) = crate::adapters::http::response_bytes(&r, false, true) {
                let _ = reply.send(crate::adapters::http::Reply { bytes, close });
            }
        }
        let mut exit = false;
        for a in adm {
            if let Some(c) = a.cause {
                eprintln!("rce host: seq {} refused ({c})", a.seq);
            }
            for d in &a.display {
                write!(out, "{d}").map_err(w)?;
            }
            exit |= a.exit;
        }
        out.flush().map_err(w)?;
        if exit || end {
            break 'serve;
        }
        if let (Some(t), Some(s)) = (stopping, http.as_deref()) {
            let idle = s.active() == 0 && open.as_ref().is_none_or(|o| o.borrow().is_empty());
            if idle || t.elapsed() >= Duration::from_millis(STOP_MS) {
                break 'serve;
            }
        }
    }
    writeln!(out, "\ndurable {} entries={} state_hash {}", h.durable.map_or("none".into(), |d| d.to_string()), h.len(), h.state_hash()).map_err(w)?;
    Ok(OK)
}

/// After SIGTERM or SIGINT, the longest the host loop waits for open
/// requests before it takes its final barrier (HC-24).
pub const STOP_MS: u64 = 5_000;

/// The most queued requests the host loop performs in one drain (HC-24);
/// the rest are performed in the next round, after a stop has been read.
pub const DRAIN_MAX: usize = 64;

/// What the host loop receives, from any source, on one channel.
enum Inbound {
    Line(Vec<u8>),
    Eof,
    Http(crate::adapters::http::HttpIn),
    Stop,
}

#[cfg(unix)]
fn stop_signals() -> libc::sigset_t {
    // SAFETY: sigemptyset and sigaddset initialise the set they are given.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGINT);
        set
    }
}

/// Block SIGTERM and SIGINT in this thread and every thread it starts from
/// now on, so only the stop thread receives them (`sigwait`).
#[cfg(unix)]
pub fn block_stop_signals() {
    let set = stop_signals();
    // SAFETY: a valid set; the old mask is not needed.
    unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) };
}

#[cfg(unix)]
fn spawn_stop_thread(tx: std::sync::mpsc::Sender<Inbound>) {
    std::thread::spawn(move || {
        let set = stop_signals();
        loop {
            let mut sig: libc::c_int = 0;
            // SAFETY: a valid set and an out pointer.
            if unsafe { libc::sigwait(&set, &mut sig) } != 0 || tx.send(Inbound::Stop).is_err() {
                return;
            }
        }
    });
}
/// P5 per-run validation: after every host run, D as left on storage is
/// verified (HS-4) and replayed with every adapter disconnected (HC-8,
/// T37); the run counts only when replay gives the live state hash. The
/// report goes to stderr so stdout stays the program's output.
fn validate_run(m: &Module, log: &str, live: &str, code: i32, err: &mut dyn Write) -> Result<i32, String> {
    let text = std::fs::read_to_string(log).map_err(|e| format!("validate {log}: {e}"))?;
    match crate::host::replay_report(m, &text) {
        Ok(r) if r.state_hash == live => {
            writeln!(err, "rce host: validated {} entries, {} refused: replay with adapters disconnected gives the live state_hash", r.entries, r.refusals.len()).map_err(|e| e.to_string())?;
            Ok(code)
        }
        Ok(r) => Err(format!("host fault: replay gives {}, the live run had {live}", r.state_hash)),
        Err(e) => Err(format!("host fault: replay failed: {e}")),
    }
}

/// `rce db D CMD [ARGS]`: RCEdb on the log D, in this process (W3). Writes
/// print `ok seq N`; a refused write prints the refusal and exits 1; rows
/// print as `KEY<TAB>ROW` in canonical key order.
fn db_cmd(a: &[String], out: &mut dyn Write) -> Result<i32, String> {
    use crate::rcedb::{Db, Dialect};
    let w = |e: std::io::Error| e.to_string();
    let path = std::path::Path::new(&a[0]);
    let c: Vec<&str> = a[1..].iter().map(String::as_str).collect();
    let mut db = Db::open(path)?;
    let rows = |out: &mut dyn Write, rs: Vec<crate::rcedb::Row>| -> Result<(), String> {
        for (_, k, r) in rs {
            writeln!(out, "{k}\t{r}").map_err(w)?;
        }
        Ok(())
    };
    let wr = match c.as_slice() {
        ["insert", t, k, r] => db.insert(t, k, r),
        ["delete", t, k] => db.delete(t, k),
        ["link", s, l, d] => db.link(s, l, d),
        ["unlink", s, l, d] => db.unlink(s, l, d),
        ["schema", t, cols] => db.schema(t, cols),
        ["index", t, col] => db.index(t, col),
        ["scan", t] | ["scan", t, _] => {
            let rs = db.scan(t, c.get(2).copied().unwrap_or(""))?;
            rows(out, rs)?;
            return Ok(0);
        }
        ["lookup", t, col, v] => {
            let rs = db.lookup(t, col, v)?;
            rows(out, rs)?;
            return Ok(0);
        }
        ["edges", s] => {
            for (_, l, d) in db.edges(s)? {
                writeln!(out, "{l}\t{d}").map_err(w)?;
            }
            return Ok(0);
        }
        ["describe"] => {
            for (t, n, cols, ix) in db.describe()? {
                writeln!(out, "{t}\trows {n}\tschema {}\tindexes {}", if cols.is_empty() { "-" } else { &cols }, if ix.is_empty() { "-" } else { &ix }).map_err(w)?;
            }
            return Ok(0);
        }
        ["dump"] => {
            let (rs, es) = db.dump()?;
            for (t, k, r) in rs {
                writeln!(out, "row\t{t}\t{k}\t{r}").map_err(w)?;
            }
            for (s, l, d) in es {
                writeln!(out, "edge\t{s}\t{l}\t{d}").map_err(w)?;
            }
            return Ok(0);
        }
        ["emit", d] => {
            let d = match *d {
                "sqlite" => Dialect::Sqlite,
                "postgres" => Dialect::Postgres,
                "duckdb" => Dialect::DuckDb,
                "cypher" => Dialect::Cypher,
                _ => return Err(format!("rce db: unknown dialect `{d}`")),
            };
            write!(out, "{}", db.emit(d)?).map_err(w)?;
            return Ok(0);
        }
        ["as-of", n, "scan", t] | ["as-of", n, "scan", t, _] => {
            let k: usize = n.parse().map_err(|_| format!("rce db: as-of needs an entry count, got `{n}`"))?;
            let mut past = db.as_of(k)?;
            rows(out, past.scan(t, c.get(4).copied().unwrap_or(""))?)?;
            return Ok(0);
        }
        _ => return Err(format!("rce db: unknown command `{}`", c.join(" "))),
    };
    match wr {
        Ok(()) => {
            writeln!(out, "ok seq {}", db.seq()).map_err(w)?;
            Ok(0)
        }
        Err(e) => {
            writeln!(out, "REFUSED {e}").map_err(w)?;
            Ok(1)
        }
    }
}
