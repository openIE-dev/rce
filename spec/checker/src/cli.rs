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
            let mut c = check::Checker::new();
            c.check_module(&m, known);
            if c.errs.is_empty() {
                interp::admit(&m)
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
    let m = parse(src)?;
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
            Ok(form_full(&parse(src)?)? + "\n")
        }
        "rust" | "python" | "js" | "scala" => {
            let d = diagnostics(src, &HashSet::new());
            if !d.is_empty() {
                return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
            }
            let lang = crate::emit::Lang::parse(target).expect("target");
            crate::emit::emit(&parse(src)?, lang)
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
        (parse(src)?, None)
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
  rce hash FILE|DIR...           the definition hash (F.4.3) of every named item of each module: HASH, kind, name, and `pub` for an export
  rce manifest-check MANIFEST [--advisories FILE] [--key ISSUER=HEXPUBKEY]...  verify a hash-pinned manifest (F.5.3, W-PIN-1) and list pinned hashes that carry a signed advisory
  rce api-diff OLD NEW           compare the exports of two module sets (FILE or DIR each); one row per change, compatible or breaking; status 1 if any is breaking
  rce v09-status                 the Appendix F rules this checker implements and the ones it refuses by version
  rce lsp                        language server on stdin/stdout
  rce index [--vectors N] FILE|DIR...  brick catalog (rce-catalog/1 JSON): every exported fn, type key, SHA-256, verdict, D.7 Form IR term
  rce search CATALOG QUERY       bricks of type QUERY (e.g. (A, B) -> R; a, b, ... are variables), best first
  rce unform FORM.json           the canonical source of a Form's terms (inverse of `rce emit form`)
  rce host run FILE --log D [--fs DIR] [--blob DIR] [--media DIR] [--wasm DIR] [--net HOST:PORT]... [--chan HOST:PORT [--tls] [--peer HOST:PORT[=KEY]]...] [--pty] [EVENTS]  run on the host runtime (Appendix E): admit EVENTS, or one batch per stdin line, into the durable log D
  rce host serve FILE --log D [--fs DIR] [--blob DIR] [--media DIR] [--wasm DIR] [--net HOST:PORT]... [--chan HOST:PORT [--tls] [--peer HOST:PORT[=KEY]]...] [--pty]  the host loop (W1d): stdin lines are inputs (event syntax, else Input(data) when declared), Tick(n) every frame when declared, Display(data) is written to stdout, Exit ends the loop
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
            let m = parse(&src)?;
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
    use crate::host::{Host, replay};
    let w = |e: std::io::Error| e.to_string();
    let (sub, file, log) = (a[0].as_str(), &a[1], a.get(3).ok_or("--log D")?);
    let src = read_file(file)?;
    let d = diagnostics(&src, &HashSet::new());
    if !d.is_empty() {
        writeln!(out, "REFUSED {file}: {}", d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; ")).map_err(w)?;
        return Ok(REFUSED);
    }
    let m = parse(&src)?;
    match sub {
        "replay" => {
            let text = std::fs::read_to_string(log).map_err(|e| format!("{log}: {e}"))?;
            match replay(&m, &text) {
                Ok((n, h, s)) => {
                    writeln!(out, "replay ok: {n} entries, state_hash {h}").map_err(w)?;
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
            if let Some(me) = &chan_me {
                let xs = if tls {
                    let id = crate::tls::Identity::load_or_create(std::path::Path::new(log))?;
                    let xs = crate::adapters::chan::chan_adapters_tls(me, &pins, &id).map_err(|e| format!("--chan {me}: {e}"))?;
                    tls_id = Some(id);
                    xs
                } else {
                    crate::adapters::chan::chan_adapters(me, &peers).map_err(|e| format!("--chan {me}: {e}"))?
                };
                for x in xs {
                    adapters.push(Box::new(x));
                }
                net.push(me.clone());
            }
            if !net.is_empty() {
                for x in crate::adapters::net::net_adapters(&net) {
                    adapters.push(Box::new(x));
                }
            }
            let mut h = Host::open_file_with(&m, std::path::Path::new(log), adapters)?;
            if let Some(id) = &tls_id {
                h.tls_identity(&crate::tls::hex(&id.public))?;
            }
            if let Some(x) = &exact {
                h.exact(x)?;
            }
            if crate::sandbox::enabled() {
                let lp = std::path::Path::new(log);
                let dir = lp.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
                let writable: Vec<&std::path::Path> = fs_root.iter().chain(blob_root.iter()).chain(media_root.iter()).map(|p| p.as_path()).collect();
                let ports: Vec<u16> = net.iter().chain(peers.iter()).filter_map(|a| a.rsplit(':').next()?.parse().ok()).collect();
                match crate::sandbox::enter(dir, &writable, pty, !net.is_empty(), &ports) {
                    Ok(d) => eprintln!("rce host: {d}"),
                    Err(r) => eprintln!("rce host: sandbox OS target not entered: {r}"),
                }
            }
            if sub == "serve" {
                let code = serve_loop(&mut h, &src, out)?;
                let live = h.state_hash().to_string();
                drop(h);
                return validate_run(&m, log, &live, code, err);
            }
            let batches: Vec<String> = match rest.first() {
                Some(e) => vec![(*e).clone()],
                None => input.lines().map_while(Result::ok).filter(|l| !l.trim().is_empty()).collect(),
            };
            let mut refused = false;
            for b in batches {
                for x in h.admit(&src, &b)? {
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

/// `rce unform FORM.json`: the canonical source of a Form's `terms`.
pub fn unform(json: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("form json: {e}"))?;
    let t = v.get("terms").ok_or("the Form has no `terms` (emit it with `rce emit form`)")?;
    let m: Module = serde_json::from_value(t.clone()).map_err(|e| format!("terms: {e}"))?;
    let m = crate::v09::restore_world(m)?;
    Ok(crate::unparse::unparse(&m))
}

/// The engine's frame period for `Tick` (W1d). It is chosen by the engine,
/// not by a setting (HC-14), and it never changes meaning: each Tick is a
/// logged input with its logical number n (HC-2).
pub const FRAME_MS: u64 = 16;

/// `rce host serve`: the host loop. Inputs, in arrival order, are admitted
/// one batch at a time: a stdin line in event syntax, or `Input(data: line)`
/// when the line is not an event list and the module declares
/// `event Input(data: text)`; between inputs, `Tick` (or `Tick(n)`) every
/// frame when declared. EOF on stdin ends the loop, as does an emitted `Exit`.
fn serve_loop(h: &mut crate::host::Host, src: &str, out: &mut dyn Write) -> Result<i32, String> {
    use crate::ast::{Arg, Expr};
    use std::sync::mpsc;
    let w = |e: std::io::Error| e.to_string();
    let (tx, rx) = mpsc::channel::<Option<String>>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut l = String::new();
        loop {
            l.clear();
            match stdin.read_line(&mut l) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(None);
                    return;
                }
                Ok(_) => {
                    if tx.send(Some(l.clone())).is_err() {
                        return;
                    }
                }
            }
        }
    });
    let tick = h.event_fields("Tick");
    let input = h.event_fields("Input").filter(|f| f.len() == 1 && f[0] == "data");
    let mut n: u64 = 0;
    let frame = std::time::Duration::from_millis(FRAME_MS);
    let mut next_tick = std::time::Instant::now() + frame;
    loop {
        let wait = if tick.is_some() { next_tick.saturating_duration_since(std::time::Instant::now()) } else { std::time::Duration::from_secs(3600) };
        let adm = match rx.recv_timeout(wait) {
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Ok(Some(line)) => {
                let t = line.trim_end_matches(['\n', '\r']);
                match h.admit(src, t) {
                    Ok(a) => a,
                    Err(e) if input.is_some() && (e.starts_with("events:") || e.ends_with("is not an event")) => {
                        h.admit_event("Input", vec![Arg { label: Some("data".into()), e: Expr::Str(format!("{t}\n"), 0) }])?
                    }
                    Err(e) => {
                        writeln!(out, "input refused: {e}").map_err(w)?;
                        continue;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                next_tick += frame;
                n += 1;
                let args = match &tick {
                    Some(f) if f.len() == 1 => vec![Arg { label: Some(f[0].clone()), e: Expr::Int(n as i128, 0) }],
                    _ => vec![],
                };
                h.admit_event("Tick", args)?
            }
        };
        let mut exit = false;
        for a in adm {
            for d in &a.display {
                write!(out, "{d}").map_err(w)?;
            }
            exit |= a.exit;
        }
        out.flush().map_err(w)?;
        if exit {
            break;
        }
    }
    writeln!(out, "\ndurable {} entries={} state_hash {}", h.durable.map_or("none".into(), |d| d.to_string()), h.len(), h.state_hash()).map_err(w)?;
    Ok(OK)
}

/// P5 per-run validation: after every host run, D as left on storage is
/// verified (HS-4) and replayed with every adapter disconnected (HC-8,
/// T37); the run counts only when replay gives the live state hash. The
/// report goes to stderr so stdout stays the program's output.
fn validate_run(m: &Module, log: &str, live: &str, code: i32, err: &mut dyn Write) -> Result<i32, String> {
    let text = std::fs::read_to_string(log).map_err(|e| format!("validate {log}: {e}"))?;
    match crate::host::replay(m, &text) {
        Ok((n, h, _)) if h == live => {
            writeln!(err, "rce host: validated {n} entries: replay with adapters disconnected gives the live state_hash").map_err(|e| e.to_string())?;
            Ok(code)
        }
        Ok((_, h, _)) => Err(format!("host fault: replay gives {h}, the live run had {live}")),
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
