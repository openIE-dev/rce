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
    s.iter()
        .map(|(r, fs)| {
            let f: Vec<String> = fs
                .iter()
                .map(|(n, v, _)| format!("{n} = {}", v.show()))
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
            Ok(form_json(&parse(src)?) + "\n")
        }
        "rust" | "python" | "js" => {
            let d = diagnostics(src, &HashSet::new());
            if !d.is_empty() {
                return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
            }
            let lang = crate::emit::Lang::parse(target).expect("target");
            crate::emit::emit(&parse(src)?, lang)
        }
        t => Err(format!(
            "unknown emit target `{t}` (form, rust, python, js)"
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
pub const TRACE_FORMAT: &str = "rce-trace/1";

/// `rce trace --json FILE [EVENTS]`: the reference trace of a Form (10.9).
/// Without EVENTS, one trace per `example` of the module (its log D, 6.1);
/// with EVENTS, one trace of that event list. Each step carries the
/// canonical D fields `seq`, `event`, `state_hash`, `prev_hash`, `hash`
/// and the derived `outcome`; a trace stops at the first refused event.
/// Keys are sorted and the JSON is compact, so equal traces are equal bytes.
pub fn trace(file: &str, src: &str, events: Option<&str>) -> Result<String, String> {
    let d = diagnostics(src, &HashSet::new());
    if !d.is_empty() {
        return Err(d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
    }
    let (m, only) = match events {
        None => (parse(src)?, None),
        Some(evs) => {
            let probe = parse(&format!("{src}\nexample {evs} -> 0\n")).map_err(|e| format!("events: {e}"))?;
            let Some(Item::Example(x)) = probe.items.last() else {
                return Err("events: not an event list".into());
            };
            let line = x.line;
            (probe, Some(line))
        }
    };
    let form: serde_json::Value =
        serde_json::from_str(&crate::interp::form_json(&m)).map_err(|e| format!("form json: {e}"))?;
    let mut traces = vec![];
    for l in form["logs"].as_array().into_iter().flatten() {
        let line = l["example_line"].as_u64().unwrap_or(0) as usize;
        if only.is_some_and(|o| o != line) {
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
                })
            })
            .collect();
        let source = if only.is_some() { "events".to_string() } else { format!("example@{line}") };
        traces.push(serde_json::json!({ "source": source, "steps": steps }));
    }
    if let (Some(evs), true) = (events, traces.is_empty()) {
        return Err(format!("events: `{evs}` gives no trace (no state, or an unknown event)"));
    }
    let name = std::path::Path::new(file).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let v = serde_json::json!({
        "format": TRACE_FORMAT,
        "file": name,
        "events": events,
        "genesis": crate::interp::D_GENESIS,
        "traces": traces,
    });
    Ok(v.to_string())
}

/// Usage of the language commands of `rce` (10.2 to 10.7).
pub const USAGE_TEXT: &str = "usage:
  rce check FILE...              parse, type check and admit; one verdict per file
  rce run FILE [EVENTS]          apply EVENTS (example syntax, e.g. \"Inc, Inc, Dec\"), or run fn main()
  rce test FILE...               run every example and check; report per file
  rce trace --json FILE [EVENTS] reference trace (seq, event, state_hash per step) of the examples, or of EVENTS
  rce emit TARGET FILE           TARGET: form (Form JSON), rust, python or js (a program printing its observations)
  rce fmt [--check] FILE...      rewrite files in canonical layout (--check: only report)
  rce lsp                        language server on stdin/stdout";

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

fn run_main(a: &[String], input: &mut dyn BufRead, out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, String> {
    let rest = if a.is_empty() { &[][..] } else { &a[1..] };
    let load = |ps: &[String]| -> Result<Vec<(String, String)>, String> {
        ps.iter().map(|p| Ok((p.clone(), read_file(p)?))).collect()
    };
    let w = |e: std::io::Error| e.to_string();
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
        Some("emit") if rest.len() == 2 => match emit(&rest[0], &read_file(&rest[1])?) {
            Ok(s) => {
                write!(out, "{s}").map_err(w)?;
                OK
            }
            Err(e) => {
                writeln!(err, "{e}").map_err(w)?;
                // A program the checker refuses is status 1; an unknown target
                // or a construct an emitter does not translate is status 2.
                if ["form", "rust", "python", "js"].contains(&rest[0].as_str())
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
        Some("lsp") if rest.is_empty() => serve_dyn(input, out),
        Some("help") | Some("--help") => {
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
