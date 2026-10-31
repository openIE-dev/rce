//! Tests for each `rce` subcommand (spec section 10, book/12-tooling.md).
use rce_spec_check::cli;
use serde_json::json;
use std::collections::HashSet;

const COUNTER: &str = "../examples/01_counter.rce";
const HELLO: &str = "../examples/02_hello_main.rce";

/// `rce ARGS` in process, through the same entry point the binary uses.
fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn tmp(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const BAD: &str = "fn f() -> i64\n    \"text\"\n";

#[test]
fn check_accepts_or_requires_change() {
    assert_eq!(
        rce(&["check", COUNTER]),
        (
            0,
            format!(
                "ok {COUNTER}\nrecommended {COUNTER}: line 2: W-INT-2 the 3 example lines are the only obligations on the steps; \
                 their lookup-table control (6.9) meets them, so they do not constrain behaviour beyond the listed examples\n"
            ),
            String::new()
        )
    );
    // no example line, no recommendation (W-INT-2 needs example lines and a step)
    let p = tmp("plain.rce", "state C\n    n: u64 = 0\n\nevent Inc\n\nstep C on Inc -> n + 1\n");
    assert_eq!(rce(&["check", &p]), (0, format!("ok {p}\n"), String::new()));
    let p = tmp("bad.rce", BAD);
    let (c, out, _) = rce(&["check", &p]);
    assert_eq!(c, 1);
    assert!(out.starts_with(&format!("requires change {p}: line ")), "{out}");
    assert!(out.contains("W-TY-1"), "{out}");
}

#[test]
fn check_usage_and_missing_file() {
    assert_eq!(rce(&["check"]).0, 2);
    assert_eq!(rce(&["check", "/nonexistent/x.rce"]).0, 2);
    assert_eq!(rce(&[]).0, 2);
    assert_eq!(rce(&["help"]).0, 0);
}

#[test]
fn run_events_prints_each_state_and_goes_on_past_kept() {
    let (c, out, _) = rce(&["run", COUNTER, "Inc, Inc, Dec"]);
    assert_eq!(c, 0);
    assert_eq!(
        out,
        "0 (initial) Counter { n = 0 }\n1 Inc Counter { n = 1 }\n2 Inc Counter { n = 2 }\n3 Dec Counter { n = 1 }\n"
    );
    let (c, out, _) = rce(&["run", COUNTER, "Dec, Inc"]);
    assert_eq!(c, 1);
    assert_eq!(
        out,
        "0 (initial) Counter { n = 0 }\n1 Dec kept: Underflow\n2 Inc Counter { n = 1 }\n"
    );
}

#[test]
fn run_main_and_errors() {
    assert_eq!(
        rce(&["run", HELLO]),
        (0, "emit \"hello, world\"\n".into(), String::new())
    );
    let (c, out, _) = rce(&["run", COUNTER]);
    assert_eq!(c, 1);
    assert!(out.contains("nothing to run"));
    let (c, out, _) = rce(&["run", COUNTER, "Nope"]);
    assert_eq!(c, 1);
    assert!(out.contains("Nope"), "{out}");
}

#[test]
fn run_is_deterministic() {
    let a = rce(&["run", COUNTER, "Inc, Dec, Inc"]);
    let b = rce(&["run", COUNTER, "Inc, Dec, Inc"]);
    assert_eq!(a, b);
}

#[test]
fn test_reports_counts_and_failures() {
    let (c, out, _) = rce(&["test", COUNTER]);
    assert_eq!(c, 0);
    assert_eq!(
        out,
        format!("ok {COUNTER} (3 examples, 0 checks)\n1 passed, 0 failed\n")
    );
    let wrong = std::fs::read_to_string(COUNTER)
        .unwrap()
        .replace("-> 1", "-> 2");
    let p = tmp("wrong.rce", &wrong);
    let (c, out, _) = rce(&["test", COUNTER, &p]);
    assert_eq!(c, 1);
    assert!(
        out.contains(&format!("FAIL {p}: ")) && out.contains("W-INT-1"),
        "{out}"
    );
    assert!(out.ends_with("1 passed, 1 failed\n"));
}

#[test]
fn emit_targets_and_unknown_target() {
    let (c, out, _) = rce(&["emit", "form", COUNTER]);
    assert_eq!(c, 0);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["S"]["s0"], json!({"Counter": {"n": 0}}));
    for t in ["rust", "python", "js"] {
        let (c, out, _) = rce(&["emit", t, COUNTER]);
        assert_eq!(c, 0, "{t}");
        assert!(
            out.contains("run_example") || out.contains("runExample"),
            "{t}"
        );
    }
    let (c, out, err) = rce(&["emit", "cobol", COUNTER]);
    assert_eq!((c, out.as_str()), (2, ""));
    assert!(err.contains("unknown emit target"));
    assert_eq!(rce(&["emit", "form", &tmp("bad2.rce", BAD)]).0, 1);
}

const MESSY: &str = "\n\nstate Counter\n  n: u64 = 0   \n\n\n\nevent Inc\n# a comment\nstep Counter on Inc -> n + 1\n\n\n";
const TIDY: &str =
    "state Counter\n    n: u64 = 0\n\nevent Inc\n# a comment\nstep Counter on Inc -> n + 1\n";

#[test]
fn fmt_canonical_layout_idempotent_and_token_preserving() {
    assert_eq!(cli::fmt_text(MESSY).unwrap(), TIDY);
    assert_eq!(cli::fmt_text(TIDY).unwrap(), TIDY);
    let nested = "fn f(a: i64,\n      b: i64) -> i64\n  if a > b\n        a\n  else\n        b\n";
    let n = cli::fmt_text(nested).unwrap();
    assert_eq!(
        n,
        "fn f(a: i64,\n      b: i64) -> i64\n    if a > b\n        a\n    else\n        b\n"
    );
    assert_eq!(cli::fmt_text(&n).unwrap(), n);
    assert!(cli::fmt_text("state S\n    a: u64 = 0\n  b: u64 = 0\n").is_err());
    for e in std::fs::read_dir("../examples").unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "rce") {
            let s = std::fs::read_to_string(&p).unwrap();
            if let Ok(f) = cli::fmt_text(&s) {
                assert_eq!(cli::fmt_text(&f).unwrap(), f, "{}", p.display());
            }
        }
    }
}

#[test]
fn fmt_command_check_and_write() {
    let p = tmp("messy.rce", MESSY);
    assert_eq!(
        rce(&["fmt", "--check", &p]),
        (1, format!("would reformat {p}\n"), String::new())
    );
    assert_eq!(std::fs::read_to_string(&p).unwrap(), MESSY);
    assert_eq!(rce(&["fmt", &p]).0, 0);
    assert_eq!(std::fs::read_to_string(&p).unwrap(), TIDY);
    assert_eq!(
        rce(&["fmt", "--check", &p]),
        (0, String::new(), String::new())
    );
}

#[test]
fn diagnostics_carry_rule_and_line() {
    let d = cli::diagnostics(BAD, &HashSet::new());
    assert_eq!(d.len(), 1);
    assert!(d[0].1.contains("W-TY-1"));
    assert_eq!(cli::line_of("line 7: W-X"), 7);
    assert_eq!(cli::line_of("no line"), 1);
}

fn frame(v: &serde_json::Value) -> String {
    let s = v.to_string();
    format!("Content-Length: {}\r\n\r\n{s}", s.len())
}

#[test]
fn lsp_session_over_stdio() {
    let msgs = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///a.rce","languageId":"rce","version":1,"text":BAD}}}),
        json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///a.rce","version":2},"contentChanges":[{"text":MESSY}]}}),
        json!({"jsonrpc":"2.0","id":2,"method":"textDocument/formatting","params":{"textDocument":{"uri":"file:///a.rce"},"options":{}}}),
        json!({"jsonrpc":"2.0","id":3,"method":"textDocument/rename","params":{}}),
        json!({"jsonrpc":"2.0","id":4,"method":"shutdown"}),
        json!({"jsonrpc":"2.0","method":"exit"}),
    ];
    let input: String = msgs.iter().map(frame).collect();
    let mut out = vec![];
    let code = cli::serve(&mut input.as_bytes(), &mut out);
    assert_eq!(code, 0);
    let mut r = &out[..];
    let mut got = vec![];
    while let Some(m) = cli::read_msg(&mut r) {
        got.push(m);
    }
    assert_eq!(got.len(), 6, "{got:?}");
    assert_eq!(got[0]["result"]["capabilities"]["textDocumentSync"], 1);
    let d = &got[1]["params"]["diagnostics"];
    assert_eq!(d.as_array().unwrap().len(), 1);
    assert!(d[0]["message"].as_str().unwrap().contains("W-TY-1"));
    assert_eq!(got[2]["params"]["diagnostics"], json!([]));
    assert_eq!(got[3]["result"][0]["newText"], TIDY);
    assert_eq!(got[4]["error"]["code"], -32601);
    assert_eq!(got[5]["result"], serde_json::Value::Null);
}

#[test]
fn lsp_exit_without_shutdown_is_1_and_command_serves() {
    let input = frame(&json!({"jsonrpc":"2.0","method":"exit"}));
    assert_eq!(cli::serve(&mut input.as_bytes(), &mut vec![]), 1);
    let s = [
        json!({"jsonrpc":"2.0","id":1,"method":"shutdown"}),
        json!({"jsonrpc":"2.0","method":"exit"}),
    ]
    .iter()
    .map(frame)
    .collect::<String>();
    let mut out = vec![];
    let code = cli::main(&["lsp".to_string()], &mut s.as_bytes(), &mut out, &mut vec![]);
    assert_eq!(code, 0);
    assert!(String::from_utf8(out).unwrap().contains("\"result\":null"));
}
