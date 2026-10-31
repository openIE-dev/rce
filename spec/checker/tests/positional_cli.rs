//! Positional command lines (spec/proposals/positional-cli.md, section 4.3):
//! every usage line has no argument that starts with `-`; every usage line,
//! read back through the table of PC-8, names the command it came from;
//! each removed flag, given in its old place, exits 2, prints the one line
//! that names its new spelling (decision 2 of section 5) and writes no
//! file, in `cli::main`, in each binary of this crate and in an emitted
//! program of every covered target; PC-7 reads `./fuel` as a file and
//! `fuel` as the word.
#[path = "common/targets.rs"]
mod targets;

use rce_spec_check::args::{Parsed, find, parse};
use rce_spec_check::cli::{COMMANDS, USAGE_TEXT};
use rce_spec_check::emit::Lang;
use std::path::{Path, PathBuf};
use std::process::Command;

fn strs(a: &[&str]) -> Vec<String> {
    a.iter().map(|x| x.to_string()).collect()
}

fn cli(a: &[&str]) -> (i32, String, String) {
    let (mut out, mut err) = (vec![], vec![]);
    let c = rce_spec_check::cli::main(&strs(a), &mut &b""[..], &mut out, &mut err);
    (c, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

/// A new, empty directory for one test.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-pcli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn files_in(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

/// Run a binary in `dir` and return (status, stdout, stderr).
fn bin(exe: &str, dir: &Path, a: &[&str]) -> (Option<i32>, String, String) {
    let o = Command::new(exe).args(a).current_dir(dir).output().unwrap();
    (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// The arguments of a usage line's synopsis (the part before the two
/// spaces that start what the command does).
fn synopsis_args(line: &str) -> Vec<&str> {
    line.trim_start().split("  ").next().unwrap().split_whitespace().collect()
}

#[test]
fn usage_lines_have_no_argument_that_starts_with_a_dash() {
    for l in USAGE_TEXT.lines().filter(|l| l.starts_with("  rce ")) {
        for a in synopsis_args(l) {
            assert!(!a.starts_with('-') && !a.starts_with("[-"), "{l}");
        }
    }
    // the other binaries print their usage on a usage error
    let d = scratch("usage");
    for (exe, a) in [
        (env!("CARGO_BIN_EXE_rce-book"), vec![]),
        (env!("CARGO_BIN_EXE_rce-check"), vec![]),
        (env!("CARGO_BIN_EXE_rce-spec-data"), vec!["x", "y"]),
        (env!("CARGO_BIN_EXE_rce-spec-check"), vec!["x", "y"]),
    ] {
        let (c, _, err) = bin(exe, &d, &a);
        assert_eq!(c, Some(2), "{exe}: {err}");
        assert!(err.contains("usage"), "{exe}: {err}");
        for l in err.lines() {
            for w in l.split_whitespace() {
                assert!(!w.starts_with('-') && !w.starts_with("[-"), "{exe}: {l}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// The arguments a usage line describes, with each value and operand
/// replaced by a placeholder: the path, every word with its values, then
/// the command's least number of operands.
fn example(c: &rce_spec_check::args::Command) -> Vec<String> {
    let mut a: Vec<String> = c.path.iter().map(|p| p.to_string()).collect();
    for w in c.words {
        a.push(w.name.to_string());
        // PC-4: after `tls`, a `peer` takes its address and key
        let vals = match w.after {
            Some((f, then)) if a.iter().any(|x| x == f) => then,
            _ => w.values,
        };
        a.extend(vals.iter().map(|v| format!("V{v}").to_lowercase().replace(['|', '.'], "x")));
    }
    a.extend((0..c.min.max(1).min(c.max)).map(|i| format!("op{i}.rce")));
    a
}

#[test]
fn every_usage_line_reads_back_to_its_command() {
    for c in COMMANDS {
        let syn = c.synopsis("rce");
        assert!(USAGE_TEXT.lines().any(|l| l.trim_start().starts_with(&syn)), "{syn} is not in USAGE_TEXT");
        let a = example(c);
        let (got, rest) = find(COMMANDS, &a).unwrap_or_else(|| panic!("{a:?} names no command"));
        assert_eq!(got.path, c.path, "{a:?}");
        let p = parse(got, rest, "rce").unwrap_or_else(|e| panic!("{a:?}: {e}"));
        assert_eq!(p.words.len(), c.words.len(), "{a:?}");
        assert_eq!(p.operands.len(), c.min.max(1).min(c.max), "{a:?}");
    }
}

/// Every flag of section 1.1, in its old place, and the line that names its
/// new spelling (decision 2; decision 4 for `--key` and `--peer`).
const RCE_REMOVED: &[(&[&str], &str)] = &[
    (&["trace", "--json", "a.rce"], "--json was removed; write: rce trace a.rce"),
    (&["dim", "--json", "a.rce"], "--json was removed; write: rce dim json a.rce"),
    (&["equiv", "--json", "a.rce", "b.rce"], "--json was removed; write: rce equiv json a.rce b.rce"),
    (&["trace", "a.rce", "--streams", "s.txt"], "--streams was removed; write: rce trace streams a.rce s.txt"),
    (&["trace", "--json", "a.rce", "--streams", "s.txt"], "--json was removed; write: rce trace streams a.rce s.txt"),
    (&["fmt", "--check", "a.rce"], "--check was removed; write: rce fmt check a.rce"),
    (&["fmt", "--nameless", "a.rce"], "--nameless was removed; write: rce fmt nameless a.rce"),
    (&["change", "--fuel", "1000", "a.rce", "b.rce"], "--fuel was removed; write: rce change fuel 1000 a.rce b.rce"),
    (&["dim", "--fuel=1000", "a.rce"], "--fuel was removed; write: rce dim fuel 1000 a.rce"),
    (&["dim", "--control", "a.rce"], "--control was removed; write: rce dim control a.rce"),
    (&["fingerprint", "--depth", "3", "a.rce"], "--depth was removed; write: rce fingerprint depth 3 a.rce"),
    (&["index", "--vectors", "4", "a.rce"], "--vectors was removed; write: rce index vectors 4 a.rce"),
    (&["manifest-check", "m.json", "--advisories", "adv.json"], "--advisories was removed; write: rce manifest-check advisories adv.json m.json"),
    (&["manifest-check", "m.json", "--key", "acme=ab12"], "--key was removed; write: rce manifest-check key acme ab12 m.json"),
    (&["host", "run", "a.rce", "--log", "d.log", "Inc"], "--log was removed; write: rce host run a.rce d.log Inc"),
    (&["host", "run", "a.rce", "--fs", "root", "--log", "d.log"], "--fs was removed; write: rce host run fs root a.rce d.log"),
    (&["host", "run", "a.rce", "--blob", "b", "--log", "d.log"], "--blob was removed; write: rce host run blob b a.rce d.log"),
    (&["host", "run", "a.rce", "--media", "m", "--log", "d.log"], "--media was removed; write: rce host run media m a.rce d.log"),
    (&["host", "run", "a.rce", "--wasm", "w", "--log", "d.log"], "--wasm was removed; write: rce host run wasm w a.rce d.log"),
    (&["host", "run", "a.rce", "--net", "127.0.0.1:9000", "--log", "d.log"], "--net was removed; write: rce host run net 127.0.0.1:9000 a.rce d.log"),
    (
        &["host", "serve", "a.rce", "--chan", "127.0.0.1:7001", "--tls", "--peer", "127.0.0.1:7002=9f0c", "--log", "d.log"],
        "--chan was removed; write: rce host serve chan 127.0.0.1:7001 tls peer 127.0.0.1:7002 9f0c a.rce d.log",
    ),
    (&["host", "serve", "a.rce", "--tls", "--log", "d.log"], "--tls was removed; write: rce host serve tls a.rce d.log"),
    (&["host", "serve", "a.rce", "--peer", "127.0.0.1:7002", "--log", "d.log"], "--peer was removed; write: rce host serve peer 127.0.0.1:7002 a.rce d.log"),
    (&["host", "serve", "a.rce", "--pty", "--log", "d.log"], "--pty was removed; write: rce host serve pty a.rce d.log"),
    (&["host", "serve", "a.rce", "--http", "0.0.0.0:8080", "--log", "d.log"], "--http was removed; write: rce host serve http 0.0.0.0:8080 a.rce d.log"),
    (&["host", "replay", "a.rce", "--redact", "priv,ek", "--log", "d.log"], "--redact was removed; write: rce host replay redact priv,ek a.rce d.log"),
    (&["host", "tls-id", "a.rce", "--log", "d.log"], "--log was removed; write: rce host tls-id a.rce d.log"),
    (&["--help"], "--help was removed; write: rce help"),
    (&["fmt", "--help"], "--help was removed; write: rce help"),
    (&["-h"], "-h was removed; write: rce help"),
    (&["--version"], "--version was removed; write: rce version"),
    (&["-V"], "-V was removed; write: rce version"),
];

#[test]
fn every_removed_rce_flag_names_its_new_spelling_and_runs_nothing() {
    let d = scratch("removed");
    for (a, line) in RCE_REMOVED {
        let (c, out, err) = cli(a);
        assert_eq!((c, out.as_str(), err.as_str()), (2, "", format!("{line}\n").as_str()), "cli::main {a:?}");
        let (c, out, err) = bin(env!("CARGO_BIN_EXE_rce"), &d, a);
        assert_eq!((c, out.as_str(), err.as_str()), (Some(2), "", format!("{line}\n").as_str()), "rce {a:?}");
        assert!(files_in(&d).is_empty(), "rce {a:?} wrote {:?}", files_in(&d));
    }
    // PC-6: any other argument that starts with `-` is a usage error
    let (c, _, err) = cli(&["check", "-x.rce"]);
    assert_eq!(c, 2);
    assert!(err.contains("PC-6") && err.contains("./-x.rce"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn every_removed_flag_of_the_other_binaries_names_its_new_spelling() {
    let d = scratch("other");
    for (exe, a, line) in [
        (env!("CARGO_BIN_EXE_rce-book"), vec!["build", "--check"], "--check was removed; write: rce-book build check"),
        (env!("CARGO_BIN_EXE_rce-spec-data"), vec!["--check"], "--check was removed; write: rce-spec-data check"),
        (env!("CARGO_BIN_EXE_rce-spec-check"), vec!["--no-generate"], "--no-generate was removed; write: rce-spec-check no-generate"),
        (env!("CARGO_BIN_EXE_rce-check"), vec!["--parse-only", "a.rce"], "--parse-only was removed; write: rce-check parse-only a.rce"),
    ] {
        let (c, out, err) = bin(exe, &d, &a);
        assert_eq!((c, out.as_str(), err.as_str()), (Some(2), "", format!("{line}\n").as_str()), "{exe} {a:?}");
        assert!(files_in(&d).is_empty(), "{exe} {a:?} wrote {:?}", files_in(&d));
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn help_and_version_exit_0() {
    let (c, out, _) = cli(&["help"]);
    assert_eq!(c, 0);
    assert!(out.starts_with("usage:"), "{out}");
    let (c, out, _) = cli(&["version"]);
    assert_eq!(c, 0);
    assert!(out.starts_with("rce commit "), "{out}");
}

fn read(a: &[&str]) -> Result<Parsed, String> {
    let a = strs(a);
    let (c, rest) = find(COMMANDS, &a).ok_or("no command")?;
    parse(c, rest, "rce")
}

#[test]
fn words_come_before_operands() {
    let p = read(&["dim", "fuel", "500", "control", "json", "a.rce"]).unwrap();
    assert!(p.has("json") && p.has("control"));
    assert_eq!(p.value("fuel"), Some("500"));
    assert_eq!(p.operands, strs(&["a.rce"]));
    // PC-1: once operands begin, a word is an operand
    assert!(read(&["dim", "a.rce", "json"]).is_err());
    let p = read(&["host", "serve", "http", "0.0.0.0:8080", "fs", "/srv", "notes.rce", "/data/notes"]).unwrap();
    assert_eq!(p.value("http"), Some("0.0.0.0:8080"));
    assert_eq!(p.operands, strs(&["notes.rce", "/data/notes"]));
    // the optional EVENTS of `host run` is never read as a grant
    let p = read(&["host", "run", "a.rce", "d.log", "pty"]).unwrap();
    assert!(!p.has("pty"));
    assert_eq!(p.operands, strs(&["a.rce", "d.log", "pty"]));
}

#[test]
fn repeatable_words_and_words_given_once() {
    let h = "ab".repeat(32);
    let p = read(&["manifest-check", "key", "a", &h, "key", "b", &h, "m.json"]).unwrap();
    assert_eq!(p.all("key").len(), 2);
    assert_eq!(p.all("key")[1], strs(&["b", &h]).as_slice());
    let p = read(&["host", "run", "net", "127.0.0.1:1", "net", "127.0.0.1:2", "a.rce", "d.log"]).unwrap();
    assert_eq!(p.all("net").len(), 2);
    let e = read(&["dim", "json", "json", "a.rce"]).unwrap_err();
    assert!(e.contains("given twice"), "{e}");
    let (c, _, err) = cli(&["dim", "fuel", "9", "fuel", "9", "a.rce"]);
    assert_eq!(c, 2, "{err}");
}

#[test]
fn tls_comes_before_the_first_peer() {
    let (k1, k2) = ("11".repeat(32), "22".repeat(32));
    let p = read(&["host", "serve", "chan", "0.0.0.0:7001", "tls", "peer", "10.0.0.2:7001", &k1, "peer", "10.0.0.3:7001", &k2, "apps/chat.rce", "a.log"]).unwrap();
    assert_eq!(p.all("peer"), vec![strs(&["10.0.0.2:7001", &k1]).as_slice(), strs(&["10.0.0.3:7001", &k2]).as_slice()]);
    assert_eq!(p.operands, strs(&["apps/chat.rce", "a.log"]));
    // without tls, a peer takes the address alone
    let p = read(&["host", "serve", "chan", "127.0.0.1:7001", "peer", "127.0.0.1:7002", "apps/chat.rce", "a.log"]).unwrap();
    assert_eq!(p.all("peer"), vec![strs(&["127.0.0.1:7002"]).as_slice()]);
    // PC-4: tls after a peer is a usage error
    let e = read(&["host", "serve", "peer", "127.0.0.1:7002", "tls", "apps/chat.rce", "a.log"]).unwrap_err();
    assert!(e.contains("PC-4"), "{e}");
    let (c, out, err) = cli(&["host", "serve", "peer", "127.0.0.1:7002", "tls", "apps/chat.rce", "a.log"]);
    assert_eq!((c, out.as_str()), (2, ""), "{err}");
}

#[test]
fn values_joined_with_equals_become_two_arguments() {
    let (c, _, err) = cli(&["manifest-check", "m.json", "--key", "acme=0a0b"]);
    assert_eq!(c, 2);
    assert_eq!(err, "--key was removed; write: rce manifest-check key acme 0a0b m.json\n");
    let (c, _, err) = cli(&["host", "run", "a.rce", "--log", "d.log", "--chan", "h:1", "--tls", "--peer", "h:2=0a0b"]);
    assert_eq!(c, 2);
    assert_eq!(err, "--log was removed; write: rce host run chan h:1 tls peer h:2 0a0b a.rce d.log\n");
}

#[test]
fn a_file_named_like_a_word_is_written_with_its_directory() {
    let d = scratch("pc7");
    std::fs::copy("../examples/01_counter.rce", d.join("fuel")).unwrap();
    let rce = env!("CARGO_BIN_EXE_rce");
    // `./fuel` has a `/`, so it is never a word: it is the FILE
    let (c, out, err) = bin(rce, &d, &["dim", "./fuel"]);
    assert_eq!(c, Some(0), "{err}");
    assert!(out.contains("dim"), "{out}");
    // `fuel` is the word, and the word takes a value
    let (c, out, err) = bin(rce, &d, &["dim", "fuel"]);
    assert_eq!((c, out.as_str()), (Some(2), ""), "{err}");
    assert!(err.contains("`fuel` takes N"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// The flags of section 1.3, in their old form, and the words that replace them.
const EMITTED_REMOVED: &[(&[&str], &str)] = &[
    (&["--run"], "run"),
    (&["--trace"], "trace"),
    (&["--fuel=1000"], "fuel 1000"),
    (&["--fuel-trace"], "fuel-trace"),
    (&["--fuel-batch=10,20"], "fuel-batch 10,20"),
    (&["--link=1,partition,a,b"], "link 1,partition,a,b"),
    (&["--actor-log"], "actor-log"),
    (&["--restart-log"], "restart-log"),
    (&["--restart=1|S|c|id|0|"], "restart 1|S|c|id|0|"),
    (&["--restart-replay"], "restart-replay"),
    (&["--restart-tree=S/c"], "restart-tree S/c"),
    (&["--event=1|Inc|"], "event 1|Inc|"),
    (&["run", "--trace"], "run trace"),
];

fn tool(l: Lang) -> String {
    match l {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
        Lang::Scala => "scala-cli",
    }
    .to_string()
}

/// The emitted counter, ready to run on `l`: (program, arguments before the words).
fn emitted(l: Lang, d: &Path) -> (String, Vec<String>) {
    let src = std::fs::read_to_string("../examples/01_counter.rce").unwrap();
    let code = rce_spec_check::emit::emit(&rce_spec_check::parse::parse(&src).unwrap(), l).unwrap();
    match l {
        Lang::Rust => {
            let (rs, exe) = (d.join("counter.rs"), d.join("counter"));
            std::fs::write(&rs, code).unwrap();
            let o = Command::new("rustc").args(["--edition", "2024", "-A", "warnings", "-o"]).arg(&exe).arg(&rs).output().unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            (exe.to_string_lossy().into_owned(), vec![])
        }
        Lang::Python => {
            let p = d.join("counter.py");
            std::fs::write(&p, code).unwrap();
            ("python3".into(), vec![p.to_string_lossy().into_owned()])
        }
        Lang::Js => {
            let p = d.join("counter.mjs");
            std::fs::write(&p, code).unwrap();
            ("node".into(), vec![p.to_string_lossy().into_owned()])
        }
        Lang::Scala => unreachable!("scala is not run here"),
    }
}

#[test]
fn every_removed_flag_of_an_emitted_program_names_its_new_spelling() {
    for l in targets::covered(&[Lang::Rust, Lang::Python, Lang::Js], tool) {
        let d = scratch(&format!("emit-{}", targets::word(l)));
        let (prog, pre) = emitted(l, &d);
        let before = files_in(&d);
        let run = |a: &[&str]| {
            let o = Command::new(&prog).args(&pre).args(a).current_dir(&d).output().unwrap();
            (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
        };
        for (a, new) in EMITTED_REMOVED {
            let (c, out, err) = run(a);
            let flag = a.iter().find(|x| x.starts_with('-')).unwrap().split('=').next().unwrap();
            assert_eq!((c, out.as_str()), (Some(2), ""), "{l:?} {a:?}: {err}");
            assert!(err.starts_with(&format!("{flag} was removed; write: ")) && err.trim_end().ends_with(&format!(" {new}")), "{l:?} {a:?}: {err}");
        }
        // an argument that is not a word is a usage error; words run
        let (c, out, _) = run(&["counter.rce"]);
        assert_eq!((c, out.as_str()), (Some(2), ""), "{l:?}");
        let (c, plain, _) = run(&[]);
        assert_eq!(c, Some(0), "{l:?}");
        let (c, traced, err) = run(&["trace", "fuel-trace", "fuel", "1000000"]);
        assert_eq!(c, Some(0), "{l:?}: {err}");
        assert!(traced.lines().any(|x| x.starts_with("fuel ")) && traced.len() > plain.len(), "{l:?}");
        assert_eq!(files_in(&d), before, "{l:?} wrote a file");
        let _ = std::fs::remove_dir_all(&d);
    }
}
