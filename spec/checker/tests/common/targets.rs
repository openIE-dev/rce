//! The emit targets a test run covers (spec 10.5, "How the emitters are
//! tested"), shared by every test that runs emitted programs (`#[path]`
//! module). The targets are named as data, one word per line, in
//! `tests/emit_targets.txt` (`rust`, `python`, `js`, `scala`; `#` starts a
//! comment). A test runs its emitted programs on every covered target it
//! uses and fails when a covered target's toolchain does not start; a
//! target the file does not name is reported as not covered by this run,
//! never as a pass. No environment variable or flag changes the targets a
//! run covers.
#![allow(dead_code)]
use rce_spec_check::emit::Lang;
use std::process::Command;
use std::sync::OnceLock;

/// The data file that names the covered targets (relative to spec/checker,
/// where cargo runs the tests).
pub const FILE: &str = "tests/emit_targets.txt";

/// Every emit target, in the order the tests use them.
pub const ALL: [Lang; 4] = [Lang::Rust, Lang::Python, Lang::Js, Lang::Scala];

/// The word that names `l` in the data file and in `rce emit`.
pub fn word(l: Lang) -> &'static str {
    match l {
        Lang::Rust => "rust",
        Lang::Python => "python",
        Lang::Js => "js",
        Lang::Scala => "scala",
    }
}

/// The targets the data file names.
pub fn named() -> &'static [Lang] {
    static N: OnceLock<Vec<Lang>> = OnceLock::new();
    N.get_or_init(|| {
        let text = std::fs::read_to_string(FILE).unwrap_or_else(|e| panic!("{FILE}: {e}"));
        let mut v = vec![];
        for l in text.lines() {
            let w = l.split('#').next().unwrap_or("").trim();
            if w.is_empty() {
                continue;
            }
            let lang = Lang::parse(w).unwrap_or_else(|| panic!("{FILE}: `{w}` is not an emit target (rust, python, js, scala)"));
            if !v.contains(&lang) {
                v.push(lang);
            }
        }
        assert!(!v.is_empty(), "{FILE} names no target, so no emitted program would run");
        v
    })
}

/// Of `langs`, the targets this run covers, in the order given. A target
/// the data file does not name is reported on standard error. For a named
/// target, `TOOL --version` must run, `tool` giving the program the test
/// runs that target with; otherwise the test fails here, naming the target.
pub fn covered(langs: &[Lang], tool: impl Fn(Lang) -> String) -> Vec<Lang> {
    let mut v = vec![];
    for &l in langs {
        if !named().contains(&l) {
            eprintln!("{}: not covered by this run ({FILE} does not name it); not a pass", word(l));
            continue;
        }
        let t = tool(l);
        match Command::new(&t).arg("--version").output() {
            Ok(o) if o.status.success() => v.push(l),
            Ok(o) => panic!(
                "{}: `{t} --version` exited with {:?}; {FILE} names {}, so its toolchain must run",
                word(l),
                o.status.code(),
                word(l)
            ),
            Err(e) => panic!("{}: `{t}` does not run ({e}); {FILE} names {}, so its toolchain must run", word(l), word(l)),
        }
    }
    v
}

/// Whether this run covers the target named `w` (as `covered`, for tests
/// that name targets by word), `tool` being the program that runs it.
pub fn covers(w: &str, tool: &str) -> bool {
    let l = Lang::parse(w).unwrap_or_else(|| panic!("`{w}` is not an emit target"));
    !covered(&[l], |_| tool.to_string()).is_empty()
}

/// The program that runs target `l` where a test reads its path from the
/// environment: `RCE_RUSTC`, `RCE_PYTHON`, `RCE_NODE` or `RCE_SCALA`, else
/// `rustc`, `python3`, `node` or `scala-cli` on the path. (These variables
/// choose an oracle's path only; they never change which targets a run
/// covers.)
pub fn oracle(l: Lang) -> String {
    let (var, def) = match l {
        Lang::Rust => ("RCE_RUSTC", "rustc"),
        Lang::Python => ("RCE_PYTHON", "python3"),
        Lang::Js => ("RCE_NODE", "node"),
        Lang::Scala => ("RCE_SCALA", "scala-cli"),
    };
    std::env::var(var).ok().filter(|v| !v.is_empty()).unwrap_or(def.to_string())
}
