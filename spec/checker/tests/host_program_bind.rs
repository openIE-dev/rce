//! HC-27: a log is bound to the program that wrote it (`D.program`, the
//! module hash). A changed program is refused on that log by `rce host run`
//! and `rce host replay`; the same program in another layout is not a change.
use std::path::Path;
use std::process::Command;

const V1: &str = "module counter\n\nevent Inc\n\nstate C\n    n: i64 = 0\n\nstep C on Inc\n    n += 1\n";
const V2: &str = "module counter\n\nevent Inc\n\nstate C\n    n: i64 = 0\n\nstep C on Inc\n    n += 2\n";
// V1 with a comment and an extra blank line: the same canonical text
const V1_LAYOUT: &str = "# a counter\nmodule counter\n\n\nevent Inc\n\nstate C\n    n: i64 = 0\n\nstep C on Inc\n    n += 1\n";

fn rce(args: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(args).output().unwrap();
    (o.status.code().unwrap_or(-1), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
}

fn write(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn a_log_runs_and_replays_only_with_the_program_that_wrote_it() {
    let dir = std::env::temp_dir().join(format!("rce-hc27-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (v1, v2, v1b) = (write(&dir, "v1.rce", V1), write(&dir, "v2.rce", V2), write(&dir, "v1b.rce", V1_LAYOUT));
    let log = dir.join("d").to_string_lossy().into_owned();

    let (code, out) = rce(&["host", "run", &v1, &log, "Inc, Inc"]);
    assert_eq!(code, 0, "{out}");
    let bound = std::fs::read_to_string(format!("{log}.program")).unwrap();
    assert_eq!(bound.trim().len(), 64, "D.program holds the module hash: {bound}");

    let before = std::fs::read(&log).unwrap();
    let (code, out) = rce(&["host", "run", &v2, &log, "Inc"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("requires change") && out.contains("HC-27"), "{out}");
    assert_eq!(std::fs::read(&log).unwrap(), before, "a refused program appends nothing to D");

    let (code, out) = rce(&["host", "replay", &v2, &log]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("HC-27"), "{out}");

    let (code, out) = rce(&["host", "run", &v1b, &log, "Inc"]);
    assert_eq!(code, 0, "layout and comments are not a change: {out}");
    let (code, out) = rce(&["host", "replay", &v1, &log]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("n = 3"), "{out}");

    // a log without D.program (written before HC-27) is adopted by the
    // program that next runs on it
    std::fs::remove_file(format!("{log}.program")).unwrap();
    let (code, out) = rce(&["host", "run", &v1, &log, "Inc"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(std::fs::read_to_string(format!("{log}.program")).unwrap(), bound);
    let _ = std::fs::remove_dir_all(&dir);
}
