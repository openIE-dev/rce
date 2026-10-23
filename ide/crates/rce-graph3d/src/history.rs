//! Version history replay from git (Code-XR / Gource lesson): one Form
//! graph per commit that touched `.rce` files, in commit order, so the
//! existing time scrubber replays how the program grew. Uses the `git`
//! binary already on the machine; nothing is linked.
use crate::time::Version;
use std::path::Path;
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let o = Command::new("git").arg("-C").arg(repo).args(args).output().map_err(|e| format!("git: {e}"))?;
    if !o.status.success() {
        return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&o.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).into_owned())
}

/// A program at one commit: (commit time in ms, "hash subject", files).
pub type Program = (i64, String, Vec<(String, String)>);

/// The `.rce` program at each of the last `max` commits that changed one.
pub fn programs_from_git(repo: &Path, max: usize) -> Result<Vec<Program>, String> {
    let n = format!("-{max}");
    let log = git(repo, &["log", &n, "--reverse", "--format=%H %ct %s", "--", "*.rce"])?;
    let mut out = vec![];
    for line in log.lines() {
        let mut p = line.splitn(3, ' ');
        let (Some(h), Some(t), s) = (p.next(), p.next(), p.next().unwrap_or("")) else { continue };
        let files = git(repo, &["ls-tree", "-r", "--name-only", h])?;
        let mut prog = vec![];
        for f in files.lines().filter(|f| f.ends_with(".rce")) {
            prog.push((f.to_string(), git(repo, &["show", &format!("{h}:{f}")])?));
        }
        out.push((t.parse::<i64>().unwrap_or(0) * 1000, format!("{} {}", &h[..h.len().min(8)], s), prog));
    }
    Ok(out)
}

/// Versions for the last `max` commits that changed `.rce` files.
pub fn from_git(repo: &Path, max: usize) -> Result<Vec<Version>, String> {
    Ok(programs_from_git(repo, max)?.into_iter().map(|(t_ms, label, prog)| {
        let g = crate::native::code_graph_of(&prog);
        Version { t_ms, label, graph: g }
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replays_commits() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        let g = |a: &[&str]| { assert!(Command::new("git").arg("-C").arg(r).args(a).output().unwrap().status.success(), "{a:?}") };
        g(&["init", "-q"]);
        g(&["config", "user.email", "t@t"]); g(&["config", "user.name", "t"]);
        std::fs::write(r.join("a.rce"), "fn f() -> i64\n    return 1\n").unwrap();
        g(&["add", "."]); g(&["commit", "-qm", "one"]);
        std::fs::write(r.join("a.rce"), "fn f() -> i64\n    return 1\n\nfn h() -> i64\n    return f()\n").unwrap();
        g(&["add", "."]); g(&["commit", "-qm", "two"]);
        let v = from_git(r, 10).unwrap();
        assert_eq!(v.len(), 2);
        assert!(v[0].label.ends_with("one") && v[1].label.ends_with("two"));
        let fns = |i: usize| v[i].graph.nodes.iter().filter(|n| n.kind == crate::graph::Kind::Fn).count();
        assert_eq!((fns(0), fns(1)), (1, 2));
    }
}
