//! Live while incomplete (lesson from Hazel, at a coarser grain). When a
//! file does not pass `rce check`, each top-level item that cannot be
//! admitted becomes a hole and the rest of the program stays live: the
//! graph, E_sim and tests keep working on the admitted items. Hazel puts
//! typed holes inside expressions and keeps even the broken part running;
//! this is item-level only, and items that depend on a hole are holed too.
#[derive(Debug, Clone, PartialEq)]
pub struct Hole {
    pub file: String,
    /// 1-based first line and line count of the holed item.
    pub line: usize,
    pub lines: usize,
    pub why: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Live {
    /// The admitted program: holed items blanked out (line numbers kept).
    pub program: Vec<(String, String)>,
    pub holes: Vec<Hole>,
}

fn items(src: &str) -> Vec<(usize, usize)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut v = vec![];
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().is_empty() || lines[i].starts_with([' ', '\t']) { i += 1; continue }
        let s = i;
        i += 1;
        while i < lines.len() && (lines[i].trim().is_empty() || lines[i].starts_with([' ', '\t'])) { i += 1 }
        let mut e = i;
        while e > s + 1 && lines[e - 1].trim().is_empty() { e -= 1 }
        v.push((s, e));
    }
    v
}

fn blank(src: &str, holes: &[(usize, usize)]) -> String {
    src.lines().enumerate()
        .map(|(i, l)| if holes.iter().any(|&(a, b)| i >= a && i < b) { "" } else { l })
        .collect::<Vec<_>>().join("\n") + "\n"
}

fn errors(prog: &[(String, String)]) -> Vec<(String, String)> {
    rce_spec_check::check_program(prog, false).into_iter().filter_map(|(p, r)| r.err().map(|e| (p, e))).collect()
}

pub fn live(files: &[(String, String)]) -> Live {
    let mut holed: Vec<Vec<(usize, usize)>> = vec![vec![]; files.len()];
    let mut why: Vec<Vec<String>> = vec![vec![]; files.len()];
    let spans: Vec<Vec<(usize, usize)>> = files.iter().map(|(_, s)| items(s)).collect();
    let build = |h: &Vec<Vec<(usize, usize)>>| -> Vec<(String, String)> {
        files.iter().zip(h).map(|((p, s), h)| (p.clone(), blank(s, h))).collect()
    };
    // Syntax first: an item that does not parse on its own is a hole (a
    // parser's error line can land far past the real cause).
    for (fi, (_, src)) in files.iter().enumerate() {
        let lines: Vec<&str> = src.lines().collect();
        for &(a, b) in &spans[fi] {
            if let Err(e) = rce_spec_check::parse::parse(&(lines[a..b].join("\n") + "\n")) {
                holed[fi].push((a, b));
                why[fi].push(e);
            }
        }
    }
    // Then checking: hole the item an error points at, until it passes.
    for _ in 0..spans.iter().map(Vec::len).sum::<usize>() + 1 {
        let prog = build(&holed);
        let errs = errors(&prog);
        if errs.is_empty() { break }
        let mut progress = false;
        for (p, e) in errs {
            let Some(fi) = files.iter().position(|(q, _)| *q == p) else { continue };
            let line = rce_spec_check::cli::line_of(&e).max(1) - 1;
            let target = spans[fi].iter().copied()
                .filter(|sp| !holed[fi].contains(sp))
                .find(|&(a, b)| line >= a && line < b)
                .or_else(|| spans[fi].iter().copied().rev().find(|sp| !holed[fi].contains(sp) && sp.0 <= line));
            if let Some(sp) = target {
                holed[fi].push(sp);
                why[fi].push(e);
                progress = true;
            }
        }
        if !progress { break }
    }
    let mut out = Live { program: build(&holed), holes: vec![] };
    for (fi, hs) in holed.iter().enumerate() {
        for (k, &(a, b)) in hs.iter().enumerate() {
            out.holes.push(Hole { file: files[fi].0.clone(), line: a + 1, lines: b - a, why: why[fi][k].clone() });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broken_item_is_a_hole_and_the_rest_stays_live() {
        let src = "fn ok(x: i64) -> i64\n    return x + 1\n\nfn broken() -> i64\n    return nope(\n\nfn also_ok() -> i64\n    return ok(1)\n";
        let l = live(&[("a.rce".into(), src.into())]);
        assert_eq!(l.holes.len(), 1, "{:?}", l.holes);
        assert_eq!(l.holes[0].line, 4);
        assert!(errors(&l.program).is_empty());
        assert!(l.program[0].1.contains("fn also_ok"));
        let m = rce_spec_check::parse::parse(&l.program[0].1).unwrap();
        let t = crate::cost::Table::default();
        let e = crate::cost::e_sim(&crate::cost::analyze(&m, &t), &t);
        assert!(e.contains_key("also_ok") && !e.contains_key("broken"));
    }
    #[test]
    fn clean_program_has_no_holes() {
        let l = live(&[("a.rce".into(), "fn f() -> i64\n    return 1\n".into())]);
        assert!(l.holes.is_empty());
    }
}
