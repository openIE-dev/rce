//! Path-precise W-PAR-1 (spec C.3): branch footprints are borrow paths, so
//! branches writing disjoint fields, tuple slots or literal indices of one
//! variable are admitted (the earlier root-variable rule refused each of
//! these), while overlapping paths are still refused. Admitted programs
//! give the source-order result under every par schedule (T16).
use rce_spec_check::conc::{run_main, Outcome, ParMode, RunCfg};
use rce_spec_check::parse::parse;

const PRE: &str = "struct In\n    a: i64\n    b: i64\n\nstruct P\n    x: i64\n    y: i64\n    q: In\n    v: List[i64]\n\n";

fn prog(one: &str, two: &str) -> String {
    format!(
        "# rce-version: 0.6\n{PRE}fn main()\n    var p: P = P(x: 1, y: 2, q: In(a: 3, b: 4), v: [5, 6])\n    var t: (i64, i64) = (7, 8)\n    var arr: [i64; 4] = [0; 4]\n    let i: usize = 2\n    par\n        branch one\n{}\n        branch two\n{}\n    emit p.x + p.y + p.q.a + p.q.b + p.v[0] + p.v[1] + t.0 + t.1 + arr[0] + arr[1] + arr[2]\n",
        ind(one),
        ind(two)
    )
}
fn ind(s: &str) -> String {
    s.lines().map(|l| format!("            {l}")).collect::<Vec<_>>().join("\n")
}
fn diags(src: &str) -> Result<(), String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    d[0].1.clone().map_err(|e| format!("{e:?}"))
}

/// race-free under C.3 paths, refused by a per-root rule; expected sum
const ACCEPT: &[(&str, &str, i64)] = &[
    ("p.x = 10", "p.y = 20", 10 + 20 + 3 + 4 + 5 + 6 + 7 + 8),
    ("p.x = p.y + 1", "p.q.a = p.y * 2", 3 + 2 + 4 + 4 + 5 + 6 + 7 + 8),
    ("p.q.a = 30", "p.q.b = 40\np.x += 1", 2 + 2 + 30 + 40 + 5 + 6 + 7 + 8),
    ("t.0 = 70", "t.1 = 80", 1 + 2 + 3 + 4 + 5 + 6 + 70 + 80),
    ("arr[0] = 1", "arr[1] = 2\narr[2] = 3", 36 + 1 + 2 + 3),
    ("p.v[0] = 50", "p.v[1] = 60", 1 + 2 + 3 + 4 + 50 + 60 + 7 + 8),
    ("p.x = 9", "arr[i] = p.y", 9 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 2),
];

/// overlapping paths: still races
const REFUSE: &[(&str, &str)] = &[
    ("p.x = 1", "p.x = 2"),
    ("p.x = 1", "let z = &p\narr[0] = z.y"),
    ("p = P(x: 0, y: 0, q: In(a: 0, b: 0), v: [])", "arr[0] = p.y"),
    ("p.q.a = 1", "arr[0] = p.q.a"),
    ("p.q = In(a: 0, b: 0)", "p.q.b = 2"),
    ("arr[i] = 1", "arr[0] = 2"),
    ("arr[1] = 1", "arr[0] = arr[i]"),
    ("p.v.push(1)", "arr[0] = p.v[0]"),
    ("t.0 = 1", "arr[0] = t.0 + t.1"),
];

#[test]
fn disjoint_paths_are_admitted_and_deterministic() {
    for (one, two, want) in ACCEPT {
        let src = prog(one, two);
        diags(&src).unwrap_or_else(|e| panic!("refused: {e}\n{src}"));
        let m = parse(&src).unwrap();
        for pm in [ParMode::Seq, ParMode::Permute(3), ParMode::Permute(11), ParMode::Threads] {
            let r = run_main(&m, &RunCfg { par_mode: pm, ..RunCfg::default() }).unwrap();
            assert_eq!(r.outcome, Outcome::Done, "{one} | {two}");
            let got: Vec<String> = r.outputs.iter().map(|v| v.show()).collect();
            assert_eq!(got, vec![want.to_string()], "{one} | {two} under {pm:?}");
        }
    }
}

#[test]
fn overlapping_paths_are_refused() {
    for (one, two) in REFUSE {
        let src = prog(one, two);
        let e = diags(&src).expect_err(&format!("admitted a race:\n{src}"));
        assert!(e.contains("W-PAR-1"), "{one} | {two}: {e}");
        // the race is the only error: the rest of the program is well typed
        assert_eq!(e.matches("W-").count(), e.matches("W-PAR-1").count(), "{one} | {two}: {e}");
    }
}
