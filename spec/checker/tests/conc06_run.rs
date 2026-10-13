//! Run-time meaning of the 0.6 constructs (the draft 0.6 concurrency chapter, not published here C.4 to
//! C.7) on the tick runtime, with property tests for T16 to T22.
#![cfg(feature = "conc06")]
use rce_spec_check::conc::{run_main, Outcome, ParMode, RunCfg, RunResult};
use rce_spec_check::parse::parse;

fn run_cfg(src: &str, cfg: &RunCfg) -> RunResult {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}\n{src}", d[0].1);
    let m = parse(src).unwrap();
    run_main(&m, cfg).unwrap()
}
fn run(src: &str) -> RunResult {
    run_cfg(src, &RunCfg::default())
}
fn outs(r: &RunResult) -> Vec<String> {
    r.outputs.iter().map(|v| v.show()).collect()
}
/// small deterministic generator for the property tests
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }
}

#[test]
fn spawn_and_channels() {
    let r = run(include_str!("../../examples/13_spawn_chan.rce"));
    assert_eq!(r.outcome, Outcome::Done);
    assert_eq!(outs(&r), vec!["4950"]);
}

#[test]
fn priority_spawn_runs() {
    let r = run(include_str!("../../conc06/accept/05_priority_spawn.rce"));
    assert_eq!(r.outcome, Outcome::Done);
    assert_eq!(outs(&r), vec!["some(1)"]);
}

#[test]
fn deadlock_is_reported() {
    let r = run("fn main()\n    let c = chan[i64](1)\n    let v = <- c\n    emit v\n");
    assert_eq!(r.outcome, Outcome::Deadlock);
}

const PAR: &str = "fn main()
    var a: i64 = 0
    var b: i64 = 0
    var c: i64 = 0
    par
        branch one
            a = 3
            for i in 0..5
                a += i
        branch two
            b = 7
        branch three
            c = 11
    emit a * 100 + b * 10 + c
";

/// T16: branches with disjoint footprints give the source-order result in
/// every interleaving (sequential, any permutation, OS threads).
#[test]
fn t16_par_mode_independent() {
    let want = outs(&run(PAR));
    assert_eq!(want, vec!["1381"]);
    for seed in 1..40 {
        let cfg = RunCfg {
            par_mode: ParMode::Permute(seed),
            ..RunCfg::default()
        };
        assert_eq!(outs(&run_cfg(PAR, &cfg)), want, "seed {seed}");
    }
    let cfg = RunCfg {
        par_mode: ParMode::Threads,
        ..RunCfg::default()
    };
    for _ in 0..10 {
        assert_eq!(outs(&run_cfg(PAR, &cfg)), want);
    }
}

/// T18: a `for .. par` reduction is the fixed tree T and is independent of
/// the execution mode; for an associative operator it equals the left fold.
#[test]
fn t18_tree_reduction() {
    let mut g = Rng(99);
    for _ in 0..25 {
        let n = g.int(1, 40);
        let xs: Vec<i64> = (0..n).map(|_| g.int(-1000, 1000)).collect();
        let lit = format!("{:?}", xs);
        let src = format!(
            "fn main()\n    let xs = {lit}\n    var s: i64 = 0\n    var p: i64 = 0\n    for x in xs par\n        s += x\n    for x in xs par\n        p ^= x\n    emit s\n    emit p\n"
        );
        let want_s: i64 = xs.iter().sum();
        let want_p = xs.iter().fold(0i64, |a, x| a ^ x);
        for pm in [ParMode::Seq, ParMode::Permute(g.next()), ParMode::Threads] {
            let cfg = RunCfg {
                par_mode: pm,
                ..RunCfg::default()
            };
            let r = run_cfg(&src, &cfg);
            assert_eq!(r.outcome, Outcome::Done, "{src}");
            assert_eq!(outs(&r), vec![want_s.to_string(), want_p.to_string()]);
        }
    }
}

/// T18 for floats: the tree order is fixed, so the result is bit-for-bit
/// the same in every mode even though float + is not associative.
#[test]
fn t18_float_tree_fixed() {
    let src = "fn main()\n    var s: f64 = 0.0\n    for i in 0..37 par\n        s += 1.0 / (i as f64 + 0.3)\n    emit s\n";
    let want = outs(&run(src));
    for seed in 0..10 {
        let cfg = RunCfg {
            par_mode: ParMode::Permute(seed),
            ..RunCfg::default()
        };
        assert_eq!(outs(&run_cfg(src, &cfg)), want);
    }
}

/// T20: the cost of a block is the sum of its parts, and a budget smaller
/// than the cost traps deterministically at the same point.
#[test]
fn t20_budget_composition() {
    let prog = |b: i64, n: i64| {
        format!(
            "fn main()\n    var n: i64 = 0\n    with budget {b}\n        for i in 0..{n}\n            n += 1\n    emit n\n"
        )
    };
    // find the least budget that suffices for n iterations; it is additive
    let least = |n: i64| {
        (0..2000)
            .find(|b| run(&prog(*b, n)).outcome == Outcome::Done)
            .unwrap()
    };
    let (c1, c2, c4) = (least(1), least(2), least(4));
    assert!(c1 > 0);
    assert_eq!(c4 - c2, 2 * (c2 - c1), "per-iteration cost is constant");
    let r1 = run(&prog(c2 - 1, 2));
    let r2 = run(&prog(c2 - 1, 2));
    assert!(matches!(&r1.outcome, Outcome::Refused(e) if e.contains("budget")), "{:?}", r1.outcome);
    assert_eq!(r1.outcome, r2.outcome);
}

const NONDET: &str = "fn feed(c: chan i64, base: i64)
    for i in 0..5
        c <- base + i
fn main()
    let a = chan[i64](8)
    let b = chan[i64](8)
    spawn feed(a, 100)
    spawn feed(b, 200)
    var got: i64 = 0
    loop bound 20
        select nondet
            recv v from a -> emit v
            recv w from b -> emit w
            after 3 -> break
";

/// T21: a run with `select nondet` is reproduced exactly by replaying its
/// logged choices; different seeds may choose differently.
#[test]
fn t21_replay() {
    let mut seen = std::collections::BTreeSet::new();
    for seed in 1..20 {
        let cfg = RunCfg {
            seed,
            ..RunCfg::default()
        };
        let r = run_cfg(NONDET, &cfg);
        assert_eq!(r.outcome, Outcome::Done);
        assert_eq!(r.outputs.len(), 10);
        let replay = RunCfg {
            seed: seed.wrapping_mul(7919) + 1,
            choices: Some(r.choices.clone()),
            ..RunCfg::default()
        };
        let r2 = run_cfg(NONDET, &replay);
        assert_eq!(outs(&r2), outs(&r), "seed {seed}");
        seen.insert(outs(&r));
    }
    assert!(seen.len() > 1, "nondet should vary with the seed");
}

/// T22: a deadline cancels at the first communication step after expiry and
/// takes the `else` branch; the outcome does not depend on the seed.
#[test]
fn t22_deadline() {
    let src = "fn main()
    let c = chan[i64](1)
    with deadline 4
        let v = <- c
        emit v
    else
        emit 0
    emit 9
";
    for seed in 0..5 {
        let cfg = RunCfg {
            seed,
            ..RunCfg::default()
        };
        let r = run_cfg(src, &cfg);
        assert_eq!(r.outcome, Outcome::Done);
        assert_eq!(outs(&r), vec!["0", "9"]);
        assert!(r.log.iter().any(|e| e.what.starts_with("Cancelled")));
    }
    let ok = "fn put(c: chan i64)\n    c <- 5\nfn main()\n    let c = chan[i64](1)\n    spawn put(c)\n    with deadline 10\n        match <- c\n            some(v) -> emit v\n            none -> emit 1\n    else\n        emit 0\n";
    assert_eq!(outs(&run(ok)), vec!["5"]);
}

/// T19a: identical programs and seeds give identical logs (determinism of
/// the runtime, including vector clocks and task ids).
#[test]
fn t19a_log_determinism() {
    let a = run(NONDET);
    let b = run(NONDET);
    let la: Vec<_> = a.log.iter().map(|e| (e.tick, e.what.clone(), e.vc.clone())).collect();
    let lb: Vec<_> = b.log.iter().map(|e| (e.tick, e.what.clone(), e.vc.clone())).collect();
    assert_eq!(la, lb);
}

#[test]
fn run_main_corpus() {
    let r = run(include_str!("../../conc06/accept/08_run_main.rce"));
    assert_eq!(outs(&r), vec!["42", "45", "[0, 1, 4, 9]"]);
}
