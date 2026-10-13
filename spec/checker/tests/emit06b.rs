//! Emitters on `with budget`, `with group`, lvars, `atomic`, `with deadline`
//! and `select nondet` (the draft 0.6 concurrency chapter, not published here C.6, C.11.1, C.11.8,
//! C.11.10): each program is emitted to Rust, Python and JavaScript, run,
//! and compared line for line with the reference evaluator (`observe`, the
//! comptime model the emitters follow).
#![cfg(feature = "conc06")]
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use std::process::Command;

fn tool(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
    }
}

fn run(lang: Lang, tag: &str, code: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("rce-emit06b-{tag}-{lang:?}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("p.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join("p");
            let c = Command::new("rustc")
                .args(["--edition", "2024", "-O", "-o"])
                .arg(&exe)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(String::from_utf8_lossy(&c.stderr).to_string());
            }
            Command::new(&exe).output()
        }
        _ => Command::new(tool(lang)).arg(&src).output(),
    }
    .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        return Err(format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}

/// admitted by rce-check, then every available target prints the evaluator's lines
fn agree(tag: &str, src: &str) -> Vec<String> {
    let d = rce_spec_check::check_program(&[("e.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "{tag}: {:?}\n{src}", d[0].1);
    let m = with_probes(&parse(src).unwrap());
    let want = observe(&m);
    for lang in [Lang::Rust, Lang::Python, Lang::Js] {
        if Command::new(tool(lang)).arg("--version").output().is_err() {
            eprintln!("SKIPPED {lang:?}: {} not found (not a pass)", tool(lang));
            continue;
        }
        let code = emit(&m, lang).unwrap_or_else(|e| panic!("{tag} {lang:?}: not emitted: {e}"));
        let got = run(lang, tag, &code).unwrap_or_else(|e| panic!("{tag} {lang:?} did not run: {e}"));
        let got: Vec<&str> = got.lines().collect();
        if got != want {
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if g != w {
                    panic!("{tag} {lang:?}: line {i}: got {g:?}, want {w:?}");
                }
            }
            panic!("{tag} {lang:?}: {} lines, want {}", got.len(), want.len());
        }
    }
    want
}

const WORK: &str = "struct P
    x: i64
    f: f64

fn sq(n: i64) -> i64 = n * n

impl P
    fn bump(self, d: i64) -> P = P(x: self.x + d, f: self.f * 1.5)

fn work() -> i64
    var t: i64 = 0
    let p = P(x: 2, f: 0.5)
    let q = p.bump(3)
    t += q.x + sq(4)
    let xs = [1, 2, 3, 4, 5]
    t += xs[2] + xs.len() as i64
    let ys = [x * 2 for x in xs if x > 1]
    t += ys[0]
    let h = fn(a: i64) => a + t
    t = h(1)
    let zs = xs.map(fn(a: i64) => a - 1)
    t += zs[4]
    var f: f64 = 1.0
    for i in 0..4
        f = f * 1.25 + q.f / 2.0
    if not (t < 0) and (f > 1.0 or t > 1000)
        t += -(t % 7)
    let s = xs[1:3]
    t += s[0]
    let m = {\"a\": 1, \"b\": 2}
    t += m.len() as i64
    var acc: i64 = 0
    for i in 0..6 par
        acc += i * 3
    t += acc
    let ps = xs.pscan(fn(a: i64, b: i64) => a + b, 0)
    t + ps[4] + [0; 3].len() as i64
";

#[test]
fn budget_traps_on_the_same_operation_on_every_target() {
    // the total cost of work(), measured by the evaluator
    let probe = format!("{WORK}\nfn total() -> i64\n    var r: i64 = 0\n    with budget 1000000\n        r = work()\n    r\n");
    let lines = agree("budget-probe", &probe);
    assert!(lines.iter().any(|l| l.starts_with("fn total = ")), "{lines:?}");
    // sweep the limit across the whole run: every limit up to the cost and past it
    let mut fns = String::new();
    let limits: Vec<u32> = (0..=420).step_by(3).chain([421, 422, 423, 424, 425, 430, 500, 800]).collect();
    for n in &limits {
        fns += &format!("fn b{n}() -> i64\n    var r: i64 = 0\n    with budget {n}\n        r = work()\n    r\n\n");
    }
    // nested budgets: an inner budget also charges the outer one
    fns += "fn nested() -> i64\n    var r: i64 = 0\n    with budget 300\n        with budget 1000\n            r = work()\n    r\n\n";
    fns += "fn after_trap() -> i64\n    var r: i64 = 0\n    with budget 50\n        r = 1\n    r + work()\n";
    let src = format!("{WORK}\n{fns}");
    let lines = agree("budget-sweep", &src);
    let traps = lines.iter().filter(|l| l.ends_with("trap: budget")).count();
    let done = lines.iter().filter(|l| l.starts_with("fn b") && !l.contains("trap")).count();
    assert!(traps > 50 && done > 3, "traps {traps}, completed {done}: {lines:?}");
    eprintln!("budget sweep: {} limits, {traps} trap, {done} complete", limits.len());
}

#[test]
fn workgroups_agree() {
    let src = "fn kernel() -> i64
    let x = [5, -3, 8, 1, 9, 2, -7, 4]
    var total: i64 = 0
    var prod: i64 = 1
    with device d bound 4096
        with group g lanes 8
            var tile: [i64; 8] = [0; 8]
            tile[g.lane] = x[g.lane] * 3
            barrier
            total += tile[7 - g.lane] * (g.lane as i64 + 1)
            prod *= g.lane as i64 % 3 + 1
    total + prod

fn floats() -> f64
    var s: f64 = 0.1
    with device d bound 4096
        with group g lanes 7
            s += 1.0 / (g.lane as f64 + 0.7)
    s

fn no_lanes() -> i64
    var s: i64 = 9
    with device d bound 4096
        with group g lanes 0
            s += 1
    s

fn lanes_seen() -> i64
    var s: i64 = 0
    with device d bound 4096
        with group g lanes 5
            s += g.lanes as i64 * 10 + g.lane as i64
    s
";
    let l = agree("group", src);
    assert!(l.iter().any(|x| x.starts_with("fn no_lanes = 9")), "{l:?}");
}

#[test]
fn lvars_agree() {
    let src = "fn basic() -> i64
    let x = lvar_new(0)
    x.put(4)
    x.put(2)
    x.put(7)
    x.freeze()

fn sets() -> List[i64]
    let s = lvar_new([0])
    for i in 0..9 par
        s.put([i * 3 % 7])
    s.freeze()

fn conflict() -> i64
    let x = lvar_new(0, 10)
    x.put(3)
    x.put(10)
    x.freeze()

fn waits_ok() -> i64
    let x = lvar_new(0)
    x.put(5)
    x.wait([3])

fn waits_never() -> i64
    let x = lvar_new(0)
    x.put(1)
    x.wait([3])

fn waits_on_later_branch() -> i64
    let x = lvar_new(0)
    let y = lvar_new(0)
    var a: i64 = 0
    var b: i64 = 0
    par
        branch first
            a = x.wait([5]) + 1
            y.put(a)
        branch second
            x.put(7)
            b = y.wait([6]) * 2
    a * 100 + b + x.freeze() + y.freeze()

fn stuck() -> i64
    let x = lvar_new(0)
    let y = lvar_new(0)
    var a: i64 = 0
    par
        branch one
            a = x.wait([1])
        branch two
            let z = y.wait([1])
    a

fn par_conflict() -> i64
    let x = lvar_new(0, 10)
    par
        branch a
            x.put(3)
        branch b
            x.put(10)
    x.freeze()
";
    let l = agree("lvar", src);
    for want in [
        "fn basic = 7",
        "fn waits_ok = 3",
        "fn waits_on_later_branch = 625",
        "fn conflict trap: lvar conflict",
        "fn waits_never trap: lvar wait cannot be satisfied",
        "fn stuck trap: lvar wait cannot be satisfied",
    ] {
        assert!(l.iter().any(|x| x == want), "missing {want:?} in {l:?}");
    }
}

#[test]
fn deadline_nondet_and_atomic_agree() {
    let dl = "fn main()
    var n: i64 = 0
    with deadline 3
        for i in 0..4
            n += i
        emit n
    else
        emit 0 - 1
    emit n * 2
";
    let l = agree("deadline", dl);
    assert_eq!(l.iter().filter(|x| x.starts_with("emit")).count(), 2, "{l:?}");
    let at = "fn main()
    emit 1
    atomic
        emit 2
    emit 3
";
    let l = agree("atomic", at);
    assert!(l.iter().any(|x| x.contains("concurrency is not available at comptime")), "{l:?}");
    let nd = "fn main()
    emit 1
    let c = chan[i64](1)
    select nondet
        recv v from c -> emit v
        default -> emit 0
";
    agree("nondet", nd);
}

#[test]
fn budget_with_group_lvar_and_par() {
    let src = "fn mix() -> i64
    var s: i64 = 0
    var u: i64 = 0
    with budget 400
        with device d bound 4096
            with group g lanes 4
                s += g.lane as i64 * 2
        let x = lvar_new(0)
        par
            branch a
                x.put(s)
            branch b
                u = 1
        s += x.freeze() + u
    s

fn mix_short() -> i64
    var s: i64 = 0
    with budget 40
        with device d bound 4096
            with group g lanes 4
                s += g.lane as i64 * 2
        par
            branch a
                s += 5
    s
";
    agree("mix", src);
}

/// T29 under the restart scheduler: random chains of lvar waits between
/// branches (a branch may wait for any later or earlier one) give the
/// evaluator's result on every target, including the stuck case.
#[test]
fn lvar_wait_chains_agree() {
    let mut x: u64 = 0x2929;
    let mut next = |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % m
    };
    let mut fns = String::new();
    for f in 0..8 {
        let k = 2 + next(4) as usize;
        let order: Vec<usize> = {
            let mut o: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                o.swap(i, next(i as u64 + 1) as usize);
            }
            o
        };
        // order[j] is the j-th branch to be able to run: it waits on the
        // lvar put by order[j - 1] (or nothing for j = 0), then puts its own
        let stuck = f % 4 == 3;
        let mut decl = String::new();
        let mut branches = String::new();
        for i in 0..k {
            decl += &format!("    let v{i} = lvar_new(0)\n    var r{i}: i64 = 0\n");
        }
        for (i, _) in (0..k).enumerate() {
            let pos = order.iter().position(|&o| o == i).unwrap();
            let mut body = String::new();
            if pos > 0 {
                let prev = order[pos - 1];
                let need = if stuck && pos == k - 1 { 1000 } else { pos as i64 * 10 };
                body += &format!("            r{i} = v{prev}.wait([{need}]) + {i}\n");
            }
            body += &format!("            v{i}.put({})\n", (pos as i64 + 1) * 10);
            branches += &format!("        branch b{i}\n{body}");
        }
        let sum: Vec<String> = (0..k).map(|i| format!("r{i} * {} + v{i}.freeze()", i + 1)).collect();
        fns += &format!("fn chain{f}() -> i64\n{decl}    par\n{branches}    {}\n\n", sum.join(" + "));
    }
    let l = agree("chains", &fns);
    let stuck = l.iter().filter(|x| x.ends_with("lvar wait cannot be satisfied")).count();
    assert!(stuck >= 1 && stuck < l.len(), "{l:?}");
}
