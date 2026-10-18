//! Test vectors linking the abstract models of `Models06b.lean` to the
//! reference evaluator. Random instances are run by `conc::run_main`, and
//! each observed result becomes a Lean theorem stating that the model
//! computes the same thing (spec/lean/RCE/ModelVectors.lean; RCE_BLESS=1
//! rewrites it, and the test fails if the file is stale):
//!
//! * T27: an actor's answers to a message sequence equal `Models.actor`
//!   applied to the sequential specification (state += k, answer = state);
//! * T29: the frozen value of an lvar after puts from par branches, in
//!   several branch orders, is the `foldl max` of the puts, and a
//!   threshold read returns the threshold (`thresholdRead`);
//! * T30: atomic transactions across two actors, some refused by a
//!   handler failure, leave the state `Models.atomic` computes.
//!
//! These are correspondences on vectors, not proofs about the evaluator.
use rce_spec_check::conc::{Outcome, RunCfg, run_main};
use rce_spec_check::parse::parse;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn run(src: &str) -> Vec<String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "not admitted: {:?}\n{src}", d[0].1);
    let m = parse(src).unwrap();
    let r = run_main(&m, &RunCfg::default()).unwrap();
    assert!(matches!(r.outcome, Outcome::Done), "{:?}\n{src}", r.outcome);
    r.outputs.iter().map(|x| x.show()).collect()
}

fn lean_int(v: &str) -> String {
    if v.starts_with('-') { format!("({v})") } else { v.to_string() }
}

fn lean_list(xs: &[i64]) -> String {
    format!("[{}]", xs.iter().map(|x| lean_int(&x.to_string())).collect::<Vec<_>>().join(", "))
}

fn t27(r: &mut Rng, k: usize, out: &mut String) {
    let n = 2 + r.below(6) as usize;
    let ms: Vec<i64> = (0..n).map(|_| r.below(19) as i64 - 9).collect();
    let mut src = String::from("event Add(k: i64)\n\nactor Acc\n    n: i64 = 0\n    on Add(k: k)\n        n += k\n        emit n\n\nfn main()\n    let a = Acc()\n");
    for m in &ms {
        src += &format!("    a <- Add(k: {m})\n");
    }
    let got = run(&src);
    let outs: Vec<i64> = got.iter().map(|x| x.parse().unwrap()).collect();
    *out += &format!(
        "\n/-\n{src}-/\ntheorem t27_v{k} : Models.actor (fun (s m : Int) => s + m) (fun s m => s + m) 0 {} = {} := by decide\n",
        lean_list(&ms),
        lean_list(&outs)
    );
}

fn t29(r: &mut Rng, k: usize, out: &mut String) {
    let nb = 2 + r.below(3) as usize;
    let puts: Vec<Vec<i64>> = (0..nb).map(|_| (0..1 + r.below(3)).map(|_| r.below(50) as i64).collect()).collect();
    let all: Vec<i64> = puts.iter().flatten().copied().collect();
    let mx = *all.iter().max().unwrap();
    let t = r.below(mx as u64 + 1) as i64;
    // the same puts in two branch orders
    for (o, order) in [(0usize, (0..nb).collect::<Vec<_>>()), (1, (0..nb).rev().collect())] {
        let mut src = String::from("fn main()\n    let c = lvar_new(0)\n    par\n");
        let mut flat = vec![];
        for (j, &b) in order.iter().enumerate() {
            src += &format!("        branch b{j}\n");
            for p in &puts[b] {
                src += &format!("            c.put({p})\n");
                flat.push(*p);
            }
        }
        src += &format!("    emit c.freeze()\n    emit c.wait([{t}])\n");
        let got = run(&src);
        *out += &format!(
            "\n/-\n{src}-/\ntheorem t29_v{k}_{o} : {}.foldl max 0 = {} ∧ Models.thresholdRead {t} ({}.foldl max 0) = some {} := by decide\n",
            lean_list_nat(&flat),
            got[0],
            lean_list_nat(&flat),
            got[1]
        );
    }
}

fn lean_list_nat(xs: &[i64]) -> String {
    format!("([{}] : List Nat)", xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn t30(r: &mut Rng, k: usize, out: &mut String) {
    let nt = 1 + r.below(4) as usize;
    let txns: Vec<(i64, i64)> = (0..nt).map(|_| (r.below(13) as i64 - 3, r.below(13) as i64 - 3)).collect();
    let mut src = String::from(
        "event Add(k: i64)\nevent Show\nerror Neg\n\nactor Acc\n    n: i64 = 0\n    on Add(k: k)\n        if k < 0\n            fail Neg\n        n += k\n    on Show\n        emit n\n\nfn main()\n    let a = Acc()\n    let b = Acc()\n",
    );
    for (x, y) in &txns {
        src += &format!("    atomic\n        a <- Add(k: {x})\n        b <- Add(k: {y})\n");
    }
    src += "    a <- Show\n    b <- Show\n";
    let got = run(&src);
    // the model: one state for the pair, a transaction is a batch of two ops
    let mut lean_state = "((0 : Int), (0 : Int))".to_string();
    for (x, y) in &txns {
        lean_state = format!(
            "(Models.atomic (fun _ (o : Bool × Int) => decide (0 ≤ o.2)) (fun (s : Int × Int) (o : Bool × Int) => if o.1 then (s.1 + o.2, s.2) else (s.1, s.2 + o.2)) {lean_state} [(true, {}), (false, {})])",
            lean_int(&x.to_string()),
            lean_int(&y.to_string())
        );
    }
    *out += &format!(
        "\n/-\n{src}-/\ntheorem t30_v{k} : {lean_state} = ({}, {}) := by decide\n",
        lean_int(&got[0]),
        lean_int(&got[1])
    );
}

fn generate() -> String {
    let mut r = Rng(0x7_06b_0ce11);
    let mut out = String::from(
        "import RCE.Models06b\n/-! Generated by spec/checker/tests/lean_models.rs (RCE_BLESS=1): the\n    models of T27, T29 and T30 instantiated on random instances, each\n    equal to what the reference evaluator computed for the same program.\n    Do not edit by hand. -/\nnamespace RCE.ModelVectors\nopen RCE\n",
    );
    for k in 0..12 {
        t27(&mut r, k, &mut out);
    }
    for k in 0..8 {
        t29(&mut r, k, &mut out);
    }
    for k in 0..12 {
        t30(&mut r, k, &mut out);
    }
    out += "\nend RCE.ModelVectors\n";
    out
}

#[test]
fn lean_model_vectors_match_evaluator() {
    let text = generate();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lean/RCE/ModelVectors.lean");
    if std::env::var("RCE_BLESS").is_ok_and(|v| v == "1") {
        std::fs::write(&path, &text).unwrap();
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(have == text, "ModelVectors.lean is stale; rerun with RCE_BLESS=1 and lake build");
}
