//! HC-12/HC-13: the placement executor for lane-wise array operations,
//! `for … par` bodies and reductions, on CPU threads and on a registered
//! platform fabric, with the same bits on every fabric.
use rce_spec_check::host::{FabricExec, Host, LaneOp, MemStore, fixed_tree, fixed_tree_sum, fixed_tree_threads, lanes_threads, replay};
use rce_spec_check::parse::parse;

const M: &str = "# rce-version: 0.6\nmodule p\nstate S\n    n: i64 = 0\n";
const OPS: [LaneOp; 6] = [LaneOp::Add, LaneOp::Sub, LaneOp::Mul, LaneOp::Div, LaneOp::Neg, LaneOp::Sqrt];

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a = (0..n).map(|i| (i as f64 * 0.37).sin() * if i % 5 == 0 { 1e9 } else { 1.0 }).collect();
    let b = (0..n).map(|i| (i as f64 * 1.3).cos() + 2.0).collect();
    (a, b)
}

/// A fabric that follows the contract (stands in for a wgpu device).
struct Good;
impl FabricExec for Good {
    fn id(&self) -> &str {
        "gpu.good"
    }
    fn lanes(&self, op: LaneOp, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        Ok(a.iter().zip(b).map(|(x, y)| op.eval(*x, *y)).collect())
    }
    fn sum(&self, xs: &[f64]) -> Result<f64, String> {
        Ok(fixed_tree_sum(xs))
    }
}

/// A fabric that reassociates the sum and flushes subnormals to zero.
struct Loose;
impl FabricExec for Loose {
    fn id(&self) -> &str {
        "gpu.loose"
    }
    fn lanes(&self, op: LaneOp, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
        Ok(a.iter().zip(b).map(|(x, y)| {
            let r = op.eval(*x, *y);
            if r.is_subnormal() { 0.0 } else { r }
        }).collect())
    }
    fn sum(&self, xs: &[f64]) -> Result<f64, String> {
        Ok(xs.iter().sum())
    }
}

#[test]
fn lanes_and_trees_have_the_same_bits_for_every_thread_count() {
    let (a, b) = data(10_007);
    for op in OPS {
        let r: Vec<u64> = (0..a.len()).map(|i| op.eval(a[i], b[i]).to_bits()).collect();
        for t in [1, 2, 3, 7, 8, 64] {
            let g: Vec<u64> = lanes_threads(a.len(), t, &|i| op.eval(a[i], b[i])).iter().map(|x| x.to_bits()).collect();
            assert_eq!(g, r, "{op:?} threads {t}");
        }
    }
    // A combine that is neither associative nor commutative.
    let f = |x: &f64, y: &f64| (x + 2.0 * y) * 0.999_999_9;
    let base = fixed_tree(&a, &0.0, &f);
    for d in 0..6 {
        assert_eq!(fixed_tree_threads(&a, &0.0, &f, d).to_bits(), base.to_bits());
    }
}

#[test]
fn lane_and_par_placement_follow_d_and_replay() {
    let m = parse(M).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let (a, b) = data(50_000);
    let (f0, r0) = h.place_lanes("l0", LaneOp::Mul, &a, &b).unwrap();
    assert_eq!(f0, "cpu");
    h.calibrate("cpu.threads", "lanes.f64.mul", 300.0, 6.0).unwrap();
    assert_eq!(h.place_lanes("l1", LaneOp::Mul, &a, &b).unwrap().0, "cpu", "not probed yet");
    assert!(h.probe_fabric("cpu.threads", "lanes.f64.mul").unwrap());
    let (f2, r2) = h.place_lanes("l2", LaneOp::Mul, &a, &b).unwrap();
    assert_eq!(f2, "cpu.threads");
    assert_eq!(r0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), r2.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    // Small work stays on the CPU (overhead dominates).
    assert_eq!(h.place_lanes("l3", LaneOp::Mul, &a[..10], &b[..10]).unwrap().0, "cpu");
    // for … par: body i -> a[i] * b[i], combined by a non-associative op.
    let body = |i: usize| a[i] * b[i];
    let f = |x: &f64, y: &f64| (x + y) * 1.000_000_1;
    let (p0, v0) = h.place_par("p0", "dot", a.len(), &body, &0.0, &f).unwrap();
    assert_eq!(p0, "cpu");
    h.calibrate("cpu.threads", "par.dot", 300.0, 6.0).unwrap();
    assert!(h.probe("cpu.threads", "par.dot", "tree", "tree").unwrap());
    let (p1, v1) = h.place_par("p1", "dot", a.len(), &body, &0.0, &f).unwrap();
    assert_eq!(p1, "cpu.threads");
    assert_eq!(v0.to_bits(), v1.to_bits());
    drop(h);
    let text = st.lines.join("\n") + "\n";
    replay(&m, &text).expect("placements replay");
    let placed = st.lines.iter().filter(|l| l.contains("host.Placed")).count();
    assert_eq!(placed, 6);
}

#[test]
fn platform_fabric_is_used_only_after_an_exact_probe() {
    let m = parse(M).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.register_fabric(Box::new(Good));
    h.register_fabric(Box::new(Loose));
    let (a, b) = data(200_000);
    for f in ["gpu.good", "gpu.loose"] {
        h.calibrate(f, "sum.f64", 50.0, 100.0).unwrap();
        h.calibrate(f, "lanes.f64.div", 50.0, 100.0).unwrap();
    }
    // Before any probe both are ignored.
    assert_eq!(h.place_sum("s0", &a).unwrap().0, "cpu");
    // The loose fabric fails its probes (reassociation, subnormal flush).
    assert!(!h.probe_fabric("gpu.loose", "sum.f64").unwrap());
    assert!(!h.probe_fabric("gpu.loose", "lanes.f64.mul").unwrap());
    assert!(h.probe_fabric("gpu.good", "sum.f64").unwrap());
    assert!(h.probe_fabric("gpu.good", "lanes.f64.div").unwrap());
    let (f1, s1) = h.place_sum("s1", &a).unwrap();
    assert_eq!(f1, "gpu.good");
    assert_eq!(s1.to_bits(), fixed_tree_sum(&a).to_bits());
    let (f2, r2) = h.place_lanes("d", LaneOp::Div, &a, &b).unwrap();
    assert_eq!(f2, "gpu.good");
    assert!(r2.iter().zip(a.iter().zip(&b)).all(|(r, (x, y))| r.to_bits() == (x / y).to_bits()));
    // `for … par` bodies never go to a platform fabric.
    h.calibrate("gpu.good", "par.dot", 1.0, 1e9).unwrap();
    assert!(h.probe("gpu.good", "par.dot", "x", "x").unwrap());
    let n = h.len();
    assert!(h.place_par("p", "dot", 10, &|i| i as f64, &0.0, &|x, y| x + y).is_err());
    assert_eq!(h.len(), n, "a placement that is not performed logs nothing");
    drop(h);
    let text = st.lines.join("\n") + "\n";
    // Replay re-derives placements from D alone; it needs no executor.
    replay(&m, &text).expect("replay without the fabric");
    // A host that lacks the executor does not perform the placement, and logs nothing.
    let mut st2 = MemStore::default();
    let mut h2 = Host::resume(&m, &text, Box::new(&mut st2)).unwrap();
    h2.take_role(1, "h2").unwrap();
    let n2 = h2.len();
    assert!(h2.place_sum("s2", &a).is_err());
    assert_eq!(h2.len(), n2);
}

/// C.8 / HC-12: the reference evaluator's `for ... par` reduction and the
/// placement executor evaluate one fixed tree (`host::tree_split`): a float
/// sum, which is not associative, gives the same bits from `run_main` and
/// from `place_par` / `fixed_tree` for every length.
#[test]
fn forpar_reference_and_placement_share_the_tree() {
    use rce_spec_check::conc::{RunCfg, run_main};
    let val = |i: usize| (i as f64 + 1.0).recip() + if i % 3 == 0 { 1e16 } else { -1e16 + 3.0 };
    for n in 1..=40usize {
        let src = format!(
            "# rce-version: 0.6\nmodule t\n\nfn main()\n    var s: f64 = 0.0\n    for i in 0..{n} par\n        s += 1.0 / ((i as f64) + 1.0) + (if i % 3 == 0 then 1e16 else -1e16 + 3.0)\n    emit s\n"
        );
        let m = parse(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let r = run_main(&m, &RunCfg::default()).unwrap();
        let xs: Vec<f64> = (0..n).map(val).collect();
        let t = fixed_tree(&xs, &0.0, &|a: &f64, b: &f64| a + b);
        assert_eq!(t.to_bits(), fixed_tree_sum(&xs).to_bits());
        let want = rce_spec_check::interp::V::F(t).show();
        assert_eq!(r.outputs.last().map(|v| v.show()), Some(want), "n = {n}");
    }
}
