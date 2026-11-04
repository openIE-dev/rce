//! The graph term (spec 6.13, GT-1 to GT-8; W-GRAPH-1 to W-GRAPH-3; W7 of
//! the core): a step whose numeric body is a FunctionGraph, from `rce check`
//! through the reference, the core and the host to replay.
use rce_spec_check::host::{Host, MemStore, replay_report};
use rce_spec_check::{cli, graph, one, parse::parse};
use rce_graph::term::Level;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dense() -> String {
    std::fs::read_to_string(root().join("compute/examples/dense.rce")).unwrap()
}

/// `rce ARGS…` in this process: status, standard output, standard error.
fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let st = cli::main(&a, &mut std::io::empty(), &mut out, &mut err);
    (st, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-graph-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const EVENTS: &str = "Input(x: [1.0, 2.0, 3.0]), Input(x: [-1.0, 0.0, 1.0]), Input(x: [1.0e308, -1.0e308, 1.0e308]), Input(x: [2.0, 0.5, 0.25])";

#[test]
fn a_tensor_step_is_admitted_and_its_examples_hold() {
    let p = root().join("compute/examples/dense.rce");
    let p = p.to_str().unwrap();
    let (st, out, _) = rce(&["check", p]);
    assert_eq!(st, 0, "{out}");
    assert!(out.starts_with("ok "), "{out}");
    let (st, out, _) = rce(&["test", p]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("(4 examples, 0 checks)") && out.contains("1 passed, 0 failed"), "{out}");
}

#[test]
fn the_reference_is_byte_identical_on_every_run() {
    let p = root().join("compute/examples/dense.rce");
    let p = p.to_str().unwrap();
    let runs: Vec<(i32, String, String)> = (0..3).map(|_| rce(&["trace", p, EVENTS])).collect();
    assert_eq!(runs[0].0, 0);
    assert!(runs.iter().all(|r| r == &runs[0]), "rce trace differs between runs");
    let runs: Vec<(i32, String, String)> = (0..3).map(|_| rce(&["run", p, EVENTS])).collect();
    assert!(runs.iter().all(|r| r == &runs[0]), "rce run differs between runs");
    // the third input leaves the finite floats: kept, with the trap of 6.3 N3
    assert!(runs[0].1.contains("kept: trap(non-finite float)"), "{}", runs[0].1);
    assert!(runs[0].1.contains("y = [1.375, 0]"), "{}", runs[0].1);
}

/// The trace's state hashes, in order.
fn trace_hashes(t: &str) -> Vec<String> {
    let j: serde_json::Value = serde_json::from_str(t).unwrap();
    j["traces"][0]["steps"].as_array().unwrap().iter().map(|s| s["state_hash"].as_str().unwrap().to_string()).collect()
}

#[test]
fn the_host_records_d_and_replay_reproduces_every_state_hash() {
    let s = dense();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut live = vec![];
    for ev in EVENTS.split("), ") {
        let ev = if ev.ends_with(')') { ev.to_string() } else { format!("{ev})") };
        let a = h.admit(&s, &ev).unwrap();
        live.push((a[0].seq, a[0].cause));
    }
    let final_hash = h.state_hash().to_string();
    drop(h);
    let d: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    // D's entries for the four inputs carry the reference trace's state hashes
    let p = root().join("compute/examples/dense.rce");
    let (_, t, _) = rce(&["trace", p.to_str().unwrap(), EVENTS]);
    let want = trace_hashes(&t);
    let entries: Vec<serde_json::Value> = d.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let got: Vec<String> = live.iter().map(|(seq, _)| entries[*seq as usize]["state_hash"].as_str().unwrap().to_string()).collect();
    assert_eq!(got, want, "the host's state hashes are the reference's");
    assert_eq!(live[2].1, Some("trap"), "the non-finite input is kept with cause trap");
    // replay with every adapter disconnected re-derives every entry
    let r = replay_report(&m, &d).unwrap();
    assert_eq!(r.state_hash, final_hash);
    assert_eq!(r.kept_outcomes.len(), 1);
    let r2 = replay_report(&m, &d).unwrap();
    assert_eq!(r2.state_hash, r.state_hash, "replay is byte-identical on every run");
}

#[test]
fn the_core_holds_the_term_and_its_evaluator_gives_the_reference_trace() {
    let s = dense();
    let m = rce_spec_check::program(&s).unwrap();
    let c = one::core_of(&m).unwrap();
    assert_eq!(c.graphs.len(), 1);
    let key = c.graphs.keys().next().unwrap().clone();
    assert_eq!(graph::keys_of(&parse(&s).unwrap()), vec![key.clone()], "the checker and the lowering build one term");
    let text = one::json::module_text(&c);
    assert!(text.contains(&format!("[\"graph\",\"{key}\"")), "a group applies the term by its key, so its key covers it");
    // the core read back is well formed (W1 to W7), and the core evaluator
    // reproduces the reference trace, the kept step included
    let back = one::json::module_from_text(&text).unwrap();
    one::wf::check(&back).unwrap();
    assert_eq!(one::run_core(&back).unwrap(), one::trace::reference(&m));
    // a module without terms has no `graphs` field: its text is unchanged
    let plain = rce_spec_check::program("state S\n    n: u64 = 0\nevent Inc\nstep S on Inc -> n + 1\n").unwrap();
    assert!(!one::json::module_text(&one::core_of(&plain).unwrap()).contains("graphs"));
}

#[test]
fn wrong_shape_operand_and_undeclared_level_require_change() {
    let dir = root().join("spec/examples/change/types");
    for (file, rule, words) in [
        ("graph_shape.rce", "W-GRAPH-1", "has 3 columns and the vector [f64; 2] has 2 elements"),
        ("graph_operand.rce", "W-GRAPH-2", "has type List[f64]"),
        ("graph_level.rce", "W-GRAPH-3", "has level lossy, weaker than the level exact in force here"),
        ("graph_attention.rce", "W-GRAPH-1", "k has q's last length"),
        ("graph_integer.rce", "W-GRAPH-2", "`tensor.exp` takes tensors of f32 or f64, found [i32; 3]"),
        ("graph_quantize.rce", "W-GRAPH-3", "`tensor.round_q8_0` has level lossy"),
    ] {
        let (st, out, _) = rce(&["check", dir.join(file).to_str().unwrap()]);
        assert_eq!(st, 1, "{file}: {out}");
        assert!(out.contains("requires change") && out.contains(rule) && out.contains(words), "{file}: {out}");
    }
    // the matrix product's inner dimensions, and two element types
    let e = rce_spec_check::check_one(
        "use std.tensor\nstate S\n    a: [[f64; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n    c: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 0.0]]\nevent Go\nstep S on Go\n    c = tensor.matmul(a, a)\n",
    )
    .unwrap_err();
    assert!(e.contains("W-GRAPH-1") && e.contains("has 3 columns and the second [[f64; 3]; 2] has 2 rows"), "{e}");
    let e = rce_spec_check::check_one(
        "use std.tensor\nstate S\n    a: [f64; 2] = [1.0, 2.0]\n    b: [f32; 2] = [1.0 as f32, 2.0 as f32]\nevent Go\nstep S on Go\n    a = tensor.relu(a + b)\n",
    )
    .unwrap_err();
    assert!(e.contains("W-GRAPH-2") && e.contains("two element types"), "{e}");
}

/// A program of f32 tensors with every operation and both declarations.
const LEVELS: &str = "use std.tensor

intent \"every operation, at three levels\"
    example Go(x: [1.0, 3.0]) -> q == [1.0, 3.0] and t == 1.0 as f32 and p == [[5.0 as f32, 11.0 as f32], [11.0 as f32, 25.0 as f32]]
    example Go(x: [1.00390625, 0.5]) -> q == [1.0, 0.5]

state S
    a: [[f32; 2]; 2] = [[1.0 as f32, 2.0 as f32], [3.0 as f32, 4.0 as f32]]
    q: [f64; 2] = [0.0, 0.0]
    t: f32 = 0.0 as f32
    p: [[f32; 2]; 2] = [[0.0 as f32, 0.0 as f32], [0.0 as f32, 0.0 as f32]]

event Go(x: [f64; 2])

step S on Go(x)
    q = tensor.lossy(tensor.round_bf16(x))
    t = tensor.sum(tensor.relu(tensor.transpose(a) - a))
    p = tensor.within(4, tensor.matmul(a, tensor.transpose(a)))
";

#[test]
fn levels_are_carried_by_every_operation() {
    rce_spec_check::check_one(LEVELS).unwrap();
    let m = rce_spec_check::program(LEVELS).unwrap();
    let c = one::core_of(&m).unwrap();
    assert_eq!(c.graphs.len(), 3);
    let mut seen = vec![];
    for (k, t) in &c.graphs {
        let (_, ty) = one::wf::graph_term(k, t).unwrap();
        let ls: Vec<Level> = ty.levels.iter().map(|l| l.level).collect();
        seen.push(ls);
    }
    seen.sort_by_key(|v| format!("{v:?}"));
    assert!(seen.contains(&vec![Level::Lossy]), "{seen:?}");
    assert!(seen.contains(&vec![Level::Within { ulps: 4 }, Level::Within { ulps: 4 }]), "{seen:?}");
    assert!(seen.contains(&vec![Level::Exact, Level::Exact, Level::Exact, Level::Exact]), "{seen:?}");
    // 1 + 2^-8 + 2^-40 lies just above a tie: it rounds up to 1 + 2^-7;
    // 1 + 2^-8 is a tie and rounds to even, 1
    let x0 = 1.0 + 2f64.powi(-8) + 2f64.powi(-40);
    let ev = format!("Go(x: [{x0:e}, {:e}])", 1.0 + 2f64.powi(-8));
    let (st, out, err) = rce(&["run", tmp_file("levels.rce", LEVELS).to_str().unwrap(), &ev]);
    assert_eq!(st, 0, "{out}{err}");
    assert!(out.contains(&format!("q = [{}, 1]", 1.0 + 2f64.powi(-7))), "{out}");
    // transpose(a) - a = [[0, -1], [1, 0]]; relu, then the sum: 1
    assert!(out.contains("t = 1"), "{out}");
    // a aᵀ = [[5, 11], [11, 25]]
    assert!(out.contains("p = [[5, 11], [11, 25]]"), "{out}");
    assert_eq!(one::run_core(&c).unwrap(), one::trace::reference(&m));
}

fn tmp_file(name: &str, text: &str) -> PathBuf {
    let p = tmp(name).join(name);
    std::fs::write(&p, text).unwrap();
    p
}

#[test]
fn a_term_whose_key_is_not_its_hash_requires_change() {
    let s = dense();
    let c = one::core_of(&rce_spec_check::program(&s).unwrap()).unwrap();
    let mut j: serde_json::Value = serde_json::from_str(&one::json::module_text(&c)).unwrap();
    let k = j["graphs"].as_object().unwrap().keys().next().unwrap().clone();
    for n in j["graphs"][&k]["graph"]["nodes"].as_array_mut().unwrap() {
        if n["Apply"]["op"]["Custom"] == "relu" {
            n["Apply"]["op"]["Custom"] = "round_bf16".into();
        }
    }
    let p = tmp_file("tampered.json", &j.to_string());
    let (st, _, err) = rce(&["run-core", p.to_str().unwrap()]);
    assert_eq!(st, 1, "{err}");
    assert!(err.contains(&format!("W7 graph term {k} requires change: the key is not the content hash")), "{err}");
}

#[test]
fn w7_refuses_a_level_stronger_than_the_operation_admits() {
    let s = dense();
    let c = one::core_of(&rce_spec_check::program(&s).unwrap()).unwrap();
    let (k, t) = c.graphs.iter().next().unwrap();
    let (mut g, ty) = one::wf::graph_term(k, t).unwrap();
    for n in g.nodes.iter_mut() {
        if let rce_graph::graph::GraphNode::Apply { op: rce_graph::graph::OpKind::Custom(c), .. } = n
            && c == "relu"
        {
            *c = "round_bf16".into();
        }
    }
    // the key of the changed body, so W7 reaches the level
    let k2 = graph::key_of(&g, &ty).unwrap();
    let e = graph::check_term(&k2, &g, &ty).unwrap_err();
    assert!(e.contains("requires change") && e.contains("is declared exact and its operation admits at most lossy"), "{e}");
    // a level left out of the typing
    let mut ty2 = ty.clone();
    ty2.levels.pop();
    let k3 = graph::key_of(&g, &ty2).unwrap();
    let e = graph::check_term(&k3, &g, &ty2).unwrap_err();
    assert!(e.contains("every operation node carries one declared level"), "{e}");
}

#[test]
fn the_program_hash_binds_the_term_and_is_name_free() {
    let s = dense();
    let renamed = s
        .replace("Layer", "Dense")
        .replace("    w: ", "    weights: ")
        .replace("    b: ", "    bias: ")
        .replace("matvec(w, x) + b)", "matvec(weights, inp) + bias)")
        .replace("Input(x: [f64; 3])", "Input(inp: [f64; 3])")
        .replace("Input(x)", "Input(inp)")
        .replace("Input(x: [", "Input(inp: [");
    assert_ne!(renamed, s);
    rce_spec_check::check_one(&renamed).unwrap();
    let h = |t: &str| rce_spec_check::apidiff::program_hash(&[parse(t).unwrap()]);
    assert_eq!(h(&s), h(&renamed), "a renaming keeps the program hash");
    assert_eq!(graph::keys_of(&parse(&s).unwrap()), graph::keys_of(&parse(&renamed).unwrap()), "the term is name-free");
    // the hash covers the term: the same text without the term's key differs
    let without: String = {
        let m = parse(&s).unwrap();
        let mut m2 = m.clone();
        m2.items.retain(|it| !matches!(it, rce_spec_check::ast::Item::Use(..)));
        rce_spec_check::apidiff::program_hash(&[m2])
    };
    assert_ne!(h(&s), without);
}

#[test]
fn rce_graph_writes_inspects_and_runs_fg_files() {
    let p = root().join("compute/examples/dense.rce");
    let dir = tmp("fg");
    let (st, out, err) = rce(&["graph", "terms", p.to_str().unwrap(), dir.to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    let key = out.split_whitespace().next().unwrap().to_string();
    assert!(out.contains("tensor.relu  (f64[2,3], f64[3], f64[2]) -> f64[2]  exact"), "{out}");
    let f = dir.join(format!("{key}.fg"));
    let (st, out, _) = rce(&["graph", "inspect", f.to_str().unwrap()]);
    assert_eq!(st, 0);
    assert!(out.contains(&format!("content hash   b3:{key}")) && out.contains("matmul(0, 3) -> f64[2,1]  exact"), "{out}");
    let (st, out, _) = rce(&["graph", "run", f.to_str().unwrap(), "[[[0.5,-0.5,0.5],[-1.0,1.0,-1.0]],[1.0,2.0,3.0],[0.5,-0.25]]"]);
    assert_eq!((st, out.as_str()), (0, "[1.5,0.0]\n"));
    let (_, out, _) = rce(&["graph", "run", f.to_str().unwrap(), "[[[1.0,0.0,0.0],[0.0,1.0,0.0]],[1e308,1e308,0.0],[1e308,0.0]]"]);
    assert_eq!(out, "trap: non-finite float\n");
    // one byte changed in the file: its content hash no longer holds
    let mut b = std::fs::read(&f).unwrap();
    let n = b.len();
    b[n / 2] ^= 1;
    std::fs::write(&f, b).unwrap();
    let (st, _, err) = rce(&["graph", "inspect", f.to_str().unwrap()]);
    assert_eq!(st, 1, "{err}");
    assert!(err.starts_with("requires change"), "{err}");
    let (st, out, _) = rce(&["graph", "ops"]);
    assert_eq!(st, 0);
    for row in ["round_bf16       lossy", "round_q8_0       lossy", "attention_causal exact", "gelu_tanh        exact", "matmul_q4_0      exact"] {
        assert!(out.contains(row), "{row}: {out}");
    }
}

// ---- the model operations (S4): definitions, a block, integers, findings

fn block_path() -> String {
    root().join("compute/examples/block.rce").to_str().unwrap().to_string()
}

#[test]
fn the_model_operations_equal_their_definitions_in_rce_source() {
    // each elementwise operation, f64 and f32, against its definition
    // written in RCE over math.rce, run by the reference interpreter: the
    // CPU backend's exp, tanh and erf are math.rce's, bit for bit
    let p = root().join("spec/checker/tests/graph/defs.rce");
    let (st, out, _) = rce(&["test", p.to_str().unwrap()]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("(0 examples, 8 checks)") && out.contains("1 passed"), "{out}");
}

#[test]
fn a_causal_attention_block_is_admitted_traced_and_replayed() {
    let p = block_path();
    let (st, out, _) = rce(&["test", &p]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("(3 examples, 0 checks)"), "{out}");
    let evs = "Tokens(ids: [0, 1, 2]), Tokens(ids: [4, 3, 2]), Tokens(ids: [5, 0, 1]), Tokens(ids: [1, 1, 0])";
    let runs: Vec<(i32, String, String)> = (0..3).map(|_| rce(&["trace", &p, evs])).collect();
    assert_eq!(runs[0].0, 0, "{}", runs[0].2);
    assert!(runs.iter().all(|r| r == &runs[0]), "rce trace differs between runs");
    let (_, out, _) = rce(&["run", &p, evs]);
    assert!(out.contains("kept: trap(index 5 out of range)"), "{out}");
    // the core holds the block's terms; its evaluator gives the reference trace
    let s = std::fs::read_to_string(&p).unwrap();
    let m = rce_spec_check::program(&s).unwrap();
    let c = one::core_of(&m).unwrap();
    // eight tensor expressions; the three projections q, k and v are one
    // term (the same graph and typing, so the same key)
    assert_eq!(c.graphs.len(), 6);
    let back = one::json::module_from_text(&one::json::module_text(&c)).unwrap();
    one::wf::check(&back).unwrap();
    assert_eq!(one::run_core(&back).unwrap(), one::trace::reference(&m));
    // the host writes D; replay with every adapter disconnected re-derives it
    let pm = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&pm, Box::new(&mut st)).unwrap();
    for ev in ["Tokens(ids: [0, 1, 2])", "Tokens(ids: [5, 0, 1])", "Tokens(ids: [1, 1, 0])"] {
        h.admit(&s, ev).unwrap();
    }
    let fin = h.state_hash().to_string();
    drop(h);
    let d: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    let r = replay_report(&pm, &d).unwrap();
    assert_eq!(r.state_hash, fin);
    assert_eq!(r.kept_outcomes.len(), 1);
}

/// Integer tensors: lane-wise arithmetic, sums and data movement, each
/// result checked against its type as RCE's integers are (6.3 N2).
const INTS: &str = "use std.tensor

intent \"integer tensors are checked as RCE integers\"
    example Add(d: [27, -28]) -> a == [127, -128] and s == -1
    example Add(d: [28, 0]) -> kept: trap(\"overflow: 128 is outside i8\")
    example Div(d: [0, 1]) -> kept: trap(\"division by zero\")
    example Join -> j == [[1, 2], [3, 4], [100, -100]] and s == 0

state S
    a: [i8; 2] = [100, -100]
    s: i32 = 0
    j: [[i32; 2]; 3] = [[0, 0], [0, 0], [0, 0]]

event Add(d: [i8; 2])
event Div(d: [i8; 2])
event Join

step S on Add(d)
    a = tensor.add(a, d)
    s = tensor.sum(d) as i32

step S on Div(d)
    a = tensor.reshape(tensor.reshape(a, [1, 2]) / tensor.reshape(d, [1, 2]), [2])

step S on Join
    let x: [[i32; 2]; 2] = [[1, 2], [3, 4]]
    let y: [[i32; 2]; 1] = [[100, -100]]
    j = tensor.concat(x, y, 0)
";

#[test]
fn integer_tensors_are_checked_as_rce_integers() {
    let p = tmp_file("ints.rce", INTS);
    let (st, out, _) = rce(&["test", p.to_str().unwrap()]);
    assert_eq!(st, 0, "{out}");
    let m = rce_spec_check::program(INTS).unwrap();
    assert_eq!(one::run_core(&one::core_of(&m).unwrap()).unwrap(), one::trace::reference(&m));
}

#[test]
fn shapes_operands_and_literals_of_the_model_operations_require_change() {
    let pre = "use std.tensor\nstate S\n    x: [[f64; 4]; 2] = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]\n    w: [f64; 3] = [0.0, 0.0, 0.0]\n    q: [[i8; 32]; 2] = [[0; 32], [0; 32]]\n    d: [[f32; 1]; 2] = [[0.0 as f32], [0.0 as f32]]\n    z: [[f32; 32]; 1] = [[0.0 as f32; 32]]\n    y: [[f64; 4]; 2] = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]\nevent Go\nstep S on Go\n";
    for (body, rule, words) in [
        ("    y = tensor.rmsnorm(x, w, 0.00001)\n", "W-GRAPH-1", "the parameters of [[f64; 4]; 2] are of the shape [f64; 4], found [f64; 3]"),
        ("    y = tensor.reshape(x, [3, 3])\n", "W-GRAPH-1", "has 8 elements and the shape [3, 3] holds 9"),
        ("    y = tensor.permute(x, [0, 0])\n", "W-GRAPH-1", "is not a permutation of the axes"),
        ("    y = tensor.softmax(x, 2)\n", "W-GRAPH-1", "axis 2 of [[f64; 4]; 2], which has rank 2"),
        ("    y = tensor.add(x, tensor.reshape(w, [3, 1]))\n", "W-GRAPH-1", "do not broadcast"),
        ("    y = tensor.softmax(x, -1)\n", "W-GRAPH-2", "an axis, is an integer literal from 0"),
        ("    y = tensor.rmsnorm(x, x, y)\n", "W-GRAPH-2", "an epsilon, is a float literal of at least 0"),
        ("    y = tensor.embed(x, w)\n", "W-GRAPH-2", "takes a table of f32 or f64 and ids of i32"),
        ("    z = tensor.matmul_q8_0(x, q, d)\n", "W-GRAPH-2", "takes x of f32, q of i8 and d of f32"),
        ("    z = tensor.matmul_q4_0(tensor.transpose(z), q, d)\n", "W-GRAPH-1", "k a multiple of 32"),
        ("    y = tensor.cumsum(x)\n", "W-GRAPH-2", "is not an operation of std.tensor"),
    ] {
        let e = rce_spec_check::check_one(&format!("{pre}{body}")).unwrap_err();
        assert!(e.contains(rule) && e.contains(words), "{body}: {e}");
    }
}

#[test]
fn rce_graph_runs_a_quantized_product_from_its_fg_file() {
    let src = "use std.tensor\nstate S\n    q: [[i8; 32]; 2] = [[1; 32], [-2; 32]]\n    d: [[f32; 1]; 2] = [[0.5 as f32], [0.25 as f32]]\n    y: [[f32; 2]; 1] = [[0.0 as f32, 0.0 as f32]]\nevent Go(x: [[f32; 32]; 1])\nstep S on Go(x)\n    y = tensor.matmul_q8_0(x, q, d)\n";
    let p = tmp_file("q.rce", src);
    let dir = tmp("qfg");
    let (st, out, err) = rce(&["graph", "terms", p.to_str().unwrap(), dir.to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    assert!(out.contains("tensor.matmul_q8_0  (f32[1,32], i8[2,32], f32[2,1]) -> f32[1,2]  exact"), "{out}");
    let key = out.split_whitespace().next().unwrap();
    let f = dir.join(format!("{key}.fg"));
    let ones = format!("[[{}]]", vec!["1.0"; 32].join(","));
    let q = format!("[[{}],[{}]]", vec!["1"; 32].join(","), vec!["-2"; 32].join(","));
    let (st, out, err) = rce(&["graph", "run", f.to_str().unwrap(), &format!("[{ones},{q},[[0.5],[0.25]]]")]);
    assert_eq!((st, out.as_str()), (0, "[[16.0,-16.0]]\n"), "{err}");
    // a Q8_0 value outside i8 is not an operand of the term
    let bad = q.replacen("1,", "128,", 1);
    let (st, _, err) = rce(&["graph", "run", f.to_str().unwrap(), &format!("[{ones},{bad},[[0.5],[0.25]]]")]);
    assert_eq!(st, 2, "{err}");
    assert!(err.contains("128 is not an integer of i8"), "{err}");
}
