//! `rce emit verilog` (spec 10.5, `spec/proposals/emit-verilog.md`).
//!
//! Run everywhere, with no simulator:
//!
//! * the module of a program is the same bytes on every emit, and equals
//!   its golden text in `tests/verilog/golden/` (all but the first line,
//!   which names the checker commit);
//! * the programs of `spec/examples/` that the subset admits are the ones
//!   listed here, and each program outside it gets one `requires change
//!   for target verilog: REASON, line N` and nothing on standard output;
//! * the harness `hdl/bench.py`, given a register dump that holds the
//!   reference evaluator's states and outcomes in the layout of the
//!   module's `rce-hdl/1` line, prints the reference's trace of every
//!   example: this checks the layout, the trap and error tables and the
//!   harness's state hashes, not the circuit.
//!
//! Run by `.github/workflows/verilog.yml`, where Icarus Verilog, Verilator
//! and Yosys are installed (marked ignored here, and run there with
//! `--include-ignored`; a tool that does not start fails the test): every
//! program is emitted, simulated by each simulator on its examples, and
//! the trace the harness computes from the dumps must equal the
//! reference's line for line; then Yosys synthesizes each module and the
//! cell counts are printed.
use rce_spec_check::emit::{trace_example, verilog};
use rce_spec_check::interp::{Interp, V};
use rce_spec_check::{ast, cli, parse};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

/// The programs whose modules have golden texts and run on the simulators:
/// the counter, the door, the flag game and the key handler of the laws
/// tests, and two programs of this test (an overflow that keeps the state,
/// and a register file with arrays, loops, functions, kinds and emits).
const FIXTURES: [(&str, &str); 6] = [
    ("counter", "../examples/01_counter.rce"),
    ("door", "../examples/30_state_machine_guarded.rce"),
    ("flag_walls", "tests/law/flag_walls.rce"),
    ("keys", "tests/law/keys.rce"),
    ("overflow", "tests/verilog/overflow.rce"),
    ("regfile", "tests/verilog/regfile.rce"),
];

/// The programs of spec/examples that the subset admits (EV-1).
const ADMITTED_EXAMPLES: [&str; 6] = [
    "01_counter.rce",
    "17_traffic_light.rce",
    "30_state_machine_guarded.rce",
    "59_dim_redundant_field.rce",
    "60_dim_hidden_period.rce",
    "61_dim_minimal.rce",
];

/// Every program the simulator tests run: the fixtures, then the admitted
/// examples not among them.
fn programs() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = FIXTURES.iter().map(|(n, p)| (n.to_string(), p.to_string())).collect();
    for e in ADMITTED_EXAMPLES {
        let p = format!("../examples/{e}");
        if !v.iter().any(|x| x.1 == p) {
            v.push((e.trim_end_matches(".rce").to_string(), p));
        }
    }
    v
}

fn module_of(src: &str) -> ast::Module {
    rce_spec_check::program(src).unwrap()
}

fn emit_file(path: &str) -> String {
    let (code, out, err) = rce(&["emit", "verilog", path]);
    assert_eq!(code, 0, "{path}: {err}");
    out
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-verilog-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The reference's trace of every example, as the harness prints it:
/// `example LINE`, then the block of `emit::trace_example` without its
/// `fuel` line.
fn want(src: &str) -> Vec<String> {
    let m = module_of(src);
    let mut out = vec![];
    for x in examples(&m) {
        out.push(format!("example {}", x.line));
        out.extend(trace_example(&m, x).into_iter().filter(|l| !l.starts_with("fuel ")));
    }
    out
}

fn examples(m: &ast::Module) -> Vec<&ast::Example> {
    let mut xs = vec![];
    for i in &m.items {
        match i {
            ast::Item::Intent(_, ex, _) => xs.extend(ex.iter()),
            ast::Item::Example(x) => xs.push(x),
            _ => {}
        }
    }
    xs
}

#[test]
fn two_emits_give_the_same_bytes() {
    for (n, p) in programs() {
        let a = emit_file(&p);
        let b = emit_file(&p);
        assert_eq!(a, b, "{n}: two emits differ");
        assert!(a.lines().nth(1).unwrap().starts_with("// rce emit verilog"), "{n}");
    }
}

/// The golden modules. A change to the emitter that changes a module
/// changes its golden text in the same commit:
/// `rce emit verilog FILE | tail -n +2 > tests/verilog/golden/NAME.v`.
#[test]
fn modules_equal_their_golden_text() {
    let mut differ = vec![];
    for (n, p) in FIXTURES {
        let got = emit_file(p);
        let got: String = got.split_once('\n').unwrap().1.to_string();
        let gp = format!("tests/verilog/golden/{n}.v");
        let want = std::fs::read_to_string(&gp).unwrap_or_else(|e| panic!("{gp}: {e}"));
        if got != want {
            let i = got.lines().zip(want.lines()).position(|(a, b)| a != b).unwrap_or(0);
            differ.push(format!("{gp}: line {}: want {:?}, got {:?}", i + 2, want.lines().nth(i), got.lines().nth(i)));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

#[test]
fn the_admitted_examples_are_the_listed_ones() {
    let mut admitted = vec![];
    let mut files: Vec<PathBuf> = std::fs::read_dir("../examples").unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "rce")).collect();
    files.sort();
    for f in &files {
        let (code, out, err) = rce(&["emit", "verilog", f.to_str().unwrap()]);
        if code == 0 {
            admitted.push(f.file_name().unwrap().to_string_lossy().to_string());
        } else {
            assert!(out.is_empty(), "{}: a refused program printed text", f.display());
            assert!(err.starts_with("requires change for target verilog: ") || err.contains("W-"), "{}: {err}", f.display());
        }
    }
    assert_eq!(admitted, ADMITTED_EXAMPLES, "the subset admits other examples");
}

/// A program outside the subset: one line naming the reason and the line,
/// status 1, nothing printed.
#[test]
fn programs_outside_the_subset_require_change() {
    let base = "state S\n    n: u8 = 0\n\nevent E\n\n";
    let cases: Vec<(String, &str)> = vec![
        (format!("{base}step S on E\n    let t = \"hi\"\n    n = t.len_bytes() as u8\n"), "text (EV-1: no dynamic text), line 7"),
        ("state S\n    xs: List[u8] = []\n\nevent E\n\nstep S on E\n    pass\n".into(), "the type `List[u8]`: the circuit holds integers, bool, kinds without fields and fixed arrays (EV-1), line 2"),
        (format!("{base}step S on E\n    var i: u8 = 0\n    while i < n\n        i += 1\n"), "a loop without a constant `bound` (EV-4), line 8"),
        ("state S\n    n: u64 = 0\n\nevent E\n\nstep S on E\n    var i: u64 = 0\n    while i < 3 bound n\n        i += 1\n".into(), "a loop bound that is not a constant (EV-4), line 8"),
        (format!("{base}fn main()\n    emit 1\n\nstep S on E -> n\n"), "`fn main` (tasks run on the tick runtime, not in a step), line 6"),
        (format!("{base}step S on E -> n.min(3)\n"), "the method `.min` (EV-1), line 6"),
        (format!("{base}step S on E\n    var x = 0\n    if n > 3\n        x = n\n    n = x\n"), "`x` is an integer (u8) on one path and an integer (lit) on another, so its type is not static (EV-2), line 9"),
        (format!("{base}fn f(x: u8) -> u8 bound 4\n    if x == 0\n        return 0\n    return f(x - 1)\n\nstep S on E -> f(n)\n"), "the recursive function `f` (EV-1: calls are inlined), line 9"),
        ("state S\n    n: f64 = 0.0\n\nevent E\n\nstep S on E -> n\n".into(), "the type `f64`: the circuit holds integers, bool, kinds without fields and fixed arrays (EV-1), line 2"),
        (format!("{base}step S on E\n    for _i in 0..30000\n        n = 1\n"), "a `for` range of 30000 items, above the size ceiling of 20000 (EV-4), line 7"),
    ];
    let dir = scratch("refuse");
    for (i, (src, msg)) in cases.iter().enumerate() {
        let p = dir.join(format!("r{i}.rce"));
        std::fs::write(&p, src).unwrap();
        let (code, out, err) = rce(&["emit", "verilog", p.to_str().unwrap()]);
        assert!(out.is_empty(), "case {i}: printed {out}");
        assert_eq!(err.trim_end(), format!("requires change for target verilog: {msg}"), "case {i}:\n{src}");
        assert_eq!(code, 1, "case {i}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------- the harness, replayed

fn iface(module: &str) -> Value {
    let l = module.lines().find(|l| l.starts_with("// rce-hdl/1 ")).unwrap();
    serde_json::from_str(&l["// rce-hdl/1 ".len()..]).unwrap()
}

fn ty_width(t: &Value, kinds: &Value) -> usize {
    match t {
        Value::String(s) if s == "bool" => 1,
        Value::String(s) => match s.as_str() {
            "i8" | "u8" => 8,
            "i16" | "u16" => 16,
            "i32" | "u32" => 32,
            _ => 64,
        },
        Value::Object(o) if o.contains_key("kind") => {
            let n = kinds[o["kind"].as_str().unwrap()].as_array().unwrap().len();
            let mut b = 1;
            while (1usize << b) < n {
                b += 1;
            }
            b
        }
        Value::Object(o) => ty_width(&o["array"], kinds) * o["len"].as_u64().unwrap() as usize,
        _ => panic!("type {t}"),
    }
}

/// The bits of `v` (type `t` of the interface), least significant first.
fn bits(t: &Value, v: &V, kinds: &Value, out: &mut Vec<bool>) {
    match v {
        V::Int(x, _) => (0..ty_width(t, kinds)).for_each(|i| out.push((x >> i) & 1 == 1)),
        V::Bool(b) => out.push(*b),
        V::Rec(n, _) => {
            let k = kinds[t["kind"].as_str().unwrap()].as_array().unwrap().iter().position(|x| x == n).unwrap();
            (0..ty_width(t, kinds)).for_each(|i| out.push((k >> i) & 1 == 1));
        }
        V::List(xs) => xs.iter().for_each(|x| bits(&t["array"], x, kinds, out)),
        o => panic!("value {}", o.show()),
    }
}

fn hex(b: &[bool]) -> String {
    let mut s = String::new();
    for d in (0..b.len().div_ceil(4).max(1)).rev() {
        let v = (0..4).filter(|i| b.get(d * 4 + i).copied().unwrap_or(false)).fold(0u32, |a, i| a | 1 << i);
        s.push(std::char::from_digit(v, 16).unwrap());
    }
    s
}

fn int_bits(x: i128, w: usize) -> Vec<bool> {
    (0..w).map(|i| (x >> i.min(127)) & 1 == 1).collect()
}

/// The register dump a correct module would give on the program's
/// examples, from the reference evaluator, in the layout of `module`.
fn reference_dump(src: &str, module: &str) -> String {
    let m = module_of(src);
    let i = iface(module);
    let kinds = &i["kinds"];
    let regions = i["regions"].as_array().unwrap();
    let state_bits = |s: &[rce_spec_check::interp::Region]| {
        let mut b = vec![];
        for (r, reg) in s.iter().zip(regions) {
            for ((_, v, _), f) in r.1.iter().zip(reg["fields"].as_array().unwrap()) {
                bits(&f["type"], v, kinds, &mut b);
            }
        }
        hex(&b)
    };
    let errors: Vec<&str> = i["errors"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let traps = i["traps"].as_array().unwrap();
    let tvw = i["trap_value_width"].as_u64().unwrap() as usize;
    let (etw, evw) = (i["emit_type_width"].as_u64().unwrap() as usize, i["emit_value_width"].as_u64().unwrap() as usize);
    let emit_types = i["emit_types"].as_array().unwrap();
    let mut out = String::new();
    for x in examples(&m) {
        let mut it = Interp::new(&m);
        let mut s = it.initial_state().unwrap();
        out += &format!("R {}\n", state_bits(&s));
        for (evn, args) in &x.events {
            it.scopes = vec![HashMap::new()];
            let ev = it.construct(evn, args).unwrap().unwrap();
            let (mut err, mut trap, mut tval) = (0, 0, 0i128);
            let (mut ok, mut et, mut ev_bits, mut n) = (false, vec![], vec![], 0);
            match it.delta(&s, &ev) {
                Ok((s2, outs)) => {
                    s = s2;
                    ok = true;
                    for o in &outs {
                        let (t, b) = match o {
                            V::Int(x, _) => ("int".to_string(), int_bits(*x, evw)),
                            V::Bool(b) => ("bool".to_string(), int_bits(*b as i128, evw)),
                            V::Rec(n, _) => {
                                let k = kinds.as_object().unwrap().iter().find(|(_, vs)| vs.as_array().unwrap().iter().any(|v| v == n)).unwrap();
                                let idx = k.1.as_array().unwrap().iter().position(|v| v == n).unwrap();
                                (k.0.clone(), int_bits(idx as i128, evw))
                            }
                            o => panic!("emit {}", o.show()),
                        };
                        let ti = emit_types.iter().position(|x| x == &Value::String(t.clone())).unwrap_or_else(|| panic!("no emit type {t}"));
                        et.extend(int_bits(ti as i128, etw));
                        ev_bits.extend(b);
                        n += 1;
                    }
                }
                Err(r) => {
                    if let Some(k) = errors.iter().position(|e| *e == r) {
                        err = k + 1;
                    } else {
                        let msg = r.strip_prefix("trap(").and_then(|x| x.strip_suffix(')')).unwrap_or_else(|| panic!("reason {r}"));
                        let pos = if let Some(rest) = msg.strip_prefix("overflow: ") {
                            let (v, t) = rest.split_once(" is outside ").unwrap();
                            tval = v.parse().unwrap();
                            traps.iter().position(|x| x["kind"] == "overflow" && x["type"] == t)
                        } else if let Some(rest) = msg.strip_prefix("index ").and_then(|x| x.strip_suffix(" out of range")) {
                            tval = rest.parse().unwrap();
                            traps.iter().position(|x| x["kind"] == "index")
                        } else {
                            traps.iter().position(|x| x["kind"] == "text" && x["text"] == msg)
                        };
                        trap = pos.unwrap_or_else(|| panic!("the module has no trap for {msg}")) + 1;
                    }
                }
            }
            out += &format!(
                "S {} {:x} {:x} {} {:x} {} {} {}\n",
                ok as u8,
                err,
                trap,
                hex(&int_bits(tval, tvw)),
                n,
                hex(&et),
                hex(&ev_bits),
                state_bits(&s)
            );
        }
    }
    out
}

fn bench(word: &str, module: &Path, third: &Path) -> Vec<String> {
    let o = Command::new("python3").arg("hdl/bench.py").arg(word).arg(module).arg(third).output().unwrap();
    assert!(o.status.success(), "bench.py {word} {}: {}{}", module.display(), String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(String::from).collect()
}

fn first_diff(want: &[String], got: &[String]) -> Option<String> {
    if want == got {
        return None;
    }
    let i = (0..want.len().max(got.len())).find(|&i| want.get(i) != got.get(i)).unwrap();
    Some(format!("line {}: want {:?}, got {:?}", i + 1, want.get(i), got.get(i)))
}

/// The harness prints the reference's trace from a dump of the
/// reference's states: the interface's layout, tables and hashing agree
/// with the reference on every program (no simulator involved).
#[test]
fn the_harness_prints_the_reference_trace_from_a_reference_dump() {
    let dir = scratch("replay");
    for (n, p) in programs() {
        let src = std::fs::read_to_string(&p).unwrap();
        let module = emit_file(&p);
        let mf = dir.join(format!("{n}.v"));
        std::fs::write(&mf, &module).unwrap();
        let df = dir.join(format!("{n}.dump"));
        std::fs::write(&df, reference_dump(&src, &module)).unwrap();
        let got = bench("replay", &mf, &df);
        if let Some(d) = first_diff(&want(&src), &got) {
            panic!("{n}: {d}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------- simulators and synthesis (CI)

fn tool_version(tool: &str, arg: &str) -> String {
    let o = Command::new(tool).arg(arg).output().unwrap_or_else(|e| panic!("`{tool}` does not run ({e}); this test needs it on PATH"));
    let s = String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr);
    s.lines().next().unwrap_or("").to_string()
}

fn simulate_all(sim: &str) {
    let dir = scratch(sim);
    let mut lines = 0;
    let mut kept = 0;
    for (n, p) in programs() {
        let src = std::fs::read_to_string(&p).unwrap();
        let mf = dir.join(format!("{n}.v"));
        std::fs::write(&mf, emit_file(&p)).unwrap();
        let w = dir.join(n.clone());
        let got = bench(sim, &mf, &w);
        let want = want(&src);
        if let Some(d) = first_diff(&want, &got) {
            panic!("{sim}: {n}: {d}");
        }
        let k = want.iter().filter(|l| l.starts_with("d kept ")).count();
        eprintln!("{sim}: {n}: {} examples, {} trace lines, {} steps kept: equal to the reference", examples(&module_of(&src)).len(), want.len(), k);
        lines += want.len();
        kept += k;
    }
    eprintln!("{sim}: {} programs agree with the reference ({lines} trace lines, {kept} kept steps)", programs().len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs Icarus Verilog; run by .github/workflows/verilog.yml"]
fn icarus_traces_equal_the_reference() {
    eprintln!("iverilog: {}", tool_version("iverilog", "-V"));
    simulate_all("iverilog");
}

#[test]
#[ignore = "needs Verilator; run by .github/workflows/verilog.yml"]
fn verilator_traces_equal_the_reference() {
    eprintln!("verilator: {}", tool_version("verilator", "--version"));
    simulate_all("verilator");
}

#[test]
#[ignore = "needs Yosys; run by .github/workflows/verilog.yml"]
fn yosys_synthesizes_every_module() {
    eprintln!("yosys: {}", tool_version("yosys", "-V"));
    let dir = scratch("yosys");
    for (n, p) in programs() {
        let mf = dir.join(format!("{n}.v"));
        std::fs::write(&mf, emit_file(&p)).unwrap();
        let out = bench("yosys", &mf, &dir.join(n.clone()));
        let cells = out.iter().find_map(|l| l.strip_prefix("cells ")).unwrap_or("?");
        assert!(cells.parse::<u64>().is_ok_and(|c| c > 0), "{n}: no cell count:\n{}", out.join("\n"));
        eprintln!("yosys: {n}: {cells} cells");
        for l in out.iter().skip(2) {
            eprintln!("yosys: {n}: {}", l.trim());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_emitter_and_the_cli_agree() {
    let src = std::fs::read_to_string("../examples/01_counter.rce").unwrap();
    let a = verilog::emit(&module_of(&src)).unwrap();
    assert_eq!(a, emit_file("../examples/01_counter.rce"));
    let _ = parse::parse(&src).unwrap();
}
