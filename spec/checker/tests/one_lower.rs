//! The lowering of the one emitter (spec/proposals/one-emitter.md ONE-4,
//! section 3.5; order of work step 3), tested differentially on
//! spec/examples: for every program the lowering covers, the trace printed
//! by evaluating its core (`rce run-core`) equals the reference
//! interpreter's trace byte for byte. Every example is listed below, either
//! as covered or with the first construct the lowering does not yet cover;
//! a program that is not covered is refused with that construct named, and
//! never printed differently.
//!
//! The conformance suite is tested the same way in one_conformance.rs.
use rce_spec_check::one;
use std::path::PathBuf;

/// The programs of spec/examples whose core prints the reference's trace.
/// Functions are lowered when the trace first reaches them, so a program
/// is covered when every construct its observations run is covered; the
/// probes below reach every function body.
const COVERED: &[&str] = &[
    "01_counter", "02_hello_main", "03_fib_loop", "04_bank_account", "05_kind_match", "06_option",
    "07_error_union", "08_traits", "09_generics", "10_comptime", "11_arena", "12_borrow", "14_select",
    "16_multi_region", "17_traffic_light", "18_while_bound", "19_floats", "20_strings", "21_maps",
    "22_tuples", "23_patterns", "24_lambda", "25_labels", "27_bitops", "28_slices", "29_nested_kind",
    "30_state_machine_guarded", "31_inventory", "32_comptime_block", "33_in_operator", "34_chan_types",
    "35_nested_if", "36_multiline_brackets", "38_hex_bin", "40_extern", "41_arena_bound_wrapping", "42_f32",
    "43_simd_lanes", "44_device_region", "45_byte_text", "46_u64", "47_select_fair", "48_linear_handle",
    "49_loop_limit", "50_iterators", "51_refined_input", "52_context_text", "53_capability", "54_secret",
    "55_rcedb", "56_utf8", "57_float_vectors", "58_float_keys", "59_dim_redundant_field",
    "60_dim_hidden_period", "61_dim_minimal",
];

/// The programs the lowering does not yet cover, and the construct it
/// names first. Each is refused with that construct; none prints a trace.
const NOT_COVERED: &[(&str, &str)] = &[
    ("13_spawn_chan", "a channel"),
    ("15_actor", "the type chan i64"),
    ("37_actor_spawn", "the type chan u64"),
    ("39_structural_recursion", "a recursive type"),
];

/// The covered programs whose probes (the extra observations of the
/// emitter tests, `emit::with_probes`, which call every function on sample
/// arguments) also lower with identical traces.
const COVERED_WITH_PROBES: &[&str] = &[
    "01_counter", "02_hello_main", "03_fib_loop", "04_bank_account", "05_kind_match", "06_option",
    "07_error_union", "08_traits", "09_generics", "10_comptime", "11_arena", "12_borrow", "14_select",
    "16_multi_region", "17_traffic_light", "18_while_bound", "19_floats", "20_strings", "22_tuples",
    "23_patterns", "24_lambda", "25_labels", "27_bitops", "28_slices", "30_state_machine_guarded",
    "31_inventory", "32_comptime_block", "33_in_operator", "34_chan_types", "35_nested_if",
    "36_multiline_brackets", "38_hex_bin", "40_extern", "41_arena_bound_wrapping", "42_f32", "43_simd_lanes",
    "44_device_region", "45_byte_text", "46_u64", "47_select_fair", "48_linear_handle", "49_loop_limit",
    "50_iterators", "51_refined_input", "52_context_text", "53_capability", "54_secret", "55_rcedb",
    "56_utf8", "57_float_vectors", "58_float_keys", "59_dim_redundant_field", "60_dim_hidden_period",
    "61_dim_minimal",
];

fn examples() -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("../examples")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.into_iter().map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap())).collect()
}

fn module(src: &str) -> rce_spec_check::ast::Module {
    assert!(rce_spec_check::cli::diagnostics(src, &Default::default()).is_empty());
    rce_spec_check::program(src).unwrap()
}

fn first_diff(want: &str, got: &str) -> String {
    let (w, g): (Vec<&str>, Vec<&str>) = (want.lines().collect(), got.lines().collect());
    let k = w.iter().zip(&g).position(|(a, b)| a != b).unwrap_or(w.len().min(g.len()));
    format!("line {}: want {:?}, got {:?}", k + 1, w.get(k), g.get(k))
}

/// The differential test of the lowering on spec/examples.
#[test]
fn examples_lower_to_the_reference_trace() {
    let all = examples();
    assert_eq!(all.len(), COVERED.len() + NOT_COVERED.len(), "every example is listed once");
    let mut same = 0;
    for (stem, src) in &all {
        let m = module(src);
        let want = one::trace::reference(&m);
        let got = one::run_core_of(&m);
        if COVERED.contains(&stem.as_str()) {
            let got = got.unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(got == want, "{stem}: {}", first_diff(&want, &got));
            same += 1;
        } else {
            let (_, why) = NOT_COVERED.iter().find(|(n, _)| n == stem).unwrap_or_else(|| panic!("{stem} is not listed"));
            let e = got.err().unwrap_or_else(|| panic!("{stem} is now covered; move it to COVERED"));
            assert!(e.starts_with("not covered by the lowering: ") && e.contains(why), "{stem}: {e}");
        }
    }
    eprintln!("examples: {}; identical traces: {same}; not yet covered: {}", all.len(), all.len() - same);
}

/// The same with the probe observations, which reach every function body.
#[test]
fn examples_with_probes_lower_to_the_reference_trace() {
    for (stem, src) in examples() {
        let m = rce_spec_check::emit::with_probes(&module(&src));
        let want = one::trace::reference(&m);
        match one::run_core_of(&m) {
            Ok(got) => {
                assert!(COVERED_WITH_PROBES.contains(&stem.as_str()), "{stem} with probes is now covered; list it");
                assert!(got == want, "{stem} with probes: {}", first_diff(&want, &got));
            }
            Err(e) => {
                assert!(!COVERED_WITH_PROBES.contains(&stem.as_str()), "{stem} with probes: {e}");
                assert!(e.starts_with("not covered by the lowering: "), "{stem} with probes: {e}");
            }
        }
    }
}

/// Small programs for constructs the examples reach little: lambdas a
/// `let` makes (captures, calls in a loop, the trap of the last parameter
/// whose fit fails), `set_of` in canonical order, sums of f64 (and the zero
/// they start from), a call of a foreign function, which traps, and
/// iterators (nested, left by `break`, and trapping after items the
/// consumer has shown); and the JSON sizes a region with a bound adds up
/// (R3).
const PROGRAMS: &[&str] = &[
    "fn main()\n    let k = 10\n    let add = fn(x: i64) => x + k\n    let j = 99\n    emit add(1)\n    emit add(j)\n    let s = \"ab\"\n    let cat = fn(t: text) => t + s\n    emit cat(cat(\"y\"))\n",
    "type Pct = u8 where self <= 100\ntype Name = text where self.len() >= 1 and self.len() <= 2\n\nfn main()\n    let pair = fn(a: Pct, b: Name) => b.len_bytes() + a.tostr().len_bytes()\n    emit pair(1, \"x\")\n    let n: u8 = 200\n    let t = \"long\"\n    emit pair(n, t)\n",
    "fn twice(n: i64) -> i64\n    let f = fn(x: i64) => x * 2\n    var t = 0\n    for i in 0..n\n        t += f(i)\n    t\n\nfn main()\n    emit twice(5)\n",
    "fn main()\n    let xs = [1.5, -0.0, 2.25]\n    emit xs.sum()\n    let ys = [-0.0]\n    emit ys.sum()\n    let zs = [1e308, 1e308]\n    emit zs.sum()\n",
    "fn main()\n    emit set_of([(2, \"b\"), (1, \"z\"), (2, \"a\"), (1, \"z\")])\n    emit set_of([(3, 1), (2, 0), (3, 1)]).len()\n    emit set_of([2.5, -0.0, 0.0, 1.0, 2.5])\n    emit set_of([true, false, true]).tolist()\n    let e: List[i64] = []\n    emit set_of(e)\n",
    "extern \"C\" fn crc32(w: &World, data: &[u8], seed: u32) -> u32\n\nfn main(w: &World)\n    let bytes: List[u8] = [1, 2, 3]\n    emit 1\n    emit crc32(w, &bytes, 0)\n    emit 2\n",
    "fn ramp(n: u8) -> iter u8\n    var i: u8 = 250\n    loop\n        yield i\n        i += n\n\nfn pairs(n: u64) -> iter (u64, text)\n    for i in 0..n\n        if i % 2 == 0\n            yield (i, \"even\")\n        else\n            yield (i, \"odd\")\n\nfn evens(n: u64) -> iter u64\n    for x in pairs(n)\n        if x.1 == \"even\"\n            yield x.0\n    return\n\nfn main()\n    for x in pairs(3)\n        emit x\n    for y in evens(200)\n        if y > 140\n            break\n        if y % 20 != 0\n            continue\n        emit y\n    emit evens(7).collect()\n    for z in ramp(2)\n        emit z\n",
];

/// A region with a bound holding values of most types: run at bounds that
/// trap after different statements, and at one that holds them all.
const REGION: &str = "struct P\n    x: i64\n    name: text\n\nkind Shape\n    Circle(r: f64)\n    Dot\n\nfn sizes(limit: u64) -> i64\n    var n: i64 = 0\n    with arena a bound limit\n        let xs: List[i64] = [1, -22, 333]\n        n += xs.len() as i64\n        let p = P(x: -5, name: \"a\\\"b\\n\\u{1}é\")\n        n += p.x\n        let s = Circle(r: 0.1)\n        let d = Dot\n        if s == d\n            n += 1\n        let o: i64? = some(3)\n        let q: i64? = none\n        n += o.or(0) + q.or(1)\n        let m: Map[text, f64] = {\"k\": 1e21, \"j\": 1.5e-7, \"z\": -0.0}\n        n += m.len() as i64\n        let t = (true, false, 2.5 as f32, 1e16)\n        if t.0\n            n += 1\n        var v: List[text] = [\"é\", \"\\t\"]\n        v.push(\"z\")\n        n += v.len() as i64\n        let e = fails(n) catch 0\n        n += e\n    return n\n\nerror Bad(code: i64)\n\nfn fails(k: i64) -> i64 ! Bad\n    if k > 0\n        fail Bad(code: k)\n    return k\n\nfn main()\n    emit sizes(LIMIT)\n";

/// Budgets (C.5, C.6) around most constructs, run at limits that stop the
/// block after different charges, and at one that lets it finish: the
/// cost is charged where the reference charges it.
const BUDGETS: &[(&str, &[i64])] = &[
    ("struct P\n    x: i64\n    ys: List[i64]\n\nfn sq(n: i64) -> i64 = n * n\n\nfn evens(n: u64) -> iter u64\n    var i: u64 = 0\n    while i < n\n        yield i * 2\n        i += 1\n\nfn work() -> i64\n    var total: i64 = 0\n    var p = P(x: 1, ys: [1, 2, 3])\n    with budget LIMIT\n        let xs = [3, 1, 2]\n        var m: Map[text, i64] = {\"a\": 1}\n        m[\"b\"] = 2\n        m.insert(\"c\", 3)\n        total += m.get(\"z\") catch 7\n        total += xs[0] + xs.len() as i64\n        let s = \"hello\"\n        total += s[1:3].len_bytes() as i64\n        p.x += 5\n        var q = copy(p.ys)\n        q[1] = -q[0]\n        total += q[1]\n        let t = (4, 2.5)\n        if t.0 > 3 and not (t.1 < 1.0) or false\n            total += sq(t.0)\n        let f = fn(a: i64) => a + 1\n        total += f(2)\n        let ys = xs.map(fn(a: i64) => a * 2).filter(fn(a: i64) => a > 2)\n        total += ys.fold(0, fn(a: i64, b: i64) => a + b)\n        let zs = [k * k for k in 0..4 if k != 2]\n        total += zs.len() as i64\n        for k in evens(3)\n            total += k as i64\n        var fl: f64 = 1.5\n        fl = fl * 2.0 / 4.0\n        if fl > 0.5\n            total += 1\n        var acc = 0\n        for i in 0..4 par\n            acc += i\n        total += acc\n        var u = 0\n        var v = 0\n        par\n            branch one\n                u = sq(3)\n            branch two\n                v = 2\n        total += u + v\n        with budget 1000000\n            total += -1\n        else\n            total += 100\n        var w: List[i64] = []\n        w.push(4)\n        w.push(5)\n        total += w.len() as i64\n        if 2 in 0..3 and 1 in xs\n            total += 1\n    else as e\n        total = 0 - (e.used as i64) - 1000 * (e.limit as i64)\n    return total + p.x\n\nfn main()\n    emit work()\n", &[0, 9, 37, 64, 101, 150, 199, 250, 330, 1000]),
    ("kind Shape\n    Circle(r: f64)\n    Square(s: i64)\n\nerror Bad(code: i64)\n\nfn area(s: Shape) -> i64\n    match s\n        Circle(r: r) -> (r * r) as i64\n        Square(s: q) -> q * q\n\nfn chk(k: i64) -> i64 ! Bad\n    if k > 3\n        fail Bad(code: k)\n    return k * 2\n\nfn nat(n: u64) -> iter u64\n    for i in 0..n\n        yield i\n\nfn work(seed: i64) -> i64\n    var total: i64 = seed\n    with budget LIMIT\n        let s = \"a,bb,ccc\"\n        let parts = s.split(\",\")\n        total += parts.len() as i64\n        let up = s.upper()\n        if up.contains(\"BB\")\n            total += 1\n        var xs = [5, 3, 9, 1]\n        xs.sort()\n        total += xs[0]\n        let j = \",\".join(parts)\n        total += j.len_bytes() as i64\n        let st = set_of([3, 1, 3])\n        total += st.len() as i64\n        var k = 0\n        while k < 3\n            total += area(Square(s: k))\n            k += 1\n        total += area(Circle(r: 1.5))\n        let c = chk(2) catch 0\n        let d = chk(9) catch 1\n        total += c + d\n        let m: Map[text, i64] = {\"x\": 1, \"y\": 2}\n        for kv in m\n            total += kv.1\n        let g = nat(4).collect()\n        total += g.len() as i64\n        let rep = [0; 5]\n        total += rep.len() as i64\n        let sq = [x * 2 for x in xs if x > 2]\n        total += sq.len() as i64\n        let fl = 7.0 / 2.0\n        let fm = fl * 3.0 - 1.0\n        if fm >= 9.5\n            total += fm as i64\n        let w = xs[1:3]\n        total += w[0]\n        let tx = s + \"!\"\n        total += tx.len_bytes() as i64\n        total += if total > 10 then 1 else 2\n        with budget 20\n            var z = 0\n            for i in 0..100\n                z += i\n            total += z\n        else as e\n            total += e.used as i64\n    else as e\n        total = 0 - (e.used as i64)\n    return total\n\nfn main()\n    emit work(1)\n", &[0, 13, 52, 99, 140, 181, 222, 263, 1000]),
];

#[test]
fn programs_lower_to_the_reference_trace() {
    let mut extra: Vec<String> = [0, 26, 40, 61, 100, 140, 158, 1000].iter().map(|b| REGION.replace("LIMIT", &b.to_string())).collect();
    for (t, limits) in BUDGETS {
        extra.extend(limits.iter().map(|b| t.replace("LIMIT", &b.to_string())));
    }
    for src in PROGRAMS.iter().copied().chain(extra.iter().map(|s| s.as_str())) {
        let m = module(src);
        let want = one::trace::reference(&m);
        let got = one::run_core_of(&m).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert!(got == want, "{src}: {}", first_diff(&want, &got));
    }
}

/// 3.4 on lowered programs: the JSON form reads back to the same module and
/// its text is canonical; `rce run-core` of that JSON prints the reference's
/// trace, and `rce emit core` of it prints it again (ONE-1: FILE may be a
/// core module).
#[test]
fn core_json_round_trip() {
    for (stem, src) in examples() {
        if !COVERED.contains(&stem.as_str()) {
            continue;
        }
        let m = module(&src);
        let c = one::core_of(&m).unwrap();
        let text = one::json::module_text(&c);
        let back = one::json::module_from_text(&text).unwrap();
        assert!(back == c, "{stem}: the JSON does not read back to the same module");
        assert_eq!(one::json::module_text(&back), text, "{stem}: the text is not canonical");
        assert!(text.starts_with("{\"defs\":{"), "{stem}: keys in sorted order");
        let t = rce_spec_check::cli::run_core(&text).unwrap();
        assert_eq!(t, one::trace::reference(&m), "{stem}: the trace of the core JSON");
        assert_eq!(rce_spec_check::cli::emit("core", &text).unwrap(), text + "\n");
    }
}

/// The lowering is deterministic: the same program gives the same bytes.
#[test]
fn core_is_deterministic() {
    let src = std::fs::read_to_string("../examples/04_bank_account.rce").unwrap();
    let a = rce_spec_check::cli::emit("core", &src).unwrap();
    let b = rce_spec_check::cli::emit("core", &src).unwrap();
    assert_eq!(a, b);
}

/// `rce emit core FILE` and `rce run-core FILE` through the command line:
/// status 0 with the bytes on standard output; a program the lowering does
/// not cover is status 2 with the construct named; a program the checker
/// refuses is status 1. `trace` is no longer an emit target: the core
/// evaluator's output is not the expected bytes.
#[test]
fn emit_core_and_run_core_commands() {
    let rce = |args: &[&str]| {
        let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (mut out, mut err) = (vec![], vec![]);
        let code = rce_spec_check::cli::main(&a, &mut &b""[..], &mut out, &mut err);
        (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
    };
    let counter = "../examples/01_counter.rce";
    let (c, out, _) = rce(&["run-core", counter]);
    assert_eq!(c, 0);
    assert_eq!(out, one::trace::reference(&module(&std::fs::read_to_string(counter).unwrap())));
    assert!(out.starts_with("example "), "{out}");
    let (c, core, _) = rce(&["emit", "core", counter]);
    assert_eq!(c, 0);
    let v: serde_json::Value = serde_json::from_str(&core).unwrap();
    assert_eq!(v["format"], "rce-core/1");
    let dir = std::env::temp_dir().join(format!("rce-one-lower-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("counter.core.json");
    std::fs::write(&p, &core).unwrap();
    let ps = p.to_string_lossy().into_owned();
    assert_eq!(rce(&["run-core", &ps]), (0, out.clone(), String::new()));
    assert_eq!(rce(&["emit", "core", &ps]), (0, core.clone(), String::new()));
    // a core module that is not well formed is refused when it is read
    let bad = core.replacen("\"ret\"", "\"halt\"", 1);
    std::fs::write(&p, &bad).unwrap();
    assert_eq!(rce(&["run-core", &ps]).0, 1);
    let (c, _, err) = rce(&["run-core", "../examples/13_spawn_chan.rce"]);
    assert_eq!(c, 2);
    assert!(err.starts_with("not covered by the lowering: "), "{err}");
    let q = dir.join("bad.rce");
    std::fs::write(&q, "fn f() -> i64\n    \"text\"\n").unwrap();
    assert_eq!(rce(&["run-core", &q.to_string_lossy()]).0, 1);
    let (c, _, err) = rce(&["emit", "trace", counter]);
    assert_eq!(c, 2);
    assert!(err.contains("rce run-core FILE"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
