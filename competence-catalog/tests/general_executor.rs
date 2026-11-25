//! The general G0 executor: a shape-driven bridge executes any basis operation
//! whose signature can be marshalled from JSON — scalars, lists, text,
//! references, and error-union returns — at the interpreter floor, and reports
//! unsupported shapes honestly rather than guessing. Correctness is checked
//! against independently computed reference values.

use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn bridge() -> RuntimeBridge {
    RuntimeBridge::new_inprocess()
}

fn call(module: &str, name: &str, shape: &str, args: serde_json::Value) -> Result<String, String> {
    bridge().call_general(module, name, shape, &args)
}

fn approx(got: &str, want: f64) {
    let g: f64 = got.parse().unwrap_or_else(|_| panic!("not a float: {got}"));
    assert!((g - want).abs() <= 1e-9 * want.abs().max(1.0), "{got} != {want}");
}

#[test]
fn scalar_math_is_correctly_rounded() {
    approx(&call("math", "sin", "(f64)->f64", json!([0.5])).unwrap(), 0.5f64.sin());
    approx(&call("math", "sqrt", "(f64)->f64", json!([2.0])).unwrap(), 2.0f64.sqrt());
    approx(&call("math", "exp", "(f64)->f64", json!([1.0])).unwrap(), 1.0f64.exp());
}

#[test]
fn slice_reference_hash_matches_reference_vector() {
    // FNV-1a 32 of "hi" — independently computed reference value.
    let r = call("hash", "fnv1a32", "(&[u8])->u32", json!([[104, 105]])).unwrap();
    assert_eq!(r, "1748694682");
}

#[test]
fn list_and_scalar_search() {
    let r = call("col.search", "binary_search", "(&[i64],i64)->i64",
        json!([[1, 3, 5, 7, 9, 11], 7])).unwrap();
    assert_eq!(r, "3");
    let miss = call("col.search", "binary_search", "(&[i64],i64)->i64",
        json!([[1, 3, 5, 7, 9, 11], 8])).unwrap();
    assert_eq!(miss, "-1");
}

#[test]
fn text_references_return_bool() {
    let r = call("text.glob", "glob_match", "(&text,&text)->bool",
        json!(["*.txt", "file.txt"])).unwrap();
    assert_eq!(r, "true");
}

#[test]
fn calendar_scalar_to_bool() {
    assert_eq!(call("time.civil", "is_leap", "(i64)->bool", json!([2024])).unwrap(), "true");
    assert_eq!(call("time.civil", "is_leap", "(i64)->bool", json!([1900])).unwrap(), "false");
}

#[test]
fn error_union_return_executes() {
    // num.stats.mean returns f64 ! StatsError; the ok value is the mean.
    let r = call("num.stats", "mean", "(&[f64])->f64!StatsError",
        json!([[1.0, 2.0, 3.0]])).unwrap();
    approx(&r, 2.0);
}

#[test]
fn struct_args_and_returns_marshal() {
    // Complex numbers: struct args from JSON objects, struct return.
    let add = call("num.complex", "cx_add", "(&Complex,&Complex)->Complex",
        json!([{"re": 1.0, "im": 2.0}, {"re": 3.0, "im": 4.0}])).unwrap();
    assert_eq!(add, "Complex(re: 4, im: 6)");
    let mul = call("num.complex", "cx_mul", "(&Complex,&Complex)->Complex",
        json!([{"re": 1.0, "im": 2.0}, {"re": 3.0, "im": 4.0}])).unwrap();
    assert_eq!(mul, "Complex(re: -5, im: 10)");
    let norm = call("num.complex", "cx_norm2", "(&Complex)->f64",
        json!([{"re": 3.0, "im": 4.0}])).unwrap();
    approx(&norm, 25.0);
}

#[test]
fn stateful_mut_list_ops_execute() {
    // heap_push: &mut List, unit return — the mutated list is read back.
    let r = call("col", "heap_push", "(&mutList[T],T)->()", json!([[3, 1, 2], 4])).unwrap();
    assert!(r.starts_with('[') && r.contains('4'), "mutated heap returned: {r}");
    // set_add: &mut List, bool return (true when the element was added).
    let added = call("col", "set_add", "(&mutList[K],K)->bool", json!([[1, 2, 3], 4])).unwrap();
    assert_eq!(added, "true");
    let dup = call("col", "set_add", "(&mutList[K],K)->bool", json!([[1, 2, 3], 2])).unwrap();
    assert_eq!(dup, "false");
}

#[test]
fn generic_collections_execute() {
    // dq_len: a Deque marshalled from {front, back}.
    let r = call("col", "dq_len", "(&Deque[T0])->usize",
        json!([{"front": [1, 2, 3], "back": [4, 5]}])).unwrap();
    assert_eq!(r, "5");
    // dq_push_back: &mut Deque, unit return — the mutated deque is read back.
    let p = call("col", "dq_push_back", "(&mutDeque[T0],T0)->()",
        json!([{"front": [1], "back": []}, 2])).unwrap();
    assert!(p.starts_with("Deque("), "mutated deque returned: {p}");
    // om_insert: &mut OrdMap from {keys, vals}.
    let m = call("col", "om_insert", "(&mutOrdMap[K,T1],K,T1)->()",
        json!([{"keys": [1], "vals": [10]}, 2, 20])).unwrap();
    assert!(m.starts_with("OrdMap("), "mutated ordmap returned: {m}");
}

#[test]
fn unmarshalable_mut_shape_is_refused_not_guessed() {
    // A &mut generic struct (Deque) the registry cannot marshal is refused with
    // a reason, not fabricated.
    let r = call("col", "dq_push_back", "(&mutDeque[T0],T0)->()", json!([{}, 1]));
    assert!(r.is_err(), "an unmarshalable &mut shape must be refused");
}

#[test]
fn defunctionalized_op_combinator_executes() {
    // col.func.map takes a defunctionalized Op tag (Reynolds/Danvy): the
    // combinator applies the tag via an interpreter. Marshal the tag as data.
    let r = call("col.func", "map", "(&[i64],&Op)->List[i64]",
        json!([[-1, 2, -3], "Abs"])).unwrap();
    assert_eq!(r, "[1, 2, 3]");
    let even = call("col.func", "filter", "(&[i64],&Op)->List[i64]",
        json!([[1, 2, 3, 4], "Even"])).unwrap();
    assert_eq!(even, "[2, 4]");
}

#[test]
fn recursive_parser_kind_executes() {
    // text.parse.run takes a recursive defunctionalized PParser term.
    let r = call("text.parse", "run", "(&PParser,&List[text],usize)->List[i64]",
        json!([{"PChar": {"want": "a"}}, ["a", "b"], 0])).unwrap();
    assert_eq!(r, "[1, 0, 1]", "PChar('a') matches at pos 0");
    // A composed parser: PSeq(PChar('a'), PChar('b')) over ["a","b"].
    let seq = call("text.parse", "run", "(&PParser,&List[text],usize)->List[i64]",
        json!([{"PSeq": {"a": {"PChar": {"want": "a"}}, "b": {"PChar": {"want": "b"}}}}, ["a", "b"], 0])).unwrap();
    assert!(seq.starts_with("[1"), "PSeq(a,b) matches: {seq}");
    // Nullary tag PDigit.
    let d = call("text.parse", "run", "(&PParser,&List[text],usize)->List[i64]",
        json!(["PDigit", ["7"], 0])).unwrap();
    assert!(d.starts_with("[1"), "PDigit matches a digit: {d}");
}

#[test]
fn recursive_plist_kind_executes() {
    // col.persistent.PList = Nil | Cons(head, tail: List[PList]); the general
    // recursive kind registry marshals nested terms from JSON.
    let r = call("col.persistent", "plen", "(&PList)->i64",
        json!([{"Cons": {"head": 1, "tail": [{"Cons": {"head": 2, "tail": ["Nil"]}}]}}])).unwrap();
    assert_eq!(r, "2");
    let nil = call("col.persistent", "plen", "(&PList)->i64", json!(["Nil"])).unwrap();
    assert_eq!(nil, "0");
}

#[test]
fn generic_return_types_instantiate() {
    // The return generic (K, T0) is resolved by unifying against the arg types.
    let u = call("col", "set_union", "(&[K],&[K])->List[K]", json!([[1, 2], [2, 3]])).unwrap();
    assert_eq!(u, "[1, 2, 3]");
    let d = call("col", "dq_to_list", "(&Deque[T0])->List[T0]",
        json!([{"front": [1, 2], "back": [3]}])).unwrap();
    assert!(d.starts_with('['), "deque to list: {d}");
}
