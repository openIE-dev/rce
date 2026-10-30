//! A deep value is cloned and compared without running out of the host's
//! stack (6.7: a deep program meets the defined `evaluation depth` trap or
//! finishes, never the host's stack size). `rce test` of a structural
//! recursion over a 1,500-deep value aborted with a native stack overflow
//! from depth 807 before `V::clone` and `V::eq` grew the stack.
use rce_spec_check::cli;

const SRC: &str = "kind Nest
    Leaf
    Node(inner: Box[Nest])

fn depth(n: &Nest) -> i64
    match n
        Leaf -> return 0
        Node(inner) -> return 1 + depth(&inner)

fn build(k: i64) -> Nest
    var t: Nest = Leaf
    for _i in 0..k
        t = Node(inner: Box(t))
    return t

fn measure(k: i64) -> i64
    let t = build(k)
    let u = t
    return depth(&u)

fn same(k: i64) -> bool = build(k) == build(k)

check measure(1500) == 1500
check same(1500)
";

#[test]
fn deep_values_clone_and_compare_without_overflow() {
    let d = std::env::temp_dir().join(format!("rce-deep-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("deep.rce");
    std::fs::write(&p, SRC).unwrap();
    let a = vec!["test".to_string(), p.to_string_lossy().into_owned()];
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    let out = String::from_utf8_lossy(&out);
    assert_eq!(code, 0, "{out}{}", String::from_utf8_lossy(&err));
    assert!(out.contains("2 checks") && out.contains("1 passed, 0 failed"), "{out}");
}
