use competence_catalog::jit::TracingJit;

#[test]
fn hot_subtree_compiles_once_then_reuses_native() {
    let mut j = TracingJit::new();
    let (v1, n1) = j.run("2+3").unwrap();
    assert_eq!(v1, "5");
    assert!(!n1);
    let (v2, n2) = j.run("2+3").unwrap();
    assert_eq!(v2, "5");
    assert!(n2);
    assert_eq!(j.subtree_compiles, 1);
    let (v3, n3) = j.run("2+3").unwrap();
    assert_eq!(v3, "5");
    assert!(n3);
    assert_eq!(j.subtree_compiles, 1); // compiled once, reused
}

#[test]
fn native_result_matches_interpreter() {
    let mut j = TracingJit::new();
    j.run("2*3+4").unwrap();
    let (v, native) = j.run("2*3+4").unwrap();
    assert!(native);
    assert_eq!(v, "10");
}

#[test]
fn guard_failure_deopts_never_wrong_answer() {
    let mut j = TracingJit::new();
    assert!(j.run("x + 1").is_err());
    assert!(j.run("x + 1").is_err()); // hot now; native compile of open form fails
    assert_eq!(j.deopts, 1);
}

#[test]
fn distinct_subtrees_each_pay_once() {
    let mut j = TracingJit::new();
    for _ in 0..2 {
        j.run("2+3").unwrap();
        j.run("4+5").unwrap();
    }
    assert_eq!(j.subtree_compiles, 2);
}
