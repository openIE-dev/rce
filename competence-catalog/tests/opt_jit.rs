use competence_catalog::jit::{OptJit, Val};

#[test]
fn type_feedback_specializes_then_deopts_on_signature_change() {
    let mut j = OptJit::new();
    let ints = vec![("x".to_string(), Val::Int(2)), ("y".to_string(), Val::Int(3))];
    let (v1, n1) = j.run("x + y", &ints).unwrap();
    assert_eq!(v1, "5");
    assert!(!n1);
    let (v2, n2) = j.run("x + y", &ints).unwrap();
    assert_eq!(v2, "5");
    assert!(n2, "should be specialized native");
    assert_eq!(j.specs, 1);
    // same source, different signature -> deopt to interpreter, correct answer
    let reals = vec![("x".to_string(), Val::Real(2.5)), ("y".to_string(), Val::Real(3.5))];
    let (v3, n3) = j.run("x + y", &reals).unwrap();
    assert_eq!(v3, "6");
    assert!(!n3);
    assert_eq!(j.deopts, 1);
}

#[test]
fn specialized_native_matches_interpreter() {
    let mut j = OptJit::new();
    let args = vec![("a".to_string(), Val::Int(6)), ("b".to_string(), Val::Int(7))];
    j.run("a * b", &args).unwrap();
    let (native, _) = j.run("a * b", &args).unwrap();
    assert_eq!(native, "42");
}

#[test]
fn bool_signature_specializes() {
    let mut j = OptJit::new();
    let args = vec![("p".to_string(), Val::Bool(true)), ("q".to_string(), Val::Bool(false))];
    j.run("p && q", &args).unwrap();
    let (v, native) = j.run("p && q", &args).unwrap();
    assert!(native);
    assert_eq!(v, "0");
}
