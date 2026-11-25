use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::types::{exec_typed, monomorphize, Ty, TypedFn};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

fn id_fn() -> TypedFn {
    TypedFn {
        name: "id".into(),
        tparams: vec!["T".into()],
        params: vec![("x".into(), Ty::Var("T".into()))],
        ret: Ty::Var("T".into()),
        body: Stmt::Return(e("x")),
    }
}

#[test]
fn generic_id_checks_and_executes_per_instantiation() {
    let tf = id_fn();
    assert!(monomorphize(&tf, &[Ty::I64]).is_ok());
    assert!(monomorphize(&tf, &[Ty::I128]).is_ok());
    assert_eq!(exec_typed(&tf, &[Ty::I64], &[42]).unwrap(), 42);
    assert_eq!(exec_typed(&tf, &[Ty::I128], &[i64::MAX as i128 + 1]).unwrap(), i64::MAX as i128 + 1);
}

#[test]
fn i64_instantiation_range_checks_arguments() {
    let tf = id_fn();
    assert!(exec_typed(&tf, &[Ty::I64], &[i64::MAX as i128 + 1]).is_err());
}

#[test]
fn bool_param_in_arithmetic_is_declined() {
    let tf = TypedFn {
        name: "bad".into(),
        tparams: vec![],
        params: vec![("b".into(), Ty::Bool)],
        ret: Ty::I128,
        body: Stmt::Return(e("b + 1")),
    };
    assert!(monomorphize(&tf, &[]).is_err());
}

#[test]
fn return_type_mismatch_is_declined() {
    let tf = TypedFn {
        name: "bad".into(),
        tparams: vec![],
        params: vec![("x".into(), Ty::I128)],
        ret: Ty::Bool,
        body: Stmt::Return(e("x")),
    };
    assert!(monomorphize(&tf, &[]).is_err());
}

#[test]
fn type_arity_mismatch_is_declined() {
    let tf = id_fn();
    assert!(monomorphize(&tf, &[]).is_err());
    assert!(monomorphize(&tf, &[Ty::I64, Ty::I64]).is_err());
}

#[test]
fn typed_comparison_executes() {
    let tf = TypedFn {
        name: "cmp".into(),
        tparams: vec![],
        params: vec![("x".into(), Ty::I64)],
        ret: Ty::Bool,
        body: Stmt::Seq(vec![
            Stmt::Let("y".into(), e("x + 1")),
            Stmt::Return(e("y > x")),
        ]),
    };
    assert_eq!(exec_typed(&tf, &[], &[3]).unwrap(), 1);
}
