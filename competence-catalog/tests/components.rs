//! Layer-separation falsifiers: the six components are independent projections of
//! one Form. Swapping/breaking any one cannot bias the others (zero accidental
//! mutual information) — the operational resolution of the chicken-and-egg. A
//! stateful (coupled) component is caught; the clean stack passes; the projections
//! converge on the Form's meaning.

use std::cell::Cell;
use std::rc::Rc;

use competence_catalog::components::{
    convergence, project_all, separation_holds, separation_report, six, Core, Engine, Form,
    Mutant, Projection, Spec, Vc,
};

#[test]
fn six_components_are_named() {
    let names: Vec<&str> = six().iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["RCE-SPEC", "RCE", "RCE-Compile", "RCE-Engine", "RCE-JEPA", "RCE-VC"]);
}

#[test]
fn every_component_is_independent() {
    let form = Form::with_accept("2 + 3", 5);
    let (rows, all) = separation_report(&form);
    assert_eq!(rows.len(), 6);
    assert!(all, "each component survives every other's swap: {rows:?}");
    assert!(rows.iter().all(|(_, ok)| *ok));
}

#[test]
fn a_stateful_coupled_component_is_caught() {
    // A component whose output is NOT a pure function of the Form (it carries
    // accidental state) leaks: its output drifts between projections. The
    // falsifier must catch it — this is what makes the check non-vacuous.
    let counter = Rc::new(Cell::new(0u64));
    struct Leaky { c: Rc<Cell<u64>> }
    impl Projection for Leaky {
        fn name(&self) -> &'static str { "leaky" }
        fn project(&self, _f: &Form) -> Result<String, String> {
            let n = self.c.get();
            self.c.set(n + 1);
            Ok(n.to_string())
        }
    }
    let comps: Vec<Box<dyn Projection>> = vec![
        Box::new(Spec),
        Box::new(Leaky { c: counter.clone() }),
        Box::new(Engine),
    ];
    let form = Form::new("2 + 3");
    // Swap component 0; the leaky component 1 must be detected as non-independent.
    assert!(!separation_holds(&form, &comps, 0, &Mutant));
}

#[test]
fn convergence_on_meaning() {
    assert!(convergence(&Form::with_accept("2 + 3", 5)));
    assert!(convergence(&Form::with_accept("2 + 3", 999))); // verdict consistent (refuse)
    assert!(!convergence(&Form::new("a + b"))); // not closed: no behavior to converge on
}

#[test]
fn projections_agree_on_the_form() {
    let form = Form::with_accept("2 + 3", 5);
    let out = project_all(&six(), &form);
    assert_eq!(out[0].as_deref(), Ok("5"));        // RCE-SPEC canonical meaning (folded)
    assert_eq!(out[3].as_deref(), Ok("5"));        // RCE-Engine behavior — agrees
    assert_eq!(out[5].as_deref(), Ok("admit"));    // RCE-VC verdict
    // RCE content key is name-free: equivalent surfaces share it.
    let k1 = Core.project(&Form::new("2 + 3")).unwrap();
    let k2 = Core.project(&Form::new("3 + 2")).unwrap();
    assert_eq!(k1, k2);
}

#[test]
fn mutant_is_non_vacuous() {
    let form = Form::new("2 + 3");
    let mutant = Mutant.project(&form).unwrap();
    let real = Spec.project(&form).unwrap();
    assert_ne!(mutant, real);
    // The clean stack is invariant to the mutant swap (independence holds).
    assert!(separation_holds(&form, &six(), 2, &Mutant));
}

#[test]
fn vc_refuses_a_failed_contract() {
    let form = Form::with_accept("2 + 3", 6);
    assert_eq!(Vc.project(&form).as_deref(), Ok("refuse"));
}