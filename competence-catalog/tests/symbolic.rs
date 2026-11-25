//! Symbolic canonical equivalence falsifiers: unbounded (all-inputs) equivalence
//! on the polynomial fragment via canonical algebraic form — no inputs tested.

use competence_catalog::symbolic::{canonical_poly, symbolically_equivalent};

#[test]
fn associativity_holds_for_all_inputs() {
    assert_eq!(symbolically_equivalent("(a + b) + c", "a + (b + c)"), Some(true));
}

#[test]
fn commutativity_holds_for_all_inputs() {
    assert_eq!(symbolically_equivalent("a + b", "b + a"), Some(true));
    assert_eq!(symbolically_equivalent("a * b", "b * a"), Some(true));
}

#[test]
fn distributivity_holds_for_all_inputs() {
    assert_eq!(symbolically_equivalent("a * (b + c)", "a * b + a * c"), Some(true));
}

#[test]
fn difference_of_squares() {
    assert_eq!(symbolically_equivalent("(a + b) * (a - b)", "a * a - b * b"), Some(true));
}

#[test]
fn distinct_polynomials_are_not_equivalent() {
    assert_eq!(symbolically_equivalent("a + b", "a * b"), Some(false));
    assert_eq!(symbolically_equivalent("a + 1", "a + 2"), Some(false));
}

#[test]
fn canonical_form_is_unique() {
    assert_eq!(canonical_poly("a * (b + c)").unwrap(), canonical_poly("c * a + b * a").unwrap());
}

#[test]
fn outside_fragment_falls_back() {
    // Division is not polynomial: symbolic equivalence declines (None).
    assert_eq!(symbolically_equivalent("a / b", "a / b"), None);
}
