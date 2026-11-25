//! Symbolic canonical equivalence — unbounded-domain equivalence on the
//! arithmetic fragment.
//!
//! Bounded exhaustive equivalence (semantic.rs) is complete only on a finite
//! domain. For the polynomial fragment of arithmetic, a *canonical algebraic
//! form* gives equivalence for ALL inputs: expand an expression to a sorted sum
//! of monomials (a polynomial over the variables), and two expressions are
//! symbolically equivalent iff their canonical polynomials are identical. This
//! proves associativity, commutativity and distributivity identities without
//! testing any inputs — `(a+b)+c ≡ a+(b+c)`, `a*(b+c) ≡ a*b+a*c`,
//! `(a+b)*(a-b) ≡ a*a-b*b` — for every integer assignment.
//!
//! This is a decision procedure for polynomial identity (undecidable in full
//! generality, decidable on this fragment), the tractable core of unbounded
//! equivalence. Non-polynomial constructs (division, comparison, calls) fall
//! back to bounded equivalence.

use std::collections::BTreeMap;

use rce_spec_check::ast::Expr;

/// A polynomial: monomial (sorted variable list, with multiplicity) → coefficient.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    terms: BTreeMap<Vec<String>, i128>,
}

impl Poly {
    fn constant(n: i128) -> Poly {
        let mut p = Poly::default();
        if n != 0 {
            p.terms.insert(vec![], n);
        }
        p
    }

    fn var(x: &str) -> Poly {
        let mut p = Poly::default();
        p.terms.insert(vec![x.to_string()], 1);
        p
    }

    fn add(&self, other: &Poly) -> Poly {
        let mut out = self.clone();
        for (m, c) in &other.terms {
            let e = out.terms.entry(m.clone()).or_insert(0);
            *e += c;
            if *e == 0 {
                out.terms.remove(m);
            }
        }
        out
    }

    fn neg(&self) -> Poly {
        let mut out = Poly::default();
        for (m, c) in &self.terms {
            out.terms.insert(m.clone(), -c);
        }
        out
    }

    fn mul(&self, other: &Poly) -> Poly {
        let mut out = Poly::default();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let mut m = m1.clone();
                m.extend(m2.iter().cloned());
                m.sort();
                let e = out.terms.entry(m).or_insert(0);
                *e += c1 * c2;
            }
        }
        out.terms.retain(|_, c| *c != 0);
        out
    }

    /// The canonical string: sorted monomials with coefficients.
    pub fn canonical(&self) -> String {
        if self.terms.is_empty() {
            return "0".into();
        }
        let mut parts = Vec::new();
        for (m, c) in &self.terms {
            let monomial = if m.is_empty() {
                String::new()
            } else {
                m.join("*")
            };
            parts.push(format!("{c}*{monomial}"));
        }
        parts.join("+")
    }
}

/// Expand an expression to a polynomial, or None if it is outside the polynomial
/// fragment (division, comparison, calls, etc.).
pub fn to_poly(e: &Expr) -> Option<Poly> {
    use Expr::*;
    match e {
        Int(n, _) => Some(Poly::constant(*n)),
        Name(x, _) => Some(Poly::var(x)),
        Neg(a, _) => Some(to_poly(a)?.neg()),
        Bin(op, a, b, _) => {
            let pa = to_poly(a)?;
            let pb = to_poly(b)?;
            match op.as_str() {
                "+" => Some(pa.add(&pb)),
                "-" => Some(pa.add(&pb.neg())),
                "*" => Some(pa.mul(&pb)),
                _ => Option::None,
            }
        }
        _ => Option::None,
    }
}

fn parse(src: &str) -> Result<Expr, String> {
    let toks = rce_spec_check::parse::lex(src).map_err(|e| e.to_string())?;
    let mut p = rce_spec_check::parse::Parser::new(toks);
    p.expr().map_err(|e| format!("{e:?}"))
}

/// The canonical polynomial form of an arithmetic expression.
pub fn canonical_poly(src: &str) -> Result<String, String> {
    let e = parse(src)?;
    let p = to_poly(&e).ok_or_else(|| "outside the polynomial fragment".to_string())?;
    Ok(p.canonical())
}

/// Symbolic (unbounded) equivalence on the polynomial fragment: two expressions
/// are equal for ALL integer assignments iff their canonical polynomials match.
/// Returns None if either is outside the fragment (fall back to bounded testing).
pub fn symbolically_equivalent(a: &str, b: &str) -> Option<bool> {
    let pa = to_poly(&parse(a).ok()?)?;
    let pb = to_poly(&parse(b).ok()?)?;
    Some(pa == pb)
}
