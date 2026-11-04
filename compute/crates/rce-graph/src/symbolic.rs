//! Symbolic computation: expression trees, symbolic differentiation,
//! simplification, evaluation, pretty printing.

use std::fmt;

// ═══════════════════════════════════════════════════════════════════
// EXPRESSION TREE
// ═══════════════════════════════════════════════════════════════════

/// Symbolic expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Tan(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
}

impl Expr {
    pub fn constant(v: f64) -> Self { Expr::Const(v) }
    pub fn var(name: &str) -> Self { Expr::Var(name.to_string()) }
    pub fn add(a: Expr, b: Expr) -> Self { Expr::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Expr, b: Expr) -> Self { Expr::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Expr, b: Expr) -> Self { Expr::Mul(Box::new(a), Box::new(b)) }
    pub fn div(a: Expr, b: Expr) -> Self { Expr::Div(Box::new(a), Box::new(b)) }
    pub fn pow(a: Expr, b: Expr) -> Self { Expr::Pow(Box::new(a), Box::new(b)) }
    pub fn neg(a: Expr) -> Self { Expr::Neg(Box::new(a)) }
    pub fn sin(a: Expr) -> Self { Expr::Sin(Box::new(a)) }
    pub fn cos(a: Expr) -> Self { Expr::Cos(Box::new(a)) }
    pub fn tan(a: Expr) -> Self { Expr::Tan(Box::new(a)) }
    pub fn exp(a: Expr) -> Self { Expr::Exp(Box::new(a)) }
    pub fn ln(a: Expr) -> Self { Expr::Ln(Box::new(a)) }
    pub fn sqrt(a: Expr) -> Self { Expr::Sqrt(Box::new(a)) }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Const(v) => write!(f, "{}", v),
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Tan(a) => write!(f, "tan({})", a),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Sqrt(a) => write!(f, "sqrt({})", a),
            Expr::Abs(a) => write!(f, "abs({})", a),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// EVALUATION
// ═══════════════════════════════════════════════════════════════════

/// Evaluate expression with variable bindings.
pub fn eval(expr: &Expr, vars: &std::collections::HashMap<String, f64>) -> f64 {
    match expr {
        Expr::Const(v) => *v,
        Expr::Var(name) => *vars.get(name).unwrap_or(&0.0),
        Expr::Add(a, b) => eval(a, vars) + eval(b, vars),
        Expr::Sub(a, b) => eval(a, vars) - eval(b, vars),
        Expr::Mul(a, b) => eval(a, vars) * eval(b, vars),
        Expr::Div(a, b) => eval(a, vars) / eval(b, vars),
        Expr::Pow(a, b) => eval(a, vars).powf(eval(b, vars)),
        Expr::Neg(a) => -eval(a, vars),
        Expr::Sin(a) => eval(a, vars).sin(),
        Expr::Cos(a) => eval(a, vars).cos(),
        Expr::Tan(a) => eval(a, vars).tan(),
        Expr::Exp(a) => eval(a, vars).exp(),
        Expr::Ln(a) => eval(a, vars).ln(),
        Expr::Sqrt(a) => eval(a, vars).sqrt(),
        Expr::Abs(a) => eval(a, vars).abs(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// SYMBOLIC DIFFERENTIATION
// ═══════════════════════════════════════════════════════════════════

/// Symbolic derivative of expr with respect to variable.
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Const(_) => Expr::Const(0.0),
        Expr::Var(name) => if name == var { Expr::Const(1.0) } else { Expr::Const(0.0) },
        Expr::Add(a, b) => Expr::add(differentiate(a, var), differentiate(b, var)),
        Expr::Sub(a, b) => Expr::sub(differentiate(a, var), differentiate(b, var)),
        Expr::Mul(a, b) => {
            // Product rule: d(fg) = f'g + fg'
            Expr::add(
                Expr::mul(differentiate(a, var), *b.clone()),
                Expr::mul(*a.clone(), differentiate(b, var)),
            )
        }
        Expr::Div(a, b) => {
            // Quotient rule: d(f/g) = (f'g - fg') / g²
            Expr::div(
                Expr::sub(
                    Expr::mul(differentiate(a, var), *b.clone()),
                    Expr::mul(*a.clone(), differentiate(b, var)),
                ),
                Expr::pow(*b.clone(), Expr::Const(2.0)),
            )
        }
        Expr::Pow(base, exp) => {
            // Power rule: d(f^g) = f^g * (g' ln(f) + g f'/f)
            if let Expr::Const(n) = exp.as_ref() {
                // Simple case: d(f^n) = n f^{n-1} f'
                Expr::mul(
                    Expr::mul(Expr::Const(*n), Expr::pow(*base.clone(), Expr::Const(n - 1.0))),
                    differentiate(base, var),
                )
            } else {
                // General: f^g (g' ln(f) + g f'/f)
                Expr::mul(
                    Expr::pow(*base.clone(), *exp.clone()),
                    Expr::add(
                        Expr::mul(differentiate(exp, var), Expr::ln(*base.clone())),
                        Expr::mul(*exp.clone(), Expr::div(differentiate(base, var), *base.clone())),
                    ),
                )
            }
        }
        Expr::Neg(a) => Expr::neg(differentiate(a, var)),
        Expr::Sin(a) => Expr::mul(Expr::cos(*a.clone()), differentiate(a, var)),
        Expr::Cos(a) => Expr::neg(Expr::mul(Expr::sin(*a.clone()), differentiate(a, var))),
        Expr::Tan(a) => {
            // d(tan(f)) = f'/cos²(f)
            Expr::div(differentiate(a, var), Expr::pow(Expr::cos(*a.clone()), Expr::Const(2.0)))
        }
        Expr::Exp(a) => Expr::mul(Expr::exp(*a.clone()), differentiate(a, var)),
        Expr::Ln(a) => Expr::div(differentiate(a, var), *a.clone()),
        Expr::Sqrt(a) => {
            // d(√f) = f'/(2√f)
            Expr::div(differentiate(a, var), Expr::mul(Expr::Const(2.0), Expr::sqrt(*a.clone())))
        }
        Expr::Abs(_) => Expr::Const(0.0), // not differentiable; return 0 as placeholder
    }
}

// ═══════════════════════════════════════════════════════════════════
// SIMPLIFICATION
// ═══════════════════════════════════════════════════════════════════

/// Simplify expression (basic algebraic rules).
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (Expr::Const(0.0), _) => sb,
                (_, Expr::Const(0.0)) => sa,
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                _ => Expr::add(sa, sb),
            }
        }
        Expr::Sub(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (_, Expr::Const(0.0)) => sa,
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                _ if sa == sb => Expr::Const(0.0),
                _ => Expr::sub(sa, sb),
            }
        }
        Expr::Mul(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                (Expr::Const(c), _) if (*c - 1.0).abs() < 1e-15 => sb,
                (_, Expr::Const(c)) if (*c - 1.0).abs() < 1e-15 => sa,
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                _ => Expr::mul(sa, sb),
            }
        }
        Expr::Div(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (Expr::Const(0.0), _) => Expr::Const(0.0),
                (_, Expr::Const(c)) if (*c - 1.0).abs() < 1e-15 => sa,
                (Expr::Const(a), Expr::Const(b)) if b.abs() > 1e-15 => Expr::Const(a / b),
                _ if sa == sb => Expr::Const(1.0),
                _ => Expr::div(sa, sb),
            }
        }
        Expr::Pow(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (_, Expr::Const(c)) if (*c - 0.0).abs() < 1e-15 => Expr::Const(1.0),
                (_, Expr::Const(c)) if (*c - 1.0).abs() < 1e-15 => sa,
                _ => Expr::pow(sa, sb),
            }
        }
        Expr::Neg(a) => {
            let sa = simplify(a);
            match &sa {
                Expr::Const(c) => Expr::Const(-c),
                Expr::Neg(inner) => *inner.clone(),
                _ => Expr::neg(sa),
            }
        }
        Expr::Sin(a) => { let sa = simplify(a); if let Expr::Const(v) = sa { Expr::Const(v.sin()) } else { Expr::sin(sa) } }
        Expr::Cos(a) => { let sa = simplify(a); if let Expr::Const(v) = sa { Expr::Const(v.cos()) } else { Expr::cos(sa) } }
        Expr::Exp(a) => { let sa = simplify(a); if let Expr::Const(v) = sa { Expr::Const(v.exp()) } else { Expr::exp(sa) } }
        Expr::Ln(a) => { let sa = simplify(a); if let Expr::Const(v) = sa { Expr::Const(v.ln()) } else { Expr::ln(sa) } }
        Expr::Sqrt(a) => { let sa = simplify(a); if let Expr::Const(v) = sa { Expr::Const(v.sqrt()) } else { Expr::sqrt(sa) } }
        _ => expr.clone(),
    }
}

/// Repeatedly simplify until fixed point.
pub fn full_simplify(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    for _ in 0..20 {
        let simplified = simplify(&current);
        if simplified == current { break; }
        current = simplified;
    }
    current
}

// ═══════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════

/// Collect all variable names in expression.
pub fn free_variables(expr: &Expr) -> Vec<String> {
    let mut vars = std::collections::HashSet::new();
    collect_vars(expr, &mut vars);
    let mut v: Vec<String> = vars.into_iter().collect();
    v.sort();
    v
}

fn collect_vars(expr: &Expr, vars: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Var(name) => { vars.insert(name.clone()); }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_vars(a, vars); collect_vars(b, vars);
        }
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
            collect_vars(a, vars);
        }
        Expr::Const(_) => {}
    }
}

/// Substitute variable with expression.
pub fn substitute(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Var(name) if name == var => replacement.clone(),
        Expr::Add(a, b) => Expr::add(substitute(a, var, replacement), substitute(b, var, replacement)),
        Expr::Sub(a, b) => Expr::sub(substitute(a, var, replacement), substitute(b, var, replacement)),
        Expr::Mul(a, b) => Expr::mul(substitute(a, var, replacement), substitute(b, var, replacement)),
        Expr::Div(a, b) => Expr::div(substitute(a, var, replacement), substitute(b, var, replacement)),
        Expr::Pow(a, b) => Expr::pow(substitute(a, var, replacement), substitute(b, var, replacement)),
        Expr::Neg(a) => Expr::neg(substitute(a, var, replacement)),
        Expr::Sin(a) => Expr::sin(substitute(a, var, replacement)),
        Expr::Cos(a) => Expr::cos(substitute(a, var, replacement)),
        Expr::Tan(a) => Expr::tan(substitute(a, var, replacement)),
        Expr::Exp(a) => Expr::exp(substitute(a, var, replacement)),
        Expr::Ln(a) => Expr::ln(substitute(a, var, replacement)),
        Expr::Sqrt(a) => Expr::sqrt(substitute(a, var, replacement)),
        Expr::Abs(a) => Expr::Abs(Box::new(substitute(a, var, replacement))),
        _ => expr.clone(),
    }
}

/// Count nodes in expression tree.
pub fn expr_size(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            1 + expr_size(a) + expr_size(b)
        }
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
            1 + expr_size(a)
        }
    }
}

/// Depth of expression tree.
pub fn expr_depth(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            1 + expr_depth(a).max(expr_depth(b))
        }
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
            1 + expr_depth(a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_eval() {
        let expr = Expr::add(Expr::var("x"), Expr::Const(1.0));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        assert!((eval(&expr, &vars) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_differentiate_polynomial() {
        // d/dx(x²) = 2x
        let expr = Expr::pow(Expr::var("x"), Expr::Const(2.0));
        let deriv = differentiate(&expr, "x");
        let simplified = full_simplify(&deriv);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        assert!((eval(&simplified, &vars) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_differentiate_sin() {
        // d/dx(sin(x)) = cos(x)
        let expr = Expr::sin(Expr::var("x"));
        let deriv = differentiate(&expr, "x");
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        assert!((eval(&deriv, &vars) - 1.0).abs() < 1e-10); // cos(0) = 1
    }

    #[test]
    fn test_differentiate_exp() {
        // d/dx(e^x) = e^x
        let expr = Expr::exp(Expr::var("x"));
        let deriv = differentiate(&expr, "x");
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        assert!((eval(&deriv, &vars) - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_simplify() {
        // 0 + x = x
        let expr = Expr::add(Expr::Const(0.0), Expr::var("x"));
        assert_eq!(simplify(&expr), Expr::var("x"));
        // x * 1 = x
        let expr = Expr::mul(Expr::var("x"), Expr::Const(1.0));
        assert_eq!(simplify(&expr), Expr::var("x"));
        // x * 0 = 0
        let expr = Expr::mul(Expr::var("x"), Expr::Const(0.0));
        assert_eq!(simplify(&expr), Expr::Const(0.0));
    }

    #[test]
    fn test_free_variables() {
        let expr = Expr::add(Expr::var("x"), Expr::mul(Expr::var("y"), Expr::Const(2.0)));
        assert_eq!(free_variables(&expr), vec!["x", "y"]);
    }

    #[test]
    fn test_substitute() {
        let expr = Expr::add(Expr::var("x"), Expr::Const(1.0));
        let sub = substitute(&expr, "x", &Expr::Const(5.0));
        assert_eq!(eval(&sub, &HashMap::new()), 6.0);
    }

    #[test]
    fn test_display() {
        let expr = Expr::add(Expr::var("x"), Expr::Const(1.0));
        assert_eq!(format!("{}", expr), "(x + 1)");
    }

    #[test]
    fn test_expr_size_depth() {
        let expr = Expr::add(Expr::var("x"), Expr::mul(Expr::var("y"), Expr::Const(2.0)));
        assert_eq!(expr_size(&expr), 5); // add, x, mul, y, 2
        assert_eq!(expr_depth(&expr), 2);
    }
}
