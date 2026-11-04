//! Automatic differentiation: forward-mode dual numbers,
//! reverse-mode tape, symbolic derivatives, gradient computation.

use std::ops::{Add, Sub, Mul, Div, Neg};

// ═══════════════════════════════════════════════════════════════════
// DUAL NUMBERS (forward-mode AD)
// ═══════════════════════════════════════════════════════════════════

/// Dual number: value + ε·derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub val: f64,
    pub deriv: f64,
}

impl Dual {
    pub fn new(val: f64, deriv: f64) -> Self { Self { val, deriv } }
    pub fn constant(val: f64) -> Self { Self { val, deriv: 0.0 } }
    pub fn variable(val: f64) -> Self { Self { val, deriv: 1.0 } }

    pub fn sin(self) -> Self { Dual { val: self.val.sin(), deriv: self.deriv * self.val.cos() } }
    pub fn cos(self) -> Self { Dual { val: self.val.cos(), deriv: -self.deriv * self.val.sin() } }
    pub fn tan(self) -> Self { let c = self.val.cos(); Dual { val: self.val.tan(), deriv: self.deriv / (c * c) } }
    pub fn exp(self) -> Self { let e = self.val.exp(); Dual { val: e, deriv: self.deriv * e } }
    pub fn ln(self) -> Self { Dual { val: self.val.ln(), deriv: self.deriv / self.val } }
    pub fn sqrt(self) -> Self { let s = self.val.sqrt(); Dual { val: s, deriv: self.deriv / (2.0 * s) } }
    pub fn abs(self) -> Self { Dual { val: self.val.abs(), deriv: self.deriv * self.val.signum() } }
    pub fn powf(self, n: f64) -> Self { Dual { val: self.val.powf(n), deriv: self.deriv * n * self.val.powf(n - 1.0) } }
    pub fn powi(self, n: i32) -> Self { self.powf(n as f64) }
    pub fn tanh(self) -> Self { let t = self.val.tanh(); Dual { val: t, deriv: self.deriv * (1.0 - t * t) } }
    pub fn sinh(self) -> Self { Dual { val: self.val.sinh(), deriv: self.deriv * self.val.cosh() } }
    pub fn cosh(self) -> Self { Dual { val: self.val.cosh(), deriv: self.deriv * self.val.sinh() } }
    pub fn asin(self) -> Self { Dual { val: self.val.asin(), deriv: self.deriv / (1.0 - self.val * self.val).sqrt() } }
    pub fn acos(self) -> Self { Dual { val: self.val.acos(), deriv: -self.deriv / (1.0 - self.val * self.val).sqrt() } }
    pub fn atan(self) -> Self { Dual { val: self.val.atan(), deriv: self.deriv / (1.0 + self.val * self.val) } }
    pub fn log2(self) -> Self { Dual { val: self.val.log2(), deriv: self.deriv / (self.val * 2.0f64.ln()) } }
    pub fn log10(self) -> Self { Dual { val: self.val.log10(), deriv: self.deriv / (self.val * 10.0f64.ln()) } }
    pub fn sigmoid(self) -> Self { let s = 1.0 / (1.0 + (-self.val).exp()); Dual { val: s, deriv: self.deriv * s * (1.0 - s) } }
    pub fn relu(self) -> Self { if self.val > 0.0 { self } else { Dual { val: 0.0, deriv: 0.0 } } }
    pub fn max(self, other: Dual) -> Self { if self.val >= other.val { self } else { other } }
    pub fn min(self, other: Dual) -> Self { if self.val <= other.val { self } else { other } }
}

impl Add for Dual { type Output = Self; fn add(self, rhs: Self) -> Self { Dual { val: self.val + rhs.val, deriv: self.deriv + rhs.deriv } } }
impl Sub for Dual { type Output = Self; fn sub(self, rhs: Self) -> Self { Dual { val: self.val - rhs.val, deriv: self.deriv - rhs.deriv } } }
impl Mul for Dual { type Output = Self; fn mul(self, rhs: Self) -> Self { Dual { val: self.val * rhs.val, deriv: self.val * rhs.deriv + self.deriv * rhs.val } } }
impl Div for Dual { type Output = Self; fn div(self, rhs: Self) -> Self { Dual { val: self.val / rhs.val, deriv: (self.deriv * rhs.val - self.val * rhs.deriv) / (rhs.val * rhs.val) } } }
impl Neg for Dual { type Output = Self; fn neg(self) -> Self { Dual { val: -self.val, deriv: -self.deriv } } }
impl Add<f64> for Dual { type Output = Self; fn add(self, rhs: f64) -> Self { Dual { val: self.val + rhs, deriv: self.deriv } } }
impl Mul<f64> for Dual { type Output = Self; fn mul(self, rhs: f64) -> Self { Dual { val: self.val * rhs, deriv: self.deriv * rhs } } }

// ═══════════════════════════════════════════════════════════════════
// FORWARD-MODE DERIVATIVE HELPERS
// ═══════════════════════════════════════════════════════════════════

/// Compute derivative of f at x using forward-mode AD.
pub fn derivative(f: impl Fn(Dual) -> Dual, x: f64) -> f64 {
    f(Dual::variable(x)).deriv
}

/// Compute value and derivative simultaneously.
pub fn value_and_derivative(f: impl Fn(Dual) -> Dual, x: f64) -> (f64, f64) {
    let r = f(Dual::variable(x));
    (r.val, r.deriv)
}

/// Second derivative using nested duals (conceptually).
/// Computed via central difference of AD derivative.
pub fn second_derivative(f: impl Fn(Dual) -> Dual, x: f64) -> f64 {
    let h = 1e-5;
    let d1 = derivative(&f, x + h);
    let d2 = derivative(&f, x - h);
    (d1 - d2) / (2.0 * h)
}

/// Gradient of f: R^n → R using forward-mode AD.
pub fn ad_gradient(f: impl Fn(&[Dual]) -> Dual, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| {
        let inputs: Vec<Dual> = x.iter().enumerate()
            .map(|(j, &v)| if j == i { Dual::variable(v) } else { Dual::constant(v) })
            .collect();
        f(&inputs).deriv
    }).collect()
}

/// Jacobian of f: R^n → R^m using forward-mode AD.
pub fn ad_jacobian(f: impl Fn(&[Dual]) -> Vec<Dual>, x: &[f64], m: usize) -> Vec<f64> {
    let n = x.len();
    let mut jac = vec![0.0; m * n];
    for j in 0..n {
        let inputs: Vec<Dual> = x.iter().enumerate()
            .map(|(k, &v)| if k == j { Dual::variable(v) } else { Dual::constant(v) })
            .collect();
        let outputs = f(&inputs);
        for i in 0..m { jac[i * n + j] = outputs[i].deriv; }
    }
    jac
}

/// Directional derivative: ∇f(x) · v.
pub fn directional_derivative(f: impl Fn(&[Dual]) -> Dual, x: &[f64], direction: &[f64]) -> f64 {
    let inputs: Vec<Dual> = x.iter().zip(direction)
        .map(|(&v, &d)| Dual::new(v, d))
        .collect();
    f(&inputs).deriv
}

// ═══════════════════════════════════════════════════════════════════
// REVERSE-MODE AD (simple tape)
// ═══════════════════════════════════════════════════════════════════

/// Node in reverse-mode computation graph.
#[derive(Clone, Debug)]
struct TapeNode {
    val: f64,
    deps: Vec<(usize, f64)>, // (parent_index, local_gradient)
}

/// Reverse-mode AD tape.
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    pub fn new() -> Self { Self { nodes: Vec::new() } }

    /// Create a variable (leaf node).
    pub fn var(&mut self, val: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val, deps: vec![] });
        idx
    }

    /// Create constant.
    pub fn constant(&mut self, val: f64) -> usize { self.var(val) }

    /// Get value of node.
    pub fn value(&self, idx: usize) -> f64 { self.nodes[idx].val }

    /// Add two nodes.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let val = self.nodes[a].val + self.nodes[b].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val, deps: vec![(a, 1.0), (b, 1.0)] });
        idx
    }

    /// Multiply two nodes.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let va = self.nodes[a].val;
        let vb = self.nodes[b].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: va * vb, deps: vec![(a, vb), (b, va)] });
        idx
    }

    /// Subtract.
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let val = self.nodes[a].val - self.nodes[b].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val, deps: vec![(a, 1.0), (b, -1.0)] });
        idx
    }

    /// Divide.
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let va = self.nodes[a].val;
        let vb = self.nodes[b].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: va / vb, deps: vec![(a, 1.0 / vb), (b, -va / (vb * vb))] });
        idx
    }

    /// Sin.
    pub fn sin(&mut self, a: usize) -> usize {
        let v = self.nodes[a].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: v.sin(), deps: vec![(a, v.cos())] });
        idx
    }

    /// Cos.
    pub fn cos(&mut self, a: usize) -> usize {
        let v = self.nodes[a].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: v.cos(), deps: vec![(a, -v.sin())] });
        idx
    }

    /// Exp.
    pub fn exp(&mut self, a: usize) -> usize {
        let e = self.nodes[a].val.exp();
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: e, deps: vec![(a, e)] });
        idx
    }

    /// Ln.
    pub fn ln(&mut self, a: usize) -> usize {
        let v = self.nodes[a].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: v.ln(), deps: vec![(a, 1.0 / v)] });
        idx
    }

    /// Power: a^n.
    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let v = self.nodes[a].val;
        let idx = self.nodes.len();
        self.nodes.push(TapeNode { val: v.powf(n), deps: vec![(a, n * v.powf(n - 1.0))] });
        idx
    }

    /// Backward pass: compute gradients w.r.t. all nodes.
    pub fn backward(&self, output: usize) -> Vec<f64> {
        let mut grads = vec![0.0; self.nodes.len()];
        grads[output] = 1.0;
        for i in (0..=output).rev() {
            let grad = grads[i];
            if (grad as f64).abs() < 1e-30 { continue; }
            for &(parent, local_grad) in &self.nodes[i].deps {
                grads[parent] += grad * local_grad;
            }
        }
        grads
    }
}

/// Convenience: compute gradient of scalar function using reverse-mode.
pub fn reverse_gradient(
    build_fn: impl Fn(&mut Tape, &[usize]) -> usize,
    x: &[f64],
) -> Vec<f64> {
    let mut tape = Tape::new();
    let vars: Vec<usize> = x.iter().map(|&v| tape.var(v)).collect();
    let output = build_fn(&mut tape, &vars);
    let grads = tape.backward(output);
    vars.iter().map(|&i| grads[i]).collect()
}

// ═══════════════════════════════════════════════════════════════════
// FINITE DIFFERENCE VERIFICATION
// ═══════════════════════════════════════════════════════════════════

/// Check AD gradient against finite differences.
pub fn check_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], ad_grad: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let mut errors = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (f(&xp) - f(&xm)) / (2.0 * eps);
        errors.push((ad_grad[i] - fd).abs());
    }
    errors
}

/// Numerical gradient (central differences).
pub fn numerical_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        (f(&xp) - f(&xm)) / (2.0 * eps)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_arithmetic() {
        let x = Dual::variable(3.0);
        let y = Dual::constant(2.0);
        let z = x * x + y; // x² + 2, d/dx = 2x = 6
        assert!((z.val - 11.0).abs() < 1e-10);
        assert!((z.deriv - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_sin() {
        let x = Dual::variable(0.0);
        let y = x.sin(); // sin(0) = 0, cos(0) = 1
        assert!((y.val - 0.0).abs() < 1e-10);
        assert!((y.deriv - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_exp() {
        let x = Dual::variable(1.0);
        let y = x.exp(); // e^1 = e, d/dx = e
        assert!((y.val - std::f64::consts::E).abs() < 1e-10);
        assert!((y.deriv - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_derivative() {
        // d/dx(x³) at x=2 = 12
        let d = derivative(|x| x.powi(3), 2.0);
        assert!((d - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_ad_gradient() {
        // f(x,y) = x²y, ∂f/∂x = 2xy, ∂f/∂y = x²
        let grad = ad_gradient(|x| x[0] * x[0] * x[1], &[3.0, 4.0]);
        assert!((grad[0] - 24.0).abs() < 1e-10); // 2*3*4
        assert!((grad[1] - 9.0).abs() < 1e-10);  // 3²
    }

    #[test]
    fn test_reverse_mode() {
        // f(x,y) = x*y + sin(x)
        let grad = reverse_gradient(|tape, vars| {
            let xy = tape.mul(vars[0], vars[1]);
            let sinx = tape.sin(vars[0]);
            tape.add(xy, sinx)
        }, &[1.0, 2.0]);
        // ∂f/∂x = y + cos(x) = 2 + cos(1)
        assert!((grad[0] - (2.0 + 1.0f64.cos())).abs() < 1e-10);
        // ∂f/∂y = x = 1
        assert!((grad[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_second_derivative() {
        // d²/dx²(x³) at x=2 = 12
        let d2 = second_derivative(|x| x.powi(3), 2.0);
        assert!((d2 - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_check_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = vec![3.0, 4.0];
        let ad = ad_gradient(|d| d[0] * d[0] + d[1] * d[1], &x);
        let errors = check_gradient(f, &x, &ad, 1e-5);
        assert!(errors.iter().all(|&e| e < 1e-4));
    }

    #[test]
    fn test_dual_chain() {
        // d/dx[sin(x²)] at x=1 = 2x·cos(x²) = 2·cos(1)
        let d = derivative(|x| (x * x).sin(), 1.0);
        assert!((d - 2.0 * 1.0f64.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = Dual::variable(0.0);
        let s = x.sigmoid();
        assert!((s.val - 0.5).abs() < 1e-10);
        assert!((s.deriv - 0.25).abs() < 1e-10); // σ'(0) = σ(0)(1-σ(0)) = 0.25
    }
}
