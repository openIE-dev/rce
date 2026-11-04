//! Partial differential equations: finite difference methods,
//! heat equation, wave equation, Laplace/Poisson, advection,
//! Crank-Nicolson, spectral methods basics.

// ═══════════════════════════════════════════════════════════════════
// 1D HEAT EQUATION: ∂u/∂t = α ∂²u/∂x²
// ═══════════════════════════════════════════════════════════════════

/// Solve 1D heat equation with explicit (FTCS) method.
/// u0: initial condition (n_x points), alpha: diffusivity,
/// dx, dt: space/time steps, n_t: time steps.
/// Boundary: u[0] = u[n-1] = 0 (Dirichlet).
pub fn heat_1d_explicit(u0: &[f64], alpha: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let r = alpha * dt / (dx * dx);
    let mut result = vec![u0.to_vec()];
    let mut u = u0.to_vec();
    for _ in 0..n_t {
        let mut u_new = vec![0.0; nx];
        for i in 1..nx - 1 {
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
        u = u_new;
        result.push(u.clone());
    }
    result
}

/// Solve 1D heat equation with Crank-Nicolson (implicit).
pub fn heat_1d_crank_nicolson(u0: &[f64], alpha: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let r = alpha * dt / (2.0 * dx * dx);
    let mut result = vec![u0.to_vec()];
    let mut u = u0.to_vec();
    for _ in 0..n_t {
        // Build tridiagonal system: (I - r*D²)u_new = (I + r*D²)u_old
        let mut rhs = vec![0.0; nx];
        for i in 1..nx - 1 {
            rhs[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
        // Solve tridiagonal: -r u_{i-1} + (1+2r) u_i - r u_{i+1} = rhs_i
        let a = vec![-r; nx]; // sub-diagonal
        let b: Vec<f64> = (0..nx).map(|i| if i == 0 || i == nx - 1 { 1.0 } else { 1.0 + 2.0 * r }).collect();
        let c = vec![-r; nx]; // super-diagonal
        u = tridiag_solve(&a, &b, &c, &rhs);
        u[0] = 0.0; u[nx - 1] = 0.0;
        result.push(u.clone());
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// 1D WAVE EQUATION: ∂²u/∂t² = c² ∂²u/∂x²
// ═══════════════════════════════════════════════════════════════════

/// Solve 1D wave equation with explicit method.
/// u0: initial displacement, v0: initial velocity.
pub fn wave_1d_explicit(u0: &[f64], v0: &[f64], c: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let r2 = (c * dt / dx).powi(2);
    let mut result = vec![u0.to_vec()];
    // First step using initial velocity
    let mut u_prev = u0.to_vec();
    let mut u_curr = vec![0.0; nx];
    for i in 1..nx - 1 {
        u_curr[i] = u0[i] + v0[i] * dt + 0.5 * r2 * (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]);
    }
    result.push(u_curr.clone());
    for _ in 1..n_t {
        let mut u_next = vec![0.0; nx];
        for i in 1..nx - 1 {
            u_next[i] = 2.0 * u_curr[i] - u_prev[i] + r2 * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
        }
        u_prev = u_curr;
        u_curr = u_next;
        result.push(u_curr.clone());
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// 2D LAPLACE/POISSON: ∇²u = f
// ═══════════════════════════════════════════════════════════════════

/// Solve 2D Laplace equation (f=0) on unit square with Gauss-Seidel.
/// boundary: value on edges, nx×ny grid.
pub fn laplace_2d(nx: usize, ny: usize, boundary_top: f64, boundary_bottom: f64,
                  boundary_left: f64, boundary_right: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let mut u = vec![0.0; nx * ny];
    // Set boundaries
    for j in 0..ny { u[j] = boundary_bottom; u[(nx - 1) * ny + j] = boundary_top; }
    for i in 0..nx { u[i * ny] = boundary_left; u[i * ny + ny - 1] = boundary_right; }
    for _ in 0..max_iter {
        let mut max_diff = 0.0f64;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let new_val = 0.25 * (u[(i + 1) * ny + j] + u[(i - 1) * ny + j] + u[i * ny + j + 1] + u[i * ny + j - 1]);
                let diff = (new_val - u[i * ny + j]).abs();
                max_diff = max_diff.max(diff);
                u[i * ny + j] = new_val;
            }
        }
        if max_diff < tol { break; }
    }
    u
}

/// Solve 2D Poisson equation ∇²u = f with Gauss-Seidel.
pub fn poisson_2d(f: &[f64], nx: usize, ny: usize, dx: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let mut u = vec![0.0; nx * ny];
    let dx2 = dx * dx;
    for _ in 0..max_iter {
        let mut max_diff = 0.0f64;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let new_val = 0.25 * (u[(i + 1) * ny + j] + u[(i - 1) * ny + j] + u[i * ny + j + 1] + u[i * ny + j - 1] - dx2 * f[i * ny + j]);
                let diff = (new_val - u[i * ny + j]).abs();
                max_diff = max_diff.max(diff);
                u[i * ny + j] = new_val;
            }
        }
        if max_diff < tol { break; }
    }
    u
}

// ═══════════════════════════════════════════════════════════════════
// 1D ADVECTION: ∂u/∂t + c ∂u/∂x = 0
// ═══════════════════════════════════════════════════════════════════

/// Upwind scheme for 1D advection (c > 0).
pub fn advection_1d_upwind(u0: &[f64], c: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let cfl = c * dt / dx;
    let mut result = vec![u0.to_vec()];
    let mut u = u0.to_vec();
    for _ in 0..n_t {
        let mut u_new = u.clone();
        if c > 0.0 {
            for i in 1..nx { u_new[i] = u[i] - cfl * (u[i] - u[i - 1]); }
        } else {
            for i in 0..nx - 1 { u_new[i] = u[i] - cfl * (u[i + 1] - u[i]); }
        }
        u = u_new;
        result.push(u.clone());
    }
    result
}

/// Lax-Wendroff scheme for 1D advection.
pub fn advection_1d_lax_wendroff(u0: &[f64], c: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let cfl = c * dt / dx;
    let cfl2 = cfl * cfl;
    let mut result = vec![u0.to_vec()];
    let mut u = u0.to_vec();
    for _ in 0..n_t {
        let mut u_new = u.clone();
        for i in 1..nx - 1 {
            u_new[i] = u[i] - 0.5 * cfl * (u[i + 1] - u[i - 1]) + 0.5 * cfl2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
        u = u_new;
        result.push(u.clone());
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// BURGERS EQUATION: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
// ═══════════════════════════════════════════════════════════════════

/// Solve 1D viscous Burgers equation.
pub fn burgers_1d(u0: &[f64], nu: f64, dx: f64, dt: f64, n_t: usize) -> Vec<Vec<f64>> {
    let nx = u0.len();
    let mut result = vec![u0.to_vec()];
    let mut u = u0.to_vec();
    for _ in 0..n_t {
        let mut u_new = u.clone();
        for i in 1..nx - 1 {
            let advection = -u[i] * (u[i] - u[i - 1]) / dx;
            let diffusion = nu * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
            u_new[i] = u[i] + dt * (advection + diffusion);
        }
        u = u_new;
        result.push(u.clone());
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// TRIDIAGONAL SOLVER (Thomas algorithm)
// ═══════════════════════════════════════════════════════════════════

/// Solve tridiagonal system Ax = d where A has diagonals (a, b, c).
pub fn tridiag_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if m.abs() < 1e-15 { cp[i] = 0.0; dp[i] = 0.0; continue; }
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() { x[i] = dp[i] - cp[i] * x[i + 1]; }
    x
}

/// SOR (Successive Over-Relaxation) for Ax = b.
pub fn sor_solve(a: &[f64], b: &[f64], n: usize, omega: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for _ in 0..max_iter {
        let mut max_diff = 0.0f64;
        for i in 0..n {
            let mut sigma = b[i];
            for j in 0..n { if j != i { sigma -= a[i * n + j] * x[j]; } }
            let new_xi = (1.0 - omega) * x[i] + omega * sigma / a[i * n + i];
            max_diff = max_diff.max((new_xi - x[i]).abs());
            x[i] = new_xi;
        }
        if max_diff < tol { break; }
    }
    x
}

// ═══════════════════════════════════════════════════════════════════
// GRID GENERATION
// ═══════════════════════════════════════════════════════════════════

/// Uniform 1D grid.
pub fn uniform_grid_1d(x_min: f64, x_max: f64, n: usize) -> Vec<f64> {
    let dx = (x_max - x_min) / (n - 1) as f64;
    (0..n).map(|i| x_min + i as f64 * dx).collect()
}

/// Chebyshev grid (clustered at boundaries).
pub fn chebyshev_grid(x_min: f64, x_max: f64, n: usize) -> Vec<f64> {
    let mid = (x_min + x_max) / 2.0;
    let half = (x_max - x_min) / 2.0;
    (0..n).map(|i| mid - half * (std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect()
}

/// 2D meshgrid from 1D grids (returns x_grid, y_grid each nx*ny).
pub fn meshgrid(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nx = x.len();
    let ny = y.len();
    let mut xg = vec![0.0; nx * ny];
    let mut yg = vec![0.0; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            xg[i * ny + j] = x[i];
            yg[i * ny + j] = y[j];
        }
    }
    (xg, yg)
}

/// CFL condition check: returns max stable dt.
pub fn cfl_max_dt(c: f64, dx: f64, cfl_limit: f64) -> f64 {
    cfl_limit * dx / c.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_1d_explicit() {
        // Initial: delta function in middle
        let nx = 21;
        let mut u0 = vec![0.0; nx];
        u0[10] = 1.0;
        let result = heat_1d_explicit(&u0, 0.01, 0.1, 0.01, 100);
        assert_eq!(result.len(), 101);
        // Heat should spread: max should decrease
        assert!(result[100].iter().copied().fold(0.0, f64::max) < 1.0);
    }

    #[test]
    fn test_wave_1d() {
        let nx = 51;
        let mut u0 = vec![0.0; nx];
        for i in 20..30 { u0[i] = ((i - 20) as f64 * std::f64::consts::PI / 10.0).sin(); }
        let v0 = vec![0.0; nx];
        let result = wave_1d_explicit(&u0, &v0, 1.0, 0.1, 0.05, 50);
        assert_eq!(result.len(), 51);
    }

    #[test]
    fn test_laplace_2d() {
        let u = laplace_2d(11, 11, 100.0, 0.0, 0.0, 0.0, 10000, 1e-6);
        // Interior should be between 0 and 100
        assert!(u[5 * 11 + 5] > 0.0 && u[5 * 11 + 5] < 100.0);
    }

    #[test]
    fn test_tridiag_solve() {
        // Simple: [2,-1,0; -1,2,-1; 0,-1,2] x = [1,0,1]
        let a = vec![0.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, 0.0];
        let d = vec![1.0, 0.0, 1.0];
        let x = tridiag_solve(&a, &b, &c, &d);
        assert_eq!(x.len(), 3);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
        assert!((x[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_grid() {
        let g = uniform_grid_1d(0.0, 1.0, 5);
        assert_eq!(g, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_chebyshev_grid() {
        let g = chebyshev_grid(-1.0, 1.0, 5);
        assert_eq!(g.len(), 5);
        assert!((g[0] - (-1.0)).abs() < 1e-10);
        assert!((g[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_meshgrid() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0, 2.0];
        let (xg, yg) = meshgrid(&x, &y);
        assert_eq!(xg.len(), 6);
        assert_eq!(yg.len(), 6);
    }

    #[test]
    fn test_cfl() {
        let dt = cfl_max_dt(1.0, 0.1, 0.5);
        assert!((dt - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_crank_nicolson() {
        let nx = 21;
        let mut u0 = vec![0.0; nx];
        u0[10] = 1.0;
        let result = heat_1d_crank_nicolson(&u0, 0.01, 0.1, 0.05, 50);
        assert!(result.len() > 1);
    }
}
