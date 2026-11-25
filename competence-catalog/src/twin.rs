//! Model-twin admission seed — sim2real's tractable core: a controller is admitted
//! only if its recorded ticks reproduce a deterministic plant model within tol.
//! Not a physics simulator (that edge stays open, registered); a double-integrator
//! plant is the spectest, exactly as replay gates competence. Mujin's live twin is
//! the peer; the delta is that here the twin's verdict is a gate, not a demo.

/// A deterministic double-integrator plant (position/velocity), fixed dt.
#[derive(Clone, Copy, Debug)]
pub struct Plant {
    pub dt: f64,
}

/// Simulate the plant under an action stream; returns the position trace.
pub fn simulate(p: &Plant, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += a * p.dt;
            x += v * p.dt;
            x
        })
        .collect()
}

/// Max abs divergence between a recorded trace and the twin's trace; None if lengths differ.
pub fn divergence(recorded: &[f64], sim: &[f64]) -> Option<f64> {
    if recorded.len() != sim.len() {
        return Option::None;
    }
    recorded
        .iter()
        .zip(sim.iter())
        .map(|(r, s)| (r - s).abs())
        .fold(Option::Some(0.0f64), |acc, d| match acc {
            Option::None => Option::None,
            Option::Some(m) => {
                if d.is_finite() { Option::Some(m.max(d)) } else { Option::None }
            }
        })
}

/// Gate: admit the controller iff its recorded trace reproduces the twin within tol.
pub fn admit_controller(p: &Plant, acts: &[f64], recorded: &[f64], tol: f64) -> bool {
    if !tol.is_finite() || tol < 0.0 || !p.dt.is_finite() || p.dt <= 0.0 {
        return false;
    }
    match divergence(recorded, &simulate(p, acts)) {
        Option::Some(d) => d <= tol,
        Option::None => false,
    }
}

/// Spring-damper contact force while the plant presses past a surface at `wall`:
/// F = k*penetration + c*approach-velocity. A force-model seed toward ISO/TS 15066
/// (collaborative force/pressure limits); full pressure/area mapping stays open.
pub fn contact_force(p: &Plant, acts: &[f64], wall: f64, k: f64, c: f64) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += a * p.dt;
            x += v * p.dt;
            let pen = (x - wall).max(0.0);
            if pen > 0.0 { k * pen + c * v.max(0.0) } else { 0.0 }
        })
        .collect()
}

/// Force-limit gate: every contact sample finite and <= limit (Newtons).
pub fn force_limit_ok(forces: &[f64], limit_n: f64) -> bool {
    limit_n.is_finite() && limit_n >= 0.0 && forces.iter().all(|f| f.is_finite() && *f <= limit_n)
}
