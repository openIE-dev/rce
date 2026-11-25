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

/// Contact pressure = force / contact area (Pa) — the second half of the ISO/TS 15066
/// force/pressure pair. None if the area is not a positive finite number.
pub fn contact_pressure(forces: &[f64], area_m2: f64) -> Option<Vec<f64>> {
    if !area_m2.is_finite() || area_m2 <= 0.0 {
        return Option::None;
    }
    Option::Some(forces.iter().map(|f| f / area_m2).collect())
}

/// Pressure-limit gate: every sample finite and <= limit (Pascals).
pub fn pressure_limit_ok(pressures: &[f64], limit_pa: f64) -> bool {
    limit_pa.is_finite() && limit_pa >= 0.0 && pressures.iter().all(|p| p.is_finite() && *p <= limit_pa)
}

/// A richer plant for physics-fidelity sim2real: double integrator + gravity + linear drag
/// + a floor at `wall` with coefficient of restitution. Still a seed — full rigid-body /
/// multi-contact solving remains the open frontier.
#[derive(Clone, Copy, Debug)]
pub struct Plant2 {
    pub dt: f64,
    pub gravity: f64,
    pub drag: f64,
    pub wall: f64,
    pub restitution: f64,
}

/// Simulate Plant2; returns the position trace. Contact clamps to the floor and reflects
/// velocity by the restitution coefficient.
pub fn simulate2(p: &Plant2, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            let acc = a - p.gravity - p.drag * v;
            v += acc * p.dt;
            x += v * p.dt;
            if x < p.wall {
                x = p.wall;
                v = -v * p.restitution;
            }
            x
        })
        .collect()
}

/// A two-boundary contact plant (floor + ceiling) with per-contact restitution and tangential
/// friction. A multi-contact seed; full rigid-body / arbitrary contact manifolds stay open.
#[derive(Clone, Copy, Debug)]
pub struct Plant3 {
    pub dt: f64,
    pub gravity: f64,
    pub lo: f64,
    pub hi: f64,
    pub restitution: f64,
    pub friction: f64,
}

/// Simulate Plant3; the trace stays within [lo, hi]; contact reflects velocity by restitution
/// and bleeds it by friction.
pub fn simulate3(p: &Plant3, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += (a - p.gravity) * p.dt;
            x += v * p.dt;
            if x < p.lo {
                x = p.lo;
                v = -v * p.restitution;
                v *= 1.0 - p.friction;
            }
            if x > p.hi {
                x = p.hi;
                v = -v * p.restitution;
                v *= 1.0 - p.friction;
            }
            x
        })
        .collect()
}

/// Real-to-sim over the richer parameter set: pick the candidate plant whose trace best
/// explains the recording; None when no candidate is within tol.
pub fn identify2(acts: &[f64], recorded: &[f64], candidates: &[Plant2], tol: f64) -> Option<(Plant2, f64)> {
    if !tol.is_finite() || tol < 0.0 {
        return Option::None;
    }
    let mut best: Option<(Plant2, f64)> = Option::None;
    for p in candidates {
        if !p.dt.is_finite() || p.dt <= 0.0 {
            continue;
        }
        let d = match divergence(recorded, &simulate2(p, acts)) {
            Option::Some(d) => d,
            Option::None => continue,
        };
        let better = match best {
            Option::None => true,
            Option::Some((_, bd)) => d < bd,
        };
        if better {
            best = Option::Some((*p, d));
        }
    }
    match best {
        Option::Some((p, d)) if d <= tol => Option::Some((p, d)),
        _ => Option::None,
    }
}

/// Real-to-sim seed (system identification): pick the candidate dt whose simulated trace
/// best explains the recorded trace; return (dt, divergence) only if within tol.
/// None when no candidate explains the data — the sim2real gap stays visible, not hidden.
pub fn identify(acts: &[f64], recorded: &[f64], candidates: &[f64], tol: f64) -> Option<(f64, f64)> {
    if !tol.is_finite() || tol < 0.0 {
        return Option::None;
    }
    let mut best: Option<(f64, f64)> = Option::None;
    for dt in candidates {
        if !dt.is_finite() || *dt <= 0.0 {
            continue;
        }
        let d = match divergence(recorded, &simulate(&Plant { dt: *dt }, acts)) {
            Option::Some(d) => d,
            Option::None => continue,
        };
        let better = match best {
            Option::None => true,
            Option::Some((_, bd)) => d < bd,
        };
        if better {
            best = Option::Some((*dt, d));
        }
    }
    match best {
        Option::Some((dt, d)) if d <= tol => Option::Some((dt, d)),
        _ => Option::None,
    }
}
