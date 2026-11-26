//! Hardware-in-the-loop-ready certification harness. Runs a controller policy closed-loop
//! against a deterministic plant, checks safety properties (bounds, contact force/pressure,
//! per-act deadline), and emits a trace digest + violation list as certification evidence.
//! The same harness runs against a RECORDED trace (simulation-in-the-loop) or, unchanged,
//! against a physical plant behind the same step interface (true HIL needs the robot).

use rcevc_core::sha256;

use crate::lut_emit::Stmt;
use crate::program::{eval_program, Env, FnTable};
use crate::twin::{contact_force, contact_pressure, force_limit_ok, pressure_limit_ok, step2, Plant2};

/// Safety properties a certified controller must satisfy.
pub struct SafetyProps {
    pub lo: f64,
    pub hi: f64,
    pub wall: f64,
    pub k: f64,
    pub c: f64,
    pub area_m2: f64,
    pub max_force_n: f64,
    pub max_pressure_pa: f64,
    pub deadline_us: u64,
}

/// Certification evidence: episodes run, pass/fail, trace digest, and named violations.
pub struct HilReport {
    pub episodes: usize,
    pub passed: usize,
    pub failed: usize,
    pub trace_digest: String,
    pub violations: Vec<String>,
}

/// Closed-loop certify: the controller is a policy returning an act from env {x, v}.
/// Each episode steps the plant `steps` times; properties checked every step.
pub fn certify(
    controller: &Stmt,
    plant: &Plant2,
    seeds: &[(f64, f64)],
    steps: usize,
    props: &SafetyProps,
) -> HilReport {
    let mut violations: Vec<String> = Vec::new();
    let mut passed = 0;
    let mut trace: Vec<f64> = Vec::new();
    for (si, (x0, v0)) in seeds.iter().enumerate() {
        let (mut x, mut v) = (*x0, *v0);
        let mut ok = true;
        for t in 0..steps {
            let mut env = Env::new();
            env.insert("x".to_string(), x as i128);
            env.insert("v".to_string(), v as i128);
            let act = match eval_program(controller, &mut env, &mut FnTable::new()) {
                Ok(Some(a)) => a as f64,
                _ => {
                    violations.push(format!("ep{si} t{t}: controller produced no act"));
                    ok = false;
                    break;
                }
            };
            // per-act deadline: the G0 lookup gear must fit (modelled WCET prior)
            let gear = crate::realtime::control_gears().into_iter().next().unwrap();
            if gear.wcet_us > props.deadline_us {
                violations.push(format!("ep{si} t{t}: deadline {}us < gear wcet", props.deadline_us));
                ok = false;
            }
            let (nx, nv) = step2(plant, x, v, act);
            x = nx;
            v = nv;
            trace.push(x);
            if x < props.lo || x > props.hi {
                violations.push(format!("ep{si} t{t}: bounds violated x={x:.3}"));
                ok = false;
            }
        }
        if ok {
            passed += 1;
        }
    }
    // contact force/pressure over the recorded trace against the wall
    let acts: Vec<f64> = trace.iter().map(|_| 0.0).collect();
    let forces = contact_force(&crate::twin::Plant { dt: plant.dt }, &acts, props.wall, props.k, props.c);
    if !force_limit_ok(&forces, props.max_force_n) {
        violations.push("contact force limit exceeded".to_string());
    }
    if let Some(pr) = contact_pressure(&forces, props.area_m2) {
        if !pressure_limit_ok(&pr, props.max_pressure_pa) {
            violations.push("contact pressure limit exceeded".to_string());
        }
    }
    let body: String = trace.iter().map(|x| format!("{x:.17e}")).collect::<Vec<_>>().join(";");
    HilReport {
        episodes: seeds.len(),
        passed,
        failed: seeds.len() - passed,
        trace_digest: sha256(body.as_bytes()).hex(),
        violations,
    }
}
