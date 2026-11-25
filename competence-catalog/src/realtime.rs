//! Real-time gate — deadline- and memory-gated gear selection for control loops.
//!
//! Physical-AI fan-out (rce-physical-ai-bottlenecks.md): robot control runs at
//! 50–1000 Hz while VLA planners infer slower than the loop; ROS 2 on Linux is
//! not real-time; Figure/Tesla are memory-bound. Parity seed with mixed-criticality
//! RTOS scheduling (QNX/seL4) and async action-chunk execution (PI/FASTER):
//! choose the highest-value gear whose WORST-CASE latency and memory fit the
//! loop, else refuse (fail-safe) — never silently miss a deadline.
//!
//! Gear figures here are Modelled seeds, not measurements.

use crate::value::value;

/// A control loop's hard budget.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    /// Loop period, microseconds (1000 Hz = 1000).
    pub period_us: u64,
    /// Worst-case latency allowed for one act, microseconds (<= period).
    pub deadline_us: u64,
    /// Memory available to the act, kibibytes.
    pub memory_kib: u64,
}

/// A candidate execution gear with worst-case (not average) figures.
#[derive(Clone, Debug)]
pub struct Gear {
    pub name: &'static str,
    /// Worst-case latency per act, µs (includes compile spikes for JIT).
    pub wcet_us: u64,
    /// Resident memory, KiB.
    pub memory_kib: u64,
    /// Benefit weight per act (boundary input, as in value.rs).
    pub benefit: f64,
    /// Energy per act, µJ.
    pub joules: f64,
}

/// The gate's verdict: admit the best fitting gear, or refuse fail-safe.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Admit { gear: &'static str, value: f64 },
    Refuse { reason: String },
}

/// The modelled control gears: G0 lookup, hot JIT, cold JIT (compile spike),
/// and the proposer. The proposer never fits a fast loop — by construction.
pub fn control_gears() -> Vec<Gear> {
    vec![
        Gear { name: "g0-lookup", wcet_us: 5, memory_kib: 64, benefit: 1.0, joules: 1.0 },
        Gear { name: "g2-jit-hot", wcet_us: 50, memory_kib: 512, benefit: 2.0, joules: 20.0 },
        Gear { name: "g2-jit-cold", wcet_us: 20_000, memory_kib: 2048, benefit: 2.0, joules: 20.0 },
        Gear { name: "g4-proposer", wcet_us: 2_000_000, memory_kib: 1_048_576, benefit: 8.0, joules: 5_000_000.0 },
    ]
}

/// Select the highest benefit/joule gear whose WCET and memory fit the loop.
/// Ties break by name (deterministic). Refuse if the budget is incoherent or no
/// gear fits — a missed deadline is a refusal, never a silent overrun.
pub fn select(lp: &Loop, gears: &[Gear]) -> Decision {
    if lp.period_us == 0 || lp.deadline_us == 0 || lp.deadline_us > lp.period_us {
        return Decision::Refuse { reason: "deadline must be > 0 and <= period".into() };
    }
    let mut best: Option<(&Gear, f64)> = Option::None;
    for g in gears {
        let v = match value(g.benefit, g.joules) {
            Ok(v) => v,
            Err(e) => return Decision::Refuse { reason: e },
        };
        if g.wcet_us > lp.deadline_us || g.memory_kib > lp.memory_kib {
            continue;
        }
        let better = match &best {
            Option::None => true,
            Some((bg, bv)) => v > *bv || (v == *bv && g.name < bg.name),
        };
        if better {
            best = Some((g, v));
        }
    }
    match best {
        Some((g, v)) => Decision::Admit { gear: g.name, value: v },
        Option::None => {
            let min_wcet = gears.iter().map(|g| g.wcet_us).min().unwrap_or(u64::MAX);
            let min_mem = gears.iter().map(|g| g.memory_kib).min().unwrap_or(u64::MAX);
            Decision::Refuse {
                reason: format!(
                    "no gear fits: min wcet {}us vs deadline {}us; min mem {}KiB vs budget {}KiB",
                    min_wcet, lp.deadline_us, min_mem, lp.memory_kib
                ),
            }
        }
    }
}

/// A slow planner's action chunk and the fast executor's tick stream.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Waypoints the slow plan committed to.
    pub plan: Vec<f64>,
    /// The fast loop's executed ticks, in order.
    pub ticks: Vec<f64>,
}

/// Coherent iff the tick stream visits every plan waypoint in order within tol
/// (the async-chunk transition invariant; seed of PI real-time chunking / F2S).
pub fn chunk_coherent(c: &Chunk, tol: f64) -> bool {
    if !tol.is_finite() || tol < 0.0 {
        return false;
    }
    let mut i = 0;
    for t in &c.ticks {
        if i < c.plan.len() && (t - c.plan[i]).abs() <= tol {
            i += 1;
        }
    }
    i == c.plan.len()
}

/// A wall-measured gear: `wcet_us` is the worst of N wall-clock samples (Measured),
/// not an analytical bound; sub-µs samples round up to the 1 µs clock floor. Joules
/// stays the G0 floor prior until energy metering is wired per gear.
pub struct Measured {
    pub gear: Gear,
    pub provenance: crate::Provenance,
    pub samples: u64,
}

/// Time the G0 lookup handler N times on the wall clock; the worst sample becomes
/// the gear's WCET. A reading, never a fabricated 0.
pub fn measure_lookup(samples: u64) -> Measured {
    let n = samples.max(1);
    let mut worst = 0u64;
    for i in 0..n {
        let probe = format!("probe {i}");
        let t0 = std::time::Instant::now();
        let _ = crate::run_handler("upper", &probe);
        worst = worst.max(t0.elapsed().as_micros() as u64);
    }
    Measured {
        gear: Gear { name: "g0-lookup-measured", wcet_us: worst.max(1), memory_kib: 64, benefit: 1.0, joules: 1.0 },
        provenance: crate::Provenance::Measured,
        samples: n,
    }
}
