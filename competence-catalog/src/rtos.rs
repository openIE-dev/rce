//! Real-time scheduling discipline seed — the QNX/seL4 axis without the port:
//! rate-monotonic schedulability (Liu–Layland sufficient bound) and a cyclic executive
//! (time-triggered) table. Above the bound the answer is "unknown", and we refuse —
//! never an optimistic admit. The port itself remains a process edge.

/// A periodic real-time task with a worst-case execution time.
#[derive(Clone, Debug)]
pub struct Task {
    pub name: &'static str,
    pub period_us: u64,
    pub wcet_us: u64,
}

/// Liu–Layland utilization bound for n tasks: n * (2^(1/n) - 1).
pub fn rm_bound(n: usize) -> f64 {
    let n = n as f64;
    n * (2f64.powf(1.0 / n) - 1.0)
}

/// Total utilization sum(WCET/period).
pub fn utilization(ts: &[Task]) -> f64 {
    ts.iter().map(|t| t.wcet_us as f64 / t.period_us as f64).sum()
}

/// Sufficient schedulability test: every WCET fits its period and utilization <= bound.
/// Above the bound: refuse (unknown), never optimistic.
pub fn schedulable(ts: &[Task]) -> bool {
    !ts.is_empty()
        && ts.iter().all(|t| t.period_us > 0 && t.wcet_us <= t.period_us)
        && utilization(ts) <= rm_bound(ts.len())
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// Cyclic executive table: (task, release offset) within the hyperperiod, sorted by
/// offset then name. None if the hyperperiod/table would explode (> 4096 entries).
pub fn cyclic_table(ts: &[Task]) -> Option<Vec<(&'static str, u64)>> {
    if ts.is_empty() || ts.iter().any(|t| t.period_us == 0) {
        return Option::None;
    }
    let hyper = ts.iter().fold(1u64, |h, t| lcm(h, t.period_us));
    let count: u64 = ts.iter().map(|t| hyper / t.period_us).sum();
    if count > 4096 {
        return Option::None;
    }
    let mut table = Vec::new();
    for t in ts {
        let mut off = 0;
        while off < hyper {
            table.push((t.name, off));
            off += t.period_us;
        }
    }
    table.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
    Option::Some(table)
}
