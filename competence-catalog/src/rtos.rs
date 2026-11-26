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

/// Deterministic priority dispatch: at each tick, among released tasks (period divides
/// tick) run the highest priority (shortest period, then name). None if not schedulable.
pub fn priority_dispatch(ts: &[Task], ticks: u64) -> Option<Vec<String>> {
    if !schedulable(ts) {
        return Option::None;
    }
    let mut out = Vec::new();
    for t in 0..ticks {
        let mut released: Vec<&Task> = ts.iter().filter(|x| t % x.period_us == 0).collect();
        released.sort_by(|a, b| a.period_us.cmp(&b.period_us).then_with(|| a.name.cmp(b.name)));
        if let Some(x) = released.first() {
            out.push(x.name.to_string());
        }
    }
    Option::Some(out)
}

/// Emit a Zephyr-shaped project (prj.conf + main.c) for a schedulable task set: rate-monotonic
/// priorities (shortest period = highest priority), periodic k_msleep loops. Open-source RTOS,
/// no licence fee. None if the set is not schedulable.
pub fn zephyr_project(ts: &[Task]) -> Option<(String, String)> {
    if !schedulable(ts) {
        return Option::None;
    }
    let mut order: Vec<&Task> = ts.iter().collect();
    order.sort_by(|a, b| a.period_us.cmp(&b.period_us).then_with(|| a.name.cmp(b.name)));
    let conf = String::from("CONFIG_MULTITHREADING=y\nCONFIG_TIMESLICING=n\nCONFIG_THREAD_NAME=y\n");
    let mut c = String::from("#include <zephyr/kernel.h>\n");
    for (i, t) in order.iter().enumerate() {
        c.push_str(&format!(
            "void {name}_entry(void *a, void *b, void *c) {{ for (;;) {{ /* wcet {wcet}us */ k_msleep({period}); }} }}\n",
            name = t.name, wcet = t.wcet_us, period = t.period_us / 1000
        ));
        let _ = i;
    }
    for i in 0..order.len() {
        c.push_str(&format!("K_THREAD_STACK_DEFINE(stack{i}, 1024);\nstatic struct k_thread t{i};\n"));
    }
    c.push_str("int main(void) {\n");
    for (i, t) in order.iter().enumerate() {
        c.push_str(&format!(
            "  k_thread_create(&t{i}, stack{i}, K_THREAD_STACK_SIZEOF(stack{i}), {name}_entry, NULL, NULL, NULL, {prio}, 0, K_NO_WAIT);\n",
            i = i, name = t.name, prio = i
        ));
    }
    c.push_str("  return 0;\n}\n");
    Option::Some((conf, c))
}

/// Emit a seL4-shaped manifest (JSON): one protected thread per task with its priority,
/// period and WCET, plus the scheduler's admission verdict. seL4 is open source with public
/// machine-checked proofs; no licence fee.
pub fn sel4_manifest(ts: &[Task]) -> Option<String> {
    if !schedulable(ts) {
        return Option::None;
    }
    let mut order: Vec<&Task> = ts.iter().collect();
    order.sort_by(|a, b| a.period_us.cmp(&b.period_us).then_with(|| a.name.cmp(b.name)));
    let threads: Vec<serde_json::Value> = order.iter().enumerate()
        .map(|(i, t)| serde_json::json!({"name": t.name, "priority": i, "period_us": t.period_us, "wcet_us": t.wcet_us}))
        .collect();
    Option::Some(serde_json::json!({
        "schema": "rce-sel4-manifest/1",
        "kernel": "seL4 (open source, machine-checked)",
        "scheduler": "rate-monotonic, admission-checked",
        "utilization": utilization(ts),
        "bound": rm_bound(ts.len()),
        "threads": threads
    }).to_string())
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
