//! The MEI turn statechart and the per-gear child charts.
use mei_machine::Transition;
use mei_machine::{ChartActor, Configuration, StateNode, Statechart};

pub const GEARS: [&str; 5] = ["gear0", "gear1", "gear2", "gear3", "gear4"];

fn atomic(names: &[&str]) -> Vec<(String, StateNode)> {
    names
        .iter()
        .map(|n| (n.to_string(), StateNode::atomic()))
        .collect()
}

/// G0: run the checker, then the tests.
pub fn g0_chart() -> Statechart {
    Statechart::try_new(
        "g0",
        StateNode::compound("g0_idle", atomic(&["g0_idle", "g0_checked", "g0_tested"])),
        ["run_check", "run_tests"],
        [
            Transition::new("g0_idle", "run_check", "g0_checked"),
            Transition::new("g0_checked", "run_tests", "g0_tested"),
        ],
    )
    .expect("g0 chart")
}

/// G1: look the task up in the rules table.
pub fn g1_chart() -> Statechart {
    Statechart::try_new(
        "g1",
        StateNode::compound("g1_idle", atomic(&["g1_idle", "g1_hit", "g1_miss"])),
        ["lookup_hit", "lookup_miss"],
        [
            Transition::new("g1_idle", "lookup_hit", "g1_hit"),
            Transition::new("g1_idle", "lookup_miss", "g1_miss"),
        ],
    )
    .expect("g1 chart")
}

/// G2: ask the System-One model a typed question.
pub fn g2_chart() -> Statechart {
    Statechart::try_new(
        "g2",
        StateNode::compound(
            "g2_idle",
            atomic(&["g2_idle", "g2_asked", "g2_answered", "g2_failed"]),
        ),
        ["ask", "answered", "failed"],
        [
            Transition::new("g2_idle", "ask", "g2_asked"),
            Transition::new("g2_asked", "answered", "g2_answered"),
            Transition::new("g2_asked", "failed", "g2_failed"),
            Transition::new("g2_idle", "failed", "g2_failed"),
        ],
    )
    .expect("g2 chart")
}

/// G3 and G4: select a provider, request, then a response or a failure.
pub fn model_chart(p: &str) -> Statechart {
    let s = |x: &str| format!("{p}_{x}");
    let names = [
        s("idle"),
        s("selected"),
        s("waiting"),
        s("answered"),
        s("failed"),
    ];
    Statechart::try_new(
        p,
        StateNode::compound(
            s("idle"),
            names
                .iter()
                .map(|n| (n.clone(), StateNode::atomic()))
                .collect::<Vec<_>>(),
        ),
        ["select", "request", "response", "failed"],
        [
            Transition::new(s("idle"), "select", s("selected")),
            Transition::new(s("idle"), "failed", s("failed")),
            Transition::new(s("selected"), "request", s("waiting")),
            Transition::new(s("selected"), "failed", s("failed")),
            Transition::new(s("waiting"), "response", s("answered")),
            Transition::new(s("waiting"), "failed", s("failed")),
        ],
    )
    .expect("model chart")
}

fn g_budget_ok(c: &Configuration, _: &str) -> bool {
    c.contains("budget_ok")
}
fn g_accept(c: &Configuration, _: &str) -> bool {
    c.contains("v_done")
}
fn g_escalate(c: &Configuration, _: &str) -> bool {
    (c.contains("v_reject") || c.contains("v_not_done"))
        && c.contains("budget_ok")
        && !c.contains("gear4")
}
fn g_in_escalate(c: &Configuration, _: &str) -> bool {
    c.contains("escalate")
}

/// The turn chart.
pub fn turn_chart() -> Statechart {
    let gears = vec![
        (
            "gear0".to_string(),
            StateNode::atomic().with_invoke(g0_chart()),
        ),
        (
            "gear1".to_string(),
            StateNode::atomic().with_invoke(g1_chart()),
        ),
        (
            "gear2".to_string(),
            StateNode::atomic().with_invoke(g2_chart()),
        ),
        (
            "gear3".to_string(),
            StateNode::atomic().with_invoke(model_chart("g3")),
        ),
        (
            "gear4".to_string(),
            StateNode::atomic().with_invoke(model_chart("g4")),
        ),
    ];
    let root = StateNode::parallel([
        (
            "turn",
            StateNode::compound(
                "plan",
                atomic(&["plan", "propose", "check", "accept", "escalate", "failed"]),
            ),
        ),
        ("gears", StateNode::compound("gear0", gears)),
        (
            "verdict",
            StateNode::compound(
                "v_none",
                atomic(&["v_none", "v_done", "v_not_done", "v_reject"]),
            ),
        ),
        (
            "budget",
            StateNode::compound("budget_ok", atomic(&["budget_ok", "budget_out"])),
        ),
    ]);
    let mut t = vec![
        Transition::new("plan", "start", "propose").with_guard("budget_ok"),
        Transition::new("plan", "start", "failed"),
        Transition::new("propose", "proposed", "check"),
        Transition::new("check", "decide", "accept").with_guard("accept"),
        Transition::new("check", "decide", "escalate").with_guard("escalate"),
        Transition::new("check", "decide", "failed"),
        Transition::new("escalate", "next", "propose"),
        Transition::new("v_none", "set_done", "v_done"),
        Transition::new("v_none", "set_not_done", "v_not_done"),
        Transition::new("v_none", "set_reject", "v_reject"),
        Transition::new("v_done", "next", "v_none"),
        Transition::new("v_not_done", "next", "v_none"),
        Transition::new("v_reject", "next", "v_none"),
        Transition::new("budget_ok", "budget_spent", "budget_out"),
    ];
    for k in 0..4 {
        t.push(Transition::new(GEARS[k], "next", GEARS[k + 1]).with_guard("in_escalate"));
    }
    Statechart::try_new(
        "mei",
        root,
        [
            "start",
            "proposed",
            "decide",
            "next",
            "set_done",
            "set_not_done",
            "set_reject",
            "budget_spent",
        ],
        t,
    )
    .expect("turn chart")
}

/// An actor on the turn chart with the guards registered.
pub fn actor(chart: &Statechart) -> ChartActor<'_> {
    let mut a = ChartActor::start(chart);
    a.register_guard("budget_ok", g_budget_ok);
    a.register_guard("accept", g_accept);
    a.register_guard("escalate", g_escalate);
    a.register_guard("in_escalate", g_in_escalate);
    a
}

/// The active gear index.
pub fn gear_of(a: &ChartActor<'_>) -> usize {
    GEARS
        .iter()
        .position(|g| a.is_active(g))
        .expect("one gear is active")
}
