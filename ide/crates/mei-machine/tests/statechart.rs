use mei_machine::{ChartActor, StateNode, Statechart, Transition};

fn nested() -> Statechart {
    Statechart::try_new(
        "app",
        StateNode::compound(
            "work",
            [
                (
                    "work",
                    StateNode::compound(
                        "editing",
                        [
                            (
                                "editing",
                                StateNode::compound(
                                    "draft",
                                    [
                                        ("draft", StateNode::atomic()),
                                        ("polished", StateNode::atomic()),
                                    ],
                                ),
                            ),
                            ("review", StateNode::atomic()),
                        ],
                    ),
                ),
                ("done", StateNode::atomic()),
            ],
        ),
        ["save", "cancel"],
        [
            Transition::new("draft", "save", "polished"),
            Transition::new("work", "save", "done"),
            Transition::new("work", "cancel", "done"),
        ],
    )
    .expect("nested chart")
}

fn active(actor: &ChartActor) -> Vec<String> {
    actor.configuration().states().map(str::to_string).collect()
}

#[test]
fn nested_initial_child() {
    let chart = nested();
    let actor = ChartActor::start(&chart);
    assert_eq!(active(&actor), vec!["app", "draft", "editing", "work"]);
    assert!(actor.is_active("draft"));
    assert!(!actor.is_active("polished"));
    assert!(!actor.is_active("review"));
    assert!(!actor.is_active("done"));
    assert_eq!(actor.configuration(), chart.initial_configuration());
}

#[test]
fn event_handled_by_child() {
    let chart = nested();
    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("save"));
    // `work` also has `save`, to `done`. The innermost state wins.
    assert_eq!(active(&actor), vec!["app", "editing", "polished", "work"]);
    assert!(!actor.is_active("done"));
}

#[test]
fn event_handled_by_parent_when_child_has_none() {
    let chart = nested();
    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("cancel"));
    assert_eq!(active(&actor), vec!["app", "done"]);

    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("save"));
    assert!(actor.is_active("polished"));
    // `polished` and `editing` have no `save`. `work` does.
    assert!(actor.send("save"));
    assert_eq!(active(&actor), vec!["app", "done"]);
}

#[test]
fn illegal_event_refused_configuration_unchanged() {
    let chart = nested();
    let mut actor = ChartActor::start(&chart);
    let snapshot = actor.configuration();
    assert!(!actor.send("nope"));
    assert_eq!(actor.configuration(), snapshot);
    assert!(actor.send("save"));
    let snapshot = actor.configuration();
    assert!(!actor.send("cancel-not-declared"));
    assert!(!actor.send("nope"));
    assert_eq!(actor.configuration(), snapshot);
    assert_eq!(active(&actor), vec!["app", "editing", "polished", "work"]);
}

#[test]
fn parallel_regions_advance_independently() {
    let chart = Statechart::try_new(
        "rig",
        StateNode::parallel([
            (
                "motion",
                StateNode::compound(
                    "still",
                    [
                        ("still", StateNode::atomic()),
                        ("moving", StateNode::atomic()),
                    ],
                ),
            ),
            (
                "lamp",
                StateNode::compound(
                    "dark",
                    [("dark", StateNode::atomic()), ("lit", StateNode::atomic())],
                ),
            ),
        ]),
        ["tick", "halt"],
        [
            Transition::new("still", "tick", "moving"),
            Transition::new("dark", "tick", "lit"),
            Transition::new("moving", "halt", "still"),
        ],
    )
    .expect("parallel chart");

    let mut actor = ChartActor::start(&chart);
    assert_eq!(
        active(&actor),
        vec!["dark", "lamp", "motion", "rig", "still"]
    );

    let snapshot = actor.configuration();
    assert!(!actor.send("halt"));
    assert_eq!(actor.configuration(), snapshot);

    assert!(actor.send("tick"));
    assert_eq!(
        active(&actor),
        vec!["lamp", "lit", "motion", "moving", "rig"]
    );

    assert!(actor.send("halt"));
    assert_eq!(
        active(&actor),
        vec!["lamp", "lit", "motion", "rig", "still"]
    );
    assert!(actor.is_active("lit"));
    assert!(actor.is_active("still"));
}

#[test]
fn same_machine_same_events_same_configuration() {
    let left = nested();
    let right = nested();
    assert_eq!(left, right);

    let mut a = ChartActor::start(&left);
    let mut b = ChartActor::start(&right);
    let events = ["save", "nope", "cancel", "save", "cancel"];
    for event in events {
        assert_eq!(a.send(event), b.send(event));
        assert_eq!(a.configuration(), b.configuration());
    }
    assert_eq!(active(&a), vec!["app", "done"]);
    assert_eq!(active(&b), vec!["app", "done"]);
}

#[test]
fn ancestor_handles_when_no_region_does() {
    let chart = Statechart::try_new(
        "app",
        StateNode::compound(
            "rig",
            [
                (
                    "rig",
                    StateNode::parallel([
                        (
                            "motion",
                            StateNode::compound(
                                "still",
                                [
                                    ("still", StateNode::atomic()),
                                    ("moving", StateNode::atomic()),
                                ],
                            ),
                        ),
                        (
                            "lamp",
                            StateNode::compound(
                                "dark",
                                [("dark", StateNode::atomic()), ("lit", StateNode::atomic())],
                            ),
                        ),
                    ]),
                ),
                ("stopped", StateNode::atomic()),
            ],
        ),
        ["tick", "cut"],
        [
            Transition::new("still", "tick", "moving"),
            Transition::new("dark", "tick", "lit"),
            Transition::new("app", "tick", "stopped"),
            Transition::new("app", "cut", "stopped"),
        ],
    )
    .expect("ancestor chart");

    let mut actor = ChartActor::start(&chart);
    // Regions take `tick`, so the ancestor transition on `tick` does not fire.
    assert!(actor.send("tick"));
    assert!(actor.is_active("moving"));
    assert!(actor.is_active("lit"));
    assert!(!actor.is_active("stopped"));

    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("cut"));
    assert_eq!(active(&actor), vec!["app", "stopped"]);
}

#[test]
fn conflicting_region_steps_are_refused() {
    let chart = Statechart::try_new(
        "app",
        StateNode::compound(
            "rig",
            [
                (
                    "rig",
                    StateNode::parallel([
                        (
                            "motion",
                            StateNode::compound(
                                "still",
                                [
                                    ("still", StateNode::atomic()),
                                    ("moving", StateNode::atomic()),
                                ],
                            ),
                        ),
                        (
                            "lamp",
                            StateNode::compound(
                                "dark",
                                [("dark", StateNode::atomic()), ("lit", StateNode::atomic())],
                            ),
                        ),
                    ]),
                ),
                ("stopped", StateNode::atomic()),
            ],
        ),
        ["boom"],
        [
            Transition::new("still", "boom", "stopped"),
            Transition::new("dark", "boom", "lit"),
        ],
    )
    .expect("conflict chart");

    let mut actor = ChartActor::start(&chart);
    let snapshot = actor.configuration();
    assert!(!actor.send("boom"));
    assert_eq!(actor.configuration(), snapshot);
}

#[test]
fn entering_parallel_starts_every_region() {
    let chart = Statechart::try_new(
        "app",
        StateNode::compound(
            "idle",
            [
                ("idle", StateNode::atomic()),
                (
                    "rig",
                    StateNode::parallel([
                        (
                            "motion",
                            StateNode::compound(
                                "still",
                                [
                                    ("still", StateNode::atomic()),
                                    ("moving", StateNode::atomic()),
                                ],
                            ),
                        ),
                        (
                            "lamp",
                            StateNode::compound(
                                "dark",
                                [("dark", StateNode::atomic()), ("lit", StateNode::atomic())],
                            ),
                        ),
                    ]),
                ),
            ],
        ),
        ["start", "focus"],
        [
            Transition::new("idle", "start", "rig"),
            Transition::new("idle", "focus", "moving"),
        ],
    )
    .expect("entry chart");

    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("start"));
    assert_eq!(
        active(&actor),
        vec!["app", "dark", "lamp", "motion", "rig", "still"]
    );

    let mut actor = ChartActor::start(&chart);
    assert!(actor.send("focus"));
    assert_eq!(
        active(&actor),
        vec!["app", "dark", "lamp", "motion", "moving", "rig"]
    );
}

#[test]
fn rejects_ill_formed_charts() {
    let atomic = || ("only", StateNode::atomic());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound("missing", [("idle", StateNode::atomic())]),
        ["go"],
        [],
    )
    .is_none());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound("idle", Vec::<(String, StateNode)>::new()),
        ["go"],
        [],
    )
    .is_none());
    assert!(Statechart::try_new("app", StateNode::parallel([atomic()]), ["go"], []).is_none());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound(
            "idle",
            [("idle", StateNode::atomic()), ("idle", StateNode::atomic())],
        ),
        ["go"],
        [],
    )
    .is_none());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound(
            "a",
            [
                (
                    "a",
                    StateNode::compound(
                        "x",
                        [("x", StateNode::atomic()), ("same", StateNode::atomic())]
                    ),
                ),
                (
                    "b",
                    StateNode::compound("same", [("same", StateNode::atomic())]),
                ),
            ],
        ),
        ["go"],
        [],
    )
    .is_none());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound(
            "idle",
            [("idle", StateNode::atomic()), ("run", StateNode::atomic())],
        ),
        ["go"],
        [
            Transition::new("idle", "go", "run"),
            Transition::new("idle", "go", "idle"),
        ],
    )
    .is_none());
    assert!(Statechart::try_new(
        "app",
        StateNode::compound("idle", [("idle", StateNode::atomic())]),
        ["go"],
        [Transition::new("idle", "go", "missing")],
    )
    .is_none());
    assert!(Statechart::try_new(
        "rig",
        StateNode::parallel([
            (
                "motion",
                StateNode::compound(
                    "still",
                    [
                        ("still", StateNode::atomic()),
                        ("moving", StateNode::atomic())
                    ],
                ),
            ),
            (
                "lamp",
                StateNode::compound(
                    "dark",
                    [("dark", StateNode::atomic()), ("lit", StateNode::atomic())],
                ),
            ),
        ]),
        ["jump"],
        [Transition::new("still", "jump", "lit")],
    )
    .is_none());
}
