use mei_machine::{ChartActor, Configuration, StateNode, Statechart, Transition};

fn sub() -> Statechart {
    Statechart::try_new(
        "sub",
        StateNode::compound(
            "x",
            [("x", StateNode::atomic()), ("y", StateNode::atomic())],
        ),
        ["flip"],
        [Transition::new("x", "flip", "y")],
    )
    .expect("sub chart")
}

fn session() -> Statechart {
    Statechart::try_new(
        "session",
        StateNode::compound(
            "idle",
            [
                ("idle", StateNode::atomic()),
                ("run", StateNode::atomic().with_invoke(sub())),
            ],
        ),
        ["go", "reset"],
        [
            Transition::new("idle", "go", "run"),
            Transition::new("run", "reset", "idle"),
        ],
    )
    .expect("session chart")
}

fn toggle() -> Statechart {
    Statechart::try_new(
        "app",
        StateNode::compound(
            "on",
            [
                ("on", StateNode::atomic().with_invoke(session())),
                ("off", StateNode::atomic()),
            ],
        ),
        ["toggle", "poke"],
        [
            Transition::new("on", "toggle", "off"),
            Transition::new("off", "toggle", "on"),
        ],
    )
    .expect("toggle chart")
}

fn kept() -> Statechart {
    Statechart::try_new(
        "app",
        StateNode::compound(
            "work",
            [
                (
                    "work",
                    StateNode::compound(
                        "draft",
                        [
                            ("draft", StateNode::atomic()),
                            ("polished", StateNode::atomic()),
                        ],
                    )
                    .with_invoke(session()),
                ),
                ("done", StateNode::atomic()),
            ],
        ),
        ["save", "finish"],
        [
            Transition::new("draft", "save", "polished"),
            Transition::new("work", "finish", "done"),
        ],
    )
    .expect("kept chart")
}

fn reenter() -> Statechart {
    Statechart::try_new(
        "app",
        StateNode::compound(
            "work",
            [(
                "work",
                StateNode::compound(
                    "editing",
                    [("editing", StateNode::atomic().with_invoke(session()))],
                ),
            )],
        ),
        ["up"],
        [Transition::new("editing", "up", "work")],
    )
    .expect("reenter chart")
}

fn states_of(actor: &ChartActor) -> Vec<String> {
    actor.configuration().states().map(str::to_string).collect()
}

fn expect_states(names: &[&str]) -> Vec<String> {
    names.iter().copied().map(str::to_string).collect()
}

fn child_states(actor: &ChartActor) -> Option<Vec<String>> {
    actor.child().map(states_of)
}

fn grand_states(actor: &ChartActor) -> Option<Vec<String>> {
    let child = actor.child()?;
    let grand = child.child()?;
    Some(states_of(grand))
}

fn child_configuration(actor: &ChartActor) -> Option<Configuration> {
    actor.child().map(ChartActor::configuration)
}

fn grand_configuration(actor: &ChartActor) -> Option<Configuration> {
    actor
        .child()
        .and_then(ChartActor::child)
        .map(ChartActor::configuration)
}

#[test]
fn entering_starts_child_at_initial() {
    let chart = toggle();
    let actor = ChartActor::start(&chart);
    assert_eq!(states_of(&actor), vec!["app", "on"]);
    assert_eq!(actor.invoking_state(), Some("on"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["idle", "session"]))
    );
    assert_eq!(
        actor.child().unwrap().configuration(),
        chart.invoked("on").unwrap().initial_configuration()
    );
    assert_eq!(chart.invoked("on"), Some(&session()));
    assert!(chart.invoked("off").is_none());
    assert!(chart.invoked("missing").is_none());
    assert!(grand_states(&actor).is_none());
    assert!(!actor.configuration().contains("idle"));
    assert!(!actor.configuration().contains("session"));
}

#[test]
fn leaving_stops_the_child() {
    let chart = toggle();
    let mut actor = ChartActor::start(&chart);
    assert!(actor.send_to_child("go"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["run", "session"]))
    );
    assert_eq!(grand_states(&actor), Some(expect_states(&["sub", "x"])));
    assert!(actor.send("toggle"));
    assert_eq!(states_of(&actor), vec!["app", "off"]);
    assert!(actor.child().is_none());
    assert!(actor.invoking_state().is_none());
    let parent = actor.configuration();
    assert!(!actor.send_to_child("go"));
    assert!(!actor.send_to_child("flip"));
    assert_eq!(actor.configuration(), parent);
    assert!(actor.child().is_none());
}

#[test]
fn kept_child_event_changes_nothing() {
    let chart = toggle();
    let mut actor = ChartActor::start(&chart);
    assert!(actor.send_to_child("go"));
    let parent = actor.configuration();
    let child = actor.child().unwrap().configuration();
    let grand = actor.child().unwrap().child().unwrap().configuration();
    assert!(!actor.send_to_child("go"));
    assert!(!actor.send_to_child("missing"));
    assert_eq!(actor.configuration(), parent);
    assert_eq!(actor.child().unwrap().configuration(), child);
    assert_eq!(
        actor.child().unwrap().child().unwrap().configuration(),
        grand
    );
    assert!(!actor.send("poke"));
    assert_eq!(actor.configuration(), parent);
    assert_eq!(actor.child().unwrap().configuration(), child);
    assert!(!actor.send("reset"));
    assert_eq!(actor.configuration(), parent);
    assert_eq!(actor.child().unwrap().configuration(), child);
    assert!(actor.action_log().is_empty());
}

#[test]
fn child_event_does_not_change_the_parent() {
    let chart = toggle();
    let mut actor = ChartActor::start(&chart);
    let parent = actor.configuration();
    assert!(actor.send_to_child("go"));
    assert_eq!(actor.configuration(), parent);
    assert!(actor.action_log().is_empty());
    assert!(actor.child_mut().unwrap().send_to_child("flip"));
    assert_eq!(actor.configuration(), parent);
    assert_eq!(grand_states(&actor), Some(expect_states(&["sub", "y"])));
    assert!(actor.action_log().is_empty());
}

#[test]
fn transition_that_does_not_exit_the_state_keeps_the_child() {
    let chart = kept();
    let mut actor = ChartActor::start(&chart);
    assert_eq!(actor.invoking_state(), Some("work"));
    assert!(actor.send_to_child("go"));
    assert!(actor.child_mut().unwrap().send_to_child("flip"));
    let child = actor.child().unwrap().configuration();
    let grand = actor.child().unwrap().child().unwrap().configuration();
    assert!(actor.send("save"));
    assert_eq!(states_of(&actor), vec!["app", "polished", "work"]);
    assert_eq!(actor.invoking_state(), Some("work"));
    assert_eq!(actor.child().unwrap().configuration(), child);
    assert_eq!(
        actor.child().unwrap().child().unwrap().configuration(),
        grand
    );
    assert!(actor.send("finish"));
    assert_eq!(states_of(&actor), vec!["app", "done"]);
    assert!(actor.child().is_none());
}

#[test]
fn reentering_starts_a_fresh_child() {
    let chart = toggle();
    let mut actor = ChartActor::start(&chart);
    assert!(actor.send_to_child("go"));
    assert!(actor.child_mut().unwrap().send_to_child("flip"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["run", "session"]))
    );
    assert_eq!(grand_states(&actor), Some(expect_states(&["sub", "y"])));
    assert!(actor.send("toggle"));
    assert!(actor.send("toggle"));
    assert_eq!(states_of(&actor), vec!["app", "on"]);
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["idle", "session"]))
    );
    assert!(grand_states(&actor).is_none());
    assert!(actor.send_to_child("go"));
    assert_eq!(grand_states(&actor), Some(expect_states(&["sub", "x"])));
}

#[test]
fn exiting_and_entering_again_in_one_step_starts_fresh() {
    let chart = reenter();
    let mut actor = ChartActor::start(&chart);
    let parent = actor.configuration();
    assert!(actor.send_to_child("go"));
    assert!(actor.child_mut().unwrap().send_to_child("flip"));
    assert!(actor.send("up"));
    assert_eq!(actor.configuration(), parent);
    assert_eq!(actor.invoking_state(), Some("editing"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["idle", "session"]))
    );
    assert!(grand_states(&actor).is_none());
}

#[derive(Clone, Copy)]
enum Step {
    Parent(&'static str),
    Child(&'static str),
    Grand(&'static str),
}

fn apply(actor: &mut ChartActor, step: Step) -> bool {
    match step {
        Step::Parent(event) => actor.send(event),
        Step::Child(event) => actor.send_to_child(event),
        Step::Grand(event) => match actor.child_mut() {
            Some(child) => child.send_to_child(event),
            None => false,
        },
    }
}

#[test]
fn same_events_same_parent_and_child_configurations() {
    let left = toggle();
    let right = toggle();
    assert_eq!(left, right);

    let mut a = ChartActor::start(&left);
    let mut b = ChartActor::start(&right);
    let steps = [
        Step::Child("go"),
        Step::Grand("flip"),
        Step::Child("go"),
        Step::Child("missing"),
        Step::Parent("poke"),
        Step::Parent("reset"),
        Step::Parent("toggle"),
        Step::Child("go"),
        Step::Grand("flip"),
        Step::Parent("toggle"),
        Step::Child("go"),
        Step::Grand("flip"),
    ];

    let mut seen_fresh = false;
    for (index, step) in steps.iter().copied().enumerate() {
        assert_eq!(apply(&mut a, step), apply(&mut b, step));
        assert_eq!(a.configuration(), b.configuration());
        assert_eq!(child_configuration(&a), child_configuration(&b));
        assert_eq!(grand_configuration(&a), grand_configuration(&b));
        if index == 9 {
            assert_eq!(states_of(&a), vec!["app", "on"]);
            assert_eq!(child_states(&a), Some(expect_states(&["idle", "session"])));
            assert!(grand_states(&a).is_none());
            seen_fresh = true;
        }
    }
    assert!(seen_fresh);
    assert_eq!(states_of(&a), vec!["app", "on"]);
    assert_eq!(child_states(&a), Some(expect_states(&["run", "session"])));
    assert_eq!(grand_states(&a), Some(expect_states(&["sub", "y"])));
}

#[test]
fn parallel_and_coactive_invokes_are_rejected() {
    assert!(Statechart::try_new(
        "rig",
        StateNode::parallel([
            ("motion", StateNode::atomic()),
            ("lamp", StateNode::atomic()),
        ])
        .with_invoke(sub()),
        ["tick"],
        [],
    )
    .is_none());

    assert!(Statechart::try_new(
        "app",
        StateNode::compound(
            "work",
            [(
                "work",
                StateNode::compound("draft", [("draft", StateNode::atomic().with_invoke(sub()))],)
                    .with_invoke(session()),
            )],
        ),
        ["go"],
        [],
    )
    .is_none());

    assert!(Statechart::try_new(
        "rig",
        StateNode::parallel([
            ("motion", StateNode::atomic().with_invoke(sub())),
            ("lamp", StateNode::atomic().with_invoke(session())),
        ]),
        ["tick"],
        [],
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
                        ("still", StateNode::atomic().with_invoke(sub())),
                        ("moving", StateNode::atomic()),
                    ],
                ),
            ),
            ("lamp", StateNode::atomic()),
        ]),
        ["tick"],
        [],
    )
    .is_some());
}

#[test]
fn exclusive_states_invoke_their_own_charts() {
    let chart = Statechart::try_new(
        "app",
        StateNode::compound(
            "left",
            [
                ("left", StateNode::atomic().with_invoke(session())),
                ("right", StateNode::atomic().with_invoke(sub())),
            ],
        ),
        ["switch"],
        [
            Transition::new("left", "switch", "right"),
            Transition::new("right", "switch", "left"),
        ],
    )
    .expect("exclusive invokes");

    let mut actor = ChartActor::start(&chart);
    assert!(actor.send_to_child("go"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["run", "session"]))
    );
    assert!(actor.send("switch"));
    assert_eq!(actor.invoking_state(), Some("right"));
    assert_eq!(child_states(&actor), Some(expect_states(&["sub", "x"])));
    assert!(actor.child().unwrap().child().is_none());
    assert!(actor.send_to_child("flip"));
    assert_eq!(child_states(&actor), Some(expect_states(&["sub", "y"])));
    assert!(actor.send("switch"));
    assert_eq!(actor.invoking_state(), Some("left"));
    assert_eq!(
        child_states(&actor),
        Some(expect_states(&["idle", "session"]))
    );
    assert!(grand_states(&actor).is_none());
}
