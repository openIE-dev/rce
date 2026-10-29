use mei_machine::{
    ActionRecord, Actor, ChartActor, Configuration, Machine, StateNode, Statechart, Transition,
};

fn allow(_configuration: &Configuration, _event: &str) -> bool {
    true
}

fn deny(_configuration: &Configuration, _event: &str) -> bool {
    false
}

fn sees_target(from: &str, _event: &str, to: &str, configuration: &Configuration) {
    assert!(
        configuration.contains(to),
        "action ran before {to} was active"
    );
    assert!(
        !configuration.contains(from),
        "action ran before {from} was left"
    );
}

fn panic_action(from: &str, event: &str, to: &str, _configuration: &Configuration) {
    panic!("action must not run for {from} -{event}-> {to}");
}

fn chart(transitions: Vec<Transition>) -> Statechart {
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
                            ("review", StateNode::atomic()),
                        ],
                    ),
                ),
                ("done", StateNode::atomic()),
            ],
        ),
        ["go", "stop"],
        transitions,
    )
    .expect("chart")
}

fn active(actor: &ChartActor) -> Vec<String> {
    actor.configuration().states().map(str::to_string).collect()
}

#[test]
fn guard_false_keeps_the_state_and_logs_nothing() {
    let machine = chart(vec![Transition::new("draft", "stop", "polished")
        .with_guard("stop_guard")
        .with_action("take_stop")]);
    let mut actor = ChartActor::start(&machine);
    actor.register_guard("stop_guard", deny);
    actor.register_action("take_stop", panic_action);
    let snapshot = actor.configuration();
    assert!(!actor.send("stop"));
    assert_eq!(actor.configuration(), snapshot);
    assert!(actor.is_active("draft"));
    assert!(actor.action_log().is_empty());
}

#[test]
fn guard_true_action_sees_committed_target() {
    let machine = chart(vec![
        Transition::new("draft", "go", "polished")
            .with_guard("first")
            .with_action("take_polished"),
        Transition::new("draft", "go", "review")
            .with_guard("second")
            .with_action("take_review"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    let mut actor = ChartActor::start(&machine);
    actor.register_guard("first", allow);
    actor.register_guard("second", deny);
    actor.register_action("take_polished", sees_target);
    actor.register_action("take_review", panic_action);
    actor.register_action("take_done", panic_action);

    assert!(actor.send("go"));
    assert_eq!(active(&actor), vec!["app", "polished", "work"]);
    let log = actor.action_log();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].action, "take_polished");
    assert_eq!(log[0].from, "draft");
    assert_eq!(log[0].event, "go");
    assert_eq!(log[0].to, "polished");
    assert!(log[0].configuration.contains("polished"));
    assert!(!log[0].configuration.contains("draft"));
    assert_eq!(log[0].configuration, actor.configuration());
}

#[test]
fn missing_guard_does_not_take_the_transition() {
    let machine = chart(vec![Transition::new("draft", "stop", "polished")
        .with_guard("not_registered")
        .with_action("take_stop")]);
    let mut actor = ChartActor::start(&machine);
    actor.register_action("take_stop", panic_action);
    let snapshot = actor.configuration();
    assert!(!actor.send("stop"));
    assert_eq!(actor.configuration(), snapshot);
    assert!(actor.action_log().is_empty());
}

#[test]
fn next_candidate_on_the_same_state() {
    let machine = chart(vec![
        Transition::new("draft", "go", "polished")
            .with_guard("first")
            .with_action("take_polished"),
        Transition::new("draft", "go", "review")
            .with_guard("second")
            .with_action("take_review"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    let mut actor = ChartActor::start(&machine);
    actor.register_guard("first", deny);
    actor.register_guard("second", allow);
    actor.register_action("take_polished", panic_action);
    actor.register_action("take_review", sees_target);
    actor.register_action("take_done", panic_action);

    assert!(actor.send("go"));
    assert_eq!(active(&actor), vec!["app", "review", "work"]);
    assert_eq!(
        actor
            .action_log()
            .iter()
            .map(|entry| entry.action.as_str())
            .collect::<Vec<_>>(),
        vec!["take_review"]
    );
    assert!(actor.action_log()[0].configuration.contains("review"));
}

#[test]
fn parent_transition_when_child_guard_fails() {
    let machine = chart(vec![
        Transition::new("draft", "go", "polished")
            .with_guard("first")
            .with_action("take_polished"),
        Transition::new("draft", "go", "review")
            .with_guard("second")
            .with_action("take_review"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    let mut actor = ChartActor::start(&machine);
    actor.register_guard("first", deny);
    actor.register_guard("second", deny);
    actor.register_action("take_polished", panic_action);
    actor.register_action("take_review", panic_action);
    actor.register_action("take_done", sees_target);

    assert!(actor.send("go"));
    assert_eq!(active(&actor), vec!["app", "done"]);
    let log = actor.action_log();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].action, "take_done");
    assert_eq!(log[0].from, "work");
    assert_eq!(log[0].to, "done");
    assert!(log[0].configuration.contains("done"));
    assert!(!log[0].configuration.contains("draft"));
    assert!(!log[0].configuration.contains("work"));
    assert_eq!(log[0].configuration, actor.configuration());
}

#[test]
fn identical_charts_same_events_same_configuration_and_action_log() {
    let left = chart(vec![
        Transition::new("draft", "stop", "polished")
            .with_guard("stop_guard")
            .with_action("take_stop"),
        Transition::new("draft", "go", "polished")
            .with_guard("first")
            .with_action("take_polished"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    let right = chart(vec![
        Transition::new("draft", "stop", "polished")
            .with_guard("stop_guard")
            .with_action("take_stop"),
        Transition::new("draft", "go", "polished")
            .with_guard("first")
            .with_action("take_polished"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    assert_eq!(left, right);

    let mut a = ChartActor::start(&left);
    let mut b = ChartActor::start(&right);
    for actor in [&mut a, &mut b] {
        actor.register_guard("stop_guard", deny);
        actor.register_guard("first", allow);
        actor.register_action("take_stop", panic_action);
        actor.register_action("take_polished", sees_target);
        actor.register_action("take_done", sees_target);
    }

    let events = ["stop", "go", "nope", "go"];
    for event in events {
        assert_eq!(a.send(event), b.send(event));
        assert_eq!(a.configuration(), b.configuration());
        assert_eq!(a.action_log(), b.action_log());
    }
    assert_eq!(active(&a), vec!["app", "done"]);
    assert_eq!(
        a.action_log().iter().map(record_view).collect::<Vec<_>>(),
        vec![
            ("take_polished", "draft", "go", "polished"),
            ("take_done", "work", "go", "done"),
        ]
    );
    assert!(a.action_log()[0].configuration.contains("polished"));
    assert!(!a.action_log()[0].configuration.contains("done"));
    assert_eq!(a.action_log()[1].configuration, a.configuration());
}

fn record_view(entry: &ActionRecord) -> (&str, &str, &str, &str) {
    (&entry.action, &entry.from, &entry.event, &entry.to)
}

#[test]
fn parallel_actions_see_the_committed_configuration() {
    let machine = Statechart::try_new(
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
        ["tick"],
        [
            Transition::new("still", "tick", "moving")
                .with_guard("motion_ok")
                .with_action("move"),
            Transition::new("dark", "tick", "lit")
                .with_guard("lamp_ok")
                .with_action("light"),
        ],
    )
    .expect("parallel chart");

    let mut actor = ChartActor::start(&machine);
    actor.register_guard("motion_ok", allow);
    actor.register_guard("lamp_ok", allow);
    actor.register_action("move", sees_target);
    actor.register_action("light", sees_target);
    assert!(actor.send("tick"));
    assert_eq!(
        actor
            .action_log()
            .iter()
            .map(|entry| entry.action.as_str())
            .collect::<Vec<_>>(),
        vec!["light", "move"]
    );
    for entry in actor.action_log() {
        assert!(entry.configuration.contains("moving"));
        assert!(entry.configuration.contains("lit"));
        assert!(!entry.configuration.contains("still"));
        assert!(!entry.configuration.contains("dark"));
        assert_eq!(entry.configuration, actor.configuration());
    }
}

#[test]
fn missing_guard_skips_to_the_next_candidate() {
    let machine = chart(vec![
        Transition::new("draft", "go", "polished")
            .with_guard("not_registered")
            .with_action("take_polished"),
        Transition::new("draft", "go", "review")
            .with_guard("second")
            .with_action("take_review"),
        Transition::new("work", "go", "done").with_action("take_done"),
    ]);
    let mut actor = ChartActor::start(&machine);
    actor.register_guard("second", allow);
    actor.register_action("take_polished", panic_action);
    actor.register_action("take_review", sees_target);
    actor.register_action("take_done", panic_action);
    assert!(actor.send("go"));
    assert_eq!(active(&actor), vec!["app", "review", "work"]);
    assert_eq!(actor.action_log()[0].action, "take_review");
    assert!(actor.action_log()[0].configuration.contains("review"));
}

#[test]
fn duplicate_guard_id_is_not_a_chart() {
    let built = Statechart::try_new(
        "app",
        StateNode::compound(
            "draft",
            [
                ("draft", StateNode::atomic()),
                ("polished", StateNode::atomic()),
            ],
        ),
        ["go"],
        [
            Transition::new("draft", "go", "polished").with_guard("same"),
            Transition::new("draft", "go", "draft").with_guard("same"),
        ],
    );
    assert!(built.is_none());
}

fn flat() -> Machine {
    Machine::try_new(
        ["idle", "run", "fault"],
        ["start", "stop"],
        "idle",
        [
            Transition::new("idle", "start", "run")
                .with_guard("ready")
                .with_action("started"),
            Transition::new("idle", "start", "fault")
                .with_guard("backup")
                .with_action("failed_over"),
            Transition::new("run", "stop", "idle").with_action("stopped"),
        ],
    )
    .expect("flat machine")
}

#[test]
fn flat_actor_guards_and_actions_are_deterministic() {
    assert!(Machine::try_new(
        ["idle", "run"],
        ["start"],
        "idle",
        [
            Transition::new("idle", "start", "run").with_guard("ready"),
            Transition::new("idle", "start", "idle").with_guard("ready"),
        ],
    )
    .is_none());

    let left = flat();
    let right = flat();
    let mut a = Actor::start(&left);
    let mut b = Actor::start(&right);
    for actor in [&mut a, &mut b] {
        actor.register_guard("ready", deny);
        actor.register_guard("backup", allow);
        actor.register_action("started", panic_action);
        actor.register_action("failed_over", sees_target);
        actor.register_action("stopped", sees_target);
    }

    assert!(!a.send("stop"));
    assert!(a.action_log().is_empty());
    assert_eq!(a.state(), "idle");

    for actor in [&mut a, &mut b] {
        assert!(actor.send("start"));
        assert_eq!(actor.state(), "fault");
        assert_eq!(actor.action_log().len(), 1);
        assert_eq!(actor.action_log()[0].to, "fault");
        assert!(actor.action_log()[0].configuration.contains("fault"));
        assert!(!actor.send("stop"));
    }
    assert_eq!(a.state(), b.state());
    assert_eq!(a.action_log(), b.action_log());
}
