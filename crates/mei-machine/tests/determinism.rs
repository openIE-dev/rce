use mei_machine::{Actor, Machine, Transition};

fn machine() -> Machine {
    Machine::try_new(
        ["idle", "run", "fault"],
        ["start", "stop", "fail"],
        "idle",
        [
            Transition::new("idle", "start", "run"),
            Transition::new("run", "stop", "idle"),
            Transition::new("run", "fail", "fault"),
        ],
    )
    .expect("machine")
}

#[test]
fn identical_machines_same_events_same_state() {
    let left = machine();
    let right = machine();
    assert_eq!(left, right);

    let mut a = Actor::start(&left);
    let mut b = Actor::start(&right);
    let events = ["start", "bogus", "fail", "stop", "start", "stop"];

    for event in events {
        assert_eq!(a.send(event), b.send(event));
        assert_eq!(a.state(), b.state());
    }
    assert_eq!(a.state(), "fault");
    assert_eq!(b.state(), "fault");
}

#[test]
fn transition_fires_only_from_its_state() {
    let machine = machine();
    let mut actor = Actor::start(&machine);
    assert!(actor.send("start"));
    assert_eq!(actor.state(), "run");
    assert!(actor.send("stop"));
    assert_eq!(actor.state(), "idle");
    assert!(!actor.send("stop"));
    assert_eq!(actor.state(), "idle");
}
