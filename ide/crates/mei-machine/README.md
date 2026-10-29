# metabolic-intelligence-machine

A deterministic statechart library.

A machine is data: named states, named events, and transitions. A transition is only from one named state, on one named event, to one named state. An actor holds the current state. `Actor::send` moves only when the current state allows that event. Otherwise the event does not apply and the actor keeps its state. `send` then returns `false`. It does not panic.

Two identical machines given the same event sequence end in the same state.

A `Statechart` is the same rule with shape. A `StateNode` is atomic, compound, or parallel. A compound state has child states and one initial child. The innermost active state handles an event first. A parent handles it only when that child does not. A parallel state keeps two or more regions active. An event is offered to each region. A region with no transition for it stays put. If no region and no ancestor has a transition, the event does not apply to `ChartActor::send` and the configuration stays put.

A compound or atomic state may name one child chart. Entering the state starts that chart at its initial configuration. Leaving the state stops the child; it does not keep running. Entering again starts a fresh child at the initial configuration. `ChartActor::send_to_child` delivers an event to the active child only. If no child is active, the send does not apply and the parent configuration stays put. An event that does not apply to the child changes neither the child nor the parent. A parallel state cannot name a child chart. Two child charts that could be active at the same time are not a statechart. The same parent events and the same child events end in the same parent configuration and the same child configuration.

This crate does not ask a question, call a model, or describe hardware.
