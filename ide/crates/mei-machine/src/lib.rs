//! Deterministic statechart.
//!
//! A [`Machine`] is data: named states, named events, and [`Transition`]s.
//! An [`Actor`] holds the current state. [`Actor::send`] changes that state
//! only when the current state has a transition on the event whose guard
//! passes. A refused event leaves the state unchanged and returns `false`.
//! Refusal is not a panic.
//!
//! A [`Statechart`] is that same rule with compound states and parallel
//! regions. The innermost active state handles an event first. On that state,
//! transitions for the event are tried in the order they were declared. A
//! parent handles the event only when the child does not. Each region of a
//! parallel state is offered the event. A region with no taken transition
//! stays put. If no region and no ancestor takes a transition, nothing changes.
//!
//! A compound or atomic state may name one child chart. While that state is
//! active, the chart runs as a child actor started at its initial
//! configuration. Leaving the state stops the child. It does not keep running.
//! Entering the state again starts a new child at that initial configuration.
//! [`ChartActor::send_to_child`] delivers an event to the active child only.
//! If no child is active, or the child refuses the event, the parent
//! configuration is unchanged.
//!
//! Guards and actions are registered by id on the actor. The machine stores
//! only the ids, so two machines built from the same data stay equal.

#![forbid(unsafe_code)]

mod statechart;

pub use statechart::{
    Action, ActionRecord, ChartActor, Configuration, Guard, StateNode, Statechart,
};

use std::collections::{BTreeMap, BTreeSet};

/// One allowed step: `from` on `event` goes to `to`.
///
/// `guard` and `action` are ids registered on an actor. Neither is a function
/// on the machine. `None` means the transition is always eligible, or that
/// taking it runs no action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition {
    pub from: String,
    pub event: String,
    pub to: String,
    pub guard: Option<String>,
    pub action: Option<String>,
}

impl Transition {
    pub fn new(from: impl Into<String>, event: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            event: event.into(),
            to: to.into(),
            guard: None,
            action: None,
        }
    }

    /// Names the guard that must pass before this transition can be taken.
    pub fn with_guard(mut self, guard: impl Into<String>) -> Self {
        self.guard = Some(guard.into());
        self
    }

    /// Names the action to run after this transition is taken and the state has changed.
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge {
    to: String,
    guard: Option<String>,
    action: Option<String>,
}

/// The statechart as data. It does not run. Actors do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    states: BTreeSet<String>,
    events: BTreeSet<String>,
    initial: String,
    /// `from` -> `event` -> candidates in declaration order.
    /// At most one candidate per guard id. [`Actor::send`] takes the first
    /// whose guard passes.
    by_state: BTreeMap<String, BTreeMap<String, Vec<Edge>>>,
    transitions: Vec<Transition>,
}

impl Machine {
    /// Builds a machine.
    ///
    /// Returns `None` when `initial` is not one of `states`, when a transition
    /// names a state or event that was not given, or when two transitions share
    /// the same `from`, `event`, and guard id. An absent guard id matches
    /// another absent guard id. Those cases would not be a single determined step.
    pub fn try_new(
        states: impl IntoIterator<Item = impl Into<String>>,
        events: impl IntoIterator<Item = impl Into<String>>,
        initial: impl Into<String>,
        transitions: impl IntoIterator<Item = Transition>,
    ) -> Option<Self> {
        let states: BTreeSet<String> = states.into_iter().map(Into::into).collect();
        let events: BTreeSet<String> = events.into_iter().map(Into::into).collect();
        let initial = initial.into();
        if !states.contains(&initial) {
            return None;
        }

        let mut by_state: BTreeMap<String, BTreeMap<String, Vec<Edge>>> = BTreeMap::new();
        let mut kept = Vec::new();
        for transition in transitions {
            if !states.contains(&transition.from)
                || !states.contains(&transition.to)
                || !events.contains(&transition.event)
            {
                return None;
            }
            let candidates = by_state
                .entry(transition.from.clone())
                .or_default()
                .entry(transition.event.clone())
                .or_default();
            if candidates.iter().any(|edge| edge.guard == transition.guard) {
                return None;
            }
            candidates.push(Edge {
                to: transition.to.clone(),
                guard: transition.guard.clone(),
                action: transition.action.clone(),
            });
            kept.push(transition);
        }

        Some(Self {
            states,
            events,
            initial,
            by_state,
            transitions: kept,
        })
    }

    pub fn states(&self) -> impl Iterator<Item = &str> {
        self.states.iter().map(String::as_str)
    }

    pub fn events(&self) -> impl Iterator<Item = &str> {
        self.events.iter().map(String::as_str)
    }

    pub fn initial(&self) -> &str {
        &self.initial
    }

    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// The target of the first transition declared from `state` on `event`.
    ///
    /// Guard selection happens in [`Actor::send`], not here. When several
    /// candidates were declared, this is only the first of them.
    pub fn target(&self, state: &str, event: &str) -> Option<&str> {
        self.candidates(state, event)
            .and_then(|edges| edges.first())
            .map(|edge| edge.to.as_str())
    }

    fn candidates(&self, state: &str, event: &str) -> Option<&[Edge]> {
        self.by_state
            .get(state)
            .and_then(|events| events.get(event))
            .map(Vec::as_slice)
    }
}

/// A running machine. The machine data is borrowed. This value holds the current state,
/// the guards and actions registered by id, and the action log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Actor<'a> {
    machine: &'a Machine,
    state: String,
    guards: BTreeMap<String, Guard>,
    actions: BTreeMap<String, Action>,
    action_log: Vec<ActionRecord>,
}

impl<'a> Actor<'a> {
    pub fn start(machine: &'a Machine) -> Self {
        Self {
            machine,
            state: machine.initial.clone(),
            guards: BTreeMap::new(),
            actions: BTreeMap::new(),
            action_log: Vec::new(),
        }
    }

    pub fn state(&self) -> &str {
        &self.state
    }

    pub fn machine(&self) -> &'a Machine {
        self.machine
    }

    /// Registers `guard` under `id`. A later registration for the same id replaces it.
    pub fn register_guard(&mut self, id: impl Into<String>, guard: Guard) {
        self.guards.insert(id.into(), guard);
    }

    /// Registers `action` under `id`. A later registration for the same id replaces it.
    pub fn register_action(&mut self, id: impl Into<String>, action: Action) {
        self.actions.insert(id.into(), action);
    }

    /// Actions that ran, in order. Empty when no send has taken a transition that changed state.
    pub fn action_log(&self) -> &[ActionRecord] {
        &self.action_log
    }

    /// Moves on `event` when the current state has a candidate whose guard passes.
    ///
    /// Candidates are the transitions declared from the current state on `event`,
    /// in that order. A transition with no guard id is eligible. A guard id that
    /// was not registered is not eligible. A guard that returns `false` is not
    /// eligible. The first eligible candidate is taken. There is no parent state
    /// on a flat machine, so if none are eligible the event is refused.
    ///
    /// Returns `true` when a transition fired. Returns `false` when the event is
    /// refused. A refusal does not change the state, does not run an action, and
    /// does not panic.
    ///
    /// When the state changes, the transition's action runs after the change.
    /// The action sees a configuration that contains only the new state. A
    /// missing action id, or an action id that was not registered, runs nothing.
    /// An action cannot replace the actor's state.
    #[must_use]
    pub fn send(&mut self, event: &str) -> bool {
        let chosen = {
            let current = Configuration::from_states(BTreeSet::from([self.state.clone()]));
            let Some(edges) = self.machine.candidates(&self.state, event) else {
                return false;
            };
            let mut chosen = None;
            for edge in edges {
                if statechart::guard_allows(edge.guard.as_deref(), &current, event, &self.guards) {
                    chosen = Some((edge.to.clone(), edge.action.clone()));
                    break;
                }
            }
            chosen
        };
        let Some((to, action)) = chosen else {
            return false;
        };
        let from = std::mem::replace(&mut self.state, to);
        if self.state != from {
            self.run_action(action, &from, event);
        }
        true
    }

    fn run_action(&mut self, action: Option<String>, from: &str, event: &str) {
        let Some(action_id) = action else {
            return;
        };
        let Some(callback) = self.actions.get(&action_id).copied() else {
            return;
        };
        let record = ActionRecord {
            action: action_id,
            from: from.to_string(),
            event: event.to_string(),
            to: self.state.clone(),
            configuration: Configuration::from_states(BTreeSet::from([self.state.clone()])),
        };
        callback(
            &record.from,
            &record.event,
            &record.to,
            &record.configuration,
        );
        self.action_log.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Machine {
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
        .expect("sample machine")
    }

    #[test]
    fn refused_event_leaves_state() {
        let machine = sample();
        let mut actor = Actor::start(&machine);
        assert_eq!(actor.state(), "idle");
        assert!(!actor.send("fail"));
        assert_eq!(actor.state(), "idle");
        assert!(!actor.send("no-such-event"));
        assert_eq!(actor.state(), "idle");
        assert!(actor.send("start"));
        assert_eq!(actor.state(), "run");
        assert!(!actor.send("start"));
        assert_eq!(actor.state(), "run");
    }

    #[test]
    fn duplicate_or_unknown_is_not_a_machine() {
        assert!(Machine::try_new(
            ["idle", "run"],
            ["start"],
            "missing",
            [Transition::new("idle", "start", "run")],
        )
        .is_none());
        assert!(Machine::try_new(
            ["idle", "run"],
            ["start"],
            "idle",
            [
                Transition::new("idle", "start", "run"),
                Transition::new("idle", "start", "idle"),
            ],
        )
        .is_none());
        assert!(Machine::try_new(
            ["idle"],
            ["start"],
            "idle",
            [Transition::new("idle", "start", "run")],
        )
        .is_none());
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
    }
}
