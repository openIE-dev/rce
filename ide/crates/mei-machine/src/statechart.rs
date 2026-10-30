//! Compound states and parallel regions.
//!
//! A [`Statechart`] is data, like [`crate::Machine`]. A [`ChartActor`] holds the
//! active configuration. [`ChartActor::send`] either applies every enabled
//! transition or keeps the configuration as it was: the event does not apply.
//!
//! A transition may name a guard id and an action id. Both are registered on
//! the actor, not stored as functions in the chart. A guard is a pure function
//! of the configuration before the send and the event. No guard id means the
//! transition is eligible. A guard id that was not registered does not take
//! that transition. A guard that returns false does not take it either. The
//! next transition declared on that same state for the same event is tried.
//! When none of those pass, the event bubbles to the parent the way an event
//! with no transition does. If nothing is taken, the send does not apply and the
//! configuration is unchanged.
//!
//! An action runs only after a taken transition has changed the configuration.
//! Every action for that send runs after the new configuration is committed,
//! in source-state order. The action receives the source state, the event,
//! the target state, and the committed configuration. It cannot change the
//! configuration.
//!
//! A compound or atomic state may name one child chart. Entering that state
//! starts the child at its initial configuration. Leaving the state stops the
//! child; it does not keep running. Entering the state again starts a new
//! child at that initial configuration. [`ChartActor::send_to_child`] delivers
//! an event to the active child and to no other actor. If no child is active,
//! the send does not apply and the parent configuration is unchanged. An event
//! that does not apply to the child changes neither actor.

use std::collections::{BTreeMap, BTreeSet};

use crate::Transition;

/// One state in a [`Statechart`].
///
/// Names live on the parent map (and on the root passed to [`Statechart::try_new`]).
/// The same name cannot be used twice in one chart.
///
/// An atomic or compound state may name one child chart in `invoke`. While that
/// state is active, the chart runs as a child actor. A parallel state cannot
/// invoke. [`Statechart::try_new`] returns `None` if it names a chart, or if two
/// named charts could be active together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateNode {
    /// No children. Only this state's own transitions can handle an event here.
    Atomic {
        /// Child chart started when this state becomes active. `None` invokes nothing.
        invoke: Option<Statechart>,
    },
    /// Child states, with exactly one of them active. Entering this state enters `initial`.
    Compound {
        initial: String,
        children: Vec<(String, StateNode)>,
        /// Child chart started when this state becomes active. `None` invokes nothing.
        invoke: Option<Statechart>,
    },
    /// Two or more regions, all active at once.
    Parallel {
        regions: Vec<(String, StateNode)>,
        /// Not a legal invoke. [`Statechart::try_new`] returns `None` when this is `Some`.
        invoke: Option<Statechart>,
    },
}

impl StateNode {
    pub fn atomic() -> Self {
        Self::Atomic { invoke: None }
    }

    pub fn compound<I, N>(initial: impl Into<String>, children: I) -> Self
    where
        I: IntoIterator<Item = (N, StateNode)>,
        N: Into<String>,
    {
        Self::Compound {
            initial: initial.into(),
            children: children
                .into_iter()
                .map(|(name, node)| (name.into(), node))
                .collect(),
            invoke: None,
        }
    }

    pub fn parallel<I, N>(regions: I) -> Self
    where
        I: IntoIterator<Item = (N, StateNode)>,
        N: Into<String>,
    {
        Self::Parallel {
            regions: regions
                .into_iter()
                .map(|(name, node)| (name.into(), node))
                .collect(),
            invoke: None,
        }
    }

    /// Names `chart` as the one child chart invoked while this state is active.
    ///
    /// Entering the state starts `chart` at its initial configuration. Leaving
    /// the state stops that child. A later entry starts a new child. A parallel
    /// state may be given a chart here; [`Statechart::try_new`] rejects it.
    pub fn with_invoke(mut self, chart: Statechart) -> Self {
        let invoke = match &mut self {
            Self::Atomic { invoke }
            | Self::Compound { invoke, .. }
            | Self::Parallel { invoke, .. } => invoke,
        };
        *invoke = Some(chart);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Atomic,
    Compound,
    Parallel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    kind: Kind,
    parent: Option<String>,
    /// Child or region names, sorted.
    children: Vec<String>,
    /// Set only for [`Kind::Compound`].
    initial: Option<String>,
    /// Child chart while this state is active. Never set on a parallel state.
    invoke: Option<Box<Statechart>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step {
    from: String,
    to: String,
    action: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge {
    to: String,
    guard: Option<String>,
    action: Option<String>,
}

/// The active states of a [`ChartActor`], including ancestors.
///
/// Equality is the snapshot: two configurations are equal when the same states are active.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Configuration {
    states: BTreeSet<String>,
}

impl Configuration {
    pub(crate) fn from_states(states: BTreeSet<String>) -> Self {
        Self { states }
    }

    pub fn contains(&self, state: &str) -> bool {
        self.states.contains(state)
    }

    pub fn states(&self) -> impl Iterator<Item = &str> {
        self.states.iter().map(String::as_str)
    }
}

/// Pure predicate registered by id on an actor.
///
/// It is a function pointer so it cannot capture hidden state. The arguments
/// are the configuration before the send chooses a transition, and the event.
pub type Guard = fn(&Configuration, &str) -> bool;

/// Callback registered by id on an actor.
///
/// It is a function pointer so it cannot capture the actor. `from` is the
/// source of the transition that was taken, `event` is the event, and `to` is
/// the target. `configuration` is the configuration after every transition for
/// this send has been committed. The reference is not mutable, so the action
/// cannot change it.
pub type Action = fn(&str, &str, &str, &Configuration);

/// One action the actor ran, in the order it ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionRecord {
    pub action: String,
    pub from: String,
    pub event: String,
    pub to: String,
    /// Configuration the action was shown. This is the configuration after the send.
    pub configuration: Configuration,
}

pub(crate) fn guard_allows(
    guard: Option<&str>,
    configuration: &Configuration,
    event: &str,
    guards: &BTreeMap<String, Guard>,
) -> bool {
    match guard {
        None => true,
        Some(id) => match guards.get(id) {
            Some(guard) => guard(configuration, event),
            None => false,
        },
    }
}

struct Decision<'a> {
    active: &'a BTreeSet<String>,
    event: &'a str,
    configuration: &'a Configuration,
    guards: &'a BTreeMap<String, Guard>,
}

/// Hierarchical statechart. It does not run. [`ChartActor`] does.
///
/// Compound states have one active child. Parallel states have every region active.
/// An atomic or compound state may name one invoked chart. That chart is not a
/// child in this tree. It runs only while the naming state is active.
/// Transitions are the flat [`Transition`] type: one source, one event, one target,
/// plus optional guard and action ids. Several transitions may share a source and
/// an event when their guard ids differ. [`ChartActor::send`] tries them in the
/// order they were given. A transition whose source and target sit in different
/// regions of the same parallel state is rejected. That step would not leave a
/// single determined configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statechart {
    root: String,
    nodes: BTreeMap<String, Node>,
    events: BTreeSet<String>,
    by_state: BTreeMap<String, BTreeMap<String, Vec<Edge>>>,
    transitions: Vec<Transition>,
    initial: BTreeSet<String>,
}

impl Statechart {
    /// Builds a statechart.
    ///
    /// Returns `None` when the tree is not a statechart: a duplicated name, a compound
    /// state with no children or an `initial` that is not a child, a parallel state with
    /// fewer than two regions, a transition that names an unknown state or event, two
    /// transitions from the same state on the same event with the same guard id (an
    /// absent guard id counts), or a transition that crosses from one parallel region
    /// into another. It is also `None` when a parallel state names an invoked chart,
    /// or when more than one invoked chart could be active at the same time: a state
    /// and a descendant both name one, or two regions of one parallel state do.
    pub fn try_new(
        root: impl Into<String>,
        node: StateNode,
        events: impl IntoIterator<Item = impl Into<String>>,
        transitions: impl IntoIterator<Item = Transition>,
    ) -> Option<Self> {
        let root = root.into();
        let mut nodes = BTreeMap::new();
        insert_tree(root.clone(), &node, None, &mut nodes)?;
        if !invoke_tree_ok(&nodes, &root) {
            return None;
        }

        let events: BTreeSet<String> = events.into_iter().map(Into::into).collect();
        let mut by_state: BTreeMap<String, BTreeMap<String, Vec<Edge>>> = BTreeMap::new();
        let mut kept = Vec::new();
        for transition in transitions {
            if !nodes.contains_key(&transition.from)
                || !nodes.contains_key(&transition.to)
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

        let chart = Self {
            root,
            nodes,
            events,
            by_state,
            transitions: kept,
            initial: BTreeSet::new(),
        };
        if chart
            .transitions
            .iter()
            .any(|transition| chart.crosses_parallel(&transition.from, &transition.to))
        {
            return None;
        }

        let initial = chart.seed();
        if !chart.invariant(&initial) {
            return None;
        }
        Some(Self { initial, ..chart })
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    pub fn states(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(String::as_str)
    }

    pub fn events(&self) -> impl Iterator<Item = &str> {
        self.events.iter().map(String::as_str)
    }

    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    pub fn initial_configuration(&self) -> Configuration {
        Configuration {
            states: self.initial.clone(),
        }
    }

    /// The child chart named by `state`, if that state invokes one.
    pub fn invoked(&self, state: &str) -> Option<&Statechart> {
        self.nodes.get(state)?.invoke.as_deref()
    }

    fn seed(&self) -> BTreeSet<String> {
        let mut active = BTreeSet::new();
        active.insert(self.root.clone());
        self.complete(&self.root, &mut active);
        active
    }

    fn complete(&self, state: &str, active: &mut BTreeSet<String>) {
        match self.kind(state) {
            Kind::Atomic => {}
            Kind::Compound => {
                let child = self.nodes[state]
                    .initial
                    .as_deref()
                    .expect("compound state has an initial child");
                active.insert(child.to_string());
                self.complete(child, active);
            }
            Kind::Parallel => {
                for region in self.child_names(state) {
                    active.insert(region.clone());
                    self.complete(region, active);
                }
            }
        }
    }

    fn kind(&self, state: &str) -> Kind {
        self.nodes[state].kind
    }

    fn parent<'a>(&'a self, state: &str) -> Option<&'a str> {
        self.nodes[state].parent.as_deref()
    }

    fn child_names(&self, state: &str) -> &[String] {
        &self.nodes[state].children
    }

    fn ancestor_chain<'a>(&'a self, state: &'a str) -> Vec<&'a str> {
        let mut chain = Vec::new();
        let mut current = state;
        chain.push(current);
        while let Some(parent) = self.parent(current) {
            current = parent;
            chain.push(current);
        }
        chain.reverse();
        chain
    }

    fn lca<'a>(&'a self, left: &'a str, right: &'a str) -> &'a str {
        let left_chain = self.ancestor_chain(left);
        let right_chain = self.ancestor_chain(right);
        let mut shared = left_chain[0];
        for (left_state, right_state) in left_chain.iter().zip(right_chain.iter()) {
            if left_state != right_state {
                break;
            }
            shared = left_state;
        }
        shared
    }

    fn is_strict_descendant(&self, state: &str, ancestor: &str) -> bool {
        let mut current = self.parent(state);
        while let Some(parent) = current {
            if parent == ancestor {
                return true;
            }
            current = self.parent(parent);
        }
        false
    }

    fn contains(&self, ancestor: &str, state: &str) -> bool {
        ancestor == state || self.is_strict_descendant(state, ancestor)
    }

    /// True when `from` and `to` are in different regions of one parallel state.
    fn crosses_parallel(&self, from: &str, to: &str) -> bool {
        if from == to {
            return false;
        }
        let shared = self.lca(from, to);
        self.kind(shared) == Kind::Parallel && shared != from && shared != to
    }

    /// The one active state that names a child chart, if there is one.
    ///
    /// At most one state in a legal configuration names a chart. If that
    /// invariant is broken, the lexicographically first name is used so the
    /// choice stays determined.
    fn active_invoker<'b>(&'b self, active: &'b BTreeSet<String>) -> Option<&'b str> {
        let mut found = None;
        for state in active {
            if self.nodes[state].invoke.is_some() {
                debug_assert!(found.is_none(), "two active states invoke a child");
                if found.is_none() {
                    found = Some(state.as_str());
                }
            }
        }
        found
    }

    fn active_child<'a>(&'a self, compound: &str, active: &BTreeSet<String>) -> &'a str {
        let mut found = None;
        for child in self.child_names(compound) {
            if active.contains(child) {
                assert!(
                    found.is_none(),
                    "compound state {compound} has two active children"
                );
                found = Some(child.as_str());
            }
        }
        found.unwrap_or_else(|| panic!("compound state {compound} has no active child"))
    }

    fn local_step(&self, state: &str, decision: &Decision<'_>) -> Option<Step> {
        let edges = self
            .by_state
            .get(state)
            .and_then(|events| events.get(decision.event))?;
        for edge in edges {
            if guard_allows(
                edge.guard.as_deref(),
                decision.configuration,
                decision.event,
                decision.guards,
            ) {
                return Some(Step {
                    from: state.to_string(),
                    to: edge.to.clone(),
                    action: edge.action.clone(),
                });
            }
        }
        None
    }

    /// Transitions enabled inside `state`, not including ancestors of `state`.
    ///
    /// A compound state asks its active child first. A parallel state asks every
    /// region. The state's own transition is used only when nothing inside took the event.
    /// On one state, candidates for the event are tried in declaration order.
    /// A candidate whose guard is missing or returns false is skipped.
    fn enabled(&self, state: &str, decision: &Decision<'_>) -> Vec<Step> {
        match self.kind(state) {
            Kind::Atomic => self.local_step(state, decision).into_iter().collect(),
            Kind::Compound => {
                let child = self.active_child(state, decision.active);
                let inner = self.enabled(child, decision);
                if !inner.is_empty() {
                    inner
                } else {
                    self.local_step(state, decision).into_iter().collect()
                }
            }
            Kind::Parallel => {
                let mut steps = Vec::new();
                for region in self.child_names(state) {
                    steps.extend(self.enabled(region, decision));
                }
                if steps.is_empty() {
                    self.local_step(state, decision).into_iter().collect()
                } else {
                    steps
                }
            }
        }
    }

    fn exit_set(&self, domain: &str, active: &BTreeSet<String>) -> BTreeSet<String> {
        active
            .iter()
            .filter(|state| self.is_strict_descendant(state, domain))
            .cloned()
            .collect()
    }

    /// States left by `steps`, including a state the same steps enter again.
    fn states_exited(&self, active: &BTreeSet<String>, steps: &[Step]) -> BTreeSet<String> {
        let mut exited = BTreeSet::new();
        for step in steps {
            let domain = self.lca(&step.from, &step.to);
            exited.extend(self.exit_set(domain, active));
        }
        exited
    }

    fn enter_set(&self, domain: &str, to: &str) -> BTreeSet<String> {
        let mut enter = BTreeSet::new();
        if domain == to {
            self.complete(to, &mut enter);
            return enter;
        }
        let path = self.path_below(domain, to);
        for (index, state) in path.iter().enumerate() {
            enter.insert(state.clone());
            let is_target = index + 1 == path.len();
            if is_target {
                self.complete(state, &mut enter);
            } else if self.kind(state) == Kind::Parallel {
                self.fill_other_regions(state, to, &mut enter);
            }
        }
        enter
    }

    fn path_below(&self, domain: &str, to: &str) -> Vec<String> {
        let mut upward = Vec::new();
        let mut current = to.to_string();
        if current == domain {
            return Vec::new();
        }
        upward.push(current.clone());
        while let Some(parent) = self.parent(&current) {
            if parent == domain {
                return {
                    upward.reverse();
                    upward
                };
            }
            upward.push(parent.to_string());
            current = parent.to_string();
        }
        panic!("transition target {to} is not under {domain}");
    }

    fn fill_other_regions(&self, parallel: &str, to: &str, enter: &mut BTreeSet<String>) {
        for region in self.child_names(parallel) {
            if self.contains(region, to) {
                continue;
            }
            enter.insert(region.clone());
            self.complete(region, enter);
        }
    }

    fn select(
        &self,
        active: &BTreeSet<String>,
        event: &str,
        guards: &BTreeMap<String, Guard>,
    ) -> Option<(BTreeSet<String>, Vec<Step>)> {
        if !self.events.contains(event) {
            return None;
        }
        let configuration = Configuration::from_states(active.clone());
        let decision = Decision {
            active,
            event,
            configuration: &configuration,
            guards,
        };
        let mut steps = self.enabled(&self.root, &decision);
        if steps.is_empty() {
            return None;
        }
        steps.sort_by(|left, right| left.from.cmp(&right.from).then(left.to.cmp(&right.to)));

        let mut claimed_exit = BTreeSet::new();
        let mut claimed_enter = BTreeSet::new();
        for step in &steps {
            let domain = self.lca(&step.from, &step.to);
            let exit = self.exit_set(domain, active);
            let enter = self.enter_set(domain, &step.to);
            if overlaps(&exit, &claimed_exit)
                || overlaps(&enter, &claimed_enter)
                || overlaps(&exit, &claimed_enter)
                || overlaps(&enter, &claimed_exit)
            {
                // Two regions enabled a step, but the steps do not stay in their
                // own regions. The whole send does not apply rather than drop one of them.
                return None;
            }
            claimed_exit.extend(exit);
            claimed_enter.extend(enter);
        }

        let mut next = active.clone();
        for state in &claimed_exit {
            next.remove(state);
        }
        next.extend(claimed_enter);
        if !self.invariant(&next) {
            return None;
        }
        Some((next, steps))
    }

    fn invariant(&self, active: &BTreeSet<String>) -> bool {
        if !active.contains(&self.root) {
            return false;
        }
        for state in active {
            if let Some(parent) = self.parent(state) {
                if !active.contains(parent) {
                    return false;
                }
            }
            match self.kind(state) {
                Kind::Atomic => {
                    if !self.child_names(state).is_empty() {
                        return false;
                    }
                }
                Kind::Compound => {
                    let count = self
                        .child_names(state)
                        .iter()
                        .filter(|child| active.contains(*child))
                        .count();
                    if count != 1 {
                        return false;
                    }
                }
                Kind::Parallel => {
                    if !self
                        .child_names(state)
                        .iter()
                        .all(|region| active.contains(region))
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn overlaps(left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
    left.intersection(right).next().is_some()
}

fn boxed_invoke(invoke: &Option<Statechart>) -> Option<Box<Statechart>> {
    invoke.clone().map(Box::new)
}

fn subtree_names_invoke(nodes: &BTreeMap<String, Node>, state: &str) -> bool {
    let node = &nodes[state];
    node.invoke.is_some()
        || node
            .children
            .iter()
            .any(|child| subtree_names_invoke(nodes, child))
}

/// False when two invoked charts could be active together.
///
/// A state and its descendant are active together. Two regions of one parallel
/// state are active together. Children of one compound state are not.
fn invoke_tree_ok(nodes: &BTreeMap<String, Node>, state: &str) -> bool {
    let node = &nodes[state];
    let descendant_invokes = node
        .children
        .iter()
        .any(|child| subtree_names_invoke(nodes, child));
    if node.invoke.is_some() && descendant_invokes {
        return false;
    }
    if node.kind == Kind::Parallel {
        let regions = node
            .children
            .iter()
            .filter(|child| subtree_names_invoke(nodes, child))
            .count();
        if regions > 1 {
            return false;
        }
    }
    node.children
        .iter()
        .all(|child| invoke_tree_ok(nodes, child))
}

fn insert_tree(
    name: String,
    node: &StateNode,
    parent: Option<String>,
    nodes: &mut BTreeMap<String, Node>,
) -> Option<()> {
    if nodes.contains_key(&name) {
        return None;
    }
    match node {
        StateNode::Atomic { invoke } => {
            nodes.insert(
                name,
                Node {
                    kind: Kind::Atomic,
                    parent,
                    children: Vec::new(),
                    initial: None,
                    invoke: boxed_invoke(invoke),
                },
            );
            Some(())
        }
        StateNode::Compound {
            initial,
            children,
            invoke,
        } => {
            let child_names = unique_child_names(children)?;
            if children.is_empty() || !child_names.iter().any(|child| child == initial) {
                return None;
            }
            nodes.insert(
                name.clone(),
                Node {
                    kind: Kind::Compound,
                    parent,
                    children: child_names,
                    initial: Some(initial.clone()),
                    invoke: boxed_invoke(invoke),
                },
            );
            for (child_name, child) in children {
                insert_tree(child_name.clone(), child, Some(name.clone()), nodes)?;
            }
            Some(())
        }
        StateNode::Parallel { regions, invoke } => {
            if invoke.is_some() || regions.len() < 2 {
                return None;
            }
            let child_names = unique_child_names(regions)?;
            nodes.insert(
                name.clone(),
                Node {
                    kind: Kind::Parallel,
                    parent,
                    children: child_names,
                    initial: None,
                    invoke: None,
                },
            );
            for (region_name, region) in regions {
                insert_tree(region_name.clone(), region, Some(name.clone()), nodes)?;
            }
            Some(())
        }
    }
}

fn unique_child_names(children: &[(String, StateNode)]) -> Option<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut names = Vec::with_capacity(children.len());
    for (name, _) in children {
        if !seen.insert(name.clone()) {
            return None;
        }
        names.push(name.clone());
    }
    names.sort();
    Some(names)
}

/// A running [`Statechart`]. The chart is borrowed. This value holds the active
/// configuration, the guards and actions registered by id, the action log, and
/// at most one invoked child.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChartActor<'a> {
    chart: &'a Statechart,
    active: BTreeSet<String>,
    guards: BTreeMap<String, Guard>,
    actions: BTreeMap<String, Action>,
    action_log: Vec<ActionRecord>,
    invocation: Option<Box<Invocation<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Invocation<'a> {
    /// State that started this child. The child keeps running only while this state stays active.
    state: String,
    actor: ChartActor<'a>,
}

impl<'a> ChartActor<'a> {
    pub fn start(chart: &'a Statechart) -> Self {
        let mut actor = Self {
            chart,
            active: chart.initial.clone(),
            guards: BTreeMap::new(),
            actions: BTreeMap::new(),
            action_log: Vec::new(),
            invocation: None,
        };
        actor.sync_invocation(&BTreeSet::new());
        debug_assert!(chart.invariant(&actor.active));
        actor
    }

    pub fn chart(&self) -> &'a Statechart {
        self.chart
    }

    /// Active states of this chart, including ancestors.
    ///
    /// States of an invoked child are not included. Those are on [`ChartActor::child`].
    pub fn configuration(&self) -> Configuration {
        Configuration::from_states(self.active.clone())
    }

    /// The running child, if a state in the active configuration names a chart.
    ///
    /// Guards and actions on this actor are not copied to the child. Register
    /// them on the child when it is running.
    pub fn child(&self) -> Option<&ChartActor<'a>> {
        self.invocation.as_ref().map(|invocation| &invocation.actor)
    }

    /// The running child, if a state in the active configuration names a chart.
    pub fn child_mut(&mut self) -> Option<&mut ChartActor<'a>> {
        self.invocation
            .as_mut()
            .map(|invocation| &mut invocation.actor)
    }

    /// The state that owns the active child, if a child is running.
    pub fn invoking_state(&self) -> Option<&str> {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.state.as_str())
    }

    pub fn is_active(&self, state: &str) -> bool {
        self.active.contains(state)
    }

    /// Registers `guard` under `id`. A later registration for the same id replaces it.
    pub fn register_guard(&mut self, id: impl Into<String>, guard: Guard) {
        self.guards.insert(id.into(), guard);
    }

    /// Registers `action` under `id`. A later registration for the same id replaces it.
    pub fn register_action(&mut self, id: impl Into<String>, action: Action) {
        self.actions.insert(id.into(), action);
    }

    /// Actions that ran, in order. A send that does not apply adds nothing.
    pub fn action_log(&self) -> &[ActionRecord] {
        &self.action_log
    }

    /// Applies `event`, or keeps the configuration when it does not apply.
    ///
    /// The innermost active state is considered first. On that state, each
    /// transition on `event` is a candidate, in declaration order. A candidate
    /// is taken only when it has no guard id, or its registered guard returns
    /// true. A missing guard id skips that candidate. When no candidate on the
    /// state passes, the parent is considered the same way an event with no
    /// transition bubbles. In a parallel state the event is offered to every
    /// region. A region that takes nothing stays. Regions that take a transition
    /// move together.
    ///
    /// The new configuration is committed before any action runs. Actions for
    /// the taken transitions then run in source-state order. Each action sees
    /// that committed configuration and cannot change it. A transition with no
    /// action id, or an action id that was not registered, runs nothing. If the
    /// taken transitions do not change the configuration, no action runs.
    ///
    /// Entering a state that names a child chart starts that child at its
    /// initial configuration. Leaving that state drops the child. A transition
    /// that exits the naming state and enters it again drops the child and
    /// starts a new one at the initial configuration. A transition that does
    /// not exit the naming state leaves the running child as it is. The child
    /// is not part of this configuration. [`ChartActor::send_to_child`] is how
    /// an event is delivered to it. An event that does not apply does not start or
    /// stop a child.
    ///
    /// Returns `false` when nothing on the active path is taken, or when the
    /// enabled transitions would not leave one determined configuration. The
    /// active states do not change in that case, and no action runs.
    #[must_use]
    pub fn send(&mut self, event: &str) -> bool {
        let Some((next, steps)) = self.chart.select(&self.active, event, &self.guards) else {
            return false;
        };
        let changed = next != self.active;
        let exited = self.chart.states_exited(&self.active, &steps);
        self.active = next;
        self.sync_invocation(&exited);
        if changed {
            for step in steps {
                self.run_action(step, event);
            }
        }
        debug_assert!(self.chart.invariant(&self.active));
        true
    }

    /// Delivers `event` to the active invoked child.
    ///
    /// Returns `false` when no child is active, or when the event does not
    /// apply to the child. The child configuration and this configuration then
    /// stay as they were. A child that accepts the event changes only the child.
    /// This configuration does not change, and this actor's actions do not run.
    ///
    /// The event is not offered to a child of that child. Send to that actor
    /// through [`ChartActor::child_mut`].
    #[must_use]
    pub fn send_to_child(&mut self, event: &str) -> bool {
        match self.invocation.as_mut() {
            Some(invocation) => invocation.actor.send(event),
            None => false,
        }
    }

    /// Stops or starts the invoked child so it matches the active naming state.
    ///
    /// `exited` is the states this transition left, including a state that was
    /// entered again by the same transition. Leaving the naming state drops the
    /// child even when that state is active again afterwards.
    fn sync_invocation(&mut self, exited: &BTreeSet<String>) {
        let chart = self.chart;
        let next_state = chart.active_invoker(&self.active).map(str::to_string);
        let current = self
            .invocation
            .as_ref()
            .map(|invocation| invocation.state.clone());
        let same = match (&next_state, &current) {
            (Some(next), Some(previous)) => next == previous && !exited.contains(previous),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        let child_chart = next_state.as_deref().and_then(|state| chart.invoked(state));
        self.invocation = match (next_state, child_chart) {
            (Some(state), Some(child_chart)) => Some(Box::new(Invocation {
                actor: ChartActor::start(child_chart),
                state,
            })),
            (None, _) => None,
            (Some(state), None) => panic!("state {state} invokes, but the chart is missing"),
        };
    }

    fn run_action(&mut self, step: Step, event: &str) {
        let Some(action_id) = step.action else {
            return;
        };
        let Some(callback) = self.actions.get(&action_id).copied() else {
            return;
        };
        let record = ActionRecord {
            action: action_id,
            from: step.from,
            event: event.to_string(),
            to: step.to,
            configuration: self.configuration(),
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
