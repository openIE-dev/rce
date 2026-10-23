//! Animation engine for Photon.
//!
//! Two flavours:
//!
//! - **Implicit** — `view.animation(spec, value:)` interpolates ANY animatable
//!   modifier on `view` between old and new bounds whenever the keyed `value`
//!   changes.
//! - **Explicit** — `with_animation(spec, || state.set(new))` tags the dirty
//!   batch so every dependent view interpolates the change.
//!
//! All time math is in milliseconds; the host calls
//! [`AnimationDriver::tick`] every frame and the engine reports each
//! in-flight animation's progress via [`AnimationDriver::progress`].
//!
//! The curve evaluators are pure-function — given the same `t ∈ [0, 1]` they
//! always return the same value. Springs collapse the closed-form damped
//! oscillator solution: under-damped (`zeta < 1`), critically damped
//! (`zeta == 1`), and over-damped (`zeta > 1`) all return a number that
//! converges to `1.0` as `t → 1`. The spring solver maps `t` into a
//! half-second physics window, which makes the SwiftUI-style defaults
//! (`stiffness=170`, `damping=10`) settle inside the animation duration the
//! caller asked for.

use crate::color::Color;
use crate::layout::{EdgeInsets, Point, Rect, Size};
use crate::render::Transform2D;
use crate::signal::NodeId;
use crate::view::BackgroundFill;
use std::collections::HashMap;

// =============================================================================
// AnimationCurve — interpolation curves shared by every animation surface.
// =============================================================================

/// Animation interpolation curve.
///
/// Linear / EaseIn / EaseOut / EaseInOut are simple closed-form polynomials.
/// `Bezier(c1x, c1y, c2x, c2y)` is a CSS cubic-bezier; control points must
/// land in `[0.0, 1.0]`. `Spring` is a damped harmonic oscillator with
/// physical mass / stiffness / damping / initial velocity.
///
/// `Eq` is intentionally *not* derived because `Bezier` and `Spring` carry
/// `f32` values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AnimationCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    /// Cubic bezier; `(cp1x, cp1y, cp2x, cp2y)` — all in `[0, 1]`.
    Bezier(f32, f32, f32, f32),
    /// Damped spring: `mass / stiffness / damping / initial_velocity`.
    Spring {
        mass: f32,
        stiffness: f32,
        damping: f32,
        initial_velocity: f32,
    },
}

impl AnimationCurve {
    /// Linear: `f(t) = t`.
    pub fn linear() -> Self {
        Self::Linear
    }
    /// Quadratic ease-in.
    pub fn ease_in() -> Self {
        Self::EaseIn
    }
    /// Quadratic ease-out.
    pub fn ease_out() -> Self {
        Self::EaseOut
    }
    /// Symmetric ease-in-out.
    pub fn ease_in_out() -> Self {
        Self::EaseInOut
    }
    /// Cubic bezier convenience constructor.
    pub fn bezier(c1x: f32, c1y: f32, c2x: f32, c2y: f32) -> Self {
        Self::Bezier(c1x, c1y, c2x, c2y)
    }
    /// SwiftUI's default spring: `mass=1`, `stiffness=170`, `damping=10`.
    pub fn spring() -> Self {
        Self::Spring {
            mass: 1.0,
            stiffness: 170.0,
            damping: 10.0,
            initial_velocity: 0.0,
        }
    }
    /// SwiftUI's snappy spring: faster, slightly less bouncy than `spring`.
    pub fn snappy() -> Self {
        Self::Spring {
            mass: 1.0,
            stiffness: 280.0,
            damping: 18.0,
            initial_velocity: 0.0,
        }
    }
    /// SwiftUI's bouncy spring: under-damped, multiple oscillations.
    pub fn bouncy() -> Self {
        Self::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 5.0,
            initial_velocity: 0.0,
        }
    }

    /// Evaluate the curve at progress `t ∈ [0, 1]`. Spring curves treat `t`
    /// as `elapsed_ms / duration_ms`, mapped onto a half-second physics
    /// window so that defaults settle within a frame budget.
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t).powi(2),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::Bezier(c1x, c1y, c2x, c2y) => bezier_y(t, *c1x, *c1y, *c2x, *c2y),
            Self::Spring {
                mass,
                stiffness,
                damping,
                initial_velocity,
            } => spring_value(t, *mass, *stiffness, *damping, *initial_velocity),
        }
    }
}

/// Cubic bezier evaluator: solves x → t via Newton-Raphson, then evaluates
/// the y-axis cubic at the recovered `t`.
fn bezier_y(x: f32, c1x: f32, c1y: f32, c2x: f32, c2y: f32) -> f32 {
    let mut t = x;
    for _ in 0..8 {
        let cur_x = bezier_axis(t, c1x, c2x);
        let dx = bezier_d_axis(t, c1x, c2x);
        if dx.abs() < 1e-6 {
            break;
        }
        t -= (cur_x - x) / dx;
    }
    bezier_axis(t, c1y, c2y)
}

fn bezier_axis(t: f32, p1: f32, p2: f32) -> f32 {
    3.0 * (1.0 - t).powi(2) * t * p1 + 3.0 * (1.0 - t) * t * t * p2 + t.powi(3)
}

fn bezier_d_axis(t: f32, p1: f32, p2: f32) -> f32 {
    3.0 * (1.0 - t).powi(2) * p1 + 6.0 * (1.0 - t) * t * (p2 - p1) + 3.0 * t * t * (1.0 - p2)
}

/// Damped-spring closed-form solver. Returns the *position* of a unit-step
/// response normalised so that `f(0) = 0` and `f(t→∞) = 1`.
///
/// We map normalised animation progress `t ∈ [0, 1]` onto a half-second
/// physics window so SwiftUI's default spring (`stiffness=170`, `damping=10`)
/// settles inside the user's nominal duration.
fn spring_value(t: f32, mass: f32, k: f32, c: f32, v0: f32) -> f32 {
    if mass <= 0.0 || k <= 0.0 {
        // Degenerate physics — fall back to linear.
        return t;
    }
    let omega = (k / mass).sqrt();
    let zeta = c / (2.0 * (k * mass).sqrt());
    let t_sec = t * 0.5;

    if (zeta - 1.0).abs() < 1e-4 {
        // Critically damped.
        let exp = (-omega * t_sec).exp();
        1.0 - exp * (1.0 + (omega + v0) * t_sec)
    } else if zeta < 1.0 {
        // Under-damped: the most common case (snappy SwiftUI feel).
        let omega_d = omega * (1.0 - zeta * zeta).sqrt();
        let exp = (-zeta * omega * t_sec).exp();
        1.0 - exp
            * ((omega_d * t_sec).cos() + ((zeta * omega + v0) / omega_d) * (omega_d * t_sec).sin())
    } else {
        // Over-damped: two real exponentials.
        let disc = (zeta * zeta - 1.0).sqrt();
        let s1 = -omega * (zeta - disc);
        let s2 = -omega * (zeta + disc);
        1.0 - (s2 * (s1 * t_sec).exp() - s1 * (s2 * t_sec).exp()) / (s2 - s1)
    }
}

// =============================================================================
// AnimationSpec — the (curve, duration, delay, key) envelope.
// =============================================================================

/// Snapshot of an [`crate::view::Modifier::Animation`] envelope. Carried by
/// [`crate::render::Annotation::animation`] so the GPU bridge or compositor
/// can interpolate values between frames.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AnimationSpec {
    pub curve: AnimationCurve,
    pub duration_ms: u32,
    pub delay_ms: u32,
    pub key: u64,
}

impl AnimationSpec {
    /// Construct a spec with zero delay and key 0.
    pub fn new(curve: AnimationCurve, duration_ms: u32) -> Self {
        Self {
            curve,
            duration_ms,
            delay_ms: 0,
            key: 0,
        }
    }
    /// Add a start delay.
    pub fn delay(self, delay_ms: u32) -> Self {
        Self { delay_ms, ..self }
    }
    /// Tag this spec with a stable identifier (used by the driver to detect
    /// when the same animation is restarted vs. a brand-new one).
    pub fn keyed(self, key: u64) -> Self {
        Self { key, ..self }
    }
}

/// Linear easing for `duration_ms` milliseconds.
pub fn linear(duration_ms: u32) -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::Linear, duration_ms)
}
/// Quadratic ease-in.
pub fn ease_in(duration_ms: u32) -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::EaseIn, duration_ms)
}
/// Quadratic ease-out.
pub fn ease_out(duration_ms: u32) -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::EaseOut, duration_ms)
}
/// Symmetric ease-in-out.
pub fn ease_in_out(duration_ms: u32) -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::EaseInOut, duration_ms)
}
/// SwiftUI's default spring (500 ms target window).
pub fn spring() -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::spring(), 500)
}
/// SwiftUI's snappy spring (350 ms target window).
pub fn snappy() -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::snappy(), 350)
}
/// SwiftUI's bouncy spring (600 ms target window).
pub fn bouncy() -> AnimationSpec {
    AnimationSpec::new(AnimationCurve::bouncy(), 600)
}

// =============================================================================
// TransitionKind — view enter/exit envelopes.
// =============================================================================

/// View-tree transition (enter / exit) used by
/// [`crate::view::Modifier::Transition`]. The driver tracks each node's
/// presence (entering / stable / leaving) and the renderer multiplies in
/// the active transition factor when painting.
#[derive(Clone, Debug)]
pub enum TransitionKind {
    /// No transition — appears / disappears instantly.
    None,
    /// Cross-fade alpha from 0 → 1 on enter and 1 → 0 on exit.
    Opacity,
    /// Slide in from / out toward the given edge.
    Slide { edge: Edge },
    /// Scale from `from` to 1.0 on enter; reverse on exit.
    Scale { from: f32 },
    /// Different transitions for the enter and exit edges.
    Asymmetric {
        enter: Box<TransitionKind>,
        exit: Box<TransitionKind>,
    },
}

/// Cardinal edge — used by [`TransitionKind::Slide`] and other directional
/// modifiers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Edge {
    Top,
    Bottom,
    Leading,
    Trailing,
}

/// Presence of a node within the render tree across consecutive frames.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Presence {
    /// First time the node has appeared — its [`TransitionKind`] enter edge
    /// runs over the next animation window.
    Entering,
    /// Stable — no presence change since the previous frame.
    Stable,
    /// Was present last frame, missing this frame — the exit edge runs.
    Leaving,
}

// =============================================================================
// Animatable — values that can interpolate between two bounds.
// =============================================================================

/// A value that can interpolate between two bounds. Implemented for the
/// numeric / geometric types Photon's modifiers carry.
pub trait Animatable {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self;
}

impl Animatable for f32 {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        from + (to - from) * t
    }
}

impl Animatable for f64 {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        from + (to - from) * t as f64
    }
}

impl Animatable for Color {
    /// Color-managed lerp.
    ///
    /// We convert both endpoints into **linear sRGB** before mixing, so that
    /// `lerp(red, green, 0.5)` produces a luminous yellow rather than the
    /// muddy half-grey you get from gamma-space averaging. The result is
    /// returned in `from`'s space (so a sRGB+sRGB blend stays in sRGB) by
    /// re-encoding the linear mix; if the endpoints disagree on space we
    /// keep linear-sRGB so the compositor can decide what to do.
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        let a = from.to_linear_srgb();
        let b = to.to_linear_srgb();
        let mixed = Color {
            r: f32::lerp(&a.r, &b.r, t),
            g: f32::lerp(&a.g, &b.g, t),
            b: f32::lerp(&a.b, &b.b, t),
            a: f32::lerp(&a.a, &b.a, t),
            space: crate::color::ColorSpace::LinearSrgb,
        };
        // Re-encode into the source's authoring space when they agree —
        // this keeps existing roundtrip tests stable for the common case
        // of two sRGB colours.
        if from.space == to.space && from.space == crate::color::ColorSpace::Srgb {
            Color {
                r: linear_to_srgb(mixed.r),
                g: linear_to_srgb(mixed.g),
                b: linear_to_srgb(mixed.b),
                a: mixed.a,
                space: crate::color::ColorSpace::Srgb,
            }
        } else {
            mixed
        }
    }
}

/// Inverse of the sRGB EOTF used by [`Color::to_linear_srgb`]. Inlined here
/// to keep `photon` free of compositor dependencies.
fn linear_to_srgb(c: f32) -> f32 {
    let s = c.signum();
    let m = c.abs();
    let nl = if m <= 0.003_130_8 {
        12.92 * m
    } else {
        1.055 * m.powf(1.0 / 2.4) - 0.055
    };
    nl * s
}

impl Animatable for Size {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        Self {
            width: f32::lerp(&from.width, &to.width, t),
            height: f32::lerp(&from.height, &to.height, t),
        }
    }
}

impl Animatable for Point {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        Self {
            x: f32::lerp(&from.x, &to.x, t),
            y: f32::lerp(&from.y, &to.y, t),
        }
    }
}

impl Animatable for Rect {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        Self {
            origin: Point::lerp(&from.origin, &to.origin, t),
            size: Size::lerp(&from.size, &to.size, t),
        }
    }
}

impl Animatable for EdgeInsets {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        Self {
            top: f32::lerp(&from.top, &to.top, t),
            right: f32::lerp(&from.right, &to.right, t),
            bottom: f32::lerp(&from.bottom, &to.bottom, t),
            left: f32::lerp(&from.left, &to.left, t),
        }
    }
}

impl Animatable for Transform2D {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        let mut out = [0.0_f32; 6];
        for i in 0..6 {
            out[i] = f32::lerp(&from.0[i], &to.0[i], t);
        }
        Transform2D(out)
    }
}

impl Animatable for BackgroundFill {
    /// Interpolate component-wise when both sides are the same flavour;
    /// otherwise snap to `to` once `t >= 0.5`. Solid colours animate via
    /// per-channel lerp; gradients lerp matching stops, falling back to the
    /// `to` side when the stop counts differ.
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        match (from, to) {
            (BackgroundFill::Color(a), BackgroundFill::Color(b)) => {
                BackgroundFill::Color(Color::lerp(a, b, t))
            }
            (
                BackgroundFill::LinearGradient {
                    angle_deg: a_a,
                    stops: a_s,
                },
                BackgroundFill::LinearGradient {
                    angle_deg: b_a,
                    stops: b_s,
                },
            ) if a_s.len() == b_s.len() => {
                let stops = a_s
                    .iter()
                    .zip(b_s.iter())
                    .map(|((ap, ac), (bp, bc))| (f32::lerp(ap, bp, t), Color::lerp(ac, bc, t)))
                    .collect();
                BackgroundFill::LinearGradient {
                    angle_deg: f32::lerp(a_a, b_a, t),
                    stops,
                }
            }
            (
                BackgroundFill::RadialGradient {
                    center: a_c,
                    stops: a_s,
                },
                BackgroundFill::RadialGradient {
                    center: b_c,
                    stops: b_s,
                },
            ) if a_s.len() == b_s.len() => {
                // Alignment2D is discrete — pick one based on `t`.
                let center = if t < 0.5 { *a_c } else { *b_c };
                let stops = a_s
                    .iter()
                    .zip(b_s.iter())
                    .map(|((ap, ac), (bp, bc))| (f32::lerp(ap, bp, t), Color::lerp(ac, bc, t)))
                    .collect();
                BackgroundFill::RadialGradient { center, stops }
            }
            _ => {
                // Different fill flavours — snap halfway through.
                if t < 0.5 { from.clone() } else { to.clone() }
            }
        }
    }
}

// =============================================================================
// AnimationDriver — per-tree animation runner.
// =============================================================================

/// Entry tracked by [`AnimationDriver`] for a single in-flight animation.
#[derive(Copy, Clone, Debug)]
struct Inflight {
    spec: AnimationSpec,
    started_ms: u64,
    elapsed_ms: u32,
    finished: bool,
}

/// Runs in-flight animations. The host owns one per render tree, ticks it
/// every frame, and queries [`progress`](Self::progress) for the current
/// `t ∈ [0, 1]` on each `(node, key)` pair.
///
/// Each animation completes when `elapsed_ms >= spec.duration_ms`. After
/// completion the entry stays alive until [`gc`](Self::gc) is called so
/// callers can read the final progress one last time.
#[derive(Default)]
pub struct AnimationDriver {
    inflight: HashMap<(NodeId, u64), Inflight>,
    presence: HashMap<NodeId, Presence>,
    last_tick_ms: Option<u64>,
}

impl AnimationDriver {
    /// Construct an empty driver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin (or restart) an animation for `(node, spec.key)`. Calling this
    /// while the same `(node, key)` pair is in-flight resets `elapsed_ms`
    /// to 0 — i.e. SwiftUI semantics where re-binding to a new value
    /// restarts the curve.
    pub fn start(&mut self, node: NodeId, spec: AnimationSpec, now_ms: u64) {
        self.inflight.insert(
            (node, spec.key),
            Inflight {
                spec,
                started_ms: now_ms,
                elapsed_ms: 0,
                finished: false,
            },
        );
    }

    /// Advance time. Returns the set of node IDs that need redraw on this
    /// frame. The last-tick timestamp is tracked internally so the driver
    /// can reason about delta-time across frames.
    pub fn tick(&mut self, now_ms: u64) -> Vec<NodeId> {
        let mut redraw: Vec<NodeId> = Vec::new();
        for ((node, _key), entry) in self.inflight.iter_mut() {
            if entry.finished {
                continue;
            }
            let elapsed = now_ms.saturating_sub(entry.started_ms);
            entry.elapsed_ms = elapsed.min(u32::MAX as u64) as u32;
            if entry.elapsed_ms >= entry.spec.duration_ms.saturating_add(entry.spec.delay_ms) {
                entry.finished = true;
            }
            redraw.push(*node);
        }
        redraw.sort_unstable();
        redraw.dedup();
        self.last_tick_ms = Some(now_ms);
        redraw
    }

    /// Current progress `t ∈ [0, 1]` for `(node, key)`. Returns `None` if
    /// no animation is registered. Honours `delay_ms` (returns `0.0` while
    /// the delay window is active).
    pub fn progress(&self, node: NodeId, key: u64) -> Option<f32> {
        let entry = self.inflight.get(&(node, key))?;
        if entry.spec.duration_ms == 0 {
            return Some(1.0);
        }
        let elapsed = entry.elapsed_ms;
        if elapsed <= entry.spec.delay_ms {
            return Some(0.0);
        }
        let t = (elapsed - entry.spec.delay_ms) as f32 / entry.spec.duration_ms as f32;
        Some(t.clamp(0.0, 1.0))
    }

    /// Curve-evaluated progress `f(t)` for `(node, key)`. Equivalent to
    /// `spec.curve.evaluate(progress(...))` but performs the lookup once.
    pub fn eased_progress(&self, node: NodeId, key: u64) -> Option<f32> {
        let entry = self.inflight.get(&(node, key))?;
        let raw = self.progress(node, key)?;
        Some(entry.spec.curve.evaluate(raw))
    }

    /// Whether `(node, key)` has finished animating. Returns `false` for
    /// unknown pairs.
    pub fn is_finished(&self, node: NodeId, key: u64) -> bool {
        self.inflight
            .get(&(node, key))
            .map(|e| e.finished)
            .unwrap_or(false)
    }

    /// Number of animations currently tracked (finished or not).
    pub fn active_count(&self) -> usize {
        self.inflight.len()
    }

    /// Drop completed animations.
    pub fn gc(&mut self) {
        self.inflight.retain(|_, e| !e.finished);
    }

    /// Most recently observed `now_ms` from [`tick`](Self::tick).
    pub fn last_tick_ms(&self) -> Option<u64> {
        self.last_tick_ms
    }

    /// Update the presence of `node` for this frame. Used by the renderer
    /// to drive [`TransitionKind`] enter/exit envelopes.
    pub fn set_presence(&mut self, node: NodeId, p: Presence) {
        self.presence.insert(node, p);
    }

    /// Lookup the presence recorded for `node`. Defaults to
    /// [`Presence::Stable`] when no explicit state was recorded.
    pub fn presence(&self, node: NodeId) -> Presence {
        self.presence
            .get(&node)
            .copied()
            .unwrap_or(Presence::Stable)
    }

    /// Compute presence by diffing two consecutive node ID lists. Nodes in
    /// `current` but not `previous` are entering; nodes in `previous` but
    /// not `current` are leaving; everything else is stable. The driver's
    /// internal presence table is replaced wholesale.
    pub fn diff_presence(&mut self, previous: &[NodeId], current: &[NodeId]) {
        use std::collections::HashSet;
        let prev_set: HashSet<NodeId> = previous.iter().copied().collect();
        let curr_set: HashSet<NodeId> = current.iter().copied().collect();
        self.presence.clear();
        for n in current {
            if prev_set.contains(n) {
                self.presence.insert(*n, Presence::Stable);
            } else {
                self.presence.insert(*n, Presence::Entering);
            }
        }
        for n in previous {
            if !curr_set.contains(n) {
                self.presence.insert(*n, Presence::Leaving);
            }
        }
    }
}

// =============================================================================
// Transition rendering helpers.
// =============================================================================

/// Resolve an asymmetric transition to the side appropriate for `presence`.
pub fn resolve_transition<'a>(kind: &'a TransitionKind, presence: Presence) -> &'a TransitionKind {
    match (kind, presence) {
        (TransitionKind::Asymmetric { enter, .. }, Presence::Entering) => enter,
        (TransitionKind::Asymmetric { exit, .. }, Presence::Leaving) => exit,
        _ => kind,
    }
}

/// Compute the alpha multiplier for an `Opacity` transition at progress `t`.
/// Returns `Some(alpha)` for opacity transitions, `None` otherwise.
pub fn opacity_for_transition(kind: &TransitionKind, presence: Presence, t: f32) -> Option<f32> {
    let resolved = resolve_transition(kind, presence);
    match (resolved, presence) {
        (TransitionKind::Opacity, Presence::Entering) => Some(t.clamp(0.0, 1.0)),
        (TransitionKind::Opacity, Presence::Leaving) => Some((1.0 - t).clamp(0.0, 1.0)),
        (TransitionKind::Opacity, Presence::Stable) => Some(1.0),
        _ => None,
    }
}

/// Compute the (dx, dy) offset for a `Slide` transition at progress `t`.
/// Returns `Some((dx, dy))` for slide transitions, `None` otherwise.
///
/// `bounds` carries the node's resolved size — slides translate by one full
/// dimension on the appropriate axis.
pub fn slide_for_transition(
    kind: &TransitionKind,
    presence: Presence,
    t: f32,
    bounds: Size,
) -> Option<(f32, f32)> {
    let resolved = resolve_transition(kind, presence);
    let edge = match resolved {
        TransitionKind::Slide { edge } => *edge,
        _ => return None,
    };
    let amount = match presence {
        Presence::Entering => 1.0 - t.clamp(0.0, 1.0),
        Presence::Leaving => t.clamp(0.0, 1.0),
        Presence::Stable => 0.0,
    };
    let (dx, dy) = match edge {
        Edge::Top => (0.0, -bounds.height * amount),
        Edge::Bottom => (0.0, bounds.height * amount),
        Edge::Leading => (-bounds.width * amount, 0.0),
        Edge::Trailing => (bounds.width * amount, 0.0),
    };
    Some((dx, dy))
}

/// Compute the scale factor for a `Scale` transition at progress `t`.
/// Returns `Some(factor)` for scale transitions, `None` otherwise.
pub fn scale_for_transition(kind: &TransitionKind, presence: Presence, t: f32) -> Option<f32> {
    let resolved = resolve_transition(kind, presence);
    let from = match resolved {
        TransitionKind::Scale { from } => *from,
        _ => return None,
    };
    let t = t.clamp(0.0, 1.0);
    let factor = match presence {
        Presence::Entering => from + (1.0 - from) * t,
        Presence::Leaving => 1.0 + (from - 1.0) * t,
        Presence::Stable => 1.0,
    };
    Some(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Curve evaluation
    // -------------------------------------------------------------------------

    #[test]
    fn linear_endpoints_and_midpoint() {
        let c = AnimationCurve::linear();
        assert!((c.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ease_in_endpoints_and_curvature() {
        let c = AnimationCurve::ease_in();
        assert!((c.evaluate(0.0)).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
        // ease_in is below the line at the midpoint.
        assert!(c.evaluate(0.5) < 0.5);
    }

    #[test]
    fn ease_out_endpoints_and_curvature() {
        let c = AnimationCurve::ease_out();
        assert!((c.evaluate(0.0)).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
        assert!(c.evaluate(0.5) > 0.5);
    }

    #[test]
    fn ease_in_out_passes_through_half() {
        let c = AnimationCurve::ease_in_out();
        assert!((c.evaluate(0.0)).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn bezier_linear_via_diagonal_controls() {
        // (0.0, 0.0, 1.0, 1.0) is the linear bezier.
        let c = AnimationCurve::bezier(0.0, 0.0, 1.0, 1.0);
        for x in [0.1_f32, 0.3, 0.5, 0.7, 0.9] {
            let y = c.evaluate(x);
            assert!((y - x).abs() < 1e-2, "y={y} x={x}");
        }
    }

    #[test]
    fn bezier_endpoints() {
        let c = AnimationCurve::bezier(0.42, 0.0, 0.58, 1.0);
        assert!((c.evaluate(0.0)).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spring_default_endpoints_settle_to_one() {
        let c = AnimationCurve::spring();
        assert!((c.evaluate(0.0)).abs() < 1e-3);
        // After one full unit of normalised time the spring should be very
        // close to 1.0 with the default damping.
        let final_v = c.evaluate(1.0);
        assert!((final_v - 1.0).abs() < 0.1, "final={final_v}");
    }

    #[test]
    fn spring_critically_damped_does_not_oscillate() {
        // Pick m=1, k=100 → omega=10, c=2*sqrt(km)=20 → zeta=1.0.
        let c = AnimationCurve::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 20.0,
            initial_velocity: 0.0,
        };
        // Should be monotonically increasing toward 1 — no overshoot.
        let mut prev = c.evaluate(0.0);
        for i in 1..=20 {
            let v = c.evaluate(i as f32 / 20.0);
            assert!(v >= prev - 1e-3, "v={v} prev={prev}");
            prev = v;
        }
    }

    #[test]
    fn spring_overdamped_monotone_increasing() {
        // zeta > 1 — heavily damped.
        let c = AnimationCurve::Spring {
            mass: 1.0,
            stiffness: 50.0,
            damping: 50.0,
            initial_velocity: 0.0,
        };
        let mut prev = c.evaluate(0.0);
        for i in 1..=20 {
            let v = c.evaluate(i as f32 / 20.0);
            assert!(v + 1e-3 >= prev);
            prev = v;
        }
    }

    #[test]
    fn spring_underdamped_can_overshoot() {
        // bouncy preset is under-damped.
        let c = AnimationCurve::bouncy();
        let mut max = 0.0_f32;
        for i in 0..=20 {
            let v = c.evaluate(i as f32 / 20.0);
            if v > max {
                max = v;
            }
        }
        // Bouncy under-damped curves should overshoot 1.0 at least slightly.
        assert!(max > 1.0 - 1e-2, "max={max}");
    }

    #[test]
    fn spring_with_zero_stiffness_falls_back_linearly() {
        let c = AnimationCurve::Spring {
            mass: 1.0,
            stiffness: 0.0,
            damping: 1.0,
            initial_velocity: 0.0,
        };
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn evaluate_clamps_input_range() {
        let c = AnimationCurve::linear();
        assert!((c.evaluate(-10.0)).abs() < 1e-6);
        assert!((c.evaluate(100.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn snappy_preset_is_stiffer_than_default() {
        match AnimationCurve::snappy() {
            AnimationCurve::Spring { stiffness, .. } => assert!(stiffness > 170.0),
            _ => panic!("snappy must be a spring"),
        }
    }

    #[test]
    fn bouncy_preset_is_under_damped() {
        match AnimationCurve::bouncy() {
            AnimationCurve::Spring {
                stiffness,
                damping,
                mass,
                ..
            } => {
                let zeta = damping / (2.0 * (stiffness * mass).sqrt());
                assert!(zeta < 1.0, "zeta={zeta}");
            }
            _ => panic!("bouncy must be a spring"),
        }
    }

    // -------------------------------------------------------------------------
    // AnimationSpec builders
    // -------------------------------------------------------------------------

    #[test]
    fn spec_new_zero_delay_zero_key() {
        let s = AnimationSpec::new(AnimationCurve::Linear, 100);
        assert_eq!(s.duration_ms, 100);
        assert_eq!(s.delay_ms, 0);
        assert_eq!(s.key, 0);
    }

    #[test]
    fn spec_delay_builder_attaches() {
        let s = AnimationSpec::new(AnimationCurve::Linear, 100).delay(50);
        assert_eq!(s.delay_ms, 50);
    }

    #[test]
    fn spec_keyed_builder_attaches() {
        let s = AnimationSpec::new(AnimationCurve::Linear, 100).keyed(7);
        assert_eq!(s.key, 7);
    }

    #[test]
    fn linear_helper_constructs_curve() {
        let s = linear(200);
        assert_eq!(s.duration_ms, 200);
        assert!(matches!(s.curve, AnimationCurve::Linear));
    }

    #[test]
    fn ease_in_helper_constructs_curve() {
        let s = ease_in(150);
        assert!(matches!(s.curve, AnimationCurve::EaseIn));
    }

    #[test]
    fn ease_out_helper_constructs_curve() {
        let s = ease_out(150);
        assert!(matches!(s.curve, AnimationCurve::EaseOut));
    }

    #[test]
    fn ease_in_out_helper_constructs_curve() {
        let s = ease_in_out(150);
        assert!(matches!(s.curve, AnimationCurve::EaseInOut));
    }

    #[test]
    fn spring_helper_default_duration_is_500() {
        let s = spring();
        assert_eq!(s.duration_ms, 500);
        assert!(matches!(s.curve, AnimationCurve::Spring { .. }));
    }

    #[test]
    fn snappy_helper_default_duration_is_350() {
        let s = snappy();
        assert_eq!(s.duration_ms, 350);
    }

    #[test]
    fn bouncy_helper_default_duration_is_600() {
        let s = bouncy();
        assert_eq!(s.duration_ms, 600);
    }

    // -------------------------------------------------------------------------
    // Animatable
    // -------------------------------------------------------------------------

    #[test]
    fn animatable_f32_endpoints_and_midpoint() {
        assert!((f32::lerp(&0.0, &10.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((f32::lerp(&0.0, &10.0, 1.0) - 10.0).abs() < 1e-6);
        assert!((f32::lerp(&0.0, &10.0, 0.5) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn animatable_f64_endpoints() {
        assert!((f64::lerp(&-1.0, &1.0, 0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn animatable_color_per_channel_lerp() {
        // Color-managed lerp: black <-> white at t=0.5 in *linear* sRGB,
        // re-encoded back to sRGB, lands near 0.7354 (linear_to_srgb(0.5)),
        // not the 0.5 you'd get from naive gamma-space averaging.
        let a = Color::rgba(0.0, 0.0, 0.0, 0.0);
        let b = Color::rgba(1.0, 1.0, 1.0, 1.0);
        let mid = Color::lerp(&a, &b, 0.5);
        assert!((mid.r - 0.7354).abs() < 1e-3, "got {}", mid.r);
        // Alpha is mixed linearly (no transfer applied to alpha).
        assert!((mid.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn animatable_size_lerp() {
        let a = Size::new(0.0, 0.0);
        let b = Size::new(100.0, 200.0);
        let mid = Size::lerp(&a, &b, 0.5);
        assert_eq!(mid.width, 50.0);
        assert_eq!(mid.height, 100.0);
    }

    #[test]
    fn animatable_point_lerp() {
        let a = Point::new(-10.0, 0.0);
        let b = Point::new(10.0, 100.0);
        let mid = Point::lerp(&a, &b, 0.5);
        assert!((mid.x).abs() < 1e-6);
        assert!((mid.y - 50.0).abs() < 1e-6);
    }

    #[test]
    fn animatable_rect_lerp() {
        let a = Rect::new(0.0, 0.0, 0.0, 0.0);
        let b = Rect::new(10.0, 20.0, 30.0, 40.0);
        let mid = Rect::lerp(&a, &b, 0.5);
        assert_eq!(mid.size.width, 15.0);
        assert_eq!(mid.origin.x, 5.0);
    }

    #[test]
    fn animatable_edge_insets_lerp() {
        let a = EdgeInsets::all(0.0);
        let b = EdgeInsets::all(8.0);
        let mid = EdgeInsets::lerp(&a, &b, 0.5);
        assert!((mid.top - 4.0).abs() < 1e-6);
        assert!((mid.left - 4.0).abs() < 1e-6);
    }

    #[test]
    fn animatable_transform2d_lerp() {
        let a = Transform2D::identity();
        let b = Transform2D::translate(10.0, 20.0);
        let mid = Transform2D::lerp(&a, &b, 0.5);
        assert!((mid.0[4] - 5.0).abs() < 1e-6);
        assert!((mid.0[5] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn animatable_background_solid_color_lerps() {
        // Linear-space color blending: midpoint of black<->white encodes
        // back to sRGB ≈ 0.7354.
        let a = BackgroundFill::Color(Color::rgb(0.0, 0.0, 0.0));
        let b = BackgroundFill::Color(Color::rgb(1.0, 1.0, 1.0));
        let mid = BackgroundFill::lerp(&a, &b, 0.5);
        match mid {
            BackgroundFill::Color(c) => assert!((c.r - 0.7354).abs() < 1e-3, "got {}", c.r),
            _ => panic!("expected Color"),
        }
    }

    #[test]
    fn animatable_background_mismatched_flavours_snap() {
        let a = BackgroundFill::Color(Color::WHITE);
        let b = BackgroundFill::LinearGradient {
            angle_deg: 0.0,
            stops: vec![(0.0, Color::WHITE), (1.0, Color::BLACK)],
        };
        // Below 0.5 → from; above → to.
        match BackgroundFill::lerp(&a, &b, 0.25) {
            BackgroundFill::Color(_) => {}
            _ => panic!("expected from"),
        }
        match BackgroundFill::lerp(&a, &b, 0.75) {
            BackgroundFill::LinearGradient { .. } => {}
            _ => panic!("expected to"),
        }
    }

    #[test]
    fn animatable_background_linear_gradient_matches_lerp() {
        let a = BackgroundFill::LinearGradient {
            angle_deg: 0.0,
            stops: vec![(0.0, Color::WHITE), (1.0, Color::BLACK)],
        };
        let b = BackgroundFill::LinearGradient {
            angle_deg: 90.0,
            stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
        };
        let mid = BackgroundFill::lerp(&a, &b, 0.5);
        match mid {
            BackgroundFill::LinearGradient { angle_deg, stops } => {
                assert!((angle_deg - 45.0).abs() < 1e-6);
                assert_eq!(stops.len(), 2);
                // White<->black midpoint in linear sRGB, encoded back: ≈0.7354.
                assert!((stops[0].1.r - 0.7354).abs() < 1e-3);
            }
            _ => panic!("expected linear"),
        }
    }

    // -------------------------------------------------------------------------
    // AnimationDriver
    // -------------------------------------------------------------------------

    #[test]
    fn driver_starts_empty() {
        let d = AnimationDriver::new();
        assert_eq!(d.active_count(), 0);
        assert!(d.last_tick_ms().is_none());
    }

    #[test]
    fn driver_start_inserts_inflight() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        assert_eq!(d.active_count(), 1);
        assert!(d.progress(7, 0).is_some());
    }

    #[test]
    fn driver_progress_zero_at_start() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        assert!((d.progress(7, 0).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn driver_progress_half_at_midpoint() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(50);
        assert!((d.progress(7, 0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn driver_progress_one_at_end() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(100);
        assert!((d.progress(7, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!(d.is_finished(7, 0));
    }

    #[test]
    fn driver_progress_clamps_after_end() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(500);
        assert!((d.progress(7, 0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn driver_eased_progress_applies_curve() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::EaseIn, 100), 0);
        d.tick(50);
        // EaseIn(0.5) = 0.25.
        assert!((d.eased_progress(7, 0).unwrap() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn driver_progress_unknown_node_is_none() {
        let d = AnimationDriver::new();
        assert!(d.progress(7, 0).is_none());
    }

    #[test]
    fn driver_progress_zero_duration_is_one() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 0), 0);
        assert!((d.progress(7, 0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn driver_delay_keeps_progress_zero() {
        let mut d = AnimationDriver::new();
        d.start(
            7,
            AnimationSpec::new(AnimationCurve::Linear, 100).delay(50),
            0,
        );
        d.tick(25);
        assert!((d.progress(7, 0).unwrap() - 0.0).abs() < 1e-6);
        d.tick(75);
        // 25 ms past the 50 ms delay → 25/100 = 0.25.
        assert!((d.progress(7, 0).unwrap() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn driver_tick_returns_redraw_set() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.start(8, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let r = d.tick(50);
        assert_eq!(r, vec![7, 8]);
    }

    #[test]
    fn driver_tick_dedups_multi_key_per_node() {
        let mut d = AnimationDriver::new();
        let s1 = AnimationSpec::new(AnimationCurve::Linear, 100).keyed(1);
        let s2 = AnimationSpec::new(AnimationCurve::Linear, 100).keyed(2);
        d.start(7, s1, 0);
        d.start(7, s2, 0);
        let r = d.tick(50);
        assert_eq!(r, vec![7]);
    }

    #[test]
    fn driver_gc_drops_finished() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(200);
        d.gc();
        assert_eq!(d.active_count(), 0);
    }

    #[test]
    fn driver_restart_resets_progress() {
        let mut d = AnimationDriver::new();
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(50);
        d.start(7, AnimationSpec::new(AnimationCurve::Linear, 100), 50);
        d.tick(50);
        // Restarted at now=50, current tick=50 → elapsed = 0.
        assert!((d.progress(7, 0).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn driver_records_last_tick() {
        let mut d = AnimationDriver::new();
        d.tick(123);
        assert_eq!(d.last_tick_ms(), Some(123));
    }

    #[test]
    fn driver_default_presence_is_stable() {
        let d = AnimationDriver::new();
        assert_eq!(d.presence(42), Presence::Stable);
    }

    #[test]
    fn driver_set_presence_round_trips() {
        let mut d = AnimationDriver::new();
        d.set_presence(42, Presence::Entering);
        assert_eq!(d.presence(42), Presence::Entering);
    }

    #[test]
    fn driver_diff_presence_marks_entering_and_leaving() {
        let mut d = AnimationDriver::new();
        d.diff_presence(&[1, 2, 3], &[2, 3, 4]);
        assert_eq!(d.presence(1), Presence::Leaving);
        assert_eq!(d.presence(2), Presence::Stable);
        assert_eq!(d.presence(3), Presence::Stable);
        assert_eq!(d.presence(4), Presence::Entering);
    }

    // -------------------------------------------------------------------------
    // Transition helpers
    // -------------------------------------------------------------------------

    #[test]
    fn opacity_transition_entering_progresses_alpha() {
        assert_eq!(
            opacity_for_transition(&TransitionKind::Opacity, Presence::Entering, 0.0),
            Some(0.0)
        );
        assert_eq!(
            opacity_for_transition(&TransitionKind::Opacity, Presence::Entering, 1.0),
            Some(1.0)
        );
    }

    #[test]
    fn opacity_transition_leaving_inverts() {
        assert_eq!(
            opacity_for_transition(&TransitionKind::Opacity, Presence::Leaving, 0.0),
            Some(1.0)
        );
        assert_eq!(
            opacity_for_transition(&TransitionKind::Opacity, Presence::Leaving, 1.0),
            Some(0.0)
        );
    }

    #[test]
    fn opacity_transition_stable_full_alpha() {
        assert_eq!(
            opacity_for_transition(&TransitionKind::Opacity, Presence::Stable, 0.5),
            Some(1.0)
        );
    }

    #[test]
    fn opacity_transition_none_returns_none() {
        assert_eq!(
            opacity_for_transition(&TransitionKind::None, Presence::Entering, 0.5),
            None
        );
    }

    #[test]
    fn slide_transition_entering_starts_offscreen() {
        let kind = TransitionKind::Slide { edge: Edge::Top };
        let bounds = Size::new(10.0, 20.0);
        let off = slide_for_transition(&kind, Presence::Entering, 0.0, bounds);
        assert_eq!(off, Some((0.0, -20.0)));
        let on = slide_for_transition(&kind, Presence::Entering, 1.0, bounds);
        assert_eq!(on, Some((0.0, 0.0)));
    }

    #[test]
    fn slide_transition_leaving_ends_offscreen() {
        let kind = TransitionKind::Slide { edge: Edge::Bottom };
        let bounds = Size::new(10.0, 20.0);
        let on = slide_for_transition(&kind, Presence::Leaving, 0.0, bounds);
        assert_eq!(on, Some((0.0, 0.0)));
        let off = slide_for_transition(&kind, Presence::Leaving, 1.0, bounds);
        assert_eq!(off, Some((0.0, 20.0)));
    }

    #[test]
    fn slide_transition_leading_pushes_left() {
        let kind = TransitionKind::Slide {
            edge: Edge::Leading,
        };
        let bounds = Size::new(10.0, 20.0);
        let off = slide_for_transition(&kind, Presence::Entering, 0.0, bounds);
        assert_eq!(off, Some((-10.0, 0.0)));
    }

    #[test]
    fn slide_transition_trailing_pushes_right() {
        let kind = TransitionKind::Slide {
            edge: Edge::Trailing,
        };
        let bounds = Size::new(10.0, 20.0);
        let off = slide_for_transition(&kind, Presence::Entering, 0.0, bounds);
        assert_eq!(off, Some((10.0, 0.0)));
    }

    #[test]
    fn scale_transition_entering_grows_to_one() {
        let kind = TransitionKind::Scale { from: 0.0 };
        assert!((scale_for_transition(&kind, Presence::Entering, 0.0).unwrap() - 0.0).abs() < 1e-6);
        assert!((scale_for_transition(&kind, Presence::Entering, 1.0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_transition_leaving_shrinks_to_from() {
        let kind = TransitionKind::Scale { from: 0.5 };
        let start = scale_for_transition(&kind, Presence::Leaving, 0.0).unwrap();
        let end = scale_for_transition(&kind, Presence::Leaving, 1.0).unwrap();
        assert!((start - 1.0).abs() < 1e-6);
        assert!((end - 0.5).abs() < 1e-6);
    }

    #[test]
    fn asymmetric_picks_enter_on_entering() {
        let kind = TransitionKind::Asymmetric {
            enter: Box::new(TransitionKind::Opacity),
            exit: Box::new(TransitionKind::Slide { edge: Edge::Top }),
        };
        // Enter side should be opacity.
        assert_eq!(
            opacity_for_transition(&kind, Presence::Entering, 0.5),
            Some(0.5)
        );
        // Exit side should be slide.
        assert!(
            slide_for_transition(&kind, Presence::Leaving, 0.0, Size::new(0.0, 10.0)).is_some()
        );
    }

    #[test]
    fn resolve_transition_passthrough_for_simple_kinds() {
        let k = TransitionKind::Opacity;
        let r = resolve_transition(&k, Presence::Entering);
        assert!(matches!(r, TransitionKind::Opacity));
    }
}
