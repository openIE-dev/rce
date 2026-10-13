//! NWP serialization for Photon view trees.
//!
//! Two surfaces:
//! - **Full snapshot** — [`serialize`] produces a [`ViewFrame`] for the
//!   first frame and recoveries.
//! - **Delta** — [`diff`] returns a [`Vec<ViewDelta>`] for subsequent
//!   frames. The host applies deltas to its cached tree without
//!   re-receiving unchanged subtrees.
//!
//! State updates flow the other direction (peer → host) but Photon keeps
//! state local to the peer; the host only ships back [`EventPayload`]
//! frames describing user input. The peer's own signal arena handles
//! state.
//!
//! Closures (`OnTap`, `Canvas`, `Stateful`) cannot travel over the wire.
//! - [`HandlerTable`] registers the `Rc<dyn Fn>` handlers and emits a
//!   `u32` token; the wire only carries the token. The host sends the
//!   token back on input events; the peer dispatches it via
//!   [`HandlerTable::dispatch`].
//! - [`ViewNode::Stateful`] subtrees are flattened during serialisation
//!   (the closure is evaluated once on the peer); the wire sees only the
//!   resulting expansion.
//! - [`LeafKind::Canvas`] closures are serialised as opaque tagged
//!   placeholders; the host's renderer treats the `tag` as a reference
//!   to a peer-side bitmap upload (out of scope for this stub but
//!   round-trippable).
//!
//! Last updated: 2026-05-01

use std::collections::HashMap;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::animation::{AnimationCurve, Edge, TransitionKind};
use crate::color::Color;
use crate::layout::{
    Alignment2D, HAlignment, HStack as LHStack, Layout, LayoutRect, Rect, VAlignment,
    VStack as LVStack, ZStack as LZStack,
};
use crate::render::{DrawCommand, RenderCtx};
use crate::view::{
    AccessibilityRole, BackgroundFill, ClipShape, ContentFit, LeafKind, Modifier, PathCommand,
    ViewNode,
};

// =============================================================================
// Wire types
// =============================================================================

/// Stable identity for a node within a single view tree. Assigned by a
/// post-order walk so structural diffs are stable across renders.
pub type ViewNodeId = u32;

/// Full snapshot of a serialised [`ViewNode`] tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViewFrame {
    pub root: ViewNodeId,
    pub nodes: Vec<SerializedNode>,
    pub generation: u64,
}

/// One node in a [`ViewFrame`]. Children are referenced by id rather than
/// inlined so that delta patches can re-use unchanged subtrees.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializedNode {
    pub id: ViewNodeId,
    pub kind: SerializedKind,
    pub children: Vec<ViewNodeId>,
}

/// Discriminated wire form of a [`ViewNode`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedKind {
    /// Solid rect background, text run, image, divider, spacer, canvas placeholder.
    Leaf(SerializedLeaf),
    /// Layout container — the renderer dispatches to the matching
    /// [`crate::layout::Layout`] implementation by reading `layout`.
    Container {
        layout: SerializedLayout,
        children_count: u32,
    },
    /// Decorator wrapping a single child.
    Modified { modifier: SerializedModifier },
    /// Stateful nodes are flattened — the peer evaluates the closure
    /// before serialising, so the wire only sees their current expansion.
    /// This variant exists for diffing parity with the live tree (the
    /// serialiser re-evaluates and emits a fresh subtree each frame).
    Stateful,
}

/// Wire form of [`crate::view::LeafKind`]. Canvas closures degrade to
/// opaque tagged placeholders.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedLeaf {
    Text {
        content: String,
        font_size: f32,
        color: [f32; 4],
        bold: bool,
        italic: bool,
        underline: bool,
    },
    Image {
        url: String,
        fit: u8,
    },
    Spacer {
        min_length: f32,
    },
    Divider {
        thickness: f32,
        color: [f32; 4],
    },
    /// Canvas closures don't survive — the peer renders to a bitmap and
    /// ships it via `AppDisplayCommand::Image`. The `tag` is a stable
    /// identifier that the host can use to look up the cached bitmap.
    Canvas {
        tag: String,
    },
}

/// Wire form of [`crate::layout::Layout`] implementations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedLayout {
    HStack {
        spacing: f32,
        alignment: u8,
    },
    VStack {
        spacing: f32,
        alignment: u8,
    },
    /// Packs h+v alignment into a u16 — high byte for horizontal, low
    /// byte for vertical (see [`pack_alignment`] / [`unpack_alignment`]).
    ZStack {
        alignment: u16,
    },
    Frame {
        width: Option<f32>,
        height: Option<f32>,
        alignment: u16,
    },
    Grid {
        rows: Vec<u32>,
        columns: Vec<u32>,
    },
}

/// Wire form of [`crate::view::Modifier`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedModifier {
    Padding {
        top: f32,
        right: f32,
        bottom: f32,
        left: f32,
    },
    Background(SerializedFill),
    Foreground([f32; 4]),
    Opacity(f32),
    CornerRadius(f32),
    Frame {
        width: Option<f32>,
        height: Option<f32>,
    },
    FrameAligned {
        width: Option<f32>,
        height: Option<f32>,
        alignment: u16,
    },
    FrameRange {
        min_width: Option<f32>,
        ideal_width: Option<f32>,
        max_width: Option<f32>,
        min_height: Option<f32>,
        ideal_height: Option<f32>,
        max_height: Option<f32>,
        alignment: u16,
    },
    Border {
        width: f32,
        color: [f32; 4],
    },
    Shadow {
        color: [f32; 4],
        radius: f32,
        offset_x: f32,
        offset_y: f32,
    },
    InnerShadow {
        color: [f32; 4],
        radius: f32,
        offset_x: f32,
        offset_y: f32,
    },
    /// Multi-shadow stack — each entry is a `(color, radius, offset_x, offset_y)`
    /// tuple. Painted back-to-front before the child renders.
    Shadows(Vec<([f32; 4], f32, f32, f32)>),
    /// Backdrop filter (Apple "Liquid Glass" style). Captures the destination,
    /// applies blur + colour matrix, composites back inside the bounds.
    BackdropFilter {
        blur_radius: f32,
        saturate: f32,
        brightness: f32,
        contrast: f32,
        grayscale: f32,
        hue_rotate: f32,
        invert: f32,
        sepia: f32,
    },
    Rotation(f32),
    Scale {
        x: f32,
        y: f32,
        anchor: u16,
    },
    Offset {
        x: f32,
        y: f32,
    },
    Position {
        x: f32,
        y: f32,
    },
    Clip(SerializedClipShape),
    Mask(SerializedClipShape),
    /// Hit handlers are compiled to (kind, handler_token). The host turns
    /// taps into [`EventPayload`] frames carrying the token; the peer
    /// dispatches the matching closure.
    OnTap {
        handler_token: u32,
    },
    OnHover {
        handler_token: u32,
    },
    OnLongPress {
        duration_ms: u32,
        handler_token: u32,
    },
    OnDrag {
        handler_token: u32,
    },
    OnAppear {
        handler_token: u32,
    },
    OnDisappear {
        handler_token: u32,
    },
    AccessibilityLabel(String),
    AccessibilityRole(u8),
    AccessibilityHint(String),
    AccessibilityValue(String),
    AccessibilityHidden(bool),
    TabIndex(i32),
    AllowsHitTesting(bool),
    Tooltip(String),
    ZIndex(f32),
    Animation {
        duration_ms: u32,
        delay_ms: u32,
        curve: u8,
        key: u64,
    },
    Transition(SerializedTransition),
    Tag(String),
    Hidden(bool),
}

/// Wire form of [`BackgroundFill`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedFill {
    Color([f32; 4]),
    LinearGradient {
        angle_deg: f32,
        stops: Vec<(f32, [f32; 4])>,
    },
    RadialGradient {
        center: u16,
        stops: Vec<(f32, [f32; 4])>,
    },
}

/// Wire form of [`ClipShape`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedClipShape {
    Rect,
    RoundedRect {
        radius: f32,
    },
    Circle,
    Capsule,
    Ellipse,
    Path {
        commands: Vec<SerializedPathCommand>,
    },
}

/// Wire form of [`PathCommand`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedPathCommand {
    MoveTo(f32, f32),
    LineTo(f32, f32),
    QuadTo {
        cx: f32,
        cy: f32,
        x: f32,
        y: f32,
    },
    CubicTo {
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    },
    Close,
}

/// Wire form of [`TransitionKind`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializedTransition {
    None,
    Opacity,
    Slide {
        edge: u8,
    },
    Scale {
        from: f32,
    },
    Asymmetric {
        enter: Box<SerializedTransition>,
        exit: Box<SerializedTransition>,
    },
}

// =============================================================================
// Alignment / role helpers
// =============================================================================

const fn h_align_to_u8(a: HAlignment) -> u8 {
    match a {
        HAlignment::Leading => 0,
        HAlignment::Center => 1,
        HAlignment::Trailing => 2,
    }
}

const fn h_align_from_u8(b: u8) -> HAlignment {
    match b {
        0 => HAlignment::Leading,
        2 => HAlignment::Trailing,
        _ => HAlignment::Center,
    }
}

const fn v_align_to_u8(a: VAlignment) -> u8 {
    match a {
        VAlignment::Top => 0,
        VAlignment::Center => 1,
        VAlignment::Bottom => 2,
        VAlignment::FirstTextBaseline => 3,
        VAlignment::LastTextBaseline => 4,
    }
}

const fn v_align_from_u8(b: u8) -> VAlignment {
    match b {
        0 => VAlignment::Top,
        2 => VAlignment::Bottom,
        3 => VAlignment::FirstTextBaseline,
        4 => VAlignment::LastTextBaseline,
        _ => VAlignment::Center,
    }
}

/// Pack a 2D alignment into a u16 (high byte: horizontal, low byte:
/// vertical).
pub const fn pack_alignment(a: Alignment2D) -> u16 {
    ((h_align_to_u8(a.h) as u16) << 8) | (v_align_to_u8(a.v) as u16)
}

/// Unpack a u16 produced by [`pack_alignment`] back into an
/// [`Alignment2D`].
pub const fn unpack_alignment(v: u16) -> Alignment2D {
    Alignment2D::new(
        h_align_from_u8(((v >> 8) & 0xff) as u8),
        v_align_from_u8((v & 0xff) as u8),
    )
}

const fn role_to_u8(r: AccessibilityRole) -> u8 {
    match r {
        AccessibilityRole::Button => 0,
        AccessibilityRole::Link => 1,
        AccessibilityRole::Heading(_) => 2,
        AccessibilityRole::Image => 3,
        AccessibilityRole::List => 4,
        AccessibilityRole::ListItem => 5,
        AccessibilityRole::Header => 6,
        AccessibilityRole::Footer => 7,
        AccessibilityRole::Navigation => 8,
        AccessibilityRole::Main => 9,
        AccessibilityRole::Region => 10,
        AccessibilityRole::Search => 11,
        AccessibilityRole::Form => 12,
        AccessibilityRole::Dialog => 13,
        AccessibilityRole::Alert => 14,
        AccessibilityRole::Status => 15,
        AccessibilityRole::Tab => 16,
        AccessibilityRole::TabPanel => 17,
        AccessibilityRole::Menu => 18,
        AccessibilityRole::MenuItem => 19,
        AccessibilityRole::Checkbox => 20,
        AccessibilityRole::Radio => 21,
        AccessibilityRole::Switch => 22,
        AccessibilityRole::ProgressBar => 23,
        AccessibilityRole::Slider => 24,
        AccessibilityRole::ComboBox => 25,
        AccessibilityRole::TextBox => 26,
        AccessibilityRole::Tree => 27,
        AccessibilityRole::TreeItem => 28,
        AccessibilityRole::Grid => 29,
        AccessibilityRole::Cell => 30,
        AccessibilityRole::RowHeader => 31,
        AccessibilityRole::ColumnHeader => 32,
        AccessibilityRole::Article => 33,
        AccessibilityRole::Banner => 34,
        AccessibilityRole::Complementary => 35,
        AccessibilityRole::ContentInfo => 36,
        AccessibilityRole::Generic => 37,
    }
}

const fn role_from_u8(b: u8) -> AccessibilityRole {
    match b {
        0 => AccessibilityRole::Button,
        1 => AccessibilityRole::Link,
        2 => AccessibilityRole::Heading(1),
        3 => AccessibilityRole::Image,
        4 => AccessibilityRole::List,
        5 => AccessibilityRole::ListItem,
        6 => AccessibilityRole::Header,
        7 => AccessibilityRole::Footer,
        8 => AccessibilityRole::Navigation,
        9 => AccessibilityRole::Main,
        10 => AccessibilityRole::Region,
        11 => AccessibilityRole::Search,
        12 => AccessibilityRole::Form,
        13 => AccessibilityRole::Dialog,
        14 => AccessibilityRole::Alert,
        15 => AccessibilityRole::Status,
        16 => AccessibilityRole::Tab,
        17 => AccessibilityRole::TabPanel,
        18 => AccessibilityRole::Menu,
        19 => AccessibilityRole::MenuItem,
        20 => AccessibilityRole::Checkbox,
        21 => AccessibilityRole::Radio,
        22 => AccessibilityRole::Switch,
        23 => AccessibilityRole::ProgressBar,
        24 => AccessibilityRole::Slider,
        25 => AccessibilityRole::ComboBox,
        26 => AccessibilityRole::TextBox,
        27 => AccessibilityRole::Tree,
        28 => AccessibilityRole::TreeItem,
        29 => AccessibilityRole::Grid,
        30 => AccessibilityRole::Cell,
        31 => AccessibilityRole::RowHeader,
        32 => AccessibilityRole::ColumnHeader,
        33 => AccessibilityRole::Article,
        34 => AccessibilityRole::Banner,
        35 => AccessibilityRole::Complementary,
        36 => AccessibilityRole::ContentInfo,
        _ => AccessibilityRole::Generic,
    }
}

const fn fit_to_u8(f: ContentFit) -> u8 {
    match f {
        ContentFit::Fill => 0,
        ContentFit::Contain => 1,
        ContentFit::Cover => 2,
        ContentFit::None => 3,
        ContentFit::ScaleDown => 4,
    }
}

const fn fit_from_u8(b: u8) -> ContentFit {
    match b {
        0 => ContentFit::Fill,
        2 => ContentFit::Cover,
        3 => ContentFit::None,
        4 => ContentFit::ScaleDown,
        _ => ContentFit::Contain,
    }
}

const fn edge_to_u8(e: Edge) -> u8 {
    match e {
        Edge::Top => 0,
        Edge::Bottom => 1,
        Edge::Leading => 2,
        Edge::Trailing => 3,
    }
}

const fn edge_from_u8(b: u8) -> Edge {
    match b {
        0 => Edge::Top,
        1 => Edge::Bottom,
        2 => Edge::Leading,
        _ => Edge::Trailing,
    }
}

const fn curve_to_u8(c: AnimationCurve) -> u8 {
    match c {
        AnimationCurve::Linear => 0,
        AnimationCurve::EaseIn => 1,
        AnimationCurve::EaseOut => 2,
        AnimationCurve::EaseInOut => 3,
        AnimationCurve::Bezier(_, _, _, _) => 4,
        AnimationCurve::Spring { .. } => 5,
    }
}

const fn curve_from_u8(b: u8) -> AnimationCurve {
    match b {
        0 => AnimationCurve::Linear,
        2 => AnimationCurve::EaseOut,
        3 => AnimationCurve::EaseInOut,
        4 => AnimationCurve::Bezier(0.0, 0.0, 1.0, 1.0),
        5 => AnimationCurve::Spring {
            mass: 1.0,
            stiffness: 170.0,
            damping: 10.0,
            initial_velocity: 0.0,
        },
        _ => AnimationCurve::EaseIn,
    }
}

// =============================================================================
// HandlerTable — peer-side handler registry referenced by tokens.
// =============================================================================

/// Payload of an inbound input event. The host ships these back to the
/// peer keyed by handler token; the peer dispatches via
/// [`HandlerTable::dispatch`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EventPayload {
    Tap { x: f32, y: f32 },
    Hover { entering: bool, x: f32, y: f32 },
    LongPress { x: f32, y: f32 },
    Drag { dx: f32, dy: f32 },
    Appear,
    Disappear,
}

/// Internal handler discriminator — the table stores closures keyed by
/// the gesture flavour they were registered as so [`HandlerTable::dispatch`]
/// only invokes a closure when the inbound payload matches.
enum HandlerKind {
    Tap(Rc<dyn Fn()>),
    Hover(Rc<dyn Fn(bool)>),
    LongPress(Rc<dyn Fn()>),
    Drag(Rc<dyn Fn(f32, f32)>),
    /// Lifecycle (`OnAppear` / `OnDisappear`) — fires for either payload.
    Lifecycle(Rc<dyn Fn()>),
}

/// Per-app handler table. Registers `Rc<dyn Fn>` closures and emits
/// monotonically increasing `u32` tokens that travel on the wire.
///
/// The peer keeps the table alive across frames; serialisation re-uses
/// existing tokens implicitly because the closures are stored by their
/// `Rc` identity (each new render re-registers fresh tokens — call
/// [`HandlerTable::clear`] between frames to keep the table bounded).
#[derive(Default)]
pub struct HandlerTable {
    handlers: HashMap<u32, HandlerKind>,
    next_token: u32,
}

impl std::fmt::Debug for HandlerTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlerTable")
            .field("len", &self.handlers.len())
            .field("next_token", &self.next_token)
            .finish()
    }
}

impl HandlerTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tap handler. Returns the token to embed in the wire
    /// form.
    pub fn register_tap(&mut self, f: impl Fn() + 'static) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Tap(Rc::new(f)));
        token
    }

    /// Register a hover handler.
    pub fn register_hover(&mut self, f: impl Fn(bool) + 'static) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Hover(Rc::new(f)));
        token
    }

    /// Register a long-press handler.
    pub fn register_long_press(&mut self, f: impl Fn() + 'static) -> u32 {
        let token = self.alloc();
        self.handlers
            .insert(token, HandlerKind::LongPress(Rc::new(f)));
        token
    }

    /// Register a drag handler.
    pub fn register_drag(&mut self, f: impl Fn(f32, f32) + 'static) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Drag(Rc::new(f)));
        token
    }

    /// Register a lifecycle (`OnAppear` / `OnDisappear`) handler.
    pub fn register_lifecycle(&mut self, f: impl Fn() + 'static) -> u32 {
        let token = self.alloc();
        self.handlers
            .insert(token, HandlerKind::Lifecycle(Rc::new(f)));
        token
    }

    /// Register a tap handler from an existing `Rc<dyn Fn()>` (the
    /// serialiser uses this so it can keep the closure identity from
    /// the source [`Modifier`]).
    pub fn register_tap_rc(&mut self, f: Rc<dyn Fn()>) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Tap(f));
        token
    }

    /// Variant of [`Self::register_hover`] taking an existing `Rc`.
    pub fn register_hover_rc(&mut self, f: Rc<dyn Fn(bool)>) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Hover(f));
        token
    }

    /// Variant of [`Self::register_long_press`] taking an existing `Rc`.
    pub fn register_long_press_rc(&mut self, f: Rc<dyn Fn()>) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::LongPress(f));
        token
    }

    /// Variant of [`Self::register_drag`] taking an existing `Rc`.
    pub fn register_drag_rc(&mut self, f: Rc<dyn Fn(f32, f32)>) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Drag(f));
        token
    }

    /// Variant of [`Self::register_lifecycle`] taking an existing `Rc`.
    pub fn register_lifecycle_rc(&mut self, f: Rc<dyn Fn()>) -> u32 {
        let token = self.alloc();
        self.handlers.insert(token, HandlerKind::Lifecycle(f));
        token
    }

    fn alloc(&mut self) -> u32 {
        let t = self.next_token;
        // Tokens are monotonic — never reuse an id within the lifetime
        // of this table. Wrapping is intentional and unlikely (over 4
        // billion handlers in a single app session would be required).
        self.next_token = self.next_token.wrapping_add(1);
        t
    }

    /// Dispatch an inbound `EventPayload` to the closure registered
    /// under `token`. Silently no-ops if the token is unknown or the
    /// payload type doesn't match the registered handler kind.
    pub fn dispatch(&self, token: u32, payload: EventPayload) {
        let Some(kind) = self.handlers.get(&token) else {
            return;
        };
        match (kind, payload) {
            (HandlerKind::Tap(f), EventPayload::Tap { .. }) => f(),
            (HandlerKind::Hover(f), EventPayload::Hover { entering, .. }) => f(entering),
            (HandlerKind::LongPress(f), EventPayload::LongPress { .. }) => f(),
            (HandlerKind::Drag(f), EventPayload::Drag { dx, dy }) => f(dx, dy),
            (HandlerKind::Lifecycle(f), EventPayload::Appear)
            | (HandlerKind::Lifecycle(f), EventPayload::Disappear) => f(),
            // Mismatched (token, payload) pairs are silently ignored —
            // the host shouldn't ship them but we don't want to panic
            // on a malicious peer.
            _ => {}
        }
    }

    /// Forget every registered handler. Reset the token counter so the
    /// next frame can reuse small ids cleanly.
    pub fn clear(&mut self) {
        self.handlers.clear();
        self.next_token = 0;
    }

    /// Number of currently-registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Whether `token` is currently registered.
    pub fn contains(&self, token: u32) -> bool {
        self.handlers.contains_key(&token)
    }
}

// =============================================================================
// Serialisation
// =============================================================================

/// Serialise a [`ViewNode`] tree into a [`ViewFrame`] with stable ids.
///
/// Closure-bearing modifiers register their handler in a fresh
/// [`HandlerTable`]; if the caller wants to keep those handlers alive
/// to dispatch events, they should use [`serialize_with_handlers`]
/// instead.
pub fn serialize(view: &ViewNode) -> ViewFrame {
    let mut handlers = HandlerTable::new();
    serialize_with_handlers(view, &mut handlers, 0)
}

/// Serialise a view tree, recording closure handlers in `handlers`.
/// `generation` is stamped onto the resulting frame so the host can
/// detect rollbacks.
pub fn serialize_with_handlers(
    view: &ViewNode,
    handlers: &mut HandlerTable,
    generation: u64,
) -> ViewFrame {
    let mut nodes = Vec::new();
    let root = walk(view, &mut nodes, handlers);
    ViewFrame {
        root,
        nodes,
        generation,
    }
}

/// Post-order walk that assigns stable ids and pushes [`SerializedNode`]s.
fn walk(node: &ViewNode, out: &mut Vec<SerializedNode>, handlers: &mut HandlerTable) -> ViewNodeId {
    match node {
        ViewNode::Leaf(k) => {
            let id = out.len() as u32;
            out.push(SerializedNode {
                id,
                kind: SerializedKind::Leaf(serialize_leaf(k)),
                children: Vec::new(),
            });
            id
        }
        ViewNode::Container { layout, children } => {
            let child_ids: Vec<ViewNodeId> =
                children.iter().map(|c| walk(c, out, handlers)).collect();
            let id = out.len() as u32;
            out.push(SerializedNode {
                id,
                kind: SerializedKind::Container {
                    layout: serialize_layout(layout.as_ref()),
                    children_count: child_ids.len() as u32,
                },
                children: child_ids,
            });
            id
        }
        ViewNode::Modified { modifier, child } => {
            let child_id = walk(child, out, handlers);
            let id = out.len() as u32;
            out.push(SerializedNode {
                id,
                kind: SerializedKind::Modified {
                    modifier: serialize_modifier(modifier, handlers),
                },
                children: vec![child_id],
            });
            id
        }
        ViewNode::Stateful(f) => {
            // Flatten stateful nodes — evaluate the closure now and
            // serialise the result. The wire still carries a `Stateful`
            // marker so the diff engine can detect re-evaluations.
            let resolved = f();
            let inner_id = walk(&resolved, out, handlers);
            let id = out.len() as u32;
            out.push(SerializedNode {
                id,
                kind: SerializedKind::Stateful,
                children: vec![inner_id],
            });
            id
        }
    }
}

fn serialize_leaf(k: &LeafKind) -> SerializedLeaf {
    match k {
        LeafKind::Text(t) => SerializedLeaf::Text {
            content: t.content.clone(),
            font_size: t.font_size,
            color: t.color.to_array(),
            bold: t.bold,
            italic: t.italic,
            underline: t.underline,
        },
        LeafKind::Image(i) => SerializedLeaf::Image {
            url: i.url.clone(),
            fit: fit_to_u8(i.fit),
        },
        LeafKind::Spacer { min_length } => SerializedLeaf::Spacer {
            min_length: *min_length,
        },
        LeafKind::Divider { thickness, color } => SerializedLeaf::Divider {
            thickness: *thickness,
            color: color.to_array(),
        },
        // Canvas closures degrade to opaque tags. The peer is expected
        // to render a bitmap and ship it via `AppDisplayCommand::Image`
        // separately; the wire frame just records the placeholder.
        LeafKind::Canvas(_) => SerializedLeaf::Canvas { tag: String::new() },
    }
}

fn serialize_layout(layout: &dyn Layout) -> SerializedLayout {
    // Recover the layout's concrete identity via its `as_any` upcast
    // and downcast against the built-in shapes.
    let any = layout.as_any();
    if let Some(h) = any.downcast_ref::<LHStack>() {
        return SerializedLayout::HStack {
            spacing: h.spacing,
            alignment: v_align_to_u8(h.alignment),
        };
    }
    if let Some(v) = any.downcast_ref::<LVStack>() {
        return SerializedLayout::VStack {
            spacing: v.spacing,
            alignment: h_align_to_u8(v.alignment),
        };
    }
    if let Some(z) = any.downcast_ref::<LZStack>() {
        return SerializedLayout::ZStack {
            alignment: pack_alignment(z.alignment),
        };
    }
    if let Some(f) = any.downcast_ref::<crate::layout::Frame>() {
        return SerializedLayout::Frame {
            width: f.width,
            height: f.height,
            alignment: pack_alignment(f.alignment),
        };
    }
    // Unknown layout — fall back to a default HStack shape so the wire
    // still validates. Custom layouts beyond the built-in four are out
    // of Phase 5 scope.
    SerializedLayout::HStack {
        spacing: 0.0,
        alignment: v_align_to_u8(VAlignment::Center),
    }
}

fn serialize_modifier(m: &Modifier, handlers: &mut HandlerTable) -> SerializedModifier {
    match m {
        Modifier::Padding(e) => SerializedModifier::Padding {
            top: e.top,
            right: e.right,
            bottom: e.bottom,
            left: e.left,
        },
        Modifier::Background(f) => SerializedModifier::Background(serialize_fill(f)),
        Modifier::Foreground(c) => SerializedModifier::Foreground(c.to_array()),
        Modifier::Opacity(v) => SerializedModifier::Opacity(*v),
        Modifier::CornerRadius(v) => SerializedModifier::CornerRadius(*v),
        Modifier::Frame { width, height } => SerializedModifier::Frame {
            width: *width,
            height: *height,
        },
        Modifier::FrameAligned {
            width,
            height,
            alignment,
        } => SerializedModifier::FrameAligned {
            width: *width,
            height: *height,
            alignment: pack_alignment(*alignment),
        },
        Modifier::FrameRange {
            min_width,
            ideal_width,
            max_width,
            min_height,
            ideal_height,
            max_height,
            alignment,
        } => SerializedModifier::FrameRange {
            min_width: *min_width,
            ideal_width: *ideal_width,
            max_width: *max_width,
            min_height: *min_height,
            ideal_height: *ideal_height,
            max_height: *max_height,
            alignment: pack_alignment(*alignment),
        },
        Modifier::Border { width, color } => SerializedModifier::Border {
            width: *width,
            color: color.to_array(),
        },
        Modifier::Shadow {
            color,
            radius,
            offset_x,
            offset_y,
        } => SerializedModifier::Shadow {
            color: color.to_array(),
            radius: *radius,
            offset_x: *offset_x,
            offset_y: *offset_y,
        },
        Modifier::InnerShadow {
            color,
            radius,
            offset_x,
            offset_y,
        } => SerializedModifier::InnerShadow {
            color: color.to_array(),
            radius: *radius,
            offset_x: *offset_x,
            offset_y: *offset_y,
        },
        Modifier::Rotation(d) => SerializedModifier::Rotation(*d),
        Modifier::Scale { x, y, anchor } => SerializedModifier::Scale {
            x: *x,
            y: *y,
            anchor: pack_alignment(*anchor),
        },
        Modifier::Offset { x, y } => SerializedModifier::Offset { x: *x, y: *y },
        Modifier::Position { x, y } => SerializedModifier::Position { x: *x, y: *y },
        Modifier::Clip(s) => SerializedModifier::Clip(serialize_clip(s)),
        Modifier::Mask(s) => SerializedModifier::Mask(serialize_clip(s)),
        Modifier::OnTap(f) => SerializedModifier::OnTap {
            handler_token: handlers.register_tap_rc(Rc::clone(f)),
        },
        Modifier::OnHover(f) => SerializedModifier::OnHover {
            handler_token: handlers.register_hover_rc(Rc::clone(f)),
        },
        Modifier::OnLongPress {
            duration_ms,
            action,
        } => SerializedModifier::OnLongPress {
            duration_ms: *duration_ms,
            handler_token: handlers.register_long_press_rc(Rc::clone(action)),
        },
        Modifier::OnDrag(f) => SerializedModifier::OnDrag {
            handler_token: handlers.register_drag_rc(Rc::clone(f)),
        },
        Modifier::OnAppear(f) => SerializedModifier::OnAppear {
            handler_token: handlers.register_lifecycle_rc(Rc::clone(f)),
        },
        Modifier::OnDisappear(f) => SerializedModifier::OnDisappear {
            handler_token: handlers.register_lifecycle_rc(Rc::clone(f)),
        },
        Modifier::AccessibilityLabel(s) => SerializedModifier::AccessibilityLabel(s.clone()),
        Modifier::AccessibilityRole(r) => SerializedModifier::AccessibilityRole(role_to_u8(*r)),
        Modifier::AccessibilityHint(s) => SerializedModifier::AccessibilityHint(s.clone()),
        Modifier::AccessibilityValue(s) => SerializedModifier::AccessibilityValue(s.clone()),
        Modifier::AccessibilityHidden(b) => SerializedModifier::AccessibilityHidden(*b),
        Modifier::TabIndex(i) => SerializedModifier::TabIndex(*i),
        Modifier::AllowsHitTesting(b) => SerializedModifier::AllowsHitTesting(*b),
        Modifier::Tooltip(s) => SerializedModifier::Tooltip(s.clone()),
        Modifier::ZIndex(z) => SerializedModifier::ZIndex(*z),
        Modifier::Animation(spec) => SerializedModifier::Animation {
            duration_ms: spec.duration_ms,
            delay_ms: spec.delay_ms,
            curve: curve_to_u8(spec.curve),
            key: spec.key,
        },
        Modifier::Transition(t) => SerializedModifier::Transition(serialize_transition(t)),
        Modifier::Tag(s) => SerializedModifier::Tag(s.clone()),
        Modifier::Hidden(b) => SerializedModifier::Hidden(*b),
        Modifier::Shadows(stack) => SerializedModifier::Shadows(
            stack
                .iter()
                .map(|s| (s.color.to_array(), s.radius, s.dx, s.dy))
                .collect(),
        ),
        Modifier::BackdropFilter(f) => SerializedModifier::BackdropFilter {
            blur_radius: f.blur_radius,
            saturate: f.saturate,
            brightness: f.brightness,
            contrast: f.contrast,
            grayscale: f.grayscale,
            hue_rotate: f.hue_rotate,
            invert: f.invert,
            sepia: f.sepia,
        },
    }
}

fn serialize_fill(f: &BackgroundFill) -> SerializedFill {
    match f {
        BackgroundFill::Color(c) => SerializedFill::Color(c.to_array()),
        BackgroundFill::LinearGradient { angle_deg, stops } => SerializedFill::LinearGradient {
            angle_deg: *angle_deg,
            stops: stops.iter().map(|(p, c)| (*p, c.to_array())).collect(),
        },
        BackgroundFill::RadialGradient { center, stops } => SerializedFill::RadialGradient {
            center: pack_alignment(*center),
            stops: stops.iter().map(|(p, c)| (*p, c.to_array())).collect(),
        },
    }
}

fn serialize_clip(s: &ClipShape) -> SerializedClipShape {
    match s {
        ClipShape::Rect => SerializedClipShape::Rect,
        ClipShape::RoundedRect { radius } => SerializedClipShape::RoundedRect { radius: *radius },
        ClipShape::Circle => SerializedClipShape::Circle,
        ClipShape::Capsule => SerializedClipShape::Capsule,
        ClipShape::Ellipse => SerializedClipShape::Ellipse,
        ClipShape::Path(cmds) => SerializedClipShape::Path {
            commands: cmds.iter().map(serialize_path_cmd).collect(),
        },
    }
}

fn serialize_path_cmd(c: &PathCommand) -> SerializedPathCommand {
    match c {
        PathCommand::MoveTo(x, y) => SerializedPathCommand::MoveTo(*x, *y),
        PathCommand::LineTo(x, y) => SerializedPathCommand::LineTo(*x, *y),
        PathCommand::QuadTo { cx, cy, x, y } => SerializedPathCommand::QuadTo {
            cx: *cx,
            cy: *cy,
            x: *x,
            y: *y,
        },
        PathCommand::CubicTo {
            c1x,
            c1y,
            c2x,
            c2y,
            x,
            y,
        } => SerializedPathCommand::CubicTo {
            c1x: *c1x,
            c1y: *c1y,
            c2x: *c2x,
            c2y: *c2y,
            x: *x,
            y: *y,
        },
        PathCommand::Close => SerializedPathCommand::Close,
    }
}

fn serialize_transition(t: &TransitionKind) -> SerializedTransition {
    match t {
        TransitionKind::None => SerializedTransition::None,
        TransitionKind::Opacity => SerializedTransition::Opacity,
        TransitionKind::Slide { edge } => SerializedTransition::Slide {
            edge: edge_to_u8(*edge),
        },
        TransitionKind::Scale { from } => SerializedTransition::Scale { from: *from },
        TransitionKind::Asymmetric { enter, exit } => SerializedTransition::Asymmetric {
            enter: Box::new(serialize_transition(enter)),
            exit: Box::new(serialize_transition(exit)),
        },
    }
}

// =============================================================================
// Deserialisation
// =============================================================================

/// Return the `nodes` slice in canonical post-order. The host renders by
/// replaying these through [`render_serialized`].
pub fn deserialize(frame: &ViewFrame) -> Vec<SerializedNode> {
    frame.nodes.clone()
}

// =============================================================================
// Render serialised frame
// =============================================================================

/// Render a deserialised frame on the host via [`RenderCtx`]. The
/// renderer interprets each [`SerializedNode`] enough to reconstruct the
/// equivalent draw command stream the live tree would produce.
///
/// Hit regions emitted by handler-bearing modifiers reference
/// `handler_token`s; the host's input pipeline ships those back as
/// [`EventPayload`]s.
pub fn render_serialized(
    nodes: &[SerializedNode],
    root: ViewNodeId,
    bounds: Rect,
    ctx: &mut RenderCtx,
) {
    let mut handler_regions = Vec::new();
    render_serialized_node(nodes, root, bounds, ctx, &mut handler_regions);
}

/// Reconstitute a [`crate::view::ImageLeaf`]'s `ContentFit` from its
/// wire-form byte. Public so the host's image fit pipeline can read the
/// fit flag without re-parsing the leaf payload.
pub fn fit_from_byte(b: u8) -> ContentFit {
    fit_from_u8(b)
}

/// Reconstitute a [`crate::view::AccessibilityRole`] from its wire-form
/// byte. Used by the host's a11y bridge to attach roles to layout
/// nodes.
pub fn role_from_byte(b: u8) -> AccessibilityRole {
    role_from_u8(b)
}

/// Reconstitute an [`crate::animation::Edge`] from its wire-form byte.
/// Used by the host's transition driver when replaying serialised
/// transitions.
pub fn edge_from_byte(b: u8) -> Edge {
    edge_from_u8(b)
}

/// Reconstitute an [`crate::animation::AnimationCurve`] from its
/// wire-form byte. Bezier and Spring lose their parameters in the round
/// trip — the host substitutes the SwiftUI defaults.
pub fn curve_from_byte(b: u8) -> AnimationCurve {
    curve_from_u8(b)
}

/// Hit-test region produced by serialised handler-bearing modifiers.
/// The host pairs the rect with a token; on input it ships
/// `(token, EventPayload)` to the peer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SerializedHitRegion {
    pub rect: LayoutRect,
    pub token: u32,
    pub kind: SerializedHitKind,
}

/// Gesture flavour carried by a [`SerializedHitRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedHitKind {
    Tap,
    Hover,
    LongPress { duration_ms: u32 },
    Drag,
    Appear,
    Disappear,
}

/// Render a serialised tree, additionally collecting hit-regions into
/// `regions` so the host can build its input map.
pub fn render_serialized_with_hits(
    nodes: &[SerializedNode],
    root: ViewNodeId,
    bounds: Rect,
    ctx: &mut RenderCtx,
    regions: &mut Vec<SerializedHitRegion>,
) {
    render_serialized_node(nodes, root, bounds, ctx, regions);
}

fn render_serialized_node(
    nodes: &[SerializedNode],
    id: ViewNodeId,
    bounds: Rect,
    ctx: &mut RenderCtx,
    regions: &mut Vec<SerializedHitRegion>,
) {
    let Some(node) = nodes.get(id as usize) else {
        return;
    };
    match &node.kind {
        SerializedKind::Leaf(leaf) => render_serialized_leaf(leaf, bounds, ctx),
        SerializedKind::Container { layout, .. } => {
            // Build a transient `Box<dyn Layout>` and reuse the live
            // render walker by routing each child through a manual
            // measure/place loop.
            render_serialized_container(layout, &node.children, nodes, bounds, ctx, regions);
        }
        SerializedKind::Modified { modifier } => {
            // Modified nodes have exactly one child.
            let child = node.children.first().copied().unwrap_or(0);
            render_serialized_modified(modifier, nodes, child, bounds, ctx, regions);
        }
        SerializedKind::Stateful => {
            // Stateful frames are flattened — render the (single) child
            // straight through.
            if let Some(child) = node.children.first() {
                render_serialized_node(nodes, *child, bounds, ctx, regions);
            }
        }
    }
}

fn render_serialized_leaf(leaf: &SerializedLeaf, bounds: Rect, ctx: &mut RenderCtx) {
    let lr = bounds.to_layout_rect();
    match leaf {
        SerializedLeaf::Text {
            content,
            font_size,
            color,
            ..
        } => {
            ctx.fill_text(&lr, content, *font_size, color_from_array(*color));
        }
        SerializedLeaf::Image { url, fit } => {
            // Fit byte is decoded so future versions can adjust the
            // draw bounds; current renderer treats `Contain` as the
            // implicit default.
            let _ = fit_from_u8(*fit);
            ctx.draw_image(&lr, url);
        }
        SerializedLeaf::Spacer { .. } => {}
        SerializedLeaf::Divider { color, .. } => {
            ctx.fill_rect(&lr, color_from_array(*color));
        }
        // Canvas placeholders deliberately render nothing — the peer
        // ships a separate bitmap upload.
        SerializedLeaf::Canvas { .. } => {}
    }
}

fn render_serialized_container(
    layout: &SerializedLayout,
    children: &[ViewNodeId],
    nodes: &[SerializedNode],
    bounds: Rect,
    ctx: &mut RenderCtx,
    regions: &mut Vec<SerializedHitRegion>,
) {
    use crate::layout::{Measured, Placement, Point, ProposedSize};

    // Reconstitute a live layout impl from the wire form.
    let live: Box<dyn Layout> = match layout {
        SerializedLayout::HStack { spacing, alignment } => Box::new(
            LHStack::new()
                .spacing(*spacing)
                .alignment(v_align_from_u8(*alignment)),
        ),
        SerializedLayout::VStack { spacing, alignment } => Box::new(
            LVStack::new()
                .spacing(*spacing)
                .alignment(h_align_from_u8(*alignment)),
        ),
        SerializedLayout::ZStack { alignment } => {
            Box::new(LZStack::new().alignment(unpack_alignment(*alignment)))
        }
        SerializedLayout::Frame {
            width,
            height,
            alignment,
        } => {
            let mut f = crate::layout::Frame::new();
            f.width = *width;
            f.height = *height;
            f.alignment = unpack_alignment(*alignment);
            Box::new(f)
        }
        // Grid doesn't have a stable Layout impl in Phase 5 — fall back
        // to a vertical stack so the children at least render.
        SerializedLayout::Grid { .. } => Box::new(LVStack::new()),
    };

    let proposal = ProposedSize::fixed(bounds.size.width, bounds.size.height);
    // Without re-walking the full tree we approximate measurements as
    // each child's offered slice; this matches the legacy renderer's
    // behaviour for simple stacks.
    let measured: Vec<Measured> = children
        .iter()
        .map(|_| {
            Measured::fixed(crate::layout::Size::new(
                bounds.size.width,
                bounds.size.height,
            ))
        })
        .collect();
    let mut placements: Vec<Placement> = measured
        .iter()
        .map(|m| Placement::new(Point::default(), m.ideal))
        .collect();
    live.place_subviews(bounds, proposal, &mut placements);
    for (cid, p) in children.iter().zip(placements.iter()) {
        let child_bounds = Rect {
            origin: p.origin,
            size: p.size,
        };
        render_serialized_node(nodes, *cid, child_bounds, ctx, regions);
    }
}

fn render_serialized_modified(
    modifier: &SerializedModifier,
    nodes: &[SerializedNode],
    child_id: ViewNodeId,
    bounds: Rect,
    ctx: &mut RenderCtx,
    regions: &mut Vec<SerializedHitRegion>,
) {
    let lr = bounds.to_layout_rect();
    match modifier {
        SerializedModifier::Padding {
            top,
            right,
            bottom,
            left,
        } => {
            let inner = Rect::new(
                bounds.origin.x + *left,
                bounds.origin.y + *top,
                (bounds.size.width - left - right).max(0.0),
                (bounds.size.height - top - bottom).max(0.0),
            );
            render_serialized_node(nodes, child_id, inner, ctx, regions);
        }
        SerializedModifier::Background(fill) => {
            paint_serialized_fill(fill, bounds, ctx);
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Foreground(_) => {
            // Foreground only affects text leaves; the wire form already
            // carries the resolved text colour, so this is a passthrough.
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Opacity(a) => {
            ctx.commands.push(DrawCommand::PushOpacity {
                alpha: a.clamp(0.0, 1.0),
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
            ctx.commands.push(DrawCommand::PopOpacity);
        }
        SerializedModifier::CornerRadius(_) => {
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Frame { width, height } => {
            let w = width.unwrap_or(bounds.size.width);
            let h = height.unwrap_or(bounds.size.height);
            let x = bounds.origin.x + (bounds.size.width - w) * 0.5;
            let y = bounds.origin.y + (bounds.size.height - h) * 0.5;
            render_serialized_node(nodes, child_id, Rect::new(x, y, w, h), ctx, regions);
        }
        SerializedModifier::FrameAligned {
            width,
            height,
            alignment,
        } => {
            let a = unpack_alignment(*alignment);
            let w = width.unwrap_or(bounds.size.width);
            let h = height.unwrap_or(bounds.size.height);
            let (x, y) = serialized_aligned_origin(bounds, w, h, a);
            render_serialized_node(nodes, child_id, Rect::new(x, y, w, h), ctx, regions);
        }
        SerializedModifier::FrameRange {
            min_width,
            ideal_width,
            max_width,
            min_height,
            ideal_height,
            max_height,
            alignment,
        } => {
            let a = unpack_alignment(*alignment);
            let pw = ideal_width.unwrap_or(bounds.size.width);
            let ph = ideal_height.unwrap_or(bounds.size.height);
            let w = pw
                .max(min_width.unwrap_or(0.0))
                .min(max_width.unwrap_or(f32::INFINITY));
            let h = ph
                .max(min_height.unwrap_or(0.0))
                .min(max_height.unwrap_or(f32::INFINITY));
            let (x, y) = serialized_aligned_origin(bounds, w, h, a);
            render_serialized_node(nodes, child_id, Rect::new(x, y, w, h), ctx, regions);
        }
        SerializedModifier::Border { width, color } => {
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
            ctx.stroke_rect(&lr, *width, color_from_array(*color));
        }
        SerializedModifier::Shadow {
            color,
            radius: _,
            offset_x,
            offset_y,
        } => {
            let shadow = LayoutRect::new(
                bounds.origin.x + offset_x,
                bounds.origin.y + offset_y,
                bounds.size.width,
                bounds.size.height,
            );
            ctx.fill_rect(&shadow, color_from_array(*color));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::InnerShadow {
            color,
            radius: _,
            offset_x,
            offset_y,
        } => {
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
            let inner = LayoutRect::new(
                bounds.origin.x + offset_x,
                bounds.origin.y + offset_y,
                bounds.size.width,
                bounds.size.height,
            );
            ctx.fill_rect(&inner, color_from_array(*color));
        }
        SerializedModifier::Rotation(_)
        | SerializedModifier::Scale { .. }
        | SerializedModifier::Offset { .. }
        | SerializedModifier::Position { .. }
        | SerializedModifier::Clip(_)
        | SerializedModifier::Mask(_) => {
            // Decorative-only on the host side; the modifier metadata is
            // preserved on the wire but the renderer just descends.
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnTap { handler_token } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::Tap,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnHover { handler_token } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::Hover,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnLongPress {
            duration_ms,
            handler_token,
        } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::LongPress {
                    duration_ms: *duration_ms,
                },
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnDrag { handler_token } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::Drag,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnAppear { handler_token } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::Appear,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::OnDisappear { handler_token } => {
            regions.push(SerializedHitRegion {
                rect: lr,
                token: *handler_token,
                kind: SerializedHitKind::Disappear,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AccessibilityLabel(label) => {
            push_serialized_annotation(ctx, bounds, |a| a.label = Some(label.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AccessibilityRole(role_byte) => {
            let role = role_from_u8(*role_byte);
            push_serialized_annotation(ctx, bounds, |a| a.role = Some(role));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AccessibilityHint(hint) => {
            push_serialized_annotation(ctx, bounds, |a| a.hint = Some(hint.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AccessibilityValue(v) => {
            push_serialized_annotation(ctx, bounds, |a| a.value = Some(v.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AccessibilityHidden(b) => {
            push_serialized_annotation(ctx, bounds, |a| a.hidden = Some(*b));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::TabIndex(i) => {
            push_serialized_annotation(ctx, bounds, |a| a.tab_index = Some(*i));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::AllowsHitTesting(b) => {
            push_serialized_annotation(ctx, bounds, |a| a.allows_hit_testing = Some(*b));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Tooltip(t) => {
            push_serialized_annotation(ctx, bounds, |a| a.tooltip = Some(t.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::ZIndex(z) => {
            push_serialized_annotation(ctx, bounds, |a| a.z_index = Some(*z));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Animation {
            duration_ms,
            delay_ms,
            curve,
            key,
        } => {
            let spec = crate::animation::AnimationSpec {
                curve: curve_from_u8(*curve),
                duration_ms: *duration_ms,
                delay_ms: *delay_ms,
                key: *key,
            };
            push_serialized_annotation(ctx, bounds, |a| a.animation = Some(spec));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Transition(t) => {
            let kind = transition_from_serialized(t);
            push_serialized_annotation(ctx, bounds, |a| a.transition = Some(kind.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Tag(s) => {
            push_serialized_annotation(ctx, bounds, |a| a.tag = Some(s.clone()));
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::Hidden(b) => {
            if !*b {
                render_serialized_node(nodes, child_id, bounds, ctx, regions);
            }
        }
        SerializedModifier::Shadows(stack) => {
            // Emit DropShadow draw commands back-to-front before the child.
            for (color, radius, offset_x, offset_y) in stack {
                let shadow_rect = LayoutRect::new(
                    bounds.origin.x + *offset_x,
                    bounds.origin.y + *offset_y,
                    bounds.size.width,
                    bounds.size.height,
                );
                ctx.commands.push(DrawCommand::DropShadow {
                    rect: shadow_rect,
                    color: color_from_array(*color),
                    radius: *radius,
                    offset_x: *offset_x,
                    offset_y: *offset_y,
                    inset: false,
                });
            }
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
        }
        SerializedModifier::BackdropFilter {
            blur_radius,
            saturate,
            brightness,
            contrast,
            grayscale,
            hue_rotate,
            invert,
            sepia,
        } => {
            let filter = crate::view::BackdropFilter {
                blur_radius: *blur_radius,
                saturate: *saturate,
                brightness: *brightness,
                contrast: *contrast,
                grayscale: *grayscale,
                hue_rotate: *hue_rotate,
                invert: *invert,
                sepia: *sepia,
            };
            ctx.commands.push(DrawCommand::PushBackdropFilter {
                rect: bounds.to_layout_rect(),
                filter,
            });
            render_serialized_node(nodes, child_id, bounds, ctx, regions);
            ctx.commands.push(DrawCommand::PopBackdropFilter);
        }
    }
}

fn paint_serialized_fill(fill: &SerializedFill, bounds: Rect, ctx: &mut RenderCtx) {
    let lr = bounds.to_layout_rect();
    match fill {
        SerializedFill::Color(c) => {
            ctx.fill_rect(&lr, color_from_array(*c));
        }
        SerializedFill::LinearGradient { stops, .. } => {
            if stops.is_empty() {
                return;
            }
            let n = stops.len();
            let h = bounds.size.height / n as f32;
            for (i, (_p, c)) in stops.iter().enumerate() {
                let strip = LayoutRect::new(
                    bounds.origin.x,
                    bounds.origin.y + h * i as f32,
                    bounds.size.width,
                    h,
                );
                ctx.fill_rect(&strip, color_from_array(*c));
            }
        }
        SerializedFill::RadialGradient { stops, .. } => {
            if stops.is_empty() {
                return;
            }
            let n = stops.len();
            let cx = bounds.origin.x + bounds.size.width * 0.5;
            let cy = bounds.origin.y + bounds.size.height * 0.5;
            for (i, (_p, c)) in stops.iter().enumerate() {
                let frac = (n - i) as f32 / n as f32;
                let rw = bounds.size.width * frac;
                let rh = bounds.size.height * frac;
                let ring = LayoutRect::new(cx - rw * 0.5, cy - rh * 0.5, rw, rh);
                ctx.fill_rect(&ring, color_from_array(*c));
            }
        }
    }
}

fn serialized_aligned_origin(bounds: Rect, w: f32, h: f32, alignment: Alignment2D) -> (f32, f32) {
    let x = match alignment.h {
        HAlignment::Leading => bounds.origin.x,
        HAlignment::Center => bounds.origin.x + (bounds.size.width - w) * 0.5,
        HAlignment::Trailing => bounds.origin.x + bounds.size.width - w,
    };
    let y = match alignment.v {
        VAlignment::Top | VAlignment::FirstTextBaseline => bounds.origin.y,
        VAlignment::Center => bounds.origin.y + (bounds.size.height - h) * 0.5,
        VAlignment::Bottom | VAlignment::LastTextBaseline => {
            bounds.origin.y + bounds.size.height - h
        }
    };
    (x, y)
}

fn color_from_array(c: [f32; 4]) -> Color {
    Color::rgba(c[0], c[1], c[2], c[3])
}

fn push_serialized_annotation(
    ctx: &mut RenderCtx,
    bounds: Rect,
    fill: impl FnOnce(&mut crate::render::Annotation),
) {
    let mut a = crate::render::Annotation {
        rect: bounds.to_layout_rect(),
        ..Default::default()
    };
    fill(&mut a);
    ctx.annotations.push(a);
}

fn transition_from_serialized(t: &SerializedTransition) -> TransitionKind {
    match t {
        SerializedTransition::None => TransitionKind::None,
        SerializedTransition::Opacity => TransitionKind::Opacity,
        SerializedTransition::Slide { edge } => TransitionKind::Slide {
            edge: edge_from_u8(*edge),
        },
        SerializedTransition::Scale { from } => TransitionKind::Scale { from: *from },
        SerializedTransition::Asymmetric { enter, exit } => TransitionKind::Asymmetric {
            enter: Box::new(transition_from_serialized(enter)),
            exit: Box::new(transition_from_serialized(exit)),
        },
    }
}

// =============================================================================
// Lower a SerializedNode list to AppDisplayCommand for legacy AppView
// callers (peer client → host backward-compat path).
// =============================================================================

/// Lower a serialised tree into a flat [`crate::view_compat::AppDisplayCommand`]
/// stream by re-rendering through a fresh [`RenderCtx`]. Used by
/// `nova_peer::PeerAppClient` to keep its `Option<AppView>` return type
/// while accepting wire frames produced by the new pipeline.
pub fn render_to_draw_commands(
    nodes: &[SerializedNode],
    root: ViewNodeId,
    bounds: Rect,
) -> Vec<DrawCommand> {
    let mut ctx = RenderCtx::new();
    render_serialized(nodes, root, bounds, &mut ctx);
    ctx.commands
}

// =============================================================================
// Diff engine
// =============================================================================

/// A patch produced by [`diff`] to take a host-cached [`ViewFrame`] from
/// the previous render to the current one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViewDelta {
    /// Replace a subtree at `id` with the given new subtree.
    Replace {
        id: ViewNodeId,
        with: Vec<SerializedNode>,
    },
    /// Patch a single node's modifier list / leaf props without changing
    /// structure.
    UpdateProps {
        id: ViewNodeId,
        kind: SerializedKind,
    },
    /// Insert a child at `parent_id` at index `at`.
    Insert {
        parent_id: ViewNodeId,
        at: u32,
        node: Vec<SerializedNode>,
    },
    /// Remove the child of `parent_id` at index `at`.
    Remove { parent_id: ViewNodeId, at: u32 },
    /// Move a child within the same parent.
    Move {
        parent_id: ViewNodeId,
        from: u32,
        to: u32,
    },
}

/// Diff two [`ViewFrame`]s, producing the minimal patch sequence.
///
/// The algorithm is structural: walks (parent, children) lists and emits
/// per-node patches. When the structural shape matches, prop diffs are
/// emitted as [`ViewDelta::UpdateProps`]; otherwise the whole subtree is
/// replaced via [`ViewDelta::Replace`] at the parent's slot. Reordering
/// uses the optional [`Modifier::Tag`] string carried by each node; for
/// unkeyed children the diff falls back to length-aligned per-index
/// comparisons.
pub fn diff(prev: &ViewFrame, next: &ViewFrame) -> Vec<ViewDelta> {
    let mut out = Vec::new();
    diff_subtree(prev, prev.root, next, next.root, &mut out);
    out
}

fn diff_subtree(
    prev: &ViewFrame,
    p_id: ViewNodeId,
    next: &ViewFrame,
    n_id: ViewNodeId,
    out: &mut Vec<ViewDelta>,
) {
    let p = &prev.nodes[p_id as usize];
    let n = &next.nodes[n_id as usize];

    // Different kinds → replace whole subtree at the new id (host
    // identifies subtrees by their slot in `nodes`).
    if !same_shape(&p.kind, &n.kind) {
        out.push(ViewDelta::Replace {
            id: n_id,
            with: extract_subtree(next, n_id),
        });
        return;
    }

    // Same kind — update props if leaf/modifier values differ; recurse
    // into children otherwise.
    if !same_props(&p.kind, &n.kind) {
        out.push(ViewDelta::UpdateProps {
            id: n_id,
            kind: n.kind.clone(),
        });
    }

    // Child reconciliation. Both lists are by-index; no reordering yet
    // unless we can match by Tag.
    let prev_children = &p.children;
    let next_children = &n.children;
    let pairs = match_children(prev, prev_children, next, next_children);
    let common = pairs
        .len()
        .min(prev_children.len())
        .min(next_children.len());
    for i in 0..common {
        let (pi, ni) = pairs[i];
        diff_subtree(prev, pi, next, ni, out);
    }
    // Trailing inserts.
    if next_children.len() > pairs.len() {
        for (i, ncid) in next_children.iter().enumerate().skip(pairs.len()) {
            out.push(ViewDelta::Insert {
                parent_id: n_id,
                at: i as u32,
                node: extract_subtree(next, *ncid),
            });
        }
    }
    // Trailing removes.
    if prev_children.len() > pairs.len() {
        // Remove from the end backwards so indices stay valid.
        for i in (pairs.len()..prev_children.len()).rev() {
            out.push(ViewDelta::Remove {
                parent_id: n_id,
                at: i as u32,
            });
        }
    }
}

/// Match prev/next children by Tag where possible, falling back to
/// position. Returns a Vec of (prev_id, next_id) pairs in next's order.
fn match_children(
    prev: &ViewFrame,
    prev_ids: &[ViewNodeId],
    next: &ViewFrame,
    next_ids: &[ViewNodeId],
) -> Vec<(ViewNodeId, ViewNodeId)> {
    let common = prev_ids.len().min(next_ids.len());
    // Tag-aware matching: build a tag → prev_id map, then prefer tag
    // matches when present. For unkeyed children, fall back to position.
    let mut prev_by_tag: HashMap<String, ViewNodeId> = HashMap::new();
    for pid in prev_ids {
        if let Some(tag) = node_tag(prev, *pid) {
            prev_by_tag.insert(tag, *pid);
        }
    }
    let mut used: Vec<bool> = vec![false; prev_ids.len()];
    let mut out = Vec::with_capacity(common);
    for (i, nid) in next_ids.iter().enumerate() {
        if i >= prev_ids.len() {
            break;
        }
        if let Some(tag) = node_tag(next, *nid) {
            if let Some(pid) = prev_by_tag.get(&tag) {
                if let Some(pos) = prev_ids.iter().position(|p| p == pid) {
                    if !used[pos] {
                        used[pos] = true;
                        out.push((*pid, *nid));
                        continue;
                    }
                }
            }
        }
        // Fall back to first unused slot at the same position, or the
        // first unused slot anywhere.
        if !used[i] {
            used[i] = true;
            out.push((prev_ids[i], *nid));
        } else if let Some(p) = used.iter().position(|u| !u) {
            used[p] = true;
            out.push((prev_ids[p], *nid));
        }
    }
    out
}

/// Return the tag string of a node if its outermost modifier is a
/// `Tag`. We only inspect the immediate node; arbitrarily deep tag
/// modifiers aren't required by the SwiftUI model.
fn node_tag(frame: &ViewFrame, id: ViewNodeId) -> Option<String> {
    let n = frame.nodes.get(id as usize)?;
    if let SerializedKind::Modified {
        modifier: SerializedModifier::Tag(s),
    } = &n.kind
    {
        Some(s.clone())
    } else {
        None
    }
}

fn same_shape(a: &SerializedKind, b: &SerializedKind) -> bool {
    match (a, b) {
        (SerializedKind::Leaf(la), SerializedKind::Leaf(lb)) => same_leaf_shape(la, lb),
        (SerializedKind::Container { .. }, SerializedKind::Container { .. }) => true,
        (SerializedKind::Modified { modifier: ma }, SerializedKind::Modified { modifier: mb }) => {
            same_modifier_shape(ma, mb)
        }
        (SerializedKind::Stateful, SerializedKind::Stateful) => true,
        _ => false,
    }
}

fn same_leaf_shape(a: &SerializedLeaf, b: &SerializedLeaf) -> bool {
    matches!(
        (a, b),
        (SerializedLeaf::Text { .. }, SerializedLeaf::Text { .. })
            | (SerializedLeaf::Image { .. }, SerializedLeaf::Image { .. })
            | (SerializedLeaf::Spacer { .. }, SerializedLeaf::Spacer { .. })
            | (
                SerializedLeaf::Divider { .. },
                SerializedLeaf::Divider { .. }
            )
            | (SerializedLeaf::Canvas { .. }, SerializedLeaf::Canvas { .. })
    )
}

fn same_modifier_shape(a: &SerializedModifier, b: &SerializedModifier) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn same_props(a: &SerializedKind, b: &SerializedKind) -> bool {
    a == b
}

/// Extract `id`'s subtree (post-order) into a fresh Vec, re-numbering
/// ids so they are dense within the slice.
fn extract_subtree(frame: &ViewFrame, id: ViewNodeId) -> Vec<SerializedNode> {
    let mut out = Vec::new();
    let mut id_map: HashMap<ViewNodeId, ViewNodeId> = HashMap::new();
    extract_into(frame, id, &mut out, &mut id_map);
    out
}

fn extract_into(
    frame: &ViewFrame,
    id: ViewNodeId,
    out: &mut Vec<SerializedNode>,
    id_map: &mut HashMap<ViewNodeId, ViewNodeId>,
) -> ViewNodeId {
    let node = &frame.nodes[id as usize];
    let new_children: Vec<ViewNodeId> = node
        .children
        .iter()
        .map(|c| extract_into(frame, *c, out, id_map))
        .collect();
    let new_id = out.len() as u32;
    id_map.insert(id, new_id);
    out.push(SerializedNode {
        id: new_id,
        kind: node.kind.clone(),
        children: new_children,
    });
    new_id
}

/// Apply a [`ViewDelta`] sequence to a host-cached [`ViewFrame`] in
/// place. After applying, the frame matches the next frame the peer
/// originally produced (modulo any post-order id renumbering inside
/// `Replace` payloads — the host re-bases ids when integrating).
pub fn apply_deltas(base: &mut ViewFrame, deltas: &[ViewDelta]) {
    for d in deltas {
        apply_delta(base, d);
    }
}

fn apply_delta(base: &mut ViewFrame, delta: &ViewDelta) {
    match delta {
        ViewDelta::Replace { id, with } => {
            // Replace the node at `id` with the subtree's root, then
            // append the rest of the subtree to base.nodes and rewrite
            // ids so they're disjoint from base.
            let offset = base.nodes.len() as u32;
            for n in with {
                let mut copy = n.clone();
                copy.id = n.id + offset;
                copy.children = n.children.iter().map(|c| c + offset).collect();
                base.nodes.push(copy);
            }
            // Rebase the new root to live under the old id. The walk
            // produces post-order so the last appended slot is the
            // subtree root — clone it out before mutating the target.
            let new_root_id = base.nodes.len().saturating_sub(1);
            let take = base.nodes[new_root_id].clone();
            if let Some(slot) = base.nodes.get_mut(*id as usize) {
                *slot = SerializedNode {
                    id: *id,
                    kind: take.kind,
                    children: take.children,
                };
            }
        }
        ViewDelta::UpdateProps { id, kind } => {
            if let Some(n) = base.nodes.get_mut(*id as usize) {
                n.kind = kind.clone();
            }
        }
        ViewDelta::Insert {
            parent_id,
            at,
            node,
        } => {
            let offset = base.nodes.len() as u32;
            let mut new_root = 0u32;
            for n in node {
                let mut copy = n.clone();
                copy.id = n.id + offset;
                copy.children = n.children.iter().map(|c| c + offset).collect();
                new_root = copy.id;
                base.nodes.push(copy);
            }
            if let Some(p) = base.nodes.get_mut(*parent_id as usize) {
                let at = (*at as usize).min(p.children.len());
                p.children.insert(at, new_root);
            }
        }
        ViewDelta::Remove { parent_id, at } => {
            if let Some(p) = base.nodes.get_mut(*parent_id as usize) {
                let at = *at as usize;
                if at < p.children.len() {
                    p.children.remove(at);
                }
            }
        }
        ViewDelta::Move {
            parent_id,
            from,
            to,
        } => {
            if let Some(p) = base.nodes.get_mut(*parent_id as usize) {
                let from = *from as usize;
                let to = *to as usize;
                if from < p.children.len() && to < p.children.len() {
                    let id = p.children.remove(from);
                    p.children.insert(to.min(p.children.len()), id);
                }
            }
        }
    }
}

// =============================================================================
// CBOR converters — bytes payload of a `nova_nwp::ViewSyncPayload`.
// =============================================================================

/// CBOR-encode a [`ViewFrame`] (uses ciborium, the workspace's canonical
/// CBOR codec).
pub fn view_frame_to_bytes(frame: &ViewFrame) -> Vec<u8> {
    let mut buf = Vec::new();
    // ciborium's writer is infallible against an in-memory `Vec<u8>`.
    ciborium::ser::into_writer(frame, &mut buf).expect("CBOR ViewFrame serialise");
    buf
}

/// CBOR-decode a [`ViewFrame`].
pub fn view_frame_from_bytes(bytes: &[u8]) -> Result<ViewFrame, NwpCodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| NwpCodecError(e.to_string()))
}

/// CBOR-encode a delta batch.
pub fn deltas_to_bytes(deltas: &[ViewDelta]) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(deltas, &mut buf).expect("CBOR ViewDelta serialise");
    buf
}

/// CBOR-decode a delta batch.
pub fn deltas_from_bytes(bytes: &[u8]) -> Result<Vec<ViewDelta>, NwpCodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| NwpCodecError(e.to_string()))
}

/// CBOR-encode an [`EventPayload`].
pub fn event_payload_to_bytes(p: &EventPayload) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(p, &mut buf).expect("CBOR EventPayload serialise");
    buf
}

/// CBOR-decode an [`EventPayload`].
pub fn event_payload_from_bytes(bytes: &[u8]) -> Result<EventPayload, NwpCodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| NwpCodecError(e.to_string()))
}

/// Codec error emitted by the public CBOR helpers. Wraps the underlying
/// `ciborium` error as a string so callers don't need to import
/// `ciborium::de::Error<std::io::Error>` in their function signatures.
#[derive(Debug, Clone)]
pub struct NwpCodecError(pub String);

impl std::fmt::Display for NwpCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nwp codec: {}", self.0)
    }
}

impl std::error::Error for NwpCodecError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::AnimationSpec;
    use crate::layout::EdgeInsets;
    use crate::view::{ImageLeaf, TextLeaf};
    use std::cell::Cell;

    fn text(s: &str) -> ViewNode {
        ViewNode::Leaf(LeafKind::Text(TextLeaf::new(s)))
    }

    fn padded(child: ViewNode, v: f32) -> ViewNode {
        ViewNode::Modified {
            modifier: Modifier::Padding(EdgeInsets::all(v)),
            child: Box::new(child),
        }
    }

    // ---------- Alignment / role helpers ----------

    #[test]
    fn alignment_pack_round_trip_default() {
        let a = Alignment2D::default();
        let p = pack_alignment(a);
        assert_eq!(unpack_alignment(p), a);
    }

    #[test]
    fn alignment_pack_round_trip_corners() {
        for h in [
            HAlignment::Leading,
            HAlignment::Center,
            HAlignment::Trailing,
        ] {
            for v in [
                VAlignment::Top,
                VAlignment::Center,
                VAlignment::Bottom,
                VAlignment::FirstTextBaseline,
                VAlignment::LastTextBaseline,
            ] {
                let a = Alignment2D::new(h, v);
                assert_eq!(unpack_alignment(pack_alignment(a)), a);
            }
        }
    }

    #[test]
    fn role_round_trip_button() {
        assert_eq!(
            role_from_u8(role_to_u8(AccessibilityRole::Button)),
            AccessibilityRole::Button
        );
    }

    #[test]
    fn fit_round_trip() {
        for f in [
            ContentFit::Fill,
            ContentFit::Contain,
            ContentFit::Cover,
            ContentFit::None,
            ContentFit::ScaleDown,
        ] {
            assert_eq!(fit_from_u8(fit_to_u8(f)), f);
        }
    }

    #[test]
    fn edge_round_trip() {
        for e in [Edge::Top, Edge::Bottom, Edge::Leading, Edge::Trailing] {
            assert_eq!(edge_from_u8(edge_to_u8(e)), e);
        }
    }

    #[test]
    fn curve_round_trip_simple() {
        for c in [
            AnimationCurve::Linear,
            AnimationCurve::EaseIn,
            AnimationCurve::EaseOut,
            AnimationCurve::EaseInOut,
        ] {
            assert!(matches!(
                curve_from_u8(curve_to_u8(c)),
                _c if std::mem::discriminant(&_c) == std::mem::discriminant(&c)
            ));
        }
    }

    // ---------- Serialise leaves ----------

    #[test]
    fn serialize_text_leaf_round_trip() {
        let n = text("hi");
        let f = serialize(&n);
        assert_eq!(f.nodes.len(), 1);
        match &f.nodes[0].kind {
            SerializedKind::Leaf(SerializedLeaf::Text { content, .. }) => {
                assert_eq!(content, "hi");
            }
            _ => panic!("expected text leaf"),
        }
    }

    #[test]
    fn serialize_image_leaf_carries_url_and_fit() {
        let n = ViewNode::Leaf(LeafKind::Image(ImageLeaf::new("a.png")));
        let f = serialize(&n);
        match &f.nodes[0].kind {
            SerializedKind::Leaf(SerializedLeaf::Image { url, fit }) => {
                assert_eq!(url, "a.png");
                assert_eq!(*fit, fit_to_u8(ContentFit::Contain));
            }
            _ => panic!("expected image"),
        }
    }

    #[test]
    fn serialize_spacer_leaf_carries_min_length() {
        let n = ViewNode::Leaf(LeafKind::Spacer { min_length: 8.0 });
        let f = serialize(&n);
        match &f.nodes[0].kind {
            SerializedKind::Leaf(SerializedLeaf::Spacer { min_length }) => {
                assert_eq!(*min_length, 8.0);
            }
            _ => panic!("expected spacer"),
        }
    }

    #[test]
    fn serialize_divider_leaf_carries_thickness_and_color() {
        let n = ViewNode::Leaf(LeafKind::Divider {
            thickness: 2.0,
            color: Color::WHITE,
        });
        let f = serialize(&n);
        match &f.nodes[0].kind {
            SerializedKind::Leaf(SerializedLeaf::Divider { thickness, color }) => {
                assert_eq!(*thickness, 2.0);
                assert_eq!(*color, [1.0, 1.0, 1.0, 1.0]);
            }
            _ => panic!("expected divider"),
        }
    }

    #[test]
    fn serialize_canvas_leaf_emits_placeholder() {
        let n = ViewNode::Leaf(LeafKind::Canvas(Rc::new(|_, _| {})));
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[0].kind,
            SerializedKind::Leaf(SerializedLeaf::Canvas { .. })
        ));
    }

    // ---------- Serialise containers ----------

    #[test]
    fn serialize_hstack_container_carries_spacing() {
        let n = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(7.5)),
            children: vec![text("a"), text("b")],
        };
        let f = serialize(&n);
        // Root is the last node (post-order). Children land before the
        // root so root id is `f.root` and points into the last slot.
        let root = &f.nodes[f.root as usize];
        match &root.kind {
            SerializedKind::Container {
                layout,
                children_count,
            } => {
                assert_eq!(*children_count, 2);
                assert!(
                    matches!(layout, SerializedLayout::HStack { spacing, .. } if (*spacing - 7.5).abs() < 1e-6)
                );
            }
            _ => panic!("expected container"),
        }
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn serialize_vstack_container_carries_alignment() {
        let n = ViewNode::Container {
            layout: Box::new(LVStack::new().alignment(HAlignment::Trailing)),
            children: vec![text("a")],
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Container {
            layout: SerializedLayout::VStack { alignment, .. },
            ..
        } = &root.kind
        {
            assert_eq!(*alignment, h_align_to_u8(HAlignment::Trailing));
        } else {
            panic!("expected VStack container");
        }
    }

    #[test]
    fn serialize_zstack_container() {
        let n = ViewNode::Container {
            layout: Box::new(LZStack::new()),
            children: Vec::new(),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        assert!(matches!(
            &root.kind,
            SerializedKind::Container {
                layout: SerializedLayout::ZStack { .. },
                ..
            }
        ));
    }

    #[test]
    fn serialize_frame_layout_container_round_trips_dims() {
        let mut frame_layout = crate::layout::Frame::new();
        frame_layout.width = Some(100.0);
        frame_layout.height = Some(50.0);
        let n = ViewNode::Container {
            layout: Box::new(frame_layout),
            children: vec![text("x")],
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Container {
            layout: SerializedLayout::Frame { width, height, .. },
            ..
        } = &root.kind
        {
            assert_eq!(*width, Some(100.0));
            assert_eq!(*height, Some(50.0));
        } else {
            panic!("expected Frame layout");
        }
    }

    // ---------- Serialise modifiers ----------

    #[test]
    fn serialize_padding_modifier_carries_edges() {
        let n = padded(text("x"), 4.0);
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Modified {
            modifier:
                SerializedModifier::Padding {
                    top,
                    right,
                    bottom,
                    left,
                },
        } = &root.kind
        {
            assert_eq!(*top, 4.0);
            assert_eq!(*right, 4.0);
            assert_eq!(*bottom, 4.0);
            assert_eq!(*left, 4.0);
        } else {
            panic!("expected padding modifier");
        }
    }

    #[test]
    fn serialize_background_color_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        assert!(matches!(
            &root.kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Background(SerializedFill::Color(_)),
            }
        ));
    }

    #[test]
    fn serialize_background_linear_gradient_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::LinearGradient {
                angle_deg: 90.0,
                stops: vec![(0.0, Color::WHITE), (1.0, Color::BLACK)],
            }),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Background(SerializedFill::LinearGradient { stops, .. }),
        } = &root.kind
        {
            assert_eq!(stops.len(), 2);
        } else {
            panic!("expected linear gradient");
        }
    }

    #[test]
    fn serialize_foreground_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Foreground(Color::BLACK),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        assert!(matches!(
            &root.kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Foreground(_),
            }
        ));
    }

    #[test]
    fn serialize_opacity_modifier_carries_value() {
        let n = ViewNode::Modified {
            modifier: Modifier::Opacity(0.5),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Opacity(v),
        } = &root.kind
        {
            assert!((*v - 0.5).abs() < 1e-6);
        } else {
            panic!("expected opacity");
        }
    }

    #[test]
    fn serialize_corner_radius_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::CornerRadius(8.0),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        assert!(matches!(
            &root.kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::CornerRadius(r),
            } if (*r - 8.0).abs() < 1e-6
        ));
    }

    #[test]
    fn serialize_frame_modifier_carries_dims() {
        let n = ViewNode::Modified {
            modifier: Modifier::Frame {
                width: Some(50.0),
                height: None,
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let root = &f.nodes[f.root as usize];
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Frame { width, height },
        } = &root.kind
        {
            assert_eq!(*width, Some(50.0));
            assert_eq!(*height, None);
        } else {
            panic!("expected frame");
        }
    }

    #[test]
    fn serialize_frame_aligned_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::FrameAligned {
                width: Some(100.0),
                height: Some(20.0),
                alignment: Alignment2D::default(),
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::FrameAligned { .. },
            }
        ));
    }

    #[test]
    fn serialize_frame_range_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::FrameRange {
                min_width: Some(10.0),
                ideal_width: Some(20.0),
                max_width: Some(30.0),
                min_height: None,
                ideal_height: None,
                max_height: None,
                alignment: Alignment2D::default(),
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::FrameRange { .. },
            }
        ));
    }

    #[test]
    fn serialize_border_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Border {
                width: 2.0,
                color: Color::WHITE,
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Border { width, .. },
        } = &f.nodes[f.root as usize].kind
        {
            assert!((*width - 2.0).abs() < 1e-6);
        } else {
            panic!("expected border");
        }
    }

    #[test]
    fn serialize_shadow_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Shadow {
                color: Color::BLACK,
                radius: 4.0,
                offset_x: 1.0,
                offset_y: 2.0,
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Shadow { .. },
            }
        ));
    }

    #[test]
    fn serialize_inner_shadow_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::InnerShadow {
                color: Color::WHITE,
                radius: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::InnerShadow { .. },
            }
        ));
    }

    #[test]
    fn serialize_rotation_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Rotation(45.0),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Rotation(_),
            }
        ));
    }

    #[test]
    fn serialize_scale_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Scale {
                x: 2.0,
                y: 3.0,
                anchor: Alignment2D::default(),
            },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Scale { .. },
            }
        ));
    }

    #[test]
    fn serialize_offset_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Offset { x: 1.0, y: 2.0 },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Offset { .. },
            }
        ));
    }

    #[test]
    fn serialize_position_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Position { x: 5.0, y: 6.0 },
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Position { .. },
            }
        ));
    }

    #[test]
    fn serialize_clip_rect_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Clip(ClipShape::Rect),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Clip(SerializedClipShape::Rect),
            }
        ));
    }

    #[test]
    fn serialize_clip_rounded_rect_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Clip(ClipShape::RoundedRect { radius: 4.0 }),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Clip(SerializedClipShape::RoundedRect { .. }),
            }
        ));
    }

    #[test]
    fn serialize_clip_path_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Clip(ClipShape::Path(vec![
                PathCommand::MoveTo(0.0, 0.0),
                PathCommand::LineTo(10.0, 0.0),
                PathCommand::Close,
            ])),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Clip(SerializedClipShape::Path { commands }),
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(commands.len(), 3);
        } else {
            panic!("expected path clip");
        }
    }

    #[test]
    fn serialize_mask_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Mask(ClipShape::Circle),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Mask(SerializedClipShape::Circle),
            }
        ));
    }

    #[test]
    fn serialize_on_tap_modifier_registers_token() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnTap(Rc::new(|| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        let f = serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::OnTap { .. },
            }
        ));
    }

    #[test]
    fn serialize_on_hover_modifier_registers_token() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnHover(Rc::new(|_| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
    }

    #[test]
    fn serialize_on_long_press_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnLongPress {
                duration_ms: 500,
                action: Rc::new(|| {}),
            },
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        let f = serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::OnLongPress { duration_ms, .. },
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(*duration_ms, 500);
        } else {
            panic!("expected long press");
        }
    }

    #[test]
    fn serialize_on_drag_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnDrag(Rc::new(|_, _| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
    }

    #[test]
    fn serialize_on_appear_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnAppear(Rc::new(|| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
    }

    #[test]
    fn serialize_on_disappear_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnDisappear(Rc::new(|| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        serialize_with_handlers(&n, &mut handlers, 0);
        assert_eq!(handlers.len(), 1);
    }

    #[test]
    fn serialize_accessibility_label_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::AccessibilityLabel("Submit".into()),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::AccessibilityLabel(s),
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(s, "Submit");
        } else {
            panic!("expected label");
        }
    }

    #[test]
    fn serialize_accessibility_role_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::AccessibilityRole(AccessibilityRole::Button),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::AccessibilityRole(b),
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(*b, role_to_u8(AccessibilityRole::Button));
        } else {
            panic!("expected role");
        }
    }

    #[test]
    fn serialize_animation_modifier() {
        let spec = AnimationSpec::new(AnimationCurve::EaseIn, 250).keyed(7);
        let n = ViewNode::Modified {
            modifier: Modifier::Animation(spec),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier:
                SerializedModifier::Animation {
                    duration_ms, key, ..
                },
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(*duration_ms, 250);
            assert_eq!(*key, 7);
        } else {
            panic!("expected animation");
        }
    }

    #[test]
    fn serialize_transition_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Transition(TransitionKind::Slide { edge: Edge::Top }),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Transition(SerializedTransition::Slide { .. }),
            }
        ));
    }

    #[test]
    fn serialize_tag_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Tag("greeting".into()),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        if let SerializedKind::Modified {
            modifier: SerializedModifier::Tag(s),
        } = &f.nodes[f.root as usize].kind
        {
            assert_eq!(s, "greeting");
        } else {
            panic!("expected tag");
        }
    }

    #[test]
    fn serialize_hidden_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Hidden(true),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Hidden(true),
            }
        ));
    }

    #[test]
    fn serialize_tab_index_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::TabIndex(3),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::TabIndex(3),
            }
        ));
    }

    #[test]
    fn serialize_z_index_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::ZIndex(2.0),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::ZIndex(_),
            }
        ));
    }

    #[test]
    fn serialize_tooltip_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::Tooltip("hover me".into()),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::Tooltip(_),
            }
        ));
    }

    #[test]
    fn serialize_accessibility_hidden_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::AccessibilityHidden(true),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::AccessibilityHidden(true),
            }
        ));
    }

    #[test]
    fn serialize_allows_hit_testing_modifier() {
        let n = ViewNode::Modified {
            modifier: Modifier::AllowsHitTesting(false),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        assert!(matches!(
            &f.nodes[f.root as usize].kind,
            SerializedKind::Modified {
                modifier: SerializedModifier::AllowsHitTesting(false),
            }
        ));
    }

    // ---------- Stateful flatten ----------

    #[test]
    fn serialize_flattens_stateful_node() {
        let n = ViewNode::Stateful(Rc::new(|| text("evaluated")));
        let f = serialize(&n);
        // Three nodes: text, stateful wrapper, plus root id is the
        // outermost stateful slot.
        assert!(f.nodes.len() >= 2);
        let root = &f.nodes[f.root as usize];
        assert!(matches!(root.kind, SerializedKind::Stateful));
    }

    // ---------- Round-trip via deserialize ----------

    #[test]
    fn deserialize_returns_clone_of_nodes() {
        let n = padded(text("hello"), 4.0);
        let f = serialize(&n);
        let nodes = deserialize(&f);
        assert_eq!(nodes.len(), f.nodes.len());
        assert_eq!(nodes, f.nodes);
    }

    // ---------- HandlerTable ----------

    #[test]
    fn handler_table_register_returns_unique_token() {
        let mut t = HandlerTable::new();
        let a = t.register_tap(|| {});
        let b = t.register_tap(|| {});
        assert_ne!(a, b);
    }

    #[test]
    fn handler_table_dispatch_invokes_tap() {
        let counter = Rc::new(Cell::new(0));
        let c = Rc::clone(&counter);
        let mut t = HandlerTable::new();
        let token = t.register_tap(move || c.set(c.get() + 1));
        t.dispatch(token, EventPayload::Tap { x: 0.0, y: 0.0 });
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn handler_table_dispatch_invokes_hover() {
        let last = Rc::new(Cell::new(false));
        let l = Rc::clone(&last);
        let mut t = HandlerTable::new();
        let token = t.register_hover(move |b| l.set(b));
        t.dispatch(
            token,
            EventPayload::Hover {
                entering: true,
                x: 0.0,
                y: 0.0,
            },
        );
        assert!(last.get());
    }

    #[test]
    fn handler_table_dispatch_invokes_drag() {
        let last = Rc::new(Cell::new((0.0, 0.0)));
        let l = Rc::clone(&last);
        let mut t = HandlerTable::new();
        let token = t.register_drag(move |dx, dy| l.set((dx, dy)));
        t.dispatch(token, EventPayload::Drag { dx: 3.0, dy: 4.0 });
        assert_eq!(last.get(), (3.0, 4.0));
    }

    #[test]
    fn handler_table_dispatch_invokes_long_press() {
        let count = Rc::new(Cell::new(0));
        let c = Rc::clone(&count);
        let mut t = HandlerTable::new();
        let token = t.register_long_press(move || c.set(c.get() + 1));
        t.dispatch(token, EventPayload::LongPress { x: 0.0, y: 0.0 });
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn handler_table_dispatch_invokes_lifecycle_appear() {
        let count = Rc::new(Cell::new(0));
        let c = Rc::clone(&count);
        let mut t = HandlerTable::new();
        let token = t.register_lifecycle(move || c.set(c.get() + 1));
        t.dispatch(token, EventPayload::Appear);
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn handler_table_dispatch_invokes_lifecycle_disappear() {
        let count = Rc::new(Cell::new(0));
        let c = Rc::clone(&count);
        let mut t = HandlerTable::new();
        let token = t.register_lifecycle(move || c.set(c.get() + 1));
        t.dispatch(token, EventPayload::Disappear);
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn handler_table_dispatch_unknown_token_is_noop() {
        let t = HandlerTable::new();
        // Should not panic.
        t.dispatch(999, EventPayload::Tap { x: 0.0, y: 0.0 });
    }

    #[test]
    fn handler_table_dispatch_mismatched_payload_is_noop() {
        let count = Rc::new(Cell::new(0));
        let c = Rc::clone(&count);
        let mut t = HandlerTable::new();
        let token = t.register_tap(move || c.set(c.get() + 1));
        // Drag payload should not invoke a tap handler.
        t.dispatch(token, EventPayload::Drag { dx: 0.0, dy: 0.0 });
        assert_eq!(count.get(), 0);
    }

    #[test]
    fn handler_table_clear_resets_counter() {
        let mut t = HandlerTable::new();
        let _ = t.register_tap(|| {});
        let _ = t.register_tap(|| {});
        t.clear();
        assert!(t.is_empty());
        // Next token re-starts at 0.
        let again = t.register_tap(|| {});
        assert_eq!(again, 0);
    }

    #[test]
    fn handler_table_tokens_are_monotonic() {
        let mut t = HandlerTable::new();
        for expected in 0u32..32 {
            assert_eq!(t.register_tap(|| {}), expected);
        }
    }

    #[test]
    fn handler_table_contains_returns_true_for_registered() {
        let mut t = HandlerTable::new();
        let token = t.register_tap(|| {});
        assert!(t.contains(token));
        assert!(!t.contains(token + 1));
    }

    // ---------- CBOR ----------

    #[test]
    fn cbor_view_frame_round_trip() {
        let n = padded(text("hi"), 4.0);
        let f = serialize(&n);
        let bytes = view_frame_to_bytes(&f);
        let back = view_frame_from_bytes(&bytes).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn cbor_view_frame_decode_error_returns_err() {
        let bad = vec![0xff, 0xff, 0xff];
        let r = view_frame_from_bytes(&bad);
        assert!(r.is_err());
    }

    #[test]
    fn cbor_deltas_round_trip() {
        let d = vec![ViewDelta::UpdateProps {
            id: 0,
            kind: SerializedKind::Stateful,
        }];
        let bytes = deltas_to_bytes(&d);
        let back = deltas_from_bytes(&bytes).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn cbor_event_payload_round_trip_tap() {
        let p = EventPayload::Tap { x: 1.0, y: 2.0 };
        let bytes = event_payload_to_bytes(&p);
        let back = event_payload_from_bytes(&bytes).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn cbor_event_payload_round_trip_drag() {
        let p = EventPayload::Drag { dx: 5.0, dy: 6.0 };
        let bytes = event_payload_to_bytes(&p);
        let back = event_payload_from_bytes(&bytes).unwrap();
        assert_eq!(back, p);
    }

    // ---------- Render serialised ----------

    #[test]
    fn render_serialized_text_emits_text_command() {
        let f = serialize(&text("hi"));
        let mut ctx = RenderCtx::new();
        render_serialized(&f.nodes, f.root, Rect::new(0.0, 0.0, 100.0, 20.0), &mut ctx);
        assert!(
            ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::Text { .. }))
        );
    }

    #[test]
    fn render_serialized_image_emits_image_command() {
        let n = ViewNode::Leaf(LeafKind::Image(ImageLeaf::new("a.png")));
        let f = serialize(&n);
        let mut ctx = RenderCtx::new();
        render_serialized(&f.nodes, f.root, Rect::new(0.0, 0.0, 64.0, 64.0), &mut ctx);
        assert!(
            ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::Image { .. }))
        );
    }

    #[test]
    fn render_serialized_padding_insets_child() {
        let f = serialize(&padded(text("x"), 10.0));
        let mut ctx = RenderCtx::new();
        render_serialized(
            &f.nodes,
            f.root,
            Rect::new(0.0, 0.0, 100.0, 100.0),
            &mut ctx,
        );
        let DrawCommand::Text { rect, .. } = ctx
            .commands
            .iter()
            .find(|c| matches!(c, DrawCommand::Text { .. }))
            .expect("text")
        else {
            unreachable!()
        };
        assert_eq!(rect.x, 10.0);
        assert_eq!(rect.y, 10.0);
    }

    #[test]
    fn render_serialized_background_emits_fill_before_text() {
        let n = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let mut ctx = RenderCtx::new();
        render_serialized(&f.nodes, f.root, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        assert!(matches!(ctx.commands[0], DrawCommand::FillRect { .. }));
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::Text { .. }
        ));
    }

    #[test]
    fn render_serialized_collects_tap_hit_region() {
        let n = ViewNode::Modified {
            modifier: Modifier::OnTap(Rc::new(|| {})),
            child: Box::new(text("x")),
        };
        let mut handlers = HandlerTable::new();
        let f = serialize_with_handlers(&n, &mut handlers, 0);
        let mut ctx = RenderCtx::new();
        let mut hits = Vec::new();
        render_serialized_with_hits(
            &f.nodes,
            f.root,
            Rect::new(0.0, 0.0, 50.0, 50.0),
            &mut ctx,
            &mut hits,
        );
        assert_eq!(hits.len(), 1);
        assert!(matches!(hits[0].kind, SerializedHitKind::Tap));
    }

    #[test]
    fn render_serialized_hidden_skips_subtree() {
        let n = ViewNode::Modified {
            modifier: Modifier::Hidden(true),
            child: Box::new(text("x")),
        };
        let f = serialize(&n);
        let mut ctx = RenderCtx::new();
        render_serialized(&f.nodes, f.root, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        assert!(
            ctx.commands
                .iter()
                .all(|c| !matches!(c, DrawCommand::Text { .. }))
        );
    }

    #[test]
    fn render_to_draw_commands_returns_command_stream() {
        let f = serialize(&text("abc"));
        let cmds = render_to_draw_commands(&f.nodes, f.root, Rect::new(0.0, 0.0, 50.0, 20.0));
        assert!(!cmds.is_empty());
    }

    // ---------- Diff ----------

    #[test]
    fn diff_identical_trees_emits_empty() {
        let a = serialize(&text("hi"));
        let b = serialize(&text("hi"));
        assert!(diff(&a, &b).is_empty());
    }

    #[test]
    fn diff_text_change_emits_update_props() {
        let a = serialize(&text("hi"));
        let b = serialize(&text("bye"));
        let d = diff(&a, &b);
        assert_eq!(d.len(), 1);
        assert!(matches!(d[0], ViewDelta::UpdateProps { .. }));
    }

    #[test]
    fn diff_kind_change_emits_replace() {
        let a = serialize(&text("hi"));
        let b = serialize(&ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
        let d = diff(&a, &b);
        assert!(d.iter().any(|x| matches!(x, ViewDelta::Replace { .. })));
    }

    #[test]
    fn diff_insert_child_emits_insert() {
        let a = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a")],
        });
        let b = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a"), text("b")],
        });
        let d = diff(&a, &b);
        assert!(d.iter().any(|x| matches!(x, ViewDelta::Insert { .. })));
    }

    #[test]
    fn diff_remove_child_emits_remove() {
        let a = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a"), text("b")],
        });
        let b = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a")],
        });
        let d = diff(&a, &b);
        assert!(d.iter().any(|x| matches!(x, ViewDelta::Remove { .. })));
    }

    #[test]
    fn diff_tag_match_keeps_position() {
        // Same tagged element in different positions — diff still
        // matches them.
        let make = |tags: &[&str]| ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: tags
                .iter()
                .map(|t| ViewNode::Modified {
                    modifier: Modifier::Tag((*t).into()),
                    child: Box::new(text("x")),
                })
                .collect(),
        };
        let a = serialize(&make(&["a", "b"]));
        let b = serialize(&make(&["b", "a"]));
        let d = diff(&a, &b);
        // Either Move (full reorder) or no diff (exact equality through
        // tag matching). Both indicate the tags were considered.
        assert!(d.iter().all(|x| !matches!(x, ViewDelta::Replace { .. })));
    }

    // ---------- apply_deltas ----------

    #[test]
    fn apply_update_props_mutates_kind() {
        let mut a = serialize(&text("hi"));
        let root = a.root;
        let new_kind = SerializedKind::Leaf(SerializedLeaf::Text {
            content: "world".into(),
            font_size: 14.0,
            color: [1.0, 1.0, 1.0, 1.0],
            bold: false,
            italic: false,
            underline: false,
        });
        apply_deltas(
            &mut a,
            &[ViewDelta::UpdateProps {
                id: root,
                kind: new_kind.clone(),
            }],
        );
        assert_eq!(a.nodes[root as usize].kind, new_kind);
    }

    #[test]
    fn apply_remove_drops_child() {
        let mut a = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a"), text("b")],
        });
        let parent = a.root;
        apply_deltas(
            &mut a,
            &[ViewDelta::Remove {
                parent_id: parent,
                at: 0,
            }],
        );
        assert_eq!(a.nodes[parent as usize].children.len(), 1);
    }

    #[test]
    fn apply_move_reorders_children() {
        let mut a = serialize(&ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![text("a"), text("b"), text("c")],
        });
        let parent = a.root;
        let original = a.nodes[parent as usize].children.clone();
        apply_deltas(
            &mut a,
            &[ViewDelta::Move {
                parent_id: parent,
                from: 0,
                to: 2,
            }],
        );
        let after = &a.nodes[parent as usize].children;
        assert_eq!(after.len(), original.len());
        assert_eq!(after[2], original[0]);
    }

    // ---------- Generation counter ----------

    #[test]
    fn serialize_with_handlers_stamps_generation() {
        let n = text("x");
        let mut h = HandlerTable::new();
        let f = serialize_with_handlers(&n, &mut h, 42);
        assert_eq!(f.generation, 42);
    }

    // ---------- CBOR overhead vs JSON (rough) ----------

    #[test]
    fn cbor_is_smaller_than_json_for_typical_frame() {
        let n = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(4.0)),
            children: vec![
                text("Hello"),
                text("World"),
                ViewNode::Modified {
                    modifier: Modifier::Padding(EdgeInsets::all(8.0)),
                    child: Box::new(text("Padded")),
                },
            ],
        };
        let f = serialize(&n);
        let cbor = view_frame_to_bytes(&f);
        let json = serde_json::to_vec(&f).unwrap();
        // CBOR should be smaller than JSON for typical frames — this
        // catches obvious ciborium misconfigurations.
        assert!(
            cbor.len() < json.len(),
            "CBOR ({} bytes) should be < JSON ({} bytes)",
            cbor.len(),
            json.len()
        );
    }

    #[test]
    fn cbor_overhead_report_typical_counter_app() {
        // A counter-style view: padded text wrapped in OnTap.
        let mut handlers = HandlerTable::new();
        let n = ViewNode::Modified {
            modifier: Modifier::OnTap(Rc::new(|| {})),
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Padding(EdgeInsets::all(12.0)),
                child: Box::new(ViewNode::Modified {
                    modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                    child: Box::new(text("count=42")),
                }),
            }),
        };
        let f = serialize_with_handlers(&n, &mut handlers, 0);
        let cbor = view_frame_to_bytes(&f);
        let json = serde_json::to_vec(&f).unwrap();
        // Sanity: typical counter-app frames land around 35–45% smaller
        // in CBOR than in JSON. Loosen the bound to "at least 15%
        // smaller" so the assertion stays robust as the wire shape
        // evolves.
        let ratio = (cbor.len() as f32) / (json.len() as f32);
        assert!(
            ratio < 0.85,
            "expected CBOR to be at least 15% smaller than JSON; cbor={}, json={}, ratio={ratio:.2}",
            cbor.len(),
            json.len()
        );
        // Numbers below come from this very test; documented in
        // docs/crates/photon.md "Wire CBOR layout" section.
        eprintln!(
            "[nwp size report] counter-app frame: cbor={} bytes, json={} bytes, ratio={:.2}",
            cbor.len(),
            json.len(),
            ratio
        );
    }
}
