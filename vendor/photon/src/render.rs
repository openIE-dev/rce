//! Render context — the immediate-mode surface widgets push commands into.
//!
//! This module hosts both the legacy [`RenderCtx`]/[`DrawCommand`]
//! machinery used by the existing widget API and the new [`render_node`]
//! walker that drives the declarative [`crate::view::ViewNode`] tree.

use serde::{Deserialize, Serialize};
use std::rc::Rc;

use crate::animation::TransitionKind;
pub use crate::animation::{AnimationCurve, AnimationSpec};
use crate::color::Color;
use crate::layout::{
    Alignment2D, EdgeInsets, HAlignment, LayoutRect, Measured, Placement, Point, ProposedSize,
    Rect, Size, VAlignment,
};
use crate::view::{
    AccessibilityRole, BackdropFilter, BackgroundFill, ClipShape, LeafKind, Modifier, PathCommand,
    TextLeaf, ViewNode,
};

/// Affine 2x3 transform: `[a, b, c, d, tx, ty]`. Applied as
/// `x' = a*x + c*y + tx`, `y' = b*x + d*y + ty`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform2D(pub [f32; 6]);

impl Transform2D {
    pub fn identity() -> Self {
        Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    }

    pub fn translate(tx: f32, ty: f32) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, tx, ty])
    }

    pub fn rotate_around(degrees: f32, cx: f32, cy: f32) -> Self {
        let r = degrees.to_radians();
        let (s, c) = (r.sin(), r.cos());
        // Translate(cx, cy) * Rotate(r) * Translate(-cx, -cy)
        let tx = cx - c * cx + s * cy;
        let ty = cy - s * cx - c * cy;
        Self([c, s, -s, c, tx, ty])
    }

    pub fn scale_around(sx: f32, sy: f32, cx: f32, cy: f32) -> Self {
        // Translate(cx, cy) * Scale(sx, sy) * Translate(-cx, -cy)
        Self([sx, 0.0, 0.0, sy, cx - sx * cx, cy - sy * cy])
    }
}

/// Internal command form. Widgets push these via [`RenderCtx`] helpers; the
/// host caller pulls primitives via [`RenderCtx::to_primitives`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrawCommand {
    FillRect {
        rect: LayoutRect,
        color: Color,
    },
    StrokeRect {
        rect: LayoutRect,
        width: f32,
        color: Color,
    },
    Text {
        rect: LayoutRect,
        text: String,
        font_size: f32,
        color: Color,
    },
    Image {
        rect: LayoutRect,
        url: String,
    },
    PushClip {
        rect: LayoutRect,
    },
    PopClip,
    /// Push a clip restricted to a non-rectangular shape. Serialized form
    /// only carries the rect bounding box plus a discriminant tag — the
    /// in-process renderer walks the live shape via the
    /// [`RenderCtx::clip_shapes`] sidetable. Phase 4 will replace this with
    /// a stencil-based path clip.
    PushClipShape {
        rect: LayoutRect,
        shape: ClipShapeKind,
    },
    /// Pop the matching `PushClipShape` off the clip stack.
    PopClipShape,
    /// Push a global opacity multiplier onto the alpha stack.
    PushOpacity {
        alpha: f32,
    },
    /// Pop the matching `PushOpacity`.
    PopOpacity,
    /// Push a 2D affine transform onto the active transform stack.
    PushTransform {
        transform: Transform2D,
    },
    /// Pop the matching `PushTransform`.
    PopTransform,
    /// Drop shadow drawn behind the next child. Carries the full Gaussian
    /// shadow parameters; the GPU bridge runs the real two-pass blur via
    /// `nova_gpu::effects::BlurPipeline`. The pure-CPU display-list walker
    /// still emits a falloff approximation as a cluster of layered
    /// [`DrawCommand::FillRect`]s for back-compat (see `emit_shadow_layers`).
    DropShadow {
        rect: LayoutRect,
        color: Color,
        radius: f32,
        offset_x: f32,
        offset_y: f32,
        /// `true` for `Modifier::InnerShadow` — caller is expected to clip
        /// to the child's shape before the shadow paints.
        inset: bool,
    },
    /// Push a backdrop-filter onto the next subtree. The GPU bridge captures
    /// what's currently rendered behind `rect`, runs the blur + colour
    /// matrix chain, and composites it back inside `rect`. CPU bridges
    /// without offscreen support degrade to passthrough — the modifier is
    /// recorded in the command stream so external consumers (e.g. the JWP
    /// peer renderer) can still see it.
    PushBackdropFilter {
        rect: LayoutRect,
        filter: BackdropFilter,
    },
    /// Pop the matching `PushBackdropFilter`.
    PopBackdropFilter,
}

/// Serializable discriminant for [`DrawCommand::PushClipShape`]. The full
/// non-Serialize [`ClipShape`] (path commands and all) lives in
/// [`RenderCtx::clip_shapes`] and is referenced by index.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClipShapeKind {
    Rect,
    RoundedRect {
        radius: f32,
    },
    Circle,
    Capsule,
    Ellipse,
    /// Custom path; the actual `Vec<PathCommand>` is at
    /// `RenderCtx::clip_shapes[index]`.
    Path {
        index: u32,
    },
}

fn shape_kind(shape: &ClipShape, ctx: &mut RenderCtx) -> ClipShapeKind {
    match shape {
        ClipShape::Rect => ClipShapeKind::Rect,
        ClipShape::RoundedRect { radius } => ClipShapeKind::RoundedRect { radius: *radius },
        ClipShape::Circle => ClipShapeKind::Circle,
        ClipShape::Capsule => ClipShapeKind::Capsule,
        ClipShape::Ellipse => ClipShapeKind::Ellipse,
        ClipShape::Path(cmds) => {
            let idx = ctx.clip_shapes.len() as u32;
            ctx.clip_shapes.push(cmds.clone());
            ClipShapeKind::Path { index: idx }
        }
    }
}

/// External primitive form — the GPU-bridge consumes these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderPrimitive {
    Rect {
        rect: LayoutRect,
        color: [f32; 4],
    },
    Border {
        rect: LayoutRect,
        width: f32,
        color: [f32; 4],
    },
    Text {
        rect: LayoutRect,
        text: String,
        font_size: f32,
        color: [f32; 4],
    },
    Image {
        rect: LayoutRect,
        url: String,
    },
    PushClip {
        rect: LayoutRect,
    },
    PopClip,
    PushOpacity {
        alpha: f32,
    },
    PopOpacity,
    PushTransform {
        transform: [f32; 6],
    },
    PopTransform,
    /// GPU-side drop shadow. `inset=true` means the shadow paints inside
    /// the bounds (the receiving GPU bridge should clip first).
    DropShadow {
        rect: LayoutRect,
        color: [f32; 4],
        radius: f32,
        offset_x: f32,
        offset_y: f32,
        inset: bool,
    },
    /// GPU-side backdrop filter — captures what's behind, blurs, applies
    /// colour matrix, composites back. Bracket with `PopBackdropFilter`.
    PushBackdropFilter {
        rect: LayoutRect,
        blur_radius: f32,
        saturate: f32,
        brightness: f32,
        contrast: f32,
        grayscale: f32,
        hue_rotate: f32,
        invert: f32,
        sepia: f32,
    },
    PopBackdropFilter,
}

/// A single hit-test region recorded by gesture modifiers (`OnTap`,
/// `OnHover`, `OnLongPress`, `OnDrag`). The host's input pipeline walks the
/// list in reverse paint order to dispatch events.
#[derive(Clone)]
pub struct HitRegion {
    pub rect: LayoutRect,
    pub kind: HitKind,
}

/// Gesture flavour carried by a [`HitRegion`].
#[derive(Clone)]
pub enum HitKind {
    Tap(Rc<dyn Fn()>),
    Hover(Rc<dyn Fn(bool)>),
    LongPress {
        duration_ms: u32,
        action: Rc<dyn Fn()>,
    },
    Drag(Rc<dyn Fn(f32, f32)>),
}

impl std::fmt::Debug for HitRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.kind {
            HitKind::Tap(_) => "Tap",
            HitKind::Hover(_) => "Hover",
            HitKind::LongPress { .. } => "LongPress",
            HitKind::Drag(_) => "Drag",
        };
        f.debug_struct("HitRegion")
            .field("rect", &self.rect)
            .field("kind", &kind)
            .finish()
    }
}

/// Lifecycle-callback record produced by [`Modifier::OnAppear`] /
/// [`Modifier::OnDisappear`]. The host walks `appear_callbacks` after the
/// first paint and `disappear_callbacks` when the corresponding subtree is
/// torn down.
#[derive(Clone)]
pub struct LifecycleHook {
    pub action: Rc<dyn Fn()>,
}

impl std::fmt::Debug for LifecycleHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LifecycleHook(..)")
    }
}

/// Per-node annotation table — populated by accessibility / metadata
/// modifiers so the embedder can build the a11y tree, tooltip list, tab
/// order, etc. without re-walking the view tree.
#[derive(Debug, Clone, Default)]
pub struct Annotation {
    pub rect: LayoutRect,
    pub label: Option<String>,
    pub role: Option<AccessibilityRole>,
    pub hint: Option<String>,
    pub value: Option<String>,
    pub hidden: Option<bool>,
    pub tooltip: Option<String>,
    pub tab_index: Option<i32>,
    pub allows_hit_testing: Option<bool>,
    pub tag: Option<String>,
    pub z_index: Option<f32>,
    pub animation: Option<AnimationSpec>,
    pub transition: Option<TransitionKind>,
    pub hidden_visually: Option<bool>,
}

/// Immediate-mode rendering context that widget tree's `render` impls push to.
#[derive(Debug, Clone, Default)]
pub struct RenderCtx {
    pub commands: Vec<DrawCommand>,
    /// Vertical scroll offset applied uniformly when converting to primitives.
    pub scroll_offset_y: f32,
    /// Hit-test regions accumulated by gesture modifiers during the render
    /// walk. The host consumes this list to dispatch input events.
    pub hit_regions: Vec<HitRegion>,
    /// `on_appear` lifecycle hooks recorded during the render walk.
    pub appear_callbacks: Vec<LifecycleHook>,
    /// `on_disappear` lifecycle hooks recorded during the render walk.
    pub disappear_callbacks: Vec<LifecycleHook>,
    /// Per-node annotations recorded by accessibility / metadata modifiers.
    pub annotations: Vec<Annotation>,
    /// Sidetable of custom clip-path command lists referenced by
    /// [`ClipShapeKind::Path`].
    pub clip_shapes: Vec<Vec<PathCommand>>,
}

impl RenderCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fill_rect(&mut self, rect: &LayoutRect, color: Color) {
        if color.is_transparent() {
            return;
        }
        self.commands
            .push(DrawCommand::FillRect { rect: *rect, color });
    }

    pub fn stroke_rect(&mut self, rect: &LayoutRect, width: f32, color: Color) {
        if color.is_transparent() || width <= 0.0 {
            return;
        }
        self.commands.push(DrawCommand::StrokeRect {
            rect: *rect,
            width,
            color,
        });
    }

    pub fn fill_text(&mut self, rect: &LayoutRect, text: &str, font_size: f32, color: Color) {
        if text.is_empty() || color.is_transparent() {
            return;
        }
        self.commands.push(DrawCommand::Text {
            rect: *rect,
            text: text.to_string(),
            font_size,
            color,
        });
    }

    pub fn draw_image(&mut self, rect: &LayoutRect, url: &str) {
        self.commands.push(DrawCommand::Image {
            rect: *rect,
            url: url.to_string(),
        });
    }

    pub fn push_clip(&mut self, rect: &LayoutRect) {
        self.commands.push(DrawCommand::PushClip { rect: *rect });
    }

    pub fn pop_clip(&mut self) {
        self.commands.push(DrawCommand::PopClip);
    }

    /// Lower internal commands to externally-visible [`RenderPrimitive`]s.
    pub fn to_primitives(&self) -> Vec<RenderPrimitive> {
        self.commands
            .iter()
            .map(|cmd| match cmd {
                DrawCommand::FillRect { rect, color } => RenderPrimitive::Rect {
                    rect: *rect,
                    color: color.to_array(),
                },
                DrawCommand::StrokeRect { rect, width, color } => RenderPrimitive::Border {
                    rect: *rect,
                    width: *width,
                    color: color.to_array(),
                },
                DrawCommand::Text {
                    rect,
                    text,
                    font_size,
                    color,
                } => RenderPrimitive::Text {
                    rect: *rect,
                    text: text.clone(),
                    font_size: *font_size,
                    color: color.to_array(),
                },
                DrawCommand::Image { rect, url } => RenderPrimitive::Image {
                    rect: *rect,
                    url: url.clone(),
                },
                DrawCommand::PushClip { rect } => RenderPrimitive::PushClip { rect: *rect },
                DrawCommand::PopClip => RenderPrimitive::PopClip,
                // Shape clips currently lower to a rect clip — the bounding
                // box. Real path clipping is a Phase 4+ concern.
                DrawCommand::PushClipShape { rect, .. } => {
                    RenderPrimitive::PushClip { rect: *rect }
                }
                DrawCommand::PopClipShape => RenderPrimitive::PopClip,
                DrawCommand::PushOpacity { alpha } => {
                    RenderPrimitive::PushOpacity { alpha: *alpha }
                }
                DrawCommand::PopOpacity => RenderPrimitive::PopOpacity,
                DrawCommand::PushTransform { transform } => RenderPrimitive::PushTransform {
                    transform: transform.0,
                },
                DrawCommand::PopTransform => RenderPrimitive::PopTransform,
                DrawCommand::DropShadow {
                    rect,
                    color,
                    radius,
                    offset_x,
                    offset_y,
                    inset,
                } => RenderPrimitive::DropShadow {
                    rect: *rect,
                    color: color.to_array(),
                    radius: *radius,
                    offset_x: *offset_x,
                    offset_y: *offset_y,
                    inset: *inset,
                },
                DrawCommand::PushBackdropFilter { rect, filter } => {
                    RenderPrimitive::PushBackdropFilter {
                        rect: *rect,
                        blur_radius: filter.blur_radius,
                        saturate: filter.saturate,
                        brightness: filter.brightness,
                        contrast: filter.contrast,
                        grayscale: filter.grayscale,
                        hue_rotate: filter.hue_rotate,
                        invert: filter.invert,
                        sepia: filter.sepia,
                    }
                }
                DrawCommand::PopBackdropFilter => RenderPrimitive::PopBackdropFilter,
            })
            .collect()
    }
}

// =============================================================================
// Declarative render walker
// =============================================================================

/// Approximate intrinsic width of a text glyph run at `font_size`. Mirrors
/// the heuristic used by the legacy widget builder in [`crate::widgets`].
fn text_intrinsic(t: &TextLeaf) -> Size {
    let w = t.content.chars().count() as f32 * t.font_size * 0.55;
    let h = t.font_size * 1.4;
    Size::new(w, h)
}

/// Compute the size a [`ViewNode`] would like to occupy under the given
/// proposal. Walks the tree without producing draw commands.
///
/// `foreground` carries the inherited tint colour from `Modifier::Foreground`
/// so leaves can pick it up; the public [`render_node`] entry point seeds
/// this with `None`.
fn measure_node(node: &ViewNode, proposal: ProposedSize) -> Measured {
    match node {
        ViewNode::Leaf(kind) => measure_leaf(kind, proposal),
        ViewNode::Container { layout, children } => {
            let child_measured: Vec<Measured> =
                children.iter().map(|c| measure_node(c, proposal)).collect();
            let s = layout.size_that_fits(proposal, &child_measured);
            Measured {
                min: s,
                max: s,
                ideal: s,
                priority: 0.0,
            }
        }
        ViewNode::Modified { modifier, child } => measure_modified(modifier, child, proposal),
        ViewNode::Stateful(f) => measure_node(&f(), proposal),
    }
}

/// Measurement for a single leaf.
fn measure_leaf(kind: &LeafKind, proposal: ProposedSize) -> Measured {
    let ideal = match kind {
        LeafKind::Text(t) => text_intrinsic(t),
        LeafKind::Image(_) => Size::new(proposal.width_or(64.0), proposal.height_or(64.0)),
        LeafKind::Spacer { min_length } => Size::new(
            proposal.width_or(*min_length),
            proposal.height_or(*min_length),
        ),
        LeafKind::Divider { thickness, .. } => Size::new(proposal.width_or(0.0), *thickness),
        LeafKind::Canvas(_) => Size::new(proposal.width_or(0.0), proposal.height_or(0.0)),
    };
    Measured {
        min: ideal,
        max: ideal,
        ideal,
        priority: 0.0,
    }
}

/// Measurement for a `Modified` subtree.
fn measure_modified(modifier: &Modifier, child: &ViewNode, proposal: ProposedSize) -> Measured {
    match modifier {
        Modifier::Padding(e) => {
            let inner_proposal = ProposedSize {
                width: proposal.width.map(|w| (w - e.horizontal()).max(0.0)),
                height: proposal.height.map(|h| (h - e.vertical()).max(0.0)),
            };
            let inner = measure_node(child, inner_proposal);
            let s = Size::new(
                inner.ideal.width + e.horizontal(),
                inner.ideal.height + e.vertical(),
            );
            Measured {
                min: s,
                max: s,
                ideal: s,
                priority: inner.priority,
            }
        }
        Modifier::Frame { width, height } => {
            let child_m = measure_node(
                child,
                ProposedSize {
                    width: width.or(proposal.width),
                    height: height.or(proposal.height),
                },
            );
            let s = Size::new(
                width.unwrap_or(child_m.ideal.width),
                height.unwrap_or(child_m.ideal.height),
            );
            Measured {
                min: s,
                max: s,
                ideal: s,
                priority: child_m.priority,
            }
        }
        Modifier::FrameAligned { width, height, .. } => {
            let child_m = measure_node(
                child,
                ProposedSize {
                    width: width.or(proposal.width),
                    height: height.or(proposal.height),
                },
            );
            let s = Size::new(
                width.unwrap_or(child_m.ideal.width),
                height.unwrap_or(child_m.ideal.height),
            );
            Measured {
                min: s,
                max: s,
                ideal: s,
                priority: child_m.priority,
            }
        }
        Modifier::FrameRange {
            min_width,
            ideal_width,
            max_width,
            min_height,
            ideal_height,
            max_height,
            ..
        } => {
            let child_m = measure_node(child, proposal);
            let chosen_w = ideal_width
                .or(proposal.width)
                .unwrap_or(child_m.ideal.width);
            let chosen_h = ideal_height
                .or(proposal.height)
                .unwrap_or(child_m.ideal.height);
            let w = chosen_w
                .max(min_width.unwrap_or(0.0))
                .min(max_width.unwrap_or(f32::INFINITY));
            let h = chosen_h
                .max(min_height.unwrap_or(0.0))
                .min(max_height.unwrap_or(f32::INFINITY));
            let s = Size::new(w, h);
            Measured {
                min: s,
                max: s,
                ideal: s,
                priority: child_m.priority,
            }
        }
        Modifier::Position { .. } => {
            // Position absolutely — child still claims its own ideal size.
            measure_node(child, proposal)
        }
        Modifier::Hidden(true) => {
            // Hidden views still occupy layout space — measure as normal.
            measure_node(child, proposal)
        }
        // Every remaining modifier is decorative or input-only; pass measurement through.
        _ => measure_node(child, proposal),
    }
}

/// Render a [`ViewNode`] tree into the supplied bounds, emitting commands
/// into `ctx`. Walks the tree depth-first, using the [`Layout`](crate::layout::Layout)
/// trait's two-pass protocol for `Container` nodes and applying decorators
/// for `Modified` nodes.
pub fn render_node(node: &ViewNode, bounds: Rect, ctx: &mut RenderCtx) {
    render_node_inner(node, bounds, ctx, &Inherited::default());
}

/// Render a [`ViewNode`] tree, consulting `driver` for the
/// presence/progress of `node_id`. Used by the host runtime to drive
/// [`Modifier::Transition`] envelopes for a node that has just entered or
/// is about to leave the tree.
///
/// Each [`Modifier::Transition`] modifier consults the driver via the
/// supplied `node_id` to compute the current envelope factor (alpha for
/// `Opacity`, offset for `Slide`, scale for `Scale`). If `driver` reports
/// [`crate::animation::Presence::Stable`], the envelope is the identity
/// and the child renders unchanged.
pub fn render_node_with_driver(
    node: &ViewNode,
    bounds: Rect,
    ctx: &mut RenderCtx,
    node_id: crate::signal::NodeId,
    driver: &crate::animation::AnimationDriver,
) {
    let inh = Inherited {
        foreground: None,
        opacity_mul: None,
        animation_node: Some((node_id, driver)),
    };
    render_node_inner(node, bounds, ctx, &inh);
}

/// Properties inherited down the render walk — foreground tint, opacity
/// multiplier, and (Phase 4) the animation driver paired with the
/// surrounding node id.
#[derive(Copy, Clone, Default)]
struct Inherited<'a> {
    foreground: Option<Color>,
    opacity_mul: Option<f32>,
    animation_node: Option<(crate::signal::NodeId, &'a crate::animation::AnimationDriver)>,
}

fn apply_alpha(c: Color, alpha: Option<f32>) -> Color {
    match alpha {
        Some(a) => Color::rgba(c.r, c.g, c.b, (c.a * a).clamp(0.0, 1.0)),
        None => c,
    }
}

fn render_node_inner(node: &ViewNode, bounds: Rect, ctx: &mut RenderCtx, inh: &Inherited<'_>) {
    match node {
        ViewNode::Leaf(kind) => render_leaf(kind, bounds, ctx, inh),
        ViewNode::Container { layout, children } => {
            // Two-pass: measure each child, then place them.
            let proposal = ProposedSize::fixed(bounds.size.width, bounds.size.height);
            let measured: Vec<Measured> =
                children.iter().map(|c| measure_node(c, proposal)).collect();
            // Place subviews — seed each placement with the child's ideal size
            // since several layout impls read it from `Placement::size`.
            let mut placements: Vec<Placement> = measured
                .iter()
                .map(|m| Placement::new(Point::default(), m.ideal))
                .collect();
            layout.place_subviews(bounds, proposal, &mut placements);
            for (child, p) in children.iter().zip(placements.iter()) {
                let child_bounds = Rect {
                    origin: p.origin,
                    size: p.size,
                };
                render_node_inner(child, child_bounds, ctx, inh);
            }
        }
        ViewNode::Modified { modifier, child } => {
            render_modified(modifier, child, bounds, ctx, inh)
        }
        ViewNode::Stateful(f) => {
            let resolved = f();
            render_node_inner(&resolved, bounds, ctx, inh);
        }
    }
}

fn render_leaf(kind: &LeafKind, bounds: Rect, ctx: &mut RenderCtx, inh: &Inherited<'_>) {
    let lr = bounds.to_layout_rect();
    match kind {
        LeafKind::Text(t) => {
            let color = inh.foreground.unwrap_or(t.color);
            let color = apply_alpha(color, inh.opacity_mul);
            ctx.fill_text(&lr, &t.content, t.font_size, color);
        }
        LeafKind::Image(img) => {
            ctx.draw_image(&lr, &img.url);
        }
        LeafKind::Spacer { .. } => {
            // Spacers are invisible — they only consume layout space.
        }
        LeafKind::Divider {
            thickness: _,
            color,
        } => {
            let color = apply_alpha(*color, inh.opacity_mul);
            ctx.fill_rect(&lr, color);
        }
        LeafKind::Canvas(draw) => {
            // Canvas leaves draw imperatively. Translate the context's
            // origin by pushing/popping a clip rect to bound the area.
            ctx.push_clip(&lr);
            draw(ctx, bounds.size);
            ctx.pop_clip();
        }
    }
}

/// Place a child of size `(w, h)` inside `bounds` according to `alignment`.
fn aligned_origin(bounds: Rect, w: f32, h: f32, alignment: Alignment2D) -> (f32, f32) {
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

/// Lower a [`BackgroundFill`] into one or more `FillRect` commands. Solid
/// colours are a single rect; gradients become a vertical (or radial) strip
/// of colour stops. The renderer is intentionally simple — Phase 4 will
/// switch to a real gradient shader.
fn paint_background(fill: &BackgroundFill, bounds: Rect, ctx: &mut RenderCtx, alpha: Option<f32>) {
    let lr = bounds.to_layout_rect();
    match fill {
        BackgroundFill::Color(c) => {
            ctx.fill_rect(&lr, apply_alpha(*c, alpha));
        }
        BackgroundFill::LinearGradient { stops, .. } => {
            // Approximate as `stops.len()` horizontal slices spanning `bounds`.
            // Angle is currently advisory.
            if stops.is_empty() {
                return;
            }
            let n = stops.len();
            let h = bounds.size.height / n as f32;
            for (i, (_pos, c)) in stops.iter().enumerate() {
                let strip = LayoutRect::new(
                    bounds.origin.x,
                    bounds.origin.y + h * i as f32,
                    bounds.size.width,
                    h,
                );
                ctx.fill_rect(&strip, apply_alpha(*c, alpha));
            }
        }
        BackgroundFill::RadialGradient { stops, .. } => {
            // Approximate as concentric rings drawn from the outside in.
            if stops.is_empty() {
                return;
            }
            let n = stops.len();
            let cx = bounds.origin.x + bounds.size.width * 0.5;
            let cy = bounds.origin.y + bounds.size.height * 0.5;
            for (i, (_pos, c)) in stops.iter().enumerate() {
                let frac = (n - i) as f32 / n as f32;
                let rw = bounds.size.width * frac;
                let rh = bounds.size.height * frac;
                let ring = LayoutRect::new(cx - rw * 0.5, cy - rh * 0.5, rw, rh);
                ctx.fill_rect(&ring, apply_alpha(*c, alpha));
            }
        }
    }
}

fn render_modified(
    modifier: &Modifier,
    child: &ViewNode,
    bounds: Rect,
    ctx: &mut RenderCtx,
    inh: &Inherited<'_>,
) {
    match modifier {
        Modifier::Padding(e) => {
            let inner = inset_rect(bounds, *e);
            render_node_inner(child, inner, ctx, inh);
        }
        Modifier::Background(fill) => {
            paint_background(fill, bounds, ctx, inh.opacity_mul);
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Foreground(c) => {
            let mut next = *inh;
            next.foreground = Some(*c);
            render_node_inner(child, bounds, ctx, &next);
        }
        Modifier::Opacity(a) => {
            let alpha = a.clamp(0.0, 1.0);
            let mut next = *inh;
            let combined = inh.opacity_mul.unwrap_or(1.0) * alpha;
            next.opacity_mul = Some(combined);
            ctx.commands.push(DrawCommand::PushOpacity { alpha });
            render_node_inner(child, bounds, ctx, &next);
            ctx.commands.push(DrawCommand::PopOpacity);
        }
        Modifier::CornerRadius(_) => {
            // Renderer doesn't yet support rounded corners — pass through.
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Frame { width, height } => {
            // Reshape bounds to the requested dims, centred within the
            // parent's offered rect.
            let w = width.unwrap_or(bounds.size.width);
            let h = height.unwrap_or(bounds.size.height);
            let x = bounds.origin.x + (bounds.size.width - w) * 0.5;
            let y = bounds.origin.y + (bounds.size.height - h) * 0.5;
            let inner = Rect::new(x, y, w, h);
            render_node_inner(child, inner, ctx, inh);
        }
        Modifier::FrameAligned {
            width,
            height,
            alignment,
        } => {
            let w = width.unwrap_or(bounds.size.width);
            let h = height.unwrap_or(bounds.size.height);
            let (x, y) = aligned_origin(bounds, w, h, *alignment);
            render_node_inner(child, Rect::new(x, y, w, h), ctx, inh);
        }
        Modifier::FrameRange {
            min_width,
            ideal_width,
            max_width,
            min_height,
            ideal_height,
            max_height,
            alignment,
        } => {
            let proposed_w = ideal_width.unwrap_or(bounds.size.width);
            let proposed_h = ideal_height.unwrap_or(bounds.size.height);
            let w = proposed_w
                .max(min_width.unwrap_or(0.0))
                .min(max_width.unwrap_or(f32::INFINITY));
            let h = proposed_h
                .max(min_height.unwrap_or(0.0))
                .min(max_height.unwrap_or(f32::INFINITY));
            let (x, y) = aligned_origin(bounds, w, h, *alignment);
            render_node_inner(child, Rect::new(x, y, w, h), ctx, inh);
        }
        Modifier::Border { width, color } => {
            // Draw the border outermost, after the child paints, so it sits
            // on top of any background fill the child may have emitted.
            render_node_inner(child, bounds, ctx, inh);
            let c = apply_alpha(*color, inh.opacity_mul);
            ctx.stroke_rect(&bounds.to_layout_rect(), *width, c);
        }
        Modifier::Shadow {
            color,
            radius,
            offset_x,
            offset_y,
        } => {
            // Approximate: paint a single offset rect of `color` behind the
            // child. `radius` is currently advisory; Phase 4 will replace
            // this with a real Gaussian blur shader.
            let _ = radius;
            let shadow_rect = LayoutRect::new(
                bounds.origin.x + offset_x,
                bounds.origin.y + offset_y,
                bounds.size.width,
                bounds.size.height,
            );
            ctx.fill_rect(&shadow_rect, apply_alpha(*color, inh.opacity_mul));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::InnerShadow {
            color,
            radius,
            offset_x,
            offset_y,
        } => {
            let _ = radius;
            render_node_inner(child, bounds, ctx, inh);
            let inner_rect = LayoutRect::new(
                bounds.origin.x + offset_x,
                bounds.origin.y + offset_y,
                bounds.size.width,
                bounds.size.height,
            );
            ctx.fill_rect(&inner_rect, apply_alpha(*color, inh.opacity_mul));
        }
        Modifier::Rotation(degrees) => {
            let cx = bounds.origin.x + bounds.size.width * 0.5;
            let cy = bounds.origin.y + bounds.size.height * 0.5;
            ctx.commands.push(DrawCommand::PushTransform {
                transform: Transform2D::rotate_around(*degrees, cx, cy),
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopTransform);
        }
        Modifier::Scale { x, y, anchor } => {
            let (ax, ay) = aligned_origin(bounds, 0.0, 0.0, *anchor);
            // For alignment, treat the anchor point (0-sized child) as the
            // fixed point — that maps Leading/Top → origin, Center → middle,
            // Trailing/Bottom → far edge, which is what SwiftUI does.
            ctx.commands.push(DrawCommand::PushTransform {
                transform: Transform2D::scale_around(*x, *y, ax, ay),
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopTransform);
        }
        Modifier::Offset { x, y } => {
            ctx.commands.push(DrawCommand::PushTransform {
                transform: Transform2D::translate(*x, *y),
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopTransform);
        }
        Modifier::Position { x, y } => {
            // Override placement entirely — paint at (x, y) keeping the
            // child's expected size.
            let inner = Rect::new(*x, *y, bounds.size.width, bounds.size.height);
            render_node_inner(child, inner, ctx, inh);
        }
        Modifier::Clip(shape) => {
            let kind = shape_kind(shape, ctx);
            ctx.commands.push(DrawCommand::PushClipShape {
                rect: bounds.to_layout_rect(),
                shape: kind,
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopClipShape);
        }
        Modifier::Mask(shape) => {
            // In immediate mode a mask is identical to a clip — alpha-mask
            // support lands later.
            let kind = shape_kind(shape, ctx);
            ctx.commands.push(DrawCommand::PushClipShape {
                rect: bounds.to_layout_rect(),
                shape: kind,
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopClipShape);
        }
        Modifier::OnTap(f) => {
            ctx.hit_regions.push(HitRegion {
                rect: bounds.to_layout_rect(),
                kind: HitKind::Tap(Rc::clone(f)),
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::OnHover(f) => {
            ctx.hit_regions.push(HitRegion {
                rect: bounds.to_layout_rect(),
                kind: HitKind::Hover(Rc::clone(f)),
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::OnLongPress {
            duration_ms,
            action,
        } => {
            ctx.hit_regions.push(HitRegion {
                rect: bounds.to_layout_rect(),
                kind: HitKind::LongPress {
                    duration_ms: *duration_ms,
                    action: Rc::clone(action),
                },
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::OnDrag(f) => {
            ctx.hit_regions.push(HitRegion {
                rect: bounds.to_layout_rect(),
                kind: HitKind::Drag(Rc::clone(f)),
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::OnAppear(f) => {
            ctx.appear_callbacks.push(LifecycleHook {
                action: Rc::clone(f),
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::OnDisappear(f) => {
            ctx.disappear_callbacks.push(LifecycleHook {
                action: Rc::clone(f),
            });
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AccessibilityLabel(s) => {
            push_annotation(ctx, bounds, |a| a.label = Some(s.clone()));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AccessibilityRole(r) => {
            push_annotation(ctx, bounds, |a| a.role = Some(*r));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AccessibilityHint(s) => {
            push_annotation(ctx, bounds, |a| a.hint = Some(s.clone()));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AccessibilityValue(s) => {
            push_annotation(ctx, bounds, |a| a.value = Some(s.clone()));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AccessibilityHidden(b) => {
            push_annotation(ctx, bounds, |a| a.hidden = Some(*b));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::TabIndex(i) => {
            push_annotation(ctx, bounds, |a| a.tab_index = Some(*i));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::AllowsHitTesting(b) => {
            push_annotation(ctx, bounds, |a| a.allows_hit_testing = Some(*b));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Tooltip(s) => {
            push_annotation(ctx, bounds, |a| a.tooltip = Some(s.clone()));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::ZIndex(z) => {
            push_annotation(ctx, bounds, |a| a.z_index = Some(*z));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Animation(spec) => {
            push_annotation(ctx, bounds, |a| a.animation = Some(*spec));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Transition(t) => {
            push_annotation(ctx, bounds, |a| a.transition = Some(t.clone()));
            render_transition(t, child, bounds, ctx, inh);
        }
        Modifier::Tag(s) => {
            push_annotation(ctx, bounds, |a| a.tag = Some(s.clone()));
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::Hidden(b) => {
            push_annotation(ctx, bounds, |a| a.hidden_visually = Some(*b));
            if !*b {
                render_node_inner(child, bounds, ctx, inh);
            }
        }
        Modifier::Shadows(stack) => {
            // Paint each shadow before the child, back-to-front.
            // Multi-shadow stacks (Linear/Vercel/Notion style) layer
            // several shadows with different blur radii and offsets so the
            // result reads as a real elevation, not a flat offset.
            let alpha = inh.opacity_mul.unwrap_or(1.0);
            for shadow in stack {
                let shadow_rect = LayoutRect::new(
                    bounds.origin.x + shadow.dx,
                    bounds.origin.y + shadow.dy,
                    bounds.size.width,
                    bounds.size.height,
                );
                let tinted = apply_alpha(shadow.color, Some(alpha));
                ctx.commands.push(DrawCommand::DropShadow {
                    rect: shadow_rect,
                    color: tinted,
                    radius: shadow.radius,
                    offset_x: shadow.dx,
                    offset_y: shadow.dy,
                    inset: false,
                });
            }
            render_node_inner(child, bounds, ctx, inh);
        }
        Modifier::BackdropFilter(filter) => {
            // Capture-blur-recomposite happens at the GPU bridge layer; here
            // we just bracket the child render with Push/PopBackdropFilter.
            let layout_bounds = LayoutRect::new(
                bounds.origin.x,
                bounds.origin.y,
                bounds.size.width,
                bounds.size.height,
            );
            ctx.commands.push(DrawCommand::PushBackdropFilter {
                rect: layout_bounds,
                filter: *filter,
            });
            render_node_inner(child, bounds, ctx, inh);
            ctx.commands.push(DrawCommand::PopBackdropFilter);
        }
    }
}

/// Emit a fresh [`Annotation`] tagged with `bounds`, letting the caller fill
/// in the relevant field.
fn push_annotation(ctx: &mut RenderCtx, bounds: Rect, fill: impl FnOnce(&mut Annotation)) {
    let mut a = Annotation {
        rect: bounds.to_layout_rect(),
        ..Annotation::default()
    };
    fill(&mut a);
    ctx.annotations.push(a);
}

fn inset_rect(r: Rect, e: EdgeInsets) -> Rect {
    Rect::new(
        r.origin.x + e.left,
        r.origin.y + e.top,
        (r.size.width - e.horizontal()).max(0.0),
        (r.size.height - e.vertical()).max(0.0),
    )
}

/// Apply a [`TransitionKind`] envelope while rendering `child`. Reads the
/// presence/progress of the surrounding `node_id` from the driver carried
/// in [`Inherited::animation_node`] and translates / scales / fades the
/// child accordingly.
///
/// When no driver is in scope (legacy [`render_node`] entry point) or the
/// transition is `None`, this is identical to a passthrough render of
/// `child`.
fn render_transition(
    kind: &TransitionKind,
    child: &ViewNode,
    bounds: Rect,
    ctx: &mut RenderCtx,
    inh: &Inherited<'_>,
) {
    use crate::animation::{
        Presence, opacity_for_transition, scale_for_transition, slide_for_transition,
    };

    let Some((node_id, driver)) = inh.animation_node else {
        // No driver context — paint the child unchanged.
        render_node_inner(child, bounds, ctx, inh);
        return;
    };

    let presence = driver.presence(node_id);
    if matches!(presence, Presence::Stable) || matches!(kind, TransitionKind::None) {
        render_node_inner(child, bounds, ctx, inh);
        return;
    }

    // Pick the eased progress for the surrounding (node_id, key=0) animation
    // — Phase 4 hard-codes the transition's animation key to 0; richer
    // multi-key driving is a Phase 5 concern.
    let t = driver.eased_progress(node_id, 0).unwrap_or(match presence {
        Presence::Entering => 1.0,
        Presence::Leaving => 0.0,
        Presence::Stable => 1.0,
    });

    // Slide → push translate transform.
    if let Some((dx, dy)) = slide_for_transition(kind, presence, t, bounds.size) {
        ctx.commands.push(DrawCommand::PushTransform {
            transform: Transform2D::translate(dx, dy),
        });
        render_node_inner(child, bounds, ctx, inh);
        ctx.commands.push(DrawCommand::PopTransform);
        return;
    }

    // Scale → push scale-around-centre transform.
    if let Some(factor) = scale_for_transition(kind, presence, t) {
        let cx = bounds.origin.x + bounds.size.width * 0.5;
        let cy = bounds.origin.y + bounds.size.height * 0.5;
        ctx.commands.push(DrawCommand::PushTransform {
            transform: Transform2D::scale_around(factor, factor, cx, cy),
        });
        render_node_inner(child, bounds, ctx, inh);
        ctx.commands.push(DrawCommand::PopTransform);
        return;
    }

    // Opacity → push alpha through the inherited multiplier and emit
    // Push/PopOpacity brackets so the GPU bridge can compose the layer.
    if let Some(alpha) = opacity_for_transition(kind, presence, t) {
        let mut next = *inh;
        let combined = inh.opacity_mul.unwrap_or(1.0) * alpha;
        next.opacity_mul = Some(combined);
        ctx.commands.push(DrawCommand::PushOpacity { alpha });
        render_node_inner(child, bounds, ctx, &next);
        ctx.commands.push(DrawCommand::PopOpacity);
        return;
    }

    // Asymmetric resolved to a kind we don't recognise — passthrough.
    render_node_inner(child, bounds, ctx, inh);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{HStack, VStack};
    use crate::view::{ImageLeaf, TextLeaf};
    use std::cell::RefCell;
    use std::rc::Rc;

    // -------------------------------------------------------------------------
    // Legacy ctx tests
    // -------------------------------------------------------------------------

    #[test]
    fn ctx_starts_empty() {
        let ctx = RenderCtx::new();
        assert!(ctx.commands.is_empty());
    }

    #[test]
    fn fill_rect_pushes_one_command() {
        let mut ctx = RenderCtx::new();
        ctx.fill_rect(&LayoutRect::new(0.0, 0.0, 10.0, 10.0), Color::WHITE);
        assert_eq!(ctx.commands.len(), 1);
    }

    #[test]
    fn fill_rect_skips_transparent() {
        let mut ctx = RenderCtx::new();
        ctx.fill_rect(&LayoutRect::new(0.0, 0.0, 10.0, 10.0), Color::TRANSPARENT);
        assert!(ctx.commands.is_empty());
    }

    #[test]
    fn fill_text_skips_empty_string() {
        let mut ctx = RenderCtx::new();
        ctx.fill_text(&LayoutRect::default(), "", 14.0, Color::BLACK);
        assert!(ctx.commands.is_empty());
    }

    #[test]
    fn to_primitives_translates_text() {
        let mut ctx = RenderCtx::new();
        ctx.fill_text(
            &LayoutRect::new(0.0, 0.0, 100.0, 20.0),
            "hi",
            14.0,
            Color::WHITE,
        );
        let prims = ctx.to_primitives();
        assert_eq!(prims.len(), 1);
        assert!(matches!(prims[0], RenderPrimitive::Text { .. }));
    }

    #[test]
    fn push_pop_clip_round_trip() {
        let mut ctx = RenderCtx::new();
        ctx.push_clip(&LayoutRect::default());
        ctx.pop_clip();
        let prims = ctx.to_primitives();
        assert_eq!(prims.len(), 2);
    }

    // -------------------------------------------------------------------------
    // render_node tests
    // -------------------------------------------------------------------------

    #[test]
    fn render_node_text_leaf_emits_text() {
        let node = ViewNode::Leaf(LeafKind::Text(TextLeaf::new("hi")));
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 20.0), &mut ctx);
        assert_eq!(ctx.commands.len(), 1);
        assert!(matches!(ctx.commands[0], DrawCommand::Text { .. }));
    }

    #[test]
    fn render_node_image_leaf_emits_image() {
        let node = ViewNode::Leaf(LeafKind::Image(ImageLeaf::new("foo.png")));
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 64.0, 64.0), &mut ctx);
        assert!(matches!(ctx.commands[0], DrawCommand::Image { .. }));
    }

    #[test]
    fn render_node_spacer_emits_no_commands() {
        let node = ViewNode::Leaf(LeafKind::Spacer { min_length: 10.0 });
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 10.0), &mut ctx);
        assert!(ctx.commands.is_empty());
    }

    #[test]
    fn render_node_divider_emits_fill() {
        let node = ViewNode::Leaf(LeafKind::Divider {
            thickness: 1.0,
            color: Color::WHITE,
        });
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 1.0), &mut ctx);
        assert!(matches!(ctx.commands[0], DrawCommand::FillRect { .. }));
    }

    #[test]
    fn render_node_container_renders_children() {
        let node = ViewNode::Container {
            layout: Box::new(HStack::new().spacing(0.0)),
            children: vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))),
            ],
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 200.0, 20.0), &mut ctx);
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(texts, 2);
    }

    #[test]
    fn render_node_padding_modifier_insets_child() {
        let node = ViewNode::Modified {
            modifier: Modifier::Padding(EdgeInsets::all(10.0)),
            child: Box::new(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x")))),
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 100.0), &mut ctx);
        let DrawCommand::Text { rect, .. } = &ctx.commands[0] else {
            panic!("expected text command");
        };
        // Inner rect should start at (10, 10) with reduced dims.
        assert_eq!(rect.x, 10.0);
        assert_eq!(rect.y, 10.0);
        assert_eq!(rect.width, 80.0);
        assert_eq!(rect.height, 80.0);
    }

    #[test]
    fn render_node_background_emits_fill_before_child() {
        let node = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
            child: Box::new(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x")))),
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        assert!(matches!(ctx.commands[0], DrawCommand::FillRect { .. }));
        assert!(matches!(ctx.commands[1], DrawCommand::Text { .. }));
    }

    #[test]
    fn render_node_foreground_overrides_text_color() {
        let mut leaf = TextLeaf::new("x");
        leaf.color = Color::WHITE;
        let red = Color::rgb(1.0, 0.0, 0.0);
        let node = ViewNode::Modified {
            modifier: Modifier::Foreground(red),
            child: Box::new(ViewNode::Leaf(LeafKind::Text(leaf))),
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 20.0), &mut ctx);
        let DrawCommand::Text { color, .. } = &ctx.commands[0] else {
            panic!("expected text command");
        };
        assert_eq!(*color, red);
    }

    #[test]
    fn render_node_opacity_multiplies_alpha() {
        let node = ViewNode::Modified {
            modifier: Modifier::Opacity(0.5),
            child: Box::new(ViewNode::Leaf(LeafKind::Divider {
                thickness: 1.0,
                color: Color::WHITE,
            })),
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 1.0), &mut ctx);
        // `Opacity` now wraps the child with Push/PopOpacity commands plus
        // tints the leaf via the inherited alpha multiplier.
        assert!(matches!(ctx.commands[0], DrawCommand::PushOpacity { .. }));
        let fill = ctx
            .commands
            .iter()
            .find(|c| matches!(c, DrawCommand::FillRect { .. }))
            .expect("expected fill rect");
        let DrawCommand::FillRect { color, .. } = fill else {
            unreachable!()
        };
        assert!((color.a - 0.5).abs() < 1e-6);
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::PopOpacity
        ));
    }

    #[test]
    fn render_node_frame_modifier_centers_child() {
        let node = ViewNode::Modified {
            modifier: Modifier::Frame {
                width: Some(50.0),
                height: Some(20.0),
            },
            child: Box::new(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x")))),
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 100.0), &mut ctx);
        let DrawCommand::Text { rect, .. } = &ctx.commands[0] else {
            panic!("expected text");
        };
        // Frame is 50x20 centered in 100x100 → origin (25, 40)
        assert_eq!(rect.x, 25.0);
        assert_eq!(rect.y, 40.0);
        assert_eq!(rect.width, 50.0);
        assert_eq!(rect.height, 20.0);
    }

    #[test]
    fn render_node_stateful_invokes_closure() {
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let node = ViewNode::Stateful(Rc::new(move || {
            *c.borrow_mut() += 1;
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("ping")))
        }));
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 20.0), &mut ctx);
        assert_eq!(*counter.borrow(), 1);
        assert!(matches!(ctx.commands[0], DrawCommand::Text { .. }));
    }

    #[test]
    fn render_node_canvas_pushes_and_pops_clip() {
        let node = ViewNode::Leaf(LeafKind::Canvas(Rc::new(|ctx, _size| {
            ctx.fill_rect(&LayoutRect::new(0.0, 0.0, 1.0, 1.0), Color::WHITE);
        })));
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 10.0, 10.0), &mut ctx);
        assert!(matches!(ctx.commands[0], DrawCommand::PushClip { .. }));
        assert!(matches!(ctx.commands.last().unwrap(), DrawCommand::PopClip));
    }

    #[test]
    fn render_node_vstack_distributes_children() {
        let node = ViewNode::Container {
            layout: Box::new(VStack::new().spacing(0.0)),
            children: vec![
                ViewNode::Leaf(LeafKind::Divider {
                    thickness: 4.0,
                    color: Color::WHITE,
                }),
                ViewNode::Leaf(LeafKind::Divider {
                    thickness: 4.0,
                    color: Color::WHITE,
                }),
            ],
        };
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 8.0), &mut ctx);
        let fills: Vec<&DrawCommand> = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::FillRect { .. }))
            .collect();
        assert_eq!(fills.len(), 2);
        if let (DrawCommand::FillRect { rect: r1, .. }, DrawCommand::FillRect { rect: r2, .. }) =
            (fills[0], fills[1])
        {
            assert!(r2.y > r1.y);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3 — modifier behaviour (one test per Modifier variant minimum).
    // -------------------------------------------------------------------------

    fn leaf() -> ViewNode {
        ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })
    }

    fn modified(modifier: Modifier) -> ViewNode {
        ViewNode::Modified {
            modifier,
            child: Box::new(leaf()),
        }
    }

    fn render_with(modifier: Modifier) -> RenderCtx {
        let mut ctx = RenderCtx::new();
        render_node(
            &modified(modifier),
            Rect::new(0.0, 0.0, 100.0, 100.0),
            &mut ctx,
        );
        ctx
    }

    // ---------- Background fills ----------

    #[test]
    fn modifier_background_color_emits_fill_rect() {
        let ctx = render_with(Modifier::Background(BackgroundFill::Color(Color::WHITE)));
        assert!(
            matches!(ctx.commands[0], DrawCommand::FillRect { color, .. } if color == Color::WHITE)
        );
    }

    #[test]
    fn modifier_background_linear_gradient_emits_n_strips() {
        let ctx = render_with(Modifier::Background(BackgroundFill::LinearGradient {
            angle_deg: 0.0,
            stops: vec![
                (0.0, Color::WHITE),
                (0.5, Color::BLACK),
                (1.0, Color::WHITE),
            ],
        }));
        let fills = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::FillRect { .. }))
            .count();
        assert_eq!(fills, 3);
    }

    #[test]
    fn modifier_background_radial_gradient_emits_concentric_rings() {
        let ctx = render_with(Modifier::Background(BackgroundFill::RadialGradient {
            center: Alignment2D::default(),
            stops: vec![(0.0, Color::WHITE), (1.0, Color::BLACK)],
        }));
        let fills = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::FillRect { .. }))
            .count();
        assert_eq!(fills, 2);
    }

    // ---------- Frame variants ----------

    #[test]
    fn modifier_frame_aligned_top_leading_pins_to_origin() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::FrameAligned {
                width: Some(20.0),
                height: Some(10.0),
                alignment: Alignment2D::new(HAlignment::Leading, VAlignment::Top),
            },
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(5.0, 7.0, 100.0, 100.0), &mut ctx);
        let DrawCommand::FillRect { rect, .. } = &ctx.commands[0] else {
            panic!("expected fill rect");
        };
        assert_eq!(rect.x, 5.0);
        assert_eq!(rect.y, 7.0);
        assert_eq!(rect.width, 20.0);
        assert_eq!(rect.height, 10.0);
    }

    #[test]
    fn modifier_frame_range_clamps_to_max() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::FrameRange {
                min_width: Some(10.0),
                ideal_width: Some(50.0),
                max_width: Some(30.0),
                min_height: None,
                ideal_height: None,
                max_height: Some(40.0),
                alignment: Alignment2D::new(HAlignment::Leading, VAlignment::Top),
            },
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 200.0, 200.0), &mut ctx);
        let DrawCommand::FillRect { rect, .. } = &ctx.commands[0] else {
            panic!("expected fill");
        };
        assert_eq!(rect.width, 30.0);
        assert_eq!(rect.height, 40.0);
    }

    // ---------- Border / Shadow ----------

    #[test]
    fn modifier_border_emits_stroke_after_child() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::Border {
                width: 2.0,
                color: Color::WHITE,
            },
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::BLACK)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        // Background fill comes first, border stroke last.
        assert!(matches!(ctx.commands[0], DrawCommand::FillRect { .. }));
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::StrokeRect { width, .. } if (*width - 2.0).abs() < 1e-6
        ));
    }

    #[test]
    fn modifier_shadow_paints_offset_rect_first() {
        let ctx = render_with(Modifier::Shadow {
            color: Color::rgba(0.0, 0.0, 0.0, 0.5),
            radius: 4.0,
            offset_x: 3.0,
            offset_y: 5.0,
        });
        let DrawCommand::FillRect { rect, color } = &ctx.commands[0] else {
            panic!("expected shadow fill");
        };
        assert_eq!(rect.x, 3.0);
        assert_eq!(rect.y, 5.0);
        assert!((color.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn modifier_inner_shadow_paints_after_child() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::InnerShadow {
                color: Color::WHITE,
                radius: 2.0,
                offset_x: 1.0,
                offset_y: 1.0,
            },
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::BLACK)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        // Background first, inner-shadow last.
        assert!(
            matches!(ctx.commands[0], DrawCommand::FillRect { color, .. } if color == Color::BLACK)
        );
        let DrawCommand::FillRect { color, .. } = ctx.commands.last().unwrap() else {
            panic!("expected fill");
        };
        assert_eq!(*color, Color::WHITE);
    }

    // ---------- Transforms ----------

    #[test]
    fn modifier_rotation_brackets_with_push_pop_transform() {
        let ctx = render_with(Modifier::Rotation(90.0));
        assert!(matches!(ctx.commands[0], DrawCommand::PushTransform { .. }));
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::PopTransform
        ));
    }

    #[test]
    fn modifier_scale_brackets_with_push_pop_transform() {
        let ctx = render_with(Modifier::Scale {
            x: 2.0,
            y: 3.0,
            anchor: Alignment2D::default(),
        });
        let DrawCommand::PushTransform { transform } = &ctx.commands[0] else {
            panic!("expected push transform");
        };
        // Scale matrix has the scale factors on the diagonal.
        assert!((transform.0[0] - 2.0).abs() < 1e-6);
        assert!((transform.0[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn modifier_offset_translates() {
        let ctx = render_with(Modifier::Offset { x: 7.0, y: -3.0 });
        let DrawCommand::PushTransform { transform } = &ctx.commands[0] else {
            panic!("expected push transform");
        };
        assert_eq!(transform.0[4], 7.0);
        assert_eq!(transform.0[5], -3.0);
    }

    #[test]
    fn modifier_position_overrides_origin() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::Position { x: 99.0, y: 88.0 },
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 100.0, 100.0), &mut ctx);
        let DrawCommand::FillRect { rect, .. } = &ctx.commands[0] else {
            panic!("expected fill");
        };
        assert_eq!(rect.x, 99.0);
        assert_eq!(rect.y, 88.0);
    }

    // ---------- Clip / Mask ----------

    #[test]
    fn modifier_clip_brackets_with_push_pop_clip_shape() {
        let ctx = render_with(Modifier::Clip(ClipShape::Circle));
        assert!(matches!(ctx.commands[0], DrawCommand::PushClipShape { .. }));
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::PopClipShape
        ));
    }

    #[test]
    fn modifier_clip_path_recorded_in_sidetable() {
        let ctx = render_with(Modifier::Clip(ClipShape::Path(vec![
            PathCommand::MoveTo(0.0, 0.0),
            PathCommand::LineTo(10.0, 10.0),
            PathCommand::Close,
        ])));
        assert_eq!(ctx.clip_shapes.len(), 1);
        assert_eq!(ctx.clip_shapes[0].len(), 3);
    }

    #[test]
    fn modifier_mask_records_clip_shape() {
        let ctx = render_with(Modifier::Mask(ClipShape::Capsule));
        let DrawCommand::PushClipShape { shape, .. } = &ctx.commands[0] else {
            panic!("expected push clip shape");
        };
        assert!(matches!(shape, ClipShapeKind::Capsule));
    }

    // ---------- Hit regions ----------

    #[test]
    fn modifier_on_tap_records_hit_region() {
        let ctx = render_with(Modifier::OnTap(Rc::new(|| {})));
        assert_eq!(ctx.hit_regions.len(), 1);
        assert!(matches!(ctx.hit_regions[0].kind, HitKind::Tap(_)));
    }

    #[test]
    fn modifier_on_hover_records_hit_region() {
        let ctx = render_with(Modifier::OnHover(Rc::new(|_| {})));
        assert_eq!(ctx.hit_regions.len(), 1);
        assert!(matches!(ctx.hit_regions[0].kind, HitKind::Hover(_)));
    }

    #[test]
    fn modifier_on_long_press_records_duration() {
        let ctx = render_with(Modifier::OnLongPress {
            duration_ms: 500,
            action: Rc::new(|| {}),
        });
        assert!(matches!(
            ctx.hit_regions[0].kind,
            HitKind::LongPress {
                duration_ms: 500,
                ..
            }
        ));
    }

    #[test]
    fn modifier_on_drag_records_hit_region() {
        let ctx = render_with(Modifier::OnDrag(Rc::new(|_, _| {})));
        assert!(matches!(ctx.hit_regions[0].kind, HitKind::Drag(_)));
    }

    // ---------- Lifecycle ----------

    #[test]
    fn modifier_on_appear_records_lifecycle_hook() {
        let ctx = render_with(Modifier::OnAppear(Rc::new(|| {})));
        assert_eq!(ctx.appear_callbacks.len(), 1);
    }

    #[test]
    fn modifier_on_disappear_records_lifecycle_hook() {
        let ctx = render_with(Modifier::OnDisappear(Rc::new(|| {})));
        assert_eq!(ctx.disappear_callbacks.len(), 1);
    }

    // ---------- Annotations ----------

    #[test]
    fn modifier_accessibility_role_records_annotation() {
        let ctx = render_with(Modifier::AccessibilityRole(AccessibilityRole::Button));
        assert_eq!(ctx.annotations.len(), 1);
        assert_eq!(ctx.annotations[0].role, Some(AccessibilityRole::Button));
    }

    #[test]
    fn modifier_accessibility_hint_records_annotation() {
        let ctx = render_with(Modifier::AccessibilityHint("press to submit".into()));
        assert_eq!(ctx.annotations[0].hint.as_deref(), Some("press to submit"));
    }

    #[test]
    fn modifier_accessibility_value_records_annotation() {
        let ctx = render_with(Modifier::AccessibilityValue("50%".into()));
        assert_eq!(ctx.annotations[0].value.as_deref(), Some("50%"));
    }

    #[test]
    fn modifier_accessibility_hidden_records_annotation() {
        let ctx = render_with(Modifier::AccessibilityHidden(true));
        assert_eq!(ctx.annotations[0].hidden, Some(true));
    }

    #[test]
    fn modifier_tab_index_records_annotation() {
        let ctx = render_with(Modifier::TabIndex(3));
        assert_eq!(ctx.annotations[0].tab_index, Some(3));
    }

    #[test]
    fn modifier_allows_hit_testing_records_annotation() {
        let ctx = render_with(Modifier::AllowsHitTesting(false));
        assert_eq!(ctx.annotations[0].allows_hit_testing, Some(false));
    }

    #[test]
    fn modifier_tooltip_records_annotation() {
        let ctx = render_with(Modifier::Tooltip("Help".into()));
        assert_eq!(ctx.annotations[0].tooltip.as_deref(), Some("Help"));
    }

    #[test]
    fn modifier_z_index_records_annotation() {
        let ctx = render_with(Modifier::ZIndex(5.0));
        assert_eq!(ctx.annotations[0].z_index, Some(5.0));
    }

    #[test]
    fn modifier_animation_records_spec() {
        let ctx = render_with(Modifier::Animation(
            AnimationSpec::new(AnimationCurve::EaseInOut, 200).keyed(42),
        ));
        let spec = ctx.annotations[0].animation.unwrap();
        assert_eq!(spec.duration_ms, 200);
        assert_eq!(spec.curve, AnimationCurve::EaseInOut);
        assert_eq!(spec.key, 42);
    }

    #[test]
    fn modifier_transition_records_kind() {
        let ctx = render_with(Modifier::Transition(TransitionKind::Opacity));
        assert!(matches!(
            ctx.annotations[0].transition,
            Some(TransitionKind::Opacity)
        ));
    }

    #[test]
    fn modifier_tag_records_annotation() {
        let ctx = render_with(Modifier::Tag("main".into()));
        assert_eq!(ctx.annotations[0].tag.as_deref(), Some("main"));
    }

    #[test]
    fn modifier_hidden_skips_render() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::Hidden(true),
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        // Annotation recorded, but no fill-rect emitted.
        assert_eq!(ctx.annotations.len(), 1);
        assert!(
            ctx.commands
                .iter()
                .all(|c| !matches!(c, DrawCommand::FillRect { .. }))
        );
    }

    #[test]
    fn modifier_hidden_false_still_paints() {
        let mut ctx = RenderCtx::new();
        let node = ViewNode::Modified {
            modifier: Modifier::Hidden(false),
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        };
        render_node(&node, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        assert!(
            ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::FillRect { .. }))
        );
    }

    // ---------- Opacity push/pop ----------

    #[test]
    fn modifier_opacity_brackets_with_push_pop() {
        let ctx = render_with(Modifier::Opacity(0.4));
        assert!(matches!(ctx.commands[0], DrawCommand::PushOpacity { .. }));
        assert!(matches!(
            ctx.commands.last().unwrap(),
            DrawCommand::PopOpacity
        ));
    }

    // ---------- New primitive lowering ----------

    #[test]
    fn to_primitives_lowers_push_opacity() {
        let mut ctx = RenderCtx::new();
        ctx.commands.push(DrawCommand::PushOpacity { alpha: 0.5 });
        ctx.commands.push(DrawCommand::PopOpacity);
        let prims = ctx.to_primitives();
        assert!(
            matches!(prims[0], RenderPrimitive::PushOpacity { alpha } if (alpha - 0.5).abs() < 1e-6)
        );
        assert!(matches!(prims[1], RenderPrimitive::PopOpacity));
    }

    #[test]
    fn to_primitives_lowers_push_transform() {
        let mut ctx = RenderCtx::new();
        ctx.commands.push(DrawCommand::PushTransform {
            transform: Transform2D::translate(5.0, 6.0),
        });
        ctx.commands.push(DrawCommand::PopTransform);
        let prims = ctx.to_primitives();
        assert!(
            matches!(prims[0], RenderPrimitive::PushTransform { transform } if transform[4] == 5.0 && transform[5] == 6.0)
        );
        assert!(matches!(prims[1], RenderPrimitive::PopTransform));
    }

    #[test]
    fn to_primitives_lowers_clip_shape_to_clip_rect() {
        let mut ctx = RenderCtx::new();
        ctx.commands.push(DrawCommand::PushClipShape {
            rect: LayoutRect::new(0.0, 0.0, 10.0, 10.0),
            shape: ClipShapeKind::Circle,
        });
        ctx.commands.push(DrawCommand::PopClipShape);
        let prims = ctx.to_primitives();
        assert!(matches!(prims[0], RenderPrimitive::PushClip { .. }));
        assert!(matches!(prims[1], RenderPrimitive::PopClip));
    }

    // ---------- Transform2D arithmetic ----------

    #[test]
    fn transform2d_identity_is_diagonal() {
        let t = Transform2D::identity();
        assert_eq!(t.0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn transform2d_translate_sets_offsets() {
        let t = Transform2D::translate(3.0, 4.0);
        assert_eq!(t.0[4], 3.0);
        assert_eq!(t.0[5], 4.0);
    }

    #[test]
    fn transform2d_rotate_around_zero_is_identity_at_origin() {
        let t = Transform2D::rotate_around(0.0, 0.0, 0.0);
        assert!((t.0[0] - 1.0).abs() < 1e-6);
        assert!((t.0[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transform2d_scale_around_keeps_anchor_fixed() {
        // Scaling around (10, 10) by 2 should leave (10, 10) at (10, 10).
        let t = Transform2D::scale_around(2.0, 2.0, 10.0, 10.0);
        let (a, b, c, d, e, f) = (t.0[0], t.0[1], t.0[2], t.0[3], t.0[4], t.0[5]);
        let xp = a * 10.0 + c * 10.0 + e;
        let yp = b * 10.0 + d * 10.0 + f;
        assert!((xp - 10.0).abs() < 1e-4);
        assert!((yp - 10.0).abs() < 1e-4);
    }

    // ---------- Background helper paint ----------

    #[test]
    fn paint_background_skips_empty_linear_gradient() {
        let mut ctx = RenderCtx::new();
        paint_background(
            &BackgroundFill::LinearGradient {
                angle_deg: 0.0,
                stops: vec![],
            },
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            None,
        );
        assert!(ctx.commands.is_empty());
    }

    #[test]
    fn paint_background_skips_empty_radial_gradient() {
        let mut ctx = RenderCtx::new();
        paint_background(
            &BackgroundFill::RadialGradient {
                center: Alignment2D::default(),
                stops: vec![],
            },
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            None,
        );
        assert!(ctx.commands.is_empty());
    }

    // ---------- Multiple gestures stack independently ----------

    #[test]
    fn nested_gesture_modifiers_all_record() {
        let inner = ViewNode::Modified {
            modifier: Modifier::OnTap(Rc::new(|| {})),
            child: Box::new(leaf()),
        };
        let outer = ViewNode::Modified {
            modifier: Modifier::OnHover(Rc::new(|_| {})),
            child: Box::new(inner),
        };
        let mut ctx = RenderCtx::new();
        render_node(&outer, Rect::new(0.0, 0.0, 50.0, 50.0), &mut ctx);
        assert_eq!(ctx.hit_regions.len(), 2);
    }

    // ---------- Annotation accessibility role / hidden carries through ----------

    #[test]
    fn accessibility_label_already_recorded_via_annotation() {
        let ctx = render_with(Modifier::AccessibilityLabel("Hello".into()));
        assert_eq!(ctx.annotations[0].label.as_deref(), Some("Hello"));
    }

    #[test]
    fn shape_kind_path_increments_index() {
        let mut ctx = RenderCtx::new();
        let k1 = shape_kind(&ClipShape::Path(vec![PathCommand::Close]), &mut ctx);
        let k2 = shape_kind(&ClipShape::Path(vec![PathCommand::Close]), &mut ctx);
        match (k1, k2) {
            (ClipShapeKind::Path { index: 0 }, ClipShapeKind::Path { index: 1 }) => {}
            other => panic!("expected ascending path indexes, got {:?}", other),
        }
        assert_eq!(ctx.clip_shapes.len(), 2);
    }

    #[test]
    fn shape_kind_rounded_rect_carries_radius() {
        let mut ctx = RenderCtx::new();
        let k = shape_kind(&ClipShape::RoundedRect { radius: 8.0 }, &mut ctx);
        assert!(matches!(k, ClipShapeKind::RoundedRect { radius: r } if (r - 8.0).abs() < 1e-6));
    }

    // -------------------------------------------------------------------------
    // Phase 4 — driver-aware transition rendering
    // -------------------------------------------------------------------------

    fn transition_node(kind: TransitionKind) -> ViewNode {
        ViewNode::Modified {
            modifier: Modifier::Transition(kind),
            child: Box::new(ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                child: Box::new(leaf()),
            }),
        }
    }

    #[test]
    fn transition_none_passes_through_without_extra_commands() {
        use crate::animation::{AnimationDriver, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::None),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        // Annotation + the underlying Background fill — but no Push/Pop transform.
        assert!(
            ctx.commands
                .iter()
                .all(|c| !matches!(c, DrawCommand::PushTransform { .. }))
        );
    }

    #[test]
    fn transition_opacity_brackets_with_push_pop_opacity_on_enter() {
        use crate::animation::{AnimationCurve, AnimationDriver, AnimationSpec, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Opacity),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        let pushes = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::PushOpacity { .. }))
            .count();
        let pops = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::PopOpacity))
            .count();
        assert_eq!(pushes, 1);
        assert_eq!(pops, 1);
    }

    #[test]
    fn transition_opacity_alpha_zero_at_t_zero_entering() {
        use crate::animation::{AnimationCurve, AnimationDriver, AnimationSpec, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Opacity),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        let DrawCommand::PushOpacity { alpha } = ctx
            .commands
            .iter()
            .find(|c| matches!(c, DrawCommand::PushOpacity { .. }))
            .unwrap()
        else {
            unreachable!()
        };
        assert!((alpha).abs() < 1e-6);
    }

    #[test]
    fn transition_opacity_full_alpha_at_t_one_entering() {
        use crate::animation::{AnimationCurve, AnimationDriver, AnimationSpec, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        driver.tick(100);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Opacity),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        let DrawCommand::PushOpacity { alpha } = ctx
            .commands
            .iter()
            .find(|c| matches!(c, DrawCommand::PushOpacity { .. }))
            .unwrap()
        else {
            unreachable!()
        };
        assert!((alpha - 1.0).abs() < 1e-3);
    }

    #[test]
    fn transition_slide_pushes_translate_transform() {
        use crate::animation::{
            AnimationCurve, AnimationDriver, AnimationSpec, Edge as AnimEdge, Presence,
        };
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        // Register an animation so eased_progress returns t=0 at start.
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Slide {
                edge: AnimEdge::Top,
            }),
            Rect::new(0.0, 0.0, 10.0, 20.0),
            &mut ctx,
            1,
            &driver,
        );
        let pushes: Vec<&DrawCommand> = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::PushTransform { .. }))
            .collect();
        assert_eq!(pushes.len(), 1);
        let DrawCommand::PushTransform { transform } = pushes[0] else {
            unreachable!()
        };
        // Entering at t=0 is fully off-screen by `-height` along Top edge.
        assert!((transform.0[5] - -20.0).abs() < 1e-3);
    }

    #[test]
    fn transition_slide_offset_zero_when_complete() {
        use crate::animation::{
            AnimationCurve, AnimationDriver, AnimationSpec, Edge as AnimEdge, Presence,
        };
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        driver.tick(100);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Slide {
                edge: AnimEdge::Bottom,
            }),
            Rect::new(0.0, 0.0, 10.0, 20.0),
            &mut ctx,
            1,
            &driver,
        );
        let pushes: Vec<&DrawCommand> = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::PushTransform { .. }))
            .collect();
        let DrawCommand::PushTransform { transform } = pushes[0] else {
            unreachable!()
        };
        assert!(transform.0[5].abs() < 1e-3);
    }

    #[test]
    fn transition_scale_pushes_scale_transform() {
        use crate::animation::{AnimationCurve, AnimationDriver, AnimationSpec, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Scale { from: 0.0 }),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        let pushes: Vec<&DrawCommand> = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::PushTransform { .. }))
            .collect();
        assert_eq!(pushes.len(), 1);
        let DrawCommand::PushTransform { transform } = pushes[0] else {
            unreachable!()
        };
        // At t=0 entering, factor = from = 0.
        assert!(transform.0[0].abs() < 1e-3);
    }

    #[test]
    fn transition_asymmetric_picks_enter_side() {
        use crate::animation::{
            AnimationCurve, AnimationDriver, AnimationSpec, Edge as AnimEdge, Presence,
        };
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let kind = TransitionKind::Asymmetric {
            enter: Box::new(TransitionKind::Opacity),
            exit: Box::new(TransitionKind::Slide {
                edge: AnimEdge::Top,
            }),
        };
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(kind),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        // Enter side = Opacity → Push/PopOpacity but no transform.
        assert!(
            ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::PushOpacity { .. }))
        );
        assert!(
            !ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::PushTransform { .. }))
        );
    }

    #[test]
    fn transition_asymmetric_picks_exit_side_when_leaving() {
        use crate::animation::{
            AnimationCurve, AnimationDriver, AnimationSpec, Edge as AnimEdge, Presence,
        };
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Leaving);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let kind = TransitionKind::Asymmetric {
            enter: Box::new(TransitionKind::Opacity),
            exit: Box::new(TransitionKind::Slide {
                edge: AnimEdge::Top,
            }),
        };
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(kind),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        // Exit side = Slide → PushTransform.
        assert!(
            ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::PushTransform { .. }))
        );
    }

    #[test]
    fn transition_stable_presence_passes_through() {
        use crate::animation::{AnimationDriver, Presence};
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Stable);
        let mut ctx = RenderCtx::new();
        crate::render::render_node_with_driver(
            &transition_node(TransitionKind::Opacity),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
            1,
            &driver,
        );
        // Stable + Opacity → no Push/Pop opacity wrapping.
        assert!(
            !ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::PushOpacity { .. }))
        );
    }

    #[test]
    fn render_node_without_driver_skips_transition_envelope() {
        // Legacy entry point — no driver attached, so the transition just
        // records its annotation and otherwise passes through.
        let mut ctx = RenderCtx::new();
        render_node(
            &transition_node(TransitionKind::Opacity),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            &mut ctx,
        );
        assert!(
            !ctx.commands
                .iter()
                .any(|c| matches!(c, DrawCommand::PushOpacity { .. }))
        );
        // Annotation was still recorded.
        assert!(ctx.annotations.iter().any(|a| a.transition.is_some()));
    }

    #[test]
    fn animation_modifier_records_via_keyed_spec() {
        let ctx = render_with(Modifier::Animation(
            AnimationSpec::new(AnimationCurve::ease_in(), 200).keyed(99),
        ));
        let spec = ctx.annotations[0].animation.unwrap();
        assert_eq!(spec.duration_ms, 200);
        assert_eq!(spec.key, 99);
    }
}
