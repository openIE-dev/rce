//! View tree — both the legacy widget trait and the new declarative
//! `ViewNode` enum.
//!
//! Photon currently exposes two layered surfaces:
//!
//! 1. **Legacy widget trait** ([`View`]) — `layout(&Constraint) -> Size` and
//!    `render(&LayoutRect, &mut RenderCtx)`. Every widget in
//!    [`crate::widgets`] implements this. The [`ViewExt`] extension trait
//!    adds a `render_root` shorthand. This is what `nova_shell`'s
//!    `chrome_photon` and `perf_photon` modules use today.
//!
//! 2. **Declarative tree** ([`ViewNode`]) — an enum-dispatched tree that the
//!    new render path ([`crate::render::render_node`]) walks. Each leaf has
//!    its own struct ([`TextLeaf`], [`ImageLeaf`]) and modifiers wrap
//!    sub-trees ([`Modifier`]). New code constructs `ViewNode`s directly;
//!    legacy widgets convert via their `to_node()` methods (see
//!    [`crate::widgets`]).
//!
//! Every legacy widget gains a default `body()` hook on [`View`] so it can
//! also be rendered through the declarative path.

use crate::color::Color;
use crate::layout::{Alignment2D, Constraint, EdgeInsets, Layout, LayoutRect, Size};
use crate::render::RenderCtx;
use std::fmt;
use std::rc::Rc;

/// Style fields shared across every legacy widget. Widgets compose this
/// struct so the common builder methods can be added in one place. New
/// declarative code uses [`Modifier`] instead.
#[derive(Debug, Clone, Default)]
pub struct Style {
    pub background: Option<Color>,
    pub border_width: f32,
    pub border_color: Option<Color>,
    pub padding_x: f32,
    pub padding_y: f32,
    pub min_width: f32,
    pub min_height: f32,
    pub max_width: f32,
    pub max_height: f32,
}

/// Core view contract. Each widget implements `layout` (returning its
/// preferred size given a parent constraint) and `render` (pushing draw
/// commands into a [`RenderCtx`]).
///
/// `body` is the declarative bridge: implementations may override it to
/// return a [`ViewNode`] tree rendered via [`crate::render::render_node`].
/// Default returns an empty [`LeafKind::Spacer`] of zero length.
pub trait View {
    /// Compute and return the size this view wants, given a parent constraint.
    fn layout(&self, constraint: &Constraint) -> Size;

    /// Render this view into the given rect.
    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx);

    /// Convert this view into a [`ViewNode`] tree (declarative path).
    /// Default returns a zero-length spacer; widgets with a meaningful
    /// declarative form override this.
    fn body(&self) -> ViewNode {
        ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })
    }
}

impl View for Box<dyn View> {
    fn layout(&self, constraint: &Constraint) -> Size {
        (**self).layout(constraint)
    }
    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        (**self).render(rect, ctx)
    }
    fn body(&self) -> ViewNode {
        (**self).body()
    }
}

/// Extension trait — provides ergonomic helpers callers expect.
///
/// Two surfaces are exposed here:
///
/// 1. [`ViewExt::render_root`] — works on any `View` (object-safe slice).
/// 2. The full SwiftUI-style modifier chain (`padding`, `background`,
///    `on_tap`, …) added in Phase 3. Those methods consume `self`, so
///    they are only available on `Sized` views.
///
/// The blanket `impl<T: View + ?Sized>` implementation supplies
/// `render_root`; sized views additionally pick up the chained modifier
/// methods through this same trait.
pub trait ViewExt: View {
    /// Render at the origin with a 0×0 rect, letting the view derive its
    /// own size from layout.
    fn render_root(&self, ctx: &mut RenderCtx) {
        let c = Constraint::loose(f32::INFINITY, f32::INFINITY);
        let size = self.layout(&c);
        self.render(&LayoutRect::new(0.0, 0.0, size.width, size.height), ctx);
    }

    /// Wrap this view in a [`Modifier::Padding`].
    fn padding(self, edges: EdgeInsets) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Padding(edges))
    }

    /// Symmetric padding on every edge.
    fn padding_all(self, value: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Padding(EdgeInsets::all(value)))
    }

    /// Independent horizontal / vertical padding.
    fn padding_xy(self, x: f32, y: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Padding(EdgeInsets::xy(x, y)))
    }

    /// Pad only the top edge.
    fn padding_top(self, v: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Padding(EdgeInsets {
                top: v,
                ..EdgeInsets::default()
            }),
        )
    }

    /// Pad only the right edge.
    fn padding_right(self, v: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Padding(EdgeInsets {
                right: v,
                ..EdgeInsets::default()
            }),
        )
    }

    /// Pad only the bottom edge.
    fn padding_bottom(self, v: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Padding(EdgeInsets {
                bottom: v,
                ..EdgeInsets::default()
            }),
        )
    }

    /// Pad only the left edge.
    fn padding_left(self, v: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Padding(EdgeInsets {
                left: v,
                ..EdgeInsets::default()
            }),
        )
    }

    /// Wrap this view in a [`Modifier::Background`].
    fn background(self, fill: impl Into<BackgroundFill>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Background(fill.into()))
    }

    /// Wrap with a solid-colour background.
    fn background_color(self, c: Color) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Background(BackgroundFill::Color(c)))
    }

    /// Inherit the foreground tint to the wrapped subtree.
    fn foreground(self, c: Color) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Foreground(c))
    }

    /// Apply a global alpha multiplier (clamped 0..=1).
    fn opacity(self, v: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Opacity(v))
    }

    /// Round the wrapped child's corners by `r` pixels (advisory).
    fn corner_radius(self, r: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::CornerRadius(r))
    }

    /// Stroke an outline border around the wrapped child.
    fn border(self, width: f32, color: Color) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Border { width, color })
    }

    /// Drop a shadow behind the child.
    fn shadow(self, color: Color, radius: f32, dx: f32, dy: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Shadow {
                color,
                radius,
                offset_x: dx,
                offset_y: dy,
            },
        )
    }

    /// Stack multiple shadows behind the child. Drawn back-to-front in
    /// `stack` order, so the *first* element ends up furthest from the
    /// surface and the *last* element sits closest. Modern "elevated" UIs
    /// (Linear, Vercel, Notion) typically layer two or three:
    ///
    /// ```text
    /// vec![
    ///     Shadow::new(Color::rgba(0.0,0.0,0.0,0.04), 1.0,  0.0, 1.0),
    ///     Shadow::new(Color::rgba(0.0,0.0,0.0,0.06), 4.0,  0.0, 2.0),
    ///     Shadow::new(Color::rgba(0.0,0.0,0.0,0.10), 16.0, 0.0, 8.0),
    /// ]
    /// ```
    fn shadows(self, stack: Vec<Shadow>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Shadows(stack))
    }

    /// Inset shadow drawn over the child.
    fn inner_shadow(self, color: Color, radius: f32, dx: f32, dy: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::InnerShadow {
                color,
                radius,
                offset_x: dx,
                offset_y: dy,
            },
        )
    }

    /// Apply a `backdrop-filter` to the destination behind the child. Pairs
    /// with the GPU compositor's blur pipeline; on backends without a real
    /// blur shader the modifier degrades to a no-op.
    fn backdrop_filter(self, filter: BackdropFilter) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::BackdropFilter(filter))
    }

    /// "Glass material" preset — convenience for the popular
    /// frosted / Liquid Glass aesthetic. Combines:
    ///
    /// 1. [`BackdropFilter::glass`] (20 px blur, slight saturate boost)
    /// 2. A 0.5 px white-ish hairline border
    /// 3. A 16 px radius rounded corner
    /// 4. A single soft shadow underneath
    ///
    /// Wrap a translucent `Background` colour (e.g. `Color::rgba(1,1,1,0.6)`)
    /// inside this for the typical card.
    fn glass(self) -> ViewNode
    where
        Self: Sized,
    {
        self.backdrop_filter(BackdropFilter::glass())
            .border(0.5, Color::rgba(1.0, 1.0, 1.0, 0.18))
            .corner_radius(16.0)
            .shadows(vec![Shadow {
                color: Color::rgba(0.0, 0.0, 0.0, 0.08),
                radius: 16.0,
                dx: 0.0,
                dy: 8.0,
            }])
    }

    /// Force a fixed `width` / `height`. Pass `None` on an axis to inherit.
    fn frame(self, width: Option<f32>, height: Option<f32>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Frame { width, height })
    }

    /// Frame with explicit alignment.
    fn frame_aligned(
        self,
        width: Option<f32>,
        height: Option<f32>,
        alignment: Alignment2D,
    ) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::FrameAligned {
                width,
                height,
                alignment,
            },
        )
    }

    /// Min/max bounded frame with optional ideal sizes.
    fn frame_range(
        self,
        min_w: Option<f32>,
        ideal_w: Option<f32>,
        max_w: Option<f32>,
        min_h: Option<f32>,
        ideal_h: Option<f32>,
        max_h: Option<f32>,
        alignment: Alignment2D,
    ) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::FrameRange {
                min_width: min_w,
                ideal_width: ideal_w,
                max_width: max_w,
                min_height: min_h,
                ideal_height: ideal_h,
                max_height: max_h,
                alignment,
            },
        )
    }

    /// Rotate the wrapped subtree by `degrees` around its centre.
    fn rotation(self, degrees: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Rotation(degrees))
    }

    /// Uniform scale factor.
    fn scale(self, factor: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::Scale {
                x: factor,
                y: factor,
                anchor: Alignment2D::default(),
            },
        )
    }

    /// Independent x/y scale with an explicit anchor.
    fn scale_xy(self, x: f32, y: f32, anchor: Alignment2D) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Scale { x, y, anchor })
    }

    /// Translate the wrapped subtree by `(dx, dy)`.
    fn offset(self, dx: f32, dy: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Offset { x: dx, y: dy })
    }

    /// Place the wrapped subtree absolutely at `(x, y)`.
    fn position(self, x: f32, y: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Position { x, y })
    }

    /// Clip the subtree to a [`ClipShape`].
    fn clip(self, shape: ClipShape) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Clip(shape))
    }

    /// Convenience: clip to the parent rect.
    fn clip_rect(self) -> ViewNode
    where
        Self: Sized,
    {
        self.clip(ClipShape::Rect)
    }

    /// Convenience: clip to a capsule (rounded ends).
    fn clip_capsule(self) -> ViewNode
    where
        Self: Sized,
    {
        self.clip(ClipShape::Capsule)
    }

    /// Convenience: clip to a circle inscribed in the parent rect.
    fn clip_circle(self) -> ViewNode
    where
        Self: Sized,
    {
        self.clip(ClipShape::Circle)
    }

    /// Mask the subtree with the given shape.
    fn mask(self, shape: ClipShape) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Mask(shape))
    }

    /// Tap callback.
    fn on_tap(self, f: impl Fn() + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::OnTap(Rc::new(f)))
    }

    /// Hover callback — `bool` argument is `true` on enter, `false` on leave.
    fn on_hover(self, f: impl Fn(bool) + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::OnHover(Rc::new(f)))
    }

    /// Long-press callback.
    fn on_long_press(self, duration_ms: u32, f: impl Fn() + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(
            self,
            Modifier::OnLongPress {
                duration_ms,
                action: Rc::new(f),
            },
        )
    }

    /// Drag callback — receives `(dx, dy)` in local coordinates.
    fn on_drag(self, f: impl Fn(f32, f32) + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::OnDrag(Rc::new(f)))
    }

    /// `on_appear` lifecycle callback.
    fn on_appear(self, f: impl Fn() + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::OnAppear(Rc::new(f)))
    }

    /// `on_disappear` lifecycle callback.
    fn on_disappear(self, f: impl Fn() + 'static) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::OnDisappear(Rc::new(f)))
    }

    /// Accessibility label.
    fn accessibility_label(self, s: impl Into<String>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AccessibilityLabel(s.into()))
    }

    /// Accessibility role.
    fn accessibility_role(self, r: AccessibilityRole) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AccessibilityRole(r))
    }

    /// Accessibility hint (long-form description).
    fn accessibility_hint(self, s: impl Into<String>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AccessibilityHint(s.into()))
    }

    /// Accessibility value (e.g. slider current).
    fn accessibility_value(self, s: impl Into<String>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AccessibilityValue(s.into()))
    }

    /// Exclude from the accessibility tree.
    fn accessibility_hidden(self, b: bool) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AccessibilityHidden(b))
    }

    /// Tab-order index.
    fn tab_index(self, i: i32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::TabIndex(i))
    }

    /// Disable hit-testing on the subtree.
    fn allows_hit_testing(self, b: bool) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::AllowsHitTesting(b))
    }

    /// Tooltip text.
    fn tooltip(self, s: impl Into<String>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Tooltip(s.into()))
    }

    /// Z-index reordering.
    fn z_index(self, z: f32) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::ZIndex(z))
    }

    /// Attach an [`crate::animation::AnimationSpec`] envelope to this
    /// subtree — animatable modifiers nested inside will interpolate from
    /// their previous value to the current one over `spec.duration_ms`
    /// along `spec.curve` whenever `spec.key` changes.
    fn animation(self, spec: crate::animation::AnimationSpec) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Animation(spec))
    }

    /// Enter / exit transition.
    fn transition(self, t: TransitionKind) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Transition(t))
    }

    /// Identity tag for nav-stack matching.
    fn tag(self, s: impl Into<String>) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Tag(s.into()))
    }

    /// Visibility toggle that does not collapse layout.
    fn hidden(self, b: bool) -> ViewNode
    where
        Self: Sized,
    {
        crate::view_ext::wrap(self, Modifier::Hidden(b))
    }
}

impl<T: View + ?Sized> ViewExt for T {}

/// Helper to box a view for use in `h_stack(vec![boxed(...), ...])`.
pub fn boxed<V: View + 'static>(v: V) -> Box<dyn View> {
    Box::new(v)
}

// =============================================================================
// Declarative view tree
// =============================================================================

/// Declarative tree node. Built by `View::body` implementations and walked
/// by [`crate::render::render_node`].
pub enum ViewNode {
    /// Single drawable element — text, image, spacer, divider, or canvas.
    Leaf(LeafKind),
    /// Layout container — places `children` per the [`Layout`] strategy.
    Container {
        layout: Box<dyn Layout>,
        children: Vec<ViewNode>,
    },
    /// Decorator — wraps `child` with a [`Modifier`].
    Modified {
        modifier: Modifier,
        child: Box<ViewNode>,
    },
    /// Lazily-evaluated subtree. The closure is called each time the node
    /// is rendered, allowing it to read [`crate::signal::Signal`]s and
    /// auto-subscribe.
    Stateful(Rc<dyn Fn() -> ViewNode>),
}

impl fmt::Debug for ViewNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViewNode::Leaf(k) => f.debug_tuple("Leaf").field(k).finish(),
            ViewNode::Container { children, .. } => f
                .debug_struct("Container")
                .field("children", &children.len())
                .finish_non_exhaustive(),
            ViewNode::Modified { modifier, .. } => f
                .debug_struct("Modified")
                .field("modifier", modifier)
                .finish_non_exhaustive(),
            ViewNode::Stateful(_) => f.write_str("Stateful(..)"),
        }
    }
}

/// Treat a [`ViewNode`] as a [`View`] so the [`ViewExt`] modifier chain can
/// be applied directly to results returned from earlier modifiers (or to
/// hand-built `ViewNode` literals).
///
/// Layout/render at this level just defer to the declarative path: `body()`
/// returns a deep clone of `self`, while the legacy `layout`/`render` pair
/// degrades to a zero-size no-op. Production code that needs precise sizing
/// should route through [`crate::render::render_node`] anyway.
impl View for ViewNode {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        let r = crate::layout::Rect::from_layout_rect(*rect);
        crate::render::render_node(self, r, ctx);
    }

    fn body(&self) -> ViewNode {
        self.clone_node()
    }
}

impl ViewNode {
    /// Deep-clone a [`ViewNode`] subtree. The enum holds non-`Clone` types
    /// (`dyn Layout`, `dyn Fn`), so we provide an explicit cloner instead
    /// of deriving `Clone`. Layouts are cloned via [`Layout::clone_box`];
    /// closures (canvas draws, tap handlers, stateful evaluators) keep
    /// their arena-allocated identity via `Rc::clone`.
    pub fn clone_node(&self) -> ViewNode {
        match self {
            ViewNode::Leaf(k) => ViewNode::Leaf(k.clone_leaf()),
            ViewNode::Container { layout, children } => ViewNode::Container {
                layout: layout.clone_box(),
                children: children.iter().map(|c| c.clone_node()).collect(),
            },
            ViewNode::Modified { modifier, child } => ViewNode::Modified {
                modifier: modifier.clone_modifier(),
                child: Box::new(child.clone_node()),
            },
            ViewNode::Stateful(f) => ViewNode::Stateful(Rc::clone(f)),
        }
    }
}

impl LeafKind {
    /// Structural clone of a leaf — text/image/divider/spacer copy by
    /// value, canvas closures share an `Rc`.
    pub fn clone_leaf(&self) -> LeafKind {
        match self {
            LeafKind::Text(t) => LeafKind::Text(t.clone()),
            LeafKind::Image(i) => LeafKind::Image(i.clone()),
            LeafKind::Spacer { min_length } => LeafKind::Spacer {
                min_length: *min_length,
            },
            LeafKind::Divider { thickness, color } => LeafKind::Divider {
                thickness: *thickness,
                color: *color,
            },
            LeafKind::Canvas(d) => LeafKind::Canvas(Rc::clone(d)),
        }
    }
}

impl Modifier {
    /// Structural clone of a modifier — closure-bearing variants keep
    /// their `Rc` identity.
    pub fn clone_modifier(&self) -> Modifier {
        match self {
            Modifier::Padding(e) => Modifier::Padding(*e),
            Modifier::Background(f) => Modifier::Background(f.clone()),
            Modifier::Foreground(c) => Modifier::Foreground(*c),
            Modifier::Opacity(v) => Modifier::Opacity(*v),
            Modifier::CornerRadius(v) => Modifier::CornerRadius(*v),
            Modifier::Frame { width, height } => Modifier::Frame {
                width: *width,
                height: *height,
            },
            Modifier::OnTap(f) => Modifier::OnTap(Rc::clone(f)),
            Modifier::AccessibilityLabel(s) => Modifier::AccessibilityLabel(s.clone()),
            Modifier::FrameAligned {
                width,
                height,
                alignment,
            } => Modifier::FrameAligned {
                width: *width,
                height: *height,
                alignment: *alignment,
            },
            Modifier::FrameRange {
                min_width,
                ideal_width,
                max_width,
                min_height,
                ideal_height,
                max_height,
                alignment,
            } => Modifier::FrameRange {
                min_width: *min_width,
                ideal_width: *ideal_width,
                max_width: *max_width,
                min_height: *min_height,
                ideal_height: *ideal_height,
                max_height: *max_height,
                alignment: *alignment,
            },
            Modifier::Border { width, color } => Modifier::Border {
                width: *width,
                color: *color,
            },
            Modifier::Shadow {
                color,
                radius,
                offset_x,
                offset_y,
            } => Modifier::Shadow {
                color: *color,
                radius: *radius,
                offset_x: *offset_x,
                offset_y: *offset_y,
            },
            Modifier::InnerShadow {
                color,
                radius,
                offset_x,
                offset_y,
            } => Modifier::InnerShadow {
                color: *color,
                radius: *radius,
                offset_x: *offset_x,
                offset_y: *offset_y,
            },
            Modifier::Shadows(stack) => Modifier::Shadows(stack.clone()),
            Modifier::BackdropFilter(f) => Modifier::BackdropFilter(*f),
            Modifier::Rotation(d) => Modifier::Rotation(*d),
            Modifier::Scale { x, y, anchor } => Modifier::Scale {
                x: *x,
                y: *y,
                anchor: *anchor,
            },
            Modifier::Offset { x, y } => Modifier::Offset { x: *x, y: *y },
            Modifier::Position { x, y } => Modifier::Position { x: *x, y: *y },
            Modifier::Clip(s) => Modifier::Clip(s.clone()),
            Modifier::Mask(s) => Modifier::Mask(s.clone()),
            Modifier::OnHover(f) => Modifier::OnHover(Rc::clone(f)),
            Modifier::OnLongPress {
                duration_ms,
                action,
            } => Modifier::OnLongPress {
                duration_ms: *duration_ms,
                action: Rc::clone(action),
            },
            Modifier::OnDrag(f) => Modifier::OnDrag(Rc::clone(f)),
            Modifier::OnAppear(f) => Modifier::OnAppear(Rc::clone(f)),
            Modifier::OnDisappear(f) => Modifier::OnDisappear(Rc::clone(f)),
            Modifier::AccessibilityRole(r) => Modifier::AccessibilityRole(*r),
            Modifier::AccessibilityHint(s) => Modifier::AccessibilityHint(s.clone()),
            Modifier::AccessibilityValue(s) => Modifier::AccessibilityValue(s.clone()),
            Modifier::AccessibilityHidden(b) => Modifier::AccessibilityHidden(*b),
            Modifier::TabIndex(i) => Modifier::TabIndex(*i),
            Modifier::AllowsHitTesting(b) => Modifier::AllowsHitTesting(*b),
            Modifier::Tooltip(s) => Modifier::Tooltip(s.clone()),
            Modifier::ZIndex(z) => Modifier::ZIndex(*z),
            Modifier::Animation(spec) => Modifier::Animation(*spec),
            Modifier::Transition(t) => Modifier::Transition(t.clone()),
            Modifier::Tag(s) => Modifier::Tag(s.clone()),
            Modifier::Hidden(b) => Modifier::Hidden(*b),
        }
    }
}

/// Imperative-draw callback used by [`LeafKind::Canvas`]. Receives a mutable
/// render context plus the leaf's final size in the parent's coordinate
/// space.
pub type CanvasDraw = Rc<dyn Fn(&mut RenderCtx, Size)>;

/// Drawable leaf variants.
pub enum LeafKind {
    /// Plain text drawn with the leaf's font / colour.
    Text(TextLeaf),
    /// External image referenced by URL.
    Image(ImageLeaf),
    /// Empty space along the layout axis.
    Spacer { min_length: f32 },
    /// Coloured horizontal/vertical line.
    Divider { thickness: f32, color: Color },
    /// Imperative draw callback. The closure receives the render context and
    /// the leaf's resolved size.
    Canvas(CanvasDraw),
}

impl fmt::Debug for LeafKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeafKind::Text(t) => f.debug_tuple("Text").field(t).finish(),
            LeafKind::Image(i) => f.debug_tuple("Image").field(i).finish(),
            LeafKind::Spacer { min_length } => f
                .debug_struct("Spacer")
                .field("min_length", min_length)
                .finish(),
            LeafKind::Divider { thickness, color } => f
                .debug_struct("Divider")
                .field("thickness", thickness)
                .field("color", color)
                .finish(),
            LeafKind::Canvas(_) => f.write_str("Canvas(..)"),
        }
    }
}

/// Plain text leaf. The render walker uses these fields to position and
/// style the drawn glyphs.
#[derive(Debug, Clone)]
pub struct TextLeaf {
    pub content: String,
    pub font_size: f32,
    pub color: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

impl TextLeaf {
    /// Construct a text leaf with default font_size=14, white text, no decorations.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            font_size: 14.0,
            color: Color::WHITE,
            bold: false,
            italic: false,
            underline: false,
        }
    }
}

/// Image leaf — resolves `url` against the host's resource loader.
#[derive(Debug, Clone)]
pub struct ImageLeaf {
    pub url: String,
    pub fit: ContentFit,
}

impl ImageLeaf {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            fit: ContentFit::Contain,
        }
    }
}

/// How an [`ImageLeaf`] (or any sized media leaf) should fit its bounds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ContentFit {
    Fill,
    #[default]
    Contain,
    Cover,
    None,
    ScaleDown,
}

/// Single drop shadow — colour, blur σ, and offset. Multiple shadows can be
/// stacked behind a child via [`Modifier::Shadows`] for layered "Linear /
/// Vercel"-style depth (a soft 16 px radius at `dy=8` plus a tight 1 px
/// radius at `dy=1` reads as a real card, where any single shadow looks
/// flat).
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Shadow {
    pub color: Color,
    pub radius: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Shadow {
    /// Construct a shadow with the given colour, blur σ, and offsets.
    pub fn new(color: Color, radius: f32, dx: f32, dy: f32) -> Self {
        Self {
            color,
            radius,
            dx,
            dy,
        }
    }

    /// Standard "elevated" 1-px hairline shadow (subtle, near-the-surface).
    /// Pair with [`Shadow::soft`] for a Linear-style two-shadow stack.
    pub fn hairline(alpha: f32) -> Self {
        Self {
            color: Color::rgba(0.0, 0.0, 0.0, alpha),
            radius: 1.0,
            dx: 0.0,
            dy: 1.0,
        }
    }

    /// Wider, softer shadow used as the second layer in a stacked "elevated"
    /// material.
    pub fn soft(alpha: f32) -> Self {
        Self {
            color: Color::rgba(0.0, 0.0, 0.0, alpha),
            radius: 16.0,
            dx: 0.0,
            dy: 8.0,
        }
    }
}

/// Bundle of post-processing knobs that make up a `backdrop-filter`.
///
/// The defaults are an identity transform: no blur, no colour adjustment.
/// [`BackdropFilter::glass`] returns the Apple-style "Liquid Glass" preset
/// that pairs a 20 px blur with a small saturate / brightness boost.
///
/// Mirrors [`nova_gpu::effects::BackdropFilter`] — Photon owns the
/// declarative form and emits it as a [`DrawCommand::BackdropFilter`]; the
/// GPU compositor lifts it onto its own type before running the blur
/// pipeline.
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackdropFilter {
    pub blur_radius: f32,
    pub saturate: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub grayscale: f32,
    pub hue_rotate: f32,
    pub invert: f32,
    pub sepia: f32,
}

impl Default for BackdropFilter {
    fn default() -> Self {
        Self {
            blur_radius: 0.0,
            saturate: 1.0,
            brightness: 1.0,
            contrast: 1.0,
            grayscale: 0.0,
            hue_rotate: 0.0,
            invert: 0.0,
            sepia: 0.0,
        }
    }
}

impl BackdropFilter {
    /// Construct a pure-blur backdrop — identity for every other channel.
    pub fn blur(radius: f32) -> Self {
        Self {
            blur_radius: radius,
            ..Self::default()
        }
    }

    /// "Liquid Glass" preset: 20 px blur, slight saturation + brightness
    /// boost. Mirrors the iOS `UIBlurEffect.systemMaterial` look that
    /// Apple's UIKit ships and that Arc / macOS Big Sur popularised.
    pub fn glass() -> Self {
        Self {
            blur_radius: 20.0,
            saturate: 1.8,
            brightness: 1.05,
            contrast: 1.0,
            grayscale: 0.0,
            hue_rotate: 0.0,
            invert: 0.0,
            sepia: 0.0,
        }
    }

    /// True if every channel is at identity (renderer can elide the pass).
    pub fn is_identity(&self) -> bool {
        self.blur_radius <= 0.0
            && (self.saturate - 1.0).abs() <= f32::EPSILON
            && (self.brightness - 1.0).abs() <= f32::EPSILON
            && (self.contrast - 1.0).abs() <= f32::EPSILON
            && self.grayscale.abs() <= f32::EPSILON
            && self.hue_rotate.abs() <= f32::EPSILON
            && self.invert.abs() <= f32::EPSILON
            && self.sepia.abs() <= f32::EPSILON
    }
}

/// Background fill — solid colour or gradient. The Phase 3 renderer expands
/// solid colours into a single `FillRect`; gradients lower to a series of
/// stops drawn as horizontal slices (linear) or concentric rings (radial).
/// Future phases can swap in a real GPU shader without changing the
/// `Modifier` API.
#[derive(Clone, Debug)]
pub enum BackgroundFill {
    Color(Color),
    LinearGradient {
        angle_deg: f32,
        stops: Vec<(f32, Color)>,
    },
    RadialGradient {
        center: Alignment2D,
        stops: Vec<(f32, Color)>,
    },
}

impl From<Color> for BackgroundFill {
    fn from(c: Color) -> Self {
        BackgroundFill::Color(c)
    }
}

/// Path-clipping shape used by [`Modifier::Clip`] and [`Modifier::Mask`].
#[derive(Clone, Debug)]
pub enum ClipShape {
    Rect,
    RoundedRect { radius: f32 },
    Circle,
    Capsule,
    Ellipse,
    Path(Vec<PathCommand>),
}

/// Single primitive in a custom clip path.
#[derive(Clone, Debug, PartialEq)]
pub enum PathCommand {
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

// `AnimationCurve`, `TransitionKind`, and `Edge` live in
// [`crate::animation`] as of Phase 4. Re-exported below so existing
// `view::AnimationCurve` / `view::TransitionKind` / `view::Edge` paths keep
// resolving.
pub use crate::animation::{AnimationCurve, Edge, TransitionKind};

/// Accessibility role exposed via [`Modifier::AccessibilityRole`]. Mirrors
/// the union of ARIA / SwiftUI roles the embedder can surface to assistive
/// tech.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessibilityRole {
    Button,
    Link,
    Heading(u8),
    Image,
    List,
    ListItem,
    Header,
    Footer,
    Navigation,
    Main,
    Region,
    Search,
    Form,
    Dialog,
    Alert,
    Status,
    Tab,
    TabPanel,
    Menu,
    MenuItem,
    Checkbox,
    Radio,
    Switch,
    ProgressBar,
    Slider,
    ComboBox,
    TextBox,
    Tree,
    TreeItem,
    Grid,
    Cell,
    RowHeader,
    ColumnHeader,
    Article,
    Banner,
    Complementary,
    ContentInfo,
    Generic,
}

/// Subtree decorators — applied by walking outward from the leaf.
pub enum Modifier {
    /// Inset the child by the given edges.
    Padding(EdgeInsets),
    /// Fill the child's bounds with the given fill (solid colour or gradient)
    /// before drawing it.
    Background(BackgroundFill),
    /// Override the child's tint colour (text leaves pick this up).
    Foreground(Color),
    /// Apply a global alpha multiplier (clamped 0..=1).
    Opacity(f32),
    /// Round the child's corners by `radius` pixels (advisory — current
    /// renderer falls back to square corners).
    CornerRadius(f32),
    /// Force fixed width / height. `None` means inherit from the proposal.
    Frame {
        width: Option<f32>,
        height: Option<f32>,
    },
    /// Tap callback. Forwarded to the host's hit-test pipeline by the
    /// embedder.
    OnTap(Rc<dyn Fn()>),
    /// Accessibility label exposed via the `nova_core::accessibility` tree.
    AccessibilityLabel(String),

    /// Frame with explicit alignment (full SwiftUI `.frame`).
    FrameAligned {
        width: Option<f32>,
        height: Option<f32>,
        alignment: Alignment2D,
    },
    /// Min/max bounds with optional ideal sizes. Phase 3 resolves these
    /// against the parent's proposal greedily — full priority-based fitting
    /// lands in Phase 4.
    FrameRange {
        min_width: Option<f32>,
        ideal_width: Option<f32>,
        max_width: Option<f32>,
        min_height: Option<f32>,
        ideal_height: Option<f32>,
        max_height: Option<f32>,
        alignment: Alignment2D,
    },
    /// Border with explicit width and stroke colour, drawn as the outermost
    /// command of the wrapped subtree.
    Border { width: f32, color: Color },
    /// Drop shadow drawn behind the child.
    Shadow {
        color: Color,
        radius: f32,
        offset_x: f32,
        offset_y: f32,
    },
    /// Inner shadow drawn inside the bounds, on top of the child.
    InnerShadow {
        color: Color,
        radius: f32,
        offset_x: f32,
        offset_y: f32,
    },
    /// Stack of drop shadows drawn back-to-front behind the child. This is
    /// the modern "elevation" approach — Linear / Vercel / Notion buttons
    /// layer 2-3 shadows of differing radii to read as real cards. The first
    /// element draws furthest back, the last closest to the child.
    Shadows(Vec<Shadow>),
    /// Backdrop filter applied to the destination behind the child's bounds.
    /// CSS `backdrop-filter` analogue — combines a Gaussian blur with the
    /// saturate / brightness / contrast / grayscale / hue / invert / sepia
    /// chain. The renderer emits a [`crate::render::DrawCommand::BackdropFilter`]
    /// the GPU bridge consumes; if no GPU bridge is in scope the modifier is
    /// a no-op (the child still renders).
    BackdropFilter(BackdropFilter),
    /// Affine rotation in degrees, around the child's centre.
    Rotation(f32),
    /// Scale factor — uniform when `x == y`. Anchor selects the fixed point.
    Scale { x: f32, y: f32, anchor: Alignment2D },
    /// Translation applied to the child's bounds.
    Offset { x: f32, y: f32 },
    /// Position absolutely — overrides the parent's layout placement.
    Position { x: f32, y: f32 },
    /// Clip the wrapped subtree to a [`ClipShape`].
    Clip(ClipShape),
    /// Mask the wrapped subtree with a [`ClipShape`] — currently identical to
    /// `Clip` in the immediate-mode renderer; full alpha-mask support is a
    /// Phase 4 concern.
    Mask(ClipShape),
    /// Hover handler — `bool` indicates entering (`true`) or leaving (`false`).
    OnHover(Rc<dyn Fn(bool)>),
    /// Long-press handler — fires after `duration_ms` of held input.
    OnLongPress {
        duration_ms: u32,
        action: Rc<dyn Fn()>,
    },
    /// Drag gesture handler — receives `(dx, dy)` in local coordinates.
    OnDrag(Rc<dyn Fn(f32, f32)>),
    /// Per-render `on_appear` callback (fires when first attached to the
    /// render tree).
    OnAppear(Rc<dyn Fn()>),
    /// Per-render `on_disappear` callback (fires when detached).
    OnDisappear(Rc<dyn Fn()>),
    /// Accessibility role used by assistive tech.
    AccessibilityRole(AccessibilityRole),
    /// Accessibility hint (long-form description).
    AccessibilityHint(String),
    /// Accessibility value (e.g. slider current).
    AccessibilityValue(String),
    /// Exclude from the accessibility tree.
    AccessibilityHidden(bool),
    /// Tab order index.
    TabIndex(i32),
    /// Disable hit-testing on the wrapped subtree (purely decorative).
    AllowsHitTesting(bool),
    /// Tooltip text shown on hover.
    Tooltip(String),
    /// Z-index reordering within a parent. Phase 3 records the value but
    /// keeps the document order; Phase 4 will reorder commands.
    ZIndex(f32),
    /// Animation envelope — value-keyed transitions. Phase 4 wires the
    /// embedded [`crate::animation::AnimationSpec`] into the
    /// [`crate::animation::AnimationDriver`] so values across the wrapped
    /// subtree interpolate over `spec.duration_ms` along `spec.curve`.
    Animation(crate::animation::AnimationSpec),
    /// View enter/exit transition. Phase 3 records the kind; Phase 4 will
    /// animate the surrounding edges.
    Transition(TransitionKind),
    /// Tag for view identity / nav-stack matching.
    Tag(String),
    /// Visibility toggle that does not collapse layout. (Distinct from
    /// `if v { vn } else { Spacer }`, which removes the subtree entirely.)
    Hidden(bool),
}

impl fmt::Debug for Modifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modifier::Padding(e) => f.debug_tuple("Padding").field(e).finish(),
            Modifier::Background(fill) => f.debug_tuple("Background").field(fill).finish(),
            Modifier::Foreground(c) => f.debug_tuple("Foreground").field(c).finish(),
            Modifier::Opacity(v) => f.debug_tuple("Opacity").field(v).finish(),
            Modifier::CornerRadius(v) => f.debug_tuple("CornerRadius").field(v).finish(),
            Modifier::Frame { width, height } => f
                .debug_struct("Frame")
                .field("width", width)
                .field("height", height)
                .finish(),
            Modifier::OnTap(_) => f.write_str("OnTap(..)"),
            Modifier::AccessibilityLabel(l) => {
                f.debug_tuple("AccessibilityLabel").field(l).finish()
            }
            Modifier::FrameAligned {
                width,
                height,
                alignment,
            } => f
                .debug_struct("FrameAligned")
                .field("width", width)
                .field("height", height)
                .field("alignment", alignment)
                .finish(),
            Modifier::FrameRange {
                min_width,
                ideal_width,
                max_width,
                min_height,
                ideal_height,
                max_height,
                alignment,
            } => f
                .debug_struct("FrameRange")
                .field("min_width", min_width)
                .field("ideal_width", ideal_width)
                .field("max_width", max_width)
                .field("min_height", min_height)
                .field("ideal_height", ideal_height)
                .field("max_height", max_height)
                .field("alignment", alignment)
                .finish(),
            Modifier::Border { width, color } => f
                .debug_struct("Border")
                .field("width", width)
                .field("color", color)
                .finish(),
            Modifier::Shadow {
                color,
                radius,
                offset_x,
                offset_y,
            } => f
                .debug_struct("Shadow")
                .field("color", color)
                .field("radius", radius)
                .field("offset_x", offset_x)
                .field("offset_y", offset_y)
                .finish(),
            Modifier::InnerShadow {
                color,
                radius,
                offset_x,
                offset_y,
            } => f
                .debug_struct("InnerShadow")
                .field("color", color)
                .field("radius", radius)
                .field("offset_x", offset_x)
                .field("offset_y", offset_y)
                .finish(),
            Modifier::Shadows(stack) => f
                .debug_struct("Shadows")
                .field("count", &stack.len())
                .finish(),
            Modifier::BackdropFilter(b) => f.debug_tuple("BackdropFilter").field(b).finish(),
            Modifier::Rotation(d) => f.debug_tuple("Rotation").field(d).finish(),
            Modifier::Scale { x, y, anchor } => f
                .debug_struct("Scale")
                .field("x", x)
                .field("y", y)
                .field("anchor", anchor)
                .finish(),
            Modifier::Offset { x, y } => f
                .debug_struct("Offset")
                .field("x", x)
                .field("y", y)
                .finish(),
            Modifier::Position { x, y } => f
                .debug_struct("Position")
                .field("x", x)
                .field("y", y)
                .finish(),
            Modifier::Clip(s) => f.debug_tuple("Clip").field(s).finish(),
            Modifier::Mask(s) => f.debug_tuple("Mask").field(s).finish(),
            Modifier::OnHover(_) => f.write_str("OnHover(..)"),
            Modifier::OnLongPress { duration_ms, .. } => f
                .debug_struct("OnLongPress")
                .field("duration_ms", duration_ms)
                .finish_non_exhaustive(),
            Modifier::OnDrag(_) => f.write_str("OnDrag(..)"),
            Modifier::OnAppear(_) => f.write_str("OnAppear(..)"),
            Modifier::OnDisappear(_) => f.write_str("OnDisappear(..)"),
            Modifier::AccessibilityRole(r) => f.debug_tuple("AccessibilityRole").field(r).finish(),
            Modifier::AccessibilityHint(s) => f.debug_tuple("AccessibilityHint").field(s).finish(),
            Modifier::AccessibilityValue(s) => {
                f.debug_tuple("AccessibilityValue").field(s).finish()
            }
            Modifier::AccessibilityHidden(b) => {
                f.debug_tuple("AccessibilityHidden").field(b).finish()
            }
            Modifier::TabIndex(i) => f.debug_tuple("TabIndex").field(i).finish(),
            Modifier::AllowsHitTesting(b) => f.debug_tuple("AllowsHitTesting").field(b).finish(),
            Modifier::Tooltip(s) => f.debug_tuple("Tooltip").field(s).finish(),
            Modifier::ZIndex(z) => f.debug_tuple("ZIndex").field(z).finish(),
            Modifier::Animation(spec) => f.debug_tuple("Animation").field(spec).finish(),
            Modifier::Transition(t) => f.debug_tuple("Transition").field(t).finish(),
            Modifier::Tag(s) => f.debug_tuple("Tag").field(s).finish(),
            Modifier::Hidden(b) => f.debug_tuple("Hidden").field(b).finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{HStack as LHStack, VStack as LVStack};

    struct Dummy;
    impl View for Dummy {
        fn layout(&self, _c: &Constraint) -> Size {
            Size::new(10.0, 10.0)
        }
        fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
            ctx.fill_rect(rect, Color::WHITE);
        }
    }

    #[test]
    fn boxed_view_works() {
        let v = boxed(Dummy);
        let s = v.layout(&Constraint::loose(100.0, 100.0));
        assert_eq!(s.width, 10.0);
    }

    #[test]
    fn render_root_emits_command() {
        let v = Dummy;
        let mut ctx = RenderCtx::new();
        v.render_root(&mut ctx);
        assert_eq!(ctx.commands.len(), 1);
    }

    #[test]
    fn style_default_has_zero_padding() {
        let s = Style::default();
        assert_eq!(s.padding_x, 0.0);
        assert_eq!(s.padding_y, 0.0);
    }

    // -------------------------------------------------------------------------
    // Declarative tree
    // -------------------------------------------------------------------------

    #[test]
    fn text_leaf_constructor_defaults() {
        let t = TextLeaf::new("hello");
        assert_eq!(t.content, "hello");
        assert_eq!(t.font_size, 14.0);
        assert!(!t.bold);
    }

    #[test]
    fn image_leaf_default_fit_is_contain() {
        let i = ImageLeaf::new("foo.png");
        assert_eq!(i.fit, ContentFit::Contain);
    }

    #[test]
    fn spacer_leaf_carries_min_length() {
        let n = ViewNode::Leaf(LeafKind::Spacer { min_length: 8.0 });
        match n {
            ViewNode::Leaf(LeafKind::Spacer { min_length }) => assert_eq!(min_length, 8.0),
            _ => panic!("expected spacer"),
        }
    }

    #[test]
    fn divider_leaf_carries_thickness() {
        let n = ViewNode::Leaf(LeafKind::Divider {
            thickness: 2.0,
            color: Color::WHITE,
        });
        match n {
            ViewNode::Leaf(LeafKind::Divider { thickness, .. }) => assert_eq!(thickness, 2.0),
            _ => panic!("expected divider"),
        }
    }

    #[test]
    fn container_holds_layout_and_children() {
        let n = ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![
                ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x"))),
            ],
        };
        if let ViewNode::Container { children, .. } = n {
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected container");
        }
    }

    #[test]
    fn nested_containers_compose() {
        let inner = ViewNode::Container {
            layout: Box::new(LVStack::new()),
            children: vec![ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a")))],
        };
        let outer = ViewNode::Container {
            layout: Box::new(LHStack::new()),
            children: vec![inner, ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b")))],
        };
        match outer {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 2),
            _ => panic!("expected container"),
        }
    }

    #[test]
    fn modifier_padding_wraps_child() {
        let n = ViewNode::Modified {
            modifier: Modifier::Padding(EdgeInsets::all(4.0)),
            child: Box::new(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x")))),
        };
        match n {
            ViewNode::Modified {
                modifier: Modifier::Padding(e),
                ..
            } => {
                assert_eq!(e.top, 4.0);
            }
            _ => panic!("expected padding modifier"),
        }
    }

    #[test]
    fn modifier_frame_optional_dims() {
        let n = ViewNode::Modified {
            modifier: Modifier::Frame {
                width: Some(100.0),
                height: None,
            },
            child: Box::new(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })),
        };
        match n {
            ViewNode::Modified {
                modifier: Modifier::Frame { width, height },
                ..
            } => {
                assert_eq!(width, Some(100.0));
                assert!(height.is_none());
            }
            _ => panic!("expected frame modifier"),
        }
    }

    #[test]
    fn stateful_node_recomputes_each_call() {
        use std::cell::RefCell;
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let node = ViewNode::Stateful(Rc::new(move || {
            *c.borrow_mut() += 1;
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("tick")))
        }));
        if let ViewNode::Stateful(f) = &node {
            let _ = f();
            let _ = f();
        }
        assert_eq!(*counter.borrow(), 2);
    }

    #[test]
    fn dummy_default_body_returns_zero_spacer() {
        let d = Dummy;
        match d.body() {
            ViewNode::Leaf(LeafKind::Spacer { min_length }) => assert_eq!(min_length, 0.0),
            _ => panic!("expected default zero spacer"),
        }
    }

    #[test]
    fn modifier_debug_renders_safely() {
        // Ensures the manual Debug impl handles every variant.
        let cases = [
            format!("{:?}", Modifier::Padding(EdgeInsets::all(1.0))),
            format!(
                "{:?}",
                Modifier::Background(BackgroundFill::Color(Color::BLACK))
            ),
            format!("{:?}", Modifier::Foreground(Color::WHITE)),
            format!("{:?}", Modifier::Opacity(0.5)),
            format!("{:?}", Modifier::CornerRadius(4.0)),
            format!(
                "{:?}",
                Modifier::Frame {
                    width: None,
                    height: None
                }
            ),
            format!("{:?}", Modifier::OnTap(Rc::new(|| {}))),
            format!("{:?}", Modifier::AccessibilityLabel("hi".into())),
        ];
        assert!(cases.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn content_fit_default_is_contain() {
        assert_eq!(ContentFit::default(), ContentFit::Contain);
    }
}
