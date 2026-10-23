//! Layout primitives — rects, sizes, and parent-supplied constraints.
//!
//! This module contains two coexisting layout APIs:
//!
//! * **Legacy widget API** — [`Constraint`], [`LayoutRect`], [`Size`]. Drives
//!   the existing builder widgets in [`crate::widgets`] which call
//!   `widget.layout(&Constraint) -> Size` and
//!   `widget.render(&LayoutRect, &mut RenderCtx)`.
//!
//! * **SwiftUI-style two-pass protocol** — [`ProposedSize`], [`Point`],
//!   [`Rect`], [`EdgeInsets`], [`HAlignment`], [`VAlignment`],
//!   [`Alignment2D`], the [`Layout`] trait, and the four built-in layout
//!   strategies [`HStack`], [`VStack`], [`ZStack`], [`Frame`]. This is what
//!   the new declarative view tree (see [`crate::view::ViewNode`]) routes
//!   through.

use serde::{Deserialize, Serialize};

// =============================================================================
// Legacy widget primitives (used by the fluent builder in widgets.rs)
// =============================================================================

/// A 2D rectangle in Photon's local coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct LayoutRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LayoutRect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

/// 2D size returned by widget `layout` calls and by the new [`Layout`]
/// protocol's `size_that_fits`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Size {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

/// Min/max constraint range used by the legacy layout pass.
///
/// `tight(w, h)` and `loose(w, h)` are the two common helpers; they roughly
/// correspond to "fixed-size" and "fits-within" parents respectively.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Constraint {
    pub min: f32,
    pub max: f32,
    pub max_width: f32,
    pub max_height: f32,
}

impl Constraint {
    pub fn tight(width: f32, height: f32) -> Self {
        Self {
            min: 0.0,
            max: width.max(height),
            max_width: width,
            max_height: height,
        }
    }

    pub fn loose(max_width: f32, max_height: f32) -> Self {
        Self {
            min: 0.0,
            max: max_width.max(max_height),
            max_width,
            max_height,
        }
    }
}

// =============================================================================
// Two-pass layout protocol (SwiftUI-style)
// =============================================================================

/// Optional dimensions a parent proposes to a child. `None` on either axis
/// means "you decide".
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ProposedSize {
    pub width: Option<f32>,
    pub height: Option<f32>,
}

impl ProposedSize {
    /// Both axes are `None` — the child has full freedom.
    pub const fn unspecified() -> Self {
        Self {
            width: None,
            height: None,
        }
    }

    /// Both axes are `+inf` — "as much as you want".
    pub const fn infinity() -> Self {
        Self {
            width: Some(f32::INFINITY),
            height: Some(f32::INFINITY),
        }
    }

    /// Both axes are zero — "as little as possible".
    pub const fn zero() -> Self {
        Self {
            width: Some(0.0),
            height: Some(0.0),
        }
    }

    /// Both axes have specific finite proposals.
    pub const fn fixed(w: f32, h: f32) -> Self {
        Self {
            width: Some(w),
            height: Some(h),
        }
    }

    /// Resolve the width axis, falling back to `default` when unspecified.
    pub fn width_or(&self, default: f32) -> f32 {
        self.width.unwrap_or(default)
    }

    /// Resolve the height axis, falling back to `default` when unspecified.
    pub fn height_or(&self, default: f32) -> f32 {
        self.height.unwrap_or(default)
    }
}

/// 2D point in the parent's coordinate space.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Rectangle expressed as origin + size (the new layout protocol's form).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            origin: Point { x, y },
            size: Size::new(w, h),
        }
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.origin.x
            && p.x < self.origin.x + self.size.width
            && p.y >= self.origin.y
            && p.y < self.origin.y + self.size.height
    }

    /// Convert to the legacy `LayoutRect` representation.
    pub fn to_layout_rect(&self) -> LayoutRect {
        LayoutRect::new(
            self.origin.x,
            self.origin.y,
            self.size.width,
            self.size.height,
        )
    }

    /// Lift a `LayoutRect` into the new `Rect` representation.
    pub fn from_layout_rect(r: LayoutRect) -> Self {
        Self::new(r.x, r.y, r.width, r.height)
    }
}

/// Symmetric edge insets — used by the `Padding` modifier.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EdgeInsets {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl EdgeInsets {
    /// Same value on every edge.
    pub const fn all(v: f32) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }

    /// Horizontal `x` (left/right) and vertical `y` (top/bottom).
    pub const fn xy(x: f32, y: f32) -> Self {
        Self {
            top: y,
            right: x,
            bottom: y,
            left: x,
        }
    }

    /// Total horizontal inset (left + right).
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// Total vertical inset (top + bottom).
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Horizontal alignment along the cross axis of a vertical stack and the main
/// axis of a horizontal stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum HAlignment {
    Leading,
    #[default]
    Center,
    Trailing,
}

/// Vertical alignment along the cross axis of a horizontal stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum VAlignment {
    Top,
    #[default]
    Center,
    Bottom,
    FirstTextBaseline,
    LastTextBaseline,
}

/// 2D alignment used by [`ZStack`] and [`Frame`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Alignment2D {
    pub h: HAlignment,
    pub v: VAlignment,
}

impl Alignment2D {
    pub const fn new(h: HAlignment, v: VAlignment) -> Self {
        Self { h, v }
    }
}

/// Subview measurements collected before `place_subviews`. Returned by each
/// child's `size_that_fits` plus a layout priority used by stacks for space
/// distribution.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Measured {
    pub min: Size,
    pub max: Size,
    pub ideal: Size,
    pub priority: f32,
}

impl Measured {
    /// Construct a measurement where min == ideal == max.
    pub fn fixed(size: Size) -> Self {
        Self {
            min: size,
            max: size,
            ideal: size,
            priority: 0.0,
        }
    }
}

/// Final placement returned by `place_subviews`. Mutated in-place; the caller
/// passes a buffer sized to the number of children.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Placement {
    pub origin: Point,
    pub size: Size,
}

impl Placement {
    pub const fn new(origin: Point, size: Size) -> Self {
        Self { origin, size }
    }
}

/// Two-pass layout contract. Implementations describe how to arrange a list
/// of pre-measured subviews within a proposal:
///
/// 1. `size_that_fits` returns the size this layout wants.
/// 2. `place_subviews` writes per-child placements into the supplied buffer.
pub trait Layout {
    /// Return the size this layout wants given the parent's proposal and the
    /// child measurements.
    fn size_that_fits(&self, proposal: ProposedSize, subviews: &[Measured]) -> Size;

    /// Assign final positions and sizes to each subview. `subviews.len()`
    /// equals the number of children laid out.
    fn place_subviews(&self, bounds: Rect, proposal: ProposedSize, subviews: &mut [Placement]);

    /// Produce a boxed clone of this layout. Required so [`crate::view::ViewNode`]
    /// trees that contain `Container { layout, .. }` can be deep-cloned by
    /// helpers that need to forward a child subtree (e.g. `ScrollView`,
    /// `LazyVStack`, `Sheet::body`). Built-in layouts implement this in
    /// terms of their existing `Copy + Clone` impls.
    fn clone_box(&self) -> Box<dyn Layout>;

    /// Upcast this layout to `&dyn Any`. Used by Phase 5's
    /// [`crate::nwp`] serialiser to recognise built-in layout types and
    /// emit their wire form. Built-in layouts return `self` directly;
    /// custom layouts may return `&()` to opt out (the serialiser falls
    /// back to a default HStack shape for unknown layouts).
    fn as_any(&self) -> &dyn std::any::Any;
}

// -----------------------------------------------------------------------------
// HStack
// -----------------------------------------------------------------------------

/// Horizontal stack — distributes children left-to-right with `spacing`
/// between siblings, aligned vertically per `alignment`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HStack {
    pub spacing: f32,
    pub alignment: VAlignment,
}

impl HStack {
    pub fn new() -> Self {
        Self {
            spacing: 0.0,
            alignment: VAlignment::Center,
        }
    }

    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }

    pub fn alignment(mut self, a: VAlignment) -> Self {
        self.alignment = a;
        self
    }
}

impl Default for HStack {
    fn default() -> Self {
        Self::new()
    }
}

impl Layout for HStack {
    fn clone_box(&self) -> Box<dyn Layout> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn size_that_fits(&self, _proposal: ProposedSize, subviews: &[Measured]) -> Size {
        if subviews.is_empty() {
            return Size::zero();
        }
        let total_spacing = self.spacing * (subviews.len() - 1) as f32;
        let mut width = total_spacing;
        let mut height = 0.0_f32;
        for m in subviews {
            width += m.ideal.width;
            height = height.max(m.ideal.height);
        }
        Size::new(width, height)
    }

    fn place_subviews(&self, bounds: Rect, proposal: ProposedSize, subviews: &mut [Placement]) {
        if subviews.is_empty() {
            return;
        }
        let n = subviews.len();
        let total_spacing = self.spacing * (n - 1) as f32;
        // Capture each child's intrinsic ideal width/height before we overwrite
        // their `Placement` records.
        let ideals: Vec<Size> = subviews.iter().map(|p| p.size).collect();
        let ideal_total: f32 = ideals.iter().map(|s| s.width).sum();
        let proposed = proposal.width_or(bounds.size.width);

        // Distribute extra (or shrink) by equal split — Phase 1's simple
        // strategy. Future phases plug in priority-based allocation here.
        let available = (proposed - total_spacing).max(0.0);
        let extra = available - ideal_total;
        let per_child_extra = if n > 0 { extra / n as f32 } else { 0.0 };

        let mut x = bounds.origin.x;
        for (i, slot) in subviews.iter_mut().enumerate() {
            let w = (ideals[i].width + per_child_extra).max(0.0);
            let h = ideals[i].height.min(bounds.size.height);
            let y = match self.alignment {
                VAlignment::Top | VAlignment::FirstTextBaseline => bounds.origin.y,
                VAlignment::Center => bounds.origin.y + (bounds.size.height - h) * 0.5,
                VAlignment::Bottom | VAlignment::LastTextBaseline => {
                    bounds.origin.y + bounds.size.height - h
                }
            };
            *slot = Placement::new(Point::new(x, y), Size::new(w, h));
            x += w + self.spacing;
        }
    }
}

// -----------------------------------------------------------------------------
// VStack
// -----------------------------------------------------------------------------

/// Vertical stack — distributes children top-to-bottom with `spacing`
/// between siblings, aligned horizontally per `alignment`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VStack {
    pub spacing: f32,
    pub alignment: HAlignment,
}

impl VStack {
    pub fn new() -> Self {
        Self {
            spacing: 0.0,
            alignment: HAlignment::Center,
        }
    }

    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }

    pub fn alignment(mut self, a: HAlignment) -> Self {
        self.alignment = a;
        self
    }
}

impl Default for VStack {
    fn default() -> Self {
        Self::new()
    }
}

impl Layout for VStack {
    fn clone_box(&self) -> Box<dyn Layout> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn size_that_fits(&self, _proposal: ProposedSize, subviews: &[Measured]) -> Size {
        if subviews.is_empty() {
            return Size::zero();
        }
        let total_spacing = self.spacing * (subviews.len() - 1) as f32;
        let mut width = 0.0_f32;
        let mut height = total_spacing;
        for m in subviews {
            width = width.max(m.ideal.width);
            height += m.ideal.height;
        }
        Size::new(width, height)
    }

    fn place_subviews(&self, bounds: Rect, proposal: ProposedSize, subviews: &mut [Placement]) {
        if subviews.is_empty() {
            return;
        }
        let n = subviews.len();
        let total_spacing = self.spacing * (n - 1) as f32;
        let ideals: Vec<Size> = subviews.iter().map(|p| p.size).collect();
        let ideal_total: f32 = ideals.iter().map(|s| s.height).sum();
        let proposed = proposal.height_or(bounds.size.height);

        let available = (proposed - total_spacing).max(0.0);
        let extra = available - ideal_total;
        let per_child_extra = if n > 0 { extra / n as f32 } else { 0.0 };

        let mut y = bounds.origin.y;
        for (i, slot) in subviews.iter_mut().enumerate() {
            let h = (ideals[i].height + per_child_extra).max(0.0);
            let w = ideals[i].width.min(bounds.size.width);
            let x = match self.alignment {
                HAlignment::Leading => bounds.origin.x,
                HAlignment::Center => bounds.origin.x + (bounds.size.width - w) * 0.5,
                HAlignment::Trailing => bounds.origin.x + bounds.size.width - w,
            };
            *slot = Placement::new(Point::new(x, y), Size::new(w, h));
            y += h + self.spacing;
        }
    }
}

// -----------------------------------------------------------------------------
// ZStack
// -----------------------------------------------------------------------------

/// Z-axis stack — overlays children at the same anchor.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ZStack {
    pub alignment: Alignment2D,
}

impl ZStack {
    pub fn new() -> Self {
        Self {
            alignment: Alignment2D::default(),
        }
    }

    pub fn alignment(mut self, a: Alignment2D) -> Self {
        self.alignment = a;
        self
    }
}

impl Layout for ZStack {
    fn clone_box(&self) -> Box<dyn Layout> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn size_that_fits(&self, _proposal: ProposedSize, subviews: &[Measured]) -> Size {
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        for m in subviews {
            width = width.max(m.ideal.width);
            height = height.max(m.ideal.height);
        }
        Size::new(width, height)
    }

    fn place_subviews(&self, bounds: Rect, _proposal: ProposedSize, subviews: &mut [Placement]) {
        for slot in subviews.iter_mut() {
            let s = slot.size;
            let x = match self.alignment.h {
                HAlignment::Leading => bounds.origin.x,
                HAlignment::Center => bounds.origin.x + (bounds.size.width - s.width) * 0.5,
                HAlignment::Trailing => bounds.origin.x + bounds.size.width - s.width,
            };
            let y = match self.alignment.v {
                VAlignment::Top | VAlignment::FirstTextBaseline => bounds.origin.y,
                VAlignment::Center => bounds.origin.y + (bounds.size.height - s.height) * 0.5,
                VAlignment::Bottom | VAlignment::LastTextBaseline => {
                    bounds.origin.y + bounds.size.height - s.height
                }
            };
            *slot = Placement::new(Point::new(x, y), s);
        }
    }
}

// -----------------------------------------------------------------------------
// Frame
// -----------------------------------------------------------------------------

/// Forces a fixed width and/or height, optionally aligning the (single)
/// subview within the resulting rect. Pass `None` on an axis to inherit
/// from the proposal.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Frame {
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub alignment: Alignment2D,
}

impl Frame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn width(mut self, w: f32) -> Self {
        self.width = Some(w);
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = Some(h);
        self
    }

    pub fn alignment(mut self, a: Alignment2D) -> Self {
        self.alignment = a;
        self
    }
}

impl Layout for Frame {
    fn clone_box(&self) -> Box<dyn Layout> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn size_that_fits(&self, proposal: ProposedSize, subviews: &[Measured]) -> Size {
        let child_ideal = subviews.first().map(|m| m.ideal).unwrap_or_else(Size::zero);
        let w = self.width.or(proposal.width).unwrap_or(child_ideal.width);
        let h = self
            .height
            .or(proposal.height)
            .unwrap_or(child_ideal.height);
        Size::new(w, h)
    }

    fn place_subviews(&self, bounds: Rect, _proposal: ProposedSize, subviews: &mut [Placement]) {
        for slot in subviews.iter_mut() {
            let s = slot.size;
            let x = match self.alignment.h {
                HAlignment::Leading => bounds.origin.x,
                HAlignment::Center => bounds.origin.x + (bounds.size.width - s.width) * 0.5,
                HAlignment::Trailing => bounds.origin.x + bounds.size.width - s.width,
            };
            let y = match self.alignment.v {
                VAlignment::Top | VAlignment::FirstTextBaseline => bounds.origin.y,
                VAlignment::Center => bounds.origin.y + (bounds.size.height - s.height) * 0.5,
                VAlignment::Bottom | VAlignment::LastTextBaseline => {
                    bounds.origin.y + bounds.size.height - s.height
                }
            };
            *slot = Placement::new(Point::new(x, y), s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Legacy primitives
    // -------------------------------------------------------------------------

    #[test]
    fn rect_contains_point_inside() {
        let r = LayoutRect::new(0.0, 0.0, 10.0, 10.0);
        assert!(r.contains(5.0, 5.0));
    }

    #[test]
    fn rect_contains_point_outside() {
        let r = LayoutRect::new(0.0, 0.0, 10.0, 10.0);
        assert!(!r.contains(15.0, 5.0));
    }

    #[test]
    fn constraint_tight_sets_max_width() {
        let c = Constraint::tight(800.0, 600.0);
        assert_eq!(c.max_width, 800.0);
        assert_eq!(c.max_height, 600.0);
    }

    #[test]
    fn constraint_loose_sets_max_width() {
        let c = Constraint::loose(1024.0, 768.0);
        assert_eq!(c.max_width, 1024.0);
    }

    #[test]
    fn size_zero_is_zero() {
        let s = Size::zero();
        assert_eq!(s.width, 0.0);
        assert_eq!(s.height, 0.0);
    }

    // -------------------------------------------------------------------------
    // ProposedSize
    // -------------------------------------------------------------------------

    #[test]
    fn proposed_unspecified_is_none_none() {
        let p = ProposedSize::unspecified();
        assert!(p.width.is_none() && p.height.is_none());
    }

    #[test]
    fn proposed_infinity_is_inf() {
        let p = ProposedSize::infinity();
        assert!(p.width.unwrap().is_infinite());
        assert!(p.height.unwrap().is_infinite());
    }

    #[test]
    fn proposed_zero_is_zero() {
        let p = ProposedSize::zero();
        assert_eq!(p.width, Some(0.0));
        assert_eq!(p.height, Some(0.0));
    }

    #[test]
    fn proposed_fixed_carries_value() {
        let p = ProposedSize::fixed(100.0, 200.0);
        assert_eq!(p.width_or(0.0), 100.0);
        assert_eq!(p.height_or(0.0), 200.0);
    }

    #[test]
    fn proposed_or_falls_back() {
        let p = ProposedSize::unspecified();
        assert_eq!(p.width_or(50.0), 50.0);
        assert_eq!(p.height_or(60.0), 60.0);
    }

    // -------------------------------------------------------------------------
    // Rect / Point / EdgeInsets
    // -------------------------------------------------------------------------

    #[test]
    fn new_rect_contains_point() {
        let r = Rect::new(0.0, 0.0, 50.0, 50.0);
        assert!(r.contains(Point::new(25.0, 25.0)));
        assert!(!r.contains(Point::new(50.0, 25.0)));
    }

    #[test]
    fn rect_round_trips_through_layout_rect() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        let lr = r.to_layout_rect();
        let r2 = Rect::from_layout_rect(lr);
        assert_eq!(r, r2);
    }

    #[test]
    fn edge_insets_all_uniform() {
        let e = EdgeInsets::all(8.0);
        assert_eq!(e.horizontal(), 16.0);
        assert_eq!(e.vertical(), 16.0);
    }

    #[test]
    fn edge_insets_xy_split() {
        let e = EdgeInsets::xy(4.0, 8.0);
        assert_eq!(e.left, 4.0);
        assert_eq!(e.right, 4.0);
        assert_eq!(e.top, 8.0);
        assert_eq!(e.bottom, 8.0);
    }

    // -------------------------------------------------------------------------
    // HStack
    // -------------------------------------------------------------------------

    #[test]
    fn hstack_size_that_fits_sums_widths() {
        let layout = HStack::new().spacing(4.0);
        let subs = vec![
            Measured::fixed(Size::new(10.0, 20.0)),
            Measured::fixed(Size::new(30.0, 25.0)),
        ];
        let s = layout.size_that_fits(ProposedSize::unspecified(), &subs);
        assert_eq!(s.width, 10.0 + 4.0 + 30.0);
        assert_eq!(s.height, 25.0);
    }

    #[test]
    fn hstack_place_subviews_lays_out_left_to_right() {
        let layout = HStack::new().spacing(2.0).alignment(VAlignment::Top);
        let mut places = vec![
            Placement::new(Point::new(0.0, 0.0), Size::new(10.0, 10.0)),
            Placement::new(Point::new(0.0, 0.0), Size::new(20.0, 20.0)),
        ];
        layout.place_subviews(
            Rect::new(0.0, 0.0, 100.0, 50.0),
            ProposedSize::fixed(100.0, 50.0),
            &mut places,
        );
        // Children get equal extra; check ordering and y alignment.
        assert!(places[0].origin.x < places[1].origin.x);
        assert_eq!(places[0].origin.y, 0.0);
        assert_eq!(places[1].origin.y, 0.0);
    }

    // -------------------------------------------------------------------------
    // VStack
    // -------------------------------------------------------------------------

    #[test]
    fn vstack_size_that_fits_sums_heights() {
        let layout = VStack::new().spacing(2.0);
        let subs = vec![
            Measured::fixed(Size::new(10.0, 20.0)),
            Measured::fixed(Size::new(30.0, 25.0)),
        ];
        let s = layout.size_that_fits(ProposedSize::unspecified(), &subs);
        assert_eq!(s.height, 20.0 + 2.0 + 25.0);
        assert_eq!(s.width, 30.0);
    }

    #[test]
    fn vstack_place_subviews_centers_horizontally_by_default() {
        let layout = VStack::new();
        let mut places = vec![Placement::new(Point::new(0.0, 0.0), Size::new(20.0, 20.0))];
        layout.place_subviews(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            ProposedSize::fixed(100.0, 100.0),
            &mut places,
        );
        // (100 - 20) / 2 == 40
        assert_eq!(places[0].origin.x, 40.0);
    }

    // -------------------------------------------------------------------------
    // ZStack
    // -------------------------------------------------------------------------

    #[test]
    fn zstack_size_takes_max_dimensions() {
        let layout = ZStack::new();
        let subs = vec![
            Measured::fixed(Size::new(50.0, 30.0)),
            Measured::fixed(Size::new(20.0, 80.0)),
        ];
        let s = layout.size_that_fits(ProposedSize::unspecified(), &subs);
        assert_eq!(s.width, 50.0);
        assert_eq!(s.height, 80.0);
    }

    #[test]
    fn zstack_centers_by_default() {
        let layout = ZStack::new();
        let mut places = vec![Placement::new(Point::new(0.0, 0.0), Size::new(10.0, 10.0))];
        layout.place_subviews(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            ProposedSize::unspecified(),
            &mut places,
        );
        assert_eq!(places[0].origin.x, 45.0);
        assert_eq!(places[0].origin.y, 45.0);
    }

    #[test]
    fn zstack_top_leading_pins_to_origin() {
        let layout =
            ZStack::new().alignment(Alignment2D::new(HAlignment::Leading, VAlignment::Top));
        let mut places = vec![Placement::new(Point::new(0.0, 0.0), Size::new(10.0, 10.0))];
        layout.place_subviews(
            Rect::new(5.0, 7.0, 100.0, 100.0),
            ProposedSize::unspecified(),
            &mut places,
        );
        assert_eq!(places[0].origin.x, 5.0);
        assert_eq!(places[0].origin.y, 7.0);
    }

    // -------------------------------------------------------------------------
    // Frame
    // -------------------------------------------------------------------------

    #[test]
    fn frame_overrides_dimensions() {
        let layout = Frame::new().width(200.0).height(80.0);
        let s = layout.size_that_fits(ProposedSize::fixed(50.0, 30.0), &[]);
        assert_eq!(s.width, 200.0);
        assert_eq!(s.height, 80.0);
    }

    #[test]
    fn frame_inherits_from_proposal_when_unset() {
        let layout = Frame::new();
        let s = layout.size_that_fits(ProposedSize::fixed(120.0, 90.0), &[]);
        assert_eq!(s.width, 120.0);
        assert_eq!(s.height, 90.0);
    }

    #[test]
    fn frame_centers_child_by_default() {
        let layout = Frame::new();
        let mut places = vec![Placement::new(Point::new(0.0, 0.0), Size::new(20.0, 20.0))];
        layout.place_subviews(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            ProposedSize::fixed(100.0, 100.0),
            &mut places,
        );
        assert_eq!(places[0].origin.x, 40.0);
        assert_eq!(places[0].origin.y, 40.0);
    }

    // -------------------------------------------------------------------------
    // Alignment defaults
    // -------------------------------------------------------------------------

    #[test]
    fn alignment_defaults_center() {
        let a = Alignment2D::default();
        assert_eq!(a.h, HAlignment::Center);
        assert_eq!(a.v, VAlignment::Center);
    }
}
