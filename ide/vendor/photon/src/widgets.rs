//! Built-in Photon widgets — `text`, `h_stack`, `v_stack`, `spacer`,
//! `divider`, `container`.
//!
//! Each widget exposes a fluent builder API. The set of modifiers is the
//! same across widgets to keep call-sites uniform — see [`Modifiers`] for
//! the full list.
//!
//! Every legacy widget also implements a `to_node` method that produces a
//! [`crate::view::ViewNode`] tree, letting new code that uses the
//! declarative render path interop with the builder API.

use crate::color::Color;
use crate::layout::{
    Constraint, EdgeInsets, HStack as LayoutHStack, LayoutRect, Size, VStack as LayoutVStack,
};
use crate::render::{DrawCommand, RenderCtx};
use crate::view::{BackgroundFill, LeafKind, Modifier, Style, TextLeaf, View, ViewNode};

// Re-export `boxed` so call-sites that `use photon::widgets::*` can
// invoke it without an extra `use photon::view::boxed;`.
pub use crate::view::boxed;

// ---------------------------------------------------------------------------
// Macro: shared builder methods every widget exposes.
// ---------------------------------------------------------------------------
macro_rules! impl_modifiers {
    ($t:ty) => {
        impl $t {
            pub fn background(mut self, c: Color) -> Self {
                self.style.background = Some(c);
                self
            }
            pub fn border(mut self, width: f32, color: Color) -> Self {
                self.style.border_width = width;
                self.style.border_color = Some(color);
                self
            }
            pub fn padding(mut self, p: f32) -> Self {
                self.style.padding_x = p;
                self.style.padding_y = p;
                self
            }
            pub fn padding_xy(mut self, x: f32, y: f32) -> Self {
                self.style.padding_x = x;
                self.style.padding_y = y;
                self
            }
            pub fn min_width(mut self, w: f32) -> Self {
                self.style.min_width = w;
                self
            }
            pub fn max_width(mut self, w: f32) -> Self {
                self.style.max_width = w;
                self
            }
            pub fn min_height(mut self, h: f32) -> Self {
                self.style.min_height = h;
                self
            }
            pub fn max_height(mut self, h: f32) -> Self {
                self.style.max_height = h;
                self
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Helpers used by every widget for chrome painting.
// ---------------------------------------------------------------------------

fn paint_chrome(rect: &LayoutRect, style: &Style, ctx: &mut RenderCtx) {
    if let Some(bg) = style.background {
        if !bg.is_transparent() {
            ctx.fill_rect(rect, bg);
        }
    }
    if style.border_width > 0.0 {
        if let Some(bc) = style.border_color {
            ctx.stroke_rect(rect, style.border_width, bc);
        }
    }
}

fn apply_size_constraints(mut size: Size, style: &Style) -> Size {
    if style.min_width > 0.0 && size.width < style.min_width {
        size.width = style.min_width;
    }
    if style.min_height > 0.0 && size.height < style.min_height {
        size.height = style.min_height;
    }
    if style.max_width > 0.0 && size.width > style.max_width {
        size.width = style.max_width;
    }
    if style.max_height > 0.0 && size.height > style.max_height {
        size.height = style.max_height;
    }
    size
}

/// Wrap `inner` in the modifiers implied by `style` (padding, background,
/// border, frame). Used by every widget's `to_node` impl so the legacy
/// builder produces a structurally identical declarative tree.
fn wrap_with_style(mut inner: ViewNode, style: &Style) -> ViewNode {
    if style.padding_x > 0.0 || style.padding_y > 0.0 {
        inner = ViewNode::Modified {
            modifier: Modifier::Padding(EdgeInsets::xy(style.padding_x, style.padding_y)),
            child: Box::new(inner),
        };
    }
    if let Some(bg) = style.background {
        inner = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(bg)),
            child: Box::new(inner),
        };
    }
    if style.min_width > 0.0 || style.min_height > 0.0 {
        inner = ViewNode::Modified {
            modifier: Modifier::Frame {
                width: if style.min_width > 0.0 {
                    Some(style.min_width)
                } else {
                    None
                },
                height: if style.min_height > 0.0 {
                    Some(style.min_height)
                } else {
                    None
                },
            },
            child: Box::new(inner),
        };
    }
    inner
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Text {
    pub content: String,
    pub font_size: f32,
    pub color: Color,
    pub bold: bool,
    pub style: Style,
}

impl_modifiers!(Text);

impl Text {
    pub fn font_size(mut self, s: f32) -> Self {
        self.font_size = s;
        self
    }
    pub fn color(mut self, c: Color) -> Self {
        self.color = c;
        self
    }
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Convert this widget into a declarative [`ViewNode`] tree.
    pub fn to_node(self) -> ViewNode {
        let inner = ViewNode::Leaf(LeafKind::Text(TextLeaf {
            content: self.content,
            font_size: self.font_size,
            color: self.color,
            bold: self.bold,
            italic: false,
            underline: false,
        }));
        wrap_with_style(inner, &self.style)
    }
}

pub fn text(s: impl Into<String>) -> Text {
    Text {
        content: s.into(),
        font_size: 14.0,
        color: Color::WHITE,
        bold: false,
        style: Style::default(),
    }
}

impl View for Text {
    fn layout(&self, _c: &Constraint) -> Size {
        // Approximate text width using a rough avg-advance of 0.55 × font_size.
        let intrinsic_w = self.content.chars().count() as f32 * self.font_size * 0.55
            + self.style.padding_x * 2.0;
        let intrinsic_h = self.font_size * 1.4 + self.style.padding_y * 2.0;
        apply_size_constraints(Size::new(intrinsic_w, intrinsic_h), &self.style)
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        let inner = LayoutRect::new(
            rect.x + self.style.padding_x,
            rect.y + self.style.padding_y,
            (rect.width - self.style.padding_x * 2.0).max(0.0),
            (rect.height - self.style.padding_y * 2.0).max(0.0),
        );
        ctx.fill_text(&inner, &self.content, self.font_size, self.color);
    }

    fn body(&self) -> ViewNode {
        self.clone().to_node()
    }
}

// ---------------------------------------------------------------------------
// Spacer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Spacer {
    pub style: Style,
}

impl_modifiers!(Spacer);

pub fn spacer() -> Spacer {
    Spacer::default()
}

impl Spacer {
    /// Convert this widget into a declarative [`ViewNode`] tree.
    pub fn to_node(self) -> ViewNode {
        let inner = ViewNode::Leaf(LeafKind::Spacer { min_length: 1.0 });
        wrap_with_style(inner, &self.style)
    }
}

impl View for Spacer {
    fn layout(&self, c: &Constraint) -> Size {
        // Spacer "absorbs" available space — but we're conservative and
        // return a tiny default so it doesn't blow out parent stacks.
        let w = if c.max_width.is_finite() {
            c.max_width.clamp(1.0, 8.0)
        } else {
            1.0
        };
        apply_size_constraints(Size::new(w, 1.0), &self.style)
    }
    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
    }
    fn body(&self) -> ViewNode {
        self.clone().to_node()
    }
}

// ---------------------------------------------------------------------------
// Divider — a 1-pixel coloured line by default.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Divider {
    pub color: Color,
    pub thickness: f32,
    pub style: Style,
}

impl_modifiers!(Divider);

impl Divider {
    pub fn color(mut self, c: Color) -> Self {
        self.color = c;
        self
    }
    pub fn thickness(mut self, t: f32) -> Self {
        self.thickness = t;
        self
    }

    /// Convert this widget into a declarative [`ViewNode`] tree.
    pub fn to_node(self) -> ViewNode {
        let inner = ViewNode::Leaf(LeafKind::Divider {
            thickness: self.thickness,
            color: self.color,
        });
        wrap_with_style(inner, &self.style)
    }
}

pub fn divider() -> Divider {
    Divider {
        color: Color::rgba(0.5, 0.5, 0.5, 1.0),
        thickness: 1.0,
        style: Style::default(),
    }
}

impl View for Divider {
    fn layout(&self, c: &Constraint) -> Size {
        let w = if c.max_width.is_finite() {
            c.max_width
        } else {
            0.0
        };
        apply_size_constraints(Size::new(w, self.thickness), &self.style)
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        ctx.fill_rect(rect, self.color);
    }

    fn body(&self) -> ViewNode {
        self.clone().to_node()
    }
}

// ---------------------------------------------------------------------------
// HStack / VStack
// ---------------------------------------------------------------------------

pub struct HStack {
    pub children: Vec<Box<dyn View>>,
    pub gap: f32,
    pub style: Style,
}

impl_modifiers!(HStack);

impl HStack {
    pub fn gap(mut self, g: f32) -> Self {
        self.gap = g;
        self
    }

    /// Convert this widget into a declarative [`ViewNode`] tree, mapping
    /// each child via its [`View::body`] impl.
    pub fn to_node(self) -> ViewNode {
        let children: Vec<ViewNode> = self.children.iter().map(|c| c.body()).collect();
        let inner = ViewNode::Container {
            layout: Box::new(LayoutHStack::new().spacing(self.gap)),
            children,
        };
        wrap_with_style(inner, &self.style)
    }
}

pub fn h_stack(children: Vec<Box<dyn View>>) -> HStack {
    HStack {
        children,
        gap: 0.0,
        style: Style::default(),
    }
}

impl View for HStack {
    fn layout(&self, c: &Constraint) -> Size {
        let child_c = Constraint::loose(
            (c.max_width - self.style.padding_x * 2.0).max(0.0),
            (c.max_height - self.style.padding_y * 2.0).max(0.0),
        );
        let mut total_w = self.style.padding_x * 2.0;
        let mut max_h = 0.0_f32;
        for (i, child) in self.children.iter().enumerate() {
            if i > 0 {
                total_w += self.gap;
            }
            let s = child.layout(&child_c);
            total_w += s.width;
            max_h = max_h.max(s.height);
        }
        let h = max_h + self.style.padding_y * 2.0;
        apply_size_constraints(Size::new(total_w, h), &self.style)
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        let inner_x = rect.x + self.style.padding_x;
        let inner_y = rect.y + self.style.padding_y;
        let inner_h = (rect.height - self.style.padding_y * 2.0).max(0.0);
        let inner_w = (rect.width - self.style.padding_x * 2.0).max(0.0);
        let child_c = Constraint::loose(inner_w, inner_h);
        let mut x = inner_x;
        for (i, child) in self.children.iter().enumerate() {
            if i > 0 {
                x += self.gap;
            }
            let s = child.layout(&child_c);
            let cr = LayoutRect::new(x, inner_y, s.width, inner_h.max(s.height));
            child.render(&cr, ctx);
            x += s.width;
        }
    }

    fn body(&self) -> ViewNode {
        let children: Vec<ViewNode> = self.children.iter().map(|c| c.body()).collect();
        let inner = ViewNode::Container {
            layout: Box::new(LayoutHStack::new().spacing(self.gap)),
            children,
        };
        wrap_with_style(inner, &self.style)
    }
}

pub struct VStack {
    pub children: Vec<Box<dyn View>>,
    pub gap: f32,
    pub style: Style,
}

impl_modifiers!(VStack);

impl VStack {
    pub fn gap(mut self, g: f32) -> Self {
        self.gap = g;
        self
    }

    /// Convert this widget into a declarative [`ViewNode`] tree, mapping
    /// each child via its [`View::body`] impl.
    pub fn to_node(self) -> ViewNode {
        let children: Vec<ViewNode> = self.children.iter().map(|c| c.body()).collect();
        let inner = ViewNode::Container {
            layout: Box::new(LayoutVStack::new().spacing(self.gap)),
            children,
        };
        wrap_with_style(inner, &self.style)
    }
}

pub fn v_stack(children: Vec<Box<dyn View>>) -> VStack {
    VStack {
        children,
        gap: 0.0,
        style: Style::default(),
    }
}

impl View for VStack {
    fn layout(&self, c: &Constraint) -> Size {
        let child_c = Constraint::loose(
            (c.max_width - self.style.padding_x * 2.0).max(0.0),
            (c.max_height - self.style.padding_y * 2.0).max(0.0),
        );
        let mut total_h = self.style.padding_y * 2.0;
        let mut max_w = 0.0_f32;
        for (i, child) in self.children.iter().enumerate() {
            if i > 0 {
                total_h += self.gap;
            }
            let s = child.layout(&child_c);
            total_h += s.height;
            max_w = max_w.max(s.width);
        }
        let w = max_w + self.style.padding_x * 2.0;
        apply_size_constraints(Size::new(w, total_h), &self.style)
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        let inner_x = rect.x + self.style.padding_x;
        let inner_y = rect.y + self.style.padding_y;
        let inner_w = (rect.width - self.style.padding_x * 2.0).max(0.0);
        let inner_h = (rect.height - self.style.padding_y * 2.0).max(0.0);
        let child_c = Constraint::loose(inner_w, inner_h);
        let mut y = inner_y;
        for (i, child) in self.children.iter().enumerate() {
            if i > 0 {
                y += self.gap;
            }
            let s = child.layout(&child_c);
            let cr = LayoutRect::new(inner_x, y, inner_w.max(s.width), s.height);
            child.render(&cr, ctx);
            y += s.height;
        }
    }

    fn body(&self) -> ViewNode {
        let children: Vec<ViewNode> = self.children.iter().map(|c| c.body()).collect();
        let inner = ViewNode::Container {
            layout: Box::new(LayoutVStack::new().spacing(self.gap)),
            children,
        };
        wrap_with_style(inner, &self.style)
    }
}

// ---------------------------------------------------------------------------
// Container — fixed-size wrapper.
// ---------------------------------------------------------------------------

pub struct Container {
    pub child: Box<dyn View>,
    pub width: f32,
    pub height: f32,
    pub bg: Option<Color>,
    pub style: Style,
}

impl_modifiers!(Container);

impl Container {
    pub fn width(mut self, w: f32) -> Self {
        self.width = w;
        self
    }
    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }
    pub fn bg(mut self, c: Color) -> Self {
        self.bg = Some(c);
        self
    }

    /// Convert this widget into a declarative [`ViewNode`] tree.
    pub fn to_node(self) -> ViewNode {
        let mut inner = self.child.body();
        if let Some(bg) = self.bg {
            inner = ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(bg)),
                child: Box::new(inner),
            };
        }
        if self.width > 0.0 || self.height > 0.0 {
            inner = ViewNode::Modified {
                modifier: Modifier::Frame {
                    width: if self.width > 0.0 {
                        Some(self.width)
                    } else {
                        None
                    },
                    height: if self.height > 0.0 {
                        Some(self.height)
                    } else {
                        None
                    },
                },
                child: Box::new(inner),
            };
        }
        wrap_with_style(inner, &self.style)
    }
}

pub fn container(child: Box<dyn View>) -> Container {
    Container {
        child,
        width: 0.0,
        height: 0.0,
        bg: None,
        style: Style::default(),
    }
}

impl View for Container {
    fn layout(&self, c: &Constraint) -> Size {
        let want_w = if self.width > 0.0 {
            self.width
        } else {
            c.max_width.min(800.0)
        };
        let want_h = if self.height > 0.0 {
            self.height
        } else {
            c.max_height.min(600.0)
        };
        apply_size_constraints(Size::new(want_w, want_h), &self.style)
    }

    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        if let Some(bg) = self.bg {
            ctx.fill_rect(rect, bg);
        }
        paint_chrome(rect, &self.style, ctx);
        // Clip to the container so the child can't overflow.
        ctx.push_clip(rect);
        self.child.render(rect, ctx);
        ctx.pop_clip();
    }

    fn body(&self) -> ViewNode {
        let mut inner = self.child.body();
        if let Some(bg) = self.bg {
            inner = ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(bg)),
                child: Box::new(inner),
            };
        }
        if self.width > 0.0 || self.height > 0.0 {
            inner = ViewNode::Modified {
                modifier: Modifier::Frame {
                    width: if self.width > 0.0 {
                        Some(self.width)
                    } else {
                        None
                    },
                    height: if self.height > 0.0 {
                        Some(self.height)
                    } else {
                        None
                    },
                },
                child: Box::new(inner),
            };
        }
        wrap_with_style(inner, &self.style)
    }
}

// ---------------------------------------------------------------------------
// Button — a placeholder for production interactive widgets. Currently
// behaves as a Container around a Text label.
// ---------------------------------------------------------------------------

pub struct Button {
    pub label: String,
    pub style: Style,
}

impl_modifiers!(Button);

impl Button {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            style: Style::default(),
        }
    }
}

impl View for Button {
    fn layout(&self, _c: &Constraint) -> Size {
        let w = self.label.chars().count() as f32 * 8.0 + 24.0;
        apply_size_constraints(Size::new(w, 28.0), &self.style)
    }
    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        ctx.fill_text(rect, &self.label, 14.0, Color::WHITE);
    }
    fn body(&self) -> ViewNode {
        let inner = ViewNode::Leaf(LeafKind::Text(TextLeaf::new(self.label.clone())));
        wrap_with_style(inner, &self.style)
    }
}

// ---------------------------------------------------------------------------
// Stack — alias used for compatibility (z-order stack of views).
// ---------------------------------------------------------------------------

pub struct Stack {
    pub children: Vec<Box<dyn View>>,
    pub style: Style,
}

impl_modifiers!(Stack);

impl Stack {
    pub fn new(children: Vec<Box<dyn View>>) -> Self {
        Self {
            children,
            style: Style::default(),
        }
    }
}

impl View for Stack {
    fn layout(&self, c: &Constraint) -> Size {
        let mut max = Size::default();
        for child in &self.children {
            let s = child.layout(c);
            if s.width > max.width {
                max.width = s.width;
            }
            if s.height > max.height {
                max.height = s.height;
            }
        }
        apply_size_constraints(max, &self.style)
    }
    fn render(&self, rect: &LayoutRect, ctx: &mut RenderCtx) {
        paint_chrome(rect, &self.style, ctx);
        for child in &self.children {
            child.render(rect, ctx);
        }
    }
    fn body(&self) -> ViewNode {
        let children: Vec<ViewNode> = self.children.iter().map(|c| c.body()).collect();
        let inner = ViewNode::Container {
            layout: Box::new(crate::layout::ZStack::new()),
            children,
        };
        wrap_with_style(inner, &self.style)
    }
}

// ---------------------------------------------------------------------------
// Canvas — imperative draw API kept for parity with the audit notes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Canvas {
    pub commands: Vec<DrawCommand>,
}

impl Canvas {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.commands.push(DrawCommand::FillRect {
            rect: LayoutRect::new(x, y, w, h),
            color,
        });
    }

    pub fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color, width: f32) {
        self.commands.push(DrawCommand::StrokeRect {
            rect: LayoutRect::new(x, y, w, h),
            width,
            color,
        });
    }

    pub fn fill_rounded_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        _radius: f32,
        color: Color,
    ) {
        // Stub: rounded corners not yet rendered — fall back to a plain fill.
        self.commands.push(DrawCommand::FillRect {
            rect: LayoutRect::new(x, y, w, h),
            color,
        });
    }

    pub fn fill_text(&mut self, text: &str, x: f32, y: f32, font_size: f32, color: Color) {
        self.commands.push(DrawCommand::Text {
            rect: LayoutRect::new(
                x,
                y,
                font_size * text.chars().count() as f32,
                font_size * 1.4,
            ),
            text: text.to_string(),
            font_size,
            color,
        });
    }

    pub fn set_clip(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.commands.push(DrawCommand::PushClip {
            rect: LayoutRect::new(x, y, w, h),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_layout_returns_positive_size() {
        let t = text("hello").font_size(14.0).color(Color::WHITE);
        let s = t.layout(&Constraint::loose(800.0, 600.0));
        assert!(s.width > 0.0);
        assert!(s.height > 0.0);
    }

    #[test]
    fn h_stack_sums_widths() {
        let stack = h_stack(vec![
            Box::new(text("a").font_size(14.0)),
            Box::new(text("b").font_size(14.0)),
        ])
        .gap(4.0);
        let s = stack.layout(&Constraint::loose(800.0, 100.0));
        assert!(s.width > 0.0);
    }

    #[test]
    fn v_stack_sums_heights() {
        let stack = v_stack(vec![
            Box::new(text("a").font_size(14.0)),
            Box::new(text("b").font_size(14.0)),
        ]);
        let s = stack.layout(&Constraint::loose(800.0, 100.0));
        assert!(s.height > 14.0); // at least two lines
    }

    #[test]
    fn divider_thickness_propagates() {
        let d = divider().thickness(3.0);
        let s = d.layout(&Constraint::loose(100.0, 100.0));
        assert_eq!(s.height, 3.0);
    }

    #[test]
    fn container_fixed_size() {
        let c = container(Box::new(text("x"))).width(200.0).height(80.0);
        let s = c.layout(&Constraint::loose(800.0, 600.0));
        assert_eq!(s.width, 200.0);
        assert_eq!(s.height, 80.0);
    }

    #[test]
    fn canvas_fill_rect_pushes_command() {
        let mut c = Canvas::new();
        c.fill_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        assert_eq!(c.commands.len(), 1);
    }

    #[test]
    fn text_render_emits_text_command() {
        let t = text("hi").color(Color::WHITE);
        let mut ctx = RenderCtx::new();
        t.render(&LayoutRect::new(0.0, 0.0, 100.0, 20.0), &mut ctx);
        assert!(!ctx.commands.is_empty());
    }

    #[test]
    fn min_width_is_respected() {
        let t = text("x").min_width(120.0);
        let s = t.layout(&Constraint::loose(800.0, 600.0));
        assert!(s.width >= 120.0);
    }

    #[test]
    fn max_width_clamps() {
        let t = text("supercalifragilisticexpialidocious")
            .font_size(18.0)
            .max_width(50.0);
        let s = t.layout(&Constraint::loose(800.0, 600.0));
        assert!(s.width <= 50.0);
    }

    // -------------------------------------------------------------------------
    // to_node / body() bridges
    // -------------------------------------------------------------------------

    #[test]
    fn text_to_node_preserves_content() {
        let n = text("hello").font_size(20.0).bold().to_node();
        match n {
            ViewNode::Leaf(LeafKind::Text(t)) => {
                assert_eq!(t.content, "hello");
                assert_eq!(t.font_size, 20.0);
                assert!(t.bold);
            }
            _ => panic!("expected text leaf"),
        }
    }

    #[test]
    fn text_with_padding_wraps_in_padding_modifier() {
        let n = text("hi").padding(8.0).to_node();
        if let ViewNode::Modified {
            modifier: Modifier::Padding(e),
            ..
        } = n
        {
            assert_eq!(e.top, 8.0);
        } else {
            panic!("expected padding modifier");
        }
    }

    #[test]
    fn text_with_background_wraps_in_background_modifier() {
        let n = text("hi").background(Color::WHITE).to_node();
        if let ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(c)),
            ..
        } = n
        {
            assert_eq!(c, Color::WHITE);
        } else {
            panic!("expected background modifier");
        }
    }

    #[test]
    fn spacer_to_node_is_spacer_leaf() {
        let n = spacer().to_node();
        assert!(matches!(n, ViewNode::Leaf(LeafKind::Spacer { .. })));
    }

    #[test]
    fn divider_to_node_is_divider_leaf() {
        let n = divider().thickness(2.0).color(Color::WHITE).to_node();
        match n {
            ViewNode::Leaf(LeafKind::Divider { thickness, color }) => {
                assert_eq!(thickness, 2.0);
                assert_eq!(color, Color::WHITE);
            }
            _ => panic!("expected divider leaf"),
        }
    }

    #[test]
    fn h_stack_to_node_is_container() {
        let s = h_stack(vec![boxed(text("a")), boxed(text("b"))]).gap(4.0);
        let n = s.to_node();
        if let ViewNode::Container { children, .. } = n {
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected container");
        }
    }

    #[test]
    fn v_stack_body_returns_container() {
        let s = v_stack(vec![boxed(text("a"))]);
        let n = s.body();
        assert!(matches!(n, ViewNode::Container { .. }));
    }

    #[test]
    fn container_to_node_includes_frame_modifier() {
        let c = container(boxed(text("x"))).width(100.0).height(50.0);
        let n = c.to_node();
        // Frame modifier is the outermost wrapper before any style wrap.
        let mut found_frame = false;
        let mut current = &n;
        loop {
            match current {
                ViewNode::Modified {
                    modifier: Modifier::Frame { width, height },
                    child,
                } => {
                    assert_eq!(*width, Some(100.0));
                    assert_eq!(*height, Some(50.0));
                    found_frame = true;
                    current = child;
                }
                ViewNode::Modified { child, .. } => current = child,
                _ => break,
            }
        }
        assert!(found_frame, "Container.to_node must emit a Frame modifier");
    }

    #[test]
    fn h_stack_body_renders_two_text_commands() {
        // End-to-end: build a widget tree, convert to ViewNode, render.
        let s = h_stack(vec![boxed(text("a")), boxed(text("b"))]);
        let node = s.to_node();
        let mut ctx = RenderCtx::new();
        crate::render::render_node(
            &node,
            crate::layout::Rect::new(0.0, 0.0, 200.0, 20.0),
            &mut ctx,
        );
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(texts, 2);
    }

    #[test]
    fn boxed_view_body_dispatches_dynamically() {
        let v: Box<dyn View> = boxed(text("hi"));
        let n = v.body();
        assert!(matches!(n, ViewNode::Leaf(LeafKind::Text(_))));
    }

    #[test]
    fn button_body_emits_text_leaf() {
        let n = Button::new("ok").body();
        // Button wraps via wrap_with_style. When style is default (no padding,
        // no bg, no min size), the inner is just the text leaf.
        assert!(matches!(n, ViewNode::Leaf(LeafKind::Text(_))));
    }

    #[test]
    fn stack_body_uses_zstack_layout() {
        let s = Stack::new(vec![boxed(text("a")), boxed(text("b"))]);
        let n = s.body();
        if let ViewNode::Container { children, .. } = n {
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected container");
        }
    }
}
