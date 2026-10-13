//! Phase 2 widget primitives — SwiftUI-style declarative API.
//!
//! Each primitive in this module is a small struct with public fields, a
//! `new()` constructor, fluent builder modifiers, and a [`View::body`]
//! implementation that lowers it into a [`ViewNode`] tree. The lowering
//! reuses the layout strategies and modifiers established in Phase 1, so
//! every primitive renders through the same [`crate::render::render_node`]
//! walker.
//!
//! The primitives are split into five groups:
//!
//! * **Interactive** — [`Button`], [`TextField`], [`Toggle`], [`Slider`],
//!   [`Picker`]: bind to a [`Signal`] and forward taps via
//!   [`Modifier::OnTap`].
//! * **Layout / structural** — [`ScrollView`], [`List`], [`LazyVStack`],
//!   [`LazyHStack`], [`LazyVGrid`], [`LazyHGrid`], [`Grid`] (with the
//!   accompanying [`GridLayout`] strategy).
//! * **Image / Media** — [`Image`], [`Canvas`].
//! * **Presentation stubs** — [`NavigationStack`], [`Sheet`], [`Alert`].
//! * **Nova-specific** — [`PeerApp`] (NWP sub-tree placeholder),
//!   [`MediaSession`] (OS media-session control surface).

use std::ops::RangeInclusive;
use std::rc::Rc;

use crate::color::Color;
use crate::layout::{
    Constraint, EdgeInsets, HStack as LHStack, Layout, LayoutRect, Measured, Placement, Point,
    ProposedSize, Rect, Size, VStack as LVStack, ZStack as LZStack,
};
use crate::render::RenderCtx;
use crate::signal::Signal;
use crate::view::{
    BackgroundFill, CanvasDraw, ContentFit, ImageLeaf, LeafKind, Modifier, TextLeaf, View, ViewNode,
};

// =============================================================================
// Helpers shared by several primitives
// =============================================================================

/// Wrap `node` in an [`Modifier::OnTap`] handler.
fn on_tap(node: ViewNode, action: Rc<dyn Fn()>) -> ViewNode {
    ViewNode::Modified {
        modifier: Modifier::OnTap(action),
        child: Box::new(node),
    }
}

/// Wrap `node` in a coloured [`Modifier::Background`].
fn with_background(node: ViewNode, color: Color) -> ViewNode {
    ViewNode::Modified {
        modifier: Modifier::Background(BackgroundFill::Color(color)),
        child: Box::new(node),
    }
}

/// Wrap `node` in a fixed [`Modifier::Frame`] with optional dims.
fn with_frame(node: ViewNode, width: Option<f32>, height: Option<f32>) -> ViewNode {
    ViewNode::Modified {
        modifier: Modifier::Frame { width, height },
        child: Box::new(node),
    }
}

/// Wrap `node` in symmetric padding.
fn with_padding(node: ViewNode, x: f32, y: f32) -> ViewNode {
    ViewNode::Modified {
        modifier: Modifier::Padding(EdgeInsets::xy(x, y)),
        child: Box::new(node),
    }
}

/// Construct a text leaf with the given content/colour at the standard
/// 14 pt font.
fn text_leaf(content: impl Into<String>, color: Color) -> ViewNode {
    let mut t = TextLeaf::new(content);
    t.color = color;
    ViewNode::Leaf(LeafKind::Text(t))
}

// =============================================================================
// Button
// =============================================================================

/// Visual style applied by [`Button::body`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ButtonStyle {
    /// Plain text on a transparent background.
    #[default]
    Default,
    /// Bordered, neutral chrome.
    Bordered,
    /// Bordered with the system accent fill.
    BorderedProminent,
    /// Plain (text only, no chrome) — same as [`ButtonStyle::Default`].
    Plain,
    /// Red / destructive styling.
    Destructive,
}

/// SwiftUI-style button. Renders a label and fires `action` on tap.
pub struct Button {
    pub label: String,
    pub action: Option<Rc<dyn Fn()>>,
    pub style: ButtonStyle,
    pub disabled: bool,
}

impl Button {
    /// Create a new button with the given label and no tap handler.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            action: None,
            style: ButtonStyle::Default,
            disabled: false,
        }
    }

    /// Attach a tap handler. `f` is wrapped in an `Rc` so the resulting
    /// `ViewNode` tree can be cloned freely.
    pub fn on_tap(mut self, f: impl Fn() + 'static) -> Self {
        self.action = Some(Rc::new(f));
        self
    }

    /// Override the button style.
    pub fn style(mut self, s: ButtonStyle) -> Self {
        self.style = s;
        self
    }

    /// Disable interaction. The body still renders but no `OnTap` modifier
    /// is emitted.
    pub fn disabled(mut self, b: bool) -> Self {
        self.disabled = b;
        self
    }
}

impl View for Button {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(self.label.chars().count() as f32 * 9.0 + 16.0, 28.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let (text_color, bg) = match self.style {
            ButtonStyle::Default | ButtonStyle::Plain => (Color::WHITE, None),
            ButtonStyle::Bordered => (Color::WHITE, Some(Color::rgb(0.20, 0.22, 0.26))),
            ButtonStyle::BorderedProminent => (Color::WHITE, Some(Color::rgb(0.30, 0.55, 0.95))),
            ButtonStyle::Destructive => (Color::WHITE, Some(Color::rgb(0.80, 0.20, 0.20))),
        };
        let mut node = with_padding(text_leaf(self.label.clone(), text_color), 12.0, 6.0);
        if let Some(bg) = bg {
            node = with_background(node, bg);
        }
        if let Some(action) = self.action.clone() {
            if !self.disabled {
                node = on_tap(node, action);
            }
        }
        node
    }
}

// =============================================================================
// TextField
// =============================================================================

/// Editable single-line text field bound to a [`Signal<String>`].
pub struct TextField {
    pub binding: Signal<String>,
    pub placeholder: String,
    pub on_submit: Option<Rc<dyn Fn(&str)>>,
    pub on_change: Option<Rc<dyn Fn(&str)>>,
    pub secure: bool,
    pub disabled: bool,
}

impl TextField {
    /// Build a text field bound to `binding`. The displayed value is the
    /// signal's current contents (or `placeholder` when empty).
    pub fn new(binding: &Signal<String>, placeholder: impl Into<String>) -> Self {
        Self {
            binding: binding.clone(),
            placeholder: placeholder.into(),
            on_submit: None,
            on_change: None,
            secure: false,
            disabled: false,
        }
    }

    /// Submit handler — invoked when the user presses return (Phase 5
    /// wiring).
    pub fn on_submit(mut self, f: impl Fn(&str) + 'static) -> Self {
        self.on_submit = Some(Rc::new(f));
        self
    }

    /// Change handler — invoked on every keystroke (Phase 5 wiring).
    pub fn on_change(mut self, f: impl Fn(&str) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    /// Mask characters as bullets (e.g. for passwords).
    pub fn secure(mut self, b: bool) -> Self {
        self.secure = b;
        self
    }

    /// Reject focus / edits.
    pub fn disabled(mut self, b: bool) -> Self {
        self.disabled = b;
        self
    }
}

impl View for TextField {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(180.0, 28.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let raw = self.binding.get();
        let (display, color) = if raw.is_empty() {
            (self.placeholder.clone(), Color::rgb(0.55, 0.55, 0.6))
        } else if self.secure {
            ("\u{2022}".repeat(raw.chars().count()), Color::WHITE)
        } else {
            (raw, Color::WHITE)
        };

        let label = with_padding(text_leaf(display, color), 8.0, 4.0);
        let chrome = with_background(label, Color::rgb(0.14, 0.14, 0.18));
        let framed = with_frame(chrome, Some(180.0), Some(28.0));

        // The OnTap modifier stands in for "focus this field" until Phase 5
        // routes real key events. When the field is disabled, no tap
        // handler is attached.
        if self.disabled {
            return framed;
        }
        let binding = self.binding.clone();
        on_tap(
            framed,
            Rc::new(move || {
                // No-op focus stub; this exists so the embedder's hit-test
                // pipeline finds a tap target. The actual edit pipeline will
                // route through `on_change` / `on_submit` in Phase 5.
                let _ = binding.get();
            }),
        )
    }
}

// =============================================================================
// Toggle
// =============================================================================

/// On/off switch bound to a [`Signal<bool>`].
pub struct Toggle {
    pub binding: Signal<bool>,
    pub label: String,
}

impl Toggle {
    pub fn new(binding: &Signal<bool>, label: impl Into<String>) -> Self {
        Self {
            binding: binding.clone(),
            label: label.into(),
        }
    }
}

impl View for Toggle {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(self.label.chars().count() as f32 * 9.0 + 60.0, 24.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let on = self.binding.get();
        let track_color = if on {
            Color::rgb(0.30, 0.75, 0.40)
        } else {
            Color::rgb(0.32, 0.32, 0.36)
        };
        let thumb_color = Color::WHITE;

        let on_now = on;
        let track: ViewNode = ViewNode::Leaf(LeafKind::Canvas(Rc::new(
            move |ctx: &mut RenderCtx, size: Size| {
                let track_rect = LayoutRect::new(0.0, (size.height - 18.0) * 0.5, 40.0, 18.0);
                ctx.fill_rect(&track_rect, track_color);
                let thumb_x = if on_now { 22.0 } else { 2.0 };
                let thumb = LayoutRect::new(thumb_x, track_rect.y + 2.0, 14.0, 14.0);
                ctx.fill_rect(&thumb, thumb_color);
            },
        )));

        let track = with_frame(track, Some(40.0), Some(24.0));
        let row = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(8.0)),
            children: vec![track, text_leaf(self.label.clone(), Color::WHITE)],
        };

        let binding = self.binding.clone();
        on_tap(
            row,
            Rc::new(move || {
                let v = binding.get();
                binding.set(!v);
            }),
        )
    }
}

// =============================================================================
// Slider
// =============================================================================

/// Continuous (or discrete) numeric slider bound to a [`Signal<f64>`].
pub struct Slider {
    pub binding: Signal<f64>,
    pub range: RangeInclusive<f64>,
    pub step: Option<f64>,
}

impl Slider {
    pub fn new(binding: &Signal<f64>, range: RangeInclusive<f64>) -> Self {
        Self {
            binding: binding.clone(),
            range,
            step: None,
        }
    }

    pub fn step(mut self, s: f64) -> Self {
        self.step = Some(s);
        self
    }
}

impl View for Slider {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(160.0, 24.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let value = self.binding.get();
        let lo = *self.range.start();
        let hi = *self.range.end();
        let span = (hi - lo).max(f64::EPSILON);
        let frac = ((value - lo) / span).clamp(0.0, 1.0) as f32;

        let canvas: ViewNode = ViewNode::Leaf(LeafKind::Canvas(Rc::new(
            move |ctx: &mut RenderCtx, size: Size| {
                let track_h = 4.0;
                let y = (size.height - track_h) * 0.5;
                let track = LayoutRect::new(0.0, y, size.width, track_h);
                ctx.fill_rect(&track, Color::rgb(0.26, 0.26, 0.30));

                let filled_w = size.width * frac;
                let filled = LayoutRect::new(0.0, y, filled_w, track_h);
                ctx.fill_rect(&filled, Color::rgb(0.30, 0.55, 0.95));

                let thumb_d = 14.0;
                let thumb_x = (filled_w - thumb_d * 0.5).max(0.0);
                let thumb =
                    LayoutRect::new(thumb_x, (size.height - thumb_d) * 0.5, thumb_d, thumb_d);
                ctx.fill_rect(&thumb, Color::WHITE);
            },
        )));

        with_frame(canvas, Some(160.0), Some(24.0))
    }
}

// =============================================================================
// Picker
// =============================================================================

/// Cycling enum picker bound to a [`Signal<T>`].
pub struct Picker<T: Clone + PartialEq + 'static> {
    pub binding: Signal<T>,
    pub options: Vec<(String, T)>,
    pub label: String,
}

impl<T: Clone + PartialEq + 'static> Picker<T> {
    pub fn new(binding: &Signal<T>, label: impl Into<String>) -> Self {
        Self {
            binding: binding.clone(),
            options: Vec::new(),
            label: label.into(),
        }
    }

    /// Append an option. The same value can be reused; equality is by
    /// `T: PartialEq`.
    pub fn option(mut self, label: impl Into<String>, value: T) -> Self {
        self.options.push((label.into(), value));
        self
    }
}

impl<T: Clone + PartialEq + 'static> View for Picker<T> {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(120.0, 24.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let current = self.binding.get();
        let display = self
            .options
            .iter()
            .find(|(_, v)| v == &current)
            .map(|(l, _)| l.clone())
            .unwrap_or_else(|| "—".to_string());

        let label_node = if self.label.is_empty() {
            text_leaf(display, Color::WHITE)
        } else {
            ViewNode::Container {
                layout: Box::new(LHStack::new().spacing(6.0)),
                children: vec![
                    text_leaf(self.label.clone(), Color::rgb(0.7, 0.7, 0.75)),
                    text_leaf(display, Color::WHITE),
                ],
            }
        };
        let chrome = with_background(
            with_padding(label_node, 10.0, 4.0),
            Color::rgb(0.18, 0.18, 0.22),
        );

        let binding = self.binding.clone();
        let opts = self.options.clone();
        on_tap(
            chrome,
            Rc::new(move || {
                if opts.is_empty() {
                    return;
                }
                let cur = binding.get();
                let mut next_idx = 0usize;
                for (i, (_, v)) in opts.iter().enumerate() {
                    if v == &cur {
                        next_idx = (i + 1) % opts.len();
                        break;
                    }
                }
                binding.set(opts[next_idx].1.clone());
            }),
        )
    }
}

// =============================================================================
// ScrollView
// =============================================================================

/// Axes a [`ScrollView`] can scroll along. Phase 2 stores the value but
/// does not yet animate scroll (that lands with Phase 5 input routing).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScrollAxis {
    Horizontal,
    Vertical,
    Both,
}

/// Scrollable container. The child is laid out at its natural size and
/// clipped to the view's bounds.
pub struct ScrollView {
    pub axes: ScrollAxis,
    pub show_indicators: bool,
    pub child: Box<ViewNode>,
}

impl ScrollView {
    /// Vertical-only scroll view.
    pub fn vertical(child: ViewNode) -> Self {
        Self {
            axes: ScrollAxis::Vertical,
            show_indicators: true,
            child: Box::new(child),
        }
    }

    /// Horizontal-only scroll view.
    pub fn horizontal(child: ViewNode) -> Self {
        Self {
            axes: ScrollAxis::Horizontal,
            show_indicators: true,
            child: Box::new(child),
        }
    }

    /// Both axes can scroll independently.
    pub fn both(child: ViewNode) -> Self {
        Self {
            axes: ScrollAxis::Both,
            show_indicators: true,
            child: Box::new(child),
        }
    }

    /// Toggle the visible scroll indicators.
    pub fn show_indicators(mut self, b: bool) -> Self {
        self.show_indicators = b;
        self
    }
}

impl View for ScrollView {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let inner = self.child.clone_node();
        let container = ViewNode::Container {
            layout: Box::new(LZStack::new()),
            children: vec![inner],
        };
        with_frame(container, None, None)
    }
}

// =============================================================================
// List
// =============================================================================

/// Visual style for a [`List`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ListStyle {
    #[default]
    Plain,
    Inset,
    Grouped,
    Sidebar,
}

/// SwiftUI-style list — a vertical stack of rows separated by [`LeafKind::Divider`]s.
pub struct List {
    pub items: Vec<ViewNode>,
    pub selection: Option<Signal<usize>>,
    pub style: ListStyle,
}

impl Default for List {
    fn default() -> Self {
        Self::new()
    }
}

impl List {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selection: None,
            style: ListStyle::Plain,
        }
    }

    /// Append a row.
    pub fn item(mut self, child: ViewNode) -> Self {
        self.items.push(child);
        self
    }

    /// Bind the selected-index signal.
    pub fn selection(mut self, s: &Signal<usize>) -> Self {
        self.selection = Some(s.clone());
        self
    }

    /// Override the visual style.
    pub fn style(mut self, st: ListStyle) -> Self {
        self.style = st;
        self
    }
}

impl View for List {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let mut rows: Vec<ViewNode> = Vec::with_capacity(self.items.len() * 2);
        let divider_color = match self.style {
            ListStyle::Plain | ListStyle::Inset => Color::rgba(0.35, 0.35, 0.4, 0.6),
            ListStyle::Grouped | ListStyle::Sidebar => Color::rgba(0.25, 0.25, 0.30, 0.8),
        };

        for (i, item) in self.items.iter().enumerate() {
            let mut row = item.clone_node();
            if matches!(self.style, ListStyle::Inset | ListStyle::Sidebar) {
                row = with_padding(row, 12.0, 4.0);
            } else {
                row = with_padding(row, 0.0, 4.0);
            }
            if let Some(sel) = &self.selection {
                let sel = sel.clone();
                let idx = i;
                row = on_tap(row, Rc::new(move || sel.set(idx)));
            }
            rows.push(row);
            if i + 1 < self.items.len() {
                rows.push(ViewNode::Leaf(LeafKind::Divider {
                    thickness: 1.0,
                    color: divider_color,
                }));
            }
        }

        ViewNode::Container {
            layout: Box::new(LVStack::new().spacing(0.0)),
            children: rows,
        }
    }
}

// =============================================================================
// LazyVStack / LazyHStack — structural placeholders for virtualised stacks
// =============================================================================

/// Vertical stack that *will* virtualise its children once Phase 5 wires
/// scrolling. Phase 2 behaves like a regular VStack.
pub struct LazyVStack {
    pub items: Vec<ViewNode>,
    pub spacing: f32,
}

impl Default for LazyVStack {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyVStack {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            spacing: 0.0,
        }
    }
    pub fn item(mut self, child: ViewNode) -> Self {
        self.items.push(child);
        self
    }
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }
}

impl View for LazyVStack {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        ViewNode::Container {
            layout: Box::new(LVStack::new().spacing(self.spacing)),
            children: self.items.iter().map(|c| c.clone_node()).collect(),
        }
    }
}

/// Horizontal counterpart of [`LazyVStack`].
pub struct LazyHStack {
    pub items: Vec<ViewNode>,
    pub spacing: f32,
}

impl Default for LazyHStack {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyHStack {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            spacing: 0.0,
        }
    }
    pub fn item(mut self, child: ViewNode) -> Self {
        self.items.push(child);
        self
    }
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }
}

impl View for LazyHStack {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(self.spacing)),
            children: self.items.iter().map(|c| c.clone_node()).collect(),
        }
    }
}

// =============================================================================
// Grid items / GridLayout / LazyVGrid / LazyHGrid / Grid
// =============================================================================

/// A single column or row spec consumed by the grid layout.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GridItem {
    Fixed(f32),
    Flexible { min: f32, max: f32 },
    Adaptive { min: f32 },
}

impl GridItem {
    /// Resolve the column / row width for an Available bound.
    fn resolve(&self, available: f32) -> f32 {
        match self {
            GridItem::Fixed(v) => *v,
            GridItem::Flexible { min, max } => available.clamp(*min, *max),
            GridItem::Adaptive { min } => available.max(*min),
        }
    }
}

/// Two-pass grid layout. Lays out children in row-major order across
/// `cols.len()` columns; rows are sized to the tallest cell.
#[derive(Clone, Debug)]
pub struct GridLayout {
    pub cols: Vec<GridItem>,
    pub spacing: f32,
}

impl GridLayout {
    pub fn new(cols: Vec<GridItem>) -> Self {
        Self { cols, spacing: 0.0 }
    }
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }

    fn col_widths(&self, total_w: f32) -> Vec<f32> {
        if self.cols.is_empty() {
            return Vec::new();
        }
        let n = self.cols.len() as f32;
        let total_spacing = self.spacing * (n - 1.0).max(0.0);
        let avail_per = ((total_w - total_spacing) / n).max(0.0);
        self.cols.iter().map(|c| c.resolve(avail_per)).collect()
    }

    fn row_heights(&self, subviews: &[Measured]) -> Vec<f32> {
        if self.cols.is_empty() {
            return Vec::new();
        }
        let cols = self.cols.len();
        let rows = subviews.len().div_ceil(cols);
        let mut heights = vec![0.0_f32; rows];
        for (i, m) in subviews.iter().enumerate() {
            let r = i / cols;
            heights[r] = heights[r].max(m.ideal.height);
        }
        heights
    }
}

impl Layout for GridLayout {
    fn clone_box(&self) -> Box<dyn Layout> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn size_that_fits(&self, proposal: ProposedSize, subviews: &[Measured]) -> Size {
        if self.cols.is_empty() {
            return Size::zero();
        }
        let total_w = proposal.width.unwrap_or_else(|| {
            self.cols.iter().map(|c| c.resolve(0.0)).sum::<f32>()
                + self.spacing * (self.cols.len().saturating_sub(1)) as f32
        });
        let widths = self.col_widths(total_w);
        let heights = self.row_heights(subviews);
        let n_rows = heights.len();
        let total_h = heights.iter().sum::<f32>() + self.spacing * n_rows.saturating_sub(1) as f32;
        let total_w_used =
            widths.iter().sum::<f32>() + self.spacing * widths.len().saturating_sub(1) as f32;
        Size::new(total_w_used, total_h)
    }

    fn place_subviews(&self, bounds: Rect, _proposal: ProposedSize, subviews: &mut [Placement]) {
        if self.cols.is_empty() || subviews.is_empty() {
            return;
        }
        let cols = self.cols.len();
        let widths = self.col_widths(bounds.size.width);
        let measured: Vec<Measured> = subviews.iter().map(|p| Measured::fixed(p.size)).collect();
        let heights = self.row_heights(&measured);

        let mut y = bounds.origin.y;
        for r in 0..heights.len() {
            let mut x = bounds.origin.x;
            for c in 0..cols {
                let i = r * cols + c;
                if i >= subviews.len() {
                    break;
                }
                subviews[i] = Placement::new(Point::new(x, y), Size::new(widths[c], heights[r]));
                x += widths[c] + self.spacing;
            }
            y += heights[r] + self.spacing;
        }
    }
}

/// Lazily-virtualised vertical grid. Phase 2 lowers to a normal grid.
pub struct LazyVGrid {
    pub columns: Vec<GridItem>,
    pub items: Vec<ViewNode>,
    pub spacing: f32,
}

impl LazyVGrid {
    pub fn new(columns: Vec<GridItem>) -> Self {
        Self {
            columns,
            items: Vec::new(),
            spacing: 0.0,
        }
    }
    pub fn item(mut self, child: ViewNode) -> Self {
        self.items.push(child);
        self
    }
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }
}

impl View for LazyVGrid {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        ViewNode::Container {
            layout: Box::new(GridLayout::new(self.columns.clone()).spacing(self.spacing)),
            children: self.items.iter().map(|c| c.clone_node()).collect(),
        }
    }
}

/// Lazily-virtualised horizontal grid. Phase 2 lowers to a grid layout
/// over the row specs.
pub struct LazyHGrid {
    pub rows: Vec<GridItem>,
    pub items: Vec<ViewNode>,
    pub spacing: f32,
}

impl LazyHGrid {
    pub fn new(rows: Vec<GridItem>) -> Self {
        Self {
            rows,
            items: Vec::new(),
            spacing: 0.0,
        }
    }
    pub fn item(mut self, child: ViewNode) -> Self {
        self.items.push(child);
        self
    }
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }
}

impl View for LazyHGrid {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        ViewNode::Container {
            layout: Box::new(GridLayout::new(self.rows.clone()).spacing(self.spacing)),
            children: self.items.iter().map(|c| c.clone_node()).collect(),
        }
    }
}

/// A row in a [`Grid`]. Cells are placed in column-index order.
pub struct GridRow {
    pub cells: Vec<ViewNode>,
}

impl GridRow {
    pub fn new(cells: Vec<ViewNode>) -> Self {
        Self { cells }
    }
}

/// SwiftUI-style `Grid` — a list of rows, each holding cells. Columns are
/// sized to the widest cell across rows.
pub struct Grid {
    pub rows: Vec<GridRow>,
}

impl Default for Grid {
    fn default() -> Self {
        Self::new()
    }
}

impl Grid {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn row(mut self, row: GridRow) -> Self {
        self.rows.push(row);
        self
    }
}

impl View for Grid {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let max_cols = self.rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
        if max_cols == 0 {
            return ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 });
        }
        let columns = vec![
            GridItem::Flexible {
                min: 0.0,
                max: f32::INFINITY
            };
            max_cols
        ];
        let mut children: Vec<ViewNode> = Vec::with_capacity(self.rows.len() * max_cols);
        for r in &self.rows {
            for c in 0..max_cols {
                if let Some(cell) = r.cells.get(c) {
                    children.push(cell.clone_node());
                } else {
                    children.push(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
                }
            }
        }
        ViewNode::Container {
            layout: Box::new(GridLayout::new(columns)),
            children,
        }
    }
}

// =============================================================================
// Image
// =============================================================================

/// Resource-loaded image. Wraps [`LeafKind::Image`].
pub struct Image {
    pub url: String,
    pub fit: ContentFit,
    pub placeholder_color: Color,
}

impl Image {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            fit: ContentFit::Contain,
            placeholder_color: Color::rgb(0.16, 0.16, 0.20),
        }
    }
    pub fn fit(mut self, f: ContentFit) -> Self {
        self.fit = f;
        self
    }
    pub fn placeholder(mut self, c: Color) -> Self {
        self.placeholder_color = c;
        self
    }
}

impl View for Image {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::new(64.0, 64.0)
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let leaf = ViewNode::Leaf(LeafKind::Image(ImageLeaf {
            url: self.url.clone(),
            fit: self.fit,
        }));
        with_background(leaf, self.placeholder_color)
    }
}

// =============================================================================
// Canvas — imperative escape hatch
// =============================================================================

/// Imperative draw escape hatch. The closure receives a [`RenderCtx`] and
/// the leaf's resolved size.
pub struct Canvas {
    pub draw: CanvasDraw,
}

impl Canvas {
    /// Wrap an arbitrary draw closure into a Photon canvas leaf.
    pub fn new<F>(draw: F) -> Self
    where
        F: Fn(&mut RenderCtx, Size) + 'static,
    {
        Self {
            draw: Rc::new(draw),
        }
    }
}

impl View for Canvas {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        ViewNode::Leaf(LeafKind::Canvas(self.draw.clone()))
    }
}

// =============================================================================
// NavigationStack
// =============================================================================

/// Phase 2 stub: renders the root only. The path signal is stored so
/// later phases can pop / push routes without a structural change.
pub struct NavigationStack {
    pub path: Signal<Vec<String>>,
    pub root: Box<ViewNode>,
}

impl NavigationStack {
    pub fn new(path: &Signal<Vec<String>>, root: ViewNode) -> Self {
        Self {
            path: path.clone(),
            root: Box::new(root),
        }
    }
}

impl View for NavigationStack {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        // Reading the path subscribes any active reader to the navigation
        // signal so a later push/pop re-renders the stack.
        let _ = self.path.get();
        self.root.clone_node()
    }
}

// =============================================================================
// Sheet
// =============================================================================

/// Modal sheet. When `is_presented` is true, renders a dim overlay + the
/// content card; otherwise the body is empty.
pub struct Sheet {
    pub is_presented: Signal<bool>,
    pub content: Box<ViewNode>,
}

impl Sheet {
    pub fn new(is_presented: &Signal<bool>, content: ViewNode) -> Self {
        Self {
            is_presented: is_presented.clone(),
            content: Box::new(content),
        }
    }
}

impl View for Sheet {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        if !self.is_presented.get() {
            return ViewNode::Container {
                layout: Box::new(LZStack::new()),
                children: Vec::new(),
            };
        }
        let dim = with_background(
            ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }),
            Color::rgba(0.0, 0.0, 0.0, 0.45),
        );
        let card = with_background(
            with_padding(self.content.clone_node(), 16.0, 16.0),
            Color::rgb(0.14, 0.14, 0.18),
        );
        ViewNode::Container {
            layout: Box::new(LZStack::new()),
            children: vec![dim, card],
        }
    }
}

// =============================================================================
// Alert
// =============================================================================

/// Style applied to an [`AlertAction`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AlertActionStyle {
    #[default]
    Default,
    Cancel,
    Destructive,
}

/// Single tap target inside an [`Alert`].
pub struct AlertAction {
    pub label: String,
    pub on_tap: Rc<dyn Fn()>,
    pub style: AlertActionStyle,
}

impl AlertAction {
    pub fn new(label: impl Into<String>, on_tap: impl Fn() + 'static) -> Self {
        Self {
            label: label.into(),
            on_tap: Rc::new(on_tap),
            style: AlertActionStyle::Default,
        }
    }
    pub fn style(mut self, s: AlertActionStyle) -> Self {
        self.style = s;
        self
    }
}

/// Standard alert dialog with a title, message, and a row of actions.
pub struct Alert {
    pub is_presented: Signal<bool>,
    pub title: String,
    pub message: String,
    pub actions: Vec<AlertAction>,
}

impl Alert {
    pub fn new(
        is_presented: &Signal<bool>,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            is_presented: is_presented.clone(),
            title: title.into(),
            message: message.into(),
            actions: Vec::new(),
        }
    }

    pub fn action(mut self, a: AlertAction) -> Self {
        self.actions.push(a);
        self
    }
}

impl View for Alert {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        if !self.is_presented.get() {
            return ViewNode::Container {
                layout: Box::new(LZStack::new()),
                children: Vec::new(),
            };
        }

        let title = text_leaf(self.title.clone(), Color::WHITE);
        let message = text_leaf(self.message.clone(), Color::rgb(0.85, 0.85, 0.9));

        let mut action_nodes: Vec<ViewNode> = Vec::with_capacity(self.actions.len());
        for action in &self.actions {
            let bg = match action.style {
                AlertActionStyle::Default => Color::rgb(0.18, 0.18, 0.22),
                AlertActionStyle::Cancel => Color::rgb(0.30, 0.30, 0.34),
                AlertActionStyle::Destructive => Color::rgb(0.80, 0.22, 0.22),
            };
            let label = with_padding(text_leaf(action.label.clone(), Color::WHITE), 12.0, 6.0);
            let chrome = with_background(label, bg);
            action_nodes.push(on_tap(chrome, Rc::clone(&action.on_tap)));
        }
        let actions_row = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(8.0)),
            children: action_nodes,
        };

        let card_inner = ViewNode::Container {
            layout: Box::new(LVStack::new().spacing(8.0)),
            children: vec![title, message, actions_row],
        };
        let card = with_background(
            with_padding(card_inner, 16.0, 16.0),
            Color::rgb(0.14, 0.14, 0.18),
        );
        let dim = with_background(
            ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }),
            Color::rgba(0.0, 0.0, 0.0, 0.5),
        );
        ViewNode::Container {
            layout: Box::new(LZStack::new()),
            children: vec![dim, card],
        }
    }
}

// =============================================================================
// PeerApp — embedded NWP sub-tree placeholder
// =============================================================================

/// Embedded peer application. Phase 2 paints a placeholder card with the
/// addr / app id / route; Phase 5 will swap the canvas leaf for a real
/// NWP sub-renderer.
pub struct PeerApp {
    pub addr: String,
    pub app_id: String,
    pub route: String,
}

impl PeerApp {
    pub fn new(addr: impl Into<String>, app_id: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            app_id: app_id.into(),
            route: String::new(),
        }
    }
    pub fn route(mut self, r: impl Into<String>) -> Self {
        self.route = r.into();
        self
    }
}

impl View for PeerApp {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let label = format!("Peer: {}@{}{}", self.app_id, self.addr, self.route);
        ViewNode::Leaf(LeafKind::Canvas(Rc::new(
            move |ctx: &mut RenderCtx, size: Size| {
                let r = LayoutRect::new(0.0, 0.0, size.width, size.height);
                ctx.fill_rect(&r, Color::rgb(0.12, 0.16, 0.22));
                ctx.fill_text(
                    &LayoutRect::new(8.0, 8.0, size.width - 16.0, 18.0),
                    &label,
                    13.0,
                    Color::WHITE,
                );
            },
        )))
    }
}

// =============================================================================
// MediaSession
// =============================================================================

/// Media transport control exposed by [`MediaSession`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MediaControl {
    Play,
    Pause,
    Next,
    Previous,
    SeekForward,
    SeekBackward,
}

impl MediaControl {
    /// Glyph rendered for this control.
    pub fn glyph(self) -> &'static str {
        match self {
            MediaControl::Play => "Play",
            MediaControl::Pause => "Pause",
            MediaControl::Next => "Next",
            MediaControl::Previous => "Prev",
            MediaControl::SeekForward => ">>",
            MediaControl::SeekBackward => "<<",
        }
    }
}

/// Now-playing surface — metadata plus a row of transport controls.
pub struct MediaSession {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub artwork_url: Option<String>,
    pub controls: Vec<MediaControl>,
}

impl MediaSession {
    pub fn new(
        title: impl Into<String>,
        artist: impl Into<String>,
        album: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            artist: artist.into(),
            album: album.into(),
            artwork_url: None,
            controls: Vec::new(),
        }
    }
    pub fn artwork(mut self, url: impl Into<String>) -> Self {
        self.artwork_url = Some(url.into());
        self
    }
    pub fn control(mut self, c: MediaControl) -> Self {
        self.controls.push(c);
        self
    }
}

impl View for MediaSession {
    fn layout(&self, _c: &Constraint) -> Size {
        Size::zero()
    }
    fn render(&self, _rect: &LayoutRect, _ctx: &mut RenderCtx) {}

    fn body(&self) -> ViewNode {
        let title = text_leaf(self.title.clone(), Color::WHITE);
        let artist = text_leaf(self.artist.clone(), Color::rgb(0.8, 0.8, 0.85));
        let album = text_leaf(self.album.clone(), Color::rgb(0.65, 0.65, 0.7));

        let mut control_buttons: Vec<ViewNode> = Vec::with_capacity(self.controls.len());
        for c in &self.controls {
            let label = with_padding(text_leaf(c.glyph(), Color::WHITE), 8.0, 4.0);
            control_buttons.push(with_background(label, Color::rgb(0.20, 0.22, 0.26)));
        }
        let controls_row = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(6.0)),
            children: control_buttons,
        };

        let metadata_col = ViewNode::Container {
            layout: Box::new(LVStack::new().spacing(2.0)),
            children: vec![title, artist, album],
        };

        let mut row_children: Vec<ViewNode> = Vec::with_capacity(2);
        if let Some(url) = &self.artwork_url {
            let art = ViewNode::Leaf(LeafKind::Image(ImageLeaf {
                url: url.clone(),
                fit: ContentFit::Cover,
            }));
            row_children.push(with_frame(art, Some(48.0), Some(48.0)));
        }
        row_children.push(metadata_col);

        let row = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(10.0)),
            children: row_children,
        };

        ViewNode::Container {
            layout: Box::new(LVStack::new().spacing(8.0)),
            children: vec![row, controls_row],
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{DrawCommand, render_node};
    use crate::signal::SignalArena;
    use std::cell::Cell;

    // -------------------------------------------------------------------------
    // Button
    // -------------------------------------------------------------------------

    #[test]
    fn button_constructor_sets_label() {
        let b = Button::new("Save");
        assert_eq!(b.label, "Save");
        assert!(b.action.is_none());
        assert_eq!(b.style, ButtonStyle::Default);
        assert!(!b.disabled);
    }

    #[test]
    fn button_builder_chains_modifiers() {
        let b = Button::new("Go")
            .style(ButtonStyle::BorderedProminent)
            .disabled(true)
            .on_tap(|| {});
        assert_eq!(b.style, ButtonStyle::BorderedProminent);
        assert!(b.disabled);
        assert!(b.action.is_some());
    }

    #[test]
    fn button_body_with_default_style_has_no_background() {
        let b = Button::new("Plain").style(ButtonStyle::Default);
        // Lowering yields padding(text) — no background, no on_tap.
        match b.body() {
            ViewNode::Modified {
                modifier: Modifier::Padding(_),
                ..
            } => {}
            other => panic!("expected padding modifier, got {:?}", other),
        }
    }

    #[test]
    fn button_body_with_bordered_prominent_has_background() {
        let b = Button::new("Submit").style(ButtonStyle::BorderedProminent);
        match b.body() {
            ViewNode::Modified {
                modifier: Modifier::Background(_),
                ..
            } => {}
            other => panic!("expected background modifier, got {:?}", other),
        }
    }

    #[test]
    fn button_disabled_does_not_emit_on_tap() {
        let b = Button::new("X")
            .style(ButtonStyle::Bordered)
            .disabled(true)
            .on_tap(|| {});
        // Walk the tree looking for OnTap.
        let mut found = false;
        fn walk(n: &ViewNode, found: &mut bool) {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(_),
                    ..
                } => *found = true,
                ViewNode::Modified { child, .. } => walk(child, found),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, found);
                    }
                }
                _ => {}
            }
        }
        walk(&b.body(), &mut found);
        assert!(!found);
    }

    #[test]
    fn button_on_tap_fires_when_invoked() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let b = Button::new("X").on_tap(move || {
            c.set(c.get() + 1);
        });
        // Pull the action out and call it.
        let action = b.action.clone().expect("action attached");
        action();
        action();
        assert_eq!(counter.get(), 2);
    }

    // -------------------------------------------------------------------------
    // TextField
    // -------------------------------------------------------------------------

    #[test]
    fn text_field_constructor_clones_binding() {
        let arena = SignalArena::new();
        let sig = arena.create::<String>("hello".into());
        let tf = TextField::new(&sig, "type here");
        assert_eq!(tf.placeholder, "type here");
        assert_eq!(tf.binding.get(), "hello");
    }

    #[test]
    fn text_field_secure_masks_value() {
        let arena = SignalArena::new();
        let sig = arena.create::<String>("abcd".into());
        let tf = TextField::new(&sig, "").secure(true);
        // Walk the body looking for the rendered text leaf.
        let mut content = String::new();
        fn walk(n: &ViewNode, out: &mut String) {
            match n {
                ViewNode::Leaf(LeafKind::Text(t)) => *out = t.content.clone(),
                ViewNode::Modified { child, .. } => walk(child, out),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, out);
                    }
                }
                _ => {}
            }
        }
        walk(&tf.body(), &mut content);
        assert_eq!(content, "\u{2022}\u{2022}\u{2022}\u{2022}");
    }

    #[test]
    fn text_field_empty_shows_placeholder() {
        let arena = SignalArena::new();
        let sig = arena.create::<String>(String::new());
        let tf = TextField::new(&sig, "Search…");
        let mut content = String::new();
        fn walk(n: &ViewNode, out: &mut String) {
            match n {
                ViewNode::Leaf(LeafKind::Text(t)) => *out = t.content.clone(),
                ViewNode::Modified { child, .. } => walk(child, out),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, out);
                    }
                }
                _ => {}
            }
        }
        walk(&tf.body(), &mut content);
        assert_eq!(content, "Search…");
    }

    #[test]
    fn text_field_disabled_omits_on_tap() {
        let arena = SignalArena::new();
        let sig = arena.create::<String>("x".into());
        let tf = TextField::new(&sig, "").disabled(true);
        let mut found = false;
        fn walk(n: &ViewNode, found: &mut bool) {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(_),
                    ..
                } => *found = true,
                ViewNode::Modified { child, .. } => walk(child, found),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, found);
                    }
                }
                _ => {}
            }
        }
        walk(&tf.body(), &mut found);
        assert!(!found);
    }

    #[test]
    fn text_field_handlers_are_stored() {
        let arena = SignalArena::new();
        let sig = arena.create::<String>(String::new());
        let tf = TextField::new(&sig, "").on_change(|_| {}).on_submit(|_| {});
        assert!(tf.on_change.is_some());
        assert!(tf.on_submit.is_some());
    }

    // -------------------------------------------------------------------------
    // Toggle
    // -------------------------------------------------------------------------

    #[test]
    fn toggle_body_has_canvas_track() {
        let arena = SignalArena::new();
        let sig = arena.create::<bool>(false);
        let t = Toggle::new(&sig, "Wi-Fi");
        let mut found_canvas = false;
        fn walk(n: &ViewNode, c: &mut bool) {
            match n {
                ViewNode::Leaf(LeafKind::Canvas(_)) => *c = true,
                ViewNode::Modified { child, .. } => walk(child, c),
                ViewNode::Container { children, .. } => {
                    for ch in children {
                        walk(ch, c);
                    }
                }
                _ => {}
            }
        }
        walk(&t.body(), &mut found_canvas);
        assert!(found_canvas);
    }

    #[test]
    fn toggle_tap_flips_binding() {
        let arena = SignalArena::new();
        let sig = arena.create::<bool>(false);
        let t = Toggle::new(&sig, "On");
        let body = t.body();
        // Find the OnTap action and call it.
        fn find_tap(n: &ViewNode) -> Option<Rc<dyn Fn()>> {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(f),
                    ..
                } => Some(Rc::clone(f)),
                ViewNode::Modified { child, .. } => find_tap(child),
                ViewNode::Container { children, .. } => children.iter().find_map(find_tap),
                _ => None,
            }
        }
        let action = find_tap(&body).expect("toggle has an OnTap");
        action();
        assert!(sig.get());
        action();
        assert!(!sig.get());
    }

    #[test]
    fn toggle_label_field_set() {
        let arena = SignalArena::new();
        let sig = arena.create::<bool>(true);
        let t = Toggle::new(&sig, "Bluetooth");
        assert_eq!(t.label, "Bluetooth");
    }

    // -------------------------------------------------------------------------
    // Slider
    // -------------------------------------------------------------------------

    #[test]
    fn slider_stores_range_and_step() {
        let arena = SignalArena::new();
        let sig = arena.create::<f64>(0.5);
        let s = Slider::new(&sig, 0.0..=1.0).step(0.1);
        assert_eq!(*s.range.start(), 0.0);
        assert_eq!(*s.range.end(), 1.0);
        assert_eq!(s.step, Some(0.1));
    }

    #[test]
    fn slider_body_renders_canvas_through_render_node() {
        let arena = SignalArena::new();
        let sig = arena.create::<f64>(0.5);
        let s = Slider::new(&sig, 0.0..=1.0);
        let body = s.body();
        let mut ctx = RenderCtx::new();
        render_node(&body, Rect::new(0.0, 0.0, 160.0, 24.0), &mut ctx);
        // Track + filled + thumb + push/pop clip → at least one fill rect.
        let fills = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::FillRect { .. }))
            .count();
        assert!(fills >= 3);
    }

    #[test]
    fn slider_clamps_value_outside_range() {
        let arena = SignalArena::new();
        let sig = arena.create::<f64>(2.0); // out of range
        let s = Slider::new(&sig, 0.0..=1.0);
        let body = s.body();
        let mut ctx = RenderCtx::new();
        render_node(&body, Rect::new(0.0, 0.0, 100.0, 24.0), &mut ctx);
        // Just ensure rendering didn't panic.
        assert!(!ctx.commands.is_empty());
    }

    // -------------------------------------------------------------------------
    // Picker
    // -------------------------------------------------------------------------

    #[test]
    fn picker_options_stored() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(1);
        let p = Picker::new(&sig, "Mode").option("One", 1).option("Two", 2);
        assert_eq!(p.options.len(), 2);
    }

    #[test]
    fn picker_tap_cycles_through_options() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(1);
        let p = Picker::new(&sig, "Mode")
            .option("One", 1)
            .option("Two", 2)
            .option("Three", 3);
        let body = p.body();
        fn find_tap(n: &ViewNode) -> Option<Rc<dyn Fn()>> {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(f),
                    ..
                } => Some(Rc::clone(f)),
                ViewNode::Modified { child, .. } => find_tap(child),
                ViewNode::Container { children, .. } => children.iter().find_map(find_tap),
                _ => None,
            }
        }
        let action = find_tap(&body).expect("picker has tap");
        action();
        assert_eq!(sig.get(), 2);
        action();
        assert_eq!(sig.get(), 3);
        action();
        assert_eq!(sig.get(), 1);
    }

    #[test]
    fn picker_with_no_options_does_not_panic_on_tap() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(1);
        let p: Picker<i32> = Picker::new(&sig, "Empty");
        let body = p.body();
        fn find_tap(n: &ViewNode) -> Option<Rc<dyn Fn()>> {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(f),
                    ..
                } => Some(Rc::clone(f)),
                ViewNode::Modified { child, .. } => find_tap(child),
                ViewNode::Container { children, .. } => children.iter().find_map(find_tap),
                _ => None,
            }
        }
        find_tap(&body).expect("tap target")();
    }

    // -------------------------------------------------------------------------
    // ScrollView
    // -------------------------------------------------------------------------

    #[test]
    fn scroll_view_vertical_constructor() {
        let s = ScrollView::vertical(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
        assert_eq!(s.axes, ScrollAxis::Vertical);
        assert!(s.show_indicators);
    }

    #[test]
    fn scroll_view_horizontal_and_both() {
        let h = ScrollView::horizontal(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
        let b = ScrollView::both(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
        assert_eq!(h.axes, ScrollAxis::Horizontal);
        assert_eq!(b.axes, ScrollAxis::Both);
    }

    #[test]
    fn scroll_view_show_indicators_chains() {
        let s = ScrollView::vertical(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }))
            .show_indicators(false);
        assert!(!s.show_indicators);
    }

    #[test]
    fn scroll_view_body_is_modified_frame() {
        let s = ScrollView::vertical(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x"))));
        match s.body() {
            ViewNode::Modified {
                modifier: Modifier::Frame { .. },
                ..
            } => {}
            other => panic!("expected frame, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // List
    // -------------------------------------------------------------------------

    #[test]
    fn list_default_is_plain_and_empty() {
        let l = List::new();
        assert_eq!(l.style, ListStyle::Plain);
        assert!(l.items.is_empty());
        assert!(l.selection.is_none());
    }

    #[test]
    fn list_item_appends() {
        let l = List::new()
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))));
        assert_eq!(l.items.len(), 2);
    }

    #[test]
    fn list_body_inserts_dividers_between_rows() {
        let l = List::new()
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("c"))));
        match l.body() {
            ViewNode::Container { children, .. } => {
                // 3 items + 2 dividers
                assert_eq!(children.len(), 5);
            }
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn list_selection_emits_on_tap_and_stores_index() {
        let arena = SignalArena::new();
        let sel = arena.create::<usize>(0);
        let l = List::new()
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))))
            .selection(&sel);
        // Find the second row's OnTap and invoke it.
        let body = l.body();
        let ViewNode::Container { children, .. } = body else {
            panic!("expected container");
        };
        // children[2] is the second row (index 0 = row a, 1 = divider, 2 = row b).
        fn find_tap(n: &ViewNode) -> Option<Rc<dyn Fn()>> {
            match n {
                ViewNode::Modified {
                    modifier: Modifier::OnTap(f),
                    ..
                } => Some(Rc::clone(f)),
                ViewNode::Modified { child, .. } => find_tap(child),
                _ => None,
            }
        }
        let action = find_tap(&children[2]).expect("on_tap on second row");
        action();
        assert_eq!(sel.get(), 1);
    }

    #[test]
    fn list_inset_style_pads_rows() {
        let l = List::new()
            .style(ListStyle::Inset)
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x"))));
        assert_eq!(l.style, ListStyle::Inset);
        // body should produce padding around the row
        let body = l.body();
        let ViewNode::Container { children, .. } = body else {
            panic!("expected container");
        };
        match &children[0] {
            ViewNode::Modified {
                modifier: Modifier::Padding(e),
                ..
            } => {
                assert!(e.left > 0.0);
            }
            other => panic!("expected padding, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // LazyVStack / LazyHStack
    // -------------------------------------------------------------------------

    #[test]
    fn lazy_vstack_builder_appends_items() {
        let s = LazyVStack::new()
            .spacing(4.0)
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))));
        assert_eq!(s.items.len(), 2);
        assert_eq!(s.spacing, 4.0);
    }

    #[test]
    fn lazy_vstack_body_is_vstack_container() {
        let s = LazyVStack::new().item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))));
        match s.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 1),
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn lazy_hstack_body_is_container() {
        let s = LazyHStack::new()
            .spacing(2.0)
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("y"))));
        match s.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 2),
            other => panic!("expected container, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Grid / GridLayout / LazyVGrid / LazyHGrid
    // -------------------------------------------------------------------------

    #[test]
    fn grid_item_resolve_fixed() {
        assert_eq!(GridItem::Fixed(50.0).resolve(0.0), 50.0);
    }

    #[test]
    fn grid_item_resolve_flexible_clamps() {
        let g = GridItem::Flexible {
            min: 20.0,
            max: 80.0,
        };
        assert_eq!(g.resolve(10.0), 20.0);
        assert_eq!(g.resolve(50.0), 50.0);
        assert_eq!(g.resolve(200.0), 80.0);
    }

    #[test]
    fn grid_item_resolve_adaptive_floor() {
        let g = GridItem::Adaptive { min: 30.0 };
        assert_eq!(g.resolve(10.0), 30.0);
        assert_eq!(g.resolve(50.0), 50.0);
    }

    #[test]
    fn grid_layout_size_that_fits_with_two_columns() {
        let g = GridLayout::new(vec![GridItem::Fixed(50.0), GridItem::Fixed(50.0)]);
        let subs = vec![
            Measured::fixed(Size::new(50.0, 20.0)),
            Measured::fixed(Size::new(50.0, 30.0)),
        ];
        let s = g.size_that_fits(ProposedSize::fixed(100.0, 0.0), &subs);
        assert_eq!(s.width, 100.0);
        assert_eq!(s.height, 30.0);
    }

    #[test]
    fn grid_layout_places_in_row_major_order() {
        let g = GridLayout::new(vec![GridItem::Fixed(50.0), GridItem::Fixed(50.0)]);
        let mut ps = vec![
            Placement::new(Point::default(), Size::new(50.0, 10.0)),
            Placement::new(Point::default(), Size::new(50.0, 10.0)),
            Placement::new(Point::default(), Size::new(50.0, 10.0)),
        ];
        g.place_subviews(
            Rect::new(0.0, 0.0, 100.0, 40.0),
            ProposedSize::fixed(100.0, 40.0),
            &mut ps,
        );
        // First row: x=0, x=50 (y=0). Second row: x=0 (y=10).
        assert_eq!(ps[0].origin.x, 0.0);
        assert_eq!(ps[1].origin.x, 50.0);
        assert!(ps[2].origin.y > ps[0].origin.y);
    }

    #[test]
    fn grid_layout_clone_box_round_trips() {
        let g = GridLayout::new(vec![GridItem::Fixed(10.0)]);
        let b = g.clone_box();
        let s = b.size_that_fits(
            ProposedSize::fixed(10.0, 0.0),
            &[Measured::fixed(Size::new(10.0, 5.0))],
        );
        assert_eq!(s.width, 10.0);
    }

    #[test]
    fn grid_with_three_rows_two_columns_emits_six_cells() {
        let g = Grid::new()
            .row(GridRow::new(vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))),
            ]))
            .row(GridRow::new(vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("c"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("d"))),
            ]))
            .row(GridRow::new(vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("e"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("f"))),
            ]));
        match g.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 6),
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn grid_pads_short_rows_with_spacers() {
        let g = Grid::new()
            .row(GridRow::new(vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))),
            ]))
            .row(GridRow::new(vec![ViewNode::Leaf(LeafKind::Text(
                TextLeaf::new("c"),
            ))]));
        match g.body() {
            ViewNode::Container { children, .. } => {
                assert_eq!(children.len(), 4);
                // Last cell should be a spacer.
                assert!(matches!(
                    children[3],
                    ViewNode::Leaf(LeafKind::Spacer { .. })
                ));
            }
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn grid_empty_body_is_spacer() {
        let g = Grid::new();
        assert!(matches!(g.body(), ViewNode::Leaf(LeafKind::Spacer { .. })));
    }

    #[test]
    fn lazy_vgrid_lowers_to_grid_layout() {
        let g = LazyVGrid::new(vec![GridItem::Fixed(40.0), GridItem::Fixed(40.0)])
            .spacing(4.0)
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("x"))));
        match g.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 1),
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn lazy_hgrid_lowers_to_grid_layout() {
        let g = LazyHGrid::new(vec![GridItem::Fixed(40.0)])
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))));
        match g.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 2),
            other => panic!("expected container, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Image
    // -------------------------------------------------------------------------

    #[test]
    fn image_constructor_defaults() {
        let i = Image::new("foo.png");
        assert_eq!(i.url, "foo.png");
        assert_eq!(i.fit, ContentFit::Contain);
    }

    #[test]
    fn image_fit_chains() {
        let i = Image::new("foo.png").fit(ContentFit::Cover);
        assert_eq!(i.fit, ContentFit::Cover);
    }

    #[test]
    fn image_placeholder_color_chains() {
        let i = Image::new("foo.png").placeholder(Color::rgb(0.5, 0.5, 0.5));
        assert_eq!(i.placeholder_color.r, 0.5);
    }

    #[test]
    fn image_body_has_background_modifier() {
        let i = Image::new("x.png").placeholder(Color::rgb(0.1, 0.1, 0.1));
        match i.body() {
            ViewNode::Modified {
                modifier: Modifier::Background(BackgroundFill::Color(c)),
                ..
            } => {
                assert_eq!(c.r, 0.1);
            }
            other => panic!("expected background, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Canvas
    // -------------------------------------------------------------------------

    #[test]
    fn canvas_body_is_canvas_leaf() {
        let c = Canvas::new(|_ctx, _size| {});
        assert!(matches!(c.body(), ViewNode::Leaf(LeafKind::Canvas(_))));
    }

    #[test]
    fn canvas_draw_runs_through_render_node() {
        let counter = Rc::new(Cell::new(0));
        let cnt = counter.clone();
        let c = Canvas::new(move |_ctx, _size| {
            cnt.set(cnt.get() + 1);
        });
        let body = c.body();
        let mut ctx = RenderCtx::new();
        render_node(&body, Rect::new(0.0, 0.0, 10.0, 10.0), &mut ctx);
        assert_eq!(counter.get(), 1);
    }

    // -------------------------------------------------------------------------
    // NavigationStack
    // -------------------------------------------------------------------------

    #[test]
    fn navigation_stack_renders_root() {
        let arena = SignalArena::new();
        let path = arena.create::<Vec<String>>(Vec::new());
        let n = NavigationStack::new(&path, ViewNode::Leaf(LeafKind::Text(TextLeaf::new("home"))));
        match n.body() {
            ViewNode::Leaf(LeafKind::Text(t)) => assert_eq!(t.content, "home"),
            other => panic!("expected text, got {:?}", other),
        }
    }

    #[test]
    fn navigation_stack_path_is_cloned() {
        let arena = SignalArena::new();
        let path = arena.create::<Vec<String>>(vec!["a".into()]);
        let n = NavigationStack::new(&path, ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }));
        path.update(|p| p.push("b".into()));
        assert_eq!(n.path.get().len(), 2);
    }

    // -------------------------------------------------------------------------
    // Sheet
    // -------------------------------------------------------------------------

    #[test]
    fn sheet_when_dismissed_renders_empty_container() {
        let arena = SignalArena::new();
        let presented = arena.create::<bool>(false);
        let s = Sheet::new(
            &presented,
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("body"))),
        );
        match s.body() {
            ViewNode::Container { children, .. } => assert!(children.is_empty()),
            other => panic!("expected empty container, got {:?}", other),
        }
    }

    #[test]
    fn sheet_when_presented_emits_two_children() {
        let arena = SignalArena::new();
        let presented = arena.create::<bool>(true);
        let s = Sheet::new(
            &presented,
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("body"))),
        );
        match s.body() {
            ViewNode::Container { children, .. } => {
                // dim + card
                assert_eq!(children.len(), 2);
            }
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn sheet_toggles_with_signal() {
        let arena = SignalArena::new();
        let presented = arena.create::<bool>(false);
        let s = Sheet::new(
            &presented,
            ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }),
        );
        // Initially empty.
        if let ViewNode::Container { children, .. } = s.body() {
            assert!(children.is_empty());
        } else {
            panic!("expected container");
        }
        presented.set(true);
        if let ViewNode::Container { children, .. } = s.body() {
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected container");
        }
    }

    // -------------------------------------------------------------------------
    // Alert
    // -------------------------------------------------------------------------

    #[test]
    fn alert_when_dismissed_is_empty() {
        let arena = SignalArena::new();
        let p = arena.create::<bool>(false);
        let a = Alert::new(&p, "Oops", "Something happened");
        if let ViewNode::Container { children, .. } = a.body() {
            assert!(children.is_empty());
        } else {
            panic!("expected container");
        }
    }

    #[test]
    fn alert_action_builder_chains() {
        let a = AlertAction::new("OK", || {}).style(AlertActionStyle::Cancel);
        assert_eq!(a.label, "OK");
        assert_eq!(a.style, AlertActionStyle::Cancel);
    }

    #[test]
    fn alert_with_two_actions_renders_action_row() {
        let arena = SignalArena::new();
        let p = arena.create::<bool>(true);
        let a = Alert::new(&p, "Delete?", "This cannot be undone")
            .action(AlertAction::new("Cancel", || {}).style(AlertActionStyle::Cancel))
            .action(AlertAction::new("Delete", || {}).style(AlertActionStyle::Destructive));
        assert_eq!(a.actions.len(), 2);
        // body should contain dim + card; card holds title+message+actions row.
        match a.body() {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 2),
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn alert_action_tap_invokes_handler() {
        let counter = Rc::new(Cell::new(0));
        let c = counter.clone();
        let a = AlertAction::new("OK", move || {
            c.set(c.get() + 1);
        });
        (a.on_tap)();
        (a.on_tap)();
        assert_eq!(counter.get(), 2);
    }

    // -------------------------------------------------------------------------
    // PeerApp
    // -------------------------------------------------------------------------

    #[test]
    fn peer_app_default_route_is_empty() {
        let p = PeerApp::new("dx@nova.local", "calc");
        assert_eq!(p.route, "");
    }

    #[test]
    fn peer_app_route_chains() {
        let p = PeerApp::new("dx@nova.local", "calc").route("/sin");
        assert_eq!(p.route, "/sin");
    }

    #[test]
    fn peer_app_body_is_canvas_leaf() {
        let p = PeerApp::new("addr", "id").route("/r");
        assert!(matches!(p.body(), ViewNode::Leaf(LeafKind::Canvas(_))));
    }

    #[test]
    fn peer_app_renders_through_render_node() {
        let p = PeerApp::new("dx", "calc").route("/r");
        let body = p.body();
        let mut ctx = RenderCtx::new();
        render_node(&body, Rect::new(0.0, 0.0, 200.0, 60.0), &mut ctx);
        // Canvas pushes/pops a clip and emits at least one fill + text.
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert!(texts >= 1);
    }

    // -------------------------------------------------------------------------
    // MediaSession
    // -------------------------------------------------------------------------

    #[test]
    fn media_session_constructor_fields() {
        let m = MediaSession::new("Title", "Artist", "Album");
        assert_eq!(m.title, "Title");
        assert!(m.artwork_url.is_none());
        assert!(m.controls.is_empty());
    }

    #[test]
    fn media_session_artwork_chains() {
        let m = MediaSession::new("t", "a", "al").artwork("art.png");
        assert_eq!(m.artwork_url.as_deref(), Some("art.png"));
    }

    #[test]
    fn media_session_control_glyphs_cover_all_variants() {
        assert_eq!(MediaControl::Play.glyph(), "Play");
        assert_eq!(MediaControl::Pause.glyph(), "Pause");
        assert_eq!(MediaControl::Next.glyph(), "Next");
        assert_eq!(MediaControl::Previous.glyph(), "Prev");
        assert_eq!(MediaControl::SeekForward.glyph(), ">>");
        assert_eq!(MediaControl::SeekBackward.glyph(), "<<");
    }

    #[test]
    fn media_session_with_artwork_includes_image_in_row() {
        let m = MediaSession::new("t", "a", "al")
            .artwork("a.png")
            .control(MediaControl::Play);
        let body = m.body();
        let mut found_image = false;
        fn walk(n: &ViewNode, found: &mut bool) {
            match n {
                ViewNode::Leaf(LeafKind::Image(_)) => *found = true,
                ViewNode::Modified { child, .. } => walk(child, found),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, found);
                    }
                }
                _ => {}
            }
        }
        walk(&body, &mut found_image);
        assert!(found_image);
    }

    #[test]
    fn media_session_without_artwork_omits_image() {
        let m = MediaSession::new("t", "a", "al")
            .control(MediaControl::Play)
            .control(MediaControl::Next);
        let body = m.body();
        let mut found_image = false;
        fn walk(n: &ViewNode, found: &mut bool) {
            match n {
                ViewNode::Leaf(LeafKind::Image(_)) => *found = true,
                ViewNode::Modified { child, .. } => walk(child, found),
                ViewNode::Container { children, .. } => {
                    for c in children {
                        walk(c, found);
                    }
                }
                _ => {}
            }
        }
        walk(&body, &mut found_image);
        assert!(!found_image);
    }

    // -------------------------------------------------------------------------
    // Cross-cutting: render_node end-to-end smoke tests
    // -------------------------------------------------------------------------

    #[test]
    fn button_renders_text_through_render_node() {
        let b = Button::new("Submit").style(ButtonStyle::Bordered);
        let mut ctx = RenderCtx::new();
        render_node(&b.body(), Rect::new(0.0, 0.0, 120.0, 32.0), &mut ctx);
        let text_count = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(text_count, 1);
    }

    #[test]
    fn list_renders_dividers_and_rows() {
        let l = List::new()
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))))
            .item(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))));
        let mut ctx = RenderCtx::new();
        render_node(&l.body(), Rect::new(0.0, 0.0, 200.0, 60.0), &mut ctx);
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        let fills = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::FillRect { .. }))
            .count();
        assert_eq!(texts, 2);
        assert_eq!(fills, 1); // single divider between two rows
    }

    #[test]
    fn grid_renders_through_render_node() {
        let g = Grid::new().row(GridRow::new(vec![
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))),
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))),
        ]));
        let mut ctx = RenderCtx::new();
        render_node(&g.body(), Rect::new(0.0, 0.0, 100.0, 30.0), &mut ctx);
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(texts, 2);
    }

    #[test]
    fn scroll_view_propagates_child_render() {
        let s = ScrollView::vertical(ViewNode::Leaf(LeafKind::Text(TextLeaf::new("hello"))));
        let mut ctx = RenderCtx::new();
        render_node(&s.body(), Rect::new(0.0, 0.0, 100.0, 100.0), &mut ctx);
        let texts = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(texts, 1);
    }

    #[test]
    fn navigation_stack_renders_through_render_node() {
        let arena = SignalArena::new();
        let path = arena.create::<Vec<String>>(Vec::new());
        let n = NavigationStack::new(&path, ViewNode::Leaf(LeafKind::Text(TextLeaf::new("home"))));
        let mut ctx = RenderCtx::new();
        render_node(&n.body(), Rect::new(0.0, 0.0, 100.0, 20.0), &mut ctx);
        assert_eq!(ctx.commands.len(), 1);
    }

    // -------------------------------------------------------------------------
    // ViewNode::clone_node coverage (introduced for primitives).
    // -------------------------------------------------------------------------

    #[test]
    fn clone_node_preserves_leaf_text() {
        let n = ViewNode::Leaf(LeafKind::Text(TextLeaf::new("hi")));
        let c = n.clone_node();
        match c {
            ViewNode::Leaf(LeafKind::Text(t)) => assert_eq!(t.content, "hi"),
            other => panic!("expected text leaf, got {:?}", other),
        }
    }

    #[test]
    fn clone_node_preserves_container_layout_and_children() {
        let n = ViewNode::Container {
            layout: Box::new(LHStack::new().spacing(4.0)),
            children: vec![
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("a"))),
                ViewNode::Leaf(LeafKind::Text(TextLeaf::new("b"))),
            ],
        };
        let c = n.clone_node();
        match c {
            ViewNode::Container { children, .. } => assert_eq!(children.len(), 2),
            other => panic!("expected container, got {:?}", other),
        }
    }

    #[test]
    fn clone_node_preserves_modifier_chain() {
        let n = ViewNode::Modified {
            modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
            child: Box::new(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })),
        };
        let c = n.clone_node();
        match c {
            ViewNode::Modified {
                modifier: Modifier::Background(_),
                ..
            } => {}
            other => panic!("expected background modifier, got {:?}", other),
        }
    }

    #[test]
    fn clone_node_shares_stateful_closure() {
        let counter = Rc::new(Cell::new(0));
        let cnt = counter.clone();
        let n = ViewNode::Stateful(Rc::new(move || {
            cnt.set(cnt.get() + 1);
            ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })
        }));
        let c = n.clone_node();
        if let ViewNode::Stateful(f) = &c {
            let _ = f();
        }
        if let ViewNode::Stateful(f) = &n {
            let _ = f();
        }
        assert_eq!(counter.get(), 2);
    }
}

// Last updated: 2026-05-01
