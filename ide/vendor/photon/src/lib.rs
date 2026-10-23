//! photon — Nova's declarative GPU UI framework.
//!
//! Photon is a SwiftUI-style declarative UI library backed by Nova's GPU
//! display-list pipeline. This crate ships two layered surfaces:
//!
//! 1. **Legacy widget builder** ([`widgets`]) — fluent-builder structs
//!    (`text`, `h_stack`, `v_stack`, `container`, etc.) that implement the
//!    [`view::View`] trait with explicit `layout`/`render` methods. This is
//!    what `nova_shell` chrome and perf-dashboard rendering currently
//!    target.
//!
//! 2. **Declarative tree** ([`view::ViewNode`]) — an enum-dispatched node
//!    tree that the new render walker [`render::render_node`] processes
//!    using the SwiftUI-style two-pass [`layout::Layout`] protocol. Plus
//!    fine-grained reactive state via [`signal::SignalArena`] /
//!    [`signal::Signal`].
//!
//! New code should prefer the declarative path; the legacy widget path
//! continues to compile so `nova_shell` doesn't need updating.

#![forbid(unsafe_code)]

pub mod animation;
pub mod color;
pub mod layout;
pub mod nwp;
pub mod primitives;
pub mod reactive;
pub mod render;
pub mod signal;
pub mod view;
pub mod view_ext;
pub mod widgets;

pub use animation::{
    Animatable, AnimationCurve, AnimationDriver, AnimationSpec, Edge, Presence, TransitionKind,
    bouncy, ease_in, ease_in_out, ease_out, linear, snappy, spring,
};
pub use color::Color;
pub use layout::{
    Alignment2D, Constraint, EdgeInsets, Frame, HAlignment, HStack as LayoutHStack, Layout,
    LayoutRect, Measured, Placement, Point, ProposedSize, Rect, Size, VAlignment,
    VStack as LayoutVStack, ZStack,
};
pub use primitives::{
    Alert, AlertAction, AlertActionStyle, Button, ButtonStyle, Canvas, Grid, GridItem, GridLayout,
    GridRow, Image, LazyHGrid, LazyHStack, LazyVGrid, LazyVStack, List, ListStyle, MediaControl,
    MediaSession, NavigationStack, PeerApp, Picker, ScrollAxis, ScrollView, Sheet, Slider,
    TextField, Toggle,
};
pub use reactive::Runtime;
pub use render::{
    Annotation, ClipShapeKind, DrawCommand, HitKind, HitRegion, LifecycleHook, RenderCtx,
    RenderPrimitive, Transform2D, render_node, render_node_with_driver,
};
pub use signal::{Binding, NodeId, Signal, SignalArena, SignalId};
pub use view::{
    AccessibilityRole, BackgroundFill, CanvasDraw, ClipShape, ContentFit, ImageLeaf, LeafKind,
    Modifier, PathCommand, Style, TextLeaf, View, ViewExt, ViewNode, boxed,
};
pub use view_ext::{linear_gradient, radial_gradient};

/// Run `f` with the supplied animation envelope active on whatever
/// [`SignalArena`] is currently installed via
/// [`SignalArena::with_animation`]. When called outside any active arena
/// scope (i.e. when `current_arena()` is `None`) this is just `f()` — the
/// envelope is silently dropped.
///
/// This is the SwiftUI top-level `withAnimation { ... }` helper.
pub fn with_animation<R>(spec: AnimationSpec, f: impl FnOnce() -> R) -> R {
    match signal::current_arena() {
        Some(arena) => arena.with_animation(spec, f),
        None => f(),
    }
}

// ---------------------------------------------------------------------------
// Chrome helpers — used by `nova_shell::chrome_photon`.
//
// These mirror the surface the audit notes listed but the *real* nova_shell
// code uses its own per-module `TabInfo` / `NavState` types. To keep the
// compile path clean we still expose the helper types here as identical
// re-exports so other call-sites that import `photon::TabInfo` keep working.
// ---------------------------------------------------------------------------

/// Theme used by [`render_chrome`].
#[derive(Debug, Clone)]
pub struct ChromeTheme {
    pub background: Color,
    pub surface: Color,
    pub text: Color,
    pub accent: Color,
}

impl Default for ChromeTheme {
    fn default() -> Self {
        Self {
            background: Color::rgb(0.12, 0.12, 0.15),
            surface: Color::rgb(0.18, 0.18, 0.22),
            text: Color::rgb(0.9, 0.9, 0.92),
            accent: Color::rgb(0.35, 0.55, 0.9),
        }
    }
}

/// Tab metadata accepted by the chrome render helper.
#[derive(Debug, Clone)]
pub struct TabInfo {
    pub title: String,
    pub active: bool,
    pub loading: bool,
}

/// Navigation enable/disable state.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavState {
    pub can_back: bool,
    pub can_forward: bool,
    pub url_is_https: bool,
}

/// Possible chrome regions that can be hit-tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromeHit {
    Tab(usize),
    Back,
    Forward,
    Reload,
    NewTab,
    CloseTab(usize),
    UrlBar,
    Other,
}

/// Render the browser chrome. The full implementation lives in
/// `nova_shell::chrome_photon`; this helper is a fallback used by Nova Apps
/// that want a basic chrome row without re-implementing one themselves.
pub fn render_chrome(
    tabs: &[TabInfo],
    url: &str,
    _focused: bool,
    _nav: NavState,
    _layout: &impl std::any::Any,
    width: f32,
    theme: &ChromeTheme,
) -> Vec<RenderPrimitive> {
    let mut ctx = RenderCtx::new();
    // A flat band for the tab strip plus a short URL bar underneath.
    let chrome_h = 80.0;
    ctx.fill_rect(
        &LayoutRect::new(0.0, 0.0, width, chrome_h),
        theme.background,
    );
    ctx.fill_rect(&LayoutRect::new(0.0, 36.0, width, 44.0), theme.surface);

    let tab_w = if tabs.is_empty() {
        200.0
    } else {
        (width / tabs.len() as f32).clamp(80.0, 200.0)
    };
    for (i, tab) in tabs.iter().enumerate() {
        let x = i as f32 * tab_w;
        let bg = if tab.active {
            theme.surface
        } else {
            theme.background
        };
        ctx.fill_rect(&LayoutRect::new(x, 0.0, tab_w, 36.0), bg);
        ctx.fill_text(
            &LayoutRect::new(x + 8.0, 8.0, tab_w - 16.0, 24.0),
            &tab.title,
            12.0,
            theme.text,
        );
    }

    if !url.is_empty() {
        ctx.fill_text(
            &LayoutRect::new(80.0, 44.0, width - 100.0, 28.0),
            url,
            13.0,
            theme.text,
        );
    }

    ctx.to_primitives()
}

/// Coarse hit-test for the chrome rendered by [`render_chrome`].
pub fn hit_test(
    tabs: &[TabInfo],
    width: f32,
    x: f32,
    y: f32,
    tab_bar_height: f32,
    toolbar_height: f32,
) -> Option<ChromeHit> {
    if y < 0.0 || x < 0.0 {
        return None;
    }
    if y < tab_bar_height && !tabs.is_empty() {
        let tab_w = (width / tabs.len() as f32).clamp(80.0, 200.0);
        let idx = (x / tab_w) as usize;
        if idx < tabs.len() {
            return Some(ChromeHit::Tab(idx));
        }
        return Some(ChromeHit::NewTab);
    } else if y < tab_bar_height + toolbar_height {
        if x < 32.0 {
            return Some(ChromeHit::Back);
        }
        if x < 64.0 {
            return Some(ChromeHit::Forward);
        }
        if x < 96.0 {
            return Some(ChromeHit::Reload);
        }
        return Some(ChromeHit::UrlBar);
    }
    None
}

/// Performance-dashboard render helper. The fully featured version lives in
/// `nova_shell::perf_photon`; this is the generic placeholder reached by
/// callers that don't want to depend on `nova_shell`.
pub fn render_performance_photon<T>(
    _store: &T,
    width: f32,
    height: f32,
    _section: &mut u32,
) -> Vec<RenderPrimitive> {
    let mut ctx = RenderCtx::new();
    ctx.fill_rect(
        &LayoutRect::new(0.0, 0.0, width, height),
        Color::rgb(0.08, 0.08, 0.10),
    );
    ctx.fill_text(
        &LayoutRect::new(20.0, 20.0, width - 40.0, 28.0),
        "Performance",
        18.0,
        Color::rgb(0.92, 0.92, 0.94),
    );
    ctx.to_primitives()
}

/// Placeholder bridge from Photon primitives to a host display list. The
/// real bridge lives in `nova_shell::perf_photon::primitives_to_display_list`
/// because it depends on `nova_gpu`. This crate's version returns the
/// caller's `Default` to keep the compile path open.
pub fn primitives_to_display_list<DL>(_prims: &[RenderPrimitive]) -> DL
where
    DL: Default,
{
    DL::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_chrome_emits_primitives() {
        let tabs = vec![
            TabInfo {
                title: "A".into(),
                active: true,
                loading: false,
            },
            TabInfo {
                title: "B".into(),
                active: false,
                loading: false,
            },
        ];
        let prims = render_chrome(
            &tabs,
            "https://example.com",
            false,
            NavState::default(),
            &(),
            1280.0,
            &ChromeTheme::default(),
        );
        assert!(!prims.is_empty());
    }

    #[test]
    fn hit_test_tab() {
        let tabs = vec![
            TabInfo {
                title: "A".into(),
                active: true,
                loading: false,
            },
            TabInfo {
                title: "B".into(),
                active: false,
                loading: false,
            },
        ];
        let hit = hit_test(&tabs, 1280.0, 50.0, 10.0, 36.0, 44.0);
        assert_eq!(hit, Some(ChromeHit::Tab(0)));
    }

    #[test]
    fn hit_test_url_bar() {
        let tabs: Vec<TabInfo> = Vec::new();
        let hit = hit_test(&tabs, 1280.0, 400.0, 50.0, 36.0, 44.0);
        assert_eq!(hit, Some(ChromeHit::UrlBar));
    }

    #[test]
    fn hit_test_below_chrome_returns_none() {
        let tabs: Vec<TabInfo> = Vec::new();
        let hit = hit_test(&tabs, 1280.0, 100.0, 200.0, 36.0, 44.0);
        assert!(hit.is_none());
    }

    #[test]
    fn perf_dashboard_emits_primitives() {
        let mut section: u32 = 0;
        let store: u32 = 0;
        let prims = render_performance_photon(&store, 800.0, 600.0, &mut section);
        assert!(!prims.is_empty());
    }

    #[test]
    fn primitives_bridge_returns_default() {
        #[derive(Default, PartialEq, Debug)]
        struct DummyDl(usize);
        let prims: Vec<RenderPrimitive> = Vec::new();
        let dl: DummyDl = primitives_to_display_list(&prims);
        assert_eq!(dl, DummyDl(0));
    }

    // -------------------------------------------------------------------------
    // Phase 1 — declarative path smoke tests
    // -------------------------------------------------------------------------

    #[test]
    fn declarative_pipeline_renders_through_render_node() {
        let arena = SignalArena::new();
        let title = arena.create::<String>("Hello".to_string());
        // Build a stateful node that reads the signal each render.
        let title2 = title.clone();
        let node = ViewNode::Stateful(std::rc::Rc::new(move || ViewNode::Container {
            layout: Box::new(LayoutHStack::new().spacing(0.0)),
            children: vec![ViewNode::Leaf(LeafKind::Text(TextLeaf::new(title2.get())))],
        }));
        let mut ctx = RenderCtx::new();
        render_node(&node, Rect::new(0.0, 0.0, 200.0, 20.0), &mut ctx);
        let text_count = ctx
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::Text { .. }))
            .count();
        assert_eq!(text_count, 1);
    }

    #[test]
    fn signal_arena_top_level_export_is_usable() {
        let arena: std::rc::Rc<SignalArena> = SignalArena::new();
        let sig: Signal<i32> = arena.create(10);
        sig.set(20);
        assert_eq!(sig.get(), 20);
    }

    #[test]
    fn layout_protocol_exports_are_usable() {
        let p = ProposedSize::fixed(100.0, 50.0);
        let h = LayoutHStack::new().spacing(2.0);
        let s = h.size_that_fits(p, &[]);
        assert_eq!(s, Size::zero());
    }

    // -------------------------------------------------------------------------
    // Phase 4 — top-level animation surface
    // -------------------------------------------------------------------------

    #[test]
    fn with_animation_top_level_passes_through_when_no_arena_active() {
        let result = with_animation(linear(100), || 42_i32);
        assert_eq!(result, 42);
    }

    #[test]
    fn with_animation_top_level_dispatches_to_active_arena() {
        let arena = SignalArena::new();
        let sig = arena.create::<i32>(0);
        // Install the arena via with_animation, then call the top-level
        // helper from inside — it should land back on the same arena.
        arena.with_animation(linear(100), || {
            with_animation(ease_in(50), || {
                sig.set(7);
            });
        });
        let drained = arena.drain_dirty_with_animation();
        assert_eq!(drained.len(), 1);
        // Inner with_animation overrides the spec.
        let spec = drained[0].1.unwrap();
        assert!(matches!(spec.curve, AnimationCurve::EaseIn));
    }

    #[test]
    fn animation_driver_re_export_is_usable() {
        let mut d = AnimationDriver::new();
        d.start(0, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        d.tick(50);
        assert!(d.progress(0, 0).is_some());
    }

    #[test]
    fn animatable_re_export_lerps_color() {
        let a = Color::rgb(0.0, 0.0, 0.0);
        let b = Color::rgb(1.0, 1.0, 1.0);
        let mid = Color::lerp(&a, &b, 0.5);
        // Color-managed lerp (linear-sRGB midpoint -> back to sRGB) lands
        // at ~0.7354, not the gamma-space 0.5.
        assert!((mid.r - 0.7354).abs() < 1e-3);
    }

    #[test]
    fn presence_re_export_default_is_stable() {
        let d = AnimationDriver::new();
        assert_eq!(d.presence(0), Presence::Stable);
    }

    #[test]
    fn render_node_with_driver_drives_transition_end_to_end() {
        let arena: std::rc::Rc<SignalArena> = SignalArena::new();
        let visible = arena.create::<bool>(true);
        let visible2 = visible.clone();
        let node = ViewNode::Stateful(std::rc::Rc::new(move || {
            if visible2.get() {
                ViewNode::Modified {
                    modifier: Modifier::Transition(TransitionKind::Opacity),
                    child: Box::new(ViewNode::Modified {
                        modifier: Modifier::Background(BackgroundFill::Color(Color::WHITE)),
                        child: Box::new(ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })),
                    }),
                }
            } else {
                ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 })
            }
        }));
        let mut driver = AnimationDriver::new();
        driver.set_presence(1, Presence::Entering);
        driver.start(1, AnimationSpec::new(AnimationCurve::Linear, 100), 0);
        let mut ctx = RenderCtx::new();
        render_node_with_driver(&node, Rect::new(0.0, 0.0, 10.0, 10.0), &mut ctx, 1, &driver);
        // The transition wraps with PushOpacity at alpha=0 (just-entered).
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
}

// Last updated: 2026-05-01
