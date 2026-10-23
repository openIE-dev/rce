//! ViewExt support — modifier-wrap helper plus blanket conversions used by
//! the modifier chain on [`crate::view::ViewExt`].
//!
//! The trait itself lives in `view.rs` to keep the canonical `ViewExt` symbol
//! co-located with the [`View`] trait. This module owns the small pieces of
//! glue (the `wrap` helper, `From` impls for [`BackgroundFill`]) so the
//! `view.rs` file isn't drowned in boilerplate.

use crate::color::Color;
use crate::view::{BackgroundFill, Modifier, View, ViewNode};

/// Wrap any view in a [`Modifier`], returning a `ViewNode::Modified` tree
/// rooted at the wrapping modifier and the view's `body()` as the child.
///
/// All [`crate::view::ViewExt`] modifier methods route through this helper so
/// the wrapping shape is uniform across every variant.
pub fn wrap<V: View>(view: V, modifier: Modifier) -> ViewNode {
    ViewNode::Modified {
        modifier,
        child: Box::new(view.body()),
    }
}

/// Build a linear-gradient [`BackgroundFill`] from a list of stops + angle.
/// The pair `(stops, angle_deg)` makes the call site read like
/// `linear_gradient(vec![(0.0, RED), (1.0, BLUE)], 0.0)`.
pub fn linear_gradient(stops: Vec<(f32, Color)>, angle_deg: f32) -> BackgroundFill {
    BackgroundFill::LinearGradient { angle_deg, stops }
}

/// Build a radial-gradient [`BackgroundFill`] centred at `center`.
pub fn radial_gradient(
    stops: Vec<(f32, Color)>,
    center: crate::layout::Alignment2D,
) -> BackgroundFill {
    BackgroundFill::RadialGradient { center, stops }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Alignment2D, EdgeInsets, HAlignment, VAlignment};
    use crate::view::{
        AccessibilityRole, ClipShape, Edge, LeafKind, TextLeaf, TransitionKind, ViewExt,
    };

    struct Dummy;
    impl View for Dummy {
        fn layout(&self, _c: &crate::layout::Constraint) -> crate::layout::Size {
            crate::layout::Size::zero()
        }
        fn render(&self, _r: &crate::layout::LayoutRect, _ctx: &mut crate::render::RenderCtx) {}
        fn body(&self) -> ViewNode {
            ViewNode::Leaf(LeafKind::Text(TextLeaf::new("dummy")))
        }
    }

    fn modifier_of(node: &ViewNode) -> &Modifier {
        match node {
            ViewNode::Modified { modifier, .. } => modifier,
            _ => panic!("expected modified node"),
        }
    }

    fn child_of(node: &ViewNode) -> &ViewNode {
        match node {
            ViewNode::Modified { child, .. } => child,
            _ => panic!("expected modified node"),
        }
    }

    #[test]
    fn wrap_produces_modified_node() {
        let n = wrap(Dummy, Modifier::Opacity(0.5));
        assert!(matches!(
            n,
            ViewNode::Modified {
                modifier: Modifier::Opacity(_),
                ..
            }
        ));
    }

    #[test]
    fn linear_gradient_stores_stops_and_angle() {
        let g = linear_gradient(vec![(0.0, Color::WHITE), (1.0, Color::BLACK)], 45.0);
        match g {
            BackgroundFill::LinearGradient { angle_deg, stops } => {
                assert_eq!(angle_deg, 45.0);
                assert_eq!(stops.len(), 2);
            }
            _ => panic!("expected linear gradient"),
        }
    }

    #[test]
    fn radial_gradient_stores_center() {
        let g = radial_gradient(
            vec![(0.0, Color::WHITE)],
            crate::layout::Alignment2D::default(),
        );
        assert!(matches!(g, BackgroundFill::RadialGradient { .. }));
    }

    // -------------------------------------------------------------------------
    // ViewExt — every chain method should produce a Modified node carrying
    // the corresponding Modifier variant.
    // -------------------------------------------------------------------------

    #[test]
    fn ext_padding_wraps_with_padding_modifier() {
        let n = Dummy.padding(EdgeInsets::all(8.0));
        assert!(matches!(modifier_of(&n), Modifier::Padding(e) if e.top == 8.0));
    }

    #[test]
    fn ext_padding_all_uniform() {
        let n = Dummy.padding_all(4.0);
        assert!(matches!(modifier_of(&n), Modifier::Padding(e) if e.left == 4.0 && e.right == 4.0));
    }

    #[test]
    fn ext_padding_xy_split() {
        let n = Dummy.padding_xy(2.0, 6.0);
        let Modifier::Padding(e) = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(e.left, 2.0);
        assert_eq!(e.top, 6.0);
    }

    #[test]
    fn ext_padding_top_only() {
        let n = Dummy.padding_top(3.0);
        let Modifier::Padding(e) = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(e.top, 3.0);
        assert_eq!(e.bottom, 0.0);
    }

    #[test]
    fn ext_padding_right_only() {
        let n = Dummy.padding_right(5.0);
        let Modifier::Padding(e) = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(e.right, 5.0);
        assert_eq!(e.left, 0.0);
    }

    #[test]
    fn ext_padding_bottom_only() {
        let n = Dummy.padding_bottom(7.0);
        let Modifier::Padding(e) = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(e.bottom, 7.0);
    }

    #[test]
    fn ext_padding_left_only() {
        let n = Dummy.padding_left(9.0);
        let Modifier::Padding(e) = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(e.left, 9.0);
    }

    #[test]
    fn ext_background_color_via_into() {
        let n = Dummy.background(Color::WHITE);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Background(BackgroundFill::Color(c)) if *c == Color::WHITE
        ));
    }

    #[test]
    fn ext_background_color_helper() {
        let n = Dummy.background_color(Color::BLACK);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Background(BackgroundFill::Color(c)) if *c == Color::BLACK
        ));
    }

    #[test]
    fn ext_background_gradient_via_into() {
        let g = linear_gradient(vec![(0.0, Color::WHITE), (1.0, Color::BLACK)], 90.0);
        let n = Dummy.background(g);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Background(BackgroundFill::LinearGradient { .. })
        ));
    }

    #[test]
    fn ext_foreground_wraps() {
        let n = Dummy.foreground(Color::WHITE);
        assert!(matches!(modifier_of(&n), Modifier::Foreground(c) if *c == Color::WHITE));
    }

    #[test]
    fn ext_opacity_wraps() {
        let n = Dummy.opacity(0.25);
        assert!(matches!(modifier_of(&n), Modifier::Opacity(v) if (*v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn ext_corner_radius_wraps() {
        let n = Dummy.corner_radius(6.0);
        assert!(matches!(modifier_of(&n), Modifier::CornerRadius(r) if (*r - 6.0).abs() < 1e-6));
    }

    #[test]
    fn ext_border_wraps() {
        let n = Dummy.border(2.0, Color::WHITE);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Border { width, color } if (*width - 2.0).abs() < 1e-6 && *color == Color::WHITE
        ));
    }

    #[test]
    fn ext_shadow_wraps_with_offsets() {
        let n = Dummy.shadow(Color::BLACK, 4.0, 1.0, 2.0);
        let Modifier::Shadow {
            offset_x, offset_y, ..
        } = modifier_of(&n)
        else {
            panic!()
        };
        assert_eq!(*offset_x, 1.0);
        assert_eq!(*offset_y, 2.0);
    }

    #[test]
    fn ext_inner_shadow_wraps() {
        let n = Dummy.inner_shadow(Color::WHITE, 2.0, 0.0, 0.0);
        assert!(matches!(modifier_of(&n), Modifier::InnerShadow { .. }));
    }

    #[test]
    fn ext_frame_wraps() {
        let n = Dummy.frame(Some(100.0), None);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Frame {
                width: Some(100.0),
                height: None
            }
        ));
    }

    #[test]
    fn ext_frame_aligned_wraps() {
        let n = Dummy.frame_aligned(
            Some(50.0),
            Some(20.0),
            Alignment2D::new(HAlignment::Trailing, VAlignment::Bottom),
        );
        assert!(matches!(modifier_of(&n), Modifier::FrameAligned { .. }));
    }

    #[test]
    fn ext_frame_range_wraps() {
        let n = Dummy.frame_range(
            Some(10.0),
            Some(30.0),
            Some(50.0),
            None,
            None,
            None,
            Alignment2D::default(),
        );
        let Modifier::FrameRange {
            min_width,
            ideal_width,
            max_width,
            ..
        } = modifier_of(&n)
        else {
            panic!("expected frame range");
        };
        assert_eq!(*min_width, Some(10.0));
        assert_eq!(*ideal_width, Some(30.0));
        assert_eq!(*max_width, Some(50.0));
    }

    #[test]
    fn ext_rotation_wraps() {
        let n = Dummy.rotation(45.0);
        assert!(matches!(modifier_of(&n), Modifier::Rotation(d) if (*d - 45.0).abs() < 1e-6));
    }

    #[test]
    fn ext_scale_uniform_wraps() {
        let n = Dummy.scale(2.0);
        let Modifier::Scale { x, y, .. } = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(*x, 2.0);
        assert_eq!(*y, 2.0);
    }

    #[test]
    fn ext_scale_xy_wraps() {
        let n = Dummy.scale_xy(2.0, 3.0, Alignment2D::default());
        let Modifier::Scale { x, y, .. } = modifier_of(&n) else {
            panic!()
        };
        assert_eq!(*x, 2.0);
        assert_eq!(*y, 3.0);
    }

    #[test]
    fn ext_offset_wraps() {
        let n = Dummy.offset(5.0, 6.0);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Offset { x: 5.0, y: 6.0 }
        ));
    }

    #[test]
    fn ext_position_wraps() {
        let n = Dummy.position(10.0, 20.0);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Position { x: 10.0, y: 20.0 }
        ));
    }

    #[test]
    fn ext_clip_wraps_with_shape() {
        let n = Dummy.clip(ClipShape::Circle);
        assert!(matches!(modifier_of(&n), Modifier::Clip(ClipShape::Circle)));
    }

    #[test]
    fn ext_clip_rect_helper() {
        let n = Dummy.clip_rect();
        assert!(matches!(modifier_of(&n), Modifier::Clip(ClipShape::Rect)));
    }

    #[test]
    fn ext_clip_capsule_helper() {
        let n = Dummy.clip_capsule();
        assert!(matches!(
            modifier_of(&n),
            Modifier::Clip(ClipShape::Capsule)
        ));
    }

    #[test]
    fn ext_clip_circle_helper() {
        let n = Dummy.clip_circle();
        assert!(matches!(modifier_of(&n), Modifier::Clip(ClipShape::Circle)));
    }

    #[test]
    fn ext_mask_wraps() {
        let n = Dummy.mask(ClipShape::Ellipse);
        assert!(matches!(
            modifier_of(&n),
            Modifier::Mask(ClipShape::Ellipse)
        ));
    }

    #[test]
    fn ext_on_tap_wraps() {
        let n = Dummy.on_tap(|| {});
        assert!(matches!(modifier_of(&n), Modifier::OnTap(_)));
    }

    #[test]
    fn ext_on_hover_wraps() {
        let n = Dummy.on_hover(|_| {});
        assert!(matches!(modifier_of(&n), Modifier::OnHover(_)));
    }

    #[test]
    fn ext_on_long_press_wraps() {
        let n = Dummy.on_long_press(500, || {});
        assert!(matches!(
            modifier_of(&n),
            Modifier::OnLongPress {
                duration_ms: 500,
                ..
            }
        ));
    }

    #[test]
    fn ext_on_drag_wraps() {
        let n = Dummy.on_drag(|_, _| {});
        assert!(matches!(modifier_of(&n), Modifier::OnDrag(_)));
    }

    #[test]
    fn ext_on_appear_wraps() {
        let n = Dummy.on_appear(|| {});
        assert!(matches!(modifier_of(&n), Modifier::OnAppear(_)));
    }

    #[test]
    fn ext_on_disappear_wraps() {
        let n = Dummy.on_disappear(|| {});
        assert!(matches!(modifier_of(&n), Modifier::OnDisappear(_)));
    }

    #[test]
    fn ext_accessibility_label_wraps() {
        let n = Dummy.accessibility_label("Greeting");
        assert!(matches!(
            modifier_of(&n),
            Modifier::AccessibilityLabel(s) if s == "Greeting"
        ));
    }

    #[test]
    fn ext_accessibility_role_wraps() {
        let n = Dummy.accessibility_role(AccessibilityRole::Heading(2));
        assert!(matches!(
            modifier_of(&n),
            Modifier::AccessibilityRole(AccessibilityRole::Heading(2))
        ));
    }

    #[test]
    fn ext_accessibility_hint_wraps() {
        let n = Dummy.accessibility_hint("activate");
        assert!(matches!(
            modifier_of(&n),
            Modifier::AccessibilityHint(s) if s == "activate"
        ));
    }

    #[test]
    fn ext_accessibility_value_wraps() {
        let n = Dummy.accessibility_value("42");
        assert!(matches!(
            modifier_of(&n),
            Modifier::AccessibilityValue(s) if s == "42"
        ));
    }

    #[test]
    fn ext_accessibility_hidden_wraps() {
        let n = Dummy.accessibility_hidden(true);
        assert!(matches!(
            modifier_of(&n),
            Modifier::AccessibilityHidden(true)
        ));
    }

    #[test]
    fn ext_tab_index_wraps() {
        let n = Dummy.tab_index(2);
        assert!(matches!(modifier_of(&n), Modifier::TabIndex(2)));
    }

    #[test]
    fn ext_allows_hit_testing_wraps() {
        let n = Dummy.allows_hit_testing(false);
        assert!(matches!(modifier_of(&n), Modifier::AllowsHitTesting(false)));
    }

    #[test]
    fn ext_tooltip_wraps() {
        let n = Dummy.tooltip("Help");
        assert!(matches!(modifier_of(&n), Modifier::Tooltip(s) if s == "Help"));
    }

    #[test]
    fn ext_z_index_wraps() {
        let n = Dummy.z_index(3.0);
        assert!(matches!(modifier_of(&n), Modifier::ZIndex(z) if (*z - 3.0).abs() < 1e-6));
    }

    #[test]
    fn ext_animation_wraps() {
        use crate::animation::{AnimationSpec, ease_in_out};
        let spec: AnimationSpec = ease_in_out(150).keyed(7);
        let n = Dummy.animation(spec);
        let Modifier::Animation(got) = modifier_of(&n) else {
            panic!("expected animation modifier");
        };
        assert_eq!(got.duration_ms, 150);
        assert_eq!(got.key, 7);
    }

    #[test]
    fn ext_transition_wraps() {
        let n = Dummy.transition(TransitionKind::Slide { edge: Edge::Top });
        assert!(matches!(
            modifier_of(&n),
            Modifier::Transition(TransitionKind::Slide { edge: Edge::Top })
        ));
    }

    #[test]
    fn ext_tag_wraps() {
        let n = Dummy.tag("primary");
        assert!(matches!(modifier_of(&n), Modifier::Tag(s) if s == "primary"));
    }

    #[test]
    fn ext_hidden_wraps() {
        let n = Dummy.hidden(true);
        assert!(matches!(modifier_of(&n), Modifier::Hidden(true)));
    }

    // -------------------------------------------------------------------------
    // Chain composition — multiple modifiers stack outwards in call order.
    // -------------------------------------------------------------------------

    #[test]
    fn chain_padding_then_background_then_on_tap_nests_correctly() {
        // .padding(...).background(...).on_tap(...)
        // Innermost wraps first → on_tap is OUTERMOST.
        let n = Dummy
            .padding(EdgeInsets::all(8.0))
            .background(Color::WHITE)
            .on_tap(|| {});
        // Outer = OnTap.
        assert!(matches!(modifier_of(&n), Modifier::OnTap(_)));
        // Middle = Background.
        let mid = child_of(&n);
        assert!(matches!(modifier_of(mid), Modifier::Background(_)));
        // Inner = Padding.
        let inner = child_of(mid);
        assert!(matches!(modifier_of(inner), Modifier::Padding(_)));
    }

    #[test]
    fn chain_three_modifiers_wraps_three_times() {
        let n = Dummy.opacity(0.5).rotation(15.0).offset(2.0, 3.0);
        let mut depth = 0;
        let mut cur = &n;
        while let ViewNode::Modified { child, .. } = cur {
            depth += 1;
            cur = child;
        }
        assert_eq!(depth, 3);
    }

    #[test]
    fn render_root_still_works_through_view_ext() {
        // ViewExt has render_root for ?Sized views — ensure it's still
        // accessible when the new sized methods are also present.
        let mut ctx = crate::render::RenderCtx::new();
        Dummy.render_root(&mut ctx);
        // Dummy renders nothing, so ctx should be empty — this confirms the
        // method is dispatched without panic.
        assert!(ctx.commands.is_empty());
    }
}

// Last updated: 2026-05-01
