//! Color values used throughout Photon.
//!
//! Photon's [`Color`] is the authoring representation. Each colour carries an
//! explicit [`ColorSpace`] tag so the renderer can convert into its blending
//! space (linear sRGB, linear-P3 or scRGB depending on backbuffer choice)
//! before any per-pixel arithmetic. Existing call sites that simply call
//! [`Color::rgb`] / [`Color::rgba`] receive sRGB by default — preserving
//! back-compat with [`Color::WHITE`], [`Color::BLACK`], [`Color::TRANSPARENT`]
//! and every consumer in `chrome_photon` and `perf_photon`.

use serde::{Deserialize, Serialize};

/// Authoring colour space for a [`Color`].
///
/// Mirrors the canonical set used by `nova_gpu::color_management::ColorSpace`,
/// but is duplicated here to keep `photon` zero-dep on `nova_gpu`. Conversion
/// happens inside the compositor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
    DisplayP3,
    LinearP3,
    Rec2020,
    LinearRec2020,
    AdobeRgb,
}

impl Default for ColorSpace {
    fn default() -> Self {
        Self::Srgb
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    /// Authoring colour space. Defaults to sRGB so older serialised data
    /// (and all existing constructor sites) keep working unchanged.
    #[serde(default)]
    pub space: ColorSpace,
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        // Equality semantics match the pre-color-management struct: two
        // colours compare equal iff their components and authoring space
        // match exactly. This keeps existing tests
        // (`*c == Color::WHITE`, etc.) working because the ergonomic
        // constructors (`rgb`, `rgba`, `WHITE`, `BLACK`, `TRANSPARENT`) all
        // produce sRGB.
        self.r == other.r
            && self.g == other.g
            && self.b == other.b
            && self.a == other.a
            && self.space == other.space
    }
}

impl Color {
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
        space: ColorSpace::Srgb,
    };
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
        space: ColorSpace::Srgb,
    };
    pub const TRANSPARENT: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
        space: ColorSpace::Srgb,
    };

    /// Construct an sRGB opaque colour.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            r,
            g,
            b,
            a: 1.0,
            space: ColorSpace::Srgb,
        }
    }

    /// Construct an sRGB colour with explicit alpha.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
            space: ColorSpace::Srgb,
        }
    }

    /// Construct a Display P3 colour (CSS `color(display-p3 …)`).
    pub const fn display_p3(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
            space: ColorSpace::DisplayP3,
        }
    }

    /// Construct a Rec.2020 colour (CSS `color(rec2020 …)`).
    pub const fn rec2020(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
            space: ColorSpace::Rec2020,
        }
    }

    /// Construct a linear sRGB colour. Useful when you've already converted
    /// out of an authored space.
    pub const fn linear_srgb(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
            space: ColorSpace::LinearSrgb,
        }
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Convert this colour to **linear sRGB**, the canonical compositing
    /// space. Pure mathematical conversion; the result is suitable for
    /// blending with `Animatable::lerp` and additive compositing.
    pub fn to_linear_srgb(self) -> Self {
        match self.space {
            ColorSpace::LinearSrgb => self,
            ColorSpace::Srgb => Color {
                r: srgb_to_linear(self.r),
                g: srgb_to_linear(self.g),
                b: srgb_to_linear(self.b),
                a: self.a,
                space: ColorSpace::LinearSrgb,
            },
            ColorSpace::DisplayP3 => self.via_linear_target(ColorSpace::LinearSrgb),
            ColorSpace::LinearP3 => Self::lin_p3_to_lin_srgb(self),
            ColorSpace::Rec2020 => self.via_linear_target(ColorSpace::LinearSrgb),
            ColorSpace::LinearRec2020 => Self::lin_rec2020_to_lin_srgb(self),
            ColorSpace::AdobeRgb => self.via_linear_target(ColorSpace::LinearSrgb),
        }
    }

    /// Whether this colour is fully transparent (alpha at zero).
    pub fn is_transparent(&self) -> bool {
        self.a <= f32::EPSILON
    }

    // --- internal cross-primary helpers ---------------------------------

    fn via_linear_target(self, target: ColorSpace) -> Self {
        // 1. decode -> linear in source primaries
        let (r, g, b) = match self.space {
            ColorSpace::Srgb => (
                srgb_to_linear(self.r),
                srgb_to_linear(self.g),
                srgb_to_linear(self.b),
            ),
            ColorSpace::DisplayP3 => (
                srgb_to_linear(self.r),
                srgb_to_linear(self.g),
                srgb_to_linear(self.b),
            ),
            ColorSpace::Rec2020 => (
                rec2020_to_linear(self.r),
                rec2020_to_linear(self.g),
                rec2020_to_linear(self.b),
            ),
            ColorSpace::AdobeRgb => (
                self.r.signum() * self.r.abs().powf(563.0 / 256.0),
                self.g.signum() * self.g.abs().powf(563.0 / 256.0),
                self.b.signum() * self.b.abs().powf(563.0 / 256.0),
            ),
            ColorSpace::LinearSrgb | ColorSpace::LinearP3 | ColorSpace::LinearRec2020 => {
                (self.r, self.g, self.b)
            }
        };

        // 2. cross-primary linear-to-linear via XYZ. We only need
        //    linear-sRGB output here; that's what the compositor wants.
        let (r2, g2, b2) = match (self.space, target) {
            (ColorSpace::DisplayP3 | ColorSpace::LinearP3, ColorSpace::LinearSrgb) => {
                lin_p3_to_lin_srgb_components(r, g, b)
            }
            (ColorSpace::Rec2020 | ColorSpace::LinearRec2020, ColorSpace::LinearSrgb) => {
                lin_rec2020_to_lin_srgb_components(r, g, b)
            }
            (ColorSpace::AdobeRgb, ColorSpace::LinearSrgb) => {
                lin_adobe_to_lin_srgb_components(r, g, b)
            }
            _ => (r, g, b),
        };

        Color {
            r: r2,
            g: g2,
            b: b2,
            a: self.a,
            space: target,
        }
    }

    fn lin_p3_to_lin_srgb(self) -> Self {
        let (r, g, b) = lin_p3_to_lin_srgb_components(self.r, self.g, self.b);
        Color {
            r,
            g,
            b,
            a: self.a,
            space: ColorSpace::LinearSrgb,
        }
    }

    fn lin_rec2020_to_lin_srgb(self) -> Self {
        let (r, g, b) = lin_rec2020_to_lin_srgb_components(self.r, self.g, self.b);
        Color {
            r,
            g,
            b,
            a: self.a,
            space: ColorSpace::LinearSrgb,
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::TRANSPARENT
    }
}

// ---------------------------------------------------------------------------
// Transfer functions (kept inline so `photon` stays free of nova_gpu deps)
// ---------------------------------------------------------------------------

fn srgb_to_linear(c: f32) -> f32 {
    let s = c.signum();
    let m = c.abs();
    let lin = if m <= 0.04045 {
        m / 12.92
    } else {
        ((m + 0.055) / 1.055).powf(2.4)
    };
    lin * s
}

fn rec2020_to_linear(c: f32) -> f32 {
    const ALPHA: f32 = 1.099_296_8;
    const BETA: f32 = 0.018_053_97;
    let s = c.signum();
    let m = c.abs();
    let lin = if m < BETA * 4.5 {
        m / 4.5
    } else {
        ((m + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45)
    };
    lin * s
}

// linear P3 -> linear sRGB matrix (identical to the one in nova_gpu/color_management).
fn lin_p3_to_lin_srgb_components(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // M = XYZ_TO_LIN_SRGB * LIN_P3_TO_XYZ
    let m: [[f32; 3]; 3] = [
        [1.224_940, -0.224_945, 0.000_000],
        [-0.042_055, 1.042_058, 0.000_000],
        [-0.019_640, -0.078_634, 1.098_276],
    ];
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

fn lin_rec2020_to_lin_srgb_components(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let m: [[f32; 3]; 3] = [
        [1.660_491, -0.587_641, -0.072_852],
        [-0.124_550, 1.132_900, -0.008_348],
        [-0.018_152, -0.100_578, 1.118_730],
    ];
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

fn lin_adobe_to_lin_srgb_components(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let m: [[f32; 3]; 3] = [
        [1.398_171, -0.397_805, -0.000_366],
        [-0.000_040, 0.999_961, 0.000_079],
        [-0.012_891, -0.069_088, 1.081_979],
    ];
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn rgb_alpha_is_one() {
        let c = Color::rgb(0.5, 0.5, 0.5);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn rgba_round_trip() {
        let c = Color::rgba(0.1, 0.2, 0.3, 0.4);
        assert_eq!(c.to_array(), [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn constants_are_correct() {
        assert_eq!(Color::WHITE.r, 1.0);
        assert_eq!(Color::BLACK.b, 0.0);
        assert!(Color::TRANSPARENT.is_transparent());
    }

    #[test]
    fn default_is_transparent() {
        let c = Color::default();
        assert!(c.is_transparent());
    }

    #[test]
    fn default_space_is_srgb() {
        assert_eq!(Color::WHITE.space, ColorSpace::Srgb);
        assert_eq!(Color::rgb(0.1, 0.2, 0.3).space, ColorSpace::Srgb);
    }

    #[test]
    fn p3_constructor_tags_space() {
        let c = Color::display_p3(1.0, 0.0, 0.0, 1.0);
        assert_eq!(c.space, ColorSpace::DisplayP3);
    }

    #[test]
    fn rec2020_constructor_tags_space() {
        let c = Color::rec2020(0.5, 0.5, 0.5, 1.0);
        assert_eq!(c.space, ColorSpace::Rec2020);
    }

    #[test]
    fn srgb_to_linear_identity_at_endpoints() {
        let c = Color::rgb(0.0, 1.0, 0.0).to_linear_srgb();
        assert!(approx(c.r, 0.0, 1e-6));
        assert!(approx(c.g, 1.0, 1e-6));
        assert!(approx(c.b, 0.0, 1e-6));
        assert_eq!(c.space, ColorSpace::LinearSrgb);
    }

    #[test]
    fn srgb_50_to_linear_about_0_2140() {
        let c = Color::rgb(0.5, 0.5, 0.5).to_linear_srgb();
        assert!(approx(c.r, 0.2140, 1e-3));
    }

    #[test]
    fn p3_red_to_linear_srgb_widens_then_clamps() {
        let lin = Color::display_p3(1.0, 0.0, 0.0, 1.0).to_linear_srgb();
        // Pure P3 red is more saturated than sRGB red — linear-sRGB R will
        // exceed 1.0, and G/B will go negative.
        assert!(lin.r > 1.0, "expected r > 1, got {}", lin.r);
        assert!(lin.g < 0.0 || lin.g.abs() < 1e-3);
    }

    #[test]
    fn back_compat_white_equals_white() {
        let a = Color::WHITE;
        let b = Color::rgb(1.0, 1.0, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn back_compat_black_equals_black() {
        assert_eq!(Color::BLACK, Color::rgb(0.0, 0.0, 0.0));
    }

    #[test]
    fn space_difference_breaks_equality() {
        let a = Color::rgb(0.5, 0.5, 0.5);
        let b = Color::display_p3(0.5, 0.5, 0.5, 1.0);
        assert_ne!(a, b);
    }

    #[test]
    fn linear_srgb_passthrough() {
        let c = Color::linear_srgb(0.4, 0.4, 0.4, 1.0).to_linear_srgb();
        assert!(approx(c.r, 0.4, 1e-7));
    }

    #[test]
    fn serde_default_back_fill_for_legacy_payloads() {
        // Old serialisations had no `space` key. `#[serde(default)]` should
        // make decoding fill it in with sRGB.
        let json = "{\"r\":0.1,\"g\":0.2,\"b\":0.3,\"a\":0.4}";
        let c: Color = serde_json::from_str(json).expect("legacy color decodes");
        assert_eq!(c.space, ColorSpace::Srgb);
        assert!(approx(c.r, 0.1, 1e-7));
    }
}
