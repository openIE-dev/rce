//! Color science: color spaces, color difference metrics, color blending,
//! palette generation, color temperature, spectral to RGB.


// ═══════════════════════════════════════════════════════════════════
// COLOR SPACE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════

/// HSL to RGB. All values [0, 1].
pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    if s < 1e-15 { return (l, l, l); }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    (hue_to_rgb(p, q, h + 1.0/3.0), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1.0/3.0))
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 { t += 1.0; } if t > 1.0 { t -= 1.0; }
    if t < 1.0/6.0 { p + (q - p) * 6.0 * t }
    else if t < 1.0/2.0 { q }
    else if t < 2.0/3.0 { p + (q - p) * (2.0/3.0 - t) * 6.0 }
    else { p }
}

/// RGB to HSL. All values [0, 1].
pub fn rgb_to_hsl(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if (max - min).abs() < 1e-15 { return (0.0, 0.0, l); }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if (max - r).abs() < 1e-15 { ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0 }
            else if (max - g).abs() < 1e-15 { ((b - r) / d + 2.0) / 6.0 }
            else { ((r - g) / d + 4.0) / 6.0 };
    (h, s, l)
}

/// XYZ to Lab (CIE L*a*b*, D65 illuminant).
pub fn xyz_to_lab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (xn, yn, zn) = (0.95047, 1.0, 1.08883);
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0/116.0 };
    let l = 116.0 * f(y / yn) - 16.0;
    let a = 500.0 * (f(x / xn) - f(y / yn));
    let b = 200.0 * (f(y / yn) - f(z / zn));
    (l, a, b)
}

/// Lab to XYZ.
pub fn lab_to_xyz(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let (xn, yn, zn) = (0.95047, 1.0, 1.08883);
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let inv = |t: f64| if t > 0.206897 { t * t * t } else { (t - 16.0/116.0) / 7.787 };
    (xn * inv(fx), yn * inv(fy), zn * inv(fz))
}

/// Lab to LCH (cylindrical form of Lab).
pub fn lab_to_lch(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let c = (a * a + b * b).sqrt();
    let h = b.atan2(a).to_degrees();
    let h = if h < 0.0 { h + 360.0 } else { h };
    (l, c, h)
}

/// LCH to Lab.
pub fn lch_to_lab(l: f64, c: f64, h: f64) -> (f64, f64, f64) {
    let h_rad = h.to_radians();
    (l, c * h_rad.cos(), c * h_rad.sin())
}

/// sRGB to linear RGB.
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Linear RGB to sRGB.
pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0/2.4) - 0.055 }
}

/// XYZ to sRGB.
pub fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = linear_to_srgb( 3.2404542 * x - 1.5371385 * y - 0.4985314 * z);
    let g = linear_to_srgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z);
    let b = linear_to_srgb( 0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
    (r.max(0.0).min(1.0), g.max(0.0).min(1.0), b.max(0.0).min(1.0))
}

/// RGB to hex string (#RRGGBB).
pub fn rgb_to_hex(r: f64, g: f64, b: f64) -> String {
    format!("#{:02x}{:02x}{:02x}", (r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8)
}

/// Hex string to RGB.
pub fn hex_to_rgb(hex: &str) -> (f64, f64, f64) {
    let hex = hex.trim_start_matches('#');
    if hex.len() < 6 { return (0.0, 0.0, 0.0); }
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f64 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f64 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f64 / 255.0;
    (r, g, b)
}

// ═══════════════════════════════════════════════════════════════════
// COLOR DIFFERENCE
// ═══════════════════════════════════════════════════════════════════

/// CIE76 color difference (Euclidean in Lab).
pub fn delta_e_76(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> f64 {
    ((l1-l2).powi(2) + (a1-a2).powi(2) + (b1-b2).powi(2)).sqrt()
}

/// CIE94 color difference.
pub fn delta_e_94(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> f64 {
    let dl = l1 - l2;
    let c1 = (a1*a1 + b1*b1).sqrt();
    let c2 = (a2*a2 + b2*b2).sqrt();
    let dc = c1 - c2;
    let da = a1 - a2;
    let db = b1 - b2;
    let dh2 = da*da + db*db - dc*dc;
    let dh = if dh2 > 0.0 { dh2.sqrt() } else { 0.0 };
    let sl = 1.0;
    let sc = 1.0 + 0.045 * c1;
    let sh = 1.0 + 0.015 * c1;
    ((dl/sl).powi(2) + (dc/sc).powi(2) + (dh/sh).powi(2)).sqrt()
}

/// CIEDE2000 color difference (simplified).
pub fn delta_e_2000(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> f64 {
    // Simplified implementation — full CIEDE2000 has rotation term
    let dl = l2 - l1;
    let l_bar = (l1 + l2) / 2.0;
    let c1 = (a1*a1 + b1*b1).sqrt();
    let c2 = (a2*a2 + b2*b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;
    let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25.0f64.powi(7))).sqrt());
    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p*a1p + b1*b1).sqrt();
    let c2p = (a2p*a2p + b2*b2).sqrt();
    let dcp = c2p - c1p;
    let h1p = b1.atan2(a1p).to_degrees();
    let h1p = if h1p < 0.0 { h1p + 360.0 } else { h1p };
    let h2p = b2.atan2(a2p).to_degrees();
    let h2p = if h2p < 0.0 { h2p + 360.0 } else { h2p };
    let dhp = h2p - h1p;
    let dhp_adj = if dhp.abs() <= 180.0 { dhp } else if dhp > 180.0 { dhp - 360.0 } else { dhp + 360.0 };
    let dh = 2.0 * (c1p * c2p).sqrt() * (dhp_adj.to_radians() / 2.0).sin();
    let sl = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * (c1p + c2p) / 2.0;
    let sh = 1.0 + 0.015 * (c1p + c2p) / 2.0;
    ((dl/sl).powi(2) + (dcp/sc).powi(2) + (dh/sh).powi(2)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// COLOR BLENDING
// ═══════════════════════════════════════════════════════════════════

/// Linear blend: lerp in RGB.
pub fn blend_rgb(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64, t: f64) -> (f64, f64, f64) {
    (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))
}

/// Blend in Lab space (perceptually uniform).
pub fn blend_lab(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64, t: f64) -> (f64, f64, f64) {
    (l1 + t*(l2-l1), a1 + t*(a2-a1), b1 + t*(b2-b1))
}

/// Alpha compositing (over operation).
pub fn alpha_over(r1: f64, g1: f64, b1: f64, a1: f64, r2: f64, g2: f64, b2: f64, a2: f64) -> (f64, f64, f64, f64) {
    let ao = a1 + a2 * (1.0 - a1);
    if ao < 1e-15 { return (0.0, 0.0, 0.0, 0.0); }
    let ro = (r1 * a1 + r2 * a2 * (1.0 - a1)) / ao;
    let go = (g1 * a1 + g2 * a2 * (1.0 - a1)) / ao;
    let bo = (b1 * a1 + b2 * a2 * (1.0 - a1)) / ao;
    (ro, go, bo, ao)
}

// ═══════════════════════════════════════════════════════════════════
// PALETTE GENERATION
// ═══════════════════════════════════════════════════════════════════

/// Generate n evenly spaced hues (HSL, fixed S and L).
pub fn palette_equidistant(n: usize, saturation: f64, lightness: f64) -> Vec<(f64, f64, f64)> {
    (0..n).map(|i| hsl_to_rgb(i as f64 / n as f64, saturation, lightness)).collect()
}

/// Generate analogous palette (hues within range).
pub fn palette_analogous(base_hue: f64, n: usize, spread: f64, s: f64, l: f64) -> Vec<(f64, f64, f64)> {
    let step = spread / (n - 1).max(1) as f64;
    (0..n).map(|i| {
        let h = (base_hue - spread/2.0 + i as f64 * step + 1.0) % 1.0;
        hsl_to_rgb(h, s, l)
    }).collect()
}

/// Generate complementary pair.
pub fn palette_complementary(h: f64, s: f64, l: f64) -> [(f64, f64, f64); 2] {
    [hsl_to_rgb(h, s, l), hsl_to_rgb((h + 0.5) % 1.0, s, l)]
}

/// Generate triadic palette.
pub fn palette_triadic(h: f64, s: f64, l: f64) -> [(f64, f64, f64); 3] {
    [hsl_to_rgb(h, s, l), hsl_to_rgb((h + 1.0/3.0) % 1.0, s, l), hsl_to_rgb((h + 2.0/3.0) % 1.0, s, l)]
}

/// Linear gradient: n colors interpolated between two RGB colors.
pub fn gradient(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64, n: usize) -> Vec<(f64, f64, f64)> {
    (0..n).map(|i| {
        let t = i as f64 / (n - 1).max(1) as f64;
        blend_rgb(r1, g1, b1, r2, g2, b2, t)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// COLOR TEMPERATURE
// ═══════════════════════════════════════════════════════════════════

/// Correlated color temperature to sRGB (Tanner Helland approximation).
pub fn cct_to_rgb(kelvin: f64) -> (f64, f64, f64) {
    let temp = kelvin / 100.0;
    let r = if temp <= 66.0 { 1.0 }
            else { (329.698727446 * (temp - 60.0).powf(-0.1332047592) / 255.0).min(1.0).max(0.0) };
    let g = if temp <= 66.0 { (99.4708025861 * temp.ln() - 161.1195681661) / 255.0 }
            else { (288.1221695283 * (temp - 60.0).powf(-0.0755148492)) / 255.0 };
    let g = g.min(1.0).max(0.0);
    let b = if temp >= 66.0 { 1.0 }
            else if temp <= 19.0 { 0.0 }
            else { (138.5177312231 * (temp - 10.0).ln() - 305.0447927307) / 255.0 };
    let b = b.min(1.0).max(0.0);
    (r, g, b)
}

/// Luminance (relative luminance per WCAG).
pub fn luminance(r: f64, g: f64, b: f64) -> f64 {
    0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)
}

/// WCAG contrast ratio between two colors.
pub fn contrast_ratio(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64) -> f64 {
    let l1 = luminance(r1, g1, b1);
    let l2 = luminance(r2, g2, b2);
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// Check WCAG AA contrast (4.5:1 for normal text).
pub fn wcag_aa(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64) -> bool {
    contrast_ratio(r1, g1, b1, r2, g2, b2) >= 4.5
}

/// Check WCAG AAA contrast (7:1).
pub fn wcag_aaa(r1: f64, g1: f64, b1: f64, r2: f64, g2: f64, b2: f64) -> bool {
    contrast_ratio(r1, g1, b1, r2, g2, b2) >= 7.0
}

/// Grayscale value from RGB.
pub fn to_grayscale(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Invert color.
pub fn invert_color(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (1.0 - r, 1.0 - g, 1.0 - b)
}

/// Desaturate by amount [0, 1].
pub fn desaturate(r: f64, g: f64, b: f64, amount: f64) -> (f64, f64, f64) {
    let gray = to_grayscale(r, g, b);
    (r + (gray - r) * amount, g + (gray - g) * amount, b + (gray - b) * amount)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsl_roundtrip() {
        let (r, g, b) = hsl_to_rgb(0.5, 0.8, 0.6);
        let (h, s, l) = rgb_to_hsl(r, g, b);
        assert!((h - 0.5).abs() < 0.01);
        assert!((s - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_hex_roundtrip() {
        let hex = rgb_to_hex(1.0, 0.0, 0.5);
        let (r, g, b) = hex_to_rgb(&hex);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_delta_e_identical() {
        assert!(delta_e_76(50.0, 20.0, -10.0, 50.0, 20.0, -10.0) < 1e-10);
    }

    #[test]
    fn test_contrast_bw() {
        let cr = contrast_ratio(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert!(cr > 20.0); // black/white ≈ 21:1
    }

    #[test]
    fn test_wcag() {
        assert!(wcag_aa(1.0, 1.0, 1.0, 0.0, 0.0, 0.0));
        assert!(wcag_aaa(1.0, 1.0, 1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_cct() {
        let (r, g, b) = cct_to_rgb(6500.0);
        assert!(r > 0.9 && g > 0.9 && b > 0.9); // daylight ≈ white
    }

    #[test]
    fn test_palette() {
        let pal = palette_equidistant(5, 0.8, 0.5);
        assert_eq!(pal.len(), 5);
    }

    #[test]
    fn test_gradient() {
        let grad = gradient(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5);
        assert_eq!(grad.len(), 5);
        assert!((grad[2].0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_invert() {
        assert_eq!(invert_color(1.0, 0.0, 0.5), (0.0, 1.0, 0.5));
    }
}
