//! Image processing: convolution kernels, morphological operations,
//! edge detection, histogram operations, color space conversions,
//! geometric transforms, filters, thresholding.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// IMAGE REPRESENTATION
// ═══════════════════════════════════════════════════════════════════

/// Grayscale image as row-major f64 pixels [0.0, 1.0].
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    pub fn from_data(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self { data, width, height }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.data[y * self.width + x] = val;
    }

    pub fn get_clamped(&self, x: i64, y: i64) -> f64 {
        let cx = x.max(0).min(self.width as i64 - 1) as usize;
        let cy = y.max(0).min(self.height as i64 - 1) as usize;
        self.get(cx, cy)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2D CONVOLUTION
// ═══════════════════════════════════════════════════════════════════

/// 2D convolution with a kernel (border clamped).
pub fn convolve2d(img: &GrayImage, kernel: &[f64], kw: usize, kh: usize) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let hkw = kw as i64 / 2;
    let hkh = kh as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut sum = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let ix = x as i64 + kx as i64 - hkw;
                    let iy = y as i64 + ky as i64 - hkh;
                    sum += img.get_clamped(ix, iy) * kernel[ky * kw + kx];
                }
            }
            out.set(x, y, sum);
        }
    }
    out
}

/// Separable 2D convolution (row filter, then column filter).
pub fn convolve2d_separable(img: &GrayImage, row_kernel: &[f64], col_kernel: &[f64]) -> GrayImage {
    // Row pass
    let mut temp = GrayImage::new(img.width, img.height);
    let hk = row_kernel.len() as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut sum = 0.0;
            for (i, &k) in row_kernel.iter().enumerate() {
                sum += img.get_clamped(x as i64 + i as i64 - hk, y as i64) * k;
            }
            temp.set(x, y, sum);
        }
    }
    // Column pass
    let mut out = GrayImage::new(img.width, img.height);
    let hk = col_kernel.len() as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut sum = 0.0;
            for (i, &k) in col_kernel.iter().enumerate() {
                sum += temp.get_clamped(x as i64, y as i64 + i as i64 - hk) * k;
            }
            out.set(x, y, sum);
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// KERNELS
// ═══════════════════════════════════════════════════════════════════

/// Gaussian kernel (normalized).
pub fn gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
    let half = size as f64 / 2.0;
    let mut kernel = Vec::with_capacity(size * size);
    let mut sum = 0.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - half + 0.5;
            let dy = y as f64 - half + 0.5;
            let v = (-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).exp();
            kernel.push(v);
            sum += v;
        }
    }
    for v in kernel.iter_mut() { *v /= sum; }
    kernel
}

/// Sobel kernels (3×3). Returns (horizontal, vertical).
pub fn sobel_kernels() -> ([f64; 9], [f64; 9]) {
    let gx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let gy = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    (gx, gy)
}

/// Prewitt kernels (3×3).
pub fn prewitt_kernels() -> ([f64; 9], [f64; 9]) {
    let gx = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
    let gy = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    (gx, gy)
}

/// Laplacian kernel (3×3).
pub fn laplacian_kernel() -> [f64; 9] {
    [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]
}

/// Sharpening kernel (3×3).
pub fn sharpen_kernel() -> [f64; 9] {
    [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]
}

/// Box blur kernel (normalized).
pub fn box_kernel(size: usize) -> Vec<f64> {
    let n = (size * size) as f64;
    vec![1.0 / n; size * size]
}

// ═══════════════════════════════════════════════════════════════════
// EDGE DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Sobel edge detection. Returns gradient magnitude image.
pub fn sobel_edges(img: &GrayImage) -> GrayImage {
    let (gx, gy) = sobel_kernels();
    let ix = convolve2d(img, &gx, 3, 3);
    let iy = convolve2d(img, &gy, 3, 3);
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = (ix.data[i] * ix.data[i] + iy.data[i] * iy.data[i]).sqrt();
    }
    out
}

/// Canny edge detection (simplified).
pub fn canny_edges(img: &GrayImage, low_thresh: f64, high_thresh: f64) -> GrayImage {
    // 1. Gaussian blur
    let blurred = gaussian_blur(img, 5, 1.4);
    // 2. Gradient magnitude and direction
    let (gx_k, gy_k) = sobel_kernels();
    let gx = convolve2d(&blurred, &gx_k, 3, 3);
    let gy = convolve2d(&blurred, &gy_k, 3, 3);
    let mut mag = GrayImage::new(img.width, img.height);
    let mut dir = vec![0.0; img.width * img.height];
    for i in 0..mag.data.len() {
        mag.data[i] = (gx.data[i].powi(2) + gy.data[i].powi(2)).sqrt();
        dir[i] = gy.data[i].atan2(gx.data[i]);
    }
    // 3. Non-maximum suppression
    let mut nms = GrayImage::new(img.width, img.height);
    for y in 1..img.height - 1 {
        for x in 1..img.width - 1 {
            let angle = dir[y * img.width + x];
            let m = mag.get(x, y);
            let (n1, n2) = if angle.abs() < PI / 8.0 || angle.abs() > 7.0 * PI / 8.0 {
                (mag.get(x + 1, y), mag.get(x - 1, y))
            } else if angle > PI / 8.0 && angle < 3.0 * PI / 8.0 {
                (mag.get(x + 1, y - 1), mag.get(x - 1, y + 1))
            } else if angle.abs() > 3.0 * PI / 8.0 && angle.abs() < 5.0 * PI / 8.0 {
                (mag.get(x, y - 1), mag.get(x, y + 1))
            } else {
                (mag.get(x - 1, y - 1), mag.get(x + 1, y + 1))
            };
            if m >= n1 && m >= n2 { nms.set(x, y, m); }
        }
    }
    // 4. Double threshold + hysteresis
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        if nms.data[i] >= high_thresh {
            out.data[i] = 1.0;
        } else if nms.data[i] >= low_thresh {
            out.data[i] = 0.5; // weak edge
        }
    }
    // Hysteresis: promote weak edges connected to strong edges
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..img.height - 1 {
            for x in 1..img.width - 1 {
                if out.get(x, y) == 0.5 {
                    let has_strong = (-1i64..=1).any(|dy| (-1i64..=1).any(|dx| {
                        out.get((x as i64 + dx) as usize, (y as i64 + dy) as usize) == 1.0
                    }));
                    if has_strong { out.set(x, y, 1.0); changed = true; }
                }
            }
        }
    }
    // Remove remaining weak edges
    for v in out.data.iter_mut() { if *v < 1.0 { *v = 0.0; } }
    out
}

/// Laplacian of Gaussian (LoG) edge detection.
pub fn log_edges(img: &GrayImage, sigma: f64) -> GrayImage {
    let size = (6.0 * sigma).ceil() as usize | 1; // ensure odd
    let half = size as f64 / 2.0;
    let mut kernel = vec![0.0; size * size];
    let s2 = sigma * sigma;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - half + 0.5;
            let dy = y as f64 - half + 0.5;
            let r2 = dx * dx + dy * dy;
            kernel[y * size + x] = -(1.0 - r2 / (2.0 * s2)) * (-r2 / (2.0 * s2)).exp() / (PI * s2 * s2);
        }
    }
    convolve2d(img, &kernel, size, size)
}

// ═══════════════════════════════════════════════════════════════════
// BLUR & SMOOTHING
// ═══════════════════════════════════════════════════════════════════

/// Gaussian blur.
pub fn gaussian_blur(img: &GrayImage, size: usize, sigma: f64) -> GrayImage {
    let kernel = gaussian_kernel(size, sigma);
    convolve2d(img, &kernel, size, size)
}

/// Box blur (mean filter).
pub fn box_blur(img: &GrayImage, size: usize) -> GrayImage {
    let kernel = box_kernel(size);
    convolve2d(img, &kernel, size, size)
}

/// Median filter.
pub fn median_filter(img: &GrayImage, size: usize) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let half = size as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut vals = Vec::with_capacity(size * size);
            for ky in 0..size {
                for kx in 0..size {
                    vals.push(img.get_clamped(x as i64 + kx as i64 - half, y as i64 + ky as i64 - half));
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.set(x, y, vals[vals.len() / 2]);
        }
    }
    out
}

/// Bilateral filter (edge-preserving smoothing).
pub fn bilateral_filter(img: &GrayImage, size: usize, sigma_space: f64, sigma_range: f64) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let half = size as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let center = img.get(x, y);
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for ky in 0..size {
                for kx in 0..size {
                    let ix = x as i64 + kx as i64 - half;
                    let iy = y as i64 + ky as i64 - half;
                    let val = img.get_clamped(ix, iy);
                    let dx = kx as f64 - half as f64;
                    let dy = ky as f64 - half as f64;
                    let spatial = (-0.5 * (dx * dx + dy * dy) / (sigma_space * sigma_space)).exp();
                    let range = (-0.5 * (val - center).powi(2) / (sigma_range * sigma_range)).exp();
                    let w = spatial * range;
                    sum += val * w;
                    weight_sum += w;
                }
            }
            out.set(x, y, sum / weight_sum);
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// MORPHOLOGICAL OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Erosion (binary/grayscale, square structuring element).
pub fn erode(img: &GrayImage, size: usize) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let half = size as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut min_val = f64::INFINITY;
            for ky in 0..size {
                for kx in 0..size {
                    let v = img.get_clamped(x as i64 + kx as i64 - half, y as i64 + ky as i64 - half);
                    if v < min_val { min_val = v; }
                }
            }
            out.set(x, y, min_val);
        }
    }
    out
}

/// Dilation (binary/grayscale, square structuring element).
pub fn dilate(img: &GrayImage, size: usize) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let half = size as i64 / 2;
    for y in 0..img.height {
        for x in 0..img.width {
            let mut max_val = f64::NEG_INFINITY;
            for ky in 0..size {
                for kx in 0..size {
                    let v = img.get_clamped(x as i64 + kx as i64 - half, y as i64 + ky as i64 - half);
                    if v > max_val { max_val = v; }
                }
            }
            out.set(x, y, max_val);
        }
    }
    out
}

/// Morphological opening (erosion then dilation).
pub fn morph_open(img: &GrayImage, size: usize) -> GrayImage {
    dilate(&erode(img, size), size)
}

/// Morphological closing (dilation then erosion).
pub fn morph_close(img: &GrayImage, size: usize) -> GrayImage {
    erode(&dilate(img, size), size)
}

/// Morphological gradient (dilation − erosion).
pub fn morph_gradient(img: &GrayImage, size: usize) -> GrayImage {
    let d = dilate(img, size);
    let e = erode(img, size);
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = d.data[i] - e.data[i];
    }
    out
}

/// Top-hat (original − opening).
pub fn top_hat(img: &GrayImage, size: usize) -> GrayImage {
    let opened = morph_open(img, size);
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = img.data[i] - opened.data[i];
    }
    out
}

/// Black-hat (closing − original).
pub fn black_hat(img: &GrayImage, size: usize) -> GrayImage {
    let closed = morph_close(img, size);
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = closed.data[i] - img.data[i];
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// THRESHOLDING
// ═══════════════════════════════════════════════════════════════════

/// Simple threshold: pixel >= threshold → 1.0, else → 0.0.
pub fn threshold(img: &GrayImage, thresh: f64) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = if img.data[i] >= thresh { 1.0 } else { 0.0 };
    }
    out
}

/// Otsu's method: find optimal threshold that minimizes intra-class variance.
pub fn otsu_threshold(img: &GrayImage) -> f64 {
    let nbins = 256;
    let mut hist = vec![0usize; nbins];
    for &v in &img.data {
        let bin = (v * (nbins - 1) as f64).round() as usize;
        hist[bin.min(nbins - 1)] += 1;
    }
    let total = img.data.len() as f64;
    let mut best_thresh = 0.0;
    let mut best_var = 0.0;
    let mut w0 = 0.0;
    let mut sum0 = 0.0;
    let sum_total: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    for t in 0..nbins {
        w0 += hist[t] as f64;
        if w0 == 0.0 { continue; }
        let w1 = total - w0;
        if w1 == 0.0 { break; }
        sum0 += t as f64 * hist[t] as f64;
        let m0 = sum0 / w0;
        let m1 = (sum_total - sum0) / w1;
        let var = w0 * w1 * (m0 - m1).powi(2);
        if var > best_var {
            best_var = var;
            best_thresh = t as f64 / (nbins - 1) as f64;
        }
    }
    best_thresh
}

/// Adaptive threshold (local mean).
pub fn adaptive_threshold(img: &GrayImage, block_size: usize, c: f64) -> GrayImage {
    let blurred = box_blur(img, block_size);
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        out.data[i] = if img.data[i] >= blurred.data[i] - c { 1.0 } else { 0.0 };
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// HISTOGRAM OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Compute histogram of image (256 bins for [0,1] range).
pub fn histogram(img: &GrayImage, nbins: usize) -> Vec<usize> {
    let mut hist = vec![0usize; nbins];
    for &v in &img.data {
        let bin = (v * (nbins - 1) as f64).round() as usize;
        hist[bin.min(nbins - 1)] += 1;
    }
    hist
}

/// Histogram equalization.
pub fn histogram_equalize(img: &GrayImage) -> GrayImage {
    let nbins = 256;
    let hist = histogram(img, nbins);
    let total = img.data.len() as f64;
    let mut cdf = vec![0.0; nbins];
    cdf[0] = hist[0] as f64 / total;
    for i in 1..nbins {
        cdf[i] = cdf[i - 1] + hist[i] as f64 / total;
    }
    let mut out = GrayImage::new(img.width, img.height);
    for i in 0..out.data.len() {
        let bin = (img.data[i] * (nbins - 1) as f64).round() as usize;
        out.data[i] = cdf[bin.min(nbins - 1)];
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// COLOR SPACE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════

/// RGB → Grayscale (luminance).
pub fn rgb_to_gray(r: f64, g: f64, b: f64) -> f64 {
    0.2989 * r + 0.5870 * g + 0.1140 * b
}

/// RGB → HSV. All values in [0,1].
pub fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max < 1e-15 { 0.0 } else { delta / max };
    let h = if delta < 1e-15 {
        0.0
    } else if (max - r).abs() < 1e-15 {
        ((g - b) / delta % 6.0) / 6.0
    } else if (max - g).abs() < 1e-15 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    (if h < 0.0 { h + 1.0 } else { h }, s, v)
}

/// HSV → RGB. All values in [0,1].
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6 as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

/// RGB → XYZ (CIE 1931, D65 illuminant).
pub fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let linearize = |v: f64| if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 };
    let rl = linearize(r);
    let gl = linearize(g);
    let bl = linearize(b);
    (
        0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl,
        0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl,
        0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl,
    )
}

/// RGB → Lab (CIE L*a*b*, D65).
pub fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    // D65 reference white
    let (xn, yn, zn) = (0.95047, 1.00000, 1.08883);
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

// ═══════════════════════════════════════════════════════════════════
// GEOMETRIC TRANSFORMS
// ═══════════════════════════════════════════════════════════════════

/// Resize image using bilinear interpolation.
pub fn resize_bilinear(img: &GrayImage, new_width: usize, new_height: usize) -> GrayImage {
    let mut out = GrayImage::new(new_width, new_height);
    let sx = img.width as f64 / new_width as f64;
    let sy = img.height as f64 / new_height as f64;
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f64 + 0.5) * sx - 0.5;
            let src_y = (y as f64 + 0.5) * sy - 0.5;
            let x0 = src_x.floor() as i64;
            let y0 = src_y.floor() as i64;
            let fx = src_x - x0 as f64;
            let fy = src_y - y0 as f64;
            let v00 = img.get_clamped(x0, y0);
            let v10 = img.get_clamped(x0 + 1, y0);
            let v01 = img.get_clamped(x0, y0 + 1);
            let v11 = img.get_clamped(x0 + 1, y0 + 1);
            let val = v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy + v11 * fx * fy;
            out.set(x, y, val);
        }
    }
    out
}

/// Rotate image by angle (radians, counterclockwise) about center.
pub fn rotate_image(img: &GrayImage, angle: f64) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    let cx = img.width as f64 / 2.0;
    let cy = img.height as f64 / 2.0;
    let (sin_a, cos_a) = angle.sin_cos();
    for y in 0..img.height {
        for x in 0..img.width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let src_x = cos_a * dx + sin_a * dy + cx;
            let src_y = -sin_a * dx + cos_a * dy + cy;
            let ix = src_x.floor() as i64;
            let iy = src_y.floor() as i64;
            if ix >= 0 && ix < img.width as i64 - 1 && iy >= 0 && iy < img.height as i64 - 1 {
                let fx = src_x - ix as f64;
                let fy = src_y - iy as f64;
                let v = img.get_clamped(ix, iy) * (1.0 - fx) * (1.0 - fy)
                    + img.get_clamped(ix + 1, iy) * fx * (1.0 - fy)
                    + img.get_clamped(ix, iy + 1) * (1.0 - fx) * fy
                    + img.get_clamped(ix + 1, iy + 1) * fx * fy;
                out.set(x, y, v);
            }
        }
    }
    out
}

/// Flip image horizontally.
pub fn flip_horizontal(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(x, y, img.get(img.width - 1 - x, y));
        }
    }
    out
}

/// Flip image vertically.
pub fn flip_vertical(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(x, y, img.get(x, img.height - 1 - y));
        }
    }
    out
}

/// Transpose image.
pub fn transpose_image(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.height, img.width);
    for y in 0..img.height {
        for x in 0..img.width {
            out.set(y, x, img.get(x, y));
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE METRICS
// ═══════════════════════════════════════════════════════════════════

/// Mean Squared Error between two images.
pub fn mse(a: &GrayImage, b: &GrayImage) -> f64 {
    let n = a.data.len() as f64;
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / n
}

/// Peak Signal-to-Noise Ratio (dB).
pub fn psnr(a: &GrayImage, b: &GrayImage) -> f64 {
    let m = mse(a, b);
    if m < 1e-15 { return f64::INFINITY; }
    10.0 * (1.0 / m).log10()
}

/// Structural Similarity Index (SSIM, simplified).
pub fn ssim(a: &GrayImage, b: &GrayImage) -> f64 {
    let n = a.data.len() as f64;
    let mu_a = a.data.iter().sum::<f64>() / n;
    let mu_b = b.data.iter().sum::<f64>() / n;
    let var_a = a.data.iter().map(|x| (x - mu_a).powi(2)).sum::<f64>() / n;
    let var_b = b.data.iter().map(|x| (x - mu_b).powi(2)).sum::<f64>() / n;
    let cov = a.data.iter().zip(&b.data).map(|(x, y)| (x - mu_a) * (y - mu_b)).sum::<f64>() / n;
    let c1 = 0.01 * 0.01; // (k1*L)^2 with L=1.0
    let c2 = 0.03 * 0.03;
    (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
        / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> GrayImage {
        GrayImage::from_data(vec![
            0.0, 0.1, 0.2, 0.3,
            0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 1.0, 0.9,
            0.8, 0.7, 0.6, 0.5,
        ], 4, 4)
    }

    #[test]
    fn test_gaussian_kernel() {
        let k = gaussian_kernel(3, 1.0);
        assert_eq!(k.len(), 9);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sobel_edges() {
        let img = test_image();
        let edges = sobel_edges(&img);
        assert_eq!(edges.data.len(), img.data.len());
    }

    #[test]
    fn test_gaussian_blur() {
        let img = test_image();
        let blurred = gaussian_blur(&img, 3, 1.0);
        assert_eq!(blurred.data.len(), img.data.len());
    }

    #[test]
    fn test_median_filter() {
        let img = test_image();
        let filtered = median_filter(&img, 3);
        assert_eq!(filtered.data.len(), img.data.len());
    }

    #[test]
    fn test_erode_dilate() {
        let img = test_image();
        let eroded = erode(&img, 3);
        let dilated = dilate(&img, 3);
        // Eroded should have smaller max, dilated larger min
        assert!(eroded.data.iter().sum::<f64>() <= img.data.iter().sum::<f64>());
        assert!(dilated.data.iter().sum::<f64>() >= img.data.iter().sum::<f64>());
    }

    #[test]
    fn test_threshold() {
        let img = test_image();
        let binary = threshold(&img, 0.5);
        for &v in &binary.data { assert!(v == 0.0 || v == 1.0); }
    }

    #[test]
    fn test_otsu() {
        let img = test_image();
        let t = otsu_threshold(&img);
        assert!(t > 0.0 && t < 1.0);
    }

    #[test]
    fn test_histogram() {
        let img = test_image();
        let hist = histogram(&img, 256);
        let total: usize = hist.iter().sum();
        assert_eq!(total, img.data.len());
    }

    #[test]
    fn test_histogram_equalize() {
        let img = test_image();
        let eq = histogram_equalize(&img);
        assert_eq!(eq.data.len(), img.data.len());
    }

    #[test]
    fn test_rgb_to_hsv_roundtrip() {
        let (h, s, v) = rgb_to_hsv(0.5, 0.3, 0.8);
        let (r, g, b) = hsv_to_rgb(h, s, v);
        assert!((r - 0.5).abs() < 1e-6);
        assert!((g - 0.3).abs() < 1e-6);
        assert!((b - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_to_lab() {
        let (l, a, b) = rgb_to_lab(1.0, 1.0, 1.0);
        assert!((l - 100.0).abs() < 1.0); // White should be L≈100
    }

    #[test]
    fn test_resize() {
        let img = test_image();
        let resized = resize_bilinear(&img, 8, 8);
        assert_eq!(resized.width, 8);
        assert_eq!(resized.height, 8);
    }

    #[test]
    fn test_flip() {
        let img = test_image();
        let flipped = flip_horizontal(&img);
        assert_eq!(flipped.get(0, 0), img.get(3, 0));
    }

    #[test]
    fn test_ssim_identical() {
        let img = test_image();
        let s = ssim(&img, &img);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_psnr_identical() {
        let img = test_image();
        assert_eq!(psnr(&img, &img), f64::INFINITY);
    }
}
