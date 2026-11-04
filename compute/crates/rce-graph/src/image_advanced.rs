//! Advanced image processing: connected components, Hough transform,
//! morphological thinning, distance transform, template matching,
//! image moments, contour detection.

use crate::image_processing::GrayImage;
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════
// CONNECTED COMPONENTS
// ═══════════════════════════════════════════════════════════════════

/// Connected component labeling (4-connected).
/// Input: binary image (0 = background, >0 = foreground).
/// Returns (label_image, n_components).
pub fn connected_components(img: &GrayImage) -> (Vec<usize>, usize) {
    let (w, h) = (img.width, img.height);
    let mut labels = vec![0usize; w * h];
    let mut next_label = 1;

    for y in 0..h {
        for x in 0..w {
            if img.get(x, y) < 0.5 || labels[y * w + x] > 0 { continue; }
            // BFS flood fill
            let mut queue = VecDeque::new();
            queue.push_back((x, y));
            labels[y * w + x] = next_label;
            while let Some((cx, cy)) = queue.pop_front() {
                let neighbors = [(cx.wrapping_sub(1), cy), (cx + 1, cy), (cx, cy.wrapping_sub(1)), (cx, cy + 1)];
                for (nx, ny) in neighbors {
                    if nx < w && ny < h && img.get(nx, ny) >= 0.5 && labels[ny * w + nx] == 0 {
                        labels[ny * w + nx] = next_label;
                        queue.push_back((nx, ny));
                    }
                }
            }
            next_label += 1;
        }
    }
    (labels, next_label - 1)
}

/// Count pixels in each connected component. Returns vec indexed by label.
pub fn component_sizes(labels: &[usize], n_components: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; n_components + 1];
    for &l in labels { if l > 0 { sizes[l] += 1; } }
    sizes
}

/// Largest connected component label.
pub fn largest_component(labels: &[usize], n_components: usize) -> usize {
    let sizes = component_sizes(labels, n_components);
    sizes.iter().enumerate().skip(1).max_by_key(|(_, s)| **s).map(|(i, _)| i).unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════
// DISTANCE TRANSFORM
// ═══════════════════════════════════════════════════════════════════

/// Euclidean distance transform (approximation via two-pass).
/// Input: binary image (0 = background). Returns distance to nearest background.
pub fn distance_transform(img: &GrayImage) -> GrayImage {
    let (w, h) = (img.width, img.height);
    let mut dist = vec![f64::INFINITY; w * h];
    // Background pixels get distance 0
    for y in 0..h {
        for x in 0..w {
            if img.get(x, y) < 0.5 { dist[y * w + x] = 0.0; }
        }
    }
    // Forward pass
    for y in 0..h {
        for x in 0..w {
            if x > 0 { dist[y * w + x] = dist[y * w + x].min(dist[y * w + x - 1] + 1.0); }
            if y > 0 { dist[y * w + x] = dist[y * w + x].min(dist[(y - 1) * w + x] + 1.0); }
        }
    }
    // Backward pass
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if x + 1 < w { dist[y * w + x] = dist[y * w + x].min(dist[y * w + x + 1] + 1.0); }
            if y + 1 < h { dist[y * w + x] = dist[y * w + x].min(dist[(y + 1) * w + x] + 1.0); }
        }
    }
    GrayImage::from_data(dist, w, h)
}

// ═══════════════════════════════════════════════════════════════════
// HOUGH TRANSFORM (line detection)
// ═══════════════════════════════════════════════════════════════════

/// Hough transform for line detection.
/// Input: edge image. Returns vec of (rho, theta) pairs above threshold.
pub fn hough_lines(edges: &GrayImage, n_rho: usize, n_theta: usize, threshold: usize) -> Vec<(f64, f64)> {
    let (w, h) = (edges.width, edges.height);
    let max_rho = ((w * w + h * h) as f64).sqrt();
    let theta_step = std::f64::consts::PI / n_theta as f64;
    let rho_step = 2.0 * max_rho / n_rho as f64;
    let mut accumulator = vec![0usize; n_rho * n_theta];

    // Precompute sin/cos
    let thetas: Vec<f64> = (0..n_theta).map(|t| t as f64 * theta_step).collect();
    let sin_t: Vec<f64> = thetas.iter().map(|t| t.sin()).collect();
    let cos_t: Vec<f64> = thetas.iter().map(|t| t.cos()).collect();

    for y in 0..h {
        for x in 0..w {
            if edges.get(x, y) < 0.5 { continue; }
            for t in 0..n_theta {
                let rho = x as f64 * cos_t[t] + y as f64 * sin_t[t];
                let rho_idx = ((rho + max_rho) / rho_step) as usize;
                if rho_idx < n_rho { accumulator[rho_idx * n_theta + t] += 1; }
            }
        }
    }

    let mut lines = Vec::new();
    for r in 0..n_rho {
        for t in 0..n_theta {
            if accumulator[r * n_theta + t] >= threshold {
                let rho = r as f64 * rho_step - max_rho;
                lines.push((rho, thetas[t]));
            }
        }
    }
    lines
}

/// Hough transform for circle detection (simplified).
/// Returns vec of (cx, cy, radius) above threshold.
pub fn hough_circles(edges: &GrayImage, min_radius: usize, max_radius: usize, threshold: usize) -> Vec<(usize, usize, usize)> {
    let (w, h) = (edges.width, edges.height);
    let mut circles = Vec::new();

    for radius in min_radius..=max_radius {
        let mut accumulator = vec![0usize; w * h];
        for y in 0..h {
            for x in 0..w {
                if edges.get(x, y) < 0.5 { continue; }
                // Vote for all possible centers at this radius
                for angle_step in 0..36 {
                    let angle = angle_step as f64 * std::f64::consts::PI * 2.0 / 36.0;
                    let cx = (x as f64 + radius as f64 * angle.cos()).round() as i64;
                    let cy = (y as f64 + radius as f64 * angle.sin()).round() as i64;
                    if cx >= 0 && cx < w as i64 && cy >= 0 && cy < h as i64 {
                        accumulator[cy as usize * w + cx as usize] += 1;
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                if accumulator[y * w + x] >= threshold {
                    circles.push((x, y, radius));
                }
            }
        }
    }
    circles
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE MOMENTS
// ═══════════════════════════════════════════════════════════════════

/// Raw image moment M_{pq}.
pub fn image_moment(img: &GrayImage, p: u32, q: u32) -> f64 {
    let mut m = 0.0;
    for y in 0..img.height {
        for x in 0..img.width {
            m += (x as f64).powi(p as i32) * (y as f64).powi(q as i32) * img.get(x, y);
        }
    }
    m
}

/// Centroid of image: (x̄, ȳ) = (M10/M00, M01/M00).
pub fn image_centroid(img: &GrayImage) -> (f64, f64) {
    let m00 = image_moment(img, 0, 0);
    if m00 < 1e-15 { return (0.0, 0.0); }
    (image_moment(img, 1, 0) / m00, image_moment(img, 0, 1) / m00)
}

/// Central moment μ_{pq}.
pub fn central_moment(img: &GrayImage, p: u32, q: u32) -> f64 {
    let (cx, cy) = image_centroid(img);
    let mut m = 0.0;
    for y in 0..img.height {
        for x in 0..img.width {
            m += (x as f64 - cx).powi(p as i32) * (y as f64 - cy).powi(q as i32) * img.get(x, y);
        }
    }
    m
}

/// Normalized central moment η_{pq} = μ_{pq} / μ_{00}^{(p+q)/2 + 1}.
pub fn normalized_moment(img: &GrayImage, p: u32, q: u32) -> f64 {
    let mu00 = central_moment(img, 0, 0);
    if mu00.abs() < 1e-15 { return 0.0; }
    let gamma = (p + q) as f64 / 2.0 + 1.0;
    central_moment(img, p, q) / mu00.powf(gamma)
}

/// Hu invariant moments (7 moments, rotation/scale/translation invariant).
pub fn hu_moments(img: &GrayImage) -> [f64; 7] {
    let n20 = normalized_moment(img, 2, 0);
    let n02 = normalized_moment(img, 0, 2);
    let n11 = normalized_moment(img, 1, 1);
    let n30 = normalized_moment(img, 3, 0);
    let n12 = normalized_moment(img, 1, 2);
    let n21 = normalized_moment(img, 2, 1);
    let n03 = normalized_moment(img, 0, 3);

    [
        n20 + n02,
        (n20 - n02).powi(2) + 4.0 * n11 * n11,
        (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2),
        (n30 + n12).powi(2) + (n21 + n03).powi(2),
        (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
            + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2)),
        (n20 - n02) * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
            + 4.0 * n11 * (n30 + n12) * (n21 + n03),
        (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
            - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2)),
    ]
}

// ═══════════════════════════════════════════════════════════════════
// TEMPLATE MATCHING
// ═══════════════════════════════════════════════════════════════════

/// Template matching via normalized cross-correlation (NCC).
/// Returns correlation map (smaller than image by template size).
pub fn template_match_ncc(image: &GrayImage, template: &GrayImage) -> GrayImage {
    let (iw, ih) = (image.width, image.height);
    let (tw, th) = (template.width, template.height);
    if tw > iw || th > ih { return GrayImage::new(0, 0); }
    let ow = iw - tw + 1;
    let oh = ih - th + 1;
    let mut result = GrayImage::new(ow, oh);

    let tmean: f64 = template.data.iter().sum::<f64>() / template.data.len() as f64;
    let tstd: f64 = template.data.iter().map(|v| (v - tmean).powi(2)).sum::<f64>().sqrt();

    for y in 0..oh {
        for x in 0..ow {
            let mut sum = 0.0;
            let mut patch_sum = 0.0;
            let mut patch_sq = 0.0;
            for ty in 0..th {
                for tx in 0..tw {
                    let iv = image.get(x + tx, y + ty);
                    let tv = template.get(tx, ty);
                    patch_sum += iv;
                    patch_sq += iv * iv;
                    sum += iv * tv;
                }
            }
            let n = (tw * th) as f64;
            let pmean = patch_sum / n;
            let pstd = (patch_sq - n * pmean * pmean).max(0.0).sqrt();
            let ncc = if pstd > 1e-10 && tstd > 1e-10 {
                (sum - n * pmean * tmean) / (pstd * tstd)
            } else { 0.0 };
            result.set(x, y, ncc);
        }
    }
    result
}

/// Find best template match location. Returns (x, y, score).
pub fn template_match_best(image: &GrayImage, template: &GrayImage) -> (usize, usize, f64) {
    let ncc = template_match_ncc(image, template);
    let mut best_x = 0;
    let mut best_y = 0;
    let mut best_score = f64::NEG_INFINITY;
    for y in 0..ncc.height {
        for x in 0..ncc.width {
            let v = ncc.get(x, y);
            if v > best_score { best_score = v; best_x = x; best_y = y; }
        }
    }
    (best_x, best_y, best_score)
}

// ═══════════════════════════════════════════════════════════════════
// CONTOUR / BOUNDARY DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Extract boundary pixels from binary image. Returns list of (x, y).
pub fn boundary_pixels(img: &GrayImage) -> Vec<(usize, usize)> {
    let (w, h) = (img.width, img.height);
    let mut boundary = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if img.get(x, y) < 0.5 { continue; }
            let is_boundary = x == 0 || y == 0 || x == w - 1 || y == h - 1
                || img.get(x - 1, y) < 0.5 || img.get(x + 1, y) < 0.5
                || img.get(x, y - 1) < 0.5 || img.get(x, y + 1) < 0.5;
            if is_boundary { boundary.push((x, y)); }
        }
    }
    boundary
}

/// Perimeter of a binary region (count boundary pixels).
pub fn region_perimeter(img: &GrayImage) -> usize {
    boundary_pixels(img).len()
}

/// Area of a binary region (count foreground pixels).
pub fn region_area(img: &GrayImage) -> usize {
    img.data.iter().filter(|&&v| v >= 0.5).count()
}

/// Circularity: 4π·Area / Perimeter².
pub fn region_circularity(img: &GrayImage) -> f64 {
    let area = region_area(img) as f64;
    let perim = region_perimeter(img) as f64;
    if perim < 1e-15 { return 0.0; }
    4.0 * std::f64::consts::PI * area / (perim * perim)
}

/// Bounding box of foreground region. Returns (min_x, min_y, max_x, max_y).
pub fn region_bbox(img: &GrayImage) -> (usize, usize, usize, usize) {
    let (mut min_x, mut min_y) = (img.width, img.height);
    let (mut max_x, mut max_y) = (0, 0);
    for y in 0..img.height {
        for x in 0..img.width {
            if img.get(x, y) >= 0.5 {
                min_x = min_x.min(x); max_x = max_x.max(x);
                min_y = min_y.min(y); max_y = max_y.max(y);
            }
        }
    }
    (min_x, min_y, max_x, max_y)
}

/// Eccentricity of region (ratio of eigenvalues of covariance).
pub fn region_eccentricity(img: &GrayImage) -> f64 {
    let mu20 = central_moment(img, 2, 0);
    let mu02 = central_moment(img, 0, 2);
    let mu11 = central_moment(img, 1, 1);
    let trace = mu20 + mu02;
    let det = mu20 * mu02 - mu11 * mu11;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda1 = (trace + disc) / 2.0;
    let lambda2 = (trace - disc) / 2.0;
    if lambda1 < 1e-15 { return 0.0; }
    (1.0 - lambda2 / lambda1).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_image() -> GrayImage {
        GrayImage::from_data(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
        ], 5, 5)
    }

    #[test]
    fn test_connected_components() {
        let img = binary_image();
        let (labels, n) = connected_components(&img);
        assert_eq!(n, 2); // two separate regions
    }

    #[test]
    fn test_component_sizes() {
        let img = binary_image();
        let (labels, n) = connected_components(&img);
        let sizes = component_sizes(&labels, n);
        assert_eq!(sizes[1], 4); // 2×2 block
        assert_eq!(sizes[2], 2); // 1×2 block
    }

    #[test]
    fn test_distance_transform() {
        let img = binary_image();
        let dt = distance_transform(&img);
        assert_eq!(dt.get(0, 0), 0.0); // background → 0
        assert!(dt.get(1, 1) > 0.0); // foreground → > 0
    }

    #[test]
    fn test_image_centroid() {
        let img = GrayImage::from_data(vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ], 3, 3);
        let (cx, cy) = image_centroid(&img);
        assert!((cx - 1.0).abs() < 1e-10);
        assert!((cy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hu_moments() {
        let img = binary_image();
        let hu = hu_moments(&img);
        assert!(hu[0] > 0.0); // first Hu moment should be positive
    }

    #[test]
    fn test_boundary_pixels() {
        let img = binary_image();
        let boundary = boundary_pixels(&img);
        assert!(!boundary.is_empty());
    }

    #[test]
    fn test_region_area() {
        let img = binary_image();
        assert_eq!(region_area(&img), 6); // 4 + 2 foreground pixels
    }

    #[test]
    fn test_region_bbox() {
        let img = GrayImage::from_data(vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ], 4, 4);
        let (min_x, min_y, max_x, max_y) = region_bbox(&img);
        assert_eq!((min_x, min_y, max_x, max_y), (1, 1, 2, 2));
    }
}
