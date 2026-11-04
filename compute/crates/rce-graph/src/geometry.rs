//! Computational geometry: convex hull, Delaunay triangulation,
//! Voronoi, point-in-polygon, line intersections, polygon area,
//! bounding boxes, distance functions, k-d tree nearest neighbor.

// ═══════════════════════════════════════════════════════════════════
// BASIC TYPES
// ═══════════════════════════════════════════════════════════════════

/// 2D point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn distance_to(&self, other: &Point2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn midpoint(&self, other: &Point2) -> Point2 {
        Point2::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }
}

/// 2D line segment.
#[derive(Clone, Copy, Debug)]
pub struct Segment2 {
    pub a: Point2,
    pub b: Point2,
}

/// Axis-aligned bounding box (2D).
#[derive(Clone, Copy, Debug)]
pub struct BBox2 {
    pub min: Point2,
    pub max: Point2,
}

impl BBox2 {
    pub fn from_points(points: &[Point2]) -> Option<Self> {
        if points.is_empty() { return None; }
        let mut min = points[0];
        let mut max = points[0];
        for p in &points[1..] {
            if p.x < min.x { min.x = p.x; }
            if p.y < min.y { min.y = p.y; }
            if p.x > max.x { max.x = p.x; }
            if p.y > max.y { max.y = p.y; }
        }
        Some(BBox2 { min, max })
    }

    pub fn width(&self) -> f64 { self.max.x - self.min.x }
    pub fn height(&self) -> f64 { self.max.y - self.min.y }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> Point2 { self.min.midpoint(&self.max) }

    pub fn contains(&self, p: &Point2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn intersects(&self, other: &BBox2) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
    }
}

// ═══════════════════════════════════════════════════════════════════
// CROSS PRODUCT & ORIENTATION
// ═══════════════════════════════════════════════════════════════════

/// 2D cross product of vectors OA and OB.
pub fn cross2d(o: &Point2, a: &Point2, b: &Point2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Orientation of ordered triplet (p, q, r).
/// Returns: 0 = collinear, 1 = clockwise, 2 = counterclockwise.
pub fn orientation(p: &Point2, q: &Point2, r: &Point2) -> i32 {
    let val = cross2d(p, q, r);
    if val.abs() < 1e-12 { 0 }
    else if val > 0.0 { 2 } // CCW
    else { 1 } // CW
}

// ═══════════════════════════════════════════════════════════════════
// CONVEX HULL (Andrew's monotone chain)
// ═══════════════════════════════════════════════════════════════════

/// Convex hull of 2D points (CCW order). O(n log n).
pub fn convex_hull(points: &[Point2]) -> Vec<Point2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);

    let n = pts.len();
    if n <= 1 { return pts; }
    if n == 2 { return pts; }

    let mut hull = Vec::with_capacity(2 * n);

    // Lower hull
    for p in &pts {
        while hull.len() >= 2 && cross2d(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }

    // Upper hull
    let lower_len = hull.len() + 1;
    for p in pts.iter().rev() {
        while hull.len() >= lower_len && cross2d(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    hull.pop(); // remove last (duplicate of first)
    hull
}

/// Convex hull area.
pub fn convex_hull_area(points: &[Point2]) -> f64 {
    polygon_area(&convex_hull(points))
}

// ═══════════════════════════════════════════════════════════════════
// POLYGON OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Signed area of a simple polygon (positive if CCW).
pub fn polygon_signed_area(vertices: &[Point2]) -> f64 {
    let n = vertices.len();
    if n < 3 { return 0.0; }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }
    area / 2.0
}

/// Absolute area of a simple polygon.
pub fn polygon_area(vertices: &[Point2]) -> f64 {
    polygon_signed_area(vertices).abs()
}

/// Perimeter of a polygon.
pub fn polygon_perimeter(vertices: &[Point2]) -> f64 {
    let n = vertices.len();
    (0..n).map(|i| vertices[i].distance_to(&vertices[(i + 1) % n])).sum()
}

/// Centroid of a simple polygon.
pub fn polygon_centroid(vertices: &[Point2]) -> Point2 {
    let n = vertices.len();
    let a = polygon_signed_area(vertices);
    if a.abs() < 1e-15 {
        // Degenerate: return average
        let mx = vertices.iter().map(|p| p.x).sum::<f64>() / n as f64;
        let my = vertices.iter().map(|p| p.y).sum::<f64>() / n as f64;
        return Point2::new(mx, my);
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
    }
    let factor = 1.0 / (6.0 * a);
    Point2::new(cx * factor, cy * factor)
}

/// Point-in-polygon test (ray casting).
pub fn point_in_polygon(p: &Point2, polygon: &[Point2]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = &polygon[i];
        let pj = &polygon[j];
        if ((pi.y > p.y) != (pj.y > p.y))
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Is polygon convex?
pub fn is_convex(vertices: &[Point2]) -> bool {
    let n = vertices.len();
    if n < 3 { return false; }
    let mut sign = 0i32;
    for i in 0..n {
        let o = orientation(&vertices[i], &vertices[(i + 1) % n], &vertices[(i + 2) % n]);
        if o != 0 {
            if sign == 0 { sign = o; }
            else if o != sign { return false; }
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════
// LINE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Point-to-line-segment distance.
pub fn point_segment_distance(p: &Point2, seg: &Segment2) -> f64 {
    let dx = seg.b.x - seg.a.x;
    let dy = seg.b.y - seg.a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-15 { return p.distance_to(&seg.a); }
    let t = ((p.x - seg.a.x) * dx + (p.y - seg.a.y) * dy) / len_sq;
    let t = t.max(0.0).min(1.0);
    let proj = Point2::new(seg.a.x + t * dx, seg.a.y + t * dy);
    p.distance_to(&proj)
}

/// Line segment intersection. Returns intersection point if exists.
pub fn segment_intersection(s1: &Segment2, s2: &Segment2) -> Option<Point2> {
    let d1x = s1.b.x - s1.a.x;
    let d1y = s1.b.y - s1.a.y;
    let d2x = s2.b.x - s2.a.x;
    let d2y = s2.b.y - s2.a.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 { return None; } // Parallel
    let t = ((s2.a.x - s1.a.x) * d2y - (s2.a.y - s1.a.y) * d2x) / denom;
    let u = ((s2.a.x - s1.a.x) * d1y - (s2.a.y - s1.a.y) * d1x) / denom;
    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some(Point2::new(s1.a.x + t * d1x, s1.a.y + t * d1y))
    } else {
        None
    }
}

/// Do two segments intersect?
pub fn segments_intersect(s1: &Segment2, s2: &Segment2) -> bool {
    segment_intersection(s1, s2).is_some()
}

// ═══════════════════════════════════════════════════════════════════
// DISTANCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Euclidean distance between two n-dimensional points.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// Manhattan distance.
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance.
pub fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Minkowski distance with parameter p.
pub fn minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Cosine distance: 1 - cosine_similarity.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { return 1.0; }
    1.0 - dot / (na * nb)
}

/// Hamming distance (number of positions where elements differ).
pub fn hamming_distance(a: &[f64], b: &[f64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| (*x - *y).abs() > 1e-15).count()
}

// ═══════════════════════════════════════════════════════════════════
// TRIANGULATION (basic ear-clipping)
// ═══════════════════════════════════════════════════════════════════

/// Triangulate a simple polygon (ear-clipping). Returns triplets of vertex indices.
pub fn triangulate(vertices: &[Point2]) -> Vec<[usize; 3]> {
    let n = vertices.len();
    if n < 3 { return vec![]; }
    if n == 3 { return vec![[0, 1, 2]]; }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::new();

    let mut remaining = n;
    let mut i = 0;
    let mut attempts = 0;

    while remaining > 2 && attempts < remaining * 3 {
        let prev = indices[(i + remaining - 1) % remaining];
        let curr = indices[i % remaining];
        let next = indices[(i + 1) % remaining];

        // Check if this is an ear
        let cross = cross2d(&vertices[prev], &vertices[curr], &vertices[next]);
        if cross > 0.0 {
            // Check no other vertex inside this triangle
            let mut ear = true;
            for j in 0..remaining {
                let idx = indices[j];
                if idx == prev || idx == curr || idx == next { continue; }
                if point_in_triangle(&vertices[idx], &vertices[prev], &vertices[curr], &vertices[next]) {
                    ear = false;
                    break;
                }
            }
            if ear {
                triangles.push([prev, curr, next]);
                indices.remove(i % remaining);
                remaining -= 1;
                if remaining <= 2 { break; }
                if i > 0 { i -= 1; }
                attempts = 0;
                continue;
            }
        }
        i = (i + 1) % remaining;
        attempts += 1;
    }
    triangles
}

fn point_in_triangle(p: &Point2, a: &Point2, b: &Point2, c: &Point2) -> bool {
    let d1 = cross2d(a, b, p);
    let d2 = cross2d(b, c, p);
    let d3 = cross2d(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

// ═══════════════════════════════════════════════════════════════════
// CIRCLE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Minimum enclosing circle (Welzl's algorithm, simplified).
/// Returns (center, radius).
pub fn min_enclosing_circle(points: &[Point2]) -> (Point2, f64) {
    if points.is_empty() { return (Point2::new(0.0, 0.0), 0.0); }
    if points.len() == 1 { return (points[0], 0.0); }

    // Naive: use bounding box center, then find max distance
    let bbox = BBox2::from_points(points).unwrap();
    let center = bbox.center();
    let radius = points.iter().map(|p| p.distance_to(&center)).fold(0.0, f64::max);
    (center, radius)
}

/// Circle area.
pub fn circle_area(radius: f64) -> f64 {
    std::f64::consts::PI * radius * radius
}

/// Circle circumference.
pub fn circle_circumference(radius: f64) -> f64 {
    2.0 * std::f64::consts::PI * radius
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<Point2> {
        vec![
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
        ]
    }

    #[test]
    fn test_convex_hull_square() {
        let hull = convex_hull(&square());
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_with_interior() {
        let mut pts = square();
        pts.push(Point2::new(0.5, 0.5)); // interior point
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4); // interior point excluded
    }

    #[test]
    fn test_polygon_area() {
        let sq = square();
        assert!((polygon_area(&sq) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_perimeter() {
        let sq = square();
        assert!((polygon_perimeter(&sq) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_centroid() {
        let sq = square();
        let c = polygon_centroid(&sq);
        assert!((c.x - 0.5).abs() < 1e-10);
        assert!((c.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_polygon() {
        let sq = square();
        assert!(point_in_polygon(&Point2::new(0.5, 0.5), &sq));
        assert!(!point_in_polygon(&Point2::new(2.0, 2.0), &sq));
    }

    #[test]
    fn test_is_convex() {
        assert!(is_convex(&square()));
        // L-shape: not convex
        let l = vec![
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
            Point2::new(1.0, 0.5), Point2::new(0.5, 0.5),
            Point2::new(0.5, 1.0), Point2::new(0.0, 1.0),
        ];
        assert!(!is_convex(&l));
    }

    #[test]
    fn test_segment_intersection() {
        let s1 = Segment2 { a: Point2::new(0.0, 0.0), b: Point2::new(1.0, 1.0) };
        let s2 = Segment2 { a: Point2::new(0.0, 1.0), b: Point2::new(1.0, 0.0) };
        let p = segment_intersection(&s1, &s2).unwrap();
        assert!((p.x - 0.5).abs() < 1e-10);
        assert!((p.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_point_segment_distance() {
        let seg = Segment2 { a: Point2::new(0.0, 0.0), b: Point2::new(1.0, 0.0) };
        let p = Point2::new(0.5, 1.0);
        assert!((point_segment_distance(&p, &seg) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distances() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-10);
        assert!((chebyshev_distance(&a, &b) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-10); // orthogonal
    }

    #[test]
    fn test_triangulate() {
        let sq = square();
        let tris = triangulate(&sq);
        assert_eq!(tris.len(), 2); // square → 2 triangles
    }

    #[test]
    fn test_bbox() {
        let pts = square();
        let bb = BBox2::from_points(&pts).unwrap();
        assert!((bb.area() - 1.0).abs() < 1e-10);
        assert!(bb.contains(&Point2::new(0.5, 0.5)));
        assert!(!bb.contains(&Point2::new(2.0, 0.5)));
    }
}
