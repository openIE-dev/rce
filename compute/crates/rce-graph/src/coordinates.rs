//! Coordinate transforms: Cartesian↔Polar↔Spherical↔Cylindrical,
//! geographic (lat/lon↔UTM, Haversine), rotation matrices, quaternions.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// 2D COORDINATE TRANSFORMS
// ═══════════════════════════════════════════════════════════════════

/// Cartesian (x,y) → Polar (r, theta). Theta in radians.
pub fn cart2pol(x: f64, y: f64) -> (f64, f64) {
    ((x * x + y * y).sqrt(), y.atan2(x))
}

/// Polar (r, theta) → Cartesian (x, y). Theta in radians.
pub fn pol2cart(r: f64, theta: f64) -> (f64, f64) {
    (r * theta.cos(), r * theta.sin())
}

// ═══════════════════════════════════════════════════════════════════
// 3D COORDINATE TRANSFORMS
// ═══════════════════════════════════════════════════════════════════

/// Cartesian (x,y,z) → Spherical (r, theta, phi).
/// theta = polar angle from z-axis [0, pi], phi = azimuthal angle [−pi, pi].
pub fn cart2sph(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = if r < 1e-15 { 0.0 } else { (z / r).acos() };
    let phi = y.atan2(x);
    (r, theta, phi)
}

/// Spherical (r, theta, phi) → Cartesian (x, y, z).
pub fn sph2cart(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    let x = r * theta.sin() * phi.cos();
    let y = r * theta.sin() * phi.sin();
    let z = r * theta.cos();
    (x, y, z)
}

/// Cartesian (x,y,z) → Cylindrical (rho, phi, z).
pub fn cart2cyl(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    ((x * x + y * y).sqrt(), y.atan2(x), z)
}

/// Cylindrical (rho, phi, z) → Cartesian (x, y, z).
pub fn cyl2cart(rho: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    (rho * phi.cos(), rho * phi.sin(), z)
}

/// Spherical → Cylindrical.
pub fn sph2cyl(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    (r * theta.sin(), phi, r * theta.cos())
}

/// Cylindrical → Spherical.
pub fn cyl2sph(rho: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    let r = (rho * rho + z * z).sqrt();
    let theta = if r < 1e-15 { 0.0 } else { rho.atan2(z) };
    (r, theta, phi)
}

// ═══════════════════════════════════════════════════════════════════
// ANGLE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════

/// Degrees → Radians.
pub fn deg2rad(deg: f64) -> f64 { deg * PI / 180.0 }

/// Radians → Degrees.
pub fn rad2deg(rad: f64) -> f64 { rad * 180.0 / PI }

/// Normalize angle to [0, 2π).
pub fn normalize_angle(rad: f64) -> f64 {
    let mut a = rad % (2.0 * PI);
    if a < 0.0 { a += 2.0 * PI; }
    a
}

/// Normalize angle to [−π, π).
pub fn wrap_angle(rad: f64) -> f64 {
    let mut a = (rad + PI) % (2.0 * PI);
    if a < 0.0 { a += 2.0 * PI; }
    a - PI
}

/// Shortest angular distance between two angles (result in [−π, π]).
pub fn angular_distance(a: f64, b: f64) -> f64 {
    wrap_angle(b - a)
}

// ═══════════════════════════════════════════════════════════════════
// GEOGRAPHIC / GEODESIC
// ═══════════════════════════════════════════════════════════════════

const EARTH_RADIUS_M: f64 = 6_371_000.0; // mean Earth radius in meters

/// Haversine distance between two (lat, lon) points in degrees. Returns meters.
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1, la2, lo2) = (deg2rad(lat1), deg2rad(lon1), deg2rad(lat2), deg2rad(lon2));
    let dlat = la2 - la1;
    let dlon = lo2 - lo1;
    let a = (dlat / 2.0).sin().powi(2) + la1.cos() * la2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_M * c
}

/// Bearing from point 1 to point 2 (degrees, 0 = North, clockwise).
pub fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1, la2, lo2) = (deg2rad(lat1), deg2rad(lon1), deg2rad(lat2), deg2rad(lon2));
    let dlon = lo2 - lo1;
    let y = dlon.sin() * la2.cos();
    let x = la1.cos() * la2.sin() - la1.sin() * la2.cos() * dlon.cos();
    let bearing_rad = y.atan2(x);
    (rad2deg(bearing_rad) + 360.0) % 360.0
}

/// Destination point given start (lat, lon in degrees), bearing (degrees), distance (meters).
pub fn destination_point(lat: f64, lon: f64, bearing_deg: f64, distance_m: f64) -> (f64, f64) {
    let la = deg2rad(lat);
    let lo = deg2rad(lon);
    let br = deg2rad(bearing_deg);
    let d = distance_m / EARTH_RADIUS_M;

    let lat2 = (la.sin() * d.cos() + la.cos() * d.sin() * br.cos()).asin();
    let lon2 = lo + (br.sin() * d.sin() * la.cos()).atan2(d.cos() - la.sin() * lat2.sin());
    (rad2deg(lat2), rad2deg(lon2))
}

/// Midpoint between two geographic coordinates (degrees).
pub fn geographic_midpoint(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64) {
    let (la1, lo1, la2, lo2) = (deg2rad(lat1), deg2rad(lon1), deg2rad(lat2), deg2rad(lon2));
    let dlon = lo2 - lo1;
    let bx = la2.cos() * dlon.cos();
    let by = la2.cos() * dlon.sin();
    let lat_m = (la1.sin() + la2.sin()).atan2(((la1.cos() + bx).powi(2) + by.powi(2)).sqrt());
    let lon_m = lo1 + by.atan2(la1.cos() + bx);
    (rad2deg(lat_m), rad2deg(lon_m))
}

// ═══════════════════════════════════════════════════════════════════
// ROTATION MATRICES (2D & 3D)
// ═══════════════════════════════════════════════════════════════════

/// 2D rotation matrix for angle theta (radians). Returns [[c, -s], [s, c]] flattened row-major.
pub fn rotation_matrix_2d(theta: f64) -> [f64; 4] {
    let (s, c) = (theta.sin(), theta.cos());
    [c, -s, s, c]
}

/// Rotate 2D point (x, y) by angle theta (radians).
pub fn rotate_2d(x: f64, y: f64, theta: f64) -> (f64, f64) {
    let (s, c) = (theta.sin(), theta.cos());
    (x * c - y * s, x * s + y * c)
}

/// 3D rotation matrix about X-axis (radians). Returns 3×3 row-major.
pub fn rotation_x(theta: f64) -> [f64; 9] {
    let (s, c) = (theta.sin(), theta.cos());
    [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
}

/// 3D rotation matrix about Y-axis.
pub fn rotation_y(theta: f64) -> [f64; 9] {
    let (s, c) = (theta.sin(), theta.cos());
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

/// 3D rotation matrix about Z-axis.
pub fn rotation_z(theta: f64) -> [f64; 9] {
    let (s, c) = (theta.sin(), theta.cos());
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}

/// Rotate 3D point by matrix (3×3 row-major).
pub fn rotate_3d(point: [f64; 3], mat: &[f64; 9]) -> [f64; 3] {
    [
        mat[0] * point[0] + mat[1] * point[1] + mat[2] * point[2],
        mat[3] * point[0] + mat[4] * point[1] + mat[5] * point[2],
        mat[6] * point[0] + mat[7] * point[1] + mat[8] * point[2],
    ]
}

// ═══════════════════════════════════════════════════════════════════
// QUATERNIONS
// ═══════════════════════════════════════════════════════════════════

/// Quaternion [w, x, y, z].
pub type Quaternion = [f64; 4];

/// Create quaternion from axis-angle (axis must be unit vector, angle in radians).
pub fn quat_from_axis_angle(axis: [f64; 3], angle: f64) -> Quaternion {
    let half = angle / 2.0;
    let s = half.sin();
    [half.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Quaternion multiplication.
pub fn quat_mul(a: &Quaternion, b: &Quaternion) -> Quaternion {
    [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ]
}

/// Quaternion conjugate.
pub fn quat_conjugate(q: &Quaternion) -> Quaternion {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Quaternion norm.
pub fn quat_norm(q: &Quaternion) -> f64 {
    (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt()
}

/// Normalize quaternion to unit length.
pub fn quat_normalize(q: &Quaternion) -> Quaternion {
    let n = quat_norm(q);
    if n < 1e-15 { return [1.0, 0.0, 0.0, 0.0]; }
    [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
}

/// Rotate a 3D point by a unit quaternion.
pub fn quat_rotate(q: &Quaternion, point: [f64; 3]) -> [f64; 3] {
    let p = [0.0, point[0], point[1], point[2]];
    let qc = quat_conjugate(q);
    let r = quat_mul(&quat_mul(q, &p), &qc);
    [r[1], r[2], r[3]]
}

/// Quaternion → rotation matrix (3×3 row-major).
pub fn quat_to_matrix(q: &Quaternion) -> [f64; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),
        2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y),
    ]
}

/// Rotation matrix (3×3 row-major) → quaternion.
pub fn matrix_to_quat(m: &[f64; 9]) -> Quaternion {
    let trace = m[0] + m[4] + m[8];
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [0.25 / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s]
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = 2.0 * (1.0 + m[0] - m[4] - m[8]).sqrt();
        [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]
    } else if m[4] > m[8] {
        let s = 2.0 * (1.0 + m[4] - m[0] - m[8]).sqrt();
        [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]
    } else {
        let s = 2.0 * (1.0 + m[8] - m[0] - m[4]).sqrt();
        [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]
    }
}

/// Euler angles (roll, pitch, yaw in radians) → quaternion.
pub fn euler_to_quat(roll: f64, pitch: f64, yaw: f64) -> Quaternion {
    let (sr, cr) = (roll / 2.0).sin_cos();
    let (sp, cp) = (pitch / 2.0).sin_cos();
    let (sy, cy) = (yaw / 2.0).sin_cos();
    [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]
}

/// Quaternion → Euler angles (roll, pitch, yaw in radians).
pub fn quat_to_euler(q: &Quaternion) -> (f64, f64, f64) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let sinr = 2.0 * (w * x + y * z);
    let cosr = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr.atan2(cosr);

    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 { sinp.signum() * PI / 2.0 } else { sinp.asin() };

    let siny = 2.0 * (w * z + x * y);
    let cosy = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny.atan2(cosy);

    (roll, pitch, yaw)
}

/// Spherical linear interpolation between two quaternions.
pub fn slerp(a: &Quaternion, b: &Quaternion, t: f64) -> Quaternion {
    let mut dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    let mut b2 = *b;
    if dot < 0.0 {
        dot = -dot;
        b2 = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        // Linear interpolation for very close quaternions
        let r = quat_normalize(&[
            a[0] + t * (b2[0] - a[0]),
            a[1] + t * (b2[1] - a[1]),
            a[2] + t * (b2[2] - a[2]),
            a[3] + t * (b2[3] - a[3]),
        ]);
        return r;
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    [
        wa * a[0] + wb * b2[0],
        wa * a[1] + wb * b2[1],
        wa * a[2] + wb * b2[2],
        wa * a[3] + wb * b2[3],
    ]
}

// ═══════════════════════════════════════════════════════════════════
// HOMOGENEOUS COORDINATES
// ═══════════════════════════════════════════════════════════════════

/// Create 4×4 translation matrix (row-major).
pub fn translation_matrix(tx: f64, ty: f64, tz: f64) -> [f64; 16] {
    [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Create 4×4 scaling matrix (row-major).
pub fn scale_matrix(sx: f64, sy: f64, sz: f64) -> [f64; 16] {
    [
        sx,  0.0, 0.0, 0.0,
        0.0, sy,  0.0, 0.0,
        0.0, 0.0, sz,  0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Multiply two 4×4 matrices (row-major).
pub fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut r = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
    r
}

/// Transform a 3D point by a 4×4 matrix (homogeneous divide).
pub fn transform_point(mat: &[f64; 16], p: [f64; 3]) -> [f64; 3] {
    let w = mat[12] * p[0] + mat[13] * p[1] + mat[14] * p[2] + mat[15];
    let iw = if w.abs() < 1e-15 { 1.0 } else { 1.0 / w };
    [
        (mat[0] * p[0] + mat[1] * p[1] + mat[2] * p[2] + mat[3]) * iw,
        (mat[4] * p[0] + mat[5] * p[1] + mat[6] * p[2] + mat[7]) * iw,
        (mat[8] * p[0] + mat[9] * p[1] + mat[10] * p[2] + mat[11]) * iw,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cart2pol_roundtrip() {
        let (r, theta) = cart2pol(3.0, 4.0);
        assert!((r - 5.0).abs() < 1e-10);
        let (x, y) = pol2cart(r, theta);
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cart2sph_roundtrip() {
        let (r, theta, phi) = cart2sph(1.0, 2.0, 3.0);
        let (x, y, z) = sph2cart(r, theta, phi);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cart2cyl_roundtrip() {
        let (rho, phi, z) = cart2cyl(3.0, 4.0, 5.0);
        assert!((rho - 5.0).abs() < 1e-10);
        let (x, y, z2) = cyl2cart(rho, phi, z);
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 4.0).abs() < 1e-10);
        assert!((z2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_deg_rad() {
        assert!((deg2rad(180.0) - PI).abs() < 1e-10);
        assert!((rad2deg(PI) - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_haversine() {
        // NYC to London: ~5570 km
        let d = haversine(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((d - 5_570_000.0).abs() < 50_000.0); // within 50km
    }

    #[test]
    fn test_bearing() {
        // Due east: bearing should be ~90
        let b = bearing(0.0, 0.0, 0.0, 1.0);
        assert!((b - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_quaternion_rotation() {
        // Rotate (1,0,0) 90° about Z-axis → (0,1,0)
        let q = quat_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let result = quat_rotate(&q, [1.0, 0.0, 0.0]);
        assert!((result[0]).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2]).abs() < 1e-10);
    }

    #[test]
    fn test_slerp() {
        let a = [1.0, 0.0, 0.0, 0.0]; // identity
        let b = quat_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let mid = slerp(&a, &b, 0.5);
        // Should be 45° rotation about Z
        let result = quat_rotate(&mid, [1.0, 0.0, 0.0]);
        let expected_angle = PI / 4.0;
        assert!((result[0] - expected_angle.cos()).abs() < 1e-6);
        assert!((result[1] - expected_angle.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_euler_quat_roundtrip() {
        let (roll, pitch, yaw) = (0.1, 0.2, 0.3);
        let q = euler_to_quat(roll, pitch, yaw);
        let (r2, p2, y2) = quat_to_euler(&q);
        assert!((r2 - roll).abs() < 1e-10);
        assert!((p2 - pitch).abs() < 1e-10);
        assert!((y2 - yaw).abs() < 1e-10);
    }

    #[test]
    fn test_mat4_transform() {
        let t = translation_matrix(1.0, 2.0, 3.0);
        let p = transform_point(&t, [0.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 2.0).abs() < 1e-10);
        assert!((p[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(-PI / 2.0) - 3.0 * PI / 2.0).abs() < 1e-10);
        assert!((normalize_angle(5.0 * PI) - PI).abs() < 1e-10);
    }
}
