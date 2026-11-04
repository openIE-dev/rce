//! Geospatial: map projections, GeoJSON helpers, spatial indexing,
//! geocoding utilities, area/distance on ellipsoid.

use std::f64::consts::PI;

// WGS84 ellipsoid parameters
const WGS84_A: f64 = 6_378_137.0;       // semi-major axis (m)
const WGS84_F: f64 = 1.0 / 298.257223563; // flattening
const WGS84_B: f64 = 6_356_752.314245;    // semi-minor axis (m)
const WGS84_E2: f64 = 0.00669437999014;   // eccentricity squared

// ═══════════════════════════════════════════════════════════════════
// VINCENTY DISTANCE (ellipsoidal)
// ═══════════════════════════════════════════════════════════════════

/// Vincenty distance between two (lat, lon) points in degrees. Returns meters.
pub fn vincenty_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (la1, lo1) = (lat1.to_radians(), lon1.to_radians());
    let (la2, lo2) = (lat2.to_radians(), lon2.to_radians());
    let f = WGS84_F;
    let u1 = ((1.0 - f) * la1.tan()).atan();
    let u2 = ((1.0 - f) * la2.tan()).atan();
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();

    let mut lambda = lo2 - lo1;
    for _ in 0..100 {
        let sin_lam = lambda.sin();
        let cos_lam = lambda.cos();
        let sin_sigma = ((cos_u2 * sin_lam).powi(2) + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam).powi(2)).sqrt();
        if sin_sigma < 1e-12 { return 0.0; }
        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;
        let cos_2sm = if cos2_alpha > 1e-15 { cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha } else { 0.0 };
        let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
        let prev = lambda;
        lambda = (lo2 - lo1) + (1.0 - c) * f * sin_alpha * (sigma + c * sin_sigma * (cos_2sm + c * cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm)));
        if (lambda - prev).abs() < 1e-12 { break; }
    }

    let _u_sq = (1.0 - WGS84_E2) * ((cos_u1 * (lo2 - lo1).cos() * cos_u2 - sin_u1 * sin_u2).powi(2) + (cos_u2 * (lo2 - lo1).sin()).powi(2));
    // Simplified final calculation
    let sin_lam = lambda.sin();
    let cos_lam = lambda.cos();
    let sin_sigma = ((cos_u2 * sin_lam).powi(2) + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam).powi(2)).sqrt();
    let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam;
    let sigma = sin_sigma.atan2(cos_sigma);
    let cos2_alpha = 1.0 - (cos_u1 * cos_u2 * sin_lam / sin_sigma).powi(2);
    let u_sq2 = cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
    let a_coeff = 1.0 + u_sq2 / 16384.0 * (4096.0 + u_sq2 * (-768.0 + u_sq2 * (320.0 - 175.0 * u_sq2)));
    let b_coeff = u_sq2 / 1024.0 * (256.0 + u_sq2 * (-128.0 + u_sq2 * (74.0 - 47.0 * u_sq2)));
    let cos_2sm = if cos2_alpha > 1e-15 { cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha } else { 0.0 };
    let delta_sigma = b_coeff * sin_sigma * (cos_2sm + b_coeff / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm)));
    WGS84_B * a_coeff * (sigma - delta_sigma)
}

// ═══════════════════════════════════════════════════════════════════
// MAP PROJECTIONS
// ═══════════════════════════════════════════════════════════════════

/// Mercator projection: (lat, lon) degrees → (x, y) meters.
pub fn mercator_forward(lat: f64, lon: f64) -> (f64, f64) {
    let x = WGS84_A * lon.to_radians();
    let lat_rad = lat.to_radians();
    let y = WGS84_A * ((PI / 4.0 + lat_rad / 2.0).tan()).ln();
    (x, y)
}

/// Mercator inverse: (x, y) meters → (lat, lon) degrees.
pub fn mercator_inverse(x: f64, y: f64) -> (f64, f64) {
    let lon = (x / WGS84_A).to_degrees();
    let lat = (2.0 * (y / WGS84_A).exp().atan() - PI / 2.0).to_degrees();
    (lat, lon)
}

/// Lambert Conformal Conic projection (simplified, single standard parallel).
pub fn lambert_forward(lat: f64, lon: f64, lat0: f64, lon0: f64) -> (f64, f64) {
    let phi = lat.to_radians();
    let lam = lon.to_radians();
    let phi0 = lat0.to_radians();
    let lam0 = lon0.to_radians();
    let n = phi0.sin();
    let f_val = phi0.cos() * (PI / 4.0 + phi0 / 2.0).tan().powf(n) / n;
    let rho = WGS84_A * f_val / (PI / 4.0 + phi / 2.0).tan().powf(n);
    let rho0 = WGS84_A * f_val / (PI / 4.0 + phi0 / 2.0).tan().powf(n);
    let x = rho * (n * (lam - lam0)).sin();
    let y = rho0 - rho * (n * (lam - lam0)).cos();
    (x, y)
}

/// Equirectangular projection (plate carrée).
pub fn equirectangular_forward(lat: f64, lon: f64) -> (f64, f64) {
    (lon.to_radians() * WGS84_A, lat.to_radians() * WGS84_A)
}

/// Equirectangular inverse.
pub fn equirectangular_inverse(x: f64, y: f64) -> (f64, f64) {
    ((y / WGS84_A).to_degrees(), (x / WGS84_A).to_degrees())
}

/// Stereographic projection from (lat, lon) to (x, y).
pub fn stereographic_forward(lat: f64, lon: f64, lat0: f64, lon0: f64) -> (f64, f64) {
    let phi = lat.to_radians();
    let lam = lon.to_radians();
    let phi0 = lat0.to_radians();
    let lam0 = lon0.to_radians();
    let k = 2.0 / (1.0 + phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * (lam - lam0).cos());
    let x = WGS84_A * k * phi.cos() * (lam - lam0).sin();
    let y = WGS84_A * k * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * (lam - lam0).cos());
    (x, y)
}

/// Azimuthal equidistant projection.
pub fn azimuthal_equidistant_forward(lat: f64, lon: f64, lat0: f64, lon0: f64) -> (f64, f64) {
    let phi = lat.to_radians();
    let lam = lon.to_radians();
    let phi0 = lat0.to_radians();
    let lam0 = lon0.to_radians();
    let c = (phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * (lam - lam0).cos()).acos();
    if c.abs() < 1e-10 { return (0.0, 0.0); }
    let k = c / c.sin();
    let x = WGS84_A * k * phi.cos() * (lam - lam0).sin();
    let y = WGS84_A * k * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * (lam - lam0).cos());
    (x, y)
}

// ═══════════════════════════════════════════════════════════════════
// UTM (Universal Transverse Mercator)
// ═══════════════════════════════════════════════════════════════════

/// Determine UTM zone number from longitude.
pub fn utm_zone(longitude: f64) -> u32 {
    ((longitude + 180.0) / 6.0).floor() as u32 + 1
}

/// Convert (lat, lon) in degrees to UTM (easting, northing, zone).
pub fn lat_lon_to_utm(lat: f64, lon: f64) -> (f64, f64, u32) {
    let zone = utm_zone(lon);
    let lon0 = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0;
    let phi = lat.to_radians();
    let lam = (lon - lon0).to_radians();
    let n = WGS84_A / (1.0 - WGS84_E2 * phi.sin().powi(2)).sqrt();
    let t = phi.tan();
    let c = WGS84_E2 / (1.0 - WGS84_E2) * phi.cos().powi(2);
    let a_coeff = lam * phi.cos();
    let m = WGS84_A * ((1.0 - WGS84_E2/4.0 - 3.0*WGS84_E2*WGS84_E2/64.0) * phi
        - (3.0*WGS84_E2/8.0 + 3.0*WGS84_E2*WGS84_E2/32.0) * (2.0*phi).sin()
        + (15.0*WGS84_E2*WGS84_E2/256.0) * (4.0*phi).sin());
    let k0 = 0.9996;
    let easting = k0 * n * (a_coeff + (1.0-t*t+c)*a_coeff.powi(3)/6.0) + 500000.0;
    let northing = k0 * (m + n * t * (a_coeff*a_coeff/2.0 + (5.0-t*t+9.0*c+4.0*c*c)*a_coeff.powi(4)/24.0));
    let northing = if lat < 0.0 { northing + 10000000.0 } else { northing };
    (easting, northing, zone)
}

// ═══════════════════════════════════════════════════════════════════
// GEODESIC AREA
// ═══════════════════════════════════════════════════════════════════

/// Approximate area of a geodesic polygon (vertices in lat/lon degrees).
/// Uses the shoelace formula on a sphere. Returns area in m².
pub fn geodesic_polygon_area(lats: &[f64], lons: &[f64]) -> f64 {
    let n = lats.len();
    if n < 3 { return 0.0; }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += (lons[j] - lons[i]).to_radians() * (2.0 + lats[i].to_radians().sin() + lats[j].to_radians().sin());
    }
    (area * WGS84_A * WGS84_A / 2.0).abs()
}

/// Great circle arc length between two points (degrees). Returns meters.
pub fn great_circle_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    crate::coordinates::haversine(lat1, lon1, lat2, lon2)
}

/// Midpoint on great circle between two points (degrees).
pub fn great_circle_midpoint(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64) {
    crate::coordinates::geographic_midpoint(lat1, lon1, lat2, lon2)
}

/// Initial bearing from point 1 to point 2 (degrees).
pub fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    crate::coordinates::bearing(lat1, lon1, lat2, lon2)
}

// ═══════════════════════════════════════════════════════════════════
// BOUNDING BOX
// ═══════════════════════════════════════════════════════════════════

/// Geographic bounding box: (min_lat, min_lon, max_lat, max_lon).
pub fn geo_bbox(lats: &[f64], lons: &[f64]) -> (f64, f64, f64, f64) {
    let min_lat = lats.iter().copied().fold(f64::INFINITY, f64::min);
    let max_lat = lats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_lon = lons.iter().copied().fold(f64::INFINITY, f64::min);
    let max_lon = lons.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min_lat, min_lon, max_lat, max_lon)
}

/// Check if point is inside bounding box.
pub fn point_in_bbox(lat: f64, lon: f64, bbox: (f64, f64, f64, f64)) -> bool {
    lat >= bbox.0 && lat <= bbox.2 && lon >= bbox.1 && lon <= bbox.3
}

/// Bounding box area in m² (approximate).
pub fn bbox_area(bbox: (f64, f64, f64, f64)) -> f64 {
    let width = great_circle_distance(bbox.0, bbox.1, bbox.0, bbox.3);
    let height = great_circle_distance(bbox.0, bbox.1, bbox.2, bbox.1);
    width * height
}

/// Expand bounding box by distance (meters) in all directions.
pub fn bbox_expand(bbox: (f64, f64, f64, f64), distance_m: f64) -> (f64, f64, f64, f64) {
    let dlat = distance_m / 111320.0; // rough degrees per meter
    let dlon = distance_m / (111320.0 * ((bbox.0 + bbox.2) / 2.0).to_radians().cos());
    (bbox.0 - dlat, bbox.1 - dlon, bbox.2 + dlat, bbox.3 + dlon)
}

// ═══════════════════════════════════════════════════════════════════
// GEOHASH
// ═══════════════════════════════════════════════════════════════════

/// Encode (lat, lon) to geohash string of given precision (1-12).
pub fn geohash_encode(lat: f64, lon: f64, precision: usize) -> String {
    const BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";
    let mut lat_range = (-90.0, 90.0);
    let mut lon_range = (-180.0, 180.0);
    let mut hash = String::with_capacity(precision);
    let mut bits = 0u8;
    let mut bit_count = 0;
    let mut is_lon = true;

    while hash.len() < precision {
        if is_lon {
            let mid = (lon_range.0 + lon_range.1) / 2.0;
            if lon >= mid { bits = (bits << 1) | 1; lon_range.0 = mid; }
            else { bits <<= 1; lon_range.1 = mid; }
        } else {
            let mid = (lat_range.0 + lat_range.1) / 2.0;
            if lat >= mid { bits = (bits << 1) | 1; lat_range.0 = mid; }
            else { bits <<= 1; lat_range.1 = mid; }
        }
        is_lon = !is_lon;
        bit_count += 1;
        if bit_count == 5 {
            hash.push(BASE32[bits as usize] as char);
            bits = 0;
            bit_count = 0;
        }
    }
    hash
}

/// Decode geohash to (lat, lon, lat_err, lon_err).
pub fn geohash_decode(hash: &str) -> (f64, f64, f64, f64) {
    const BASE32: &str = "0123456789bcdefghjkmnpqrstuvwxyz";
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;

    for c in hash.chars() {
        let idx = BASE32.find(c).unwrap_or(0) as u8;
        for bit in (0..5).rev() {
            let val = (idx >> bit) & 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) / 2.0;
                if val == 1 { lon_range.0 = mid; } else { lon_range.1 = mid; }
            } else {
                let mid = (lat_range.0 + lat_range.1) / 2.0;
                if val == 1 { lat_range.0 = mid; } else { lat_range.1 = mid; }
            }
            is_lon = !is_lon;
        }
    }
    let lat = (lat_range.0 + lat_range.1) / 2.0;
    let lon = (lon_range.0 + lon_range.1) / 2.0;
    ((lat_range.1 - lat_range.0) / 2.0, (lon_range.1 - lon_range.0) / 2.0, lat, lon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercator_roundtrip() {
        let (x, y) = mercator_forward(40.0, -74.0);
        let (lat, lon) = mercator_inverse(x, y);
        assert!((lat - 40.0).abs() < 0.01);
        assert!((lon - (-74.0)).abs() < 0.01);
    }

    #[test]
    fn test_utm_zone() {
        assert_eq!(utm_zone(-73.9857), 18); // NYC
    }

    #[test]
    fn test_lat_lon_to_utm() {
        let (e, n, z) = lat_lon_to_utm(40.7128, -74.0060);
        assert_eq!(z, 18);
        assert!(e > 500000.0 && e < 700000.0);
    }

    #[test]
    fn test_vincenty() {
        // NYC to London ≈ 5570 km
        let d = vincenty_distance(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((d - 5_570_000.0).abs() < 100_000.0);
    }

    #[test]
    fn test_geohash_roundtrip() {
        let hash = geohash_encode(40.7128, -74.0060, 8);
        assert_eq!(hash.len(), 8);
        let (_, _, lat, lon) = geohash_decode(&hash);
        assert!((lat - 40.7128).abs() < 0.01);
        assert!((lon - (-74.0060)).abs() < 0.01);
    }

    #[test]
    fn test_geo_bbox() {
        let lats = vec![40.0, 41.0, 42.0];
        let lons = vec![-74.0, -73.0, -72.0];
        let bbox = geo_bbox(&lats, &lons);
        assert_eq!(bbox.0, 40.0);
        assert_eq!(bbox.2, 42.0);
    }

    #[test]
    fn test_equirectangular_roundtrip() {
        let (x, y) = equirectangular_forward(30.0, 60.0);
        let (lat, lon) = equirectangular_inverse(x, y);
        assert!((lat - 30.0).abs() < 0.01);
        assert!((lon - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_geodesic_polygon() {
        // Rough area of a 1° × 1° box near equator ≈ ~12,300 km²
        let lats = vec![0.0, 0.0, 1.0, 1.0];
        let lons = vec![0.0, 1.0, 1.0, 0.0];
        let area = geodesic_polygon_area(&lats, &lons);
        assert!(area > 1e10 && area < 2e10);
    }
}
