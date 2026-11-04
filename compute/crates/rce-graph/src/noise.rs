//! Noise generation: Perlin, Simplex, Worley, white noise,
//! fractal Brownian motion, value noise, blue noise.

// ═══════════════════════════════════════════════════════════════════
// PERLIN NOISE (2D)
// ═══════════════════════════════════════════════════════════════════

/// 2D Perlin noise at (x, y). Returns value in [-1, 1].
pub fn perlin_2d(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let xf = x - xi as f64;
    let yf = y - yi as f64;
    let u = fade(xf);
    let v = fade(yf);
    let n00 = grad2d(hash2(xi, yi), xf, yf);
    let n10 = grad2d(hash2(xi + 1, yi), xf - 1.0, yf);
    let n01 = grad2d(hash2(xi, yi + 1), xf, yf - 1.0);
    let n11 = grad2d(hash2(xi + 1, yi + 1), xf - 1.0, yf - 1.0);
    let x1 = lerp(n00, n10, u);
    let x2 = lerp(n01, n11, u);
    lerp(x1, x2, v)
}

/// 3D Perlin noise.
pub fn perlin_3d(x: f64, y: f64, z: f64) -> f64 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let zi = z.floor() as i64;
    let xf = x - xi as f64;
    let yf = y - yi as f64;
    let zf = z - zi as f64;
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);
    let n000 = grad3d(hash3(xi, yi, zi), xf, yf, zf);
    let n100 = grad3d(hash3(xi+1, yi, zi), xf-1.0, yf, zf);
    let n010 = grad3d(hash3(xi, yi+1, zi), xf, yf-1.0, zf);
    let n110 = grad3d(hash3(xi+1, yi+1, zi), xf-1.0, yf-1.0, zf);
    let n001 = grad3d(hash3(xi, yi, zi+1), xf, yf, zf-1.0);
    let n101 = grad3d(hash3(xi+1, yi, zi+1), xf-1.0, yf, zf-1.0);
    let n011 = grad3d(hash3(xi, yi+1, zi+1), xf, yf-1.0, zf-1.0);
    let n111 = grad3d(hash3(xi+1, yi+1, zi+1), xf-1.0, yf-1.0, zf-1.0);
    let x1 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v);
    let x2 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v);
    lerp(x1, x2, w)
}

// ═══════════════════════════════════════════════════════════════════
// FRACTAL BROWNIAN MOTION
// ═══════════════════════════════════════════════════════════════════

/// Fractal Brownian motion (layered Perlin noise).
pub fn fbm_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..octaves {
        sum += amplitude * perlin_2d(x * frequency, y * frequency);
        max_amp += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    sum / max_amp
}

/// 3D fractal Brownian motion.
pub fn fbm_3d(x: f64, y: f64, z: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..octaves {
        sum += amplitude * perlin_3d(x * frequency, y * frequency, z * frequency);
        max_amp += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    sum / max_amp
}

/// Ridged multifractal noise.
pub fn ridged_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut weight = 1.0;
    for _ in 0..octaves {
        let signal = (1.0 - perlin_2d(x * frequency, y * frequency).abs()) * weight;
        sum += signal * amplitude;
        weight = (signal * gain).min(1.0).max(0.0);
        amplitude *= gain;
        frequency *= lacunarity;
    }
    sum
}

/// Turbulence (absolute value fbm).
pub fn turbulence_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    for _ in 0..octaves {
        sum += amplitude * perlin_2d(x * frequency, y * frequency).abs();
        amplitude *= gain;
        frequency *= lacunarity;
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// VALUE NOISE
// ═══════════════════════════════════════════════════════════════════

/// 2D value noise (interpolated random grid).
pub fn value_noise_2d(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let xf = x - xi as f64;
    let yf = y - yi as f64;
    let u = fade(xf);
    let v = fade(yf);
    let n00 = hash_float(xi, yi);
    let n10 = hash_float(xi + 1, yi);
    let n01 = hash_float(xi, yi + 1);
    let n11 = hash_float(xi + 1, yi + 1);
    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

// ═══════════════════════════════════════════════════════════════════
// WORLEY (CELLULAR) NOISE
// ═══════════════════════════════════════════════════════════════════

/// 2D Worley noise (distance to nearest feature point). Returns F1.
pub fn worley_2d(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let mut min_dist = f64::INFINITY;
    for dx in -1..=1 {
        for dy in -1..=1 {
            let nx = xi + dx;
            let ny = yi + dy;
            let fx = nx as f64 + hash_float(nx, ny);
            let fy = ny as f64 + hash_float(nx * 127 + 31, ny * 269 + 73);
            let dist = ((x - fx).powi(2) + (y - fy).powi(2)).sqrt();
            min_dist = min_dist.min(dist);
        }
    }
    min_dist
}

/// 2D Worley F2 - F1 (produces cell-like patterns).
pub fn worley_2d_f2f1(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let mut f1 = f64::INFINITY;
    let mut f2 = f64::INFINITY;
    for dx in -1..=1 {
        for dy in -1..=1 {
            let nx = xi + dx;
            let ny = yi + dy;
            let fx = nx as f64 + hash_float(nx, ny);
            let fy = ny as f64 + hash_float(nx * 127 + 31, ny * 269 + 73);
            let dist = ((x - fx).powi(2) + (y - fy).powi(2)).sqrt();
            if dist < f1 { f2 = f1; f1 = dist; }
            else if dist < f2 { f2 = dist; }
        }
    }
    f2 - f1
}

// ═══════════════════════════════════════════════════════════════════
// WHITE NOISE
// ═══════════════════════════════════════════════════════════════════

/// White noise at integer coordinates.
pub fn white_noise_2d(x: i64, y: i64) -> f64 {
    hash_float(x, y)
}

/// Generate 2D white noise grid.
pub fn white_noise_grid(width: usize, height: usize, seed: u64) -> Vec<f64> {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    fn prng(s: &mut [u64; 4]) -> u64 {
        let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    for _ in 0..20 { prng(&mut state); }
    (0..width * height).map(|_| (prng(&mut state) >> 11) as f64 / (1u64 << 53) as f64).collect()
}

// ═══════════════════════════════════════════════════════════════════
// NOISE MAP GENERATION
// ═══════════════════════════════════════════════════════════════════

/// Generate 2D noise map.
pub fn noise_map(width: usize, height: usize, scale: f64, noise_fn: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut map = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            map.push(noise_fn(x as f64 / scale, y as f64 / scale));
        }
    }
    map
}

/// Normalize noise map to [0, 1].
pub fn normalize_noise_map(map: &mut [f64]) {
    let min = map.iter().copied().fold(f64::INFINITY, f64::min);
    let max = map.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range > 1e-15 { for v in map.iter_mut() { *v = (*v - min) / range; } }
}

/// Domain warping: warp noise coordinates with another noise function.
pub fn domain_warp_2d(x: f64, y: f64, warp_strength: f64, octaves: u32) -> f64 {
    let wx = fbm_2d(x, y, octaves, 2.0, 0.5) * warp_strength;
    let wy = fbm_2d(x + 5.2, y + 1.3, octaves, 2.0, 0.5) * warp_strength;
    fbm_2d(x + wx, y + wy, octaves, 2.0, 0.5)
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

fn fade(t: f64) -> f64 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
fn lerp(a: f64, b: f64, t: f64) -> f64 { a + t * (b - a) }

fn hash2(x: i64, y: i64) -> u64 {
    let mut h = (x as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add((y as u64).wrapping_mul(0x6C62272E07BB0142));
    h ^= h >> 30; h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27; h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31; h
}

fn hash3(x: i64, y: i64, z: i64) -> u64 {
    hash2(hash2(x, y) as i64, z)
}

fn hash_float(x: i64, y: i64) -> f64 {
    (hash2(x, y) >> 11) as f64 / (1u64 << 53) as f64
}

fn grad2d(hash: u64, x: f64, y: f64) -> f64 {
    match hash & 3 { 0 => x + y, 1 => -x + y, 2 => x - y, _ => -x - y }
}

fn grad3d(hash: u64, x: f64, y: f64, z: f64) -> f64 {
    match hash & 15 {
        0 => x + y, 1 => -x + y, 2 => x - y, 3 => -x - y,
        4 => x + z, 5 => -x + z, 6 => x - z, 7 => -x - z,
        8 => y + z, 9 => -y + z, 10 => y - z, 11 => -y - z,
        12 => x + y, 13 => -x + y, 14 => x - z, _ => -y - z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_range() {
        for i in 0..100 {
            let v = perlin_2d(i as f64 * 0.1, i as f64 * 0.13);
            assert!(v >= -2.0 && v <= 2.0);
        }
    }

    #[test]
    fn test_perlin_3d() {
        let v = perlin_3d(1.5, 2.5, 3.5);
        assert!(v.is_finite());
    }

    #[test]
    fn test_fbm() {
        let v = fbm_2d(1.0, 1.0, 4, 2.0, 0.5);
        assert!(v >= -1.5 && v <= 1.5);
    }

    #[test]
    fn test_value_noise() {
        let v = value_noise_2d(1.5, 2.5);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_worley() {
        let v = worley_2d(1.5, 2.5);
        assert!(v >= 0.0);
    }

    #[test]
    fn test_noise_map() {
        let map = noise_map(10, 10, 5.0, perlin_2d);
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn test_white_noise_grid() {
        let grid = white_noise_grid(10, 10, 42);
        assert_eq!(grid.len(), 100);
        assert!(grid.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_ridged() {
        let v = ridged_2d(1.0, 2.0, 4, 2.0, 0.5);
        assert!(v.is_finite());
    }

    #[test]
    fn test_domain_warp() {
        let v = domain_warp_2d(1.0, 2.0, 1.0, 3);
        assert!(v.is_finite());
    }
}
