//! Type conversion utilities: numeric parsing, formatting, base conversion,
//! number system representations, bit manipulation, endianness.

// ═══════════════════════════════════════════════════════════════════
// BASE CONVERSION
// ═══════════════════════════════════════════════════════════════════

/// Convert integer to binary string.
pub fn to_binary(n: u64) -> String { format!("{:b}", n) }

/// Convert integer to octal string.
pub fn to_octal(n: u64) -> String { format!("{:o}", n) }

/// Convert integer to hexadecimal string.
pub fn to_hex(n: u64) -> String { format!("{:x}", n) }

/// Convert integer to arbitrary base (2-36).
pub fn to_base(mut n: u64, base: u32) -> String {
    if n == 0 { return "0".to_string(); }
    let digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    let mut result = Vec::new();
    while n > 0 {
        result.push(digits.as_bytes()[(n % base as u64) as usize] as char);
        n /= base as u64;
    }
    result.reverse();
    result.into_iter().collect()
}

/// Parse integer from arbitrary base string.
pub fn from_base(s: &str, base: u32) -> Option<u64> {
    let mut result = 0u64;
    for c in s.chars() {
        let digit = if c.is_ascii_digit() { c as u64 - '0' as u64 }
                    else if c.is_ascii_lowercase() { c as u64 - 'a' as u64 + 10 }
                    else if c.is_ascii_uppercase() { c as u64 - 'A' as u64 + 10 }
                    else { return None; };
        if digit >= base as u64 { return None; }
        result = result * base as u64 + digit;
    }
    Some(result)
}

/// Parse binary string to integer.
pub fn from_binary(s: &str) -> Option<u64> { from_base(s, 2) }

/// Parse hex string to integer.
pub fn from_hex(s: &str) -> Option<u64> { from_base(s, 16) }

/// Parse octal string to integer.
pub fn from_octal(s: &str) -> Option<u64> { from_base(s, 8) }

// ═══════════════════════════════════════════════════════════════════
// NUMERIC FORMATTING
// ═══════════════════════════════════════════════════════════════════

/// Format number with thousands separator.
pub fn format_thousands(n: i64) -> String {
    let neg = n < 0;
    let s = n.unsigned_abs().to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    if neg { result.push('-'); }
    result.chars().rev().collect()
}

/// Format bytes as human-readable (KB, MB, GB, TB).
pub fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut val = bytes as f64;
    let mut unit_idx = 0;
    while val >= 1024.0 && unit_idx < units.len() - 1 { val /= 1024.0; unit_idx += 1; }
    if unit_idx == 0 { format!("{} B", bytes) }
    else { format!("{:.2} {}", val, units[unit_idx]) }
}

/// Format duration in seconds as human-readable.
pub fn format_duration(seconds: f64) -> String {
    if seconds < 0.001 { return format!("{:.0} μs", seconds * 1e6); }
    if seconds < 1.0 { return format!("{:.1} ms", seconds * 1000.0); }
    if seconds < 60.0 { return format!("{:.1} s", seconds); }
    if seconds < 3600.0 { return format!("{:.0}m {:.0}s", (seconds / 60.0).floor(), seconds % 60.0); }
    let hours = (seconds / 3600.0).floor();
    let mins = ((seconds % 3600.0) / 60.0).floor();
    format!("{:.0}h {:.0}m", hours, mins)
}

/// Format number in scientific notation.
pub fn format_scientific(x: f64, precision: usize) -> String {
    format!("{:.*e}", precision, x)
}

/// Format number in engineering notation (exponent multiple of 3).
pub fn format_engineering(x: f64, precision: usize) -> String {
    if x == 0.0 { return format!("{:.*}", precision, 0.0); }
    let exp = (x.abs().log10().floor() / 3.0).floor() * 3.0;
    let mantissa = x / 10.0f64.powf(exp);
    if exp == 0.0 { format!("{:.*}", precision, mantissa) }
    else { format!("{:.*}e{}", precision, mantissa, exp as i64) }
}

/// Format percentage.
pub fn format_percent(x: f64, precision: usize) -> String {
    format!("{:.*}%", precision, x * 100.0)
}

/// Format number with SI prefix (k, M, G, T, etc.).
pub fn format_si(x: f64) -> String {
    let prefixes = [("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0),
                    ("m", 1e-3), ("μ", 1e-6), ("n", 1e-9), ("p", 1e-12)];
    let abs_x = x.abs();
    for &(prefix, scale) in &prefixes {
        if abs_x >= scale * 0.999 || prefix == "p" {
            return format!("{:.3}{}", x / scale, prefix);
        }
    }
    format!("{:.3}", x)
}

// ═══════════════════════════════════════════════════════════════════
// BIT MANIPULATION
// ═══════════════════════════════════════════════════════════════════

/// Count set bits (population count / Hamming weight).
pub fn popcount(n: u64) -> u32 { n.count_ones() }

/// Count leading zeros.
pub fn leading_zeros(n: u64) -> u32 { n.leading_zeros() }

/// Count trailing zeros.
pub fn trailing_zeros(n: u64) -> u32 { n.trailing_zeros() }

/// Highest set bit position (0-indexed from LSB).
pub fn highest_bit(n: u64) -> u32 { if n == 0 { 0 } else { 63 - n.leading_zeros() } }

/// Lowest set bit position.
pub fn lowest_bit(n: u64) -> u32 { if n == 0 { 64 } else { n.trailing_zeros() } }

/// Is power of 2?
pub fn is_power_of_2(n: u64) -> bool { n > 0 && (n & (n - 1)) == 0 }

/// Next power of 2 >= n.
pub fn next_power_of_2(n: u64) -> u64 { n.next_power_of_two() }

/// Reverse bits of a u64.
pub fn reverse_bits(n: u64) -> u64 { n.reverse_bits() }

/// Rotate left by k bits.
pub fn rotate_left(n: u64, k: u32) -> u64 { n.rotate_left(k) }

/// Rotate right by k bits.
pub fn rotate_right(n: u64, k: u32) -> u64 { n.rotate_right(k) }

/// Bit extraction: get bits [lo..hi] inclusive.
pub fn extract_bits(n: u64, lo: u32, hi: u32) -> u64 {
    let width = hi - lo + 1;
    (n >> lo) & ((1u64 << width) - 1)
}

/// Set bit at position.
pub fn set_bit(n: u64, pos: u32) -> u64 { n | (1u64 << pos) }

/// Clear bit at position.
pub fn clear_bit(n: u64, pos: u32) -> u64 { n & !(1u64 << pos) }

/// Toggle bit at position.
pub fn toggle_bit(n: u64, pos: u32) -> u64 { n ^ (1u64 << pos) }

/// Test bit at position.
pub fn test_bit(n: u64, pos: u32) -> bool { (n >> pos) & 1 == 1 }

/// Integer log2 (floor).
pub fn ilog2(n: u64) -> u32 { if n == 0 { 0 } else { 63 - n.leading_zeros() } }

/// Gray code encode.
pub fn gray_encode(n: u64) -> u64 { n ^ (n >> 1) }

/// Gray code decode.
pub fn gray_decode(mut n: u64) -> u64 {
    let mut mask = n >> 1;
    while mask != 0 { n ^= mask; mask >>= 1; }
    n
}

// ═══════════════════════════════════════════════════════════════════
// ENDIANNESS
// ═══════════════════════════════════════════════════════════════════

/// Swap bytes (big-endian ↔ little-endian).
pub fn byte_swap_16(n: u16) -> u16 { n.swap_bytes() }
pub fn byte_swap_32(n: u32) -> u32 { n.swap_bytes() }
pub fn byte_swap_64(n: u64) -> u64 { n.swap_bytes() }

/// f64 to bytes (little-endian).
pub fn f64_to_le_bytes(x: f64) -> [u8; 8] { x.to_le_bytes() }
/// f64 from bytes (little-endian).
pub fn f64_from_le_bytes(bytes: [u8; 8]) -> f64 { f64::from_le_bytes(bytes) }
/// f64 to bytes (big-endian).
pub fn f64_to_be_bytes(x: f64) -> [u8; 8] { x.to_be_bytes() }
/// f64 from bytes (big-endian).
pub fn f64_from_be_bytes(bytes: [u8; 8]) -> f64 { f64::from_be_bytes(bytes) }

// ═══════════════════════════════════════════════════════════════════
// TYPE ROUNDING / CLAMPING
// ═══════════════════════════════════════════════════════════════════

/// Round to n decimal places.
pub fn round_to(x: f64, decimals: u32) -> f64 {
    let factor = 10.0f64.powi(decimals as i32);
    (x * factor).round() / factor
}

/// Floor to n decimal places.
pub fn floor_to(x: f64, decimals: u32) -> f64 {
    let factor = 10.0f64.powi(decimals as i32);
    (x * factor).floor() / factor
}

/// Ceil to n decimal places.
pub fn ceil_to(x: f64, decimals: u32) -> f64 {
    let factor = 10.0f64.powi(decimals as i32);
    (x * factor).ceil() / factor
}

/// Clamp value to [min, max].
pub fn clamp(x: f64, min: f64, max: f64) -> f64 { x.max(min).min(max) }

/// Linear interpolation: lerp(a, b, t) = a + t*(b - a).
pub fn lerp(a: f64, b: f64, t: f64) -> f64 { a + t * (b - a) }

/// Inverse lerp: find t such that lerp(a, b, t) = x.
pub fn inv_lerp(a: f64, b: f64, x: f64) -> f64 {
    if (b - a).abs() < 1e-15 { 0.0 } else { (x - a) / (b - a) }
}

/// Remap value from [in_min, in_max] to [out_min, out_max].
pub fn remap(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    out_min + (x - in_min) / (in_max - in_min) * (out_max - out_min)
}

/// Smooth step (Hermite interpolation): 0 for x<=edge0, 1 for x>=edge1.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step (quintic interpolation).
pub fn smootherstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Sign function: -1, 0, or 1.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

/// Step function: 0 for x < edge, 1 for x >= edge.
pub fn step(edge: f64, x: f64) -> f64 { if x >= edge { 1.0 } else { 0.0 } }

/// Wrap value to [min, max) range.
pub fn wrap(x: f64, min: f64, max: f64) -> f64 {
    let range = max - min;
    min + ((x - min) % range + range) % range
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_conversion() {
        assert_eq!(to_binary(42), "101010");
        assert_eq!(to_hex(255), "ff");
        assert_eq!(to_base(255, 16), "ff");
        assert_eq!(from_binary("101010"), Some(42));
        assert_eq!(from_hex("ff"), Some(255));
    }

    #[test]
    fn test_format_thousands() {
        assert_eq!(format_thousands(1234567), "1,234,567");
        assert_eq!(format_thousands(-1234), "-1,234");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_si() {
        assert!(format_si(1500.0).contains("k"));
        assert!(format_si(0.001).contains("m"));
    }

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0b1010_1010), 4);
    }

    #[test]
    fn test_power_of_2() {
        assert!(is_power_of_2(64));
        assert!(!is_power_of_2(63));
        assert_eq!(next_power_of_2(5), 8);
    }

    #[test]
    fn test_gray_code() {
        for i in 0..16 {
            assert_eq!(gray_decode(gray_encode(i)), i);
        }
    }

    #[test]
    fn test_bit_ops() {
        assert!(test_bit(0b1010, 1));
        assert!(!test_bit(0b1010, 0));
        assert_eq!(set_bit(0, 3), 8);
        assert_eq!(clear_bit(0b1111, 2), 0b1011);
        assert_eq!(extract_bits(0b11001010, 2, 5), 0b0010);
    }

    #[test]
    fn test_round_to() {
        assert!((round_to(3.14159, 2) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((inv_lerp(0.0, 10.0, 5.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_remap() {
        assert!((remap(5.0, 0.0, 10.0, 0.0, 100.0) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 0.1);
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn test_wrap() {
        assert!((wrap(370.0, 0.0, 360.0) - 10.0).abs() < 1e-10);
    }
}
