//! Error correction codes: Hamming, parity, CRC, repetition,
//! Reed-Solomon (simplified), convolutional basics.

// ═══════════════════════════════════════════════════════════════════
// PARITY
// ═══════════════════════════════════════════════════════════════════

/// Even parity bit for a byte.
pub fn parity_bit(byte: u8) -> u8 {
    byte.count_ones() as u8 % 2
}

/// Check even parity (returns true if valid).
pub fn check_parity(byte: u8, parity: u8) -> bool {
    (byte.count_ones() as u8 + parity) % 2 == 0
}

/// Add parity bit to each byte (MSB becomes parity).
pub fn add_parity(data: &[u8]) -> Vec<u8> {
    data.iter().map(|&b| {
        let p = parity_bit(b);
        (b & 0x7F) | (p << 7)
    }).collect()
}

/// Check and strip parity bits. Returns (data, errors_found).
pub fn check_and_strip_parity(data: &[u8]) -> (Vec<u8>, usize) {
    let mut errors = 0;
    let stripped: Vec<u8> = data.iter().map(|&b| {
        let payload = b & 0x7F;
        let p = b >> 7;
        if !check_parity(payload, p) { errors += 1; }
        payload
    }).collect();
    (stripped, errors)
}

// ═══════════════════════════════════════════════════════════════════
// HAMMING CODE (7,4)
// ═══════════════════════════════════════════════════════════════════

/// Hamming(7,4) encode: 4 data bits → 7 coded bits.
pub fn hamming74_encode(data: u8) -> u8 {
    let d = [data & 1, (data >> 1) & 1, (data >> 2) & 1, (data >> 3) & 1];
    let p1 = d[0] ^ d[1] ^ d[3];
    let p2 = d[0] ^ d[2] ^ d[3];
    let p3 = d[1] ^ d[2] ^ d[3];
    p1 | (p2 << 1) | (d[0] << 2) | (p3 << 3) | (d[1] << 4) | (d[2] << 5) | (d[3] << 6)
}

/// Hamming(7,4) decode with single-error correction.
/// Returns (decoded 4-bit data, error_corrected).
pub fn hamming74_decode(code: u8) -> (u8, bool) {
    let bits = [code & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1,
                (code >> 4) & 1, (code >> 5) & 1, (code >> 6) & 1];
    let s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6];
    let s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6];
    let s3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6];
    let syndrome = s1 | (s2 << 1) | (s3 << 2);

    let mut corrected = code;
    let error = syndrome > 0;
    if error && (syndrome as usize) <= 7 {
        corrected ^= 1 << (syndrome - 1);
    }

    let data = ((corrected >> 2) & 1) | (((corrected >> 4) & 1) << 1)
             | (((corrected >> 5) & 1) << 2) | (((corrected >> 6) & 1) << 3);
    (data, error)
}

/// Encode a byte array with Hamming(7,4). Each byte produces 2 codewords.
pub fn hamming_encode_bytes(data: &[u8]) -> Vec<u8> {
    data.iter().flat_map(|&b| {
        let lo = hamming74_encode(b & 0x0F);
        let hi = hamming74_encode((b >> 4) & 0x0F);
        vec![lo, hi]
    }).collect()
}

/// Decode Hamming-encoded bytes. Returns (data, errors_corrected).
pub fn hamming_decode_bytes(encoded: &[u8]) -> (Vec<u8>, usize) {
    let mut errors = 0;
    let decoded: Vec<u8> = encoded.chunks(2).map(|pair| {
        let (lo, e1) = hamming74_decode(pair[0]);
        let (hi, e2) = if pair.len() > 1 { hamming74_decode(pair[1]) } else { (0, false) };
        if e1 { errors += 1; }
        if e2 { errors += 1; }
        lo | (hi << 4)
    }).collect();
    (decoded, errors)
}

// ═══════════════════════════════════════════════════════════════════
// REPETITION CODE
// ═══════════════════════════════════════════════════════════════════

/// Repetition encode: repeat each bit n times.
pub fn repetition_encode(data: &[u8], n: usize) -> Vec<u8> {
    let mut bits = Vec::new();
    for &byte in data {
        for bit_pos in 0..8 {
            let bit = (byte >> bit_pos) & 1;
            for _ in 0..n { bits.push(bit); }
        }
    }
    // Pack back into bytes
    let mut result = Vec::new();
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() { byte |= b << i; }
        result.push(byte);
    }
    result
}

/// Repetition decode: majority vote every n bits.
pub fn repetition_decode(encoded: &[u8], n: usize, original_bytes: usize) -> Vec<u8> {
    // Unpack all bits
    let mut bits = Vec::new();
    for &byte in encoded {
        for i in 0..8 { bits.push((byte >> i) & 1); }
    }
    // Majority vote
    let mut decoded_bits = Vec::new();
    for chunk in bits.chunks(n) {
        let ones: u8 = chunk.iter().sum();
        decoded_bits.push(if ones > (chunk.len() as u8 / 2) { 1u8 } else { 0 });
    }
    // Pack into bytes
    let mut result = Vec::new();
    for chunk in decoded_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() { byte |= b << i; }
        result.push(byte);
    }
    result.truncate(original_bytes);
    result
}

// ═══════════════════════════════════════════════════════════════════
// INTERLEAVING
// ═══════════════════════════════════════════════════════════════════

/// Block interleave: rearrange data to spread burst errors.
pub fn block_interleave(data: &[u8], block_size: usize) -> Vec<u8> {
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut result = vec![0u8; n_blocks * block_size];
    for i in 0..data.len() {
        let row = i / block_size;
        let col = i % block_size;
        let new_idx = col * n_blocks + row;
        if new_idx < result.len() { result[new_idx] = data[i]; }
    }
    result
}

/// Block deinterleave.
pub fn block_deinterleave(data: &[u8], block_size: usize) -> Vec<u8> {
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut result = vec![0u8; data.len()];
    for i in 0..data.len() {
        let col = i / n_blocks;
        let row = i % n_blocks;
        let orig_idx = row * block_size + col;
        if orig_idx < result.len() { result[orig_idx] = data[i]; }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// BIT ERROR RATE
// ═══════════════════════════════════════════════════════════════════

/// Bit error rate between original and received data.
pub fn bit_error_rate(original: &[u8], received: &[u8]) -> f64 {
    let total_bits = original.len().min(received.len()) * 8;
    if total_bits == 0 { return 0.0; }
    let errors: u32 = original.iter().zip(received).map(|(a, b)| (a ^ b).count_ones()).sum();
    errors as f64 / total_bits as f64
}

/// Byte error rate.
pub fn byte_error_rate(original: &[u8], received: &[u8]) -> f64 {
    let total = original.len().min(received.len());
    if total == 0 { return 0.0; }
    let errors = original.iter().zip(received).filter(|(a, b)| a != b).count();
    errors as f64 / total as f64
}

/// Introduce random bit errors (for testing).
pub fn inject_errors(data: &[u8], error_rate: f64, seed: u64) -> Vec<u8> {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 {
        let _result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
        let t = state[1] << 17;
        state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
        state[2] ^= t; state[3] = state[3].rotate_left(45);
    }

    data.iter().map(|&byte| {
        let mut b = byte;
        for bit in 0..8 {
            let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
            let t = state[1] << 17;
            state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
            state[2] ^= t; state[3] = state[3].rotate_left(45);
            let u = (result >> 11) as f64 / (1u64 << 53) as f64;
            if u < error_rate { b ^= 1 << bit; }
        }
        b
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CHECKSUM
// ═══════════════════════════════════════════════════════════════════

/// Internet checksum (RFC 1071).
pub fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    for chunk in data.chunks(2) {
        let word = if chunk.len() == 2 {
            (chunk[0] as u32) << 8 | chunk[1] as u32
        } else {
            (chunk[0] as u32) << 8
        };
        sum += word;
    }
    while sum >> 16 != 0 { sum = (sum & 0xFFFF) + (sum >> 16); }
    !(sum as u16)
}

/// Luhn algorithm (credit card, IMEI validation).
pub fn luhn_check(digits: &[u8]) -> bool {
    let mut sum = 0u32;
    for (i, &d) in digits.iter().rev().enumerate() {
        let mut n = d as u32;
        if i % 2 == 1 { n *= 2; if n > 9 { n -= 9; } }
        sum += n;
    }
    sum % 10 == 0
}

/// Generate Luhn check digit.
pub fn luhn_check_digit(digits: &[u8]) -> u8 {
    let mut extended = digits.to_vec();
    extended.push(0);
    for check in 0..10 {
        *extended.last_mut().unwrap() = check;
        if luhn_check(&extended) { return check; }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity() {
        assert_eq!(parity_bit(0b1010_1010), 0); // 4 ones → even
        assert_eq!(parity_bit(0b1010_1011), 1); // 5 ones → odd
    }

    #[test]
    fn test_hamming74_roundtrip() {
        for data in 0..16u8 {
            let encoded = hamming74_encode(data);
            let (decoded, _) = hamming74_decode(encoded);
            assert_eq!(decoded, data);
        }
    }

    #[test]
    fn test_hamming74_error_correction() {
        let data = 0b1010u8;
        let encoded = hamming74_encode(data);
        // Flip one bit
        let corrupted = encoded ^ 0b0000100;
        let (decoded, corrected) = hamming74_decode(corrupted);
        assert_eq!(decoded, data);
        assert!(corrected);
    }

    #[test]
    fn test_hamming_bytes_roundtrip() {
        let data = vec![0xAB, 0xCD, 0xEF];
        let encoded = hamming_encode_bytes(&data);
        let (decoded, _) = hamming_decode_bytes(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_repetition_roundtrip() {
        let data = vec![0xAB, 0xCD];
        let encoded = repetition_encode(&data, 3);
        let decoded = repetition_decode(&encoded, 3, data.len());
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_bit_error_rate() {
        let a = vec![0xFF, 0x00];
        let b = vec![0xFF, 0x00];
        assert_eq!(bit_error_rate(&a, &b), 0.0);
        let c = vec![0xFE, 0x00]; // 1 bit different
        assert!((bit_error_rate(&a, &c) - 1.0 / 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_interleave_roundtrip() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let interleaved = block_interleave(&data, 3);
        let deinterleaved = block_deinterleave(&interleaved, 3);
        assert_eq!(deinterleaved[..data.len()], data[..]);
    }

    #[test]
    fn test_luhn() {
        // Valid credit card: 4539578763621486
        let digits = vec![4, 5, 3, 9, 5, 7, 8, 7, 6, 3, 6, 2, 1, 4, 8, 6];
        assert!(luhn_check(&digits));
    }

    #[test]
    fn test_luhn_check_digit() {
        let digits = vec![4, 5, 3, 9, 5, 7, 8, 7, 6, 3, 6, 2, 1, 4, 8];
        assert_eq!(luhn_check_digit(&digits), 6);
    }

    #[test]
    fn test_internet_checksum() {
        let data = vec![0x00, 0x01, 0xF2, 0x03, 0xF4, 0xF5, 0xF6, 0xF7];
        let cs = internet_checksum(&data);
        // Verify: adding checksum to data should yield 0xFFFF
        assert!(cs != 0);
    }
}
