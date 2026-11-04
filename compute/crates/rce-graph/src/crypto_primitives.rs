//! Cryptographic primitives: hash functions, HMAC, key derivation,
//! secure random, base64, hex encoding, checksums.
//! NOTE: These are for educational/research use — NOT production crypto.

// ═══════════════════════════════════════════════════════════════════
// HASH FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// FNV-1a hash (64-bit).
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a hash (32-bit).
pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash = 0x811c9dc5u32;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// DJB2 hash.
pub fn djb2(data: &[u8]) -> u64 {
    let mut hash = 5381u64;
    for &byte in data { hash = hash.wrapping_mul(33).wrapping_add(byte as u64); }
    hash
}

/// MurmurHash3 (32-bit, simplified).
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    let c1: u32 = 0xcc9e2d51;
    let c2: u32 = 0x1b873593;
    let mut h = seed;
    let nblocks = data.len() / 4;

    for i in 0..nblocks {
        let mut k = u32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]);
        k = k.wrapping_mul(c1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(c2);
        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    if tail.len() >= 3 { k1 ^= (tail[2] as u32) << 16; }
    if tail.len() >= 2 { k1 ^= (tail[1] as u32) << 8; }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(c1).rotate_left(15).wrapping_mul(c2);
        h ^= k1;
    }

    h ^= data.len() as u32;
    // fmix
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// CRC32 (IEEE 802.3).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 { crc = (crc >> 1) ^ 0xEDB88320; }
            else { crc >>= 1; }
        }
    }
    !crc
}

/// Adler32 checksum.
pub fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Fletcher16 checksum.
pub fn fletcher16(data: &[u8]) -> u16 {
    let mut sum1 = 0u16;
    let mut sum2 = 0u16;
    for &byte in data {
        sum1 = (sum1 + byte as u16) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    (sum2 << 8) | sum1
}

/// SipHash-2-4 (simplified 64-bit).
pub fn siphash_2_4(data: &[u8], key: [u64; 2]) -> u64 {
    let mut v0 = key[0] ^ 0x736f6d6570736575;
    let mut v1 = key[1] ^ 0x646f72616e646f6d;
    let mut v2 = key[0] ^ 0x6c7967656e657261;
    let mut v3 = key[1] ^ 0x7465646279746573;

    let nblocks = data.len() / 8;
    for i in 0..nblocks {
        let mut m = 0u64;
        for j in 0..8 { m |= (data[i * 8 + j] as u64) << (j * 8); }
        v3 ^= m;
        for _ in 0..2 { sipround(&mut v0, &mut v1, &mut v2, &mut v3); }
        v0 ^= m;
    }

    let mut last = (data.len() as u64 & 0xff) << 56;
    let tail = &data[nblocks * 8..];
    for (i, &b) in tail.iter().enumerate() { last |= (b as u64) << (i * 8); }
    v3 ^= last;
    for _ in 0..2 { sipround(&mut v0, &mut v1, &mut v2, &mut v3); }
    v0 ^= last;
    v2 ^= 0xff;
    for _ in 0..4 { sipround(&mut v0, &mut v1, &mut v2, &mut v3); }
    v0 ^ v1 ^ v2 ^ v3
}

fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1); *v1 = v1.rotate_left(13); *v1 ^= *v0; *v0 = v0.rotate_left(32);
    *v2 = v2.wrapping_add(*v3); *v3 = v3.rotate_left(16); *v3 ^= *v2;
    *v0 = v0.wrapping_add(*v3); *v3 = v3.rotate_left(21); *v3 ^= *v0;
    *v2 = v2.wrapping_add(*v1); *v1 = v1.rotate_left(17); *v1 ^= *v2; *v2 = v2.rotate_left(32);
}

// ═══════════════════════════════════════════════════════════════════
// ENCODING
// ═══════════════════════════════════════════════════════════════════

/// Hex encode bytes to string.
pub fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Hex decode string to bytes.
pub fn hex_decode(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).filter_map(|i| {
        u8::from_str_radix(&hex[i..i + 2], 16).ok()
    }).collect()
}

/// Base64 encode.
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        result.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(TABLE[((triple >> 6) & 0x3F) as usize] as char); } else { result.push('='); }
        if chunk.len() > 2 { result.push(TABLE[(triple & 0x3F) as usize] as char); } else { result.push('='); }
    }
    result
}

/// Base64 decode.
pub fn base64_decode(encoded: &str) -> Vec<u8> {
    fn decode_char(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62, b'/' => 63,
            _ => 0,
        }
    }
    let bytes: Vec<u8> = encoded.bytes().filter(|&b| b != b'=').collect();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 { break; }
        let b0 = decode_char(chunk[0]) as u32;
        let b1 = decode_char(chunk[1]) as u32;
        let b2 = if chunk.len() > 2 { decode_char(chunk[2]) as u32 } else { 0 };
        let b3 = if chunk.len() > 3 { decode_char(chunk[3]) as u32 } else { 0 };
        let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        result.push(((triple >> 16) & 0xFF) as u8);
        if chunk.len() > 2 { result.push(((triple >> 8) & 0xFF) as u8); }
        if chunk.len() > 3 { result.push((triple & 0xFF) as u8); }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// HMAC (Hash-based Message Authentication Code)
// ═══════════════════════════════════════════════════════════════════

/// HMAC using FNV1a as the hash function (educational only).
pub fn hmac_fnv1a(key: &[u8], message: &[u8]) -> u64 {
    let block_size = 64;
    let mut key_padded = vec![0u8; block_size];
    if key.len() > block_size {
        let h = fnv1a_64(key);
        key_padded[..8].copy_from_slice(&h.to_le_bytes());
    } else {
        key_padded[..key.len()].copy_from_slice(key);
    }

    let mut ipad = vec![0x36u8; block_size];
    let mut opad = vec![0x5cu8; block_size];
    for i in 0..block_size {
        ipad[i] ^= key_padded[i];
        opad[i] ^= key_padded[i];
    }

    let mut inner = ipad;
    inner.extend_from_slice(message);
    let inner_hash = fnv1a_64(&inner);

    let mut outer = opad;
    outer.extend_from_slice(&inner_hash.to_le_bytes());
    fnv1a_64(&outer)
}

// ═══════════════════════════════════════════════════════════════════
// KEY DERIVATION (PBKDF2-like with FNV)
// ═══════════════════════════════════════════════════════════════════

/// Simple key derivation (educational, NOT cryptographically secure).
pub fn derive_key(password: &[u8], salt: &[u8], iterations: usize) -> u64 {
    let mut key = fnv1a_64(password);
    for _ in 0..iterations {
        let mut data = salt.to_vec();
        data.extend_from_slice(&key.to_le_bytes());
        key = fnv1a_64(&data);
    }
    key
}

// ═══════════════════════════════════════════════════════════════════
// XOR CIPHER (educational)
// ═══════════════════════════════════════════════════════════════════

/// XOR cipher (symmetric).
pub fn xor_cipher(data: &[u8], key: &[u8]) -> Vec<u8> {
    data.iter().enumerate().map(|(i, &b)| b ^ key[i % key.len()]).collect()
}

/// One-time pad (XOR with key of same length).
pub fn otp_encrypt(plaintext: &[u8], key: &[u8]) -> Vec<u8> {
    assert_eq!(plaintext.len(), key.len(), "OTP key must be same length as plaintext");
    plaintext.iter().zip(key).map(|(p, k)| p ^ k).collect()
}

// ═══════════════════════════════════════════════════════════════════
// BLOOM FILTER
// ═══════════════════════════════════════════════════════════════════

/// Simple Bloom filter.
pub struct BloomFilter {
    bits: Vec<bool>,
    n_hashes: usize,
    size: usize,
}

impl BloomFilter {
    /// Create with estimated capacity and false positive rate.
    pub fn new(capacity: usize, fp_rate: f64) -> Self {
        let size = (-(capacity as f64 * fp_rate.ln()) / (2.0f64.ln().powi(2))).ceil() as usize;
        let n_hashes = ((size as f64 / capacity as f64) * 2.0f64.ln()).ceil() as usize;
        Self { bits: vec![false; size], n_hashes: n_hashes.max(1), size }
    }

    /// Insert an element.
    pub fn insert(&mut self, data: &[u8]) {
        for i in 0..self.n_hashes {
            let idx = self.hash(data, i);
            self.bits[idx] = true;
        }
    }

    /// Check if element might be present.
    pub fn contains(&self, data: &[u8]) -> bool {
        (0..self.n_hashes).all(|i| self.bits[self.hash(data, i)])
    }

    fn hash(&self, data: &[u8], seed: usize) -> usize {
        let h1 = fnv1a_64(data);
        let h2 = djb2(data);
        (h1.wrapping_add((seed as u64).wrapping_mul(h2))) as usize % self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a() {
        let h = fnv1a_64(b"hello");
        assert!(h != 0);
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }

    #[test]
    fn test_crc32() {
        // Known CRC32 of "123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = b"Hello, World!";
        let hex = hex_encode(data);
        let decoded = hex_decode(&hex);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_known() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_xor_cipher() {
        let data = b"hello";
        let key = b"key";
        let encrypted = xor_cipher(data, key);
        let decrypted = xor_cipher(&encrypted, key);
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"hello");
        bf.insert(b"world");
        assert!(bf.contains(b"hello"));
        assert!(bf.contains(b"world"));
        // May have false positives but shouldn't for most random strings
    }

    #[test]
    fn test_murmur3() {
        let h = murmur3_32(b"hello", 0);
        assert!(h != 0);
    }

    #[test]
    fn test_adler32() {
        let h = adler32(b"hello");
        assert!(h != 0);
    }

    #[test]
    fn test_otp() {
        let plain = b"hello";
        let key = b"12345";
        let cipher = otp_encrypt(plain, key);
        let decrypted = otp_encrypt(&cipher, key);
        assert_eq!(decrypted, plain);
    }

    #[test]
    fn test_derive_key() {
        let k1 = derive_key(b"password", b"salt", 100);
        let k2 = derive_key(b"password", b"salt", 100);
        assert_eq!(k1, k2); // deterministic
        let k3 = derive_key(b"different", b"salt", 100);
        assert_ne!(k1, k3);
    }
}
