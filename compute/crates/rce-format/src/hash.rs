/// BLAKE3-256 hash of arbitrary bytes. Used for content addressing.
pub fn blake3_32(bytes: &[u8]) -> [u8; 32] {
    blake3::hash(bytes).into()
}

/// Hex-encode a 32-byte hash (lowercase, no separator).
pub fn hex_encode(hash: &[u8; 32]) -> String {
    hex::encode(hash)
}
