//! Compression algorithms: Huffman coding, Run-Length Encoding,
//! LZ77, delta encoding, variable-length encoding.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ═══════════════════════════════════════════════════════════════════
// HUFFMAN CODING
// ═══════════════════════════════════════════════════════════════════

/// Huffman tree node.
#[derive(Debug, Clone)]
enum HuffNode {
    Leaf { byte: u8, freq: usize },
    Internal { freq: usize, left: Box<HuffNode>, right: Box<HuffNode> },
}
impl HuffNode {
    fn freq(&self) -> usize {
        match self { HuffNode::Leaf { freq, .. } | HuffNode::Internal { freq, .. } => *freq }
    }
}
impl PartialEq for HuffNode { fn eq(&self, other: &Self) -> bool { self.freq() == other.freq() } }
impl Eq for HuffNode {}
impl PartialOrd for HuffNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HuffNode {
    fn cmp(&self, other: &Self) -> Ordering { other.freq().cmp(&self.freq()) } // min-heap
}

/// Build Huffman code table from byte frequencies.
pub fn huffman_code_table(data: &[u8]) -> HashMap<u8, Vec<bool>> {
    let mut freq = HashMap::new();
    for &b in data { *freq.entry(b).or_insert(0usize) += 1; }
    if freq.len() <= 1 {
        let mut table = HashMap::new();
        for &b in freq.keys() { table.insert(b, vec![false]); }
        return table;
    }
    let mut heap = BinaryHeap::new();
    for (&byte, &f) in &freq { heap.push(HuffNode::Leaf { byte, freq: f }); }
    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();
        let merged = HuffNode::Internal { freq: left.freq() + right.freq(), left: Box::new(left), right: Box::new(right) };
        heap.push(merged);
    }
    let root = heap.pop().unwrap();
    let mut table = HashMap::new();
    build_codes(&root, &mut Vec::new(), &mut table);
    table
}

fn build_codes(node: &HuffNode, prefix: &mut Vec<bool>, table: &mut HashMap<u8, Vec<bool>>) {
    match node {
        HuffNode::Leaf { byte, .. } => { table.insert(*byte, prefix.clone()); }
        HuffNode::Internal { left, right, .. } => {
            prefix.push(false); build_codes(left, prefix, table); prefix.pop();
            prefix.push(true); build_codes(right, prefix, table); prefix.pop();
        }
    }
}

/// Huffman encode data. Returns (encoded_bits, code_table).
pub fn huffman_encode(data: &[u8]) -> (Vec<bool>, HashMap<u8, Vec<bool>>) {
    let table = huffman_code_table(data);
    let bits: Vec<bool> = data.iter().flat_map(|b| table.get(b).unwrap().iter().copied()).collect();
    (bits, table)
}

/// Huffman decode bits using code table.
pub fn huffman_decode(bits: &[bool], table: &HashMap<u8, Vec<bool>>) -> Vec<u8> {
    // Build reverse lookup
    let reverse: HashMap<Vec<bool>, u8> = table.iter().map(|(&b, code)| (code.clone(), b)).collect();
    let max_len = reverse.keys().map(|k| k.len()).max().unwrap_or(0);
    let mut result = Vec::new();
    let mut i = 0;
    while i < bits.len() {
        for len in 1..=max_len {
            if i + len > bits.len() { break; }
            let candidate: Vec<bool> = bits[i..i + len].to_vec();
            if let Some(&byte) = reverse.get(&candidate) {
                result.push(byte);
                i += len;
                break;
            }
        }
    }
    result
}

/// Compression ratio of Huffman encoding.
pub fn huffman_compression_ratio(data: &[u8]) -> f64 {
    let (bits, _) = huffman_encode(data);
    let compressed_bytes = (bits.len() + 7) / 8;
    data.len() as f64 / compressed_bytes.max(1) as f64
}

// ═══════════════════════════════════════════════════════════════════
// RUN-LENGTH ENCODING
// ═══════════════════════════════════════════════════════════════════

/// RLE encode bytes. Returns vec of (value, count) pairs.
pub fn rle_encode(data: &[u8]) -> Vec<(u8, usize)> {
    if data.is_empty() { return vec![]; }
    let mut result = Vec::new();
    let mut current = data[0];
    let mut count = 1;
    for &b in &data[1..] {
        if b == current && count < 255 { count += 1; }
        else { result.push((current, count)); current = b; count = 1; }
    }
    result.push((current, count));
    result
}

/// RLE decode.
pub fn rle_decode(encoded: &[(u8, usize)]) -> Vec<u8> {
    encoded.iter().flat_map(|&(val, count)| std::iter::repeat(val).take(count)).collect()
}

/// RLE encode for f64 arrays (run-length of equal values).
pub fn rle_encode_f64(data: &[f64]) -> Vec<(f64, usize)> {
    if data.is_empty() { return vec![]; }
    let mut result = Vec::new();
    let mut current = data[0];
    let mut count = 1;
    for &v in &data[1..] {
        if (v - current).abs() < 1e-15 { count += 1; }
        else { result.push((current, count)); current = v; count = 1; }
    }
    result.push((current, count));
    result
}

/// RLE decode for f64 arrays.
pub fn rle_decode_f64(encoded: &[(f64, usize)]) -> Vec<f64> {
    encoded.iter().flat_map(|&(val, count)| std::iter::repeat(val).take(count)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// DELTA ENCODING
// ═══════════════════════════════════════════════════════════════════

/// Delta encode: store differences between consecutive values.
pub fn delta_encode(data: &[f64]) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let mut result = vec![data[0]];
    for i in 1..data.len() { result.push(data[i] - data[i - 1]); }
    result
}

/// Delta decode.
pub fn delta_decode(encoded: &[f64]) -> Vec<f64> {
    if encoded.is_empty() { return vec![]; }
    let mut result = vec![encoded[0]];
    for i in 1..encoded.len() { result.push(result[i - 1] + encoded[i]); }
    result
}

/// Double delta encode (delta of delta — good for nearly linear data).
pub fn double_delta_encode(data: &[f64]) -> Vec<f64> {
    delta_encode(&delta_encode(data))
}

/// Double delta decode.
pub fn double_delta_decode(encoded: &[f64]) -> Vec<f64> {
    delta_decode(&delta_decode(encoded))
}

// ═══════════════════════════════════════════════════════════════════
// LZ77 (simplified)
// ═══════════════════════════════════════════════════════════════════

/// LZ77 token: (offset, length, next_char).
#[derive(Debug, Clone, PartialEq)]
pub struct Lz77Token {
    pub offset: usize,
    pub length: usize,
    pub next: u8,
}

/// LZ77 compress.
pub fn lz77_compress(data: &[u8], window_size: usize, lookahead_size: usize) -> Vec<Lz77Token> {
    let n = data.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let mut best_offset = 0;
        let mut best_length = 0;
        let search_start = if i > window_size { i - window_size } else { 0 };
        for j in search_start..i {
            let mut length = 0;
            while length < lookahead_size && i + length < n && data[j + length] == data[i + length] {
                length += 1;
                if j + length >= i { break; }
            }
            if length > best_length { best_length = length; best_offset = i - j; }
        }
        let next = if i + best_length < n { data[i + best_length] } else { 0 };
        tokens.push(Lz77Token { offset: best_offset, length: best_length, next });
        i += best_length + 1;
    }
    tokens
}

/// LZ77 decompress.
pub fn lz77_decompress(tokens: &[Lz77Token]) -> Vec<u8> {
    let mut output = Vec::new();
    for token in tokens {
        if token.length > 0 {
            let start = output.len() - token.offset;
            for i in 0..token.length { output.push(output[start + i]); }
        }
        if token.next != 0 || token.length == 0 { output.push(token.next); }
    }
    output
}

// ═══════════════════════════════════════════════════════════════════
// VARIABLE-LENGTH ENCODING (Elias gamma/delta, Fibonacci)
// ═══════════════════════════════════════════════════════════════════

/// Elias gamma encoding of positive integer.
pub fn elias_gamma_encode(n: u64) -> Vec<bool> {
    if n == 0 { return vec![false]; }
    let bits = 64 - n.leading_zeros() as usize;
    let mut code = vec![false; bits - 1]; // leading zeros
    for i in (0..bits).rev() { code.push((n >> i) & 1 == 1); }
    code
}

/// Elias gamma decode (from bit iterator, returns value and bits consumed).
pub fn elias_gamma_decode(bits: &[bool]) -> (u64, usize) {
    let mut zeros = 0;
    while zeros < bits.len() && !bits[zeros] { zeros += 1; }
    if zeros >= bits.len() { return (0, bits.len()); }
    let total = 2 * zeros + 1;
    let mut value = 0u64;
    for i in 0..zeros + 1 {
        if bits[zeros + i] { value |= 1 << (zeros - i); }
    }
    (value, total)
}

/// Zigzag encoding (map signed to unsigned: 0→0, -1→1, 1→2, -2→3, ...).
pub fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Zigzag decoding.
pub fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ -((n & 1) as i64)
}

/// Variable-length quantity encoding (used in MIDI, protobuf).
pub fn vlq_encode(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value > 0 { byte |= 0x80; }
        bytes.push(byte);
        if value == 0 { break; }
    }
    bytes
}

/// Variable-length quantity decoding.
pub fn vlq_decode(bytes: &[u8]) -> (u64, usize) {
    let mut value = 0u64;
    let mut shift = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 { return (value, i + 1); }
    }
    (value, bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_roundtrip() {
        let data = b"aabbbccccdddddd";
        let (bits, table) = huffman_encode(data);
        let decoded = huffman_decode(&bits, &table);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_huffman_compression() {
        let data = b"aaaaaabbbbbbccccccddddddeeeeee";
        let ratio = huffman_compression_ratio(data);
        assert!(ratio > 1.0); // should compress
    }

    #[test]
    fn test_rle_roundtrip() {
        let data = vec![1u8, 1, 1, 2, 2, 3, 3, 3, 3];
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_f64_roundtrip() {
        let data = vec![1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
        let encoded = rle_encode_f64(&data);
        let decoded = rle_decode_f64(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_roundtrip() {
        let data = vec![10.0, 12.0, 15.0, 19.0, 24.0];
        let encoded = delta_encode(&data);
        let decoded = delta_decode(&encoded);
        for (a, b) in data.iter().zip(&decoded) { assert!((a - b).abs() < 1e-10); }
    }

    #[test]
    fn test_double_delta_roundtrip() {
        let data = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let encoded = double_delta_encode(&data);
        let decoded = double_delta_decode(&encoded);
        for (a, b) in data.iter().zip(&decoded) { assert!((a - b).abs() < 1e-10); }
    }

    #[test]
    fn test_lz77_roundtrip() {
        let data = b"abcabcabc";
        let tokens = lz77_compress(data, 8, 4);
        let decoded = lz77_decompress(&tokens);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_decode(zigzag_encode(-42)), -42);
    }

    #[test]
    fn test_vlq_roundtrip() {
        for val in [0, 1, 127, 128, 300, 16384, 1_000_000u64] {
            let encoded = vlq_encode(val);
            let (decoded, _) = vlq_decode(&encoded);
            assert_eq!(decoded, val);
        }
    }

    #[test]
    fn test_elias_gamma() {
        let bits = elias_gamma_encode(7);
        let (decoded, _) = elias_gamma_decode(&bits);
        assert_eq!(decoded, 7);
    }
}
