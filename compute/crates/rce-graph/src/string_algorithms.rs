//! String algorithms: suffix array, KMP, Rabin-Karp, Boyer-Moore,
//! longest common substring, edit distance variants, string hashing.

// ═══════════════════════════════════════════════════════════════════
// PATTERN MATCHING
// ═══════════════════════════════════════════════════════════════════

/// Knuth-Morris-Pratt pattern search. Returns all match positions.
pub fn kmp_search(text: &str, pattern: &str) -> Vec<usize> {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    if p.is_empty() || t.len() < p.len() { return vec![]; }
    let fail = kmp_failure(&p);
    let mut results = Vec::new();
    let mut j = 0;
    for i in 0..t.len() {
        while j > 0 && t[i] != p[j] { j = fail[j - 1]; }
        if t[i] == p[j] { j += 1; }
        if j == p.len() { results.push(i + 1 - j); j = fail[j - 1]; }
    }
    results
}

fn kmp_failure(pattern: &[char]) -> Vec<usize> {
    let n = pattern.len();
    let mut fail = vec![0; n];
    let mut k = 0;
    for i in 1..n {
        while k > 0 && pattern[i] != pattern[k] { k = fail[k - 1]; }
        if pattern[i] == pattern[k] { k += 1; }
        fail[i] = k;
    }
    fail
}

/// Rabin-Karp pattern search (rolling hash).
pub fn rabin_karp_search(text: &str, pattern: &str) -> Vec<usize> {
    let t: Vec<u8> = text.bytes().collect();
    let p: Vec<u8> = pattern.bytes().collect();
    if p.is_empty() || t.len() < p.len() { return vec![]; }
    let base = 256u64;
    let modulus = 1_000_000_007u64;
    let m = p.len();

    let mut p_hash = 0u64;
    let mut t_hash = 0u64;
    let mut h = 1u64;
    for _ in 0..m - 1 { h = (h * base) % modulus; }

    for i in 0..m {
        p_hash = (base * p_hash + p[i] as u64) % modulus;
        t_hash = (base * t_hash + t[i] as u64) % modulus;
    }

    let mut results = Vec::new();
    for i in 0..=t.len() - m {
        if p_hash == t_hash && t[i..i + m] == p[..] { results.push(i); }
        if i + m < t.len() {
            t_hash = (base * (t_hash + modulus - (t[i] as u64 * h) % modulus) + t[i + m] as u64) % modulus;
        }
    }
    results
}

/// Boyer-Moore-Horspool (simplified bad character heuristic).
pub fn bmh_search(text: &str, pattern: &str) -> Vec<usize> {
    let t: Vec<u8> = text.bytes().collect();
    let p: Vec<u8> = pattern.bytes().collect();
    if p.is_empty() || t.len() < p.len() { return vec![]; }
    let m = p.len();
    // Bad character table
    let mut shift = vec![m; 256];
    for i in 0..m - 1 { shift[p[i] as usize] = m - 1 - i; }

    let mut results = Vec::new();
    let mut i = 0;
    while i + m <= t.len() {
        let mut j = m - 1;
        while t[i + j] == p[j] {
            if j == 0 { results.push(i); break; }
            j -= 1;
        }
        i += shift[t[i + m - 1] as usize];
    }
    results
}

/// Count occurrences of pattern in text.
pub fn count_occurrences(text: &str, pattern: &str) -> usize {
    kmp_search(text, pattern).len()
}

/// Find all overlapping occurrences.
pub fn find_all_overlapping(text: &str, pattern: &str) -> Vec<usize> {
    kmp_search(text, pattern) // KMP already finds overlapping matches
}

// ═══════════════════════════════════════════════════════════════════
// SUFFIX ARRAY
// ═══════════════════════════════════════════════════════════════════

/// Build suffix array (naive O(n² log n) — fine for moderate strings).
pub fn suffix_array(s: &str) -> Vec<usize> {
    let n = s.len();
    let bytes = s.as_bytes();
    let mut sa: Vec<usize> = (0..n).collect();
    sa.sort_by(|&a, &b| bytes[a..].cmp(&bytes[b..]));
    sa
}

/// LCP array from suffix array.
pub fn lcp_array(s: &str, sa: &[usize]) -> Vec<usize> {
    let n = s.len();
    let bytes = s.as_bytes();
    let mut rank = vec![0usize; n];
    for (i, &s) in sa.iter().enumerate() { rank[s] = i; }

    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        if rank[i] > 0 {
            let j = sa[rank[i] - 1];
            while i + h < n && j + h < n && bytes[i + h] == bytes[j + h] { h += 1; }
            lcp[rank[i]] = h;
            if h > 0 { h -= 1; }
        } else {
            h = 0;
        }
    }
    lcp
}

/// Find longest repeated substring using suffix array + LCP.
pub fn longest_repeated_substring(s: &str) -> String {
    let sa = suffix_array(s);
    let lcp = lcp_array(s, &sa);
    let max_idx = lcp.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i).unwrap_or(0);
    let max_len = lcp[max_idx];
    if max_len == 0 { return String::new(); }
    s[sa[max_idx]..sa[max_idx] + max_len].to_string()
}

/// Count distinct substrings using suffix array + LCP.
pub fn count_distinct_substrings(s: &str) -> usize {
    let n = s.len();
    let sa = suffix_array(s);
    let lcp = lcp_array(s, &sa);
    let total: usize = sa.iter().map(|&i| n - i).sum();
    let lcp_sum: usize = lcp.iter().sum();
    total - lcp_sum
}

// ═══════════════════════════════════════════════════════════════════
// LONGEST COMMON SUBSTRING / SUBSEQUENCE
// ═══════════════════════════════════════════════════════════════════

/// Longest common substring (returns the string itself).
pub fn longest_common_substring(a: &str, b: &str) -> String {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let (m, n) = (ac.len(), bc.len());
    let mut max_len = 0;
    let mut max_end = 0;
    let mut prev = vec![0usize; n + 1];
    for i in 1..=m {
        let mut curr = vec![0usize; n + 1];
        for j in 1..=n {
            if ac[i-1] == bc[j-1] {
                curr[j] = prev[j-1] + 1;
                if curr[j] > max_len { max_len = curr[j]; max_end = i; }
            }
        }
        prev = curr;
    }
    if max_len == 0 { String::new() } else { ac[max_end - max_len..max_end].iter().collect() }
}

/// Longest common subsequence (returns the subsequence).
pub fn longest_common_subsequence(a: &str, b: &str) -> String {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let (m, n) = (ac.len(), bc.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if ac[i-1] == bc[j-1] { dp[i-1][j-1] + 1 } else { dp[i-1][j].max(dp[i][j-1]) };
        }
    }
    // Traceback
    let mut result = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if ac[i-1] == bc[j-1] { result.push(ac[i-1]); i -= 1; j -= 1; }
        else if dp[i-1][j] > dp[i][j-1] { i -= 1; }
        else { j -= 1; }
    }
    result.reverse();
    result.into_iter().collect()
}

// ═══════════════════════════════════════════════════════════════════
// STRING DISTANCE
// ═══════════════════════════════════════════════════════════════════

/// Normalized Levenshtein similarity: 1 - edit_dist/max_len.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let dist = crate::text_analysis::levenshtein(a, b);
    let max_len = a.len().max(b.len());
    if max_len == 0 { return 1.0; }
    1.0 - dist as f64 / max_len as f64
}

/// Jaro similarity.
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    if ac.is_empty() && bc.is_empty() { return 1.0; }
    if ac.is_empty() || bc.is_empty() { return 0.0; }
    let match_dist = (ac.len().max(bc.len()) / 2).saturating_sub(1);
    let mut a_matched = vec![false; ac.len()];
    let mut b_matched = vec![false; bc.len()];
    let mut matches = 0.0;
    let mut transpositions = 0.0;

    for i in 0..ac.len() {
        let start = if i > match_dist { i - match_dist } else { 0 };
        let end = (i + match_dist + 1).min(bc.len());
        for j in start..end {
            if b_matched[j] || ac[i] != bc[j] { continue; }
            a_matched[i] = true;
            b_matched[j] = true;
            matches += 1.0;
            break;
        }
    }
    if matches < 1e-15 { return 0.0; }

    let mut k = 0;
    for i in 0..ac.len() {
        if !a_matched[i] { continue; }
        while !b_matched[k] { k += 1; }
        if ac[i] != bc[k] { transpositions += 1.0; }
        k += 1;
    }

    (matches / ac.len() as f64 + matches / bc.len() as f64 + (matches - transpositions / 2.0) / matches) / 3.0
}

/// Jaro-Winkler similarity.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let jaro = jaro_similarity(a, b);
    let prefix_len = a.chars().zip(b.chars()).take(4).take_while(|(x, y)| x == y).count();
    jaro + prefix_len as f64 * 0.1 * (1.0 - jaro)
}

/// Dice coefficient on character bigrams.
pub fn dice_bigram(a: &str, b: &str) -> f64 {
    if a.len() < 2 || b.len() < 2 { return 0.0; }
    let bigrams_a: Vec<String> = a.as_bytes().windows(2).map(|w| String::from_utf8_lossy(w).to_string()).collect();
    let bigrams_b: Vec<String> = b.as_bytes().windows(2).map(|w| String::from_utf8_lossy(w).to_string()).collect();
    let inter = bigrams_a.iter().filter(|bg| bigrams_b.contains(bg)).count();
    2.0 * inter as f64 / (bigrams_a.len() + bigrams_b.len()) as f64
}

// ═══════════════════════════════════════════════════════════════════
// STRING UTILITIES
// ═══════════════════════════════════════════════════════════════════

/// Check if string is a palindrome.
pub fn is_palindrome(s: &str) -> bool {
    let chars: Vec<char> = s.chars().filter(|c| c.is_alphanumeric()).map(|c| c.to_lowercase().next().unwrap()).collect();
    let n = chars.len();
    (0..n / 2).all(|i| chars[i] == chars[n - 1 - i])
}

/// Longest palindromic substring.
pub fn longest_palindrome(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n == 0 { return String::new(); }
    let mut best_start = 0;
    let mut best_len = 1;
    for center in 0..2 * n - 1 {
        let mut l = center as i64 / 2;
        let mut r = l + (center % 2) as i64;
        while l >= 0 && (r as usize) < n && chars[l as usize] == chars[r as usize] {
            let len = (r - l + 1) as usize;
            if len > best_len { best_len = len; best_start = l as usize; }
            l -= 1; r += 1;
        }
    }
    chars[best_start..best_start + best_len].iter().collect()
}

/// Run-length encode a string.
pub fn string_rle_encode(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() { return String::new(); }
    let mut result = String::new();
    let mut count = 1;
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] { count += 1; }
        else { result.push(chars[i - 1]); if count > 1 { result.push_str(&count.to_string()); } count = 1; }
    }
    result.push(*chars.last().unwrap());
    if count > 1 { result.push_str(&count.to_string()); }
    result
}

/// Reverse words in a string.
pub fn reverse_words(s: &str) -> String {
    s.split_whitespace().rev().collect::<Vec<&str>>().join(" ")
}

/// Compress consecutive spaces to single space.
pub fn compress_spaces(s: &str) -> String {
    let mut result = String::new();
    let mut prev_space = true;
    for c in s.chars() {
        if c.is_whitespace() { if !prev_space { result.push(' '); prev_space = true; } }
        else { result.push(c); prev_space = false; }
    }
    result.trim().to_string()
}

/// Rotate string left by n positions.
pub fn rotate_string(s: &str, n: usize) -> String {
    if s.is_empty() { return String::new(); }
    let n = n % s.len();
    format!("{}{}", &s[n..], &s[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmp() {
        assert_eq!(kmp_search("ababcababd", "ababd"), vec![5]);
        assert_eq!(kmp_search("aaaaaa", "aa"), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_rabin_karp() {
        assert_eq!(rabin_karp_search("hello world hello", "hello"), vec![0, 12]);
    }

    #[test]
    fn test_bmh() {
        assert_eq!(bmh_search("hello world", "world"), vec![6]);
    }

    #[test]
    fn test_suffix_array() {
        let sa = suffix_array("banana");
        assert_eq!(sa[0], 5); // "a"
    }

    #[test]
    fn test_longest_repeated_substring() {
        assert_eq!(longest_repeated_substring("banana"), "ana");
    }

    #[test]
    fn test_lcs_string() {
        assert_eq!(longest_common_substring("abcdef", "zbcdf"), "bcd");
    }

    #[test]
    fn test_lcs_subsequence() {
        assert_eq!(longest_common_subsequence("ABCBDAB", "BDCAB").len(), 4);
    }

    #[test]
    fn test_jaro_winkler() {
        let sim = jaro_winkler("martha", "marhta");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_palindrome() {
        assert!(is_palindrome("racecar"));
        assert!(!is_palindrome("hello"));
    }

    #[test]
    fn test_longest_palindrome() {
        assert_eq!(longest_palindrome("babad").len(), 3);
    }

    #[test]
    fn test_reverse_words() {
        assert_eq!(reverse_words("hello world foo"), "foo world hello");
    }

    #[test]
    fn test_rotate_string() {
        assert_eq!(rotate_string("abcdef", 2), "cdefab");
    }

    #[test]
    fn test_string_rle() {
        assert_eq!(string_rle_encode("aaabbc"), "a3b2c");
    }
}
