//! String operations: R/Python string function parity.
//!
//! Replaces R's grep/gsub/substr/paste, Python's str methods,
//! pandas .str accessor.

/// Split string by delimiter.
pub fn str_split(s: &str, sep: &str) -> Vec<String> {
    s.split(sep).map(|x| x.to_string()).collect()
}

/// Join strings with separator.
pub fn str_join(parts: &[&str], sep: &str) -> String {
    parts.join(sep)
}

/// Replace all occurrences.
pub fn str_replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

/// Replace first occurrence only.
pub fn str_replace_first(s: &str, from: &str, to: &str) -> String {
    s.replacen(from, to, 1)
}

/// Substring by byte positions (0-indexed, exclusive end).
pub fn substr(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end - start).collect()
}

/// Trim whitespace from both ends.
pub fn str_trim(s: &str) -> String { s.trim().to_string() }
/// Trim leading whitespace.
pub fn str_trim_start(s: &str) -> String { s.trim_start().to_string() }
/// Trim trailing whitespace.
pub fn str_trim_end(s: &str) -> String { s.trim_end().to_string() }

/// Convert to uppercase.
pub fn str_upper(s: &str) -> String { s.to_uppercase() }
/// Convert to lowercase.
pub fn str_lower(s: &str) -> String { s.to_lowercase() }
/// Capitalize first character.
pub fn str_capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}
/// Title case (capitalize first letter of each word).
pub fn str_title(s: &str) -> String {
    s.split_whitespace().map(|w| str_capitalize(w)).collect::<Vec<_>>().join(" ")
}

/// Check if string starts with prefix.
pub fn str_starts_with(s: &str, prefix: &str) -> bool { s.starts_with(prefix) }
/// Check if string ends with suffix.
pub fn str_ends_with(s: &str, suffix: &str) -> bool { s.ends_with(suffix) }
/// Check if string contains substring.
pub fn str_contains(s: &str, sub: &str) -> bool { s.contains(sub) }

/// Count occurrences of substring.
pub fn str_count(s: &str, sub: &str) -> usize { s.matches(sub).count() }

/// Length in characters.
pub fn str_len(s: &str) -> usize { s.chars().count() }

/// Pad string to width (left, right, or center).
pub fn str_pad_left(s: &str, width: usize, fill: char) -> String {
    if s.len() >= width { s.to_string() }
    else { format!("{}{}", std::iter::repeat(fill).take(width - s.len()).collect::<String>(), s) }
}
pub fn str_pad_right(s: &str, width: usize, fill: char) -> String {
    if s.len() >= width { s.to_string() }
    else { format!("{}{}", s, std::iter::repeat(fill).take(width - s.len()).collect::<String>()) }
}
pub fn str_center(s: &str, width: usize, fill: char) -> String {
    if s.len() >= width { return s.to_string(); }
    let pad = width - s.len();
    let left = pad / 2;
    let right = pad - left;
    format!("{}{}{}", std::iter::repeat(fill).take(left).collect::<String>(), s, std::iter::repeat(fill).take(right).collect::<String>())
}

/// Zero-fill number string.
pub fn str_zfill(s: &str, width: usize) -> String { str_pad_left(s, width, '0') }

/// Reverse a string.
pub fn str_reverse(s: &str) -> String { s.chars().rev().collect() }

/// Repeat string n times.
pub fn str_repeat(s: &str, n: usize) -> String { s.repeat(n) }

/// Find index of first occurrence (-1 if not found).
pub fn str_find(s: &str, sub: &str) -> i64 {
    s.find(sub).map(|i| i as i64).unwrap_or(-1)
}

/// Find index of last occurrence (-1 if not found).
pub fn str_rfind(s: &str, sub: &str) -> i64 {
    s.rfind(sub).map(|i| i as i64).unwrap_or(-1)
}

/// Check if string is all digits.
pub fn str_is_digit(s: &str) -> bool { !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) }
/// Check if string is all alphabetic.
pub fn str_is_alpha(s: &str) -> bool { !s.is_empty() && s.chars().all(|c| c.is_alphabetic()) }
/// Check if string is all alphanumeric.
pub fn str_is_alnum(s: &str) -> bool { !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()) }
/// Check if string is all whitespace.
pub fn str_is_space(s: &str) -> bool { !s.is_empty() && s.chars().all(|c| c.is_whitespace()) }
/// Check if string is all uppercase.
pub fn str_is_upper(s: &str) -> bool { !s.is_empty() && s == s.to_uppercase() }
/// Check if string is all lowercase.
pub fn str_is_lower(s: &str) -> bool { !s.is_empty() && s == s.to_lowercase() }

/// Remove prefix if present.
pub fn str_remove_prefix(s: &str, prefix: &str) -> String {
    s.strip_prefix(prefix).unwrap_or(s).to_string()
}
/// Remove suffix if present.
pub fn str_remove_suffix(s: &str, suffix: &str) -> String {
    s.strip_suffix(suffix).unwrap_or(s).to_string()
}

/// Wrap text to specified width.
pub fn str_wrap(s: &str, width: usize) -> String {
    let mut result = String::new();
    let mut line_len = 0;
    for word in s.split_whitespace() {
        if line_len + word.len() + 1 > width && line_len > 0 {
            result.push('\n');
            line_len = 0;
        }
        if line_len > 0 { result.push(' '); line_len += 1; }
        result.push_str(word);
        line_len += word.len();
    }
    result
}

/// Encode string to bytes (UTF-8).
pub fn str_encode(s: &str) -> Vec<u8> { s.as_bytes().to_vec() }
/// Decode bytes to string (UTF-8, lossy).
pub fn str_decode(bytes: &[u8]) -> String { String::from_utf8_lossy(bytes).to_string() }

/// Format number with thousands separator.
pub fn format_number(n: f64, decimals: usize, sep: char) -> String {
    let formatted = format!("{:.prec$}", n, prec = decimals);
    let parts: Vec<&str> = formatted.split('.').collect();
    let int_part = parts[0];
    let dec_part = if parts.len() > 1 { parts[1] } else { "" };

    let negative = int_part.starts_with('-');
    let digits: String = int_part.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut result = String::new();
    for (i, c) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(sep); }
        result.push(c);
    }
    let mut int_formatted: String = result.chars().rev().collect();
    if negative { int_formatted = format!("-{int_formatted}"); }
    if decimals > 0 { format!("{int_formatted}.{dec_part}") }
    else { int_formatted }
}

/// Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

/// Jaro-Winkler similarity (0-1, higher = more similar).
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() { return 1.0; }
    if a.is_empty() || b.is_empty() { return 0.0; }
    let match_dist = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut a_matches = vec![false; a.len()];
    let mut b_matches = vec![false; b.len()];
    let mut matches = 0.0;
    let mut transpositions = 0.0;
    for i in 0..a.len() {
        let lo = i.saturating_sub(match_dist);
        let hi = (i + match_dist + 1).min(b.len());
        for j in lo..hi {
            if !b_matches[j] && a[i] == b[j] {
                a_matches[i] = true;
                b_matches[j] = true;
                matches += 1.0;
                break;
            }
        }
    }
    if matches == 0.0 { return 0.0; }
    let mut k = 0;
    for i in 0..a.len() {
        if !a_matches[i] { continue; }
        while !b_matches[k] { k += 1; }
        if a[i] != b[k] { transpositions += 1.0; }
        k += 1;
    }
    let jaro = (matches / a.len() as f64 + matches / b.len() as f64 + (matches - transpositions / 2.0) / matches) / 3.0;
    // Winkler bonus
    let mut prefix = 0;
    for i in 0..4.min(a.len()).min(b.len()) {
        if a[i] == b[i] { prefix += 1; } else { break; }
    }
    jaro + prefix as f64 * 0.1 * (1.0 - jaro)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_join() {
        let parts = str_split("a,b,c", ",");
        assert_eq!(parts, vec!["a", "b", "c"]);
        assert_eq!(str_join(&["a", "b", "c"], "-"), "a-b-c");
    }

    #[test]
    fn test_replace() {
        assert_eq!(str_replace("hello world", "world", "rust"), "hello rust");
        assert_eq!(str_replace_first("aaa", "a", "b"), "baa");
    }

    #[test]
    fn test_case() {
        assert_eq!(str_upper("hello"), "HELLO");
        assert_eq!(str_lower("HELLO"), "hello");
        assert_eq!(str_capitalize("hello"), "Hello");
        assert_eq!(str_title("hello world"), "Hello World");
    }

    #[test]
    fn test_pad() {
        assert_eq!(str_pad_left("42", 5, '0'), "00042");
        assert_eq!(str_pad_right("hi", 5, ' '), "hi   ");
        assert_eq!(str_center("hi", 6, '-'), "--hi--");
        assert_eq!(str_zfill("7", 3), "007");
    }

    #[test]
    fn test_find_count() {
        assert_eq!(str_find("hello world", "world"), 6);
        assert_eq!(str_find("hello", "xyz"), -1);
        assert_eq!(str_count("banana", "an"), 2);
    }

    #[test]
    fn test_checks() {
        assert!(str_is_digit("12345"));
        assert!(!str_is_digit("123a5"));
        assert!(str_is_alpha("hello"));
        assert!(str_starts_with("hello", "hel"));
        assert!(str_ends_with("hello", "llo"));
        assert!(str_contains("hello world", "lo wo"));
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn test_jaro_winkler() {
        assert!((jaro_winkler("martha", "marhta") - 0.961).abs() < 0.01);
        assert!((jaro_winkler("abc", "abc") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234567.89, 2, ','), "1,234,567.89");
        assert_eq!(format_number(42.0, 0, ','), "42");
    }

    #[test]
    fn test_wrap() {
        let wrapped = str_wrap("the quick brown fox jumps over the lazy dog", 15);
        assert!(wrapped.contains('\n'));
    }

    #[test]
    fn test_substr() {
        assert_eq!(substr("hello world", 6, 11), "world");
    }

    #[test]
    fn test_reverse() {
        assert_eq!(str_reverse("hello"), "olleh");
    }
}
