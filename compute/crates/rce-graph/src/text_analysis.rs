//! Text analysis: tokenization, n-grams, TF-IDF, bag of words,
//! cosine similarity, edit distances, text statistics, hashing.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// TOKENIZATION
// ═══════════════════════════════════════════════════════════════════

/// Simple whitespace + punctuation tokenizer (lowercased).
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Tokenize into sentences (split on .!?).
pub fn sentence_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Remove common English stop words.
pub fn remove_stopwords(tokens: &[String]) -> Vec<String> {
    let stops: std::collections::HashSet<&str> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "and", "but", "or",
        "nor", "not", "so", "yet", "for", "of", "at", "by", "to", "from",
        "in", "on", "with", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "about", "up", "it", "its", "i", "me", "my", "he", "him",
        "his", "she", "her", "we", "us", "our", "they", "them", "their",
        "this", "that", "these", "those", "what", "which", "who", "whom",
    ].iter().copied().collect();
    tokens.iter().filter(|t| !stops.contains(t.as_str())).cloned().collect()
}

/// Simple stemming (Porter-like suffix stripping, very basic).
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    let suffixes = ["ings", "ing", "tion", "sion", "ment", "ness", "able", "ible",
                    "ful", "less", "ous", "ive", "ity", "ies", "es", "ed", "er", "ly", "s"];
    for suffix in &suffixes {
        if w.len() > suffix.len() + 2 && w.ends_with(suffix) {
            return w[..w.len() - suffix.len()].to_string();
        }
    }
    w
}

// ═══════════════════════════════════════════════════════════════════
// N-GRAMS
// ═══════════════════════════════════════════════════════════════════

/// Generate word n-grams.
pub fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n { return vec![]; }
    (0..=tokens.len() - n).map(|i| tokens[i..i + n].to_vec()).collect()
}

/// Generate character n-grams from a string.
pub fn char_ngrams(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n { return vec![]; }
    (0..=chars.len() - n).map(|i| chars[i..i + n].iter().collect()).collect()
}

/// Skip-grams: n-grams with up to `skip` tokens skipped between.
pub fn skip_grams(tokens: &[String], n: usize, skip: usize) -> Vec<Vec<String>> {
    if n < 2 || tokens.len() < n { return vec![]; }
    let mut result = Vec::new();
    for i in 0..tokens.len() {
        for j in i+1..=(i + skip + 1).min(tokens.len() - 1) {
            if n == 2 {
                result.push(vec![tokens[i].clone(), tokens[j].clone()]);
            }
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// BAG OF WORDS / TERM FREQUENCY
// ═══════════════════════════════════════════════════════════════════

/// Term frequency: count of each token.
pub fn term_frequency(tokens: &[String]) -> HashMap<String, f64> {
    let mut tf = HashMap::new();
    let n = tokens.len() as f64;
    for t in tokens { *tf.entry(t.clone()).or_insert(0.0) += 1.0; }
    for v in tf.values_mut() { *v /= n; }
    tf
}

/// Raw term counts.
pub fn term_counts(tokens: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for t in tokens { *counts.entry(t.clone()).or_insert(0) += 1; }
    counts
}

/// Document frequency across a corpus of token lists.
pub fn document_frequency(corpus: &[Vec<String>]) -> HashMap<String, usize> {
    let mut df = HashMap::new();
    for doc in corpus {
        let unique: std::collections::HashSet<&String> = doc.iter().collect();
        for t in unique { *df.entry(t.clone()).or_insert(0) += 1; }
    }
    df
}

/// TF-IDF for a single document given a corpus.
pub fn tfidf(tokens: &[String], corpus: &[Vec<String>]) -> HashMap<String, f64> {
    let tf = term_frequency(tokens);
    let df = document_frequency(corpus);
    let n = corpus.len() as f64;
    let mut result = HashMap::new();
    for (term, &freq) in &tf {
        let idf = (n / (*df.get(term).unwrap_or(&1) as f64)).ln() + 1.0;
        result.insert(term.clone(), freq * idf);
    }
    result
}

/// Build vocabulary from corpus. Returns sorted unique terms.
pub fn build_vocabulary(corpus: &[Vec<String>]) -> Vec<String> {
    let mut vocab: Vec<String> = corpus.iter()
        .flat_map(|doc| doc.iter().cloned())
        .collect::<std::collections::HashSet<String>>()
        .into_iter().collect();
    vocab.sort();
    vocab
}

/// Bag of words: convert document to fixed-size vector using vocabulary.
pub fn bag_of_words(tokens: &[String], vocab: &[String]) -> Vec<f64> {
    let counts = term_counts(tokens);
    vocab.iter().map(|w| *counts.get(w).unwrap_or(&0) as f64).collect()
}

/// TF-IDF matrix for a corpus. Returns (matrix row-major, vocabulary).
pub fn tfidf_matrix(corpus: &[Vec<String>]) -> (Vec<f64>, Vec<String>) {
    let vocab = build_vocabulary(corpus);
    let n_docs = corpus.len();
    let n_terms = vocab.len();
    let df = document_frequency(corpus);
    let n = n_docs as f64;

    let mut matrix = vec![0.0; n_docs * n_terms];
    for (di, doc) in corpus.iter().enumerate() {
        let tf = term_frequency(doc);
        for (ti, term) in vocab.iter().enumerate() {
            if let Some(&freq) = tf.get(term) {
                let idf = (n / *df.get(term).unwrap_or(&1) as f64).ln() + 1.0;
                matrix[di * n_terms + ti] = freq * idf;
            }
        }
    }
    (matrix, vocab)
}

// ═══════════════════════════════════════════════════════════════════
// SIMILARITY MEASURES
// ═══════════════════════════════════════════════════════════════════

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { return 0.0; }
    dot / (na * nb)
}

/// Jaccard similarity between two token sets.
pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    let set_a: std::collections::HashSet<&String> = a.iter().collect();
    let set_b: std::collections::HashSet<&String> = b.iter().collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union < 1e-15 { return 0.0; }
    inter / union
}

// ═══════════════════════════════════════════════════════════════════
// EDIT DISTANCES
// ═══════════════════════════════════════════════════════════════════

/// Levenshtein distance.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    let mut dp = vec![0usize; (m + 1) * (n + 1)];
    for i in 0..=m { dp[i * (n + 1)] = i; }
    for j in 0..=n { dp[j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
            dp[i * (n + 1) + j] = (dp[(i-1) * (n + 1) + j] + 1)
                .min(dp[i * (n + 1) + j - 1] + 1)
                .min(dp[(i-1) * (n + 1) + j - 1] + cost);
        }
    }
    dp[m * (n + 1) + n]
}

/// Damerau-Levenshtein distance (allows transpositions).
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    let mut dp = vec![0usize; (m + 1) * (n + 1)];
    for i in 0..=m { dp[i * (n + 1)] = i; }
    for j in 0..=n { dp[j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
            let mut d = (dp[(i-1) * (n + 1) + j] + 1)
                .min(dp[i * (n + 1) + j - 1] + 1)
                .min(dp[(i-1) * (n + 1) + j - 1] + cost);
            if i > 1 && j > 1 && a_chars[i-1] == b_chars[j-2] && a_chars[i-2] == b_chars[j-1] {
                d = d.min(dp[(i-2) * (n + 1) + j - 2] + cost);
            }
            dp[i * (n + 1) + j] = d;
        }
    }
    dp[m * (n + 1) + n]
}

/// Longest Common Subsequence length.
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    let mut dp = vec![0usize; (m + 1) * (n + 1)];
    for i in 1..=m {
        for j in 1..=n {
            dp[i * (n + 1) + j] = if a_chars[i-1] == b_chars[j-1] {
                dp[(i-1) * (n + 1) + j - 1] + 1
            } else {
                dp[(i-1) * (n + 1) + j].max(dp[i * (n + 1) + j - 1])
            };
        }
    }
    dp[m * (n + 1) + n]
}

/// Longest Common Substring length.
pub fn lc_substring_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    let mut max_len = 0;
    let mut prev = vec![0usize; n + 1];
    for i in 1..=m {
        let mut curr = vec![0usize; n + 1];
        for j in 1..=n {
            if a_chars[i-1] == b_chars[j-1] {
                curr[j] = prev[j - 1] + 1;
                max_len = max_len.max(curr[j]);
            }
        }
        prev = curr;
    }
    max_len
}

// ═══════════════════════════════════════════════════════════════════
// TEXT STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Word count.
pub fn word_count(text: &str) -> usize {
    tokenize(text).len()
}

/// Character count (excluding whitespace).
pub fn char_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

/// Average word length.
pub fn avg_word_length(text: &str) -> f64 {
    let tokens = tokenize(text);
    if tokens.is_empty() { return 0.0; }
    let total: usize = tokens.iter().map(|t| t.len()).sum();
    total as f64 / tokens.len() as f64
}

/// Type-token ratio (vocabulary richness).
pub fn type_token_ratio(tokens: &[String]) -> f64 {
    if tokens.is_empty() { return 0.0; }
    let unique: std::collections::HashSet<&String> = tokens.iter().collect();
    unique.len() as f64 / tokens.len() as f64
}

/// Flesch reading ease score.
pub fn flesch_reading_ease(text: &str) -> f64 {
    let sentences = sentence_tokenize(text);
    let words = tokenize(text);
    let syllables: usize = words.iter().map(|w| count_syllables(w)).sum();
    let n_sent = sentences.len().max(1) as f64;
    let n_words = words.len().max(1) as f64;
    206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables as f64 / n_words)
}

/// Flesch-Kincaid grade level.
pub fn flesch_kincaid_grade(text: &str) -> f64 {
    let sentences = sentence_tokenize(text);
    let words = tokenize(text);
    let syllables: usize = words.iter().map(|w| count_syllables(w)).sum();
    let n_sent = sentences.len().max(1) as f64;
    let n_words = words.len().max(1) as f64;
    0.39 * (n_words / n_sent) + 11.8 * (syllables as f64 / n_words) - 15.59
}

fn count_syllables(word: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let chars: Vec<char> = word.to_lowercase().chars().collect();
    let mut count = 0;
    let mut prev_vowel = false;
    for &c in &chars {
        let is_vowel = vowels.contains(&c);
        if is_vowel && !prev_vowel { count += 1; }
        prev_vowel = is_vowel;
    }
    // Silent e
    if chars.last() == Some(&'e') && count > 1 { count -= 1; }
    count.max(1)
}

// ═══════════════════════════════════════════════════════════════════
// HASHING (for feature hashing / hashing trick)
// ═══════════════════════════════════════════════════════════════════

/// Feature hashing (hashing trick). Maps tokens to fixed-size vector.
pub fn feature_hash(tokens: &[String], n_features: usize) -> Vec<f64> {
    let mut features = vec![0.0; n_features];
    for token in tokens {
        let hash = simple_hash(token);
        let idx = hash as usize % n_features;
        let sign = if (hash >> 31) & 1 == 0 { 1.0 } else { -1.0 };
        features[idx] += sign;
    }
    features
}

fn simple_hash(s: &str) -> u64 {
    let mut hash = 5381u64;
    for b in s.bytes() { hash = hash.wrapping_mul(33).wrapping_add(b as u64); }
    hash
}

/// SimHash for near-duplicate detection (returns 64-bit hash).
pub fn simhash(tokens: &[String]) -> u64 {
    let mut v = [0i64; 64];
    for token in tokens {
        let hash = simple_hash(token);
        for i in 0..64 {
            if (hash >> i) & 1 == 1 { v[i] += 1; }
            else { v[i] -= 1; }
        }
    }
    let mut result = 0u64;
    for i in 0..64 {
        if v[i] > 0 { result |= 1 << i; }
    }
    result
}

/// Hamming distance between two SimHash values.
pub fn simhash_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_ngrams() {
        let tokens: Vec<String> = vec!["a", "b", "c", "d"].into_iter().map(String::from).collect();
        let bigrams = ngrams(&tokens, 2);
        assert_eq!(bigrams.len(), 3);
    }

    #[test]
    fn test_tfidf() {
        let doc1 = tokenize("the cat sat on the mat");
        let doc2 = tokenize("the dog sat on the log");
        let corpus = vec![doc1.clone(), doc2];
        let scores = tfidf(&doc1, &corpus);
        // "cat" is unique to doc1, so its IDF should be higher than shared words
        assert!(scores.get("cat").unwrap_or(&0.0) > &0.0);
        // "cat" IDF > "sat" IDF (sat appears in both docs)
        let cat_idf = scores.get("cat").unwrap_or(&0.0) / (1.0 / 6.0);
        let sat_idf = scores.get("sat").unwrap_or(&0.0) / (1.0 / 6.0);
        assert!(cat_idf > sat_idf);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_lcs() {
        assert_eq!(lcs_length("ABCBDAB", "BDCAB"), 4);
    }

    #[test]
    fn test_word_count() {
        assert_eq!(word_count("hello world foo"), 3);
    }

    #[test]
    fn test_type_token_ratio() {
        let tokens: Vec<String> = vec!["a", "b", "a", "c"].into_iter().map(String::from).collect();
        assert!((type_token_ratio(&tokens) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_feature_hash() {
        let tokens: Vec<String> = vec!["hello", "world"].into_iter().map(String::from).collect();
        let fh = feature_hash(&tokens, 10);
        assert_eq!(fh.len(), 10);
    }

    #[test]
    fn test_simhash() {
        let a = simhash(&tokenize("the quick brown fox"));
        let b = simhash(&tokenize("the quick brown dog"));
        // Similar texts should have small Hamming distance
        assert!(simhash_distance(a, b) < 32);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let b: Vec<String> = vec!["b", "c", "d"].into_iter().map(String::from).collect();
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stem() {
        assert_eq!(stem("running"), "runn");
        assert_eq!(stem("happiness"), "happi");
    }

    #[test]
    fn test_remove_stopwords() {
        let tokens = tokenize("the cat is on the mat");
        let filtered = remove_stopwords(&tokens);
        assert!(filtered.contains(&"cat".to_string()));
        assert!(!filtered.contains(&"the".to_string()));
    }
}
