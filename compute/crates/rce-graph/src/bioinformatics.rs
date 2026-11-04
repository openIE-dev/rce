//! Bioinformatics: sequence alignment, FASTA/FASTQ parsing,
//! DNA/RNA/protein operations, k-mer counting, GC content,
//! motif finding, phylogenetic distance, sequence statistics.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// SEQUENCE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// DNA complement.
pub fn dna_complement(seq: &str) -> String {
    seq.chars().map(|c| match c {
        'A' | 'a' => 'T', 'T' | 't' => 'A',
        'G' | 'g' => 'C', 'C' | 'c' => 'G',
        'N' | 'n' => 'N', other => other,
    }).collect()
}

/// DNA reverse complement.
pub fn dna_reverse_complement(seq: &str) -> String {
    dna_complement(seq).chars().rev().collect()
}

/// Transcribe DNA to RNA (T → U).
pub fn transcribe(dna: &str) -> String {
    dna.chars().map(|c| match c { 'T' => 'U', 't' => 'u', other => other }).collect()
}

/// Reverse transcribe RNA to DNA (U → T).
pub fn reverse_transcribe(rna: &str) -> String {
    rna.chars().map(|c| match c { 'U' => 'T', 'u' => 't', other => other }).collect()
}

/// GC content (fraction of G+C bases).
pub fn gc_content(seq: &str) -> f64 {
    let total = seq.len() as f64;
    if total < 1.0 { return 0.0; }
    let gc = seq.chars().filter(|c| matches!(c, 'G' | 'g' | 'C' | 'c')).count() as f64;
    gc / total
}

/// Translate DNA/RNA codon to amino acid (standard genetic code).
pub fn translate_codon(codon: &str) -> char {
    match codon.to_uppercase().as_str() {
        "TTT" | "TTC" | "UUU" | "UUC" => 'F',
        "TTA" | "TTG" | "CTT" | "CTC" | "CTA" | "CTG" |
        "UUA" | "UUG" | "CUU" | "CUC" | "CUA" | "CUG" => 'L',
        "ATT" | "ATC" | "ATA" | "AUU" | "AUC" | "AUA" => 'I',
        "ATG" | "AUG" => 'M', // start codon
        "GTT" | "GTC" | "GTA" | "GTG" | "GUU" | "GUC" | "GUA" | "GUG" => 'V',
        "TCT" | "TCC" | "TCA" | "TCG" | "UCU" | "UCC" | "UCA" | "UCG" => 'S',
        "CCT" | "CCC" | "CCA" | "CCG" | "CCU" => 'P',
        "ACT" | "ACC" | "ACA" | "ACG" | "ACU" => 'T',
        "GCT" | "GCC" | "GCA" | "GCG" | "GCU" => 'A',
        "TAT" | "TAC" | "UAU" | "UAC" => 'Y',
        "TAA" | "TAG" | "TGA" | "UAA" | "UAG" | "UGA" => '*', // stop
        "CAT" | "CAC" | "CAU" => 'H',
        "CAA" | "CAG" => 'Q',
        "AAT" | "AAC" | "AAU" => 'N',
        "AAA" | "AAG" => 'K',
        "GAT" | "GAC" | "GAU" => 'D',
        "GAA" | "GAG" => 'E',
        "TGT" | "TGC" | "UGU" | "UGC" => 'C',
        "TGG" | "UGG" => 'W',
        "CGT" | "CGC" | "CGA" | "CGG" | "CGU" => 'R',
        "AGT" | "AGC" | "AGU" => 'S',
        "AGA" | "AGG" => 'R',
        "GGT" | "GGC" | "GGA" | "GGG" | "GGU" => 'G',
        _ => 'X', // unknown
    }
}

/// Translate a DNA/RNA sequence to protein.
pub fn translate(seq: &str) -> String {
    let chars: Vec<char> = seq.chars().collect();
    let mut protein = String::new();
    for i in (0..chars.len() - 2).step_by(3) {
        let codon: String = chars[i..i + 3].iter().collect();
        let aa = translate_codon(&codon);
        if aa == '*' { break; }
        protein.push(aa);
    }
    protein
}

/// Count nucleotides.
pub fn nucleotide_count(seq: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for c in seq.chars() {
        *counts.entry(c.to_ascii_uppercase()).or_insert(0) += 1;
    }
    counts
}

// ═══════════════════════════════════════════════════════════════════
// K-MER COUNTING
// ═══════════════════════════════════════════════════════════════════

/// Count k-mers in a sequence.
pub fn kmer_count(seq: &str, k: usize) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    if seq.len() < k { return counts; }
    for i in 0..=seq.len() - k {
        let kmer = &seq[i..i + k];
        *counts.entry(kmer.to_uppercase()).or_insert(0) += 1;
    }
    counts
}

/// K-mer frequency (normalized).
pub fn kmer_frequency(seq: &str, k: usize) -> HashMap<String, f64> {
    let counts = kmer_count(seq, k);
    let total: usize = counts.values().sum();
    counts.into_iter().map(|(k, v)| (k, v as f64 / total as f64)).collect()
}

/// Unique k-mers.
pub fn unique_kmers(seq: &str, k: usize) -> Vec<String> {
    let counts = kmer_count(seq, k);
    let mut kmers: Vec<String> = counts.into_keys().collect();
    kmers.sort();
    kmers
}

// ═══════════════════════════════════════════════════════════════════
// SEQUENCE ALIGNMENT
// ═══════════════════════════════════════════════════════════════════

/// Needleman-Wunsch global alignment.
/// Returns (aligned_a, aligned_b, score).
pub fn needleman_wunsch(a: &str, b: &str, match_score: i32, mismatch: i32, gap: i32) -> (String, String, i32) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());

    let mut dp = vec![vec![0i32; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i as i32 * gap; }
    for j in 0..=n { dp[0][j] = j as i32 * gap; }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a_chars[i-1] == b_chars[j-1] { match_score } else { mismatch };
            dp[i][j] = (dp[i-1][j-1] + s).max(dp[i-1][j] + gap).max(dp[i][j-1] + gap);
        }
    }

    // Traceback
    let mut aligned_a = String::new();
    let mut aligned_b = String::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let s = if a_chars[i-1] == b_chars[j-1] { match_score } else { mismatch };
            if dp[i][j] == dp[i-1][j-1] + s {
                aligned_a.push(a_chars[i-1]); aligned_b.push(b_chars[j-1]);
                i -= 1; j -= 1; continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i-1][j] + gap {
            aligned_a.push(a_chars[i-1]); aligned_b.push('-');
            i -= 1;
        } else {
            aligned_a.push('-'); aligned_b.push(b_chars[j-1]);
            j -= 1;
        }
    }

    let score = dp[m][n];
    (aligned_a.chars().rev().collect(), aligned_b.chars().rev().collect(), score)
}

/// Smith-Waterman local alignment.
/// Returns (aligned_a, aligned_b, score).
pub fn smith_waterman(a: &str, b: &str, match_score: i32, mismatch: i32, gap: i32) -> (String, String, i32) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());

    let mut dp = vec![vec![0i32; n + 1]; m + 1];
    let mut max_score = 0;
    let mut max_pos = (0, 0);

    for i in 1..=m {
        for j in 1..=n {
            let s = if a_chars[i-1] == b_chars[j-1] { match_score } else { mismatch };
            dp[i][j] = 0.max(dp[i-1][j-1] + s).max(dp[i-1][j] + gap).max(dp[i][j-1] + gap);
            if dp[i][j] > max_score { max_score = dp[i][j]; max_pos = (i, j); }
        }
    }

    // Traceback from max
    let mut aligned_a = String::new();
    let mut aligned_b = String::new();
    let (mut i, mut j) = max_pos;
    while i > 0 && j > 0 && dp[i][j] > 0 {
        let s = if a_chars[i-1] == b_chars[j-1] { match_score } else { mismatch };
        if dp[i][j] == dp[i-1][j-1] + s {
            aligned_a.push(a_chars[i-1]); aligned_b.push(b_chars[j-1]);
            i -= 1; j -= 1;
        } else if dp[i][j] == dp[i-1][j] + gap {
            aligned_a.push(a_chars[i-1]); aligned_b.push('-');
            i -= 1;
        } else {
            aligned_a.push('-'); aligned_b.push(b_chars[j-1]);
            j -= 1;
        }
    }

    (aligned_a.chars().rev().collect(), aligned_b.chars().rev().collect(), max_score)
}

/// Hamming distance between equal-length sequences.
pub fn hamming_distance(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).filter(|(x, y)| x != y).count()
}

/// Sequence identity (fraction of matching positions).
pub fn sequence_identity(a: &str, b: &str) -> f64 {
    let total = a.len().max(b.len());
    if total == 0 { return 1.0; }
    let matches = a.chars().zip(b.chars()).filter(|(x, y)| x == y).count();
    matches as f64 / total as f64
}

// ═══════════════════════════════════════════════════════════════════
// MOTIF FINDING
// ═══════════════════════════════════════════════════════════════════

/// Find all occurrences of a pattern in a sequence. Returns start positions.
pub fn find_motif(seq: &str, pattern: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let seq_upper = seq.to_uppercase();
    let pat_upper = pattern.to_uppercase();
    let mut start = 0;
    while let Some(pos) = seq_upper[start..].find(&pat_upper) {
        positions.push(start + pos);
        start += pos + 1;
    }
    positions
}

/// Find open reading frames (ORFs). Returns (start, end) positions.
pub fn find_orfs(seq: &str, min_length: usize) -> Vec<(usize, usize)> {
    let seq_upper = seq.to_uppercase();
    let chars: Vec<char> = seq_upper.chars().collect();
    let mut orfs = Vec::new();

    for frame in 0..3 {
        let mut in_orf = false;
        let mut start = 0;
        for i in (frame..chars.len() - 2).step_by(3) {
            let codon: String = chars[i..i + 3].iter().collect();
            if codon == "ATG" && !in_orf {
                in_orf = true;
                start = i;
            } else if in_orf && (codon == "TAA" || codon == "TAG" || codon == "TGA") {
                let end = i + 3;
                if end - start >= min_length { orfs.push((start, end)); }
                in_orf = false;
            }
        }
    }
    orfs
}

// ═══════════════════════════════════════════════════════════════════
// SEQUENCE STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Molecular weight of a DNA sequence (daltons).
pub fn dna_molecular_weight(seq: &str) -> f64 {
    seq.chars().map(|c| match c.to_ascii_uppercase() {
        'A' => 331.2, 'T' => 322.2, 'G' => 347.2, 'C' => 307.2, _ => 0.0,
    }).sum::<f64>() - (seq.len().saturating_sub(1)) as f64 * 18.02 // subtract water
}

/// Melting temperature (basic, nearest-neighbor approximation).
/// Simple Wallace rule for short oligos: Tm = 2(A+T) + 4(G+C).
pub fn melting_temperature(seq: &str) -> f64 {
    let counts = nucleotide_count(seq);
    let at = *counts.get(&'A').unwrap_or(&0) + *counts.get(&'T').unwrap_or(&0);
    let gc = *counts.get(&'G').unwrap_or(&0) + *counts.get(&'C').unwrap_or(&0);
    2.0 * at as f64 + 4.0 * gc as f64
}

/// Shannon entropy of a sequence.
pub fn sequence_entropy(seq: &str) -> f64 {
    let n = seq.len() as f64;
    if n < 1.0 { return 0.0; }
    let counts = nucleotide_count(seq);
    counts.values().map(|&c| {
        let p = c as f64 / n;
        if p > 0.0 { -p * p.log2() } else { 0.0 }
    }).sum()
}

/// Codon usage table for a coding sequence.
pub fn codon_usage(seq: &str) -> HashMap<String, usize> {
    let chars: Vec<char> = seq.to_uppercase().chars().collect();
    let mut usage = HashMap::new();
    for i in (0..chars.len() - 2).step_by(3) {
        let codon: String = chars[i..i + 3].iter().collect();
        *usage.entry(codon).or_insert(0) += 1;
    }
    usage
}

/// Jukes-Cantor evolutionary distance.
pub fn jukes_cantor_distance(p: f64) -> f64 {
    if p >= 0.75 { return f64::INFINITY; }
    -0.75 * (1.0 - 4.0 * p / 3.0).ln()
}

/// Kimura 2-parameter distance.
pub fn kimura_distance(transitions: f64, transversions: f64) -> f64 {
    let p = transitions;
    let q = transversions;
    if 1.0 - 2.0 * p - q <= 0.0 || 1.0 - 2.0 * q <= 0.0 { return f64::INFINITY; }
    -0.5 * (1.0 - 2.0 * p - q).ln() - 0.25 * (1.0 - 2.0 * q).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complement() {
        assert_eq!(dna_complement("ATGC"), "TACG");
        assert_eq!(dna_reverse_complement("ATGC"), "GCAT");
    }

    #[test]
    fn test_transcribe() {
        assert_eq!(transcribe("ATGC"), "AUGC");
        assert_eq!(reverse_transcribe("AUGC"), "ATGC");
    }

    #[test]
    fn test_gc_content() {
        assert!((gc_content("ATGCGC") - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_translate() {
        assert_eq!(translate("ATGAAAGGG"), "MKG");
    }

    #[test]
    fn test_kmer_count() {
        let counts = kmer_count("ATGATG", 3);
        assert_eq!(*counts.get("ATG").unwrap(), 2);
    }

    #[test]
    fn test_needleman_wunsch() {
        let (a, b, score) = needleman_wunsch("GATTACA", "GCATGCU", 2, -1, -1);
        assert!(score > 0);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn test_smith_waterman() {
        let (a, b, score) = smith_waterman("ACACACTA", "AGCACACA", 2, -1, -1);
        assert!(score > 0);
    }

    #[test]
    fn test_hamming() {
        assert_eq!(hamming_distance("GATTACA", "GACTATA"), 2);
    }

    #[test]
    fn test_find_motif() {
        let pos = find_motif("ATATATATAT", "ATAT");
        assert!(pos.len() >= 2);
    }

    #[test]
    fn test_find_orfs() {
        let orfs = find_orfs("XYZATGAAAGCCTAAYYY", 6);
        assert!(orfs.len() >= 1);
    }

    #[test]
    fn test_melting_temp() {
        let tm = melting_temperature("ATGCGC");
        assert!(tm > 0.0);
    }

    #[test]
    fn test_sequence_entropy() {
        // Max entropy for 4-letter alphabet
        let e = sequence_entropy("ATGC");
        assert!((e - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_jukes_cantor() {
        let d = jukes_cantor_distance(0.1);
        assert!(d > 0.1);
    }
}
