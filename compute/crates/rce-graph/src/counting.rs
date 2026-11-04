//! Counting and enumeration: frequency tables, contingency tables,
//! cross tabulation, histogram binning, empirical CDF, rank statistics.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// FREQUENCY TABLES
// ═══════════════════════════════════════════════════════════════════

/// Frequency table from categorical labels. Returns sorted (value, count, proportion).
pub fn freq_table_usize(data: &[usize]) -> Vec<(usize, usize, f64)> {
    let n = data.len() as f64;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &v in data { *counts.entry(v).or_insert(0) += 1; }
    let mut table: Vec<(usize, usize, f64)> = counts.into_iter()
        .map(|(v, c)| (v, c, c as f64 / n)).collect();
    table.sort_by_key(|t| t.0);
    table
}

/// Cumulative frequency table.
pub fn cum_freq_table(data: &[usize]) -> Vec<(usize, usize, f64)> {
    let ft = freq_table_usize(data);
    let n = data.len() as f64;
    let mut cum = 0;
    ft.iter().map(|&(v, c, _)| { cum += c; (v, cum, cum as f64 / n) }).collect()
}

/// Two-way contingency table (cross tabulation).
/// Returns (table as flat vec, row_labels, col_labels).
pub fn contingency_table(rows: &[usize], cols: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut row_labels: Vec<usize> = rows.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
    let mut col_labels: Vec<usize> = cols.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
    row_labels.sort(); col_labels.sort();
    let nr = row_labels.len();
    let nc = col_labels.len();
    let row_idx: HashMap<usize, usize> = row_labels.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let col_idx: HashMap<usize, usize> = col_labels.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut table = vec![0usize; nr * nc];
    for (&r, &c) in rows.iter().zip(cols) {
        table[row_idx[&r] * nc + col_idx[&c]] += 1;
    }
    (table, row_labels, col_labels)
}

/// Row totals of contingency table.
pub fn row_totals(table: &[usize], n_rows: usize, n_cols: usize) -> Vec<usize> {
    (0..n_rows).map(|i| table[i*n_cols..(i+1)*n_cols].iter().sum()).collect()
}

/// Column totals of contingency table.
pub fn col_totals(table: &[usize], n_rows: usize, n_cols: usize) -> Vec<usize> {
    (0..n_cols).map(|j| (0..n_rows).map(|i| table[i*n_cols+j]).sum()).collect()
}

/// Row percentages of contingency table.
pub fn row_percentages(table: &[usize], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let rt = row_totals(table, n_rows, n_cols);
    let mut result = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            result.push(if rt[i] > 0 { table[i*n_cols+j] as f64 / rt[i] as f64 * 100.0 } else { 0.0 });
        }
    }
    result
}

/// Column percentages.
pub fn col_percentages(table: &[usize], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let ct = col_totals(table, n_rows, n_cols);
    let mut result = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            result.push(if ct[j] > 0 { table[i*n_cols+j] as f64 / ct[j] as f64 * 100.0 } else { 0.0 });
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// HISTOGRAM BINNING
// ═══════════════════════════════════════════════════════════════════

/// Histogram with n_bins equal-width bins. Returns (bin_edges, counts).
pub fn histogram(data: &[f64], n_bins: usize) -> (Vec<f64>, Vec<usize>) {
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = (max - min) / n_bins as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|i| min + i as f64 * width).collect();
    let mut counts = vec![0usize; n_bins];
    for &v in data {
        let bin = ((v - min) / width).floor() as usize;
        counts[bin.min(n_bins - 1)] += 1;
    }
    (edges, counts)
}

/// Sturges' rule for number of bins: k = ceil(log2(n) + 1).
pub fn sturges_bins(n: usize) -> usize {
    ((n as f64).log2() + 1.0).ceil() as usize
}

/// Scott's rule: bin width = 3.5σ / n^{1/3}.
pub fn scott_bin_width(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    3.5 * std / n.cbrt()
}

/// Freedman-Diaconis rule: bin width = 2·IQR / n^{1/3}.
pub fn fd_bin_width(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;
    2.0 * iqr / (n as f64).cbrt()
}

/// 2D histogram. Returns (x_edges, y_edges, counts as n_xbins × n_ybins).
pub fn histogram_2d(x: &[f64], y: &[f64], n_xbins: usize, n_ybins: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let xmin = x.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xw = (xmax - xmin) / n_xbins as f64;
    let yw = (ymax - ymin) / n_ybins as f64;
    let xedges: Vec<f64> = (0..=n_xbins).map(|i| xmin + i as f64 * xw).collect();
    let yedges: Vec<f64> = (0..=n_ybins).map(|i| ymin + i as f64 * yw).collect();
    let mut counts = vec![0usize; n_xbins * n_ybins];
    for (&xi, &yi) in x.iter().zip(y) {
        let xbin = ((xi - xmin) / xw).floor() as usize;
        let ybin = ((yi - ymin) / yw).floor() as usize;
        counts[xbin.min(n_xbins-1) * n_ybins + ybin.min(n_ybins-1)] += 1;
    }
    (xedges, yedges, counts)
}

// ═══════════════════════════════════════════════════════════════════
// EMPIRICAL CDF
// ═══════════════════════════════════════════════════════════════════

/// Empirical CDF. Returns sorted (value, cumulative_probability) pairs.
pub fn ecdf(data: &[f64]) -> Vec<(f64, f64)> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    sorted.iter().enumerate().map(|(i, &v)| (v, (i + 1) as f64 / n)).collect()
}

/// Evaluate ECDF at a specific point.
pub fn ecdf_at(data: &[f64], x: f64) -> f64 {
    let below = data.iter().filter(|&&v| v <= x).count();
    below as f64 / data.len() as f64
}

/// Empirical quantile function (inverse ECDF).
pub fn empirical_quantile(data: &[f64], p: f64) -> f64 {
    crate::probability::quantile(data, p)
}

// ═══════════════════════════════════════════════════════════════════
// RANK STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Assign ranks (average method for ties).
pub fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 { j += 1; }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j { ranks[indexed[k].0] = avg; }
        i = j;
    }
    ranks
}

/// Percentile rank of each element.
pub fn percentile_ranks(data: &[f64]) -> Vec<f64> {
    let ranks = rank(data);
    let n = data.len() as f64;
    ranks.iter().map(|&r| (r - 0.5) / n * 100.0).collect()
}

/// Dense rank (no gaps).
pub fn dense_rank(data: &[f64]) -> Vec<usize> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    data.iter().map(|v| sorted.iter().position(|s| (s - v).abs() < 1e-15).unwrap() + 1).collect()
}

/// Min rank (competition ranking: 1, 2, 2, 4, ...).
pub fn min_rank(data: &[f64]) -> Vec<usize> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 { j += 1; }
        for k in i..j { ranks[indexed[k].0] = i + 1; }
        i = j;
    }
    ranks
}

/// Rank-based outlier detection: values with rank outside [lo, hi] percentile.
pub fn rank_outliers(data: &[f64], lo_pct: f64, hi_pct: f64) -> Vec<usize> {
    let pr = percentile_ranks(data);
    (0..data.len()).filter(|&i| pr[i] < lo_pct || pr[i] > hi_pct).collect()
}

/// Kendall's W (coefficient of concordance) for k raters ranking n items.
/// rankings: k × n matrix (row-major).
pub fn kendall_w(rankings: &[f64], k: usize, n: usize) -> f64 {
    let mut col_sums = vec![0.0; n];
    for j in 0..n { for i in 0..k { col_sums[j] += rankings[i * n + j]; } }
    let mean = col_sums.iter().sum::<f64>() / n as f64;
    let s: f64 = col_sums.iter().map(|c| (c - mean).powi(2)).sum();
    12.0 * s / (k * k * (n * n * n - n)) as f64
}

// ═══════════════════════════════════════════════════════════════════
// RUNNING / ONLINE STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Welford's online algorithm for running mean and variance.
pub struct RunningStats {
    pub n: usize,
    pub mean: f64,
    m2: f64,
    pub min: f64,
    pub max: f64,
}

impl RunningStats {
    pub fn new() -> Self { Self { n: 0, mean: 0.0, m2: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY } }

    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }

    pub fn variance(&self) -> f64 { if self.n < 2 { 0.0 } else { self.m2 / (self.n - 1) as f64 } }
    pub fn std(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max - self.min }
}

/// Build running stats from data.
pub fn running_stats(data: &[f64]) -> RunningStats {
    let mut rs = RunningStats::new();
    for &v in data { rs.update(v); }
    rs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_table() {
        let data = vec![1, 2, 2, 3, 3, 3];
        let ft = freq_table_usize(&data);
        assert_eq!(ft.len(), 3);
        assert_eq!(ft[2].1, 3); // 3 appears 3 times
    }

    #[test]
    fn test_contingency() {
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let (table, rl, cl) = contingency_table(&rows, &cols);
        assert_eq!(table.len(), 4); // 2×2
    }

    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (edges, counts) = histogram(&data, 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<usize>(), 5);
    }

    #[test]
    fn test_ecdf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cdf = ecdf(&data);
        assert_eq!(cdf.last().unwrap().1, 1.0);
        assert!((ecdf_at(&data, 3.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_rank() {
        let r = rank(&[30.0, 10.0, 20.0]);
        assert_eq!(r, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank_ties() {
        let r = rank(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn test_dense_rank() {
        assert_eq!(dense_rank(&[30.0, 10.0, 20.0, 20.0]), vec![3, 1, 2, 2]);
    }

    #[test]
    fn test_running_stats() {
        let rs = running_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rs.mean - 3.0).abs() < 1e-10);
        assert_eq!(rs.min, 1.0);
        assert_eq!(rs.max, 5.0);
    }

    #[test]
    fn test_sturges() {
        assert_eq!(sturges_bins(100), 8); // ceil(log2(100) + 1) = 8
    }

    #[test]
    fn test_histogram_2d() {
        let x = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let (_, _, counts) = histogram_2d(&x, &y, 2, 2);
        assert_eq!(counts.iter().sum::<usize>(), 4);
    }
}
