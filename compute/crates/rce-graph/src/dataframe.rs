//! DataFrame: columnar data structure for tabular operations.
//!
//! Replaces Python pandas, R data.frame, Julia DataFrames.jl.
//! Pure Rust, no external dependencies. Operates on Vec<f64> columns
//! with string labels.
//!
//! Designed to integrate with the graph interpreter as Value::Record
//! of named columns.

use std::collections::HashMap;

// ── Column types ───────────────────────────────────────────────────

/// A single column of data.
#[derive(Debug, Clone)]
pub enum Column {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Str(Vec<String>),
    Bool(Vec<bool>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Int(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            Column::Float(v) => v.clone(),
            Column::Int(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Bool(v) => v.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            Column::Str(_) => vec![f64::NAN; self.len()],
        }
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Column::Float(_) => "float",
            Column::Int(_) => "int",
            Column::Str(_) => "str",
            Column::Bool(_) => "bool",
        }
    }
}

// ── DataFrame ──────────────────────────────────────────────────────

/// A columnar data frame with named columns and row indexing.
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<(String, Column)>,
}

impl DataFrame {
    pub fn new() -> Self {
        Self { columns: Vec::new() }
    }

    /// Create from column vectors.
    pub fn from_columns(cols: Vec<(String, Column)>) -> Self {
        Self { columns: cols }
    }

    /// Create from a CSV string.
    pub fn from_csv(csv: &str) -> Self {
        let mut lines = csv.lines();
        let header = match lines.next() {
            Some(h) => h,
            None => return Self::new(),
        };
        let names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
        let ncols = names.len();

        let mut float_cols: Vec<Vec<f64>> = vec![vec![]; ncols];
        let mut str_cols: Vec<Vec<String>> = vec![vec![]; ncols];
        let mut is_numeric: Vec<bool> = vec![true; ncols];

        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            for (j, field) in fields.iter().enumerate() {
                if j >= ncols { break; }
                let val = field.trim();
                if is_numeric[j] {
                    if let Ok(f) = val.parse::<f64>() {
                        float_cols[j].push(f);
                        str_cols[j].push(val.to_string());
                    } else {
                        is_numeric[j] = false;
                        str_cols[j].push(val.to_string());
                    }
                } else {
                    str_cols[j].push(val.to_string());
                }
            }
        }

        let columns: Vec<(String, Column)> = names.into_iter().enumerate().map(|(j, name)| {
            if is_numeric[j] {
                (name, Column::Float(float_cols[j].clone()))
            } else {
                (name, Column::Str(str_cols[j].clone()))
            }
        }).collect();

        Self { columns }
    }

    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    pub fn ncols(&self) -> usize { self.columns.len() }

    pub fn col_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Get a column by name.
    pub fn col(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Add a column.
    pub fn add_col(&mut self, name: &str, col: Column) {
        self.columns.push((name.to_string(), col));
    }

    /// Select specific columns.
    pub fn select(&self, names: &[&str]) -> Self {
        let columns = names.iter().filter_map(|&name| {
            self.columns.iter().find(|(n, _)| n == name).cloned()
        }).collect();
        Self { columns }
    }

    /// Filter rows by a boolean mask.
    pub fn filter(&self, mask: &[bool]) -> Self {
        let columns = self.columns.iter().map(|(name, col)| {
            let filtered = match col {
                Column::Float(v) => Column::Float(v.iter().zip(mask).filter(|&(_, &m)| m).map(|(&x, _)| x).collect()),
                Column::Int(v) => Column::Int(v.iter().zip(mask).filter(|&(_, &m)| m).map(|(&x, _)| x).collect()),
                Column::Str(v) => Column::Str(v.iter().zip(mask).filter(|&(_, &m)| m).map(|(x, _)| x.clone()).collect()),
                Column::Bool(v) => Column::Bool(v.iter().zip(mask).filter(|&(_, &m)| m).map(|(&x, _)| x).collect()),
            };
            (name.clone(), filtered)
        }).collect();
        Self { columns }
    }

    /// Sort by a column.
    pub fn sort_by(&self, col_name: &str, ascending: bool) -> Self {
        let col = match self.col(col_name) {
            Some(c) => c,
            None => return self.clone(),
        };
        let vals = col.as_f64();
        let mut indices: Vec<usize> = (0..vals.len()).collect();
        indices.sort_by(|&a, &b| {
            let cmp = vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal);
            if ascending { cmp } else { cmp.reverse() }
        });

        let columns = self.columns.iter().map(|(name, col)| {
            let reordered = match col {
                Column::Float(v) => Column::Float(indices.iter().map(|&i| v[i]).collect()),
                Column::Int(v) => Column::Int(indices.iter().map(|&i| v[i]).collect()),
                Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
                Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
            };
            (name.clone(), reordered)
        }).collect();
        Self { columns }
    }

    /// Group by a column and aggregate.
    pub fn groupby_agg(&self, group_col: &str, agg_col: &str, agg_fn: AggFn) -> Self {
        let group = match self.col(group_col) {
            Some(c) => c,
            None => return Self::new(),
        };
        let values = match self.col(agg_col) {
            Some(c) => c.as_f64(),
            None => return Self::new(),
        };

        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        let group_keys: Vec<String> = match group {
            Column::Str(v) => v.clone(),
            Column::Int(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::Float(v) => v.iter().map(|x| format!("{x}")).collect(),
            Column::Bool(v) => v.iter().map(|x| x.to_string()).collect(),
        };

        for (i, key) in group_keys.iter().enumerate() {
            if i < values.len() {
                groups.entry(key.clone()).or_default().push(values[i]);
            }
        }

        let mut keys: Vec<String> = groups.keys().cloned().collect();
        keys.sort();

        let agg_values: Vec<f64> = keys.iter().map(|k| {
            let vals = &groups[k];
            match agg_fn {
                AggFn::Sum => vals.iter().sum(),
                AggFn::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
                AggFn::Count => vals.len() as f64,
                AggFn::Min => vals.iter().copied().fold(f64::INFINITY, f64::min),
                AggFn::Max => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                AggFn::Std => {
                    let m = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var = vals.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (vals.len() as f64 - 1.0).max(1.0);
                    var.sqrt()
                }
            }
        }).collect();

        Self::from_columns(vec![
            (group_col.to_string(), Column::Str(keys)),
            (format!("{}_{:?}", agg_col, agg_fn).to_lowercase(), Column::Float(agg_values)),
        ])
    }

    /// Describe: summary statistics for numeric columns.
    pub fn describe(&self) -> Self {
        let stats = vec!["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        let mut result_cols: Vec<(String, Column)> = vec![
            ("stat".to_string(), Column::Str(stats.iter().map(|s| s.to_string()).collect())),
        ];

        for (name, col) in &self.columns {
            let vals = col.as_f64();
            let valid: Vec<f64> = vals.iter().copied().filter(|x| !x.is_nan()).collect();
            if valid.is_empty() { continue; }

            let n = valid.len() as f64;
            let mean = valid.iter().sum::<f64>() / n;
            let var = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            let std = var.sqrt();
            let mut sorted = valid.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = sorted[0];
            let max = sorted[sorted.len() - 1];
            let q25 = percentile_sorted(&sorted, 0.25);
            let q50 = percentile_sorted(&sorted, 0.50);
            let q75 = percentile_sorted(&sorted, 0.75);

            result_cols.push((name.clone(), Column::Float(vec![n, mean, std, min, q25, q50, q75, max])));
        }

        Self { columns: result_cols }
    }

    /// Inner join on a key column.
    pub fn merge(&self, other: &DataFrame, on: &str) -> Self {
        let left_key = match self.col(on) { Some(c) => c, None => return Self::new() };
        let right_key = match other.col(on) { Some(c) => c, None => return Self::new() };

        let left_keys: Vec<String> = col_to_strings(left_key);
        let right_keys: Vec<String> = col_to_strings(right_key);

        // Build right index
        let mut right_index: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, k) in right_keys.iter().enumerate() {
            right_index.entry(k.as_str()).or_default().push(i);
        }

        // Collect matching row pairs
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for (li, lk) in left_keys.iter().enumerate() {
            if let Some(ris) = right_index.get(lk.as_str()) {
                for &ri in ris {
                    left_indices.push(li);
                    right_indices.push(ri);
                }
            }
        }

        // Build result columns
        let mut result_cols: Vec<(String, Column)> = Vec::new();
        for (name, col) in &self.columns {
            result_cols.push((name.clone(), reindex_col(col, &left_indices)));
        }
        for (name, col) in &other.columns {
            if name == on { continue; } // skip join key (already in left)
            let suffix = if self.col(name).is_some() { "_right" } else { "" };
            result_cols.push((format!("{name}{suffix}"), reindex_col(col, &right_indices)));
        }

        Self { columns: result_cols }
    }

    /// Pivot table.
    pub fn pivot(&self, index_col: &str, columns_col: &str, values_col: &str) -> Self {
        let idx = match self.col(index_col) { Some(c) => col_to_strings(c), None => return Self::new() };
        let cols = match self.col(columns_col) { Some(c) => col_to_strings(c), None => return Self::new() };
        let vals = match self.col(values_col) { Some(c) => c.as_f64(), None => return Self::new() };

        let mut unique_idx: Vec<String> = idx.iter().cloned().collect();
        unique_idx.sort(); unique_idx.dedup();
        let mut unique_cols: Vec<String> = cols.iter().cloned().collect();
        unique_cols.sort(); unique_cols.dedup();

        let idx_map: HashMap<&str, usize> = unique_idx.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        let col_map: HashMap<&str, usize> = unique_cols.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

        let mut grid = vec![vec![f64::NAN; unique_cols.len()]; unique_idx.len()];
        for i in 0..idx.len().min(cols.len()).min(vals.len()) {
            if let (Some(&ri), Some(&ci)) = (idx_map.get(idx[i].as_str()), col_map.get(cols[i].as_str())) {
                grid[ri][ci] = vals[i];
            }
        }

        let mut result_cols: Vec<(String, Column)> = vec![
            (index_col.to_string(), Column::Str(unique_idx)),
        ];
        for (j, col_name) in unique_cols.iter().enumerate() {
            let col_vals: Vec<f64> = grid.iter().map(|row| row[j]).collect();
            result_cols.push((col_name.clone(), Column::Float(col_vals)));
        }

        Self { columns: result_cols }
    }

    /// Export to CSV string.
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        let names: Vec<&str> = self.columns.iter().map(|(n, _)| n.as_str()).collect();
        csv.push_str(&names.join(","));
        csv.push('\n');

        let nrows = self.nrows();
        for i in 0..nrows {
            for (j, (_, col)) in self.columns.iter().enumerate() {
                if j > 0 { csv.push(','); }
                match col {
                    Column::Float(v) => csv.push_str(&format!("{}", v.get(i).copied().unwrap_or(f64::NAN))),
                    Column::Int(v) => csv.push_str(&format!("{}", v.get(i).copied().unwrap_or(0))),
                    Column::Str(v) => csv.push_str(v.get(i).map(|s| s.as_str()).unwrap_or("")),
                    Column::Bool(v) => csv.push_str(&format!("{}", v.get(i).copied().unwrap_or(false))),
                }
            }
            csv.push('\n');
        }
        csv
    }

    /// Head: first n rows.
    pub fn head(&self, n: usize) -> Self {
        let mask: Vec<bool> = (0..self.nrows()).map(|i| i < n).collect();
        self.filter(&mask)
    }

    /// Tail: last n rows.
    pub fn tail(&self, n: usize) -> Self {
        let nrows = self.nrows();
        let start = nrows.saturating_sub(n);
        let mask: Vec<bool> = (0..nrows).map(|i| i >= start).collect();
        self.filter(&mask)
    }
}

// ── Aggregation functions ──────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum AggFn {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Std,
}

// ── Helpers ────────────────────────────────────────────────────────

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn col_to_strings(col: &Column) -> Vec<String> {
    match col {
        Column::Str(v) => v.clone(),
        Column::Int(v) => v.iter().map(|x| x.to_string()).collect(),
        Column::Float(v) => v.iter().map(|x| format!("{x}")).collect(),
        Column::Bool(v) => v.iter().map(|x| x.to_string()).collect(),
    }
}

fn reindex_col(col: &Column, indices: &[usize]) -> Column {
    match col {
        Column::Float(v) => Column::Float(indices.iter().map(|&i| v[i]).collect()),
        Column::Int(v) => Column::Int(indices.iter().map(|&i| v[i]).collect()),
        Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
        Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            ("name".into(), Column::Str(vec!["Alice".into(), "Bob".into(), "Charlie".into(), "Diana".into()])),
            ("age".into(), Column::Float(vec![30.0, 25.0, 35.0, 28.0])),
            ("score".into(), Column::Float(vec![85.0, 92.0, 78.0, 95.0])),
            ("dept".into(), Column::Str(vec!["eng".into(), "eng".into(), "sales".into(), "sales".into()])),
        ])
    }

    #[test]
    fn test_from_csv() {
        let csv = "x,y,label\n1,2,a\n3,4,b\n5,6,c\n";
        let df = DataFrame::from_csv(csv);
        assert_eq!(df.ncols(), 3);
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.col_names(), vec!["x", "y", "label"]);
    }

    #[test]
    fn test_select() {
        let df = sample_df();
        let sub = df.select(&["name", "score"]);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.nrows(), 4);
    }

    #[test]
    fn test_filter() {
        let df = sample_df();
        let scores = df.col("score").unwrap().as_f64();
        let mask: Vec<bool> = scores.iter().map(|&s| s >= 90.0).collect();
        let filtered = df.filter(&mask);
        assert_eq!(filtered.nrows(), 2); // Bob(92) and Diana(95)
    }

    #[test]
    fn test_sort() {
        let df = sample_df();
        let sorted = df.sort_by("age", true);
        let ages = sorted.col("age").unwrap().as_f64();
        assert!((ages[0] - 25.0).abs() < 0.01); // Bob first
        assert!((ages[3] - 35.0).abs() < 0.01); // Charlie last
    }

    #[test]
    fn test_groupby_mean() {
        let df = sample_df();
        let grouped = df.groupby_agg("dept", "score", AggFn::Mean);
        assert_eq!(grouped.nrows(), 2);
        let scores = grouped.col("score_mean").unwrap().as_f64();
        // eng: (85+92)/2 = 88.5, sales: (78+95)/2 = 86.5
        assert!((scores[0] - 88.5).abs() < 0.01 || (scores[0] - 86.5).abs() < 0.01);
    }

    #[test]
    fn test_groupby_count() {
        let df = sample_df();
        let grouped = df.groupby_agg("dept", "score", AggFn::Count);
        let counts = grouped.col("score_count").unwrap().as_f64();
        assert_eq!(counts.iter().sum::<f64>() as i64, 4); // total 4 rows
    }

    #[test]
    fn test_describe() {
        let df = sample_df();
        let desc = df.describe();
        assert!(desc.col("stat").is_some());
        assert!(desc.col("age").is_some());
        let age_stats = desc.col("age").unwrap().as_f64();
        assert!((age_stats[0] - 4.0).abs() < 0.01); // count
        assert!((age_stats[1] - 29.5).abs() < 0.01); // mean
    }

    #[test]
    fn test_merge() {
        let left = DataFrame::from_columns(vec![
            ("id".into(), Column::Int(vec![1, 2, 3])),
            ("val".into(), Column::Float(vec![10.0, 20.0, 30.0])),
        ]);
        let right = DataFrame::from_columns(vec![
            ("id".into(), Column::Int(vec![2, 3, 4])),
            ("label".into(), Column::Str(vec!["b".into(), "c".into(), "d".into()])),
        ]);
        let merged = left.merge(&right, "id");
        assert_eq!(merged.nrows(), 2); // ids 2 and 3 match
        assert!(merged.col("label").is_some());
    }

    #[test]
    fn test_pivot() {
        let df = DataFrame::from_columns(vec![
            ("year".into(), Column::Str(vec!["2023".into(), "2023".into(), "2024".into(), "2024".into()])),
            ("quarter".into(), Column::Str(vec!["Q1".into(), "Q2".into(), "Q1".into(), "Q2".into()])),
            ("revenue".into(), Column::Float(vec![100.0, 150.0, 120.0, 180.0])),
        ]);
        let pivoted = df.pivot("year", "quarter", "revenue");
        assert_eq!(pivoted.nrows(), 2);
        assert!(pivoted.col("Q1").is_some());
        assert!(pivoted.col("Q2").is_some());
    }

    #[test]
    fn test_to_csv() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float(vec![1.0, 2.0])),
            ("y".into(), Column::Float(vec![3.0, 4.0])),
        ]);
        let csv = df.to_csv();
        assert!(csv.starts_with("x,y\n"));
        assert!(csv.contains("1"));
        assert!(csv.contains("4"));
    }

    #[test]
    fn test_head_tail() {
        let df = sample_df();
        assert_eq!(df.head(2).nrows(), 2);
        assert_eq!(df.tail(1).nrows(), 1);
    }
}
