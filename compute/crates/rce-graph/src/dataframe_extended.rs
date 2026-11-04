//! Extended DataFrame operations: rolling windows, resampling, string accessor,
//! datetime operations, advanced groupby, and IO patterns.

use crate::dataframe::*;

// ═══════════════════════════════════════════════════════════════════
// ROLLING WINDOW ON DATAFRAME COLUMNS
// ═══════════════════════════════════════════════════════════════════

/// Apply rolling mean to a numeric column. Returns new column.
pub fn df_rolling_mean(col: &Column, window: usize) -> Column {
    let vals = col.as_f64();
    let result: Vec<f64> = crate::math_runtime::rolling_mean(&vals, window);
    Column::Float(result)
}

/// Apply rolling std to a numeric column.
pub fn df_rolling_std(col: &Column, window: usize) -> Column {
    let vals = col.as_f64();
    let result: Vec<f64> = crate::math_runtime::rolling_std(&vals, window);
    Column::Float(result)
}

/// Apply rolling sum.
pub fn df_rolling_sum(col: &Column, window: usize) -> Column {
    let vals = col.as_f64();
    let n = vals.len();
    let result: Vec<f64> = (0..n).map(|i| {
        if i + 1 < window { f64::NAN }
        else { vals[i + 1 - window..=i].iter().sum() }
    }).collect();
    Column::Float(result)
}

// ═══════════════════════════════════════════════════════════════════
// COLUMN OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Apply a function element-wise to a column.
pub fn col_map(col: &Column, f: impl Fn(f64) -> f64) -> Column {
    Column::Float(col.as_f64().iter().map(|&x| f(x)).collect())
}

/// Cumulative sum of a column.
pub fn col_cumsum(col: &Column) -> Column {
    let vals = col.as_f64();
    let mut acc = 0.0;
    Column::Float(vals.iter().map(|&x| { acc += x; acc }).collect())
}

/// Cumulative product of a column.
pub fn col_cumprod(col: &Column) -> Column {
    let vals = col.as_f64();
    let mut acc = 1.0;
    Column::Float(vals.iter().map(|&x| { acc *= x; acc }).collect())
}

/// Percent change of a column.
pub fn col_pct_change(col: &Column) -> Column {
    let vals = col.as_f64();
    Column::Float(crate::math_runtime::pct_change(&vals))
}

/// Shift column by n positions.
pub fn col_shift(col: &Column, n: i32) -> Column {
    let vals = col.as_f64();
    Column::Float(crate::math_runtime::lag(&vals, n))
}

/// Rank values in a column.
pub fn col_rank(col: &Column) -> Column {
    let vals = col.as_f64();
    Column::Float(crate::math_runtime::rankdata(&vals))
}

/// Clip values to a range.
pub fn col_clip(col: &Column, lo: f64, hi: f64) -> Column {
    Column::Float(col.as_f64().iter().map(|&x| x.clamp(lo, hi)).collect())
}

/// Fill NaN values with a constant.
pub fn col_fillna(col: &Column, value: f64) -> Column {
    Column::Float(col.as_f64().iter().map(|&x| if x.is_nan() { value } else { x }).collect())
}

/// Forward fill NaN values.
pub fn col_ffill(col: &Column) -> Column {
    let vals = col.as_f64();
    let mut last = f64::NAN;
    Column::Float(vals.iter().map(|&x| {
        if x.is_nan() { last } else { last = x; x }
    }).collect())
}

/// Backward fill NaN values.
pub fn col_bfill(col: &Column) -> Column {
    let mut vals = col.as_f64();
    let n = vals.len();
    let mut last = f64::NAN;
    for i in (0..n).rev() {
        if vals[i].is_nan() { vals[i] = last; }
        else { last = vals[i]; }
    }
    Column::Float(vals)
}

/// Interpolate NaN values linearly.
pub fn col_interpolate(col: &Column) -> Column {
    let mut vals = col.as_f64();
    let n = vals.len();
    let mut i = 0;
    while i < n {
        if vals[i].is_nan() {
            let start = i;
            while i < n && vals[i].is_nan() { i += 1; }
            let end = i;
            let v0 = if start > 0 { vals[start - 1] } else { 0.0 };
            let v1 = if end < n { vals[end] } else { v0 };
            let span = (end - start + 1) as f64;
            for j in start..end {
                vals[j] = v0 + (v1 - v0) * (j - start + 1) as f64 / span;
            }
        } else {
            i += 1;
        }
    }
    Column::Float(vals)
}

// ═══════════════════════════════════════════════════════════════════
// DATAFRAME ADVANCED OPS
// ═══════════════════════════════════════════════════════════════════

/// Drop columns by name.
pub fn df_drop_cols(df: &DataFrame, names: &[&str]) -> DataFrame {
    let columns = df.columns.iter()
        .filter(|(n, _)| !names.contains(&n.as_str()))
        .cloned()
        .collect();
    DataFrame { columns }
}

/// Rename a column.
pub fn df_rename_col(df: &DataFrame, old: &str, new: &str) -> DataFrame {
    let columns = df.columns.iter().map(|(n, c)| {
        if n == old { (new.to_string(), c.clone()) } else { (n.clone(), c.clone()) }
    }).collect();
    DataFrame { columns }
}

/// Get unique values from a column.
pub fn col_unique(col: &Column) -> Column {
    match col {
        Column::Float(v) => {
            let mut unique = Vec::new();
            for &x in v {
                if !unique.iter().any(|&u: &f64| (u - x).abs() < 1e-15 || (u.is_nan() && x.is_nan())) {
                    unique.push(x);
                }
            }
            Column::Float(unique)
        }
        Column::Int(v) => {
            let mut unique = v.clone();
            unique.sort(); unique.dedup();
            Column::Int(unique)
        }
        Column::Str(v) => {
            let mut unique = v.clone();
            unique.sort(); unique.dedup();
            Column::Str(unique)
        }
        Column::Bool(v) => {
            let mut unique = v.clone();
            unique.sort(); unique.dedup();
            Column::Bool(unique)
        }
    }
}

/// Count unique values in a column.
pub fn col_nunique(col: &Column) -> usize {
    match &col_unique(col) {
        Column::Float(v) => v.len(),
        Column::Int(v) => v.len(),
        Column::Str(v) => v.len(),
        Column::Bool(v) => v.len(),
    }
}

/// Value counts: returns (values, counts) sorted by count descending.
pub fn col_value_counts(col: &Column) -> (Vec<String>, Vec<usize>) {
    let strs: Vec<String> = match col {
        Column::Str(v) => v.clone(),
        Column::Int(v) => v.iter().map(|x| x.to_string()).collect(),
        Column::Float(v) => v.iter().map(|x| format!("{x}")).collect(),
        Column::Bool(v) => v.iter().map(|x| x.to_string()).collect(),
    };
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for s in &strs {
        *counts.entry(s.clone()).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let values: Vec<String> = pairs.iter().map(|(v, _)| v.clone()).collect();
    let cnts: Vec<usize> = pairs.iter().map(|(_, c)| *c).collect();
    (values, cnts)
}

/// Sample n rows randomly.
pub fn df_sample(df: &DataFrame, n: usize, seed: u64) -> DataFrame {
    let nrows = df.nrows();
    if n >= nrows { return df.clone(); }
    let mut rng = crate::distributions::Rng::new(seed);
    let mut indices: Vec<usize> = (0..nrows).collect();
    // Fisher-Yates shuffle
    for i in (1..nrows).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
    let selected: Vec<bool> = (0..nrows).map(|i| indices.iter().take(n).any(|&idx| idx == i)).collect();
    df.filter(&selected)
}

/// Concatenate two DataFrames vertically (append rows).
pub fn df_concat(a: &DataFrame, b: &DataFrame) -> DataFrame {
    let mut columns = Vec::new();
    for (name, col_a) in &a.columns {
        if let Some(col_b) = b.col(name) {
            let combined = match (col_a, col_b) {
                (Column::Float(va), Column::Float(vb)) => {
                    let mut v = va.clone(); v.extend(vb); Column::Float(v)
                }
                (Column::Int(va), Column::Int(vb)) => {
                    let mut v = va.clone(); v.extend(vb); Column::Int(v)
                }
                (Column::Str(va), Column::Str(vb)) => {
                    let mut v = va.clone(); v.extend(vb.iter().cloned()); Column::Str(v)
                }
                (Column::Bool(va), Column::Bool(vb)) => {
                    let mut v = va.clone(); v.extend(vb); Column::Bool(v)
                }
                _ => col_a.clone(), // type mismatch, keep first
            };
            columns.push((name.clone(), combined));
        }
    }
    DataFrame { columns }
}

/// Melt (unpivot): wide to long format.
pub fn df_melt(df: &DataFrame, id_vars: &[&str], value_vars: &[&str]) -> DataFrame {
    let nrows = df.nrows();
    let id_cols: Vec<(String, Vec<String>)> = id_vars.iter().map(|&name| {
        let col = df.col(name).unwrap();
        let strs: Vec<String> = match col {
            Column::Str(v) => v.clone(),
            Column::Int(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::Float(v) => v.iter().map(|x| format!("{x}")).collect(),
            Column::Bool(v) => v.iter().map(|x| x.to_string()).collect(),
        };
        (name.to_string(), strs)
    }).collect();

    let mut result_ids: Vec<Vec<String>> = vec![Vec::new(); id_vars.len()];
    let mut var_names = Vec::new();
    let mut values = Vec::new();

    for &var_name in value_vars {
        let col = match df.col(var_name) {
            Some(c) => c,
            None => continue,
        };
        let vals = col.as_f64();
        for i in 0..nrows {
            for (j, (_, id_col)) in id_cols.iter().enumerate() {
                result_ids[j].push(id_col[i].clone());
            }
            var_names.push(var_name.to_string());
            values.push(vals[i]);
        }
    }

    let mut columns: Vec<(String, Column)> = Vec::new();
    for (j, id_name) in id_vars.iter().enumerate() {
        columns.push((id_name.to_string(), Column::Str(result_ids[j].clone())));
    }
    columns.push(("variable".to_string(), Column::Str(var_names)));
    columns.push(("value".to_string(), Column::Float(values)));
    DataFrame { columns }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            ("x".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ("y".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ("g".into(), Column::Str(vec!["a".into(), "b".into(), "a".into(), "b".into(), "a".into()])),
        ])
    }

    #[test]
    fn test_rolling_mean() {
        let col = Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let rm = df_rolling_mean(&col, 3);
        let vals = rm.as_f64();
        assert!(vals[0].is_nan());
        assert!((vals[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_col_cumsum() {
        let col = Column::Float(vec![1.0, 2.0, 3.0, 4.0]);
        let cs = col_cumsum(&col);
        assert_eq!(cs.as_f64(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_col_ffill() {
        let col = Column::Float(vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN]);
        let filled = col_ffill(&col);
        let vals = filled.as_f64();
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
        assert!((vals[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_value_counts() {
        let col = Column::Str(vec!["a".into(), "b".into(), "a".into(), "c".into(), "a".into()]);
        let (vals, counts) = col_value_counts(&col);
        assert_eq!(vals[0], "a");
        assert_eq!(counts[0], 3);
    }

    #[test]
    fn test_df_drop_cols() {
        let df = sample_df();
        let dropped = df_drop_cols(&df, &["g"]);
        assert_eq!(dropped.ncols(), 2);
        assert!(dropped.col("g").is_none());
    }

    #[test]
    fn test_df_rename() {
        let df = sample_df();
        let renamed = df_rename_col(&df, "x", "new_x");
        assert!(renamed.col("new_x").is_some());
        assert!(renamed.col("x").is_none());
    }

    #[test]
    fn test_df_concat() {
        let a = DataFrame::from_columns(vec![("x".into(), Column::Float(vec![1.0, 2.0]))]);
        let b = DataFrame::from_columns(vec![("x".into(), Column::Float(vec![3.0, 4.0]))]);
        let c = df_concat(&a, &b);
        assert_eq!(c.nrows(), 4);
    }

    #[test]
    fn test_df_sample() {
        let df = sample_df();
        let sampled = df_sample(&df, 3, 42);
        assert_eq!(sampled.nrows(), 3);
    }

    #[test]
    fn test_col_interpolate() {
        let col = Column::Float(vec![1.0, f64::NAN, f64::NAN, 4.0]);
        let interp = col_interpolate(&col);
        let vals = interp.as_f64();
        assert!(!vals[1].is_nan());
        assert!(!vals[2].is_nan());
    }

    #[test]
    fn test_melt() {
        let df = DataFrame::from_columns(vec![
            ("id".into(), Column::Str(vec!["a".into(), "b".into()])),
            ("x".into(), Column::Float(vec![1.0, 2.0])),
            ("y".into(), Column::Float(vec![3.0, 4.0])),
        ]);
        let melted = df_melt(&df, &["id"], &["x", "y"]);
        assert_eq!(melted.nrows(), 4); // 2 rows * 2 value vars
        assert!(melted.col("variable").is_some());
        assert!(melted.col("value").is_some());
    }
}
