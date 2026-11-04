//! Cross-validation and model selection: k-fold, stratified k-fold,
//! leave-one-out, train/test split, grid search, scoring metrics.

// ═══════════════════════════════════════════════════════════════════
// DATA SPLITTING
// ═══════════════════════════════════════════════════════════════════

/// Train/test split indices. Returns (train_indices, test_indices).
pub fn train_test_split(n: usize, test_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);
    let test_size = (n as f64 * test_fraction).round() as usize;
    let test = indices[..test_size].to_vec();
    let train = indices[test_size..].to_vec();
    (train, test)
}

/// K-fold cross-validation indices. Returns vec of (train, test) index pairs.
pub fn kfold(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);
    let fold_size = n / k;
    (0..k).map(|i| {
        let test_start = i * fold_size;
        let test_end = if i == k - 1 { n } else { (i + 1) * fold_size };
        let test: Vec<usize> = indices[test_start..test_end].to_vec();
        let train: Vec<usize> = indices[..test_start].iter().chain(&indices[test_end..]).copied().collect();
        (train, test)
    }).collect()
}

/// Stratified k-fold: preserves class distribution in each fold.
pub fn stratified_kfold(labels: &[usize], k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n_classes = *labels.iter().max().unwrap_or(&0) + 1;
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &l) in labels.iter().enumerate() { class_indices[l].push(i); }
    for ci in class_indices.iter_mut() { shuffle(ci, seed); }

    (0..k).map(|fold| {
        let mut test = Vec::new();
        let mut train = Vec::new();
        for ci in &class_indices {
            let fold_size = ci.len() / k;
            let start = fold * fold_size;
            let end = if fold == k - 1 { ci.len() } else { (fold + 1) * fold_size };
            test.extend_from_slice(&ci[start..end]);
            train.extend_from_slice(&ci[..start]);
            train.extend_from_slice(&ci[end..]);
        }
        (train, test)
    }).collect()
}

/// Leave-one-out cross-validation indices.
pub fn leave_one_out(n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..n).map(|i| {
        let train: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        (train, vec![i])
    }).collect()
}

/// Repeated k-fold: run k-fold n_repeats times with different seeds.
pub fn repeated_kfold(n: usize, k: usize, n_repeats: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut all_folds = Vec::new();
    for r in 0..n_repeats {
        all_folds.extend(kfold(n, k, seed.wrapping_add(r as u64)));
    }
    all_folds
}

/// Time series split (expanding window). No shuffling.
pub fn time_series_split(n: usize, n_splits: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let min_train = n / (n_splits + 1);
    (0..n_splits).map(|i| {
        let test_start = min_train + i * min_train;
        let test_end = (test_start + min_train).min(n);
        let train: Vec<usize> = (0..test_start).collect();
        let test: Vec<usize> = (test_start..test_end).collect();
        (train, test)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// SCORING METRICS (classification)
// ═══════════════════════════════════════════════════════════════════

/// Accuracy: fraction of correct predictions.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
    correct as f64 / y_true.len() as f64
}

/// Precision for binary classification (class 1 = positive).
pub fn precision(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let tp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 1 && **p == 1).count() as f64;
    let fp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 0 && **p == 1).count() as f64;
    if tp + fp < 1e-15 { return 0.0; }
    tp / (tp + fp)
}

/// Recall for binary classification.
pub fn recall(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let tp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 1 && **p == 1).count() as f64;
    let fn_ = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 1 && **p == 0).count() as f64;
    if tp + fn_ < 1e-15 { return 0.0; }
    tp / (tp + fn_)
}

/// F1 score: harmonic mean of precision and recall.
pub fn f1_score(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let p = precision(y_true, y_pred);
    let r = recall(y_true, y_pred);
    if p + r < 1e-15 { return 0.0; }
    2.0 * p * r / (p + r)
}

/// F-beta score.
pub fn fbeta_score(y_true: &[usize], y_pred: &[usize], beta: f64) -> f64 {
    let p = precision(y_true, y_pred);
    let r = recall(y_true, y_pred);
    let b2 = beta * beta;
    if (1.0 + b2) * p * r < 1e-15 { return 0.0; }
    (1.0 + b2) * p * r / (b2 * p + r)
}

/// Confusion matrix for n classes. Returns n×n matrix (row=true, col=pred).
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> Vec<usize> {
    let mut cm = vec![0usize; n_classes * n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t < n_classes && p < n_classes { cm[t * n_classes + p] += 1; }
    }
    cm
}

/// Matthews correlation coefficient (MCC).
pub fn mcc(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let tp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 1 && **p == 1).count() as f64;
    let tn = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 0 && **p == 0).count() as f64;
    let fp = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 0 && **p == 1).count() as f64;
    let fn_ = y_true.iter().zip(y_pred).filter(|(t, p)| **t == 1 && **p == 0).count() as f64;
    let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if denom < 1e-15 { return 0.0; }
    (tp * tn - fp * fn_) / denom
}

/// Cohen's kappa.
pub fn cohen_kappa(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let n = y_true.len() as f64;
    let acc = accuracy(y_true, y_pred);
    let n_classes = *y_true.iter().max().unwrap_or(&0) + 1;
    let mut pe = 0.0;
    for c in 0..n_classes {
        let true_c = y_true.iter().filter(|&&t| t == c).count() as f64 / n;
        let pred_c = y_pred.iter().filter(|&&p| p == c).count() as f64 / n;
        pe += true_c * pred_c;
    }
    if (1.0 - pe).abs() < 1e-15 { return 1.0; }
    (acc - pe) / (1.0 - pe)
}

// ═══════════════════════════════════════════════════════════════════
// SCORING METRICS (regression)
// ═══════════════════════════════════════════════════════════════════

/// Mean Squared Error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).sum::<f64>() / y_true.len() as f64
}

/// Root Mean Squared Error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 { mse(y_true, y_pred).sqrt() }

/// Mean Absolute Error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred).map(|(t, p)| (t - p).abs()).sum::<f64>() / y_true.len() as f64
}

/// R² (coefficient of determination).
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_res: f64 = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    if ss_tot < 1e-15 { return 0.0; }
    1.0 - ss_res / ss_tot
}

/// Adjusted R².
pub fn adjusted_r_squared(y_true: &[f64], y_pred: &[f64], n_features: usize) -> f64 {
    let n = y_true.len() as f64;
    let r2 = r_squared(y_true, y_pred);
    1.0 - (1.0 - r2) * (n - 1.0) / (n - n_features as f64 - 1.0)
}

/// Mean Absolute Percentage Error.
pub fn mape(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    y_true.iter().zip(y_pred).map(|(t, p)| {
        if t.abs() > 1e-15 { ((t - p) / t).abs() } else { 0.0 }
    }).sum::<f64>() / n * 100.0
}

/// Explained variance score.
pub fn explained_variance(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let residuals: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();
    let var_y: f64 = { let m = y_true.iter().sum::<f64>() / n; y_true.iter().map(|t| (t - m).powi(2)).sum::<f64>() / n };
    let var_r: f64 = { let m = residuals.iter().sum::<f64>() / n; residuals.iter().map(|r| (r - m).powi(2)).sum::<f64>() / n };
    if var_y < 1e-15 { return 0.0; }
    1.0 - var_r / var_y
}

/// Max error.
pub fn max_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred).map(|(t, p)| (t - p).abs()).fold(0.0, f64::max)
}

/// Median absolute error.
pub fn median_ae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mut errors: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).abs()).collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = errors.len();
    if n % 2 == 0 { (errors[n/2 - 1] + errors[n/2]) / 2.0 } else { errors[n/2] }
}

// ═══════════════════════════════════════════════════════════════════
// GRID/RANDOM SEARCH HELPERS
// ═══════════════════════════════════════════════════════════════════

/// Generate parameter grid from ranges. Each param: (min, max, n_steps).
pub fn param_grid(ranges: &[(f64, f64, usize)]) -> Vec<Vec<f64>> {
    let grids: Vec<Vec<f64>> = ranges.iter().map(|&(lo, hi, n)| {
        if n <= 1 { vec![lo] }
        else { (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect() }
    }).collect();

    let mut combos = vec![Vec::new()];
    for grid in &grids {
        let mut new_combos = Vec::new();
        for combo in &combos {
            for &val in grid {
                let mut c = combo.clone();
                c.push(val);
                new_combos.push(c);
            }
        }
        combos = new_combos;
    }
    combos
}

/// Random parameter search. Returns n_iter random parameter combinations.
pub fn random_param_search(ranges: &[(f64, f64)], n_iter: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed_state(seed);
    (0..n_iter).map(|_| {
        ranges.iter().map(|&(lo, hi)| lo + prng_f64(&mut state) * (hi - lo)).collect()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

fn shuffle<T>(data: &mut [T], seed: u64) {
    let mut state = seed_state(seed);
    let n = data.len();
    for i in (1..n).rev() {
        let j = prng_next(&mut state) as usize % (i + 1);
        data.swap(i, j);
    }
}

fn prng_next(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = state[3].rotate_left(45);
    result
}

fn prng_f64(state: &mut [u64; 4]) -> f64 {
    (prng_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn seed_state(seed: u64) -> [u64; 4] {
    let mut s = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                 seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                 seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut s); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let (train, test) = train_test_split(100, 0.2, 42);
        assert_eq!(train.len() + test.len(), 100);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_kfold() {
        let folds = kfold(100, 5, 42);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_stratified_kfold() {
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let folds = stratified_kfold(&labels, 2, 42);
        assert_eq!(folds.len(), 2);
    }

    #[test]
    fn test_loo() {
        let folds = leave_one_out(5);
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[0].0.len(), 4);
        assert_eq!(folds[0].1.len(), 1);
    }

    #[test]
    fn test_time_series_split() {
        let splits = time_series_split(100, 5);
        assert_eq!(splits.len(), 5);
        // Train should be expanding
        assert!(splits[0].0.len() < splits[4].0.len());
    }

    #[test]
    fn test_accuracy() {
        assert!((accuracy(&[0, 1, 1, 0], &[0, 1, 0, 0]) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall() {
        let t = vec![1, 1, 0, 0, 1, 0];
        let p = vec![1, 0, 0, 1, 1, 0];
        let prec = precision(&t, &p);
        let rec = recall(&t, &p);
        assert!(prec > 0.0 && prec <= 1.0);
        assert!(rec > 0.0 && rec <= 1.0);
    }

    #[test]
    fn test_f1() {
        let t = vec![1, 1, 1, 0, 0, 0];
        let p = vec![1, 1, 1, 0, 0, 0];
        assert!((f1_score(&t, &p) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let t = vec![0, 0, 1, 1];
        let p = vec![0, 1, 0, 1];
        let cm = confusion_matrix(&t, &p, 2);
        assert_eq!(cm, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_mse_rmse() {
        let t = vec![1.0, 2.0, 3.0];
        let p = vec![1.0, 2.0, 3.0];
        assert!(mse(&t, &p).abs() < 1e-10);
        assert!(rmse(&t, &p).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared() {
        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((r_squared(&t, &p) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_param_grid() {
        let grid = param_grid(&[(0.0, 1.0, 3), (10.0, 20.0, 2)]);
        assert_eq!(grid.len(), 6); // 3 × 2
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        let t = vec![0, 1, 2, 0, 1, 2];
        let p = vec![0, 1, 2, 0, 1, 2];
        assert!((cohen_kappa(&t, &p) - 1.0).abs() < 1e-10);
    }
}
