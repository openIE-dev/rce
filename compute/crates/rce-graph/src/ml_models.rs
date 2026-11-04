//! ML models: decision tree, logistic regression, naive Bayes, DBSCAN.
//!
//! Replaces scikit-learn's core estimators. Pure Rust, no dependencies.

use crate::math_runtime::{mean, variance};

// ═══════════════════════════════════════════════════════════════════
// LOGISTIC REGRESSION (binary, gradient descent)
// ═══════════════════════════════════════════════════════════════════

/// Logistic regression model.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LogisticRegression {
    /// Train via gradient descent. y must be 0.0 or 1.0.
    pub fn fit(x: &[Vec<f64>], y: &[f64], lr: f64, max_iter: usize) -> Self {
        let n = x.len();
        let dim = if n > 0 { x[0].len() } else { 0 };
        let mut w = vec![0.0; dim];
        let mut b = 0.0;

        for _ in 0..max_iter {
            let mut dw = vec![0.0; dim];
            let mut db = 0.0;
            for i in 0..n {
                let z: f64 = w.iter().zip(&x[i]).map(|(&wi, &xi)| wi * xi).sum::<f64>() + b;
                let pred = 1.0 / (1.0 + (-z).exp());
                let err = pred - y[i];
                for j in 0..dim { dw[j] += err * x[i][j]; }
                db += err;
            }
            for j in 0..dim { w[j] -= lr * dw[j] / n as f64; }
            b -= lr * db / n as f64;
        }
        Self { weights: w, bias: b }
    }

    /// Predict probabilities.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| {
            let z: f64 = self.weights.iter().zip(xi).map(|(&w, &x)| w * x).sum::<f64>() + self.bias;
            1.0 / (1.0 + (-z).exp())
        }).collect()
    }

    /// Predict class labels (threshold = 0.5).
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        self.predict_proba(x).iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// DECISION TREE (CART, classification + regression)
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum TreeNode {
    Leaf { value: f64 },
    Split { feature: usize, threshold: f64, left: Box<TreeNode>, right: Box<TreeNode> },
}

/// Decision tree regressor.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: TreeNode,
    pub max_depth: usize,
}

impl DecisionTree {
    /// Fit a regression tree.
    pub fn fit_regressor(x: &[Vec<f64>], y: &[f64], max_depth: usize, min_samples: usize) -> Self {
        let indices: Vec<usize> = (0..x.len()).collect();
        let root = build_tree(x, y, &indices, 0, max_depth, min_samples);
        Self { root, max_depth }
    }

    /// Predict values.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| predict_node(&self.root, xi)).collect()
    }

    /// Fit a classification tree (uses same structure, predicts majority class).
    pub fn fit_classifier(x: &[Vec<f64>], y: &[usize], max_depth: usize, min_samples: usize) -> Self {
        let y_f: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        Self::fit_regressor(x, &y_f, max_depth, min_samples)
    }

    /// Predict class labels (rounds predicted value).
    pub fn predict_class(&self, x: &[Vec<f64>]) -> Vec<usize> {
        self.predict(x).iter().map(|&v| v.round() as usize).collect()
    }
}

fn build_tree(x: &[Vec<f64>], y: &[f64], indices: &[usize], depth: usize, max_depth: usize, min_samples: usize) -> TreeNode {
    if indices.len() <= min_samples || depth >= max_depth {
        let avg = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len().max(1) as f64;
        return TreeNode::Leaf { value: avg };
    }

    let dim = if !x.is_empty() { x[0].len() } else { return TreeNode::Leaf { value: 0.0 }; };
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_score = f64::INFINITY;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    for feat in 0..dim {
        let mut vals: Vec<f64> = indices.iter().map(|&i| x[i][feat]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        for window in vals.windows(2) {
            let threshold = (window[0] + window[1]) / 2.0;
            let left: Vec<usize> = indices.iter().copied().filter(|&i| x[i][feat] <= threshold).collect();
            let right: Vec<usize> = indices.iter().copied().filter(|&i| x[i][feat] > threshold).collect();
            if left.is_empty() || right.is_empty() { continue; }
            let score = mse_impurity(y, &left) * left.len() as f64 + mse_impurity(y, &right) * right.len() as f64;
            if score < best_score {
                best_score = score;
                best_feature = feat;
                best_threshold = threshold;
                best_left = left;
                best_right = right;
            }
        }
    }

    if best_left.is_empty() || best_right.is_empty() {
        let avg = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
        return TreeNode::Leaf { value: avg };
    }

    TreeNode::Split {
        feature: best_feature,
        threshold: best_threshold,
        left: Box::new(build_tree(x, y, &best_left, depth + 1, max_depth, min_samples)),
        right: Box::new(build_tree(x, y, &best_right, depth + 1, max_depth, min_samples)),
    }
}

fn mse_impurity(y: &[f64], indices: &[usize]) -> f64 {
    let n = indices.len() as f64;
    if n < 1.0 { return 0.0; }
    let m: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
    indices.iter().map(|&i| (y[i] - m).powi(2)).sum::<f64>() / n
}

fn predict_node(node: &TreeNode, x: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf { value } => *value,
        TreeNode::Split { feature, threshold, left, right } => {
            if x[*feature] <= *threshold { predict_node(left, x) }
            else { predict_node(right, x) }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RANDOM FOREST (ensemble of decision trees)
// ═══════════════════════════════════════════════════════════════════

/// Random forest regressor.
#[derive(Debug, Clone)]
pub struct RandomForest {
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    /// Fit a random forest with bootstrap sampling.
    pub fn fit_regressor(x: &[Vec<f64>], y: &[f64], n_trees: usize, max_depth: usize, seed: u64) -> Self {
        let n = x.len();
        let mut rng = crate::distributions::Rng::new(seed);
        let trees: Vec<DecisionTree> = (0..n_trees).map(|_| {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n).map(|_| (rng.next_u64() as usize) % n).collect();
            let x_boot: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
            let y_boot: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            DecisionTree::fit_regressor(&x_boot, &y_boot, max_depth, 2)
        }).collect();
        Self { trees }
    }

    /// Predict by averaging tree predictions.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut preds = vec![0.0; n];
        for tree in &self.trees {
            let tp = tree.predict(x);
            for i in 0..n { preds[i] += tp[i]; }
        }
        let nt = self.trees.len() as f64;
        preds.iter_mut().for_each(|p| *p /= nt);
        preds
    }

    /// Fit classifier and predict class by majority vote.
    pub fn fit_classifier(x: &[Vec<f64>], y: &[usize], n_trees: usize, max_depth: usize, seed: u64) -> Self {
        let y_f: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        Self::fit_regressor(x, &y_f, n_trees, max_depth, seed)
    }

    pub fn predict_class(&self, x: &[Vec<f64>]) -> Vec<usize> {
        self.predict(x).iter().map(|&v| v.round() as usize).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// GAUSSIAN NAIVE BAYES
// ═══════════════════════════════════════════════════════════════════

/// Gaussian Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct GaussianNB {
    pub classes: Vec<usize>,
    pub means: Vec<Vec<f64>>,    // means[class][feature]
    pub vars: Vec<Vec<f64>>,     // vars[class][feature]
    pub priors: Vec<f64>,
}

impl GaussianNB {
    pub fn fit(x: &[Vec<f64>], y: &[usize]) -> Self {
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort(); classes.dedup();
        let dim = if !x.is_empty() { x[0].len() } else { 0 };
        let n = x.len() as f64;

        let mut means = Vec::new();
        let mut vars = Vec::new();
        let mut priors = Vec::new();

        for &c in &classes {
            let class_data: Vec<&Vec<f64>> = x.iter().zip(y).filter(|(_, yi)| **yi == c).map(|(xi, _)| xi).collect();
            let nc = class_data.len() as f64;
            priors.push(nc / n);
            let mut cm = vec![0.0; dim];
            let mut cv = vec![0.0; dim];
            for d in 0..dim {
                let vals: Vec<f64> = class_data.iter().map(|xi| xi[d]).collect();
                cm[d] = mean(&vals);
                cv[d] = variance(&vals).max(1e-9); // floor to avoid zero variance
            }
            means.push(cm);
            vars.push(cv);
        }
        Self { classes, means, vars, priors }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|xi| {
            let mut best_class = self.classes[0];
            let mut best_log_prob = f64::NEG_INFINITY;
            for (ci, &c) in self.classes.iter().enumerate() {
                let mut log_prob = self.priors[ci].ln();
                for d in 0..xi.len() {
                    let mu = self.means[ci][d];
                    let var = self.vars[ci][d];
                    log_prob += -0.5 * ((xi[d] - mu).powi(2) / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
                }
                if log_prob > best_log_prob { best_log_prob = log_prob; best_class = c; }
            }
            best_class
        }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// DBSCAN CLUSTERING
// ═══════════════════════════════════════════════════════════════════

/// DBSCAN: density-based clustering. Returns labels (-1 = noise).
pub fn dbscan(data: &[Vec<f64>], eps: f64, min_samples: usize) -> Vec<i32> {
    let n = data.len();
    let mut labels = vec![-1i32; n]; // -1 = unvisited/noise
    let mut cluster_id: i32 = 0;

    for i in 0..n {
        if labels[i] != -1 { continue; }
        let neighbors = range_query(data, i, eps);
        if neighbors.len() < min_samples { continue; } // noise (stays -1)

        labels[i] = cluster_id;
        let mut queue = neighbors;
        let mut qi = 0;
        while qi < queue.len() {
            let j = queue[qi];
            qi += 1;
            if labels[j] == -1 { labels[j] = cluster_id; } // was noise, now border
            else if labels[j] != -1 && labels[j] != cluster_id { continue; } // already assigned
            else { labels[j] = cluster_id; }

            let j_neighbors = range_query(data, j, eps);
            if j_neighbors.len() >= min_samples {
                for &k in &j_neighbors {
                    if labels[k] == -1 || labels[k] == -1 {
                        if !queue.contains(&k) { queue.push(k); }
                        labels[k] = cluster_id;
                    }
                }
            }
        }
        cluster_id += 1;
    }
    labels
}

fn range_query(data: &[Vec<f64>], idx: usize, eps: f64) -> Vec<usize> {
    let eps2 = eps * eps;
    (0..data.len()).filter(|&j| {
        data[idx].iter().zip(&data[j]).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>() <= eps2
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LINEAR DISCRIMINANT ANALYSIS (LDA)
// ═══════════════════════════════════════════════════════════════════

/// LDA: project data to lower dimension that maximizes class separation.
/// Returns projected data (n × n_components).
pub fn lda_transform(x: &[Vec<f64>], y: &[usize], n_components: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let dim = x[0].len();
    let mut classes: Vec<usize> = y.iter().copied().collect();
    classes.sort(); classes.dedup();

    // Overall mean
    let overall_mean: Vec<f64> = (0..dim).map(|d| x.iter().map(|xi| xi[d]).sum::<f64>() / n as f64).collect();

    // Between-class and within-class scatter
    let mut sb = vec![vec![0.0; dim]; dim];
    let mut sw = vec![vec![0.0; dim]; dim];

    for &c in &classes {
        let class_data: Vec<&Vec<f64>> = x.iter().zip(y).filter(|(_, yi)| **yi == c).map(|(xi, _)| xi).collect();
        let nc = class_data.len() as f64;
        let class_mean: Vec<f64> = (0..dim).map(|d| class_data.iter().map(|xi| xi[d]).sum::<f64>() / nc).collect();

        // Between-class
        for i in 0..dim {
            for j in 0..dim {
                sb[i][j] += nc * (class_mean[i] - overall_mean[i]) * (class_mean[j] - overall_mean[j]);
            }
        }
        // Within-class
        for xi in &class_data {
            for i in 0..dim {
                for j in 0..dim {
                    sw[i][j] += (xi[i] - class_mean[i]) * (xi[j] - class_mean[j]);
                }
            }
        }
    }

    // Solve Sw^-1 * Sb via eigendecomposition of Sw^-1 * Sb
    // Simplified: use inverse of Sw then multiply
    let sw_inv = match crate::math_runtime::inverse(&sw) {
        Some(inv) => inv,
        None => return x.to_vec(), // fallback
    };
    let m = crate::math_runtime::mat_mul(&sw_inv, &sb);
    let _eigs = crate::math_runtime::eigenvalues_symmetric(&m, 200);

    // Use top n_components eigenvectors (approximated via power iteration on m)
    let k = n_components.min(dim);
    let mut components = Vec::new();
    let mut deflated = m.clone();
    for _ in 0..k {
        let mut v: Vec<f64> = vec![1.0; dim];
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        for _ in 0..100 {
            let mut new_v = vec![0.0; dim];
            for i in 0..dim { for j in 0..dim { new_v[i] += deflated[i][j] * v[j]; } }
            let n2: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n2 < 1e-15 { break; }
            v = new_v.iter().map(|x| x / n2).collect();
        }
        let eigenvalue: f64 = v.iter().enumerate().map(|(i, &vi)| {
            (0..dim).map(|j| deflated[i][j] * v[j]).sum::<f64>() * vi
        }).sum();
        for i in 0..dim { for j in 0..dim { deflated[i][j] -= eigenvalue * v[i] * v[j]; } }
        components.push(v);
    }

    // Project
    x.iter().map(|xi| {
        components.iter().map(|comp| {
            xi.iter().zip(comp).map(|(&x, &c)| (x - overall_mean[xi.len().min(1) - 1]) * c).sum() // simplified
        }).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = LogisticRegression::fit(&x, &y, 0.5, 1000);
        let pred = model.predict(&x);
        assert_eq!(pred[0], 0);
        assert_eq!(pred[5], 1);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let tree = DecisionTree::fit_regressor(&x, &y, 3, 1);
        let pred = tree.predict(&x);
        // Predictions should be close to actual values
        for (p, &actual) in pred.iter().zip(&y) {
            assert!((p - actual).abs() < 2.0, "pred={p}, actual={actual}");
        }
    }

    #[test]
    fn test_decision_tree_classifier() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0], vec![11.0], vec![12.0]];
        let y = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTree::fit_classifier(&x, &y, 3, 1);
        let pred = tree.predict_class(&x);
        assert_eq!(pred[0], 0);
        assert_eq!(pred[5], 1);
    }

    #[test]
    fn test_random_forest() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0], vec![11.0], vec![12.0]];
        let y = vec![0, 0, 0, 1, 1, 1];
        let rf = RandomForest::fit_classifier(&x, &y, 10, 3, 42);
        let pred = rf.predict_class(&x);
        assert_eq!(pred[0], 0);
        assert_eq!(pred[5], 1);
    }

    #[test]
    fn test_gaussian_nb() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.5, 0.5], vec![5.0, 5.0], vec![6.0, 6.0], vec![5.5, 5.5]];
        let y = vec![0, 0, 0, 1, 1, 1];
        let model = GaussianNB::fit(&x, &y);
        let pred = model.predict(&x);
        assert_eq!(pred[0], 0);
        assert_eq!(pred[5], 1);
    }

    #[test]
    fn test_dbscan() {
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.0, 5.2],
            vec![100.0, 100.0], // noise
        ];
        let labels = dbscan(&data, 0.5, 2);
        assert_eq!(labels.len(), 7);
        // First 3 should be same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        // Next 3 should be same cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // Different clusters
        assert_ne!(labels[0], labels[3]);
    }
}
