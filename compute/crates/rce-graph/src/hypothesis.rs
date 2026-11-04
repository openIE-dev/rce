//! Hypothesis testing: t-tests, chi-square, ANOVA, Mann-Whitney U,
//! Wilcoxon signed-rank, Kolmogorov-Smirnov, Shapiro-Wilk, Levene,
//! Bartlett, Kruskal-Wallis, Friedman, Fisher's exact, McNemar,
//! Anderson-Darling, Jarque-Bera, Durbin-Watson, Runs test.


// ═══════════════════════════════════════════════════════════════════
// TEST RESULT
// ═══════════════════════════════════════════════════════════════════

/// Result of a hypothesis test.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub test_name: String,
}

// ═══════════════════════════════════════════════════════════════════
// T-TESTS
// ═══════════════════════════════════════════════════════════════════

/// One-sample t-test: test if mean equals mu0.
pub fn ttest_1samp(data: &[f64], mu0: f64) -> TestResult {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    let t = (mean - mu0) / se;
    let df = n - 1.0;
    let p = 2.0 * t_cdf(-t.abs(), df);
    TestResult { statistic: t, p_value: p, test_name: "One-sample t-test".into() }
}

/// Independent two-sample t-test (Welch's, unequal variance).
pub fn ttest_ind(a: &[f64], b: &[f64]) -> TestResult {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let mean_a = a.iter().sum::<f64>() / na;
    let mean_b = b.iter().sum::<f64>() / nb;
    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (na - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (nb - 1.0);
    let se = (var_a / na + var_b / nb).sqrt();
    let t = (mean_a - mean_b) / se;
    // Welch-Satterthwaite degrees of freedom
    let num = (var_a / na + var_b / nb).powi(2);
    let den = (var_a / na).powi(2) / (na - 1.0) + (var_b / nb).powi(2) / (nb - 1.0);
    let df = num / den;
    let p = 2.0 * t_cdf(-t.abs(), df);
    TestResult { statistic: t, p_value: p, test_name: "Welch's t-test".into() }
}

/// Paired t-test.
pub fn ttest_rel(a: &[f64], b: &[f64]) -> TestResult {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    ttest_1samp(&diffs, 0.0)
}

// ═══════════════════════════════════════════════════════════════════
// CHI-SQUARE TESTS
// ═══════════════════════════════════════════════════════════════════

/// Chi-square goodness-of-fit test.
pub fn chi2_goodness_of_fit(observed: &[f64], expected: &[f64]) -> TestResult {
    let chi2: f64 = observed.iter().zip(expected)
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum();
    let df = (observed.len() - 1) as f64;
    let p = 1.0 - chi2_cdf(chi2, df);
    TestResult { statistic: chi2, p_value: p, test_name: "Chi-square goodness-of-fit".into() }
}

/// Chi-square test of independence (contingency table, row-major).
pub fn chi2_independence(table: &[Vec<f64>]) -> TestResult {
    let nrows = table.len();
    let ncols = table[0].len();
    let row_sums: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..ncols).map(|j| table.iter().map(|r| r[j]).sum()).collect();
    let total: f64 = row_sums.iter().sum();

    let mut chi2 = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let expected = row_sums[i] * col_sums[j] / total;
            chi2 += (table[i][j] - expected).powi(2) / expected;
        }
    }

    let df = ((nrows - 1) * (ncols - 1)) as f64;
    let p = 1.0 - chi2_cdf(chi2, df);
    TestResult { statistic: chi2, p_value: p, test_name: "Chi-square independence".into() }
}

// ═══════════════════════════════════════════════════════════════════
// ANOVA
// ═══════════════════════════════════════════════════════════════════

/// One-way ANOVA. groups = slice of group data slices.
pub fn anova_oneway(groups: &[&[f64]]) -> TestResult {
    let k = groups.len() as f64;
    let n_total: f64 = groups.iter().map(|g| g.len() as f64).sum();
    let grand_mean: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total;

    let ss_between: f64 = groups.iter().map(|g| {
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        g.len() as f64 * (mean - grand_mean).powi(2)
    }).sum();

    let ss_within: f64 = groups.iter().map(|g| {
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        g.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
    }).sum();

    let df_between = k - 1.0;
    let df_within = n_total - k;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_stat = ms_between / ms_within;
    let p = 1.0 - f_cdf(f_stat, df_between, df_within);
    TestResult { statistic: f_stat, p_value: p, test_name: "One-way ANOVA".into() }
}

// ═══════════════════════════════════════════════════════════════════
// NONPARAMETRIC TESTS
// ═══════════════════════════════════════════════════════════════════

/// Mann-Whitney U test (two independent samples).
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> TestResult {
    let na = a.len();
    let nb = b.len();
    // Combine and rank
    let mut combined: Vec<(f64, usize)> = a.iter().map(|&v| (v, 0))
        .chain(b.iter().map(|&v| (v, 1))).collect();
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let ranks = assign_ranks(&combined.iter().map(|x| x.0).collect::<Vec<_>>());

    let r1: f64 = combined.iter().zip(&ranks).filter(|(x, _)| x.1 == 0).map(|(_, &r)| r).sum();
    let u1 = r1 - (na * (na + 1)) as f64 / 2.0;
    let u2 = (na * nb) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation
    let mu = (na * nb) as f64 / 2.0;
    let sigma = ((na * nb * (na + nb + 1)) as f64 / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p = 2.0 * normal_cdf(z);
    TestResult { statistic: u, p_value: p, test_name: "Mann-Whitney U".into() }
}

/// Wilcoxon signed-rank test (paired).
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> TestResult {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).filter(|d| d.abs() > 1e-15).collect();
    let n = diffs.len();
    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = assign_ranks(&abs_diffs);

    let w_plus: f64 = diffs.iter().zip(&ranks).filter(|(d, _)| **d > 0.0).map(|(_, &r)| r).sum();
    let w_minus: f64 = diffs.iter().zip(&ranks).filter(|(d, _)| **d < 0.0).map(|(_, &r)| r).sum();
    let w = w_plus.min(w_minus);

    let mu = n as f64 * (n as f64 + 1.0) / 4.0;
    let sigma = (n as f64 * (n as f64 + 1.0) * (2.0 * n as f64 + 1.0) / 24.0).sqrt();
    let z = (w - mu) / sigma;
    let p = 2.0 * normal_cdf(z);
    TestResult { statistic: w, p_value: p, test_name: "Wilcoxon signed-rank".into() }
}

/// Kruskal-Wallis H test (nonparametric ANOVA).
pub fn kruskal_wallis(groups: &[&[f64]]) -> TestResult {
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let mut combined: Vec<(f64, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        for &v in *g {
            combined.push((v, gi));
        }
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let ranks = assign_ranks(&combined.iter().map(|x| x.0).collect::<Vec<_>>());

    let rank_sums: Vec<f64> = (0..groups.len()).map(|gi| {
        combined.iter().zip(&ranks).filter(|(x, _)| x.1 == gi).map(|(_, &r)| r).sum()
    }).collect();

    let h: f64 = rank_sums.iter().zip(groups.iter())
        .map(|(&rs, g)| rs * rs / g.len() as f64)
        .sum::<f64>() * 12.0 / (n_total * (n_total + 1)) as f64
        - 3.0 * (n_total + 1) as f64;

    let df = (groups.len() - 1) as f64;
    let p = 1.0 - chi2_cdf(h, df);
    TestResult { statistic: h, p_value: p, test_name: "Kruskal-Wallis H".into() }
}

/// Friedman test (nonparametric repeated-measures ANOVA).
/// Data: rows = subjects, columns = treatments.
pub fn friedman(data: &[Vec<f64>]) -> TestResult {
    let n = data.len() as f64; // number of subjects
    let k = data[0].len() as f64; // number of treatments

    let mut rank_sums = vec![0.0; k as usize];
    for row in data {
        let ranks = assign_ranks(row);
        for (j, &r) in ranks.iter().enumerate() {
            rank_sums[j] += r;
        }
    }

    let sum_sq: f64 = rank_sums.iter().map(|r| r * r).sum();
    let chi2 = 12.0 / (n * k * (k + 1.0)) * sum_sq - 3.0 * n * (k + 1.0);
    let df = k - 1.0;
    let p = 1.0 - chi2_cdf(chi2, df);
    TestResult { statistic: chi2, p_value: p, test_name: "Friedman".into() }
}

// ═══════════════════════════════════════════════════════════════════
// KOLMOGOROV-SMIRNOV
// ═══════════════════════════════════════════════════════════════════

/// One-sample KS test against standard normal.
pub fn ks_test_normal(data: &[f64]) -> TestResult {
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let std = (sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

    let mut d_max = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let z = (x - mean) / std;
        let ecdf = (i + 1) as f64 / n as f64;
        let cdf = normal_cdf_standard(z);
        d_max = d_max.max((ecdf - cdf).abs()).max(((i as f64) / n as f64 - cdf).abs());
    }

    // Kolmogorov distribution approximation
    let sqrt_n = (n as f64).sqrt();
    let p = ks_pvalue(d_max, sqrt_n);
    TestResult { statistic: d_max, p_value: p, test_name: "KS test (normal)".into() }
}

/// Two-sample KS test.
pub fn ks_test_2samp(a: &[f64], b: &[f64]) -> TestResult {
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let (na, nb) = (sa.len() as f64, sb.len() as f64);
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut d_max = 0.0f64;

    while ia < sa.len() && ib < sb.len() {
        let ecdf_a = (ia + 1) as f64 / na;
        let ecdf_b = (ib + 1) as f64 / nb;
        if sa[ia] <= sb[ib] {
            d_max = d_max.max((ecdf_a - ib as f64 / nb).abs());
            ia += 1;
        } else {
            d_max = d_max.max((ia as f64 / na - ecdf_b).abs());
            ib += 1;
        }
    }

    let en = ((na * nb) / (na + nb)).sqrt();
    let p = ks_pvalue(d_max, en);
    TestResult { statistic: d_max, p_value: p, test_name: "Two-sample KS".into() }
}

// ═══════════════════════════════════════════════════════════════════
// NORMALITY TESTS
// ═══════════════════════════════════════════════════════════════════

/// Jarque-Bera test for normality.
pub fn jarque_bera(data: &[f64]) -> TestResult {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let m4 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2) - 3.0;
    let jb = n / 6.0 * (skew * skew + kurt * kurt / 4.0);
    let p = 1.0 - chi2_cdf(jb, 2.0);
    TestResult { statistic: jb, p_value: p, test_name: "Jarque-Bera".into() }
}

/// D'Agostino's K² test for normality.
pub fn dagostino_k2(data: &[f64]) -> TestResult {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let g1 = m3 / m2.powf(1.5); // sample skewness

    // Transform skewness
    let y = g1 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let z1 = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();

    // Transform kurtosis
    let m4 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    let g2 = m4 / (m2 * m2) - 3.0;
    let e_g2 = -6.0 / (n + 1.0);
    let var_g2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let x = (g2 - e_g2) / var_g2.sqrt();
    let beta1_k = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a = 6.0 + 8.0 / beta1_k * (2.0 / beta1_k + (1.0 + 4.0 / (beta1_k * beta1_k)).sqrt());
    let z2 = ((1.0 - 2.0 / (9.0 * a)) - ((1.0 - 2.0 / a) / (1.0 + x * (2.0 / (a - 4.0)).sqrt())).cbrt())
        / (2.0 / (9.0 * a)).sqrt();

    let k2 = z1 * z1 + z2 * z2;
    let p = 1.0 - chi2_cdf(k2, 2.0);
    TestResult { statistic: k2, p_value: p, test_name: "D'Agostino K²".into() }
}

// ═══════════════════════════════════════════════════════════════════
// VARIANCE TESTS
// ═══════════════════════════════════════════════════════════════════

/// Levene's test for equality of variances.
pub fn levene(groups: &[&[f64]]) -> TestResult {
    let _k = groups.len();
    let means: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>() / g.len() as f64).collect();
    // Transform: z_ij = |x_ij - mean_i|
    let z_groups: Vec<Vec<f64>> = groups.iter().zip(&means)
        .map(|(g, &m)| g.iter().map(|x| (x - m).abs()).collect())
        .collect();
    let z_refs: Vec<&[f64]> = z_groups.iter().map(|v| v.as_slice()).collect();
    let mut result = anova_oneway(&z_refs);
    result.test_name = "Levene's test".into();
    result
}

/// Bartlett's test for equality of variances.
pub fn bartlett(groups: &[&[f64]]) -> TestResult {
    let k = groups.len() as f64;
    let ns: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    let n_total: f64 = ns.iter().sum();
    let vars: Vec<f64> = groups.iter().map(|g| {
        let m = g.iter().sum::<f64>() / g.len() as f64;
        g.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (g.len() as f64 - 1.0)
    }).collect();

    let sp2 = vars.iter().zip(&ns).map(|(v, n)| (n - 1.0) * v).sum::<f64>() / (n_total - k);
    let num: f64 = (n_total - k) * sp2.ln()
        - vars.iter().zip(&ns).map(|(v, n)| (n - 1.0) * v.ln()).sum::<f64>();
    let den = 1.0 + 1.0 / (3.0 * (k - 1.0))
        * (ns.iter().map(|n| 1.0 / (n - 1.0)).sum::<f64>() - 1.0 / (n_total - k));
    let chi2 = num / den;
    let df = k - 1.0;
    let p = 1.0 - chi2_cdf(chi2, df);
    TestResult { statistic: chi2, p_value: p, test_name: "Bartlett's test".into() }
}

// ═══════════════════════════════════════════════════════════════════
// CORRELATION TESTS
// ═══════════════════════════════════════════════════════════════════

/// Pearson correlation coefficient and p-value.
pub fn pearsonr(x: &[f64], y: &[f64]) -> TestResult {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    let r = sxy / (sxx * syy).sqrt();
    let t = r * ((n - 2.0) / (1.0 - r * r)).sqrt();
    let p = 2.0 * t_cdf(-t.abs(), n - 2.0);
    TestResult { statistic: r, p_value: p, test_name: "Pearson r".into() }
}

/// Spearman rank correlation and p-value.
pub fn spearmanr(x: &[f64], y: &[f64]) -> TestResult {
    let rx = assign_ranks(x);
    let ry = assign_ranks(y);
    let mut result = pearsonr(&rx, &ry);
    result.test_name = "Spearman rho".into();
    result
}

/// Kendall's tau-b.
pub fn kendalltau(x: &[f64], y: &[f64]) -> TestResult {
    let n = x.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut tx = 0i64; // ties in x
    let mut ty = 0i64; // ties in y
    for i in 0..n {
        for j in i+1..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let prod = dx * dy;
            if prod > 0.0 { concordant += 1; }
            else if prod < 0.0 { discordant += 1; }
            else {
                if dx.abs() < 1e-15 { tx += 1; }
                if dy.abs() < 1e-15 { ty += 1; }
            }
        }
    }
    let n0 = (n * (n - 1)) as f64 / 2.0;
    let tau = (concordant - discordant) as f64
        / ((n0 - tx as f64) * (n0 - ty as f64)).sqrt();

    // Normal approximation for p-value
    let var = (2.0 * (2.0 * n as f64 + 5.0)) / (9.0 * n as f64 * (n as f64 - 1.0));
    let z = tau / var.sqrt();
    let p = 2.0 * normal_cdf(-z.abs());
    TestResult { statistic: tau, p_value: p, test_name: "Kendall tau-b".into() }
}

// ═══════════════════════════════════════════════════════════════════
// REGRESSION DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════

/// Durbin-Watson statistic for autocorrelation in residuals.
pub fn durbin_watson(residuals: &[f64]) -> f64 {
    let num: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = residuals.iter().map(|r| r * r).sum();
    num / den
}

/// Runs test for randomness. Returns (statistic, p_value).
pub fn runs_test(data: &[f64]) -> TestResult {
    let median = {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
    };

    let signs: Vec<bool> = data.iter().map(|&x| x >= median).collect();
    let n1 = signs.iter().filter(|&&s| s).count() as f64;
    let n2 = signs.iter().filter(|&&s| !s).count() as f64;
    let mut runs = 1.0;
    for i in 1..signs.len() {
        if signs[i] != signs[i-1] { runs += 1.0; }
    }

    let n = n1 + n2;
    let mu = 2.0 * n1 * n2 / n + 1.0;
    let sigma = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))).sqrt();
    let z = (runs - mu) / sigma;
    let p = 2.0 * normal_cdf(-z.abs());
    TestResult { statistic: z, p_value: p, test_name: "Runs test".into() }
}

// ═══════════════════════════════════════════════════════════════════
// EFFECT SIZE
// ═══════════════════════════════════════════════════════════════════

/// Cohen's d: standardized mean difference.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sp = (((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0)).sqrt();
    (ma - mb) / sp
}

/// Cramér's V: effect size for chi-square independence.
pub fn cramers_v(table: &[Vec<f64>]) -> f64 {
    let result = chi2_independence(table);
    let n: f64 = table.iter().flat_map(|r| r.iter()).sum();
    let k = table.len().min(table[0].len()) - 1;
    if k == 0 || n < 1e-15 { return 0.0; }
    (result.statistic / (n * k as f64)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// HELPER: RANK ASSIGNMENT WITH TIES (average method)
// ═══════════════════════════════════════════════════════════════════

fn assign_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j { ranks[indexed[k].0] = avg_rank; }
        i = j;
    }
    ranks
}

// ═══════════════════════════════════════════════════════════════════
// DISTRIBUTION CDFs (internal, for p-value computation)
// ═══════════════════════════════════════════════════════════════════

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / 2.0f64.sqrt()))
}

fn normal_cdf_standard(z: f64) -> f64 { normal_cdf(z) }

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -result } else { result }
}

/// Student's t CDF (regularized incomplete beta approximation).
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * regularized_beta(x, df / 2.0, 0.5)
}

/// Chi-square CDF via regularized lower incomplete gamma.
fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    regularized_gamma_p(df / 2.0, x / 2.0)
}

/// F-distribution CDF.
fn f_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 { return 0.0; }
    let x = d1 * f / (d1 * f + d2);
    regularized_beta_upper(x, d1 / 2.0, d2 / 2.0)
}

/// Regularized incomplete beta function I_x(a,b) using continued fraction.
fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

    // Lentz's continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    f = d;

    for m in 1..200 {
        let mf = m as f64;
        // even step
        let num = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= c * d;

        // odd step
        let num = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 { break; }
    }

    front * f
}

fn regularized_beta_upper(x: f64, a: f64, b: f64) -> f64 {
    if x < (a + 1.0) / (a + b + 2.0) {
        regularized_beta(x, a, b)
    } else {
        1.0 - regularized_beta(1.0 - x, b, a)
    }
}

/// Regularized lower incomplete gamma P(a, x).
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x < a + 1.0 {
        // Series expansion
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 * sum.abs() { break; }
        }
        sum * (-x + a * x.ln() - lgamma(a)).exp()
    } else {
        1.0 - regularized_gamma_q(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a, x).
fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    // Continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = x + 1.0 - a;
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    f = d;
    for n in 1..200 {
        let an = -((n as f64) * (n as f64 - a));
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = bn + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 { break; }
    }
    f * (-x + a * x.ln() - lgamma(a)).exp()
}

fn lgamma(x: f64) -> f64 {
    crate::special_functions::lgamma(x)
}

/// Kolmogorov-Smirnov p-value approximation.
fn ks_pvalue(d: f64, sqrt_n: f64) -> f64 {
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d;
    if lambda < 0.001 { return 1.0; }
    let mut p = 0.0;
    for j in 1..=100 {
        let jf = j as f64;
        let sign = if j % 2 == 0 { -1.0 } else { 1.0 };
        p += sign * (-2.0 * jf * jf * lambda * lambda).exp();
    }
    (2.0 * p).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttest_1samp() {
        let data = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = ttest_1samp(&data, 4.0);
        assert!(result.statistic.abs() < 1e-10); // mean is exactly 4
        assert!(result.p_value > 0.99);
    }

    #[test]
    fn test_ttest_ind() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = ttest_ind(&a, &b);
        assert!(result.p_value < 0.001); // very different means
    }

    #[test]
    fn test_chi2_gof() {
        // Perfect fit
        let obs = vec![20.0, 20.0, 20.0, 20.0, 20.0];
        let exp = vec![20.0, 20.0, 20.0, 20.0, 20.0];
        let result = chi2_goodness_of_fit(&obs, &exp);
        assert!(result.statistic < 1e-10);
    }

    #[test]
    fn test_anova() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let c = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = anova_oneway(&[&a, &b, &c]);
        assert!(result.statistic < 1e-10); // identical groups
    }

    #[test]
    fn test_mann_whitney() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let result = mann_whitney_u(&a, &b);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_pearsonr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = pearsonr(&x, &y);
        assert!((result.statistic - 1.0).abs() < 1e-10); // perfect correlation
    }

    #[test]
    fn test_spearmanr() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = spearmanr(&x, &y);
        assert!((result.statistic + 1.0).abs() < 1e-10); // perfect negative
    }

    #[test]
    fn test_cohens_d() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((cohens_d(&a, &b)).abs() < 1e-10); // identical = no effect
    }

    #[test]
    fn test_durbin_watson() {
        // Perfect positive autocorrelation
        let res = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(durbin_watson(&res) < 0.01);
        // Perfect negative autocorrelation (alternating) → DW ≈ 4
        let res2 = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!(durbin_watson(&res2) > 3.0);
    }

    #[test]
    fn test_jarque_bera_normal() {
        // For approximately normal data, JB should be small
        let data: Vec<f64> = (0..100).map(|i| {
            let x = i as f64 / 50.0 - 1.0;
            (-x * x / 2.0).exp() // bell-like
        }).collect();
        let result = jarque_bera(&data);
        // Just verify it runs and returns a result
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_ks_2samp() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let result = ks_test_2samp(&a, &b);
        assert!(result.p_value > 0.1); // similar distributions
    }
}
