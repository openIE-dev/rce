//! Survival analysis: Kaplan-Meier, Cox proportional hazards,
//! Nelson-Aalen, hazard functions, life tables, competing risks.

// ═══════════════════════════════════════════════════════════════════
// KAPLAN-MEIER ESTIMATOR
// ═══════════════════════════════════════════════════════════════════

/// Kaplan-Meier survival curve.
/// times: event times, events: 1=event, 0=censored.
/// Returns (unique_times, survival_probabilities, standard_errors).
pub fn kaplan_meier(times: &[f64], events: &[u8]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = times.len();
    let mut data: Vec<(f64, u8)> = times.iter().zip(events).map(|(&t, &e)| (t, e)).collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut unique_times = Vec::new();
    let mut surv = Vec::new();
    let mut se = Vec::new();
    let mut s = 1.0;
    let mut at_risk = n as f64;
    let mut var_sum = 0.0;

    let mut i = 0;
    while i < n {
        let t = data[i].0;
        let mut d = 0.0; // deaths
        let mut c = 0.0; // censored at this time
        while i < n && (data[i].0 - t).abs() < 1e-15 {
            if data[i].1 == 1 { d += 1.0; } else { c += 1.0; }
            i += 1;
        }
        if d > 0.0 {
            let h = d / at_risk;
            s *= 1.0 - h;
            if at_risk > d { var_sum += d / (at_risk * (at_risk - d)); }
            unique_times.push(t);
            surv.push(s);
            se.push(s * var_sum.sqrt());
        }
        at_risk -= d + c;
    }
    (unique_times, surv, se)
}

/// Median survival time from Kaplan-Meier curve.
pub fn median_survival(times: &[f64], surv: &[f64]) -> f64 {
    for i in 0..surv.len() {
        if surv[i] <= 0.5 { return times[i]; }
    }
    f64::INFINITY
}

/// Restricted mean survival time (area under KM curve up to tau).
pub fn restricted_mean_survival(times: &[f64], surv: &[f64], tau: f64) -> f64 {
    let mut rmst = 0.0;
    let mut prev_t = 0.0;
    let mut prev_s = 1.0;
    for (&t, &s) in times.iter().zip(surv) {
        if t > tau { rmst += prev_s * (tau - prev_t); break; }
        rmst += prev_s * (t - prev_t);
        prev_t = t;
        prev_s = s;
    }
    if times.last().map_or(true, |&t| t <= tau) { rmst += prev_s * (tau - prev_t); }
    rmst
}

// ═══════════════════════════════════════════════════════════════════
// NELSON-AALEN ESTIMATOR
// ═══════════════════════════════════════════════════════════════════

/// Nelson-Aalen cumulative hazard estimator.
/// Returns (unique_times, cumulative_hazard).
pub fn nelson_aalen(times: &[f64], events: &[u8]) -> (Vec<f64>, Vec<f64>) {
    let n = times.len();
    let mut data: Vec<(f64, u8)> = times.iter().zip(events).map(|(&t, &e)| (t, e)).collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut unique_times = Vec::new();
    let mut cum_hazard = Vec::new();
    let mut h = 0.0;
    let mut at_risk = n as f64;

    let mut i = 0;
    while i < n {
        let t = data[i].0;
        let mut d = 0.0;
        let mut c = 0.0;
        while i < n && (data[i].0 - t).abs() < 1e-15 {
            if data[i].1 == 1 { d += 1.0; } else { c += 1.0; }
            i += 1;
        }
        if d > 0.0 {
            h += d / at_risk;
            unique_times.push(t);
            cum_hazard.push(h);
        }
        at_risk -= d + c;
    }
    (unique_times, cum_hazard)
}

// ═══════════════════════════════════════════════════════════════════
// LOG-RANK TEST
// ═══════════════════════════════════════════════════════════════════

/// Log-rank test comparing two groups.
/// Returns (test_statistic, p_value).
pub fn log_rank_test(times1: &[f64], events1: &[u8], times2: &[f64], events2: &[u8]) -> (f64, f64) {
    let mut all_events: Vec<(f64, u8, u8)> = Vec::new(); // (time, event, group)
    for (&t, &e) in times1.iter().zip(events1) { all_events.push((t, e, 0)); }
    for (&t, &e) in times2.iter().zip(events2) { all_events.push((t, e, 1)); }
    all_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut at_risk = [times1.len() as f64, times2.len() as f64];
    let mut observed_minus_expected = 0.0;
    let mut variance = 0.0;

    let mut i = 0;
    let n = all_events.len();
    while i < n {
        let t = all_events[i].0;
        let mut d = [0.0, 0.0]; // deaths per group
        let mut c = [0.0, 0.0]; // censored per group
        while i < n && (all_events[i].0 - t).abs() < 1e-15 {
            let g = all_events[i].2 as usize;
            if all_events[i].1 == 1 { d[g] += 1.0; } else { c[g] += 1.0; }
            i += 1;
        }
        let d_total = d[0] + d[1];
        let n_total = at_risk[0] + at_risk[1];
        if d_total > 0.0 && n_total > 1.0 {
            let expected_1 = d_total * at_risk[0] / n_total;
            observed_minus_expected += d[0] - expected_1;
            variance += d_total * at_risk[0] * at_risk[1] * (n_total - d_total) / (n_total * n_total * (n_total - 1.0));
        }
        at_risk[0] -= d[0] + c[0];
        at_risk[1] -= d[1] + c[1];
    }

    let z = if variance > 1e-15 { observed_minus_expected / variance.sqrt() } else { 0.0 };
    let chi2 = z * z;
    let p = 1.0 - chi2_cdf_approx(chi2, 1.0);
    (chi2, p)
}

// ═══════════════════════════════════════════════════════════════════
// HAZARD FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Exponential hazard: h(t) = λ (constant).
pub fn hazard_exponential(lambda: f64, _t: f64) -> f64 { lambda }

/// Exponential survival: S(t) = exp(-λt).
pub fn survival_exponential(lambda: f64, t: f64) -> f64 { (-lambda * t).exp() }

/// Weibull hazard: h(t) = (k/λ)(t/λ)^{k-1}.
pub fn hazard_weibull(k: f64, lambda: f64, t: f64) -> f64 {
    (k / lambda) * (t / lambda).powf(k - 1.0)
}

/// Weibull survival: S(t) = exp(-(t/λ)^k).
pub fn survival_weibull(k: f64, lambda: f64, t: f64) -> f64 {
    (-(t / lambda).powf(k)).exp()
}

/// Log-logistic survival: S(t) = 1 / (1 + (t/α)^β).
pub fn survival_log_logistic(alpha: f64, beta: f64, t: f64) -> f64 {
    1.0 / (1.0 + (t / alpha).powf(beta))
}

/// Gompertz hazard: h(t) = b·exp(ct).
pub fn hazard_gompertz(b: f64, c: f64, t: f64) -> f64 { b * (c * t).exp() }

/// Gompertz survival: S(t) = exp(-b/c·(exp(ct) - 1)).
pub fn survival_gompertz(b: f64, c: f64, t: f64) -> f64 {
    (-b / c * ((c * t).exp() - 1.0)).exp()
}

// ═══════════════════════════════════════════════════════════════════
// LIFE TABLE
// ═══════════════════════════════════════════════════════════════════

/// Actuarial life table. intervals: [(start, end)], deaths per interval, censored per interval.
/// Returns (interval_survival, cumulative_survival, hazard).
pub fn life_table(n_start: f64, deaths: &[f64], censored: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = deaths.len();
    let mut interval_surv = Vec::with_capacity(k);
    let mut cum_surv = Vec::with_capacity(k);
    let mut hazards = Vec::with_capacity(k);
    let mut at_risk = n_start;

    for i in 0..k {
        let effective = at_risk - censored[i] / 2.0; // adjusted for censoring
        let q = if effective > 0.0 { deaths[i] / effective } else { 0.0 };
        let p = 1.0 - q;
        interval_surv.push(p);
        let s = if i == 0 { p } else { cum_surv[i - 1] * p };
        cum_surv.push(s);
        hazards.push(q);
        at_risk -= deaths[i] + censored[i];
    }
    (interval_surv, cum_surv, hazards)
}

/// Life expectancy from life table survival probabilities and interval widths.
pub fn life_expectancy(cum_surv: &[f64], interval_width: f64) -> f64 {
    cum_surv.iter().sum::<f64>() * interval_width
}

// ═══════════════════════════════════════════════════════════════════
// COX REGRESSION UTILITIES
// ═══════════════════════════════════════════════════════════════════

/// Partial likelihood for Cox model (single covariate, for concept).
/// beta: coefficient, x: covariate values, times: event times, events: 0/1.
pub fn cox_partial_likelihood(beta: f64, x: &[f64], times: &[f64], events: &[u8]) -> f64 {
    let n = x.len();
    let mut data: Vec<(f64, u8, f64)> = (0..n).map(|i| (times[i], events[i], x[i])).collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut log_pl = 0.0;
    for i in 0..n {
        if data[i].1 != 1 { continue; }
        let xi = data[i].2;
        let mut risk_sum = 0.0;
        for j in i..n { risk_sum += (beta * data[j].2).exp(); }
        log_pl += beta * xi - risk_sum.ln();
    }
    log_pl
}

/// Concordance index (C-index) for survival predictions.
pub fn concordance_index(predicted: &[f64], times: &[f64], events: &[u8]) -> f64 {
    let n = predicted.len();
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    for i in 0..n {
        if events[i] != 1 { continue; }
        for j in 0..n {
            if i == j { continue; }
            if times[j] > times[i] {
                if predicted[j] < predicted[i] { concordant += 1.0; }
                else if predicted[j] > predicted[i] { discordant += 1.0; }
                else { concordant += 0.5; discordant += 0.5; }
            }
        }
    }
    let total = concordant + discordant;
    if total < 1e-15 { return 0.5; }
    concordant / total
}

/// Brier score for survival predictions at time t.
pub fn brier_score(predicted_survival: &[f64], times: &[f64], events: &[u8], t: f64) -> f64 {
    let n = predicted_survival.len();
    let mut score = 0.0;
    for i in 0..n {
        let y = if times[i] <= t && events[i] == 1 { 1.0 } else { 0.0 };
        let s = predicted_survival[i];
        score += (y - (1.0 - s)).powi(2);
    }
    score / n as f64
}

fn chi2_cdf_approx(x: f64, df: f64) -> f64 {
    let z = (x / df).cbrt() - (1.0 - 2.0 / (9.0 * df));
    let z = z / (2.0 / (9.0 * df)).sqrt();
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -r } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaplan_meier() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![1, 0, 1, 0, 1, 1, 0, 1];
        let (t, s, _se) = kaplan_meier(&times, &events);
        assert!(s[0] < 1.0 && s[0] > 0.0);
        // Survival should be non-increasing
        for i in 1..s.len() { assert!(s[i] <= s[i-1] + 1e-10); }
    }

    #[test]
    fn test_nelson_aalen() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![1, 1, 0, 1, 1];
        let (t, h) = nelson_aalen(&times, &events);
        // Cumulative hazard should be non-decreasing
        for i in 1..h.len() { assert!(h[i] >= h[i-1] - 1e-10); }
    }

    #[test]
    fn test_survival_exponential() {
        assert!((survival_exponential(0.1, 0.0) - 1.0).abs() < 1e-10);
        assert!(survival_exponential(0.1, 10.0) < 1.0);
    }

    #[test]
    fn test_survival_weibull() {
        assert!((survival_weibull(1.0, 1.0, 0.0) - 1.0).abs() < 1e-10);
        // Weibull with k=1 = exponential
        let s_weibull = survival_weibull(1.0, 2.0, 1.0);
        let s_exp = survival_exponential(0.5, 1.0);
        assert!((s_weibull - s_exp).abs() < 1e-6);
    }

    #[test]
    fn test_life_table() {
        let deaths = vec![10.0, 15.0, 20.0];
        let censored = vec![5.0, 3.0, 2.0];
        let (is, cs, h) = life_table(100.0, &deaths, &censored);
        assert_eq!(is.len(), 3);
        assert!(cs[0] > cs[1]); // cumulative survival decreasing
    }

    #[test]
    fn test_concordance() {
        // Perfect concordance
        let predicted = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![1, 1, 1, 1, 1];
        let c = concordance_index(&predicted, &times, &events);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_rank() {
        // Same distribution → non-significant
        let t1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e1 = vec![1, 1, 1, 1, 1];
        let t2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e2 = vec![1, 1, 1, 1, 1];
        let (stat, p) = log_rank_test(&t1, &e1, &t2, &e2);
        assert!(p > 0.05); // not significant
    }

    #[test]
    fn test_restricted_mean() {
        let times = vec![1.0, 2.0, 3.0];
        let surv = vec![0.9, 0.7, 0.5];
        let rmst = restricted_mean_survival(&times, &surv, 4.0);
        assert!(rmst > 0.0 && rmst < 4.0);
    }
}
