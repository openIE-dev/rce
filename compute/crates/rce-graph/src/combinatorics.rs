//! Combinatorics: factorials, binomial coefficients, permutations,
//! combinations, partitions, Catalan numbers, Stirling numbers,
//! Bell numbers, derangements, multinomials, integer partitions.

// ═══════════════════════════════════════════════════════════════════
// FACTORIALS & BASIC COUNTING
// ═══════════════════════════════════════════════════════════════════

/// Factorial n! (returns f64 for large values).
pub fn factorial(n: u64) -> f64 {
    if n <= 1 { return 1.0; }
    (2..=n).fold(1.0, |acc, i| acc * i as f64)
}

/// Log-factorial: ln(n!).
pub fn log_factorial(n: u64) -> f64 {
    if n <= 1 { return 0.0; }
    (2..=n).fold(0.0, |acc, i| acc + (i as f64).ln())
}

/// Double factorial n!! (1·3·5·...·n for odd, 2·4·6·...·n for even).
pub fn double_factorial(n: u64) -> f64 {
    if n <= 1 { return 1.0; }
    let mut result = 1.0;
    let mut k = n;
    while k > 1 {
        result *= k as f64;
        k -= 2;
    }
    result
}

/// Subfactorial !n (number of derangements).
pub fn subfactorial(n: u64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return 0.0; }
    let mut d0 = 1.0;
    let mut d1 = 0.0;
    for i in 2..=n {
        let d2 = (i as f64 - 1.0) * (d1 + d0);
        d0 = d1;
        d1 = d2;
    }
    d1
}

/// Falling factorial: x · (x-1) · ... · (x-n+1).
pub fn falling_factorial(x: f64, n: u64) -> f64 {
    (0..n).fold(1.0, |acc, i| acc * (x - i as f64))
}

/// Rising factorial (Pochhammer): x · (x+1) · ... · (x+n-1).
pub fn rising_factorial(x: f64, n: u64) -> f64 {
    (0..n).fold(1.0, |acc, i| acc * (x + i as f64))
}

// ═══════════════════════════════════════════════════════════════════
// BINOMIAL COEFFICIENTS
// ═══════════════════════════════════════════════════════════════════

/// Binomial coefficient C(n, k).
pub fn binomial(n: u64, k: u64) -> f64 {
    if k > n { return 0.0; }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Log-binomial: ln(C(n,k)).
pub fn log_binomial(n: u64, k: u64) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Multinomial coefficient: n! / (k1! · k2! · ... · km!).
pub fn multinomial(ks: &[u64]) -> f64 {
    let n: u64 = ks.iter().sum();
    let log_num = log_factorial(n);
    let log_den: f64 = ks.iter().map(|&k| log_factorial(k)).sum();
    (log_num - log_den).exp()
}

// ═══════════════════════════════════════════════════════════════════
// PERMUTATIONS & COMBINATIONS (enumeration)
// ═══════════════════════════════════════════════════════════════════

/// Count of k-permutations: P(n,k) = n!/(n-k)!.
pub fn count_permutations(n: u64, k: u64) -> f64 {
    if k > n { return 0.0; }
    falling_factorial(n as f64, k)
}

/// Count of k-combinations: C(n,k).
pub fn count_combinations(n: u64, k: u64) -> f64 { binomial(n, k) }

/// Count combinations with repetition: C(n+k-1, k).
pub fn count_combinations_with_rep(n: u64, k: u64) -> f64 {
    binomial(n + k - 1, k)
}

/// Generate all permutations of a slice (Heap's algorithm).
pub fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let n = items.len();
    if n == 0 { return vec![vec![]]; }
    let mut result = Vec::new();
    let mut arr = items.to_vec();
    let mut c = vec![0; n];
    result.push(arr.clone());
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 { arr.swap(0, i); } else { arr.swap(c[i], i); }
            result.push(arr.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    result
}

/// Generate all k-combinations of indices 0..n.
pub fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k > n { return vec![]; }
    if k == 0 { return vec![vec![]]; }
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn build(start: usize, n: usize, k: usize, combo: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        for i in start..=n-(k-combo.len()) {
            combo.push(i);
            build(i + 1, n, k, combo, result);
            combo.pop();
        }
    }
    build(0, n, k, &mut combo, &mut result);
    result
}

/// Generate all subsets (power set) of indices 0..n.
pub fn power_set(n: usize) -> Vec<Vec<usize>> {
    let total = 1usize << n;
    (0..total).map(|mask| {
        (0..n).filter(|&i| mask & (1 << i) != 0).collect()
    }).collect()
}

/// Next permutation (lexicographic order). Returns false if at last permutation.
pub fn next_permutation(arr: &mut [usize]) -> bool {
    let n = arr.len();
    if n < 2 { return false; }
    let mut i = n - 2;
    while arr[i] >= arr[i + 1] {
        if i == 0 { return false; }
        i -= 1;
    }
    let mut j = n - 1;
    while arr[j] <= arr[i] { j -= 1; }
    arr.swap(i, j);
    arr[i + 1..].reverse();
    true
}

// ═══════════════════════════════════════════════════════════════════
// SPECIAL SEQUENCES
// ═══════════════════════════════════════════════════════════════════

/// Catalan number C_n.
pub fn catalan(n: u64) -> f64 {
    binomial(2 * n, n) / (n as f64 + 1.0)
}

/// Bell number B_n (number of set partitions).
pub fn bell(n: usize) -> f64 {
    if n == 0 { return 1.0; }
    let mut triangle = vec![vec![0.0; n + 1]; n + 1];
    triangle[0][0] = 1.0;
    for i in 1..=n {
        triangle[i][0] = triangle[i - 1][i - 1];
        for j in 1..=i {
            triangle[i][j] = triangle[i][j - 1] + triangle[i - 1][j - 1];
        }
    }
    triangle[n][0]
}

/// Stirling number of the second kind S(n, k): partitions of n elements into k non-empty subsets.
pub fn stirling2(n: usize, k: usize) -> f64 {
    if k == 0 { return if n == 0 { 1.0 } else { 0.0 }; }
    if k > n { return 0.0; }
    if k == n { return 1.0; }
    // Inclusion-exclusion formula
    let mut sum = 0.0;
    for j in 0..=k {
        let sign = if (k - j) % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * binomial(k as u64, j as u64) * (j as f64).powi(n as i32);
    }
    sum / factorial(k as u64)
}

/// Stirling number of the first kind (unsigned) |s(n,k)|.
pub fn stirling1(n: usize, k: usize) -> f64 {
    if k == 0 { return if n == 0 { 1.0 } else { 0.0 }; }
    if k > n { return 0.0; }
    // Recurrence: |s(n,k)| = (n-1)|s(n-1,k)| + |s(n-1,k-1)|
    let mut prev = vec![0.0; k + 1];
    prev[0] = 1.0;
    for i in 1..=n {
        let mut curr = vec![0.0; k + 1];
        for j in 1..=k.min(i) {
            curr[j] = (i as f64 - 1.0) * prev[j] + prev[j - 1];
        }
        prev = curr;
    }
    prev[k]
}

/// Euler number E(n) (tangent/secant numbers).
pub fn euler_number(n: usize) -> f64 {
    if n % 2 != 0 { return 0.0; }
    // Using secant numbers via recurrence
    let m = n / 2;
    let mut e = vec![0.0; m + 1];
    e[0] = 1.0;
    for i in 1..=m {
        for j in (0..i).rev() {
            e[j + 1] = (j as f64 + 1.0) * e[j] + (j as f64 + 2.0) * e[j + 1];
        }
        // Actually, this gets Euler zigzag numbers; use different approach
    }
    // Simpler: alternating sign sum of binomial * Bernoulli-like
    // Use tangent number approach
    let mut a = vec![0.0; n + 2];
    a[0] = 1.0;
    for i in 1..=n {
        a[i] = 0.0;
        for k in 0..i {
            a[i] += binomial(i as u64, k as u64) * a[k] / 2.0;
        }
        a[i] = if i == 1 { -0.5 } else { -a[i] + if i % 2 == 0 { 1.0 } else { 0.0 } };
    }
    a[n] * factorial(n as u64)
}

/// Fibonacci number F_n.
pub fn fibonacci(n: u64) -> f64 {
    if n <= 1 { return n as f64; }
    let mut a = 0.0;
    let mut b = 1.0;
    for _ in 2..=n {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

/// Lucas number L_n.
pub fn lucas(n: u64) -> f64 {
    if n == 0 { return 2.0; }
    if n == 1 { return 1.0; }
    let mut a = 2.0;
    let mut b = 1.0;
    for _ in 2..=n {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

// ═══════════════════════════════════════════════════════════════════
// INTEGER PARTITIONS
// ═══════════════════════════════════════════════════════════════════

/// Number of integer partitions of n.
pub fn partition_count(n: usize) -> f64 {
    let mut dp = vec![0.0; n + 1];
    dp[0] = 1.0;
    for k in 1..=n {
        for j in k..=n {
            dp[j] += dp[j - k];
        }
    }
    dp[n]
}

/// Generate all integer partitions of n.
pub fn integer_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    fn build(remaining: usize, max_val: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if remaining == 0 {
            result.push(current.clone());
            return;
        }
        for k in (1..=remaining.min(max_val)).rev() {
            current.push(k);
            build(remaining - k, k, current, result);
            current.pop();
        }
    }
    let mut current = Vec::new();
    build(n, n, &mut current, &mut result);
    result
}

// ═══════════════════════════════════════════════════════════════════
// GCD / LCM / MODULAR ARITHMETIC
// ═══════════════════════════════════════════════════════════════════

/// Greatest common divisor.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { return 0; }
    a / gcd(a, b) * b
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 { return (b, 0, 1); }
    let (g, x1, y1) = extended_gcd(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp /= 2;
        base = base.wrapping_mul(base) % modulus;
    }
    result
}

/// Modular inverse: a^(-1) mod m (if it exists).
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 { return None; }
    Some(((x % m as i64 + m as i64) % m as i64) as u64)
}

/// Euler's totient function φ(n).
pub fn euler_totient(mut n: u64) -> u64 {
    let mut result = n;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 { n /= p; }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 { result -= result / n; }
    result
}

/// Check if n is prime (trial division, fine for moderate n).
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

/// Sieve of Eratosthenes: all primes up to n.
pub fn sieve_primes(n: usize) -> Vec<usize> {
    if n < 2 { return vec![]; }
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2;
    while p * p <= n {
        if is_prime[p] {
            let mut m = p * p;
            while m <= n { is_prime[m] = false; m += p; }
        }
        p += 1;
    }
    (2..=n).filter(|&i| is_prime[i]).collect()
}

/// Prime factorization of n. Returns list of (prime, exponent) pairs.
pub fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            let mut exp = 0u32;
            while n % d == 0 { n /= d; exp += 1; }
            factors.push((d, exp));
        }
        d += 1;
    }
    if n > 1 { factors.push((n, 1)); }
    factors
}

/// Count divisors of n.
pub fn count_divisors(n: u64) -> u64 {
    factorize(n).iter().map(|(_, e)| *e as u64 + 1).product()
}

/// Sum of divisors of n.
pub fn sum_divisors(n: u64) -> u64 {
    factorize(n).iter().map(|(p, e)| {
        (p.pow(*e + 1) - 1) / (p - 1)
    }).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert!((factorial(0) - 1.0).abs() < 1e-10);
        assert!((factorial(5) - 120.0).abs() < 1e-10);
        assert!((factorial(10) - 3628800.0).abs() < 1e-3);
    }

    #[test]
    fn test_binomial() {
        assert!((binomial(5, 2) - 10.0).abs() < 1e-10);
        assert!((binomial(10, 5) - 252.0).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial() {
        // 6!/(2!2!2!) = 90
        assert!((multinomial(&[2, 2, 2]) - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_permutations() {
        let perms = permutations(&[1, 2, 3]);
        assert_eq!(perms.len(), 6);
    }

    #[test]
    fn test_combinations() {
        let combs = combinations(5, 3);
        assert_eq!(combs.len(), 10);
    }

    #[test]
    fn test_power_set() {
        assert_eq!(power_set(3).len(), 8);
    }

    #[test]
    fn test_catalan() {
        assert!((catalan(5) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell() {
        assert!((bell(5) - 52.0).abs() < 1e-10);
    }

    #[test]
    fn test_stirling2() {
        // S(4,2) = 7
        assert!((stirling2(4, 2) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci() {
        assert!((fibonacci(10) - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_count() {
        // p(5) = 7
        assert!((partition_count(5) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_integer_partitions() {
        let parts = integer_partitions(4);
        assert_eq!(parts.len(), 5); // [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(lcm(4, 6), 12);
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 mod 1000
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_sieve() {
        let primes = sieve_primes(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_factorize() {
        let f = factorize(60);
        assert_eq!(f, vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(euler_totient(12), 4);
    }

    #[test]
    fn test_subfactorial() {
        // !4 = 9
        assert!((subfactorial(4) - 9.0).abs() < 1e-10);
    }
}
