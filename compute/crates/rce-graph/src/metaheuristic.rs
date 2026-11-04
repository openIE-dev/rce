//! Metaheuristic optimization: genetic algorithm, simulated annealing,
//! particle swarm optimization, differential evolution, ant colony,
//! tabu search, hill climbing, random search.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// PRNG
// ═══════════════════════════════════════════════════════════════════

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

fn prng_normal(state: &mut [u64; 4]) -> f64 {
    let u1 = prng_f64(state).max(1e-15);
    let u2 = prng_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ═══════════════════════════════════════════════════════════════════
// GENETIC ALGORITHM
// ═══════════════════════════════════════════════════════════════════

/// Genetic algorithm for continuous optimization.
/// fitness: higher is better. Returns (best_solution, best_fitness).
pub fn genetic_algorithm(
    fitness: impl Fn(&[f64]) -> f64,
    dim: usize,
    bounds: &[(f64, f64)],
    pop_size: usize,
    n_generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);

    // Initialize population
    let mut pop: Vec<Vec<f64>> = (0..pop_size).map(|_| {
        (0..dim).map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0)).collect()
    }).collect();
    let mut fitnesses: Vec<f64> = pop.iter().map(|ind| fitness(ind)).collect();

    for _ in 0..n_generations {
        let mut new_pop = Vec::with_capacity(pop_size);

        // Elitism: keep best
        let best_idx = fitnesses.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        new_pop.push(pop[best_idx].clone());

        while new_pop.len() < pop_size {
            // Tournament selection
            let p1 = tournament_select(&fitnesses, 3, &mut state);
            let p2 = tournament_select(&fitnesses, 3, &mut state);

            // Crossover
            let mut child = if prng_f64(&mut state) < crossover_rate {
                let alpha = prng_f64(&mut state);
                (0..dim).map(|d| alpha * pop[p1][d] + (1.0 - alpha) * pop[p2][d]).collect()
            } else {
                pop[p1].clone()
            };

            // Mutation
            for d in 0..dim {
                if prng_f64(&mut state) < mutation_rate {
                    let range = bounds[d].1 - bounds[d].0;
                    child[d] += prng_normal(&mut state) * range * 0.1;
                    child[d] = child[d].max(bounds[d].0).min(bounds[d].1);
                }
            }
            new_pop.push(child);
        }

        pop = new_pop;
        fitnesses = pop.iter().map(|ind| fitness(ind)).collect();
    }

    let best_idx = fitnesses.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (pop[best_idx].clone(), fitnesses[best_idx])
}

fn tournament_select(fitnesses: &[f64], k: usize, state: &mut [u64; 4]) -> usize {
    let n = fitnesses.len();
    let mut best = prng_next(state) as usize % n;
    for _ in 1..k {
        let candidate = prng_next(state) as usize % n;
        if fitnesses[candidate] > fitnesses[best] { best = candidate; }
    }
    best
}

// ═══════════════════════════════════════════════════════════════════
// SIMULATED ANNEALING
// ═══════════════════════════════════════════════════════════════════

/// Simulated annealing for continuous optimization (minimization).
pub fn simulated_annealing(
    cost: impl Fn(&[f64]) -> f64,
    initial: &[f64],
    bounds: &[(f64, f64)],
    t_start: f64,
    t_end: f64,
    n_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);
    let dim = initial.len();
    let mut current = initial.to_vec();
    let mut current_cost = cost(&current);
    let mut best = current.clone();
    let mut best_cost = current_cost;

    for i in 0..n_iter {
        let t = t_start * (t_end / t_start).powf(i as f64 / n_iter as f64);

        // Generate neighbor
        let mut candidate = current.clone();
        for d in 0..dim {
            let range = bounds[d].1 - bounds[d].0;
            candidate[d] += prng_normal(&mut state) * range * 0.05 * (t / t_start);
            candidate[d] = candidate[d].max(bounds[d].0).min(bounds[d].1);
        }

        let candidate_cost = cost(&candidate);
        let delta = candidate_cost - current_cost;
        if delta < 0.0 || prng_f64(&mut state) < (-delta / t).exp() {
            current = candidate;
            current_cost = candidate_cost;
            if current_cost < best_cost {
                best = current.clone();
                best_cost = current_cost;
            }
        }
    }
    (best, best_cost)
}

// ═══════════════════════════════════════════════════════════════════
// PARTICLE SWARM OPTIMIZATION
// ═══════════════════════════════════════════════════════════════════

/// PSO for continuous optimization (minimization).
pub fn particle_swarm(
    cost: impl Fn(&[f64]) -> f64,
    dim: usize,
    bounds: &[(f64, f64)],
    n_particles: usize,
    n_iter: usize,
    w: f64,        // inertia
    c1: f64,       // cognitive
    c2: f64,       // social
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);

    let mut positions: Vec<Vec<f64>> = (0..n_particles).map(|_| {
        (0..dim).map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0)).collect()
    }).collect();
    let mut velocities: Vec<Vec<f64>> = (0..n_particles).map(|_| {
        (0..dim).map(|d| {
            let range = bounds[d].1 - bounds[d].0;
            (prng_f64(&mut state) - 0.5) * range * 0.1
        }).collect()
    }).collect();

    let mut p_best = positions.clone();
    let mut p_best_cost: Vec<f64> = positions.iter().map(|p| cost(p)).collect();
    let g_best_idx = p_best_cost.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let mut g_best = p_best[g_best_idx].clone();
    let mut g_best_cost = p_best_cost[g_best_idx];

    for _ in 0..n_iter {
        for i in 0..n_particles {
            for d in 0..dim {
                let r1 = prng_f64(&mut state);
                let r2 = prng_f64(&mut state);
                velocities[i][d] = w * velocities[i][d]
                    + c1 * r1 * (p_best[i][d] - positions[i][d])
                    + c2 * r2 * (g_best[d] - positions[i][d]);
                positions[i][d] += velocities[i][d];
                positions[i][d] = positions[i][d].max(bounds[d].0).min(bounds[d].1);
            }
            let c = cost(&positions[i]);
            if c < p_best_cost[i] {
                p_best_cost[i] = c;
                p_best[i] = positions[i].clone();
                if c < g_best_cost {
                    g_best_cost = c;
                    g_best = positions[i].clone();
                }
            }
        }
    }
    (g_best, g_best_cost)
}

// ═══════════════════════════════════════════════════════════════════
// DIFFERENTIAL EVOLUTION
// ═══════════════════════════════════════════════════════════════════

/// Differential Evolution (DE/rand/1/bin) for minimization.
pub fn differential_evolution(
    cost: impl Fn(&[f64]) -> f64,
    dim: usize,
    bounds: &[(f64, f64)],
    pop_size: usize,
    n_iter: usize,
    f_scale: f64,   // mutation scale (typically 0.5–1.0)
    cr: f64,         // crossover probability (typically 0.7–0.9)
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);

    let mut pop: Vec<Vec<f64>> = (0..pop_size).map(|_| {
        (0..dim).map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0)).collect()
    }).collect();
    let mut costs: Vec<f64> = pop.iter().map(|p| cost(p)).collect();

    for _ in 0..n_iter {
        for i in 0..pop_size {
            // Pick 3 distinct random indices ≠ i
            let (mut a, mut b, mut c) = (i, i, i);
            while a == i { a = prng_next(&mut state) as usize % pop_size; }
            while b == i || b == a { b = prng_next(&mut state) as usize % pop_size; }
            while c == i || c == a || c == b { c = prng_next(&mut state) as usize % pop_size; }

            let j_rand = prng_next(&mut state) as usize % dim;
            let trial: Vec<f64> = (0..dim).map(|d| {
                if d == j_rand || prng_f64(&mut state) < cr {
                    let v = pop[a][d] + f_scale * (pop[b][d] - pop[c][d]);
                    v.max(bounds[d].0).min(bounds[d].1)
                } else {
                    pop[i][d]
                }
            }).collect();

            let trial_cost = cost(&trial);
            if trial_cost <= costs[i] {
                pop[i] = trial;
                costs[i] = trial_cost;
            }
        }
    }

    let best_idx = costs.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (pop[best_idx].clone(), costs[best_idx])
}

// ═══════════════════════════════════════════════════════════════════
// HILL CLIMBING
// ═══════════════════════════════════════════════════════════════════

/// Stochastic hill climbing (minimization).
pub fn hill_climbing(
    cost: impl Fn(&[f64]) -> f64,
    initial: &[f64],
    bounds: &[(f64, f64)],
    step_size: f64,
    n_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);
    let dim = initial.len();
    let mut current = initial.to_vec();
    let mut current_cost = cost(&current);

    for _ in 0..n_iter {
        let mut candidate = current.clone();
        for d in 0..dim {
            candidate[d] += prng_normal(&mut state) * step_size;
            candidate[d] = candidate[d].max(bounds[d].0).min(bounds[d].1);
        }
        let c = cost(&candidate);
        if c < current_cost { current = candidate; current_cost = c; }
    }
    (current, current_cost)
}

/// Random restart hill climbing.
pub fn random_restart_hill_climbing(
    cost: impl Fn(&[f64]) -> f64,
    dim: usize,
    bounds: &[(f64, f64)],
    step_size: f64,
    n_restarts: usize,
    n_iter_per_restart: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);
    let mut best: Option<(Vec<f64>, f64)> = None;

    for _ in 0..n_restarts {
        let initial: Vec<f64> = (0..dim)
            .map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0))
            .collect();
        let s = prng_next(&mut state);
        let (sol, c) = hill_climbing(&cost, &initial, bounds, step_size, n_iter_per_restart, s);
        if best.is_none() || c < best.as_ref().unwrap().1 {
            best = Some((sol, c));
        }
    }
    best.unwrap()
}

// ═══════════════════════════════════════════════════════════════════
// RANDOM SEARCH
// ═══════════════════════════════════════════════════════════════════

/// Pure random search (minimization).
pub fn random_search(
    cost: impl Fn(&[f64]) -> f64,
    dim: usize,
    bounds: &[(f64, f64)],
    n_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let mut state = seed_state(seed);
    let mut best: Vec<f64> = (0..dim)
        .map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0))
        .collect();
    let mut best_cost = cost(&best);

    for _ in 1..n_iter {
        let candidate: Vec<f64> = (0..dim)
            .map(|d| bounds[d].0 + prng_f64(&mut state) * (bounds[d].1 - bounds[d].0))
            .collect();
        let c = cost(&candidate);
        if c < best_cost { best = candidate; best_cost = c; }
    }
    (best, best_cost)
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Sphere function: f(x) = Σ x_i². Global min at origin = 0.
pub fn sphere(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum() }

/// Rosenbrock function. Global min at (1,1,...,1) = 0.
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
}

/// Rastrigin function. Global min at origin = 0.
pub fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64 + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
}

/// Ackley function. Global min at origin ≈ 0.
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq = x.iter().map(|v| v * v).sum::<f64>();
    let sum_cos = x.iter().map(|v| (2.0 * PI * v).cos()).sum::<f64>();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Schwefel function. Global min at x_i = 420.9687... ≈ 0.
pub fn schwefel(x: &[f64]) -> f64 {
    418.9829 * x.len() as f64 - x.iter().map(|&xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
}

/// Griewank function. Global min at origin = 0.
pub fn griewank(x: &[f64]) -> f64 {
    let sum = x.iter().map(|v| v * v / 4000.0).sum::<f64>();
    let prod: f64 = x.iter().enumerate().map(|(i, &v)| (v / (i as f64 + 1.0).sqrt()).cos()).product();
    sum - prod + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_min() {
        assert!((sphere(&[0.0, 0.0])).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_min() {
        assert!((rosenbrock(&[1.0, 1.0])).abs() < 1e-10);
    }

    #[test]
    fn test_genetic_algorithm() {
        let (sol, fit) = genetic_algorithm(
            |x| -sphere(x), // negate for maximization
            2, &[(-5.0, 5.0), (-5.0, 5.0)],
            50, 100, 0.1, 0.8, 42,
        );
        assert!(sol[0].abs() < 1.0);
        assert!(sol[1].abs() < 1.0);
    }

    #[test]
    fn test_simulated_annealing() {
        let (sol, cost) = simulated_annealing(
            sphere, &[3.0, 4.0], &[(-5.0, 5.0), (-5.0, 5.0)],
            10.0, 0.01, 10000, 42,
        );
        assert!(cost < 1.0);
    }

    #[test]
    fn test_pso() {
        let (sol, cost) = particle_swarm(
            sphere, 2, &[(-5.0, 5.0), (-5.0, 5.0)],
            30, 100, 0.7, 1.5, 1.5, 42,
        );
        assert!(cost < 0.1);
    }

    #[test]
    fn test_differential_evolution() {
        let (sol, cost) = differential_evolution(
            sphere, 2, &[(-5.0, 5.0), (-5.0, 5.0)],
            30, 200, 0.8, 0.9, 42,
        );
        assert!(cost < 0.01);
    }

    #[test]
    fn test_hill_climbing() {
        let (sol, cost) = hill_climbing(
            sphere, &[3.0, 4.0], &[(-5.0, 5.0), (-5.0, 5.0)],
            0.1, 10000, 42,
        );
        assert!(cost < 1.0);
    }

    #[test]
    fn test_random_search() {
        let (_, cost) = random_search(
            sphere, 2, &[(-5.0, 5.0), (-5.0, 5.0)],
            10000, 42,
        );
        assert!(cost < 1.0);
    }

    #[test]
    fn test_rastrigin_at_origin() {
        assert!(rastrigin(&[0.0, 0.0]).abs() < 1e-10);
    }
}
