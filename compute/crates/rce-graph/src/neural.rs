//! Neural network primitives: layers, activations, loss functions,
//! optimizers, forward/backward pass, simple MLP, convolution layers.

// ═══════════════════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// ReLU activation.
pub fn relu(x: f64) -> f64 { x.max(0.0) }

/// ReLU derivative.
pub fn relu_derivative(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }

/// Leaky ReLU.
pub fn leaky_relu(x: f64, alpha: f64) -> f64 { if x > 0.0 { x } else { alpha * x } }

/// Leaky ReLU derivative.
pub fn leaky_relu_derivative(x: f64, alpha: f64) -> f64 { if x > 0.0 { 1.0 } else { alpha } }

/// ELU activation.
pub fn elu(x: f64, alpha: f64) -> f64 { if x > 0.0 { x } else { alpha * (x.exp() - 1.0) } }

/// SELU activation.
pub fn selu(x: f64) -> f64 {
    let lambda = 1.0507009873554805;
    let alpha = 1.6732632423543772;
    lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// Sigmoid activation.
pub fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Sigmoid derivative (given output).
pub fn sigmoid_derivative(y: f64) -> f64 { y * (1.0 - y) }

/// Tanh activation.
pub fn tanh_activation(x: f64) -> f64 { x.tanh() }

/// Tanh derivative (given output).
pub fn tanh_derivative(y: f64) -> f64 { 1.0 - y * y }

/// Swish activation: x * sigmoid(x).
pub fn swish(x: f64) -> f64 { x * sigmoid(x) }

/// GELU activation (Gaussian Error Linear Unit).
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3)) / std::f64::consts::SQRT_2).tanh())
}

/// Softplus: ln(1 + exp(x)).
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Mish activation: x * tanh(softplus(x)).
pub fn mish(x: f64) -> f64 { x * softplus(x).tanh() }

/// Softmax on a vector (in-place safe).
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

/// Log-softmax.
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + x.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    x.iter().map(|v| v - lse).collect()
}

// ═══════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Mean Squared Error loss.
pub fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter().zip(target).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
}

/// MSE gradient.
pub fn mse_grad(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;
    pred.iter().zip(target).map(|(p, t)| 2.0 * (p - t) / n).collect()
}

/// Cross-entropy loss (binary).
pub fn binary_cross_entropy(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    pred.iter().zip(target).map(|(p, t)| {
        let p = p.max(1e-15).min(1.0 - 1e-15);
        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
    }).sum::<f64>() / n
}

/// Cross-entropy loss (multi-class, pred = softmax output).
pub fn categorical_cross_entropy(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter().zip(target).map(|(p, t)| {
        if *t > 0.0 { -t * p.max(1e-15).ln() } else { 0.0 }
    }).sum()
}

/// Huber loss (smooth L1).
pub fn huber_loss(pred: &[f64], target: &[f64], delta: f64) -> f64 {
    let n = pred.len() as f64;
    pred.iter().zip(target).map(|(p, t)| {
        let diff = (p - t).abs();
        if diff <= delta { 0.5 * diff * diff } else { delta * diff - 0.5 * delta * delta }
    }).sum::<f64>() / n
}

/// Hinge loss (for SVM-like classifiers, target in {-1, 1}).
pub fn hinge_loss(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    pred.iter().zip(target).map(|(p, t)| (1.0 - t * p).max(0.0)).sum::<f64>() / n
}

/// KL divergence: KL(p || q).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(pi, qi)| {
        if *pi > 1e-15 { pi * (pi / qi.max(1e-15)).ln() } else { 0.0 }
    }).sum()
}

// ═══════════════════════════════════════════════════════════════════
// DENSE LAYER
// ═══════════════════════════════════════════════════════════════════

/// Dense (fully connected) layer: y = W*x + b.
/// weights: out_dim × in_dim (row-major), bias: out_dim.
pub fn dense_forward(input: &[f64], weights: &[f64], bias: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {
        let mut sum = bias[i];
        for j in 0..in_dim {
            sum += weights[i * in_dim + j] * input[j];
        }
        output[i] = sum;
    }
    output
}

/// Dense layer backward pass. Returns (grad_input, grad_weights, grad_bias).
pub fn dense_backward(input: &[f64], grad_output: &[f64], weights: &[f64], in_dim: usize, out_dim: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // grad_input = W^T * grad_output
    let mut grad_input = vec![0.0; in_dim];
    for j in 0..in_dim {
        for i in 0..out_dim {
            grad_input[j] += weights[i * in_dim + j] * grad_output[i];
        }
    }
    // grad_weights = grad_output ⊗ input
    let mut grad_weights = vec![0.0; out_dim * in_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            grad_weights[i * in_dim + j] = grad_output[i] * input[j];
        }
    }
    // grad_bias = grad_output
    let grad_bias = grad_output.to_vec();
    (grad_input, grad_weights, grad_bias)
}

/// Batch normalization forward.
/// input: batch_size × dim. Returns (normalized, mean, var).
pub fn batch_norm_forward(input: &[f64], batch_size: usize, dim: usize, gamma: &[f64], beta: &[f64], eps: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0; dim];
    let mut var = vec![0.0; dim];
    for i in 0..batch_size {
        for j in 0..dim { mean[j] += input[i * dim + j]; }
    }
    for m in mean.iter_mut() { *m /= batch_size as f64; }
    for i in 0..batch_size {
        for j in 0..dim { var[j] += (input[i * dim + j] - mean[j]).powi(2); }
    }
    for v in var.iter_mut() { *v /= batch_size as f64; }

    let mut output = vec![0.0; batch_size * dim];
    for i in 0..batch_size {
        for j in 0..dim {
            let normalized = (input[i * dim + j] - mean[j]) / (var[j] + eps).sqrt();
            output[i * dim + j] = gamma[j] * normalized + beta[j];
        }
    }
    (output, mean, var)
}

/// Layer normalization.
pub fn layer_norm(input: &[f64], dim: usize, gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    let mean = input.iter().sum::<f64>() / dim as f64;
    let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
    let std = (var + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * (x - mean) / std + beta[i]).collect()
}

/// Dropout (training mode). Returns (output, mask).
pub fn dropout(input: &[f64], rate: f64, seed: u64) -> (Vec<f64>, Vec<bool>) {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut state); }
    let scale = 1.0 / (1.0 - rate);
    let mut output = Vec::with_capacity(input.len());
    let mut mask = Vec::with_capacity(input.len());
    for &v in input {
        let u = (prng_next(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        let keep = u >= rate;
        mask.push(keep);
        output.push(if keep { v * scale } else { 0.0 });
    }
    (output, mask)
}

// ═══════════════════════════════════════════════════════════════════
// 1D CONVOLUTION LAYER
// ═══════════════════════════════════════════════════════════════════

/// 1D convolution. input: length, kernel: k_len.
/// Returns output of length (input_len - k_len + 1).
pub fn conv1d(input: &[f64], kernel: &[f64], stride: usize) -> Vec<f64> {
    let k_len = kernel.len();
    if input.len() < k_len { return vec![]; }
    let mut output = Vec::new();
    let mut i = 0;
    while i + k_len <= input.len() {
        let sum: f64 = input[i..i + k_len].iter().zip(kernel).map(|(x, k)| x * k).sum();
        output.push(sum);
        i += stride;
    }
    output
}

/// Max pooling 1D.
pub fn max_pool1d(input: &[f64], pool_size: usize, stride: usize) -> Vec<f64> {
    let mut output = Vec::new();
    let mut i = 0;
    while i + pool_size <= input.len() {
        let max = input[i..i + pool_size].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        output.push(max);
        i += stride;
    }
    output
}

/// Average pooling 1D.
pub fn avg_pool1d(input: &[f64], pool_size: usize, stride: usize) -> Vec<f64> {
    let mut output = Vec::new();
    let mut i = 0;
    while i + pool_size <= input.len() {
        let avg = input[i..i + pool_size].iter().sum::<f64>() / pool_size as f64;
        output.push(avg);
        i += stride;
    }
    output
}

/// Global average pooling.
pub fn global_avg_pool(input: &[f64]) -> f64 {
    input.iter().sum::<f64>() / input.len() as f64
}

// ═══════════════════════════════════════════════════════════════════
// OPTIMIZERS
// ═══════════════════════════════════════════════════════════════════

/// SGD update: w = w - lr * grad.
pub fn sgd_update(weights: &mut [f64], grads: &[f64], lr: f64) {
    for (w, g) in weights.iter_mut().zip(grads) { *w -= lr * g; }
}

/// SGD with momentum.
pub fn sgd_momentum_update(weights: &mut [f64], grads: &[f64], velocity: &mut [f64], lr: f64, momentum: f64) {
    for i in 0..weights.len() {
        velocity[i] = momentum * velocity[i] - lr * grads[i];
        weights[i] += velocity[i];
    }
}

/// Adam optimizer step.
pub fn adam_update(weights: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64],
                   lr: f64, beta1: f64, beta2: f64, eps: f64, t: usize) {
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    for i in 0..weights.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// RMSProp update.
pub fn rmsprop_update(weights: &mut [f64], grads: &[f64], cache: &mut [f64], lr: f64, decay: f64, eps: f64) {
    for i in 0..weights.len() {
        cache[i] = decay * cache[i] + (1.0 - decay) * grads[i] * grads[i];
        weights[i] -= lr * grads[i] / (cache[i].sqrt() + eps);
    }
}

/// AdaGrad update.
pub fn adagrad_update(weights: &mut [f64], grads: &[f64], cache: &mut [f64], lr: f64, eps: f64) {
    for i in 0..weights.len() {
        cache[i] += grads[i] * grads[i];
        weights[i] -= lr * grads[i] / (cache[i].sqrt() + eps);
    }
}

// ═══════════════════════════════════════════════════════════════════
// WEIGHT INITIALIZATION
// ═══════════════════════════════════════════════════════════════════

/// Xavier/Glorot uniform initialization.
pub fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    random_uniform(fan_in * fan_out, -limit, limit, seed)
}

/// He/Kaiming normal initialization.
pub fn he_normal(fan_in: usize, n: usize, seed: u64) -> Vec<f64> {
    let std = (2.0 / fan_in as f64).sqrt();
    random_normal(n, 0.0, std, seed)
}

/// LeCun normal initialization.
pub fn lecun_normal(fan_in: usize, n: usize, seed: u64) -> Vec<f64> {
    let std = (1.0 / fan_in as f64).sqrt();
    random_normal(n, 0.0, std, seed)
}

// ═══════════════════════════════════════════════════════════════════
// LEARNING RATE SCHEDULES
// ═══════════════════════════════════════════════════════════════════

/// Step decay: lr = lr0 * drop^(epoch / step_size).
pub fn lr_step_decay(lr0: f64, drop: f64, epoch: usize, step_size: usize) -> f64 {
    lr0 * drop.powi((epoch / step_size) as i32)
}

/// Exponential decay: lr = lr0 * exp(-k * epoch).
pub fn lr_exp_decay(lr0: f64, k: f64, epoch: usize) -> f64 {
    lr0 * (-k * epoch as f64).exp()
}

/// Cosine annealing: lr oscillates with cosine schedule.
pub fn lr_cosine_annealing(lr0: f64, lr_min: f64, epoch: usize, t_max: usize) -> f64 {
    lr_min + (lr0 - lr_min) * (1.0 + (std::f64::consts::PI * epoch as f64 / t_max as f64).cos()) / 2.0
}

/// Warmup linear: linearly increase lr from 0 to lr0 over warmup_steps.
pub fn lr_warmup_linear(lr0: f64, step: usize, warmup_steps: usize) -> f64 {
    if step < warmup_steps { lr0 * step as f64 / warmup_steps as f64 } else { lr0 }
}

// ═══════════════════════════════════════════════════════════════════
// SIMPLE MLP (forward only)
// ═══════════════════════════════════════════════════════════════════

/// Multi-layer perceptron forward pass.
/// layer_weights: vec of (weights, bias) per layer. Activation: ReLU for hidden, none for output.
pub fn mlp_forward(input: &[f64], layers: &[(Vec<f64>, Vec<f64>, usize, usize)]) -> Vec<f64> {
    let mut x = input.to_vec();
    let n_layers = layers.len();
    for (li, (weights, bias, in_dim, out_dim)) in layers.iter().enumerate() {
        let z = dense_forward(&x, weights, bias, *in_dim, *out_dim);
        x = if li < n_layers - 1 {
            z.iter().map(|&v| relu(v)).collect()
        } else {
            z
        };
    }
    x
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

fn prng_next(state: &mut [u64; 4]) -> u64 {
    let result = (state[0].wrapping_add(state[3])).rotate_left(23).wrapping_add(state[0]);
    let t = state[1] << 17;
    state[2] ^= state[0]; state[3] ^= state[1]; state[1] ^= state[2]; state[0] ^= state[3];
    state[2] ^= t; state[3] = state[3].rotate_left(45);
    result
}

fn random_uniform(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut state); }
    (0..n).map(|_| {
        let u = (prng_next(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }).collect()
}

fn random_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    for _ in 0..20 { prng_next(&mut state); }
    (0..n).map(|_| {
        let u1 = ((prng_next(&mut state) >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        let u2 = (prng_next(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(2.0), 2.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_softmax() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!((mse_loss(&pred, &target)).abs() < 1e-10);
    }

    #[test]
    fn test_dense_forward() {
        // Identity-like: W = [[1,0],[0,1]], b = [0,0]
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0, 0.0];
        let out = dense_forward(&[3.0, 4.0], &w, &b, 2, 2);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn test_dense_backward() {
        let input = vec![1.0, 2.0];
        let grad_out = vec![1.0];
        let weights = vec![3.0, 4.0]; // 1×2
        let (gi, gw, gb) = dense_backward(&input, &grad_out, &weights, 2, 1);
        assert_eq!(gi, vec![3.0, 4.0]); // W^T * grad_out
        assert_eq!(gw, vec![1.0, 2.0]); // grad_out ⊗ input
        assert_eq!(gb, vec![1.0]);
    }

    #[test]
    fn test_conv1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0]; // edge detection
        let out = conv1d(&input, &kernel, 1);
        assert_eq!(out.len(), 3);
        assert!((out[0] - (-2.0)).abs() < 1e-10); // 1*1 + 2*0 + 3*(-1) = -2
    }

    #[test]
    fn test_max_pool() {
        let input = vec![1.0, 3.0, 2.0, 4.0, 1.0, 5.0];
        let out = max_pool1d(&input, 2, 2);
        assert_eq!(out, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_adam() {
        let mut w = vec![1.0, 2.0];
        let g = vec![0.1, 0.2];
        let mut m = vec![0.0, 0.0];
        let mut v = vec![0.0, 0.0];
        adam_update(&mut w, &g, &mut m, &mut v, 0.01, 0.9, 0.999, 1e-8, 1);
        assert!(w[0] < 1.0); // should decrease
    }

    #[test]
    fn test_xavier() {
        let w = xavier_uniform(100, 100, 42);
        assert_eq!(w.len(), 10000);
        let limit = (6.0 / 200.0f64).sqrt();
        assert!(w.iter().all(|&v| v >= -limit && v <= limit));
    }

    #[test]
    fn test_lr_cosine() {
        let lr = lr_cosine_annealing(0.01, 0.0001, 50, 100);
        assert!(lr > 0.0001 && lr < 0.01);
    }

    #[test]
    fn test_dropout() {
        let input = vec![1.0; 100];
        let (output, mask) = dropout(&input, 0.5, 42);
        let kept = mask.iter().filter(|&&m| m).count();
        assert!(kept > 20 && kept < 80); // roughly half
        // Kept values should be scaled by 1/(1-0.5) = 2
        for (i, &m) in mask.iter().enumerate() {
            if m { assert!((output[i] - 2.0).abs() < 1e-10); }
            else { assert_eq!(output[i], 0.0); }
        }
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(kl_divergence(&p, &q).abs() < 1e-10);
    }

    #[test]
    fn test_batch_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let (out, _, _) = batch_norm_forward(&input, 2, 2, &gamma, &beta, 1e-5);
        // Should be roughly centered at 0
        let mean: f64 = out.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 0.1);
    }
}
