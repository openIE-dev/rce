//! Multi-layer LSTM forward — matches PyTorch `nn.LSTM(...).forward(x, (h0, c0))`
//! bit-close (within f32 noise).
//!
//! PyTorch convention (unidirectional, batch_first = false, bias = true):
//!
//!   For each layer L:
//!     i_t = σ(W_ii · x_t + b_ii + W_hi · h_{t-1} + b_hi)
//!     f_t = σ(W_if · x_t + b_if + W_hf · h_{t-1} + b_hf)
//!     g_t = tanh(W_ig · x_t + b_ig + W_hg · h_{t-1} + b_hg)
//!     o_t = σ(W_io · x_t + b_io + W_ho · h_{t-1} + b_ho)
//!     c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//!     h_t = o_t ⊙ tanh(c_t)
//!
//! Weights are stored fused as `weight_ih_l<L>` shape `[4·H, input_size]` and
//! `weight_hh_l<L>` shape `[4·H, H]`, with the row-blocks ordered (i, f, g, o).

#[derive(Clone, Debug)]
pub struct LstmLayer {
    /// `[4·H, input_size]`  (row blocks: i, f, g, o)
    pub weight_ih: Vec<f32>,
    /// `[4·H, H]`
    pub weight_hh: Vec<f32>,
    /// `[4·H]`
    pub bias_ih: Vec<f32>,
    /// `[4·H]`
    pub bias_hh: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct LstmConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Compute `y = W · x + b` where `W` is `[out_size, in_size]` row-major.
fn linear_into(out: &mut [f32], x: &[f32], w: &[f32], b: &[f32], in_size: usize) {
    debug_assert_eq!(x.len(), in_size);
    debug_assert_eq!(out.len(), b.len());
    let out_size = out.len();
    for o in 0..out_size {
        let mut acc = b[o];
        let wr = &w[o*in_size..(o+1)*in_size];
        for i in 0..in_size { acc += x[i] * wr[i]; }
        out[o] = acc;
    }
}

/// Forward pass.  `x` is laid out as `[T, input_size]` (batch_first = false, B = 1).
///
/// Returns `(outputs, hn, cn)`:
///   - `outputs` is `[T, H]` — the **top** layer's `h_t` at every time step.
///   - `hn` is `[num_layers, H]`.
///   - `cn` is `[num_layers, H]`.
pub fn lstm_forward(
    x: &[f32],
    t: usize,
    cfg: &LstmConfig,
    layers: &[LstmLayer],
    h0: Option<&[f32]>,
    c0: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(layers.len(), cfg.num_layers);
    assert_eq!(x.len(), t * cfg.input_size);

    let h = cfg.hidden_size;
    let four_h = 4 * h;

    let mut layer_h: Vec<Vec<f32>> = (0..cfg.num_layers).map(|i| {
        match h0 { Some(s) => s[i*h..(i+1)*h].to_vec(), None => vec![0.0; h] }
    }).collect();
    let mut layer_c: Vec<Vec<f32>> = (0..cfg.num_layers).map(|i| {
        match c0 { Some(s) => s[i*h..(i+1)*h].to_vec(), None => vec![0.0; h] }
    }).collect();

    let mut current_x: Vec<f32> = x.to_vec();
    let mut current_in = cfg.input_size;
    let mut layer_out = Vec::new();
    let mut gates_ih = vec![0.0f32; four_h];
    let mut gates_hh = vec![0.0f32; four_h];

    for li in 0..cfg.num_layers {
        let layer = &layers[li];
        assert_eq!(layer.weight_ih.len(), four_h * current_in);
        assert_eq!(layer.weight_hh.len(), four_h * h);
        assert_eq!(layer.bias_ih.len(), four_h);
        assert_eq!(layer.bias_hh.len(), four_h);

        layer_out = vec![0.0f32; t * h];

        // Aliases for h_prev / c_prev — we mutate them in place across time steps.
        let h_prev = &mut layer_h[li];
        let c_prev = &mut layer_c[li];

        for ti in 0..t {
            let xt = &current_x[ti*current_in..(ti+1)*current_in];
            linear_into(&mut gates_ih, xt, &layer.weight_ih, &layer.bias_ih, current_in);
            linear_into(&mut gates_hh, h_prev, &layer.weight_hh, &layer.bias_hh, h);

            // Compose i, f, g, o; update c_t and h_t in place.
            for k in 0..h {
                let i_pre = gates_ih[k]       + gates_hh[k];
                let f_pre = gates_ih[h + k]   + gates_hh[h + k];
                let g_pre = gates_ih[2*h + k] + gates_hh[2*h + k];
                let o_pre = gates_ih[3*h + k] + gates_hh[3*h + k];
                let i_g = sigmoid(i_pre);
                let f_g = sigmoid(f_pre);
                let g_g = g_pre.tanh();
                let o_g = sigmoid(o_pre);
                c_prev[k] = f_g * c_prev[k] + i_g * g_g;
                h_prev[k] = o_g * c_prev[k].tanh();
                layer_out[ti*h + k] = h_prev[k];
            }
        }

        current_x = std::mem::take(&mut layer_out);
        current_in = h;
    }

    let outputs = current_x;        // [T, H]  — top layer
    let mut hn = vec![0.0f32; cfg.num_layers * h];
    let mut cn = vec![0.0f32; cfg.num_layers * h];
    for i in 0..cfg.num_layers {
        hn[i*h..(i+1)*h].copy_from_slice(&layer_h[i]);
        cn[i*h..(i+1)*h].copy_from_slice(&layer_c[i]);
    }
    (outputs, hn, cn)
}

/// Incremental single-step forward — `x_t` shape `[input_size]`.
/// Mutates `layer_h` and `layer_c` in place (each shape `[num_layers, H]`).
/// Returns the top-layer `h_t` (length `H`).  Useful for autoregressive decoding.
pub fn lstm_step(
    x_t: &[f32],
    cfg: &LstmConfig,
    layers: &[LstmLayer],
    layer_h: &mut [f32],
    layer_c: &mut [f32],
) -> Vec<f32> {
    assert_eq!(layers.len(), cfg.num_layers);
    let h = cfg.hidden_size;
    let four_h = 4 * h;
    let mut current_x: Vec<f32> = x_t.to_vec();
    let mut current_in = cfg.input_size;
    let mut gates_ih = vec![0.0f32; four_h];
    let mut gates_hh = vec![0.0f32; four_h];

    for li in 0..cfg.num_layers {
        let layer = &layers[li];
        let h_prev = &mut layer_h[li*h..(li+1)*h];
        let c_prev = &mut layer_c[li*h..(li+1)*h];
        linear_into(&mut gates_ih, &current_x, &layer.weight_ih, &layer.bias_ih, current_in);
        linear_into(&mut gates_hh, h_prev, &layer.weight_hh, &layer.bias_hh, h);
        let mut h_new = vec![0.0f32; h];
        for k in 0..h {
            let i_g = sigmoid(gates_ih[k]       + gates_hh[k]);
            let f_g = sigmoid(gates_ih[h + k]   + gates_hh[h + k]);
            let g_g = (gates_ih[2*h + k] + gates_hh[2*h + k]).tanh();
            let o_g = sigmoid(gates_ih[3*h + k] + gates_hh[3*h + k]);
            c_prev[k] = f_g * c_prev[k] + i_g * g_g;
            h_new[k]  = o_g * c_prev[k].tanh();
        }
        h_prev.copy_from_slice(&h_new);
        current_x = h_new;
        current_in = h;
    }
    current_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn read_f32(name: &str) -> Option<Vec<f32>> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
        let bytes = std::fs::read(p).ok()?;
        Some(bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
    }
    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max)
    }
    fn mean_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x,y)| (x-y).abs()).sum::<f32>() / a.len() as f32
    }

    /// 2-layer LSTM, input=12, hidden=8, T=5 — matches PyTorch nn.LSTM bit-close.
    #[test]
    fn lstm_matches_pytorch_golden() {
        let Some(x) = read_f32("x.f32") else { eprintln!("skip: fixtures missing"); return; };
        let Some(h0) = read_f32("h0.f32") else { return; };
        let Some(c0) = read_f32("c0.f32") else { return; };
        let Some(golden_y) = read_f32("y.f32") else { return; };
        let Some(golden_hn) = read_f32("hn.f32") else { return; };
        let Some(golden_cn) = read_f32("cn.f32") else { return; };

        let cfg = LstmConfig { input_size: 12, hidden_size: 8, num_layers: 2 };
        let layers = vec![
            LstmLayer {
                weight_ih: read_f32("weight_ih_l0.f32").unwrap(),
                weight_hh: read_f32("weight_hh_l0.f32").unwrap(),
                bias_ih:   read_f32("bias_ih_l0.f32").unwrap(),
                bias_hh:   read_f32("bias_hh_l0.f32").unwrap(),
            },
            LstmLayer {
                weight_ih: read_f32("weight_ih_l1.f32").unwrap(),
                weight_hh: read_f32("weight_hh_l1.f32").unwrap(),
                bias_ih:   read_f32("bias_ih_l1.f32").unwrap(),
                bias_hh:   read_f32("bias_hh_l1.f32").unwrap(),
            },
        ];

        let (y, hn, cn) = lstm_forward(&x, 5, &cfg, &layers, Some(&h0), Some(&c0));
        let dy = max_diff(&y,  &golden_y);
        let dh = max_diff(&hn, &golden_hn);
        let dc = max_diff(&cn, &golden_cn);
        let my = mean_diff(&y,  &golden_y);
        eprintln!("y  max_abs_diff = {dy:.4e}, mean = {my:.4e}");
        eprintln!("hn max_abs_diff = {dh:.4e}");
        eprintln!("cn max_abs_diff = {dc:.4e}");
        assert!(dy < 1e-5, "y drift {dy}");
        assert!(dh < 1e-5, "hn drift {dh}");
        assert!(dc < 1e-5, "cn drift {dc}");
    }

    /// `lstm_step` produces the same output as `lstm_forward` when called
    /// repeatedly with the running (h, c) state.
    #[test]
    fn lstm_step_matches_lstm_forward() {
        let Some(x) = read_f32("x.f32") else { eprintln!("skip"); return; };
        let Some(h0) = read_f32("h0.f32") else { return; };
        let Some(c0) = read_f32("c0.f32") else { return; };
        let Some(golden_y) = read_f32("y.f32") else { return; };

        let cfg = LstmConfig { input_size: 12, hidden_size: 8, num_layers: 2 };
        let layers = vec![
            LstmLayer {
                weight_ih: read_f32("weight_ih_l0.f32").unwrap(),
                weight_hh: read_f32("weight_hh_l0.f32").unwrap(),
                bias_ih:   read_f32("bias_ih_l0.f32").unwrap(),
                bias_hh:   read_f32("bias_hh_l0.f32").unwrap(),
            },
            LstmLayer {
                weight_ih: read_f32("weight_ih_l1.f32").unwrap(),
                weight_hh: read_f32("weight_hh_l1.f32").unwrap(),
                bias_ih:   read_f32("bias_ih_l1.f32").unwrap(),
                bias_hh:   read_f32("bias_hh_l1.f32").unwrap(),
            },
        ];

        let mut hl = h0.clone();
        let mut cl = c0.clone();
        let mut y_streaming = vec![0.0f32; 5 * 8];
        for ti in 0..5 {
            let xt = &x[ti*12..(ti+1)*12];
            let ht = lstm_step(xt, &cfg, &layers, &mut hl, &mut cl);
            y_streaming[ti*8..(ti+1)*8].copy_from_slice(&ht);
        }
        let d = max_diff(&y_streaming, &golden_y);
        eprintln!("streaming vs forward: max_abs_diff = {d:.4e}");
        assert!(d < 1e-5, "streaming drift {d}");
    }
}
