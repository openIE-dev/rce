//! WAT kernel templates emitted as private helpers in every output
//! module. Mirrors the helper functions of the retired graph benchmark
//! WAT (matmul, relu, add_bias, softmax_row, fast_exp) so the
//! emitted module exhibits the same numerical behaviour as the
//! rce-cpu interpreter.
//!
//! All kernels operate on f32 row-major contiguous buffers in linear
//! memory. Pointers are byte offsets; lengths are passed in as i32
//! params so the same kernel handles every shape.

/// Schraudolph fast f32 exp with Β [-80, 80] clamp — same trick used
/// by the bench's `BENCH_WAT` so emitted modules match the bench
/// numerically. ~3% max error; matmul-dominated workloads don't
/// notice.
pub const FAST_EXP_WAT: &str = r#"
  (func $fast_exp (param $x f32) (result f32)
    (local $cx f32)
    (local.set $cx
      (f32.min (f32.const 80.0)
        (f32.max (f32.const -80.0) (local.get $x))))
    (f32.reinterpret_i32
      (i32.trunc_f32_s
        (f32.add
          (f32.mul (local.get $cx) (f32.const 12102203.0))
          (f32.const 1064866805.0)))))
"#;

/// 2-D row-major matmul into a caller-provided output region.
/// `matmul(a_off, b_off, out_off, m, k, n)`.
pub const MATMUL_WAT: &str = r#"
  (func $matmul
        (param $a i32) (param $b i32) (param $out i32)
        (param $m i32) (param $k i32) (param $n i32)
    (local $i i32) (local $j i32) (local $kk i32) (local $acc f32)
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_next
        (br_if $i_done (i32.ge_u (local.get $i) (local.get $m)))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_next
            (br_if $j_done (i32.ge_u (local.get $j) (local.get $n)))
            (local.set $acc (f32.const 0))
            (local.set $kk (i32.const 0))
            (block $kk_done
              (loop $kk_next
                (br_if $kk_done (i32.ge_u (local.get $kk) (local.get $k)))
                (local.set $acc
                  (f32.add (local.get $acc)
                    (f32.mul
                      (f32.load offset=0
                        (i32.add (local.get $a)
                          (i32.shl
                            (i32.add (i32.mul (local.get $i) (local.get $k))
                                     (local.get $kk))
                            (i32.const 2))))
                      (f32.load offset=0
                        (i32.add (local.get $b)
                          (i32.shl
                            (i32.add (i32.mul (local.get $kk) (local.get $n))
                                     (local.get $j))
                            (i32.const 2)))))))
                (local.set $kk (i32.add (local.get $kk) (i32.const 1)))
                (br $kk_next)))
            (f32.store offset=0
              (i32.add (local.get $out)
                (i32.shl
                  (i32.add (i32.mul (local.get $i) (local.get $n))
                           (local.get $j))
                  (i32.const 2)))
              (local.get $acc))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_next)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_next))))
"#;

/// Element-wise binary op into a caller-provided output region.
/// `binop_<op>(a_off, b_off, out_off, n_elements)`. Same shape
/// required for `a` and `b`; broadcast-add is handled by the dedicated
/// `add_bias` kernel below.
pub const ELEMWISE_ADD_WAT: &str = r#"
  (func $elemwise_add (param $a i32) (param $b i32) (param $out i32) (param $n i32)
    (local $i i32) (local $off i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $off (i32.shl (local.get $i) (i32.const 2)))
        (f32.store offset=0
          (i32.add (local.get $out) (local.get $off))
          (f32.add
            (f32.load offset=0 (i32.add (local.get $a) (local.get $off)))
            (f32.load offset=0 (i32.add (local.get $b) (local.get $off)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next))))
"#;

pub const ELEMWISE_MUL_WAT: &str = r#"
  (func $elemwise_mul (param $a i32) (param $b i32) (param $out i32) (param $n i32)
    (local $i i32) (local $off i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $off (i32.shl (local.get $i) (i32.const 2)))
        (f32.store offset=0
          (i32.add (local.get $out) (local.get $off))
          (f32.mul
            (f32.load offset=0 (i32.add (local.get $a) (local.get $off)))
            (f32.load offset=0 (i32.add (local.get $b) (local.get $off)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next))))
"#;

/// 2-D bias add: `out[i, j] = a[i, j] + bias[j]`. `a` is (rows, cols);
/// `bias` is (cols,). Output written to `out` in row-major (rows, cols).
pub const ADD_BIAS_WAT: &str = r#"
  (func $add_bias
        (param $a i32) (param $bias i32) (param $out i32)
        (param $rows i32) (param $cols i32)
    (local $i i32) (local $j i32) (local $a_off i32) (local $b_off i32) (local $o_off i32)
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_next
        (br_if $i_done (i32.ge_u (local.get $i) (local.get $rows)))
        (local.set $j (i32.const 0))
        (block $j_done
          (loop $j_next
            (br_if $j_done (i32.ge_u (local.get $j) (local.get $cols)))
            (local.set $a_off
              (i32.shl
                (i32.add (i32.mul (local.get $i) (local.get $cols))
                         (local.get $j))
                (i32.const 2)))
            (local.set $b_off (i32.shl (local.get $j) (i32.const 2)))
            (local.set $o_off (local.get $a_off))
            (f32.store offset=0
              (i32.add (local.get $out) (local.get $o_off))
              (f32.add
                (f32.load offset=0 (i32.add (local.get $a) (local.get $a_off)))
                (f32.load offset=0 (i32.add (local.get $bias) (local.get $b_off)))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_next)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_next))))
"#;

/// Element-wise relu into a caller-provided output region. (Output
/// can alias input for in-place; the loop is single-pass.)
pub const RELU_WAT: &str = r#"
  (func $relu (param $a i32) (param $out i32) (param $n i32)
    (local $i i32) (local $off i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $off (i32.shl (local.get $i) (i32.const 2)))
        (f32.store offset=0
          (i32.add (local.get $out) (local.get $off))
          (f32.max
            (f32.load offset=0 (i32.add (local.get $a) (local.get $off)))
            (f32.const 0)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next))))
"#;

/// Per-row softmax. Output written to `out`; (rows, cols) shape.
/// Numerically stable: subtract max, exp via `$fast_exp`, divide by sum.
pub const SOFTMAX_ROW_WAT: &str = r#"
  (func $softmax_row
        (param $a i32) (param $out i32)
        (param $rows i32) (param $cols i32)
    (local $i i32) (local $j i32)
    (local $a_row i32) (local $o_row i32)
    (local $max f32) (local $sum f32) (local $v f32) (local $e f32)
    (local $a_off i32) (local $o_off i32)
    (local.set $i (i32.const 0))
    (block $i_done
      (loop $i_next
        (br_if $i_done (i32.ge_u (local.get $i) (local.get $rows)))
        (local.set $a_row
          (i32.add (local.get $a)
            (i32.shl (i32.mul (local.get $i) (local.get $cols)) (i32.const 2))))
        (local.set $o_row
          (i32.add (local.get $out)
            (i32.shl (i32.mul (local.get $i) (local.get $cols)) (i32.const 2))))
        ;; pass 1: max
        (local.set $max (f32.load offset=0 (local.get $a_row)))
        (local.set $j (i32.const 1))
        (block $max_done
          (loop $max_next
            (br_if $max_done (i32.ge_u (local.get $j) (local.get $cols)))
            (local.set $v
              (f32.load offset=0
                (i32.add (local.get $a_row)
                         (i32.shl (local.get $j) (i32.const 2)))))
            (local.set $max (f32.max (local.get $max) (local.get $v)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $max_next)))
        ;; pass 2: exp + sum + write
        (local.set $sum (f32.const 0))
        (local.set $j (i32.const 0))
        (block $exp_done
          (loop $exp_next
            (br_if $exp_done (i32.ge_u (local.get $j) (local.get $cols)))
            (local.set $a_off
              (i32.add (local.get $a_row) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $o_off
              (i32.add (local.get $o_row) (i32.shl (local.get $j) (i32.const 2))))
            (local.set $v (f32.load offset=0 (local.get $a_off)))
            (local.set $e
              (call $fast_exp (f32.sub (local.get $v) (local.get $max))))
            (local.set $sum (f32.add (local.get $sum) (local.get $e)))
            (f32.store offset=0 (local.get $o_off) (local.get $e))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $exp_next)))
        ;; pass 3: divide
        (local.set $j (i32.const 0))
        (block $div_done
          (loop $div_next
            (br_if $div_done (i32.ge_u (local.get $j) (local.get $cols)))
            (local.set $o_off
              (i32.add (local.get $o_row) (i32.shl (local.get $j) (i32.const 2))))
            (f32.store offset=0 (local.get $o_off)
              (f32.div (f32.load offset=0 (local.get $o_off)) (local.get $sum)))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $div_next)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_next))))
"#;
