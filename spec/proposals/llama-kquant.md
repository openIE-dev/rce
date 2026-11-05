# Proposal: `llama` lifting, rotary position embeddings and the K formats

Status: accepted (ledger row of 2026-11-04). Implemented in
`compute/crates/rce-vm-backend-cpu/src/crmath.rs` (`sin`, `cos`),
`compute/crates/rce-vm-backend-cpu/src/ml/kquant.rs` (the K-format
products, rounding operations and reference quantizer),
`compute/crates/rce-vm-backend-cpu/src/ml/nn.rs` (`rope`),
`spec/checker/src/graph.rs` and `spec/checker/src/typed.rs` (the new
operations) and `spec/checker/src/lift_gguf.rs` (`lift_llama`,
`rope_freqs`, the K weights), specified in 6.13 GT-4 and GT-10 and 10.17
LIFT-5 and LIFT-6 of the specification. Owner: David Charlot.

## 1. Terms

- *Super-block*: 256 consecutive values along a matrix's last axis.
- *Sub-block*: a part of a super-block with its own scale: 32 values for
  Q4_K and Q5_K, 16 for Q6_K.
- *K formats*: GGUF's Q4_K, Q5_K and Q6_K, which store per super-block the
  binary16 numbers d (and dmin) and per sub-block small integers sc (and
  m), so a value is d · sc · q − dmin · m.
- *Rotary position embedding* (RoPE): each adjacent pair of a head's
  elements turned by the angle position · frequency.
- *Grouped-query attention* (GQA): fewer key and value heads than query
  heads, each shared by a group of query heads.

## 2. The state before

10.17 lifted only `gpt2`, with F32, F16, Q8_0 and Q4_0 weights.
`compute/INTEGRATION.md` listed as left: architectures with rotary position
embeddings, which need `sin` and `cos` of `math.rce` in the backend, and
the formats Q4_K, Q5_K and Q6_K.

## 3. Decision

1. `crmath.rs` computes `math.sin` and `math.cos` as `math.rce` does,
   correctly rounded, so the backend has S(x) and C(x) of GT-4.
2. Seven operations join `std.tensor`: `rope` (exact),
   `matmul_q4_k`, `matmul_q5_k`, `matmul_q6_k` (exact), and
   `round_q4_k`, `round_q5_k`, `round_q6_k` (lossy, W-GRAPH-3).
3. A K-format matrix is held as its values (`i8`) and its per-sub-block
   scales and offsets as `f32` (GT-10). d · sc and dmin · m are exact in
   binary32 (an 11-bit significand times an integer below 2^8), so these
   blobs are the file's values exactly and a product needs no binary16.
4. The order of each K product is that of `rce-op-qmm`'s serial row
   kernels, so a host kernel can be judged against the reference bit for
   bit.
5. The rounding operations use a reference quantizer stated in GT-10,
   with `round` half away from zero. A lifted program never uses it.
6. `rope` takes its frequencies as an operand. The lift computes
   base^(−2i / hd) in binary64 with the platform's `powf` and rounds once
   to `f32`, and the result enters the program as a blob, so the program
   holds the bits and its hash covers them.
7. `rce lift gguf` lifts `llama` (LIFT-5) with GQA, SwiGLU and a tied or
   untied output.

## 4. Why

- One element order per product makes a K product exact like Q8_0 and
  Q4_0, and the order of `rce-op-qmm` is the one the hosts already run.
- `crmath.rs` has no `log` or `pow`, so the frequencies cannot be an
  operation of the term with the same bits everywhere; the lift is a tool,
  not a program, and its output is fixed by the blob digest. Another
  `powf` could give another last bit; the digest then differs and the
  program is another program, never the same program with other results.
- GGUF's own K quantizers search for scales and differ between versions;
  a rounding operation must be a function of its operand's bits, so RCE
  states its own, simpler quantizer and names it.
- The dispatched (vector) K kernels of `rce-op-qmm` reproduced every bit
  on the Apple M5 Max, but the test asserts only the serial kernels,
  since another processor's vector path may sum in another order.

## 5. Results

`spec/checker/tests/lift_llama.rs` builds a two-block `llama` model of
width 256, 4 query heads over 2 key and value heads, head length 64,
feed-forward width 512, vocabulary 32 and context 8 from the seed
`0x5eed_11a3`, with products in Q4_K, Q5_K, Q6_K, Q8_0 and F16
(`tests/support/toy_llama.rs`):

- the lifted program is admitted, and `rce test` and `rce trace` give the
  same bytes across runs;
- every K weight's held values equal `rce-ingest-gguf`'s
  `dequantize_to_f32` exactly;
- the logits equal a forward pass written in Rust bit for bit; the
  `rce-op-qmm` K products differ by 0 units in the last place, serial and
  dispatched, on 22,528 (Q4_K), 10,240 (Q5_K) and 8,704 (Q6_K) elements.

## 6. Not done here

- Other rotary layouts (the NeoX halves of `gptneox` and kin) and the
  scaled-frequency variants: they need their own lifts.
- Sequences shorter than the context, as for `gpt2`.
- Metal, MPS, AMX and CUDA K kernels: they need their hardware.
