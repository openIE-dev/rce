# Emission parity against the reference

| adapter | agree | disagree | n/a | unsupported | not run |
|---|---|---|---|---|---|
| emit-js | 1459 | 0 | 520 | 0 | 0 |
| emit-python | 1459 | 0 | 520 | 0 | 0 |
| emit-rust | 1459 | 0 | 520 | 0 | 0 |
| emit-scala | 1459 | 0 | 520 | 0 | 0 |
| lean-core | 210 | 1 | 0 | 1766 | 2 |
| wasm-playground | 419 | 16 | 0 | 1504 | 40 |

## Disagreements

- `lexing-parsing/targeted/r5-generic-call` lean-core: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-moved-in-loop-reject` wasm-playground: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-moved-on-one-path-reject` wasm-playground: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-moved-twice-reject` wasm-playground: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-send-after-close-reject` wasm-playground: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-use-after-move-reject` wasm-playground: reference pass, adapter fail
- `concurrency/harvested/w-ses-4-use-after-send-reject` wasm-playground: reference pass, adapter fail
- `evaluation-fuel/harvested/ok-7000-accept` wasm-playground: reference pass, adapter fail
- `evaluation-fuel/harvested/trap-10000-reject` wasm-playground: reference pass, adapter fail
- `numbers/targeted/n6-int-float-equality` wasm-playground: reference pass, adapter fail
- `numbers/targeted/w-ty-3-mixing` wasm-playground: reference pass, adapter fail
- `numbers/targeted/w-ty-3-narrowing` wasm-playground: reference pass, adapter fail
- `types-checking/targeted/w-mod-1-private` wasm-playground: reference pass, adapter fail
- `types-checking/targeted/w-mod-2-cycle` wasm-playground: reference pass, adapter fail
- `types-checking/targeted/w-mod-3-clash` wasm-playground: reference pass, adapter fail
- `types-checking/targeted/w-state-3-borrow-field` wasm-playground: reference pass, adapter fail
- `types-checking/targeted/w-trait-1-impl-mismatch` wasm-playground: reference pass, adapter fail
