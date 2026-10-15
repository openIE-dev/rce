# Emission parity against the reference

| adapter | agree | disagree | unsupported | not run |
|---|---|---|---|---|
| emit-js | 1241 | 1 | 603 | 0 |
| emit-python | 1241 | 1 | 603 | 0 |
| wasm-playground | 239 | 2 | 1254 | 350 |

## Disagreements

- `forms-log/targeted/s1-guard-trap-refuses` emit-js: reference pass, adapter fail
- `forms-log/targeted/s1-guard-trap-refuses` emit-python: reference pass, adapter fail
- `evaluation-fuel/harvested/ok-7000-accept` wasm-playground: reference pass, adapter fail
- `evaluation-fuel/harvested/trap-10000-reject` wasm-playground: reference pass, adapter fail
