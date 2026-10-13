# Fuzz results

Run of 100,000 programs, 2026-10-12: 8 parts × 12,500, seeds 1001–1008,
`python3 fuzz/gen.py 12500 <seed> results/s<seed>.json` (per-part counts in
`results/`).

| | count |
|---|---|
| programs | 100,000 |
| parsed by LALR(1) | 100,000 |
| exactly one Earley parse (explicit ambiguity) | 100,000 |
| accepted by `rce-check` | 100,000 |
| rule near-misses refused | 516,012 / 516,012 |
| syntax near-misses refused (by grammar and `rce-check`) | 392,362 / 392,362 |
| failures | 0 |
