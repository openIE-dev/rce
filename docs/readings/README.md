# Energy readings

Readings of a machine's package energy counter taken by RCE's own
toolchain, kept as they were written. Each one has the D.11 form
(Appendix D): a figure, the basis `reported` (an on-chip counter, not a
meter on the supply), and the provenance that says which counter was read,
over what interval and when, on which machine, inside which boundary, and
with how many wraparounds. `tests/host_energy.rs` in `spec/checker` checks
every file here by D.11 on each CI run.

Each figure is the package's: CPU, GPU and Neural Engine together, with
everything else the machine ran in the interval. No idle baseline is
subtracted (`idle_baseline_w` is 0). None of these runs was made on a quiet
machine, so a figure says what the counter read during the run, not what
the program alone drew.

## 2026-11-04: `host.Energy` entries of `rce host run energy`

[`2026-11-04-host-energy.jsonl`](2026-11-04-host-energy.jsonl) holds the
`host.Energy` lines of four logs, byte for byte as the host wrote them
(Appendix E.4j, HC-38), at commit `4cc4d423`. Each log is of
`spec/examples/01_counter.rce`:

```
rce host run energy spec/examples/01_counter.rce counter.d "Inc, Inc, Dec"
rce host run energy spec/examples/01_counter.rce long.d < inc.txt   # 2,000 lines `Inc`
```

`rce host replay` of each log reported `replay ok` with the live state
hash. The rest of each log is not kept, so the lines here carry their
`prev_hash` but cannot be replayed alone.

| Line | Machine | Run | Entries covered | Joules | Interval (s) | Joules ÷ interval (W) | Joules per input |
|---|---|---|---|---|---|---|---|
| 1 | MacBook Pro, Apple M5 Max (Mac17,6), macOS 27.2 | 3 inputs | 3 to 6 | 0.237353376 | 0.014753709 | 16.09 | |
| 2 | MacBook Pro, Apple M5 Max | 2,000 inputs | 3 to 4002 | 74.048002445 | 7.698080667 | 9.62 | 0.0370 |
| 3 | Mac Studio, Apple M4 Max (Mac16,9), macOS 27.2 | 3 inputs | 3 to 6 | 0.269747937 | 0.016153709 | 16.70 | |
| 4 | Mac Studio, Apple M4 Max | 2,000 inputs | 3 to 4002 | 113.130351864 | 13.291327959 | 8.51 | 0.0566 |

The last two columns are computed from the first figures and are not
readings. The run's operator recorded the load on each machine minutes
afterwards, which the files do not hold: load averages of 3.88, 3.53 and 3.10 on
the MacBook Pro, with another job at 74% CPU; 22.74, 10.38 and 6.68 on the
Mac Studio, just after a build, with three processes at 99%, 55% and 22%
CPU.

## 2026-11-03: a reading of `rce test measure`

[`2026-11-03-test-measure.json`](2026-11-03-test-measure.json) is the
reading that the IDE build's `rce test measure spec/examples/16_multi_region.rce`
printed on the MacBook Pro M5 Max, as a D.11 term: 0.01978864 J over
0.015071584 s while two examples ran. It is a standalone reading (D.11: a
test run's reading, which no receipt holds), kept here as printed.
