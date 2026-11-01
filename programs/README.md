# RCE examples

Programs written in RCE, grouped by what they are for: simulated worlds,
control of physical devices, media, networks, text, data, money, science,
systems and hardware. Each program is small enough to read in one sitting,
carries its own tests, and is checked by the RCE reference checker.

In plain words:

1. **Each folder is one kind of problem.** Pick the area closest to what
   you want to build and start from its programs.
2. **Each program tests itself.** The `example` and `check` lines inside
   a program say what it must do, and the program requires change if
   they do not hold.
3. **The limits are written down.** Where RCE cannot yet express something
   an area needs, the folder's README says so plainly.

## Quickstart

Install `rce` from the release of the newest `main` commit (Linux, macOS on
Apple silicon, or Windows with Git Bash; the script checks the archive
against the release's SHA256SUMS), or build it from source as the
[rce README](../README.md) describes, then:

```sh
curl -fsSL https://raw.githubusercontent.com/openIE-dev/rce/main/scripts/install.sh | sh
git clone https://github.com/openIE-dev/rce
cd rce/programs
rce test money/account.rce
rce run systems/ring_buffer.rce "Push(v: 7), Push(v: 8), Pop, Pop, Pop"
```

To check a program without installing anything, paste it into the
[RCE playground](https://openie-dev.github.io/rce/).

To check every program in the repository:

```sh
RCE=/path/to/rce scripts/check.sh
```

## Areas

| Folder | Programs | What they show |
|---|---|---|
| [world/](world/) | `cellular_automaton.rce`, `queue_sim.rce` | discrete simulation as a state that changes on each event |
| [body/](body/) | `thermostat.rce`, `arm.rce` | control loops that react to sensor readings and keep the state on impossible ones |
| [media/](media/) | `gain.rce`, `moving_average.rce` | fixed-size buffers, lane-wise arithmetic, clipping, filters |
| [network/](network/) | `handshake.rce`, `server.rce`, `notes_http.rce` | a protocol as a state machine; a server as an actor; a JSON service on the HTTP event host |
| [text/](text/) | `word_count.rce`, `edit_distance.rce`, `json_nest.rce` | a byte-by-byte lexer; dynamic programming over text; JSON of any depth parsed by loops over the input |
| [data/](data/) | `totals.rce` | records, totals and a group-by |
| [money/](money/) | `account.rce`, `ledger.rce`, `checkout.rce` | balances in integer cents; double-entry bookkeeping with an invariant; a checkout of 40 independent options |
| [science/](science/) | `numerics.rce`, `knapsack.rce`, `graph.rce`, `dpll.rce` | dot product, mean, Newton's method; dynamic programming; shortest paths over an input graph; a SAT solver and the bound it has to state |
| [systems/](systems/) | `ring_buffer.rce`, `reconcile.rce`, `accounts.rce`, `ls.rce` | a ring buffer that never overwrites; a reconcile loop; a long-running service on the host, one bounded step per event; `ls` brought in from an existing Go program, compared byte for byte with coreutils |
| [hardware/](hardware/) | `bits.rce` | bit counting, bit fields, a CRC-32 table built while checking |
| [multi/](multi/) | `elevator.rce`, `vending.rce`, `crossing.rce`, `parking.rce`, `printer.rce`, `lobby.rce`, `card.rce`, `washer.rce`, `session.rce`, `player.rce`, `climate.rce`, `reservoir.rce` | twelve devices whose state is split over two regions the same events drive; every one emits or keeps the state, and every one is explored completely by `rce dim` |

## Areas without examples yet

Some kinds of program have no example here, because the RCE language of
the current specification draft has no construct for them:

- **Machine learning.** There is no tensor type and no automatic
  differentiation.
- **Logic and proof.** There is no logic or constraint programming in the
  language. (The language itself is modelled in Lean, in the `rce`
  repository, but that is a proof *about* RCE, not a feature *of* it.)
- **Quantum computing.** There is no construct for qubits or circuits.

## How the checks work

`scripts/check.sh` applies four rules: a program outside a `change/`
folder must pass `rce test`; a program in `change/` must require change,
with the text of its `.expect` file in the report; a `.cmd` file holds a command whose
output must equal the `.out` file next to it; and a folder with a
`program.txt` is tested as one program.

All programs are in the one language of the current specification draft
(spec 2.7: a `# rce-version:` label selects nothing). They are checked by `programs.yml` against the reference checker built from
the same commit, so there is no commit to name: `scripts/check.sh` must report
every program passed at the current head (43 checks, 0 failed, on 1 November 2026). The
`ecosystem/` programs import modules of the standard library (`use hpke`,
`use pq`, `use mls`, `use crypto`), which the reference checker has built in
(specification 4.1); `scripts/check.sh` leaves them out, and `programs.yml`
runs them with `ecosystem/run.sh`, which also checks their results.

## Contributing

New examples are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Everyone
taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2026 Open Interface Engineering, Inc.

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in this work shall be dual licensed
as above, without any additional terms or conditions.
