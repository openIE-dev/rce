# RCE examples

Programs written in RCE, grouped by what they are for: simulated worlds,
control of physical devices, media, networks, text, data, money, science,
systems and hardware. Each program is small enough to read in one sitting,
carries its own tests, and is checked by the RCE reference checker.

In plain words:

1. **Each folder is one kind of problem.** Pick the area closest to what
   you want to build and start from its programs.
2. **Each program tests itself.** The `example` and `check` lines inside
   a program say what it must do, and the checker refuses the program if
   they do not hold.
3. **The limits are written down.** Where RCE cannot yet express something
   an area needs, the folder's README says so plainly.

## Quickstart

Build `rce` from the [rce repository](https://github.com/openIE-dev/rce),
then:

```sh
git clone https://github.com/openIE-dev/rce-examples
cd rce-examples
rce test money/account.rce
rce run systems/ring_buffer.rce "Push(v: 7), Push(v: 8), Pop, Pop, Pop"
```

To check every program in the repository:

```sh
RCE=/path/to/rce scripts/check.sh
```

## Areas

| Folder | Programs | What they show |
|---|---|---|
| [world/](world/) | `cellular_automaton.rce`, `queue_sim.rce` | discrete simulation as a state that changes on each event |
| [body/](body/) | `thermostat.rce`, `arm.rce` | control loops that react to sensor readings and refuse impossible ones |
| [media/](media/) | `gain.rce`, `moving_average.rce` | fixed-size buffers, lane-wise arithmetic, clipping, filters |
| [network/](network/) | `handshake.rce`, `server.rce` | a protocol as a state machine; a server as an actor |
| [text/](text/) | `word_count.rce`, `edit_distance.rce` | a byte-by-byte lexer; dynamic programming over text |
| [data/](data/) | `totals.rce` | records, totals and a group-by |
| [money/](money/) | `account.rce`, `ledger.rce` | balances in integer cents; double-entry bookkeeping with an invariant |
| [science/](science/) | `numerics.rce`, `knapsack.rce` | dot product, mean, Newton's method; dynamic programming |
| [systems/](systems/) | `ring_buffer.rce`, `reconcile.rce` | a ring buffer that refuses to overwrite; a reconcile loop |
| [hardware/](hardware/) | `bits.rce` | bit counting, bit fields, a CRC-32 table built while checking |

## Areas without examples yet

Some kinds of program have no example here, because draft 0.5 of RCE has
no construct for them:

- **Machine learning.** There is no tensor type and no automatic
  differentiation.
- **Logic and proof.** There is no logic or constraint programming in the
  language. (The language itself is modelled in Lean, in the `rce`
  repository, but that is a proof *about* RCE, not a feature *of* it.)
- **Quantum computing.** There is no construct for qubits or circuits.

## How the checks work

`scripts/check.sh` applies four rules: a program outside a `refused/`
folder must pass `rce test`; a program in `refused/` must be refused with
the text in its `.expect` file; a `.cmd` file holds a command whose
output must equal the `.out` file next to it; and a folder with a
`program.txt` is tested as one program.

All programs follow draft 0.5 of the specification.

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
