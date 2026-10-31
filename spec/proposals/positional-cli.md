# Proposal: positional command lines (no flags)

Status: proposed, 2026-10-31. Design only: no code changes in this
proposal. The implementation is one later change, made after the open pull
requests that touch the same files have merged (section 4).

AGENTS.md already states the rule: "Commands take positional arguments
only." The tools do not follow it yet. This proposal lists every place
where they do not (section 1), gives one grammar for every command
(section 2), says what output changes (section 3), and plans the change
that removes the flags (section 4). Section 5 lists the questions the
owner has to answer.

## 0. Terms

- **Argument**: one element of a program's argument vector, as the shell
  passes it.
- **Flag**: an argument that starts with `-` or `--` and selects a mode,
  sets a bound or names a value that comes after it (`--json`,
  `--fuel 1000`, `--fuel=1000`, `-V`). This proposal removes all flags.
- **Command path**: the fixed words that name a command: `rce check`,
  `rce host serve`, `rce-book build`.
- **Operand**: what a command acts on: a file, a directory, a log D, an
  event list, a manifest, an address the command acts on.
- **Word**: an argument from a fixed list that belongs to one command. A
  word is lowercase ASCII letters, digits and hyphens, and starts with a
  letter (`json`, `fuel`, `fuel-trace`, `http`). A **mode word** takes no
  value (`json`, `control`, `check`, `pty`). A **setting word** takes
  exactly one value, the next argument (`fuel 1000`, `fs DIR`,
  `http 0.0.0.0:8080`).
- **Emitted program**: a program that `rce emit rust|python|js|scala FILE`
  prints. It has a command line of its own, read by the runtime that
  `spec/checker/src/emit/rt.*` holds.
- **Caller**: any file that runs a binary with arguments (workflow,
  script, test, listing, Dockerfile), or a document that tells a reader
  what to type.

## 1. Inventory

Counted on `main` at 0c015c43, and on `origin/claude/book-0.9` (pull
request #81, which rewrites `book/`) for `book/`. A count is the number of
times the flag's text occurs in tracked files, attributed to the binary it
belongs to. Uses that belong to other programs were left out by hand:
`lake env lean --run`, `unshare --net`, `tree-sitter build --wasm`,
`gh release list --json`, `cargo install ... --version`, and `--version`
given to rustc, node, python3, scala-cli, mdbook, lean or wasmtime. Cargo's
own flags (`--release`, `--bin`, `--features tree-sitter`) are cargo's,
not RCE's, and stay.

Columns: **src** the definition, its parsing and the messages that name it
(`spec/checker/src`); **tests** `spec/checker/tests`; **run** callers that
execute it (workflows, `scripts/`, `stdlib/scripts/`, `programs/*.sh`,
`conformance/` adapters and tools, `book/listings/*.cmd`, `deploy/`,
code under `ide/`); **text** prose and comments (`spec/book`, `book/src`,
`README.md`, `AGENTS.md`, `docs/`, `spec/README.md`, Markdown and `.rce`
comments in `stdlib/`, `programs/`, `conformance/`, `spec/lean`); **gen**
generated files (`spec/RCE-SPEC.md`, `spec/data/*.json`), regenerated,
never edited; **prop** `spec/proposals` other than the ledger; **#81** the
book on pull request #81.

### 1.1 `rce` (`spec/checker/src/cli.rs`, `src/bin/rce.rs`)

Parsed in `cli::run_main` and `host_cmd`; listed in `USAGE_TEXT`.

| Flag | Command | src | tests | run | text | gen | prop | total on main | #81 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `--json` | `trace` (required), `dim`, `equiv`, `fingerprint` | 15 | 22 | 10 | 15 | 7 | 4 | 73 | 4 |
| `--streams PATH` | `trace` | 4 | 1 | 4 | 1 | 1 | 1 | 12 | 0 |
| `--check` | `fmt` | 2 | 2 | 0 | 5 | 2 | 2 | 13 | 2 |
| `--nameless` | `fmt` | 4 | 16 | 6 | 7 | 4 | 0 | 37 | 1 |
| `--fuel N` | `dim`, `equiv`, `fingerprint`, `change` | 21 | 14 | 4 | 15 | 6 | 3 | 63 | 17 |
| `--control` | `dim` | 7 | 6 | 1 | 9 | 5 | 1 | 29 | 5 |
| `--depth N` | `fingerprint` | 6 | 4 | 0 | 1 | 0 | 1 | 12 | 1 |
| `--vectors N` | `index` | 5 | 1 | 0 | 4 | 3 | 0 | 13 | 0 |
| `--advisories FILE` | `manifest-check` | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| `--key ISSUER=HEX` | `manifest-check` | 4 | 0 | 0 | 1 | 1 | 0 | 6 | 0 |
| `--log D` | `host run`, `serve`, `tls-id`, `replay` (required) | 9 | 33 | 19 | 10 | 1 | 0 | 72 | 9 |
| `--fs DIR` | `host run`, `serve` | 4 | 1 | 9 | 7 | 0 | 0 | 21 | 0 |
| `--blob DIR` | `host run`, `serve` | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| `--media DIR` | `host run`, `serve` | 3 | 0 | 0 | 5 | 0 | 0 | 8 | 3 |
| `--wasm DIR` | `host run`, `serve` | 4 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| `--net HOST:PORT` (repeatable) | `host run`, `serve` | 6 | 7 | 0 | 5 | 3 | 0 | 21 | 2 |
| `--chan HOST:PORT` | `host run`, `serve` | 7 | 1 | 3 | 2 | 1 | 0 | 14 | 1 |
| `--tls` | `host run`, `serve` | 6 | 2 | 0 | 2 | 0 | 0 | 10 | 1 |
| `--peer HOST:PORT[=KEY]` (repeatable) | `host run`, `serve` | 9 | 1 | 3 | 2 | 1 | 0 | 16 | 1 |
| `--pty` | `host run`, `serve` | 7 | 6 | 1 | 0 | 0 | 0 | 14 | 0 |
| `--http HOST:PORT` | `host serve` | 16 | 10 | 2 | 15 | 7 | 3 | 53 | 6 |
| `--redact F,...` | `host replay` | 4 | 3 | 4 | 5 | 0 | 0 | 16 | 4 |
| `--help`, `-h` | any command | 5 | 4 | 0 | 0 | 0 | 0 | 9 | 0 |
| `--version`, `-V` | `rce` alone | 4 | 3 | 4 | 6 | 0 | 0 | 17 | 4 |

26 flags, 539 occurrences on main, 61 more on #81. Callers that run them:
`.github/workflows/deploy.yml` (`host serve ... --log ... --http`,
`host replay ... --log`, `--version`), `release.yml` (`--version` writes
`VERSION`), `deploy/Dockerfile` (`CMD ["--version"]` and the serve line in
its header), `docs/DEPLOY.md`, `scripts/` (`corpus.py`, `edit-landscape.py`,
`table-control.py`, `t7-equiv.sh`, `xarch-hashes.sh`, `xarch-streams.py`,
`partner-demo.sh`, `chat-ui.py`, `install.sh`), `stdlib/scripts/`
(`check.sh`, `check-all.sh`, `gen/feed_gen.py`, `mut/io.py`,
`mut/energy.py`), `programs/ecosystem/run.sh`, `book/listings/ch18/door_dim_control.cmd`,
`book/listings/tour/equiv_diff.cmd`, `book/listings/tour/equiv_same.cmd`
(and on #81 also `ch16/bound_log.cmd`, `ch18/change_erratum.cmd`,
`ch18/change_meaning.cmd`, `ch18/change_rename.cmd`). The tests are
`cli.rs`, `dim.rs`, `nameless.rs`, `traces.rs`, `float_rules.rs`,
`lift_go.rs`, `brick.rs`, `bugfix_sweep.rs`, `version.rs` and the eleven
`host_*.rs` files. The spec chapters that name flags are
`spec/book/12-tooling.md` (10.1, 10.6, 10.9 to 10.11), `16-host-contract.md`
(HC-5a, HC-23, HC-24, HD-2, the `host.HttpListen` row), `17-draft09.md`
(`--key`), `15-handler-spec-profile.md` (`--vectors`), `08-steps-loops-concurrency.md`,
`10-theorems.md` (T61), `07-form-evaluation.md`; `README.md`'s command
table; `book/src/appendix-a-commands.md`, `ch01`, `ch16`, `ch17`, `ch18`
and `tour.md`.

Two related findings:

- `stdlib/tests/io/io_host.rce` carries the comment `# host-flags: --fs`,
  which `stdlib/scripts/check.sh` and `check-all.sh` read to decide to
  grant a file root. A comment that a tool reads is a directive (ledger row
  of 2026-10-27 on `# uses:`), so it goes in the same change.
- `conformance/clauses.json` holds section titles with flags in them
  (10.9 `rce trace --json FILE [EVENTS]`, 10.10 `rce index [--vectors N]
  ...`). It is extracted from the spec, so the new titles change two clause
  keys, and `conformance/clause-repin.tsv` needs two rows.

### 1.2 The other binaries of `spec/checker`

| Binary | Flag | src | callers | total |
|---|---|---:|---:|---:|
| `rce-book` | `build --check` | 3 | `ci.yml`, `regen-spec.sh`, `AGENTS.md`, `spec/README.md` (4) | 7 |
| `rce-spec-data` | `--check` | 2 | `regen-spec.sh`, `spec/README.md`, 10.1, generated and proposal text (6) | 8 |
| `rce-spec-check` | `--no-generate` | 4 | none | 4 |
| `rce-check` | `--parse-only` | 2 | `conformance/tools/ts_crosscheck.py` (2) | 4 |

`rce-fuzz N SEED STATS`, `rce-c1 N SEED OUTDIR [JOBS]`, `rce-bench-det
filter|emit|run ...`, `rce-conform` (JSON lines on stdin) and
`rce-conform-lean` (JSON lines, environment variables) already take no
flags. The flags `rce-c1` passes to rustc and scala-cli belong to those
programs.

### 1.3 Emitted programs (`spec/checker/src/emit/rt.{rs,py,js,scala}`, `prog.rs`)

| Flag | Meaning | src | tests | run | text | gen | prop | total | #81 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `--run` | run `fn main` on the task runtime | 12 | 23 | 3 | 6 | 4 | 6 | 54 | 4 |
| `--trace` | print every entry of D | 13 | 7 | 7 | 6 | 1 | 5 | 39 | 0 |
| `--fuel=N` | fuel limit | 6 | 2 | 2 | 3 | 1 | 2 | 16 | 0 |
| `--fuel-trace` | print `fuel K` per observation | 8 | 9 | 5 | 5 | 2 | 5 | 34 | 0 |
| `--fuel-batch=N,...` | every observation once per limit | 8 | 2 | 0 | 0 | 0 | 2 | 12 | 0 |
| `--link=T,partition\|heal,A,B` | network link events | 5 | 2 | 0 | 0 | 0 | 1 | 8 | 0 |
| `--actor-log` | print actor decisions | 4 | 2 | 0 | 1 | 1 | 0 | 8 | 0 |
| `--restart-log` | print restart decisions | 5 | 4 | 0 | 2 | 8 | 1 | 20 | 0 |
| `--restart=LIST` | replay logged restarts | 10 | 4 | 0 | 1 | 4 | 2 | 21 | 0 |
| `--restart-replay` | replay with no list | 5 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| `--restart-tree=S/c,...` | the supervision tree | 8 | 1 | 0 | 1 | 4 | 0 | 14 | 0 |
| `--event=TICK\|Name\|f=v,...` | program events as inputs | 13 | 3 | 0 | 1 | 1 | 2 | 20 | 0 |

12 flags, 251 occurrences, 4 more on #81 (`ch14/channels_python.cmd`,
`ch14`, `ch15`). Callers: `conformance/adapters/emit_adapter.py`
(`--run`, `--fuel=`, `--trace --fuel-trace`), `conformance/ADAPTERS.md`
(the adapter protocol tells other implementations to pass them),
`stdlib/scripts/ct_check.sh`, `scripts/bench-determinism.py`,
`examples/interop/README.md`, and the tests `emit.rs`, `emit_run.rs`,
`fuel.rs`, `fuel_run.rs`, `one_core.rs`, `indexed.rs`, `loop_cost.rs`,
`method_costs.rs`, `refined.rs`, `sha256.rs`, `write_back.rs`.

### 1.4 IDE binaries (`ide/crates`)

The IDE calls the checker in process (`rce_spec_check::cli::main`) and so
takes the new grammar with no edit of its own, but its binaries have flags:

- `rce` (IDE build, `ide/crates/rce-ide/src/bin/rce.rs`): `test ... --energy-budget J
  [--measured J | --measure] [--strict]`, and `--version`, `-V`.
- `rce-ide`: `--root DIR --file FILE [--lsp CMD...] [--no-memory]
  [--history N] [--demo QUERY] [--export-model OUT] [--offscreen OUT]
  [--size WxH]`, `--version`, `-V`, `--help`, `-h`; `rce-ide test`
  export with `--perfetto` and `-o OUT`; `rce-ide telemetry calibrate
  [--out FILE] [--reps N] [--fake-meter WATTS]`.
- `rce-mcp`: `--root DIR [--ledger FILE] [--key FILE] [--calibration FILE]
  [--no-memory] [--http ADDR --token-file FILE [--sse] [--allow-origin O]...
  [--allow-host H]...] [--no-jail]`, `--help`, `-h`.
- `rce-lsp` takes no arguments.

35 flags; their text occurs 166 times under `ide/`, `ide/docs/PARITY.md` and
the workflows (a text count: `--out` and `--root` include a few uses by
other tools). `acp-client` spawns `rce-mcp` with `--http`, `--token-file`,
`--root`, `--ledger` and `--key`; `mock-acp-agent` with `--no-memory`.

### 1.5 Repository scripts

The scripts are commands too. About 35 flags, about 120 occurrences:
`--check` of `scripts/regen-spec.sh`, `claim-counts.py`,
`canonical-json-vectors.py` (also `--generate`),
`conformance/tools/readme_status.py` (also `--print`),
`spec/examples/form/make.py` and `stdlib/scripts/gen/math_const.py`;
`--bless` of `book/scripts/check-listings.sh` and `programs/scripts/check.sh`;
`--me --log --rce --app --chan --peer --poll` of `scripts/chat-ui.py`;
`--rce --out --mutants --typed-depth --typed-depth-with-examples
--equiv-fuel` of `scripts/corpus.py`; `--rce --out --jobs` of
`edit-landscape.py`; `--rce --out` of `table-control.py`; `--adapter
--tests --filter --out --jobs --quiet --known-failures --stop-on-fail` of
`conformance/harness/run.py`; `--ref` of `stdlib/scripts/gen/effect_gen.py`;
`--cm --sample --out --shard` of `math_vec.py`; `--measure --out` of
`stats_nist_gen.py` and `stats_scipy_gen.py`; `--refresh` of
`view_gen.py`; `--serve` of the two IDE web build scripts.

### 1.6 Totals

| Group | Flags | Occurrences on main | On #81 |
|---|---:|---:|---:|
| `rce` | 26 | 539 | 61 |
| other checker binaries | 4 | 23 | 0 |
| emitted programs | 12 | 251 | 4 |
| sections 1.1 to 1.3 | **42** | **813** | **65** |
| IDE binaries | 35 | about 166 | 0 |
| repository scripts | about 35 | about 120 | 3 |

## 2. The positional grammar

### 2.1 The rule

One rule holds for `rce`, every other binary, every emitted program and
every script:

> **PC-1.** An invocation is `PROGRAM COMMAND-PATH WORD... OPERAND...`.
> After the command path come the command's words, in any order among
> themselves; the first argument that is not one of the command's words
> begins the operands, and from there on every argument is an operand, in
> the order the usage line gives, even one spelled like a word. A setting
> word takes the next argument as its value, whatever it spells.

The other rules follow from it or close the edge cases:

- **PC-2. Operands are positional.** What a command acts on (a FILE, a
  DIR, the log D, EVENTS, OLD and NEW, a MANIFEST) is an operand and has
  a position, never a name. The log D of the host commands is an operand.
- **PC-3. Modes are words.** An option that selects what a command does or
  prints becomes a mode word (`rce fmt check FILE`, `rce dim json FILE`),
  or disappears when only one mode is left: `rce trace` prints only JSON,
  so `--json` goes and nothing replaces it.
- **PC-4. Bounds and grants are setting words.** `fuel 1000`, `depth 8`,
  `vectors 4`, `fs DIR`, `http HOST:PORT`. A value keeps its spelling of
  today (`HOST:PORT`, `HOST:PORT=KEY`, `ISSUER=HEXPUBKEY`, `F,G`,
  `TICK|Name|f=v,...`). A word may be given once, except those the usage
  line marks repeatable (`net`, `peer`, `key`, and the emitted program's
  `link`, `restart`, `event`).
- **PC-5. The name of a word is the flag's name without its dashes.**
  `--fuel-trace` becomes `fuel-trace`, `--no-generate` becomes
  `no-generate`. The exceptions are the ones PC-2 and PC-3 force
  (`--log D` becomes the operand D; `trace --json` becomes `trace`) and
  `--help`, `-h`, `--version` and `-V`, which become the commands `help`
  and `version`.
- **PC-6. No argument starts with `-`.** An argument that does is a usage
  error, exit status 2, wherever it stands. A file whose name starts with
  `-` is written `./-name`.
- **PC-7. A word and a file of the same name.** Only the first operand can
  be read as a word, because once operands begin nothing is. Every word is
  lowercase letters, digits and hyphens, so an argument with a `.`, `/`,
  `:`, `,`, a space, a parenthesis or a capital letter is never a word:
  `.rce` files, paths with a directory, addresses and event lists never
  collide. A first operand spelled exactly like one of its command's words
  (a file called `fuel` given to `rce dim`) is written `./fuel`.
- **PC-8. One table per binary.** Each binary declares its commands, words,
  arities and operands once, as data; the parser reads it and `USAGE_TEXT`
  is printed from it. The usage text and the parser cannot then disagree,
  and a test can read the table (section 4.3).

### 2.2 Why words come before operands

The owner's sketch put grants after the operands
(`rce host serve FILE D http 0.0.0.0:8080 fs DIR`). PC-1 puts them before
(`rce host serve http 0.0.0.0:8080 fs DIR FILE D`), for three reasons:

1. `rce host run FILE D [EVENTS]` has an optional last operand. With grants
   after D, the third argument is either EVENTS or a grant, and only the
   word list can tell. An event list of one lowercase event named `pty`
   would then grant a terminal. All 399 `.rce` files of the repository
   name events with a capital letter, but the grammar of section 3 does
   not require it.
2. `check`, `test`, `fmt`, `hash`, `index` and `fingerprint` take a list
   of FILEs. With words after the list, the end of the list is found only
   by recognising a word, so every FILE position can collide with a word.
   With words first, only the first operand can (PC-7).
3. The parser is one left-to-right pass with no lookahead, and the same
   pass serves every binary.

The order reads as "serve, over HTTP on 0.0.0.0:8080, the program
notes_http.rce on the log /data/notes". Section 5 asks the owner whether to
keep it.

### 2.3 Why `fuel 1000`

Three forms were considered for a numeric bound:

- **An ordered positional parameter** (`rce dim FILE 1000`,
  `rce fingerprint 8 1000 FILE...`). A number is a valid file name; an
  optional trailing number competes with an optional operand and with a
  list of files; with two bounds the reader has to remember which comes
  first. Declined.
- **`fuel=1000`.** Excluded by the owner, and `=` already appears inside
  values (`ISSUER=KEY`, `f=v`).
- **A setting word, `fuel 1000`.** It names what the number bounds, the
  value is always taken as the value, and `fuel` cannot be a path with a
  directory or an extension. Chosen.

Defaults stay: `fuel` 100,000 states, `depth` 8, `vectors` 0.

### 2.4 `rce`

| Today | Proposed |
|---|---|
| `rce check FILE\|DIR...` | unchanged |
| `rce run FILE [EVENTS]` | unchanged |
| `rce test FILE\|DIR...` | unchanged |
| `rce trace --json FILE [EVENTS]` | `rce trace FILE [EVENTS]` (JSON is the only output) |
| `rce trace --json FILE --streams PATH` | `rce trace streams FILE PATH` |
| `rce emit TARGET FILE` | unchanged: TARGET is already a mode word |
| `rce run-core FILE`, `rce form-check FORM [VECTORS]` | unchanged |
| `rce fmt [--check] FILE...` | `rce fmt FILE...`, `rce fmt check FILE...` |
| `rce fmt --nameless FILE` | `rce fmt nameless FILE` |
| `rce dim [--fuel N] [--control] [--json] FILE` | `rce dim [control] [json] [fuel N] FILE` |
| `rce equiv [--fuel N] [--json] A B` | `rce equiv [json] [fuel N] A B` |
| `rce fingerprint [--depth N] [--fuel N] [--json] FILE...` | `rce fingerprint [json] [depth N] [fuel N] FILE...` |
| `rce change [--fuel N] OLD NEW` | `rce change [fuel N] OLD NEW` |
| `rce lib [MODULE]`, `hash`, `manifest DIR`, `api-diff OLD NEW`, `v09-status`, `lsp`, `search CATALOG QUERY`, `unform FORM.json` | unchanged |
| `rce manifest-check MANIFEST [--advisories FILE] [--key ISSUER=HEX]...` | `rce manifest-check [advisories FILE] [key ISSUER=HEX]... MANIFEST` |
| `rce index [--vectors N] FILE\|DIR...` | `rce index [vectors N] FILE\|DIR...` |
| `rce host run FILE --log D [GRANTS] [EVENTS]` | `rce host run [GRANT...] FILE D [EVENTS]` |
| `rce host serve FILE --log D [GRANTS] [--http HOST:PORT]` | `rce host serve [GRANT...] [http HOST:PORT] FILE D` |
| `rce host tls-id FILE --log D` | `rce host tls-id FILE D` |
| `rce host replay FILE --log D [--redact F,...]` | `rce host replay [redact F,...] FILE D` |
| `rce db D CMD [ARGS]` | unchanged: CMD and ARGS are operands, a statement to the database |
| `rce SUB --help`, `rce -h`, `rce --help` | `rce help` |
| `rce --version`, `rce -V` | `rce version` |

GRANT is one of `fs DIR`, `blob DIR`, `media DIR`, `wasm DIR`,
`net HOST:PORT` (repeatable), `chan HOST:PORT`, `tls`,
`peer HOST:PORT[=KEY]` (repeatable) and `pty`. Examples:

```sh
rce host serve http 0.0.0.0:8080 /srv/notes_http.rce /data/notes
rce host serve chan 127.0.0.1:7001 peer 127.0.0.1:7002 apps/chat.rce a.log
rce host serve chan 0.0.0.0:7001 tls peer 10.0.0.2:7001=9f0c...e1 apps/chat.rce a.log
rce host run fs /tmp/root net 127.0.0.1:9000 io_host.rce d.log "Roll(n: 2)"
rce host replay redact priv,ek,dk pqxdh.rce pq.log
rce dim control json door.rce
rce change fuel 1000 counter.rce counter_v2.rce
rce trace streams app.rce streams.txt
```

### 2.5 Other checker binaries

| Today | Proposed |
|---|---|
| `rce-book build [--check] [SPEC_DIR]` | `rce-book build [check] [SPEC_DIR]` |
| `rce-book verify RECEIPT.json [SPEC_DIR]` | unchanged |
| `rce-spec-data [--check]` | `rce-spec-data [check]` |
| `rce-spec-check [--no-generate] [SPEC_DIR]` | `rce-spec-check [no-generate] [SPEC_DIR]` |
| `rce-check [--parse-only] FILE...` | `rce-check [parse-only] FILE...` |

The word `check` means the same after every command that writes files
(`rce fmt`, `rce-book build`, `rce-spec-data`, the scripts): write nothing,
report what differs, exit 1 if anything does.

### 2.6 Emitted programs

An emitted program has words and no operands:

```
PROGRAM [run] [trace] [fuel N] [fuel-trace] [fuel-batch N,N,...]
        [link T,partition|heal,A,B]... [actor-log] [restart-log]
        [restart TICK|SUP|CHILD|ID|STOP|RESET,...]... [restart-replay]
        [restart-tree S/c,...] [event TICK|Name|f=v,...]...
```

For example `python3 counter.py trace fuel-trace` and
`node app.js run fuel 1000`. An argument that is not one of these words is
a usage error, exit status 2, as `rce` gives; today an emitted program
ignores an argument it does not know.

### 2.7 IDE binaries and scripts

The same rule, word for word by PC-5:

- `rce test [energy-budget J] [measured J] [measure] [strict] FILE...`
  (IDE build; `measured` and `measure` stay exclusive), `rce version`.
- `rce-ide [no-memory] [history N] [demo QUERY] [export-model OUT.json]
  [offscreen OUT.png] [size WxH] ROOT FILE [LSP-COMMAND...]`: the language
  server command line becomes the trailing operands, since `--lsp CMD...`
  took the rest of the arguments. `rce-ide test [perfetto] [out FILE] ...`,
  `rce-ide telemetry calibrate [out FILE] [reps N] [fake-meter WATTS]`,
  `rce-ide version`, `rce-ide help`.
- `rce-mcp [ledger FILE] [key FILE] [calibration FILE] [no-memory]
  [http ADDR] [token-file FILE] [sse] [allow-origin O]... [allow-host H]...
  [no-jail] ROOT`, `rce-mcp help`.
- Scripts: `scripts/regen-spec.sh [check]`, `claim-counts.py [check]`,
  `book/scripts/check-listings.sh [bless]`, `chat-ui.py [chan ADDR]
  [peer ADDR]... [poll S] [rce PATH] [app PATH] ME LOG`,
  `conformance/harness/run.py [quiet] [jobs N] [filter F]
  [known-failures FILE] [stop-on-fail] [out FILE] ADAPTER TESTS`, and so
  on: a required value becomes an operand, everything else a word.

## 3. Exit statuses and output

Exit statuses do not change: 0 success, 1 requires change or a kept step
or a difference found by `check`, 2 usage error, 3 undecided (10.1). Two
cases move between them because the grammar moves: `rce trace FILE`, a
usage error (2) today, becomes the trace (0); and every removed flag
becomes a usage error (2). `rce help` and `rce version` exit 0 as
`--help` and `--version` do.

Output that names a flag has to change, and only that:

- `USAGE_TEXT` and the usage lines of `rce-book`, `rce-ide`, `rce-mcp`
  and the calibrate command, printed from the tables of PC-8;
- the error texts `--fuel N: N is a count of states`, `--depth N: ...`,
  `--vectors N: not a count`, `--key ISSUER=HEXPUBKEY`, `--log D`,
  `--http: one listening address per host`, `--http needs rce host serve`,
  `--tls: every --peer needs a pinned key`, `--peer X: not a 32-byte hex
  key`, `--chan ADDR: ...` (`cli.rs`, `dim.rs`), which become `fuel N: ...`
  and so on;
- the verdict lines `undecided ... raise --fuel` of `rce change` and
  `rce equiv` and `fuel reached before depth N ...; raise --fuel or lower
  --depth` of `rce fingerprint` (`dim.rs`), which become `raise fuel` and
  `raise fuel or lower depth`; the book listings that show them are
  re-blessed;
- the five delivery notes `--http: event HttpRequest is not declared ...`
  (`host.rs`), which become `http: ...`. They go to standard error and
  are not in D, so no log and no state hash changes;
- `grammar: generation check skipped (--no-generate)` of `rce-spec-check`.

No trace, D entry, state hash, Form, emitted program output or program
hash changes. The text of an emitted program changes (its argument
reader), so tests that compare emitted text are re-blessed; its output
under the same words is byte for byte what it printed under the flags.

## 4. Migration

### 4.1 One change, no aliases

The owner wrote "remove", so the change keeps no alias: no old flag runs
anything after it. The change is one commit, made after pull request #81
and the other open pull requests that touch `cli.rs`, the host tests or
`book/` have merged, so that it can move every caller at once:

1. `spec/checker/src/cli.rs`: the table of PC-8, one reader, the new
   `USAGE_TEXT`; `src/bin/rce.rs` (`version`), `book.rs`, `spec_data.rs`,
   `spec_check.rs`, `main.rs`; the messages of section 3 in `cli.rs`,
   `dim.rs` and `host.rs`.
2. The emitted runtimes `emit/rt.rs`, `rt.py`, `rt.js`, `rt.scala` and the
   entry code in `emit/prog.rs`; then every test and golden that holds
   emitted text.
3. Every caller of section 1: the workflows (`ci.yml`, `deploy.yml`,
   `release.yml`), `deploy/Dockerfile` (`CMD ["version"]`), `docs/DEPLOY.md`,
   `scripts/`, `stdlib/scripts/`, `programs/ecosystem/run.sh`, `.githooks/pre-push`,
   `conformance/adapters/emit_adapter.py`, `conformance/tools/ts_crosscheck.py`,
   `conformance/ADAPTERS.md`, the tests of `spec/checker/tests`, `README.md`,
   `AGENTS.md`, `CONTRIBUTING.md`, `spec/README.md`, `examples/interop`.
4. The specification: section 10 of `spec/book/12-tooling.md` (the headings
   of 10.6, 10.9, 10.10 and 10.11 lose their flags), Appendix E in
   `16-host-contract.md`, `17-draft09.md`, `15-handler-spec-profile.md`,
   `14-draft06-concurrency.md` (the restart words), `08`, `10`, `07`; then
   `scripts/regen-spec.sh` for `spec/RCE-SPEC.md` and `spec/data`, and
   two rows in `conformance/clause-repin.tsv` for the two clause keys that
   change.
5. The book: `book/src/appendix-a-commands.md` (the command table, written
   from the same table as `USAGE_TEXT` where the book's tools allow it),
   `ch01`, `ch14` to `ch18`, `tour.md`, and the `.cmd` files of
   `book/listings`, whose outputs `book/scripts/check-listings.sh bless`
   rewrites.
6. `# host-flags: --fs` in `stdlib/tests/io/io_host.rce` is removed;
   `stdlib/scripts/check.sh` and `check-all.sh` read the grants of each
   host test from a data file beside them (path and words, one line per
   test) instead of a comment.
7. The IDE (`rce-ide`, `rce-mcp`, the IDE build of `rce`, `acp-client` and
   `mock-acp-agent`, which spawn `rce-mcp`), its README and `ide/docs/PARITY.md`,
   and the scripts of section 1.5, unless the owner splits them off
   (section 5).
8. Open proposals that name a flag use the new spelling. The ledger, dated
   release notes (`ide/release/NOTES-*.md`, `RELEASE.md`) and dated
   surveys keep their words, as the vocabulary check already allows.

Risks of removing without aliases, and what limits each:

- **The production host and other deployments** run
  `rce host serve ... --log ... --http ...` from a unit or a container
  command. A new image fails at start with status 2 until its command line
  is changed. `deploy.yml` and `docs/DEPLOY.md` change in the same commit;
  anything outside the repository has to be changed by hand at the same
  release.
- **Released binaries and their users.** `scripts/install.sh` installs the
  newest release, so a user's scripts break at the first update. The
  release notes of that release give the table of section 2.4.
- **Other implementations of the adapter protocol.** `conformance/ADAPTERS.md`
  tells them to run emitted programs with `--run`, `--fuel=N`, `--trace`
  and `--fuel-trace`. The protocol text changes in the same commit; an
  outside adapter that still passes flags gets status 2 and fails every
  test, which is loud rather than wrong.
- **Branches open at the time of the change** that add a flag use fail the
  check of 4.3 after they merge `main`, and are fixed there.
- **Old commands in prose outside the repository** (issues, chat logs)
  stop working. A removed flag prints a usage error that names the new
  spelling if the owner allows it (section 5); it never runs the command.

### 4.2 Tree-sitter grammar, language server and IDE

- The tree-sitter grammar describes RCE program text, not command lines,
  and does not change; `tree-sitter/package.json` passes `--wasm` to the
  tree-sitter command line, which is that tool's own flag.
- The language server (`rce lsp`, 10.7) takes no arguments and its
  messages do not change. `rce-lsp` takes none either. Editors that start
  `rce lsp` need no change.
- The IDE calls `cli::main` in process, so its `rce` passes every command
  but `test` through to the new grammar with no edit. Its own flags change
  as in 2.7, and with them the code that spawns them (`rce-ide` starting its language server,
  `acp-client/src/client.rs` starting `rce-mcp`) and the IDE's tests.

### 4.3 How CI shows that no caller still uses a flag

Two checks, both run by `ci.yml`, both positional themselves:

1. **A test of the grammar**, `spec/checker/tests/positional_cli.rs`:
   - every line of `USAGE_TEXT` and of the other binaries' usage texts has
     no argument that starts with `-`;
   - every usage line, read back through the table of PC-8, gives the
     command it came from, so the text and the parser agree;
   - each of the 42 flags of sections 1.1 to 1.3, given in its old place
     to `cli::main`, to each `CARGO_BIN_EXE_*` binary and to an emitted
     program of every available target, exits 2 and writes no file;
   - PC-7: a file named `fuel` given as `./fuel` to `rce dim` is read as
     a file, and given as `fuel` is read as the word.
2. **A scan of the tree**, `scripts/check-flags.py`, which fails when any
   tracked file has
   - a command line of an RCE binary with an argument that starts with
     `-`: the pattern `\brce(-[a-z-]+)?( [^ |;&]+)* --?[A-Za-z]` (the
     owner's `rce [a-z-]* --`, widened to any number of arguments before
     the flag and to short flags) in shell, YAML, Markdown, Dockerfiles and
     `.cmd` listings;
   - in Rust, Python and JavaScript, a string literal that starts with `-`
     in the argument list of a call that runs an RCE binary or an emitted
     program (`rce(&[...])`, `Command::new(env!("CARGO_BIN_EXE_...`,
     `rce.run(...)`, `[RCE, ...]`, `"$RCE"`, the emitted program's path);
   - any of the removed flag names, anywhere, outside the lines listed in
     `scripts/flag-allow.tsv` (path glob, line pattern, reason, in the
     shape of `scripts/vocab-allow.tsv`): uses by other programs
     (`lake env lean --run`, `unshare --net`, `tree-sitter build --wasm`,
     `gh ... --json`, cargo, and `--version` of rustc, node, python3,
     scala-cli, mdbook, lean and wasmtime), the ledger, this proposal and
     dated release notes.

   The scan prints every remaining line with its path and line number. On
   the day of the change it reports 0; on `main` today it would report
   the 813 occurrences that sections 1.1 to 1.3 count.

## 5. Open questions for the owner

1. **Order of grants.** Words before operands as proposed
   (`rce host serve http 0.0.0.0:8080 FILE D`), or FILE and D first as
   in the owner's sketch? The second needs either EVENTS behind a word
   (`events "Inc, Dec"`) or a rule that event names start with a capital
   letter.
2. **The message for a removed flag.** May a removed flag print a usage
   error that names its new spelling (`--fuel: write fuel 1000 before the
   files`), exit 2 and run nothing, or only the plain usage text?
3. **Scope of the one change.** The IDE binaries (35 flags) and the
   repository scripts (about 35 flags) in the same change as `rce`, or in
   changes of their own right after it?
4. **Values that join two things with `=`** (`peer ADDR=KEY`,
   `key ISSUER=HEX`): keep them as one value, as proposed, or split them
   into two values (`key ISSUER HEX`)?
