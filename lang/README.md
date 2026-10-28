# `lang/`: pinned oracles and corpora for lifting other languages

This directory holds the data that a lift of a program from another
language into RCE will be judged against, in the formats that
[`spec/proposals/rce-form.md`](../spec/proposals/rce-form.md) defines
(section 16 for the per-language files, section 20 for acceptance). It holds
that data for three languages:

| Language id | Oracle | Programs |
|---|---|---|
| `go1.24` | the gc toolchain, Go 1.24.7 | see `go1.24/corpus/manifest.toml` |
| `python3.13` | CPython 3.13.16 | see `python3.13/corpus/manifest.toml` |
| `es2020` | Node.js 22.22.0 (V8) | see `es2020/corpus/manifest.toml` |

**No lifter exists yet.** Nothing in this repository lifts a Go, Python or
JavaScript program, and no grammar lock, lift table, runtime library or
print table of section 16 is written. What is here is the other side of the
comparison: for each language, a pinned implementation (the oracle), the
rule that turns one of its runs into an observation, and a corpus of
programs, each with the observation recorded from the oracle. When a lift
exists, a case holds when the lifted program's observation equals the
recorded one byte for byte.

## Terms

- **Oracle**: a pinned implementation of a language, whose runs give the
  observations a lift is compared with.
- **Case**: one program with one input. Every case here has the same input:
  no arguments, no standard input, and no environment beyond the oracle's
  pins.
- **Observation** (`rce-obs/1`): the record of one run. The `command`
  profile compares three fields: the exit status, the standard output as
  exact bytes (`stdout_hex`), and the class of the standard error
  (`stderr_class`).
- **Key**: the lower-case hexadecimal SHA-256 of a value's canonical JSON
  (specification 6.1.1). A TOML record's key is the key of the value it
  parses to, so comments do not change it.
- **Admitted**: a program enters the corpus only when every run of it under
  the oracle gives the same observation (rule K1 and the case classes of
  section 20).

## Files of one language

```
lang/<L>/
  oracle/<name>.toml      rce-oracle/1: the pinned implementation
  observe.toml            rce-observe/1: how a run becomes an observation
  corpus/manifest.toml    rce-corpus/1: one [[case]] per admitted program
  corpus/files/           the programs, each stored under its SHA-256
  corpus/expected/        one rce-obs/1 file per case, in canonical JSON
  corpus/admission.txt    every program offered to the corpus and what happened to it
  corpus/verify.txt       the output of the last verify run that was recorded
```

### The oracle record

Each record pins its toolchain by version and by digest, and holds as data
the commands it runs (positional arguments only; `{src}`, `{bin}`, `{run}`
and `{cache}` are filled in by `corpus.py`), the whole environment of a run,
and the perturbations used at admission.

- **Toolchain digests.** The proposal names an OCI image digest; no image is
  built yet, so `[toolchain]` holds what stands for it: the SHA-256 of the
  executable, of every file of its toolchain directory where it has one
  (Go's `GOROOT`; CPython's standard library, with bytecode caches and
  installed packages left out), and of each shared object the executable
  loads as `ldd` lists it. The tree digest is the SHA-256 of the lines
  `<sha256>  <relative path>`, one per file, sorted by path; a symbolic link
  contributes `link:<target>`. `version_output` is what the `version`
  command prints.
- **Environment.** A run receives exactly `pins.env` (and a Go build exactly
  `pins.build_env`), in a new, empty directory that holds only the program,
  in a new network namespace with no interface up (`unshare --net`).
- **What is not pinned.** This host cannot set the wall clock of one
  process, so no clock is pinned or perturbed. Each record lists, under
  `[[unpinned]]`, patterns for the names that read the clock, a source of
  entropy, the file system outside the run directory, or the identity of
  the process; a program that names one is not admitted. The patterns are
  conservative: a match in a comment or a string also keeps a program out.
- **Perturbations.** At admission every program runs twice as the record
  says (`runs = 2`) and once more under each perturbation: the environment
  in reverse order, the run confined to 1 and then 4 processors, and one
  variable changed per language (Go's `GOMAXPROCS`, CPython's
  `PYTHONHASHSEED`, libuv's `UV_THREADPOOL_SIZE`). A program whose two runs
  differ is `oracle-nondet`; one whose observation changes under a
  perturbation is `nondet:<perturbation>`; neither is admitted.

Differences from the proposal's examples, each for a stated reason:

- The proposal names the language `go1.23`; this host has Go 1.24.7, so the
  directory is `go1.24`. A file built outside a module takes the toolchain's
  language version, go1.24.
- CPython runs with `-P -s`, not `-I`: `-I` implies `-E`, which would make
  the interpreter ignore the pinned `PYTHONHASHSEED`; the environment is
  replaced whole instead.
- Node runs the program as `main.cjs`, so it is read as a CommonJS script
  whatever `package.json` lies above the run directory. The corpus programs
  use the language of ECMAScript 2020 and, from the host, `console.log`,
  `console.error` and `process.exitCode` only.

### Observations and the standard error

`observe.toml` gives the class of a run's standard error: an empty string
when nothing was written, `signal:<n>` for a run ended by a signal (its exit
status is then 128 + n), otherwise the class of the first row whose pattern
matches (for example `exc:ZeroDivisionError`, `panic:runtime error: integer
divide by zero`, `resource:stack`), and when no row matches,
`bytes:<sha256>` of the standard error with the run directory's path written
`{run}`, so that comparison stays exact. A class that starts with
`resource:` names a limit of the oracle (its recursion or stack limit), and
a case with such a class is admitted with the class `resource` rather than
`deterministic`, as section 20 says.

An expected file holds the observation and, under `recorded`, the key of the
oracle record it was recorded under, the keys of the observations of both
runs, and the key of the observation under each perturbation (rule K1). The
manifest's `expected_key` is the key of the observation without `recorded`.
A case's key is the key of `{"inputs": …, "profile": "command", "sha256":
<the file's SHA-256>}`.

### The corpus

Every program was written for this repository and is licensed MIT OR
Apache-2.0; each file says so on one of its first lines, and admission
checks that line (rule K2). The tiers are the proposal's: `basics` (one
construct at a time), `algorithms` (small complete programs), `boundaries`
(the edges a lift must carry exactly: integer overflow and division, float
printing, strings and Unicode, iteration order, errors and exceptions,
resource limits) and `generated` (tables over edge values written by
`generate.py`, whose SHA-256 each such case records). A case is never edited
(rule K3): a changed program is a new file under a new hash.

## Reproducing the recorded expectations

On x86-64 Linux with the same toolchains at the same paths, `unshare` and
`ldd` (and `taskset` for admission), and permission to make a network
namespace (root, or unprivileged user namespaces):

```sh
python3 lang/corpus.py pins go1.24        # prints the digests found and whether every pin holds
python3 lang/corpus.py verify go1.24      # MATCH, or the first differing byte, for every case
```

`verify` runs every case `runs` times and prints one line per case: `MATCH`
with the observation's key, or `DIFFERS` with the run and the first field
that differs (for standard output, the offset of the first differing byte
and the two byte values). Its output holds no times and no paths, so it can
be compared with `corpus/verify.txt` byte for byte. When a pin does not
hold, every case is `NOT-RUN` and the line names the pin. On another host,
or with another release, the record does not apply: a new oracle record and
a new admission are needed.

To add programs, write them under a directory with one subdirectory per
tier, each with the licence line, and run
`python3 lang/corpus.py admit LANG DIR`; the generated tier comes from
`python3 lang/generate.py LANG DIR`.
