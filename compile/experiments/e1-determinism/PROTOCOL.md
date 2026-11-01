# E1: determinism of lowering (reproducibility)

Tests H1 of [HYPOTHESES.md](../../HYPOTHESES.md): with declared inputs, a
compiler's output bytes are the same wherever it runs. The results are in
[RESULTS.md](RESULTS.md), which `report.py` generates from `results/*.json`,
which `e1.py` writes. No figure in either is typed.

## Method

The declared builds are in [matrix.json](matrix.json): one *configuration*
per compiler and flag set, each naming its programs (under
[programs/](programs/)), its build command, and the same command with the
toolchain's documented path remapping. For every configuration whose tool
is installed, and every program it names, `e1.py` makes these builds, each
in a fresh directory that holds only the copied source:

| Builds | Where | Comparison |
|---|---|---|
| A1, A2 | twice in directory A | *same directory* |
| A1, B1 | A, and B, a second absolute path of another length | *two directories* |
| Ar, Br | A and B, with path remapping | *two directories, remapped* |
| C1, C2 | twice in A with `SOURCE_DATE_EPOCH` unset, at least 1.1 s apart | *same directory, clock free* |

Every build except C1 and C2 runs with the declared environment: `PATH` and
`HOME` from the caller, `LC_ALL=C`, `TZ=UTC`, `SOURCE_DATE_EPOCH=1735689600`
and the configuration's own variables. A configuration's caches (Go's
`GOCACHE`, Zig's cache directories) are new empty directories, outside the
build directory, for every build, so the second build of a pair is a real
rebuild and not a cache hit.

An *output* is every file the build leaves in its directory other than the
source. For each pair of builds, `e1.py` records per output file the two
sizes and the *differing byte count*: the number of byte positions, below
the longer of the two lengths, at which the files differ or one file has
ended. The count is positional, so a path of another length that shifts
the rest of a file counts every shifted byte. It also records the *path
copies*, the number of times the absolute build directory occurs in the
outputs of B1 and of Br, and the SHA-256 of every output of A1 and Ar, so
that two results files from different machines can be compared where the
tool version strings agree.

## Path remapping used

The remapping options are the ones the toolchains document; where none was
found, the remapped comparison is "not tested".

| Toolchain | Remapping | Source |
|---|---|---|
| gcc, g++, gfortran, gnatmake | `-ffile-prefix-map=DIR=.` (passed to the compiler by `gnatmake -cargs`) | GCC manual [K] |
| clang, clang++ | `-ffile-prefix-map=DIR=.` | Clang command-line reference [K] |
| rustc | `--remap-path-prefix=DIR=.` | the rustc book [K] |
| go | `-trimpath` | `go help build` [K]; Go 1.21 was "the first Go toolchain with perfectly reproducible builds" (Go blog, 2023) [S] |
| erlc | `+deterministic` | erlc and compile module documentation [K, unverified] |
| ocamlopt | `BUILD_PATH_PREFIX_MAP=.=DIR` | reproducible-builds.org specification; OCaml's support for it is unverified |
| julia | none; the remapped run writes the code without source lines (`debuginfo = :none`) | Julia documentation of `code_llvm` [K, unverified] |
| lean, wasmtime compile, zig, ghc, cobc | none found | |

`go build` is measured twice: in module mode (Go's default) and in GOPATH
mode (`GO111MODULE=off`), because a trial run of this harness found that
the build differs across directories after `-trimpath` in GOPATH mode.

## Toolchains

On the machine of the committed results: gcc 13.3, clang 18.1, rustc 1.97
(with the `wasm32-wasip1` target), go 1.24, gfortran 13.3, Lean 4.12 and
Wasmtime 26.0.1; `RESULTS.md` records the exact version strings. The
workflow [`.github/workflows/compile-e1.yml`](../../../.github/workflows/compile-e1.yml)
runs the same harness on a GitHub-hosted Ubuntu 24.04 runner, adds ghc,
gnat, gnucobol, ocaml, erlang and julia from apt where that release has
them, and Zig 0.17.0 from its official tarball checked against the SHA-256
that ziglang.org publishes, and uploads its results file and write-up as
the artifact `compile-e1-results`. No binary outside the machine's package
manager is downloaded on the machine of the committed results.

## Running it

```sh
python3 e1.py run results/NAME.json
python3 report.py write
python3 report.py check
```

`report.py check` fails when `RESULTS.md` is not what `report.py write`
would print from the committed results; the workflow runs it.

## Limits

- The programs are small: one or two per language. A difference in a large
  program that these do not reach is not tested.
- A cross-machine comparison needs the same tool version string on both
  machines, so it covers only the toolchains whose package versions agree.
- Link-time optimisation, profile-guided builds and parallel code
  generation other than rustc's defaults are not in the matrix yet.
- The harness checks bytes. It does not check that the program's meaning
  was preserved, which is H3 and H7.
