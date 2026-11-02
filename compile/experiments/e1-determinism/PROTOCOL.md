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
| clang on macOS, compile and link split (configurations marked macOS; added 2026-11-02) | `-ffile-prefix-map={dir}=.` on the compile, `-Wl,-oso_prefix,{dir}/` on the link; one of the two configurations also sets `ZERO_AR_DATE=1` | Clang command-line reference [K]; Apple ld, ld(1) on macOS, `-oso_prefix` [K, unverified]; `ZERO_AR_DATE` read by Apple ld [K, unverified] |
| rustc on macOS (added 2026-11-02) | `--remap-path-prefix={dir}=.` and `-C link-arg=-Wl,-oso_prefix,{dir}/` | the rustc book [K]; Apple ld, ld(1) on macOS [K, unverified] |

`go build` is measured three times: in module mode (Go's default); in
GOPATH mode (`GO111MODULE=off`) with the source file named on the command
line; and in GOPATH mode with the source placed at `gopath/src/e1` under
the build directory, `GOPATH` set to that `gopath`, and the package built
by its import path `e1`. A trial run of this harness found the second
differing across directories after `-trimpath`; the third tests the cause
(below).

For a differing ELF file, `e1.py` also counts the differing bytes per
section, from the file's own section headers. For a differing Mach-O file
(the format of macOS executables, object files and the DWARF file inside a
`.dSYM` bundle) it counts the differing bytes per region and records the
debug map; see "Mach-O attribution" below. For the Go configurations it
runs the remapped build once more in each directory with
`GODEBUG=gocachehash=1`, which makes `go build` print every input it
hashes into each action ID, and records the lines that name the build
directory.

## The Go GOPATH-mode difference

In GOPATH mode, a package named by a file outside every `GOPATH` is a
*local* package, and Go gives it the import path `_` followed by its
absolute directory. The `gocachehash` trace shows that path among the
inputs of the compile and link action IDs
(`local true prefix "_/…/build"`), and `-trimpath` does not remove it.
The Go build ID that `go tool buildid` prints is built from those hashes;
of its four parts, the first two differ between the directories and the
last two are equal. The linker writes the build ID into the
`.note.go.buildid` section, and the 20-byte GNU build ID in
`.note.gnu.build-id` differs with it. These two notes are the only
sections that differ: code and data are identical. With the package given a declared
import path in a `GOPATH`, the lines naming the directory disappear and the
remapped builds are identical across directories. `RESULTS.md` shows both
from the harness's own records.

## Changes for macOS, 2026-11-02

Made after the first macOS run (`results/macos-27.2-arm64.json`, a MacBook
Pro M5 Max, macOS 27.2, arm64) and before its rerun. No registered
configuration changes its command, its remapping or its programs.

1. **The build directory is named by its real path.** `e1.py` resolves
   every symbolic link in its work directory before it makes the build
   directories, so `{dir}`, the remappings that use it, the path count and
   the `{dir}` of the traces all name the path that `getcwd()` returns.
   *Why:* on macOS the system temporary directory is under `/var/folders`,
   and `/var` is a symbolic link to `/private/var` [K]. The harness passes
   no `PWD`, so a tool that asks for its working directory gets the
   resolved path. The first macOS run shows this for Go: its trace names
   the directory as `_/private{dir}`. A check on Linux (gcc 13.3,
   clang 18.1, rustc 1.99.1, 2026-11-02), building `hello.c` and `hello.rs`
   with `-g` in a directory reached through a symbolic link and with no
   `PWD`, found the real path once in each output when the remapping named
   the linked path, and no copy when it named the real path. On the Linux
   machines of the committed results the work directory has no link in
   it, so their builds are unchanged.
2. **Machine facts on macOS.** `machdep.cpu.brand_string` and `hw.model`
   from `sysctl`, and the release from `sw_vers` (product name, version and
   build), recorded as `cpu`, `model` and `os`. The first macOS run
   recorded neither, because `/proc/cpuinfo` and `/etc/os-release` do not
   exist there.
3. **Mach-O attribution** (below), with the results schema raised to
   `rce-compile-e1-results/2`. `report.py` still reads schema 1 and marks a
   schema-1 macOS file "without Mach-O attribution".
4. **Three new configurations, marked (macOS)**, which run only where
   `uname -s` prints `Darwin`. When `clang` compiles and links in one call
   with `-g`, the driver writes the object file to the temporary directory
   under a random name and the linker's debug map names that file by its
   absolute path [K]; `-ffile-prefix-map` does not apply to that path [K].
   Compiling and linking in two calls puts the object at `{dir}/{stem}.o`,
   a declared path, and `-oso_prefix` removes `{dir}/` from it in the debug
   map. The debug map also holds the object file's modification time;
   `ZERO_AR_DATE=1` is the variable Apple ld reads to write 0 there [K,
   unverified], so it is a configuration of its own, to separate the two
   inputs. For rustc, whose objects are written in the build directory,
   `-oso_prefix` is added to the documented remapping. These configurations
   are additions: the registered `clang -O2 -g` and `rustc -O -g` keep
   their meaning and stay in the matrix.

## Mach-O attribution

For a differing thin 64-bit little-endian Mach-O file (arm64 or x86_64),
`e1.py` reads the load commands with its own parser and divides the file
into regions: the header; each load command (`LC_UUID` on its own, and a
segment command by its segment name); each section with file contents
(for example `__TEXT,__text`, `__DWARF,__debug_info`); the pieces of the
`__LINKEDIT` segment that the load commands locate (the symbol table, where
the debug map's stab entries are, the string table, the code signature,
the chained fixups, the exports trie, the function starts and the others);
and, for bytes in none of these, the segment that holds them or "outside
segments". Each region's bytes are compared with the same region of the
other file, from the region's own start, so a region that grows does not
shift the count of the regions after it. A fat (universal) file, a 32-bit
file and a big-endian file are not attributed.

For the same files it records, for both builds, the 16 bytes of `LC_UUID`
and the debug map: every `N_SO` entry (source directory and file) and every
`N_OSO` entry (object file path and the value stored with it, the object
file's modification time). In the recorded paths the build directory is
written `{dir}` and the system temporary directory `{tmpdir}`. `RESULTS.md`
shows the `LC_UUID` of the two builds and the entries found in only one of
them.

## The first macOS run, 2026-11-02

What `results/macos-27.2-arm64.json`, written before the changes above,
shows, from its own rows:

- Every `-g` configuration of C, C++ and Fortran (Apple clang 21.0 as
  `gcc`, `clang`, `g++` and `clang++`; Homebrew gfortran 16.2) and
  `rustc -O -g` leaves a `.dSYM` bundle beside the executable:
  `Info.plist`, the DWARF file and a relocations file. `Info.plist` and the
  relocations file are identical in every comparison.
- Same directory twice, C, C++ and Fortran with `-g`: the executable
  differs in 37 or 38 bytes at equal size; the DWARF file is identical,
  except for `gcc -O2 -g` on `hello.c`, where the executable differs in 85
  bytes and the DWARF file in 16. With `SOURCE_DATE_EPOCH` unset and the
  builds at least 1.1 s apart, every executable differs (85 to 119 bytes)
  and every DWARF file differs (14 to 16 bytes). `rustc -O -g` is
  identical in the same directory, in both comparisons. Which bytes these
  are is not recorded.
- Two directories: the second directory's path is 16 bytes longer than the
  first's. Each C, C++ and Fortran executable is 16 bytes longer and its
  DWARF file 32 bytes longer; for rustc, the executable is 32 bytes longer
  and the DWARF file 16. Three copies of the build directory are counted in
  each set of outputs.
- Two directories, remapped: the same size differences, the same three
  copies, and differing byte counts within one byte of the unremapped
  ones. The remapping removed no copy of the directory, consistent with
  change 1 above; whether the copies left after the rerun are the ones the
  remapping cannot reach (the debug map) is what the rerun records.
- `go build, GOPATH mode`, remapped: 119 and 120 differing bytes at equal
  size and no copy of the directory; its trace names
  `_/private{dir}` in the compile and link action IDs, as on Linux;
  `go build, GOPATH mode, import path declared` and `go build` are
  identical across directories when remapped. Which sections or regions
  hold the 120 bytes is not recorded.
- `zig build-exe` (Debug) differs in the same directory (402 bytes at
  equal size, 396 with the clock free); `zig build-exe -O ReleaseSmall` is
  identical in every comparison.
- `ocamlopt -g` differs across directories when remapped (669 bytes, one
  copy of the directory); `BUILD_PATH_PREFIX_MAP` names `{dir}`, so change 1
  applies to it too.

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
python3 -m unittest test_e1
```

`test_e1.py` tests the Mach-O parser on a file built from bytes in the
test, so it runs on any system. `report.py check` fails when `RESULTS.md`
is not what `report.py write` would print from the committed results; the
workflow runs both.

## Limits

- The programs are small: one or two per language. A difference in a large
  program that these do not reach is not tested.
- A cross-machine comparison needs the same tool version string on both
  machines, so it covers only the toolchains whose package versions agree.
- Link-time optimisation, profile-guided builds and parallel code
  generation other than rustc's defaults are not in the matrix yet.
- The harness checks bytes. It does not check that the program's meaning
  was preserved, which is H3 and H7.
- Mach-O attribution covers thin 64-bit little-endian files only. It
  attributes bytes to regions; it does not decode DWARF or the code
  signature.
