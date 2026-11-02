# E5: R, name-free identity across builds (protocol)

Status: protocol and harness, written and dated **2026-11-02, before the
first run**. Tests H5 of [HYPOTHESES.md](../../HYPOTHESES.md): a name-free
hash raises reuse hits on renamed but equal programs, without false hits.
It implements part R of the Compiler Form ([CHARTER.md](../../CHARTER.md)
section 4.2), which the charter records as absent from every compiler the
founding report surveyed: "a canonical representation with an equivalence
relation and a name-free hash: bound names replaced by positions, the build
path removed, then hashed". E5 is independent of RCE: nothing here builds
or calls RCE's checker, Form or specification.

## Sources

Marks are those of the charter (section 2). For the sources of this
experiment the second report of the project, *Lift for replication intent
and classification* (2 November 2026; not in this repository), gives the
marks shown unless the row says the source was read for E5.

| Source | What E5 takes from it | Mark |
|---|---|---|
| Binary Ninja's WARP ([github.com/Vector35/warp](https://github.com/Vector35/warp), README) | the function ID: "The function GUID is the UUIDv5 of the basic block GUIDs", "sorted highest to lowest start address"; "The basic block GUID is the UUIDv5 of the byte sequence of the instructions", "sorted in execution order", after "Zero out all relocatable instructions", "Exclude all NOP instructions" and "Exclude all instructions that set a register to itself if they are effectively NOPs", with "`mov edi, edi` will _not_ be removed" on x86-64; the two namespace UUIDs; "Constraints allow us to further disambiguate between functions with the same GUID" | [F], the README read for E5 on 2026-11-02 |
| WARP constraints | that the constraints a matcher uses are the function's callers, callees and neighbours | [K] |
| Ghidra BSim ([BSim tutorial introduction](https://github.com/NationalSecurityAgency/ghidra/blob/master/GhidraDocs/GhidraClass/BSim/BSimTutorial_Intro.md)) | "values of constants, names of registers, and data types, are intentionally not incorporated" in its features, and differences of compiler and architecture "typically result in vectors which are close but not identical" | [F], read for E5 on 2026-11-02 |
| Maziarz, Ellis, Lawrence, Fitzgibbon and Peyton Jones, "Hashing Modulo Alpha-Equivalence", PLDI 2021 (arXiv 2105.02856; code at [microsoft/hash-modulo-alpha](https://github.com/microsoft/hash-modulo-alpha)) | that a hash of a syntax tree can ignore the names of bound variables, in O(n log² n) time with a proved low collision probability | title and authors [F] from the repository's README, read for E5; venue [K]; the complexity and collision bound [S]; the paper was not read (arxiv.org could not be reached) |
| Unison | that each definition is identified by a hash of its syntax tree with names removed, so a rename does not change its identity | [K]; its documentation could not be reached |

E5's IR-level hash is not the algorithm of Maziarz et al.: it is a
canonical form (positions for bound indices, then SHA-256), which takes the
idea, not the method. Its machine-code level follows WARP's README where the
README is specific and states its own rule where it is not (blocks, the
NOP lists, how a relocated instruction is found, the ID of a whole object).

## Terms

- An *identifier* of a program is a name it binds: a local variable,
  parameter, function, type, constant, static variable or macro in the
  source; an index into one of a module's index spaces in WebAssembly.
- A *free name* is a name a program uses but does not bind: a library
  function, the exported `run`, an import.
- *Alpha-equivalence*: two programs are alpha-equivalent when one becomes
  the other by a one-to-one renaming of bound identifiers.
- A *relocation* is a record in an unlinked object file that tells the
  linker to patch bytes at an offset with an address it decides.
- A *key* is the value a cache would look a result up by. E5 computes two
  baseline keys and four R hashes per build output.

## What R hashes, and how

### The machine-code level ([machine.py](machine.py))

Input: an unlinked object file (the output of `-c` or `--emit=obj`), in
one of three formats:

| Format | Architecture | Functions located by | Relocations read |
|---|---|---|---|
| ELF relocatable | x86-64 | `STT_FUNC` symbols of executable sections, value and size (a size of 0 runs to the next function or the section end); aliases at one address are one function | `SHT_RELA` and `SHT_REL` sections, by the section each applies to |
| Mach-O object (read with E1's `macho_load_commands`) | arm64 | `N_SECT` symbols of sections with `S_ATTR_PURE_INSTRUCTIONS`, except assembler-local labels (names starting `l` or `L`, such as `ltmp0`); a function runs to the next symbol's address or the section end | each section's relocation entries (`r_address`) |
| WebAssembly object | wasm32 | the code section, which delimits every body; names from the `linking` section's symbol table | the `reloc.CODE` section (offsets into the code section) |

An object with no symbol table gets the status "functions cannot be
located: no symbol table" and no hash. (A WebAssembly object without a
`linking` section is still delimited by its code section and is hashed
with no names.)

Per function:

1. **Decode.** x86-64: an instruction-length decoder of the legacy, REX
   and VEX encodings (EVEX and 3DNow! are reported as not decoded). arm64:
   fixed 4-byte words. wasm32: the instruction decoder of
   [irhash.py](irhash.py). A function that does not decode gives its
   object the status "not decoded" with the reason.
2. **Blocks.** x86-64 and arm64: a new block starts at the function's
   start, at the target of every direct branch inside the function, and
   after every conditional or unconditional branch, return, indirect jump
   and trap (`ud2`, `int3`, `hlt`, `brk`); a call does not end a block.
   wasm32 has structured control, so a block ends after every `end`, `br`,
   `br_if`, `br_table`, `return` and `unreachable`, and starts again at
   every `block`, `loop`, `if` and `else`; the local declarations belong to
   the first block.
3. **Zero relocatable instructions.** An instruction is relocatable when
   a relocation entry's offset falls inside its bytes; all its bytes are
   replaced by zero bytes of the same length. This is the rule for unlinked
   objects. Linked executables and linked modules carry no relocation
   records, so their relocatable operands cannot be told from other
   constants without a model of the address space; E5 does not hash them
   at this level.
4. **Leave out NOPs and self-moves**, by this declared list:

   | Architecture | Left out |
   |---|---|
   | x86-64 | `0x90` without REX.B and without an F3 prefix (F3 90 is PAUSE); `0F 1F /0` with any prefixes (multi-byte NOP); `MOV r64, r64` (REX.W `89` or `8B`, register form) with the same source and destination. `MOV r32, r32` to itself is kept: it clears the upper 32 bits |
   | arm64 | `NOP` (`0xD503201F`); `MOV Xd, Xd` (`ORR Xd, XZR, Xd`); `MOV SP, SP` (`ADD SP, SP, #0`). `MOV Wd, Wd` is kept: it clears the upper 32 bits |
   | wasm32 | `nop` (`0x01`); WebAssembly has no register self-move |

   A block left empty is left out.
5. **IDs.** Each block's ID is UUIDv5 in WARP's block namespace
   (`0192a178-7a5f-7936-8653-3cbaa7d6afe7`) of its remaining bytes in
   address order; the function's ID is UUIDv5 in WARP's function namespace
   (`0192a179-61ac-7cef-88ed-012296e9492f`) of its block IDs' 16 bytes each,
   from the highest start address to the lowest.

Two R hashes come from these IDs:

- **`machine-run`**: the ID of the function named `run` (`_run` on
  Mach-O). The symbol name is used only to choose the function.
- **`machine-module`**: SHA-256 of the sorted list of the IDs of every
  function of the object. Names do not enter it. WARP's constraints are
  not implemented, so two functions with one ID are not told apart.

### The IR level ([irhash.py](irhash.py))

Input: a linked WebAssembly module (the E4 build). The declared IR is
WebAssembly 1.0 in its binary format with the sign-extension, multi-value,
bulk-memory and reference-type instructions; SIMD and exception handling
are reported as not hashed. A module binds its identifiers as indices into
six spaces (types, functions, tables, globals, data and element segments)
and per function the locals. Its free names are its import and export
names, which are its linking interface and are hashed as text. Labels are
relative depths already, de Bruijn indices in WebAssembly's own design, and
are kept. Custom sections (the names section, producers, target features)
are left out, so a renamed symbol in the names section does not reach the
hash.

The canonical form:

1. A type index becomes the signature it names.
2. Locals by position: parameters keep their positions (the calling
   convention); every other local is numbered by its first reference in
   the body, its declared type following that order; locals never referred
   to are written as a sorted list of types.
3. Functions by call-graph shape: each defined function starts as a hash
   of its export names, signature and body with callees left blank; each
   round replaces it by a hash of itself and its body with each callee
   written as the callee's hash from the round before (colour refinement in
   the Weisfeiler–Lehman style); rounds stop when one splits no class.
   Imported functions are anchored by their import names. Defined functions
   are ordered by the final hash; a call becomes the callee's position in
   that order. Functions with one final hash keep their order in the module
   and are counted as *ties*.
4. Globals are numbered by first reference in the canonical function order,
   then by exports; unreferenced globals follow, sorted by type, mutability
   and initial value.
5. The order of independent definitions is normalised: imports and exports
   sort by name, active data and element segments by offset and contents,
   passive data segments by first reference.

**`ir-module`** is SHA-256 of a prefix-free byte encoding of this form.
**`ir-run`** is the same for the part of the module that the export `run`
reaches: the functions reachable from it by calls, `ref.func` and (when it
makes an indirect call) the table, the globals they name, and the memory,
data segments and table; the other exports and unreachable functions are
left out.

Properties, tested in [test_e5.py](test_e5.py) on modules built
instruction by instruction: permuting every index space and changing every
name in the names section leaves `ir-module` and `ir-run` unchanged (40
random renamings); changing any one opcode to another of its class, any
constant, any memory offset, a global's initial value, a data byte, an
export name, or which function a call names changes `ir-module`.

What equality means: equal `ir-module` hashes mean equal canonical forms
(barring a SHA-256 collision), and the canonical form determines the module
up to a permutation of its bound indices and its custom sections, so the
two modules are alpha-equivalent. The converse can fail: two
alpha-equivalent modules whose tied functions refer to different but
equivalent parts could be ordered differently; the tie count of every
module is recorded.

## Corpus and builds

- **E4's corpus**, fixed by [../e4-energy/matrix.json](../e4-energy/matrix.json)
  (whose SHA-256 the results record): 8 kernels in C and in Rust, at
  `clang -O0, -O1, -O2, -O3, -Os, -Oz` and `rustc opt-level=0, 1, 2, 3, s,
  z`. Per row four outputs ([matrix.json](matrix.json)): an x86-64 host
  object (`-c`; Rust `--crate-type lib --emit=obj`), an arm64 Mach-O object
  (C only, `clang --target=arm64-apple-macos11 -ffreestanding -c`; no Rust
  target for it is installed), a WebAssembly object (`-c`, `--emit=obj`)
  and the linked WebAssembly module of E4.
- **E1's C and Rust programs**: `c/hello.c`, `c/sort.c`, `rust/hello.rs`,
  `rust/generic.rs`, at the same six levels, host object only. They print
  through the C and Rust libraries, so they have no wasm32 build. E1's
  other languages have no renamer here.
- **Variants.** Each source is written to directory A, its same bytes to
  directory B, and its renamed copy to directory N; the three paths have
  equal length and differ in one character. Source paths are absolute, as
  in E1 and E4, so a compiler that records the path records a different one
  in each directory.

### Renamed sources ([rename.py](rename.py))

The renamer lexes C or Rust and renames every identifier the program binds
to `n<k>_` and 8 hexadecimal digits of the SHA-256 of the old name, k
counting distinct identifiers in order of first appearance; it removes
comments. It keeps keywords, preprocessor directive names, built-in and
library type names, `run` and `main`, and every identifier that the file
uses in a library position (after `.`; in Rust also next to `::`, before
`!`, and captured by a format string), everywhere in the file. The renaming
depends on the source text alone.

*Evidence that the renamed source is the same program*: for every E4 row,
the renamed module gives `reference.py`'s value for each declared input of
the kernel (run by E4's `wasm_eps.py`), and so does the original; for E1's
programs, the original and the renamed program, built at the lowest level
and run, print the same bytes. A renamed pair without this evidence is
reported as such.

## Keys

| Key | Applies to | Definition |
|---|---|---|
| `output-bytes` (baseline) | every output | SHA-256 of the output file: a cache keyed by output bytes |
| `source-key` (baseline) | every output | SHA-256 of the source bytes and the build command with the build directory written `{dir}`: a cache keyed by source text and flags with the path remapped |
| `machine-module`, `machine-run` | objects | as above |
| `ir-module`, `ir-run` | linked WebAssembly modules | as above |

H5's metric names the compiler's own cache key. E5 does not read ccache's,
rustc's or any other compiler's cache keys; the two baselines stand for
the two kinds of key the charter describes, keys over bytes and keys over
named source text with the path removed (charter, part U).

## Comparisons and predictions

| Comparison | Pairs | Declared | Prediction |
|---|---|---|---|
| `two_dir` | A against B, per program, level and output | equal | every R hash equal in every pair |
| `renamed` | A against N | equal | every R hash equal in every pair with rename evidence; `source-key` unequal in every pair; `output-bytes` unequal in some |
| `opt_levels` | A at two levels of one toolchain, every pair of the six levels | not equal | R hashes unequal in most pairs; the rate of equal pairs is reported, and an equal pair means the two levels produced identical code (canonically), which is not a false hit |
| `c_vs_rust` | the C and the Rust kernel of one algorithm at the same position in the list of levels | not equal | machine-code R hashes unequal in every pair; `ir-run` also unequal in every pair, since the two front ends produce different LLVM IR |
| `different_kernels` | every pair of A builds of two different E4 kernels, per output | not equal | no R hash equal in any pair |
| `mutations` | each A module against itself with one opcode changed within its class, one constant changed or one memory offset changed | not equal | `ir-module` changes in every case |

The rate reported per comparison and key is the share of pairs with equal
keys; for a comparison declared equal it is the rate of equality where
equality is declared, and for one declared not equal, one minus it is the
rate of inequality where inequality is declared.

**H5 by these comparisons.** A *hit* is an equal key on a `two_dir` or
`renamed` pair; a *false hit* is an equal key on a `different_kernels`
pair (two programs that compute different functions) or an unchanged
`ir-module` under a mutation. An equal key across `opt_levels` or
`c_vs_rust` is counted and reported, not counted as false, because those
pairs are the same program. *Supported* if, on the `renamed` pairs, every
R hash has a higher hit rate than both baselines, and there is no false
hit. *Falsified* if any R hash has a false hit, or if no R hash has a
higher hit rate than both baselines on the `renamed` pairs.

## Equivalence levels, linking assumption and trusted base

| R hash | What equality establishes | What it does not |
|---|---|---|
| `ir-module` | the two linked modules are alpha-equivalent: the same instructions, constants, data and interface up to a permutation of indices. Their behaviour under the WebAssembly semantics is therefore the same for every input, which is the **exact** level for integer results and **bitwise within an accumulation class** for floating point (the evaluation order is the same), except where the WebAssembly semantics itself leaves a NaN's bits open | anything about the source, about the native code built from the same source, or about modules that differ: unequal hashes say nothing about equivalence (the `opt_levels` pairs are the same program with different hashes) |
| `ir-run` | the same for the part reachable from `run`, with the memory and data included | anything about the other exports |
| `machine-run`, `machine-module` | the function bytes are equal after relocatable instructions are zeroed and NOPs and self-moves are left out | **no equivalence level**: a zeroed instruction may call or load a different symbol, and the hash does not see which, so two functions with one ID can compute different things once linked; WARP leaves the choice to its constraints |

If an R hash keys reuse (rung G0 of the gear ladder), the gate statement
the charter requires (section 4.4) is:

- **Property class**: for `ir-module` and `ir-run`, equivalence at the
  exact level of the whole module (or of the part `run` reaches), which
  preserves every trace property and hyperproperty observable through the
  module's interface; for the machine-code IDs, none on their own.
- **Linking assumption**: the module is linked and called by its import
  and export names only, by a host that does not read custom sections; for
  the machine-code IDs, that every zeroed relocation resolves to the same
  target in both objects, which the hash does not check.
- **Trusted computing base**: `irhash.py` and `machine.py` (the decoders,
  the canonicaliser and the relocation readers), E4's `wasm_eps.Reader`,
  E1's Mach-O reader, Python's `hashlib` (SHA-256) and `uuid` (UUIDv5,
  over SHA-1), and the WebAssembly specification as the semantics; the
  engine that later runs the module is outside what the hash decides.

## Files and running it

| File | Contents |
|---|---|
| [matrix.json](matrix.json) | toolchains, levels, build commands per output, E1's programs, the keys and the comparisons |
| [irhash.py](irhash.py) | the IR-level hash and the mutations |
| [machine.py](machine.py) | the object readers, the decoders and the WARP-style IDs |
| [rename.py](rename.py) | the renamer |
| [e5.py](e5.py) | the harness, which writes `results/*.json` |
| [report.py](report.py) | writes [RESULTS.md](RESULTS.md) from the results files and checks it |
| [test_e5.py](test_e5.py) | unit tests |

On a Linux x86-64 host, after `sudo apt-get install clang lld llvm` and
`rustup target add wasm32-unknown-unknown`:

```sh
cd compile/experiments/e5-name-free
python3 test_e5.py
python3 e5.py run results/NAME.json
python3 report.py write
python3 report.py check
```

When `llvm-objdump` is on PATH, the harness also compares the instruction
boundaries of the x86-64 decoder with llvm-objdump's for every A host
object and records the agreement. `report.py check` fails when `RESULTS.md`
is not what `report.py write` prints from `results/*.json`. The workflow
[`.github/workflows/compile-e5.yml`](../../../.github/workflows/compile-e5.yml)
runs the unit tests and `report.py check`.

## Limits

- The machine-code level reads unlinked objects only, two native
  architectures and wasm32; it does not implement WARP's constraints.
- The x86-64 decoder covers what is listed above; its boundaries are
  checked against llvm-objdump, not against a specification.
- The IR level hashes WebAssembly, the one IR of this corpus that both
  toolchains emit and that the project already parses; it is not LLVM IR,
  where the two front ends would be compared before their back ends.
- Ties in the function order are counted, not resolved.
- The corpus is small (16 kernels and 4 programs), and the renamer covers
  the constructs that occur in it.
