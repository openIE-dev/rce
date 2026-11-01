# Systems: buffers and control loops

Programs of the kind found in operating systems and infrastructure tools.

| Program | What it shows |
|---|---|
| `ring_buffer.rce` | a fixed-size ring buffer that never overwrites when full |
| `reconcile.rce` | a reconcile loop that moves the actual state one step towards the desired state |
| `accounts.rce` | a long-running account service on the host: an unbounded run of bounded steps, where a transfer that fails half way keeps the state |
| `ls.rce` | `ls`, `ls -a` and `ls -l` brought in from an existing Go program: a pure core of mode strings, names and rows, and a directory read through the host's `fs` grant |

**A service as a stream of bounded steps.** `accounts.rce` has no main
loop. The host (`rce host serve accounts.rce D`, Appendix E) reads
one line of events at a time from stdin, applies each as one step and
answers on its Display sink: `ok`, or `kept SEQ CAUSE` from the `StepKept`
event the host delivers after a kept step (HC-21). The run ends when the
host stops; each step is bounded, because the state holds at most
`MAX_ACCOUNTS` accounts and the one loop (a fee charged to every account)
runs over them, and the host gives every input its own fuel (HC-7a).

A transfer writes the debit before it checks the credit, so a credit over
`CAP` fails in the middle of the step. The event then does not apply to
any region: the debit is not kept, and the `Audit` count of transfers does
not move (6.4, T8). `accounts.cmd` shows a kept transfer; an `example` line
cannot state the state after a kept event (6.9: a kept outcome is the last
step of an example), so the property is checked on the host's log D instead. In
openie-research-experiments `rce/bench/bounded-programs`, 100,000 events
driven one at a time through `rce host serve` gave a median reply time of
4.0 ms in every block of 10,000 (the host's durable append per input is
most of it), and each of the 17,766 kept steps left the state hash
unchanged.

## `ls.rce`: an existing Go program brought into RCE

**The reference.** [tonybanters/go-lsbtw](https://github.com/tonybanters/go-lsbtw)
is a from-scratch `ls` in Go: one file, `ls.go` (109 lines), and a README
from a tutorial video. The repository has no licence file, so no code is
taken from it: `ls.rce` is written for RCE, and go-lsbtw is cited as the
reference behaviour only. Read as text, `ls.go` does the following:

1. With no path argument it lists `.`; an argument that does not start
   with `-` is the path, and the last one given wins.
2. An argument that starts with `-` is a group of letters: `a` and `l`
   combine (`-a`, `-l`, `-la`, `-al`); any other letter prints a usage
   line on stderr and exits 1; a lone `-` changes nothing.
3. A directory that cannot be read prints the error on stderr and exits 1.
4. The names come from `os.ReadDir`, which drops `.` and `..` and sorts by
   bytes.
5. Without `a`, a name that starts with `.` is left out. With `a`, `.` and
   `..` are printed first, then every name.
6. Without `l`, one name per line.
7. With `l`, each name is `os.Lstat`ed (a symbolic link is not followed);
   an `Lstat` error prints `name: error` on stderr and the listing goes on.
8. The mode string is a type letter (`d l c b p s`, otherwise `-`) and the
   nine `rwx` places of the 0777 bits; set-user-ID, set-group-ID and sticky
   are not shown.
9. Then the link count, the owner and group names of the numeric user id
   (uid) and group id (gid) (`user.LookupId`, `user.LookupGroupId`, `?`
   when the lookup fails), the size in bytes,
   and the modification time in the local zone in Go's layout
   `Jan _2 15:04` (always the time of day, never the year).
10. Every field is separated by one space, the name is printed alone (no
    `-> target`), and there is no `total` line.

Its README leaves two exercises: right-aligned columns, and `link ->
target` for a symbolic link.

**The lift.** RCE's one Go lifter is the probe
[`spec/lift/go/lift_go.py`](../../spec/lift/go/lift_go.py), which reads Go
source as a tree-sitter syntax tree (it runs nothing) and lifts one
pattern: a struct with one pointer-receiver method per event, `if field {
return Err }` guards and field assignments. Until 2026-11-01, on `ls.go` it printed an empty
Form (four blank lines, lift hash `6a3cf5192354f716`) and exited 0. It
carried none of the program: `ls.go` has no type declaration, no method
and no `errors.New` variable, and the probe looks at nothing else. It also
reported nothing as requiring change: the 3 function declarations, 4
`for ... range` loops, 2 `switch` statements, 8 `if` statements, 32 calls
into `os`, `os/user`, `syscall`, `fmt`, `strings`, `strconv` and
`path/filepath`, the type assertion to `*syscall.Stat_t` and the `[10]byte`
array were dropped without a word. The probe
now does what the Python lifter does: it names a rule and a line for every
construct it does not carry (`L-GO-FUNC` for a function, `L-GO-FOR`,
`L-GO-SWITCH`, `L-GO-IF`, `L-GO-CALL`, `L-GO-ASSERT`, `L-GO-COLLECTION` for
an array, `L-GO-IMPORT` for a package other than `errors`), prints no Form
and exits 1, so a program like `ls.go` requires change instead of lifting
to an empty Form. `ls.go` is not in this repository; the probe's report on
its own fixture with these constructs is
`spec/lift/go/fixtures/counter.requires-change`. The probe still
reproduces `door.lifted.rce` from `door.go` exactly.

**The program.** `ls.rce` is written by hand:

- the mode string (`mode_string`, GNU's, with `s S t T`; and
  `ref_mode_string`, the reference's case 8);
- the visible names (`visible`: cases 4 and 5);
- the reference's long row (`row`: cases 8 to 10, the time from seconds
  and a UTC offset through the standard library's `time` module);
- exercise 1, GNU's columns (`aligned_row`, `widths`): the link count and
  size right-aligned, owner and group left-aligned, and GNU's half-year
  rule, which shows the year instead of the time for an old or future
  file; with GNU's `total` line (`total_line`);
- exercise 2, `name -> target` (`name_field`).

31 `check` lines pin these, among them byte-exact rows for a fixed set of
five entries, and four examples drive the steps. Cases 1 and 2 do not
carry over: RCE's command line takes words, never flags, so the path and
`all` are the fields of the `Ls` event. Case 3 becomes a `FileListDone`
whose status is not `delivered`, which counts in `failed` and prints
`ls: ` and the host's error text. The mode string is written here rather than taken from the
standard library's `os.perm.filemode` because a module cannot `use` both
`os.perm` and `time` today: both declare a function `digits` (W-MOD-3,
"imported names are not renamed yet").

**On the host.** `rce host serve fs DIR ls.rce D`, given the line
`Ls(path: "", all: false)` on stdin, sends one `FileList` request under the
`fs DIR` grant (HC-3, HC-5) and prints the names on the Display sink;
stdout is that listing alone, since the host writes its closing status
line, `durable N entries=M state_hash H`, to stderr (HC-28). A
long listing cannot be made on the host: its stat records come in a
`Stats` event, typed by whoever writes the input (HC-10b), because these
effects do not exist:

| Gap | What is missing | The rule it would need |
|---|---|---|
| lstat | a file request that answers mode, link count, uid, gid, size, mtime and blocks without following a final symbolic link | HC-3 (one typed result event per request), under the `fs` grant of HC-5 |
| readlink | a file request that answers a link's text; the file adapter resolves a path by canonicalising it, so a link whose target leaves the root is refused even though neither lstat nor readlink follows it | HC-3, and HC-5's root check applied to the parent directory, not the target |
| user and group names | no adapter turns a uid or gid into a name (`/etc/passwd`, `/etc/group`, or the other sources of the Name Service Switch) | HC-3 and a capability of HC-5 for the account databases, the answer logged as HC-10 requires |
| local time | time enters only as events, and `WallTime(ns)` carries no UTC offset; the zone's offset at each mtime (the rules of its time zone file) is not an input | HC-2 (clocks only as events) and HC-10: the offset delivered as an input or a logged answer |
| exact names | `FileList` answers names as text made by a lossy conversion: the two names `a\xfeb` and `a\xffb` both arrive as `a\u{fffd}b` | HC-10a (bytes delivered exactly): names as bytes |

**Against coreutils.** [`ls_coreutils.sh`](ls_coreutils.sh) makes a
directory with two regular files and a hard link to one, a dotfile, a
set-user-ID file, a named pipe, a subdirectory, a sticky directory, a symbolic
link and a dangling one, modes set with `chmod` and times pinned with
`touch` in UTC. GNU `stat` and `readlink` give each entry's record (the
inputs), and coreutils `ls` in the C locale gives the expected bytes. On
1 November 2026, with GNU coreutils on Linux: the short listing through the
host's `FileList` equals `ls -1` (9 names) and `ls -1a` (12 names), and the
Display text of the pure core equals `ls -la` (13 lines) and `ls -l` (10
lines) byte for byte, `total` line, columns, `s` and `t`, `->` targets and
the year of the 2020 file included. The first run differed: with `all`,
`.` and `..` appeared twice, because the stat records include the
directory's own two and `visible` added them again; `visible` now drops a
`.` or `..` among its input names, and a check pins it.

Two differences from GNU `ls` follow the reference and are not exercised
by the test: `.` and `..` always come first (GNU sorts them with the
other names, so a name that sorts before `.` in bytes, such as
`-x`, comes before them), and an unknown file type is `-` in `row` but `?`
in `mode_string`. For a uid or gid without a name the reference prints `?`
where GNU prints the number; `ls.rce` prints the name its `Entry` record
carries.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
