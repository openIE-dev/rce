# Systems: buffers and control loops

Programs of the kind found in operating systems and infrastructure tools.

| Program | What it shows |
|---|---|
| `ring_buffer.rce` | a fixed-size ring buffer that never overwrites when full |
| `reconcile.rce` | a reconcile loop that moves the actual state one step towards the desired state |
| `accounts.rce` | a long-running account service on the host: an unbounded run of bounded steps, where a transfer that fails half way keeps the state |
| `ls.rce` | `ls`, `ls -a`, `ls -l` and `ls -la` brought in from an existing Go program: a pure core of mode strings, names and rows, and every name, record, link text, owner, group, clock reading and UTC offset read through host requests |

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
return Err }` guards and field assignments. On `ls.go` it printed an empty
Form (four blank lines, lift hash `6a3cf5192354f716`) and exited 0. It
carried none of the program: `ls.go` has no type declaration, no method
and no `errors.New` variable, and the probe looks at nothing else. It also
reported nothing as requiring change: the 3 function declarations, 4
`for ... range` loops, 2 `switch` statements, 8 `if` statements, 32 calls
into `os`, `os/user`, `syscall`, `fmt`, `strings`, `strconv` and
`path/filepath`, the type assertion to `*syscall.Stat_t` and the `[10]byte`
array were dropped without a word. The Python lifter names a rule and a
line for every construct outside its subset; the Go probe does not yet, so
an empty result from it does not mean a program is empty. The probe still
reproduces `door.lifted.rce` from `door.go` exactly.

**The program.** `ls.rce` is written by hand:

- the mode string (`mode_string`, GNU's, with `s S t T`; and
  `ref_mode_string`, the reference's case 8);
- the visible names (`visible`: cases 4 and 5, with `.` and `..` sorted
  among the other names in byte order, as GNU `ls` does in the C locale);
- the reference's long row (`row`: cases 8 to 10, the time from seconds
  and a UTC offset through the standard library's `time` module);
- exercise 1, GNU's columns (`aligned_row`, `widths`): the link count and
  size right-aligned, owner and group left-aligned, and GNU's half-year
  rule, to the nanosecond, which shows the year instead of the time for an
  old or future file; with GNU's `total` line (`total_line`);
- exercise 2, `name -> target` (`name_field`).

Names and link texts are bytes (`List[i64]`), never text, so a name that
is not UTF-8 is listed exactly, and the output is written as bytes
(`Display(data: List[i64])`). 34 `check` lines pin the core, among them
byte-exact rows for a fixed set of five entries and two names that differ
only in bytes that are not UTF-8, and four examples drive the steps.
Cases 1 and 2 do not carry over: RCE's command line takes words, never
flags, so the path, `all` and `long` are the fields of the `Ls` event.
Case 3 becomes a `FileListDone` whose status is not `delivered`, which
counts in `failed` and prints `ls: ` and the host's error text. The mode
string is written here rather than taken from the standard library's
`os.perm.filemode` because a module cannot `use` both `os.perm` and `time`
today: both declare a function `digits` (W-MOD-3, "imported names are not
renamed yet").

**On the host.** `rce host run fs DIR names ls.rce D`, given
`Ls(path: "d", all: true, long: true)`, makes only host requests (Appendix
E): `FileList` for the names, as bytes (HC-30), and `WallClock` for the
time now (HC-32); then for each name `FileLstat`, which does not follow a
final symbolic link, and `FileReadlink` for a link's text (HC-31);
`LocalOffset` for the UTC offset of the host's zone at its mtime (HC-32);
and `UserName` and `GroupName`, which exist only under the grant word
`names` (HC-33). Each answer is a result event in D (HC-10), so `rce host
replay ls.rce D` re-derives the listing with every adapter disconnected.
A uid or gid without a name is printed as the number, as GNU `ls` does.
The OS target the host enters reads only under the log directory, `/dev`
and DIR, and the two account databases only under `names` (HC-5a).

The listing of DIR itself (`path: ""`) has no `..` row: `..` of the
granted root is outside it, so its lstat is refused. List a directory
under DIR instead, as the test below does. Device files are listed with
their size, not GNU's `major, minor`, because the lstat record carries no
device number.

**Against coreutils.** [`ls_coreutils.sh`](ls_coreutils.sh) makes
`WORKDIR/t/d` with two regular files and a hard link to one, a dotfile, a
set-user-ID file, a named pipe, a subdirectory, a sticky directory, a
symbolic link, a dangling one and one whose target leaves the granted
root, the names `x\376` and `x\377` and `caf\351`, which are not UTF-8, and,
when run as root, a file whose owner and group have no name. Modes are set
with `chmod` and times pinned with `touch` in the zone
`EST5EDT,M3.2.0,M11.1.0`, so the files of October 2026 are at UTC-4 and
the file of January 2020 at UTC-5. It runs the program on the host under
`fs WORKDIR/t names` for `ls`, `ls -a`, `ls -l` and `ls -la` of `d`, and
compares the bytes of its output with coreutils `ls` in the C locale. On
1 November 2026, with GNU coreutils 9.4 on Linux, all four are equal byte
for byte (14, 17, 15 and 18 lines), `total` line, columns, `s` and `t`,
`->` targets, the numeric owner, the bytes that are not UTF-8 and the year
of the 2020 file included, and the log of `ls -la` (90 entries) replays.

One difference from GNU `ls` follows the reference and is not exercised by
the test: an unknown file type is `-` in `row` but `?` in `mode_string`.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
