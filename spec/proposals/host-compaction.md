# Proposal: host-compaction

Status: accepted and implemented as HC-37 (Appendix E, E.4i) by the change
on branch `claude/host-compaction` (ledger row of 2026-11-03). Sections 1
to 5 describe what that change builds and checks; section 6 lists what it
does not build.

The host's durable log D (Appendix E, E.3) only grows. Every open verifies
and folds it from the start (HS-4, HC-8), so the cost of opening a host,
and of replay, grows with the whole life of the program. This proposal
adds compaction to the host's core: a later point of D from which a host
and replay can start, with the rule that the state after every entry is
the fold of D kept exactly.

## 0. Terms

- **D**: the host's log (6.1, E.3); **entry N − 1**: the entry of D whose
  seq is N − 1, the last of the first N entries.
- **Checkpoint**: the entry `host.Checkpoint` that takes the place of the
  first N entries of D.
- **D'**: the compacted log: the checkpoint, then the entries of D from
  seq N on, unchanged.
- **Archive**: the replaced lines of D, byte for byte, kept beside D as
  `D.archive.SHA256`, SHA256 being the SHA-256 of those bytes.
- **Fold at N**: the state and the host state a replay of D has reached
  after entry N − 1.

## 1. Source: compaction in the Infinity runtime

Infinity (github.com/hydro-project/infinity, Apache-2.0; Amazon Web
Services and the UC Berkeley Hydro project), whose RAP protocol HC-34 to
HC-36 already serve (`host-rap.md`), compacts the history of a long-lived
agent thread. Its runtime documentation
(`docs/docs/infinity-runtime/architecture.md`, section "Compaction") says
that when a thread's history approaches the model's context window "the
runtime will spawn a compaction thread that summarizes the conversation
and stores the summary in the `ConversationStore`, tagged with the history
index it covers. Subsequent slices will load the summary plus only the
messages after that index. Because summaries are indexed by position,
child threads that were spawned before a compaction can still reconstruct
the exact history they inherited." The history manager's page
(`low-level/history-manager.md`) adds that `apply_compaction()` "tracks
the absolute store index it covers so that a second compaction on top will
compute the right split", and that a spawn point, `safe_spawn_point()`,
"excludes trailing tool calls that have no result yet, so a child will
never inherit a dangling call".

## 2. What was taken and what was not

Taken:

- **A stand-in tagged with the position it covers.** The checkpoint names
  its position: its seq is N − 1, and it carries `head`, the hash of entry
  N − 1, which the next entry of D' links to.
- **Load the stand-in plus only what follows.** A host and replay load the
  checkpoint, then fold only the entries after it.
- **Positions stay absolute.** Seqs in D' are those of D, so every id a
  seq names (an HTTP request id, a RAP call, `host.Restarted` below) is
  unchanged, a restart never reuses an id (HC-22), and a second compaction
  of D' splits at an absolute N.
- **No cut through unfinished work.** As `safe_spawn_point()` leaves out
  dangling tool calls, a checkpoint does not stand where D owes entries
  after entry N − 1 (a kept step's `host.StepKept` and its delivery, a
  restart's delivery, a RAP step's event, a typed input) or between a
  `host.Input` and its input. Requests in flight are not a reason to
  refuse: the checkpoint carries them.
- **The history before the stand-in stays readable.** Infinity's older
  children keep their exact history; here the archive keeps the exact
  lines, and the checkpoint names it by its SHA-256.

Not taken: **the summary.** Infinity's summary is text a model writes. It
is lossy, and nothing can check it against the messages it replaces. A
checkpoint is the opposite: a value, the fold of D at N, admitted only
where the replay of D' from it equals the full replay of D, and provable
again by anyone holding the archive. A summary of state, a checkpoint
taken by a model, or one accepted on a statistical comparison would not
be admissible under HC-37.

## 3. What the checkpoint carries, and why

Each item is something the replay of the entries after N reads. The list
was made by going through every field of the reference host
(`spec/checker/src/host.rs`, `struct Host`) and asking which ones the
fold of D sets and a later entry reads:

| Item | Why |
|---|---|
| `state` | the state in the canonical named form of 6.1, decoded field by field by the declared types; the SHA-256 of its positional form is the state hash unless a task is live. The positional form alone is not carried: it leaves out what only the types say (a variant is its index, an optional value is the value), and a decoding that gave back the same bytes but another value would fold later entries differently |
| `head` | the hash of entry N − 1: the next entry's `prev_hash` (HS-4) |
| `queue` | requests emitted and not answered (outputs of a declared struct with an integer `req`), with the seq that asked: a later result answers one, and a restart interrupts each one a host performs (HC-3, HC-22). All are carried, since which ones a host performs depends on its adapters, and a host opened on D' keeps those its adapters perform |
| `descends` | requests whose results descend from a delivered `StepKept`, so a later kept step is not delivered again (HC-21) |
| `admitted`, `window`, `closed_tick` | per source the highest seq in D, the next window id, the last closed tick: dedupe and windows (HC-15 to HC-17) |
| `term` | the admitter's term; a later `host.Term` must exceed it (HC-19) |
| `task` | a live `fn main` task, part of the state D folds to (HC-6) |
| `calib`, `exacts` | calibrations and exact probes, which placement reads (HC-12, HC-13) |
| `last_tick` | the tick a host loop continues (HC-22) |
| `http` | HTTP inputs still open, as [id, head, close]: a response after N is made from them (HC-26) |
| `rap` | every RAP invocation: its key, which dedupes every later invocation (HC-35); whether it was stepped; its outcome; the length and SHA-256 of its result; the status that settled it (HC-36) |
| `archive` | the SHA-256 of the replaced lines |

Not carried: the text of a kept step (HC-21 keeps it out of D). For a
kept RAP step, whose text a later duplicate's result needs, the checkpoint
carries the state the step was kept on, in its named form, and the
text is derived again from it when the checkpoint is loaded. Not carried
either: what the module and the adapters give (bindings, the start
notices), and what replay only reports (the kept steps before N).

A module that declares `Frame` is not compacted: HC-29 derives its frames
from every entry of D, and already allows no snapshot of it.

## 4. Admission and verification

`rce host compact FILE D N` replays D in full, takes the fold at N on the
way, writes the checkpoint, replays D' from it, and writes nothing unless
the two agree after the last entry on the state hash, on the outputs of
every entry after N, on the kept steps and their texts, on every RAP
result byte for byte, on the HTTP inputs, and on everything else the fold
reaches. A refusal is exit status 2. A D whose tail does not verify (a
torn or corrupted tail) is refused: a host cuts it on open (HS-4).

Loading a checkpoint checks, and fails closed on any difference: that it
is the first entry; its hash over its stored event; that the next entry
links to `head`; that the state decodes and, with the task, hashes to the
checkpoint's state hash (the command also requires the decoded state to
be the value the full replay reached, part for part and type for type);
each kept text derived again with its cause;
each result derived again against its length and SHA-256; and that the
values read back give the payload byte for byte. A host does not open,
and does not cut, a D whose checkpoint fails. `rce host replay` proves
each checkpoint again from its archive when the archive is beside D.

## 5. Retention: `archive none` is refused

The command takes the setting word `archive none`, and refuses it with
exit status 2. HS-1 makes D the base truth and snapshots caches; HS-9
makes every past state queryable from D; HC-20 never removes committed
entries; and law 1 of E.2c makes D checked, not trusted. Discarding the
archive would remove committed entries, end HS-9 for every seq below N,
and leave the checkpoint trusted rather than checked. With the archive,
D' and the archive together hold every line of D, so nothing is removed;
an operator may move the archive elsewhere, since the checkpoint names it
by its SHA-256.

## 6. Not built

- Compaction by the host loop itself, on a size or time rule: compaction
  is an operator's command on a log no host has open (the command reads D
  again before the swap and refuses if it changed).
- Compaction of a replicated D (E.2b): HC-18a's snapshot install is
  unchanged.
- A module that declares `Frame` (section 3).
