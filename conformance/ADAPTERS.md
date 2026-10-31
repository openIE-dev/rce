# Adapter protocol rce-adapter/1

An adapter is a process. The harness writes one JSON request per line to its
stdin and reads one JSON response per line from its stdout. The adapter file
`adapters/<name>.json` gives `cmd` (env vars expanded), `ops` and `features`.

Requests (`op`):

- `hello` -> `{"protocol":"rce-adapter/1","implementation":...}`; the
  reference adds `commit` (the commit it was built from; there is no typed
  version), an emitter adds `emitted_by` (the
  checker named in an emitted file's first line) and `toolchain` (the
  target's version line); the harness records the answer in the results
- `check {files}` -> `{"ok":true}` or `{"ok":false,"diagnostics":[...]}`
- `run {source, events?, fuel?, schedule_seed?}` -> `{"outcome", "outputs"}`
- `trace {source, events}` -> `{"head", "lines"}`: the whole trace of the
  example the events form. `lines` is every line an emitted program run
  with `--trace --fuel-trace` prints for that example, after its
  `example LINE` header: the state after each event, each entry of D as
  `d applied HASH BODY` or `d kept HASH BODY` before the event's line
  (BODY the canonical JSON `{event, prev_hash, seq, state_hash}`, HASH its
  SHA-256), the outputs, the reason an event was kept, `trace HEAD` and
  `fuel N`. `head` is the hash of the last entry. The harness compares the
  lines with the test's `trace_lines` byte for byte
- `host {source, steps, tear?}` -> live, replay, reopen state hashes and torn results
- `replicate {source, nodes, script}` -> per-node committed logs and terms

Any response may be `{"unsupported": reason}` or `{"error": message}`.

Shipped adapters: `reference` (the `rce-conform` binary from
`spec/checker`), `emit-rust`, `emit-python`, `emit-js`,
`emit-scala` (via `adapters/emit_adapter.py TARGET`: emit, compile, run
with `--run --fuel=N`, events as appended `example` blocks, traces with
`--trace --fuel-trace`), `wasm-playground`
(check only, via node), `lean-core` (`rce-conform-lean` from
`spec/checker`: reference entry, then the Lean core semantics in
`spec/lean` on the elaborated `fn main`; run op only, programs outside the
core subset are `unsupported`; env `RCE_CONFORM_LEAN`, `RCE_LEAN_DIR`). Emitted programs cannot observe a trap on the last
event (see finding EXAMPLE-TRAP), so those cases are `unsupported`.
