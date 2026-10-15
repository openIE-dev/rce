# Adapter protocol rce-adapter/1

An adapter is a process. The harness writes one JSON request per line to its
stdin and reads one JSON response per line from its stdout. The adapter file
`adapters/<name>.json` gives `cmd` (env vars expanded), `ops` and `features`.

Requests (`op`):

- `hello` -> `{"protocol":"rce-adapter/1","name":...}`
- `check {files}` -> `{"ok":true}` or `{"ok":false,"diagnostics":[...]}`
- `run {source, events?, fuel?, schedule_seed?}` -> `{"outcome", "outputs"}`
- `trace {source, events}` -> `{"head", "steps"}`
- `host {source, steps, tear?}` -> live, replay, reopen state hashes and torn results
- `replicate {source, nodes, script}` -> per-node committed logs and terms

Any response may be `{"unsupported": reason}` or `{"error": message}`.

Shipped adapters: `reference` (the `rce-conform` binary from
`crates/rce/spec/checker`), `emit-rust`, `emit-python`, `emit-js`,
`emit-scala` (via `adapters/emit_adapter.py`: emit, compile, run with
`--run --fuel=N`, events as appended `example` blocks), `wasm-playground`
(check only, via node). Emitted programs cannot observe a trap on the last
event (see finding EXAMPLE-TRAP), so those cases are `unsupported`.
