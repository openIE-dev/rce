# `examples/`

Small programs that show one feature each, outside the example set of the
specification (`spec/examples/`) and the larger programs of `programs/`.

| Path | What it is |
|---|---|
| `surface/counter.rce` | The counter of the RCE-JEPA paper: a state that never goes negative, an event that does not apply keeps the state (`kept: Underflow`). `rce check` and `rce test` pass. |
| `interop/` | Emitted programs embedded in ordinary services (Python asyncio, FastAPI and numpy; Rust tokio; JavaScript node:http, React and a host with no Node.js runtime), each compared with `rce run` byte for byte. See `interop/README.md`. |
| `engine-spec/` | Intent specifications in the format of the RCE engine's planner, with two of its outputs. They are not RCE source and `rce check` does not read them; see `engine-spec/README.md`. |
