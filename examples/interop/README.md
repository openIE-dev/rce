# Emitted programs in ordinary hosts

A program printed by `rce emit rust|python|js FILE` is the step function of
the module (spec 10.5, "Embedding an emitted program"): a host calls the
initial state once and `delta(state, event)` once per input, and the host
contract of Appendix E is met around it. This directory holds one host per
language, built on that language's usual libraries, and a test that holds
each one to the reference.

| Language | Host | Event source | Files |
|---|---|---|---|
| Python | asyncio service | newline-delimited JSON on an asyncio TCP stream | `python/service.py` |
| Python | FastAPI service | `POST /events`, driven by FastAPI's test client | `python/fastapi_app.py` |
| Python | numpy values | `numpy.int64` and `numpy.float64` scalars handed to the adapter, no JSON | `python/numpy_drive.py` |
| Rust | tokio service | newline-delimited JSON on a tokio `TcpListener`; one admitter task owns the state | `rust/src/main.rs` |
| JavaScript | `node:http` service | the server's `request` handler, driven by `fetch` | `js/service.mjs` |
| JavaScript | React component | a `rce-event` listener on an `EventTarget`; `delta` is the `useReducer` reducer, under `StrictMode` | `js/Program.mjs`, `js/react_drive.mjs` |
| JavaScript | a realm with no `process` (as in a browser) | a loop over the stream, in `node:vm` | `js/browser_drive.mjs` |

Each language has one adapter, written once for every program:
`python/rce_host.py`, `js/rce_host.mjs` and `rust/src/adapter.rs`. It reads
an event from JSON by the program's own field table (`EVENT_FIELDS`,
`event_fields`), checks integer ranges with the runtime's `chk`, calls the
step after resetting the fuel (HC-7a), and writes the lines `rce run` prints
and the state as JSON. A kept step leaves the state as it was (HC-21).

`hand/` holds the same eight programs written by hand as idiomatic Python,
JavaScript and Rust, run through the same comparison, so that the emitted
code can be compared with what a person would write.

## The test

```sh
RCE=spec/checker/target/release/rce python3 examples/interop/check.py
```

For each program and event stream of `streams.json` (eight programs of
`spec/examples/`, eleven streams), the expected bytes are what
`rce run FILE EVENTS` prints. Each host prints the transcript of the
stream it was sent, and passes only when the transcript equals those bytes.
A stream with a kept step ends with it, because `rce run` stops at the
first kept step while a host goes on with the state unchanged. A host whose
toolchain or library is missing is reported as not run, never as passed;
`INTEROP_REQUIRE=python,rust,...` makes it a failure, as the `interop`
workflow does. The React host needs `npm ci` in `js/`; the FastAPI host
needs `fastapi` and `httpx`, and the numpy host `numpy`.

## What was in the way, and what changed

Before this directory existed, no emitted file could be embedded as
printed (measured on `main` at 8ce91cea): imported, the Python module
printed every example's observations to standard output (2 of 11
transcripts equal, the two programs without examples); the JavaScript
module exported nothing (0 of 11) and stopped with a `ReferenceError`
where there is no `process`; and the Rust file did not compile as a
module, because `initial_state`, `show_state` and `event_fields` were
private (E0603). Three small changes to the emitter fixed these, and every
emit test still passes:

1. Python: the observations run under `if __name__ == "__main__":`.
2. JavaScript: the observations run only when `node` runs that file; the
   module exports its step API; `worker_threads` and `crypto` are loaded
   only where `process` exists.
3. Rust: `initial_state`, `show_state` and `event_fields` are `pub`.

These remain, each with the change that would remove it:

- **Every emitted file carries its own runtime.** About 2,000 lines of
  runtime come before the program in each language. In Rust each file has
  its own value type `V`, so two programs in one binary have two runtimes
  and two incompatible value types, and the adapter is instantiated once
  per program (`include!("adapter.rs")` in each module). Proposed: an emit
  target for the runtime alone (a crate, a Python package, an npm module)
  and a program target that imports it.
- **The Rust file is a crate root, not an includable module.** It opens
  with an inner attribute (`#![allow(...)]`), so `include!` cannot take it,
  and `#[path]` takes only a literal, so a `build.rs` cannot place it in
  `OUT_DIR`; `check.py` copies it into `rust/src/gen/`. Proposed: print the
  allowances as outer attributes on the items.
- **A kept step is a string.** `delta` reports a kept step as the display
  text of its reason (`Insufficient`, `trap(overflow: ...)`): the declared
  error's payload (`need: 70`) and the cause of HC-21 (fail, fuel, size,
  value, depth or trap) are lost, and a host that needs them parses text.
  Proposed: a kept record {cause, error value, text}.
- **The values are the runtime's, not the language's.** State is a list of
  (region, [(field, value)]) pairs of runtime classes (`Int`, `Rec`, `Mp`,
  BigInt in JavaScript, `V::Int(i128, &'static str)` in Rust), and an event
  is built from the runtime's classes and a table of type names. There are
  no typed records, dataclasses or structs, and no accessor such as
  `state.Account.balance`. Proposed: per-program constructors for events
  and typed views of the state, printed beside the step.
- **No JSON reader or writer at the boundary.** Each adapter writes its own
  event reader (the larger part of each adapter) although the runtime
  already writes canonical JSON for the log D (`fv_json`). In JavaScript,
  integers are BigInt inside the program, so `JSON.parse` loses digits
  above 2^53 and `JSON.stringify` throws on a BigInt and writes -0 as `0`
  (which merges the two keys of `58_float_keys`); the adapter reads numbers
  from their source text and writes them with `JSON.rawJSON` (Node 21 and
  later). In Rust, `serde_json` without `arbitrary_precision` cannot carry
  an `i128`. Proposed: `event_from_json` and `state_json` in every
  runtime, on the canonical JSON of 6.1.
- **The process configures the module.** The Python and JavaScript
  runtimes read `--fuel=`, `--trace` and `--fuel-trace` from the process's
  arguments when they load, and the Python runtime raises the recursion
  limit of the whole process to 1,000,000. A host started with such an
  argument changes the program. `delta` does not reset the fuel itself, so
  a host that forgets `fuel_reset` before each input departs from HC-7a.
  Proposed: a `step(state, event)` entry that resets the fuel and returns
  the outcome, with limits passed as arguments.
- **The names are positional.** Steps are `s_5`, `s_6`, locals `l_n`,
  functions `u_NAME`; a stack trace from inside a step names those and the
  runtime's helpers, not the source.
- **Not covered here.** The Scala target has no host in this directory.
  The one emitter (`spec/checker/src/one`, the proposal `one-emitter.md`)
  prints the core as JSON and evaluates it; it prints no language yet, so
  there is nothing of it to embed.

The measurements (lines of emitted code against the hand-written
versions, the size of each host's glue, and the cost of each call at the
boundary) are in the RCE project's experiments repository,
`rce/bench/projections-interop/`.
