# Proposal: host-rap

Status: accepted and implemented as HC-34 to HC-36 (Appendix E, E.4h) by
the change on branch `claude/host-rap` (ledger row of 2026-11-02). Every
section below describes what that change builds and checks, except
section 6, which lists what it does not build.

RAP, the Reactive Agent Protocol, is the protocol of the Infinity project
(github.com/hydro-project/infinity, Apache-2.0; Amazon Web Services and the
UC Berkeley Hydro project) between agent runtimes and tools. This proposal
serves an RCE program as a RAP tool from the RCE host, so that an agent
runtime can call a program's events as tools, and states each design
decision with the rule of the RAP specification or of the host contract
that decides it.

## 0. Terms

- **RAP runtime** (or runtime): the agent host process that calls tools.
  **Tool server** (or tool): the HTTP service that serves them; here, the
  RCE host.
- **Invocation**: the runtime's `POST` of one call, with `operation`,
  `arguments`, `id`, `callback_url`, `group_id`, and optionally `call_id`,
  `thread_ancestors` and `user_id` (RAP `spec/basic/tool-invocation.md`).
- **Acknowledgement** (ack): the tool's HTTP 200 answer to an invocation.
- **Tool result** (result): the tool's later `POST` of
  `{type: "tool_result", group_id, id, call_id, text}` to `callback_url`
  (RAP `spec/basic/tool-result.md`).
- **Toolset**: the JSON manifest of a tool's operations, each with a JSON
  Schema `inputSchema` (RAP `spec/basic/toolsets.md`).
- **D**: the host's durable log (Appendix E, E.3). **Round**: one pass of
  the host loop of `rce host serve`, which admits what arrived and takes
  one barrier (HC-24).

## 1. What RAP requires of a tool

Read from `spec/basic/` (lifecycle, toolsets, transport, tool-invocation,
tool-result, tool-cancellation, thread-closure, migration), `spec/server/`
(subscription-events, view-updates, user-choice, oauth) and `about/`
(rap-servers, architecture, subscription-events) of the RAP documentation:

1. Serve the toolset at `{endpoint_base}/.well-known/rap-toolset`; a
   toolset has a name of 1 to 128 characters, an `endpoint`, and at least
   one tool, each with a name, a description and an `inputSchema`.
2. Acknowledge an invocation with HTTP 200 immediately, and never block
   the acknowledgement on processing.
3. Deliver exactly one terminal message per invocation to `callback_url`,
   with its `group_id` and `id`, echoing `call_id`; never drop an
   invocation silently.
4. Answer an unknown operation or arguments that do not conform with 200
   and then an error result, its text starting `Error: `.
5. Retry a callback on 5xx or a failed connection, with exponential
   backoff, and not on 4xx. Runtimes process results idempotently, since a
   result may arrive more than once.
6. Answer `/cancel_tool_call` and `/close_thread` with 200 whatever is
   done; acting on them is optional.
7. Expose `/migrate` only when the toolset says `needsMigration: true`.

## 2. What the host contract requires

HC-10 (every value that enters a run is an event or a logged answer),
HC-5 (no ambient authority), HS-2 (no output reported before its entry is
durable), HC-16 and law 3 of E.2c (exactly once is at-least-once delivery
plus idempotent dedupe), HC-21 (a kept step is an input, its text not in
D), HC-22 (a restart is an input, ids never reused, outstanding requests
`interrupted`), HC-26 (every byte the host writes is a function of D),
HC-27 (a log is bound to its program) and the positional command line
(`positional-cli.md`, PC-1).

## 3. Decisions

Each decision names the rule that decides it.

| # | Decision | Reason |
|---|---|---|
| R1 | The command is `rce host serve rap HOST:PORT [callback HOST:PORT]... FILE D`: two setting words of the `serve` family, before FILE and D. | PC-1 and PC-3: setting words before operands, no flags. D is an operand as for every `serve` (HC-1: no admission without D). |
| R2 | A result goes only to an address granted by `callback HOST:PORT`, compared by its exact text; an invocation whose `callback_url` names another address is refused with 400. | HC-5: a callback is a connection the host opens, so its address is authority, granted as `net` and `peer` addresses are (HC-23); the sandbox (HC-5a) allows connecting to exactly these ports. RAP leaves callback authentication to the implementation (`transport.md`). |
| R3 | Each event the program accepts from outside is one operation, named as the event, in declaration order; the events the host delivers (HC-10b's, `Tick`, `Input`) and `extern` events are not. | The task's rule and HC-10b: an event the host delivers is not an input a runtime may invent; HC-3: a foreign runtime's answers come only from that runtime. RAP's tool names (letters, digits, `_`, `-`) admit every RCE identifier. |
| R4 | `inputSchema` is the JSON Schema of the event's fields with `additionalProperties: false`, integers with their range; an event with a field of another type (a kind, a map, a tuple) is left out and the host says so. | RAP asks for a JSON Schema per tool and an explicit `additionalProperties: false` for a tool with no parameters (`toolsets.md`); E.4c delivers only exact conversions, so the schema states exactly what converts. A kind has no single JSON form an agent can be told today. |
| R5 | The toolset is a function of the module and the endpoint; `host.RapListen` records its SHA-256, which replay derives again. A bind failure stops the host. | HC-26: what the host writes is a function of D. The endpoint is in the toolset, so the host cannot serve a toolset before its address is bound (HC-24's retry does not apply). |
| R6 | An invocation without a string `id` and `group_id`, or without an `http://` `callback_url` to a granted address, is refused with 400 and nothing enters D; every other invocation is acknowledged with 200. | RAP: a non-200 tells the runtime the tool could not accept the call, and it records a failed call (`tool-invocation.md`), so the 400 is not a silent drop; such an invocation can never be answered, since results are routed by id and group_id to the callback URL. RAP rule 4 for everything else. Only `http://` because the host has no general TLS client; an `https://` callback is refused, not downgraded. |
| R7 | The invocations of a round are appended as `host.RapInvoke`, made durable by one barrier, acknowledged, and then stepped (`host.RapStep` {call}, then the event). | HS-2 and RAP rule 3: an ack after the barrier means an acknowledged invocation survives any crash; RAP rule 2: the ack waits for the barrier only, never for δ or a result. Group commit keeps one barrier per round (HC-24). |
| R8 | `host.RapInvoke` holds RAP's eight invocation fields, each null when absent; other fields are dropped. | HC-10 (the host's whole value, as E.4c's `host.Kept` holds it); the field table of `tool-invocation.md` names exactly eight. |
| R9 | The dedupe key is (group_id, id). A repeated key is `host.RapDuplicate` {call_id, callback_url, group_id, id, of}: never stepped, answered with the first invocation's result text, its own call_id and callback_url. | HC-16 and law 3 of E.2c, with RAP's key: RAP routes results by (group_id, id) (`tool-result.md`, Routing) and delivers invocations at least once. HC-22: an id is never applied twice. |
| R10 | The result's text is `{"outputs":[...],"seq":N,"state_hash":H}` for an applied step, each output as D writes an event; `Error: the step of E at seq N was kept (cause C): T; the state is unchanged` for a kept step; `Error: ` and the reason for an invocation that does not conform. | RAP: `text` may be JSON-encoded structured data, and errors start with `Error: `. The outputs are what the step emits, the whole λ output of the input; seq and state hash let the agent cite D. HC-21: the cause is named. Effects' later results are not part of the step. |
| R11 | `host.RapResult` {call, length, sha256} records the digest of the result, not its bytes; replay derives the bytes and compares. | HC-26: the bytes are a function of D, so D need not hold them; HC-21: the kept text, which can name a declared error, stays out of D (T7). |
| R12 | The result's POST is an effect under HC-16's rule: retried until an answer settles it (any status but 5xx), 100 ms doubling to 30 s; `host.RapSent` {call, status} records the settling answer; a host opened on D delivers again every result without `host.RapSent` (one delivery, retried until it settles), and never a settled one. HC-22's `interrupted` is not used. | RAP rules 3 and 5: at-least-once with an idempotent runtime; HC-22 would give an unsettled POST the outcome `interrupted` and drop a result, which RAP forbids. HS-2: a result is POSTed only after its `host.RapResult` is durable. |
| R13 | `/cancel_tool_call` and `/close_thread` are acknowledged with 200 and not acted on; nothing enters D. | RAP rule 6. A step is applied when it is admitted, so nothing is in flight to cancel, and a result after a cancellation is allowed (`tool-cancellation.md`); the program's state is not per thread. Since nothing the host writes depends on them, HC-10 needs no entry. |
| R14 | No `/migrate`; the toolset omits `needsMigration`. | RAP rule 7. The state is D; moving a session is copying D and `D.program` (HC-27). |
| R15 | A body nested deeper than 64 levels is refused before it is parsed. | The JSON parser recurses; a body of at most 1,000,000 bytes (HC-25) could otherwise exhaust its stack. |
| R16 | `rce host replay rap FILE D` prints each result, `rap CALL STATUS BODY`. | HC-8: replay re-derives every byte the host wrote, with every adapter disconnected. |

## 4. Composition before addition

No construct is added to the language: a program is served unchanged,
and the toolset, the invocations and the results are host entries and
host code (CONTRIBUTING rule 2). The same service written with existing
constructs is a program on `http HOST:PORT` that parses RAP's JSON itself
and POSTs results with `net`; it would have to repeat, in every program,
the toolset, the dedupe by (group_id, id) and the retry of results, and
the toolset would not be derived from the declarations.

## 5. Checks

- `spec/checker/tests/host_rap.rs` runs `rce host serve rap` against an
  in-process callback receiver on loopback: the toolset of the declared
  events and its digest in D; acknowledged, stepped and answered
  invocations, kept steps and invocations that do not conform; refusals
  that append nothing; duplicates answered from D; replay giving every
  result byte for byte; a lost POST retried until it settles; a kill -9
  with a result not settled, sent again once; a kill -9 after the
  acknowledgement and before the step; forged RAP entries found by
  replay; and RAP's message shapes against examples copied from the
  specification (`tests/host/rap/NOTICE.md` names their source and
  license).
- `spec/checker/tests/host_faults.rs` serves `programs/money/account.rce`
  as a RAP tool under seeded drops, duplicates and reorders of
  invocations, lost result POSTs and kills (among the arrivals, between
  the acknowledgements and the steps, before the round is durable, before
  its results are sent), with torn tails and power cuts, and requires
  replay to hold, no invocation stepped twice, every result received equal
  to the one replay derives, and one result text per id.

## 6. Not built

- Subscription events, OAuth, user choice and view updates
  (`spec/server/`): no message other than the one result is sent per
  invocation. A program's later outputs (the results of its effects)
  could become `subscription_event` messages; that is a plan.
- `https://` callbacks (R6) and authentication of invocations or
  callbacks, which RAP leaves to the implementation.
- The Infinity side: pointing an Infinity runtime at a served program is
  described in chapter 16 of the book and was not run in this change.
