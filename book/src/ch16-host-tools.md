# Host tools: RCEdb, HTTP, media, redacted replay and chat

This chapter covers five tools of the RCE host runtime, the program
that runs an RCE program against the outside world and keeps its log D
of admitted events. Codes such as HC-3 (host contract rule 3), HS-9 (host
storage rule 9) and T8 (theorem 8), and section numbers such as 6.7 or
E.8, name rules of [the RCE specification](https://github.com/openIE-dev/rce/blob/main/spec/RCE-SPEC.md).

These commands ship in the `rce` binary. The program of this chapter,
`hello_http.rce`, is checked with `rce test` like the listings of the
other chapters; serving it needs a network, so the exchange shown with it
is an example, not checked output.

## `rce db D CMD [ARGS]`: RCEdb from the command line

RCEdb is the database of an RCE system: its tables, rows, edges, schemas
and indexes are state of an RCE module (`rcedb/rcedb.rce`), and its only
storage is a durable log D of the host runtime. Appendix E.8 gives the
normative rules; this section describes the command. `rce db` is
serverless and in-process: the directory D is the database, and each
command opens it, admits at most one event and closes it. There is no
daemon, no network listener and no query engine outside RCE.

Write commands admit one event to D and print `ok seq N`, where N is the
sequence number of the entry they appended, or `REFUSED E` with the
error E the module raised; a refused write changes no state (T8).

| Command | Effect |
|---|---|
| `insert T K ROW` | add row ROW under key K of table T (refused with `Exists` when K is present) |
| `delete T K` | remove row K (refused with `Missing` when absent) |
| `link S L D2` / `unlink S L D2` | add or remove the edge from S to D2 with label L |
| `schema T COLS` | give table T a typed schema, COLS as `name:text`, `name:i64` or `name:bool`, comma-separated; allowed only while T has no rows |
| `index T COL` | keep a secondary index on column COL of T, a fold of D (HS-7) |

Under a schema a row is its cells in column order, separated by the
horizontal tab U+0009; a row whose cells do not parse as their column
types is refused with `BadRow`. Read commands append nothing (HS-8):

| Command | Output |
|---|---|
| `scan T [START]` | `KEY<TAB>ROW` per row of T from key START, in canonical key order |
| `lookup T COL VALUE` | the rows whose column COL equals VALUE, through the index when one exists |
| `edges S` | `LABEL<TAB>DST` per edge from S |
| `describe` | per table: row count, schema and indexes |
| `dump` | every row and edge, one line each |
| `emit sqlite\|postgres\|duckdb\|cypher` | the state as Structured Query Language (SQL) statements for SQLite, PostgreSQL or DuckDB, or openCypher `MERGE` statements, headed by the entry count and state hash of D they project |
| `as-of N scan T [START]` | the scan of T over the fold of the first N entries of D (time travel, HS-9) |

An emitted store is a projection of D and never feeds back into it: the
foreign engine's durability and ordering do not reach δ. A runnable example is `run.sh rcedb` in the examples of the host runtime.

## Serving HTTP: `rce host serve FILE --log D --http HOST:PORT`

`rce host serve` is the host loop. It admits each input into D as it
arrives, performs the effects the step emits, and admits each effect's
result as another input. A line on standard input is an input written in
event syntax, a declared `Tick(n)` arrives every frame, and `Display(data)`
is written to standard output. When the loop ends, D is replayed with
every adapter disconnected, and the run counts only if replay gives the
state hash the live run had.

With `--http HOST:PORT` the host listens on that address itself and turns
each request of the Hypertext Transfer Protocol (HTTP/1.1) into one
input. The listing declares the request, the malformed request, the
response and its result with the host's own shapes, so each is delivered
exactly. A program may declare them with other fields, or not at all:
the host then delivers each value as the declaration takes it, matching
fields by name, and records in D what it did not deliver and why (Appendix E, E.4c, HD-2).

```rce
{{#include ../listings/ch16/hello_http.rce}}
```

Each request arrives as `HttpRequest(id, method, target, version, names,
values, body)`. The `id` is the sequence number of the request's own entry
in D, and the step answers by emitting `HttpRespond` with the same `req`. Run
`rce host serve hello_http.rce --log D --http 127.0.0.1:8080` and, in
another terminal, `curl -i http://127.0.0.1:8080/hello` prints:

```text
HTTP/1.1 200 OK
Content-Type: text/plain
Content-Length: 6

hello
```

The host wrote the status line and `Content-Length`; every other byte
came from the program. Whether the host also writes `Connection: close`
and closes the connection depends on the request's version and its
Connection field, which D records, so every response is a function of D
(HC-26). A
response the host cannot write correctly, for example one with a status
outside 200 to 599 or a `Content-Length` written by the program, fails
the effect, with the reason in `HttpRespondDone`, and leaves the request
open, so the step that receives that result can answer again. A `delivered` result means the host
handed the response to the connection; a client that has gone by the
time it is written is not reported.

The host reads requests from many connections at once, so a client that
sends nothing holds up no other. It applies its limits before a byte
enters D: a request line over 8,192 bytes, a head over 64 KiB or 100
fields, a body over 1,000,000 bytes, a head not complete within 10
seconds of its first byte, and any error of syntax arrive as
`HttpMalformed(id, status, why)`, with 414, 431, 413, 408 or 400 as the
status, and the program answers them like any other request (HC-25). A
connection that closes before its request is complete adds nothing. An
idle host appends nothing to D. The termination signal (SIGTERM) stops
the loop: the host stops accepting connections, gives the open requests
up to 5 seconds, makes D durable and validates it.

`programs/network/notes_http.rce` is a complete JSON notes service
written this way. Its examples test every answer without a network.

## Bytes, refusals and restarts

Three inputs reach a program on the host that it never asks for.

- **Bytes.** Bytes from outside arrive exactly as they were received, as
  a `List[i64]` of values from 0 to 255: the body of `HttpRequest`, and
  the data of `NetRecvDone` below (HC-10a). A body that is not valid UTF-8
  is therefore a value, not a trap; `utf8.utf8_valid(&b)` from `use
  text.utf8` tells the step whether it may call `b.utf8()`.
- **Refusals.** When a step is refused, by a declared error, a trap or a
  limit such as fuel, D records the refused input with the state
  unchanged (chapter 6), then `host.Refused`, and then, if the program
  declares `event Refused(seq: u64, cause: text)`, the input `Refused`
  with the refused entry's seq and a cause: `fail`, `trap`, `fuel`,
  `size`, `value` or `depth` (HC-21). On the HTTP event host the seq is
  the id of the request, so the `Refused` step of `hello_http.rce` answers
  that request with 500 instead of leaving its client waiting.
- **Restarts.** Sockets and terminals belong to one host process. A host
  started on an existing D appends `host.Restarted` with its own seq, and
  admits `Restarted(below: u64)` if the program declares it, before it
  reports every request the earlier process left unanswered as `failed`
  with `interrupted` (HC-22). New ids start at `below`, so an id the
  program kept from before the restart names no new socket.

## The net adapter: `--net HOST:PORT`

`--net HOST:PORT` grants connections of the Transmission Control
Protocol (TCP) as effects, at that exact address:
`NetConnect`, `NetListen`, `NetAccept`, `NetSend`, `NetRecv` and
`NetClose`, each with one result event (HC-23). `NetRecv` returns the
bytes that have arrived so far, possibly none, and never waits; `NetSend`
hands its bytes to a writer of the connection and returns at once. A
program on this adapter frames its own protocol and asks for more bytes
on each `Tick`, so it serves one connection per frame. For HTTP the
event host above does the framing, serves connections concurrently and
needs no `Tick`.

## Media capture and playback: `--media DIR`

`rce host run` and `rce host serve` take `--media DIR` to connect the
media adapters. A program requests them with three output kinds, each
with one result event (HC-3):

- `MediaOpen(req, source)` opens a track. The source is `dir:NAME`, the
  files of `DIR/capture/NAME` in name order, one per frame, or
  `pattern:N`, N generated test frames (at most 100000). The result
  carries the track id.
- `MediaRead(req, id)` reads the next frame of a track. Its bytes are
  stored as a blob under DIR addressed by their Secure Hash Algorithm 256
  (SHA-256) digest, and the result carries only the digest.
- `MediaPlay(req, track, digest)` plays a frame to the sink by digest;
  the sink refuses a frame whose stored bytes do not hash to the digest.

D therefore holds the track, the sequence number and the digest of each
frame, and the `Tick` that read it, never the frame bytes. Replay with
every adapter disconnected (HC-8) reaches the same state without
capturing again. In a call, frames travel between peers through a
selective forwarding unit (SFU), a relay that forwards each frame to the
other subscribers of its track without reading it; chapter 17 describes it.

## Redacted replay: `rce host replay FILE --log D --redact F,...`

`rce host replay` re-derives every state hash of D with every adapter
disconnected and prints the final state. With `--redact F,G` the fields
named F and G are printed as `<redacted sha256:H>`, where H is the first
six bytes of the SHA-256 digest of the field's value in canonical form.
Redaction changes only what is printed: D, the replay and every state
hash are unchanged, and two replays of the same field print the same H,
so a reviewer can compare runs without seeing a secret. Private keys,
ratchet state and frame digests are the fields to redact. D itself still
holds the secrets and stays on the device that wrote it.

## The chat user interface

`apps/chat.rce` is a complete chat program: workspaces, channels and
topic threads, direct messages and groups, reactions, presence, file
shares and call rooms, as one replicated RCE state. Every change is an
operation applied locally and sent to peers over remote channels (P4,
the host's channel adapter, enabled with `--chan HOST:PORT` and
`--peer HOST:PORT`, with Transport Layer Security (TLS) under `--tls`).
Replicas order messages by (Lamport clock, author) and so converge.
Direct messages are end-to-end encrypted (E2EE) as chapter 17 describes.
`apps/chat.rce` is generated from `apps/chat-app.rce` and
`lib/crypto.rce` by `scripts/build-chat.sh`.

The front end `scripts/chat-ui.py` runs `rce host serve apps/chat.rce`
as a child process and holds no chat state of its own. Each command it
accepts becomes an event in D, so a session replays:

| Command | Event |
|---|---|
| `/open CONV TOPIC` | switch to a thread and mark it read |
| TEXT | post TEXT to the current thread |
| `/edit N TEXT`, `/del N` | edit or delete your message N |
| `/react N EMOJI` | toggle a reaction |
| `/dm PEER TEXT`, `/dms PEER` | send or show an encrypted direct message thread |
| `/unread`, `/read` | unread counts per thread; mark the thread read |

It renders two queries of the program that change no state: `View(conv,
topic)`, one tab-separated line per message (`view`, the message key,
the author, the flags `new`, `edited`, `deleted` and `e2e`, the text)
ending with `view-end`, and `Unread`, one line per thread with messages
from others after the read marker. Run it with
`python3 scripts/chat-ui.py --me ann --log /tmp/ann.log [--chan ADDR --peer ADDR ...]`.
