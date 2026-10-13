# Network: protocols and services

Programs that talk to other programs.

| Program | What it shows |
|---|---|
| `handshake.rce` | a connection protocol as a state machine; a message in the wrong phase is refused |
| `server.rce` | a server written as an `actor`, which has its own state and handles one message at a time |

`server.rce` has no tests of its own: it shows the shape of an actor and
is only type checked.

Limits in draft 0.5: there are no bit-level packet layouts.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
