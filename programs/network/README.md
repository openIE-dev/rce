# Network: protocols and services

Programs that talk to other programs.

| Program | What it shows |
|---|---|
| `handshake.rce` | a connection protocol as a state machine; a message in the wrong phase is refused |
| `server.rce` | a server written as an `actor`, which has its own state and handles one message at a time |
| `notes.rce` | a JSON notes service on the HTTP event host: each request is one logged input, a body that is not UTF-8 gets 400, a note over 4096 bytes 413, and a refused step 503, 507 or 500 |

`server.rce` has no tests of its own: it shows the shape of an actor and
is only type checked.

`notes.rce` runs on the host with `rce host serve notes.rce --log D --http
127.0.0.1:8080`. The host frames each request (Appendix E HC-24 to HC-26)
and the program answers it with `HttpRespond`; its examples test every
answer without a network.

Limits in draft 0.5: there are no bit-level packet layouts.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
