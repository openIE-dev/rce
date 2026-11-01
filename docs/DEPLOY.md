# Running an RCE program in production

An RCE program runs as a service on the host runtime (specification
Appendix E): `rce host serve` admits each input as an event, applies the
program's step, and appends the entry to the durable log D before any
effect leaves the process. The state is always the replay of D, so D is the
only thing to keep. This guide runs `programs/network/notes_http.rce`, an
HTTP notes service with a `/health` route; every command below was run
against the container image this repository builds, and
`.github/workflows/deploy.yml` runs the same sequence on each change to the
image.

## Get rce

- Binary: `curl -fsSL https://raw.githubusercontent.com/openIE-dev/rce/main/scripts/install.sh | sh`
  installs the release of the newest `main` commit that passed CI
  (`rce-latest`), checked against its SHA256SUMS; `sh install.sh COMMIT`
  pins one commit, where COMMIT is the 12-hex-digit short hash of the
  release `rce-<commit>`. Only the 30 newest commit releases remain (the
  release workflow deletes older ones), so an older commit cannot be pinned
  this way. `rce version` prints the commit.
- Container: `ghcr.io/openie-dev/rce:latest`, or `ghcr.io/openie-dev/rce:<commit>`
  to pin, where `<commit>` is the full 40-hex-digit commit hash. The image runs `rce` as an unprivileged user with the working
  directory `/srv` and the volume `/data`. To build it yourself:
  `docker build -f deploy/Dockerfile -t rce .`

## Check before you deploy

```sh
rce check notes_http.rce     # the program is admitted
rce test notes_http.rce      # every example and check holds
```

## Run it as a service

With the container:

```sh
docker run -d --name notes -p 8080:8080 \
  -v rce-data:/data -v "$PWD/programs/network:/srv:ro" \
  ghcr.io/openie-dev/rce:latest \
  host serve http 0.0.0.0:8080 /srv/notes_http.rce /data/notes

curl -i http://127.0.0.1:8080/health                     # HTTP/1.1 200 OK
curl -X POST --data 'first note' http://127.0.0.1:8080/notes
curl http://127.0.0.1:8080/notes                         # [{"id":1,"text":"first note"}]
```

Without a container, the same command runs the installed binary:
`rce host serve http 127.0.0.1:8080 notes_http.rce /var/lib/notes`.
The grant words (`http`, `fs`, `net` and the others) come before the
program and the log, and no argument starts with `-`: a command line that
still gives a removed flag, such as `--log`, exits with status 2 and prints
its new spelling, so a unit or container command written for an earlier
`rce` fails at start instead of running.
A systemd unit needs only that `ExecStart=`, `Restart=on-failure`, a user
that owns the log directory, and `KillSignal=SIGTERM` (the default).

On start the host prints its sandbox: on Linux, Landlock (writes and reads
only under the log directory and the granted roots, no TCP bind or connect
after the listener is open, no exec) and a seccomp filter. The grant word
`names` lets a program turn user and group ids into names; only with it
does the host read `/etc/passwd` and `/etc/group` (Appendix E, HC-33). It works under Docker's default seccomp profile;
no extra capability is needed.

When stderr is a terminal, the host also reports its status there as OSC
7501 records (HC-28), so a terminal that reads them, such as Herdr, shows
the service as idle, working, done or in error. Under systemd, Docker without `-t`,
and in a pipe, stderr is not a terminal and no record is written;
`RCE_STATUS=on` writes them anyway and `RCE_STATUS=off` never. The report is
not in D and does not change replay or the state hash.

## Stop, restart, back up

- `docker stop` (SIGTERM) is the host loop's stop input: the loop ends after
  the entry in progress, and D is complete. The host then writes its closing status
  line, `durable N entries=M state_hash H`, to stderr (HC-28), which
  `docker logs` shows; stdout carries only the program's Display bytes.
- A new process on the same log continues where the last one stopped. A
  restart is itself recorded, and ids are never reused.
- `rce host replay FILE D` re-derives every state hash of D with every
  adapter disconnected and prints `replay ok: N entries, ...` with the final
  state. Run it after a restore, or on a copy of D to audit it.
- To back up, stop the service, copy the log directory, and check the copy
  with `rce host replay`.

## Frames and a renderer

A program that shows a view emits `Frame(view)`, a `ui.view` tree, after a
step (HC-29). The grant `frames FILE` appends one JSON line per frame to
FILE (its seq, digests, the view and the `ui.view.diff` operations from the
previous frame); a renderer reads that file. Nothing is appended to D for a
frame, since a frame is a function of D:
`rce host replay frames FILE D` prints the same lines from D alone, with
the `ui.rules` findings of each frame and the report of the session.

Under `rce host serve presented ACKS ...`, the renderer appends
`SEQ FRAME_DIGEST` to ACKS for each frame it showed; the host records each
as `host.Presented` and delivers `Presented(seq, digest)` when the program
declares it, so what a person was shown can be counted from D. A program
that declares `Frame` is always replayed from the start of D (no snapshot).

## Changing the program

A log is bound to the program that wrote it (HC-27): beside D the host
keeps `D.program`, the program's module hash, and `rce host run`, `serve`
and `replay` refuse a program whose hash differs, naming both hashes,
before D is opened. Layout and comments are not part of the hash.

- A change of meaning is a new service with a new log.
- For a change that `rce change OLD NEW` classifies as an erratum, write the
  new hash to `D.program` (the refusal prints it) and restart; that write is
  the record of the decision.
- `rce change` checks behaviour on the event alphabet it explores
  (specification 10.11): each payload field's boundary values, plus the
  literals that the steps, and the functions they pass the field to,
  compare the field with (`==`, `!=`, `<`, `<=`, `>`, `>=`, `match` arms),
  their integer neighbours, and a text that none of them equals. Both
  programs are explored over the literals of both. So a wrong method's
  status changed from 405 to 501 in `programs/network/notes_http.rce` is a
  meaning change: the request `("", "/health")` reaches the branch, and the
  report names it. The check is still not every input: a value the program
  derives with a method call or an operator before comparing it (a path
  segment of `target.split("/")`, `x + 1 > 9`) is not followed, so a branch
  reached only that way can stay unexplored, and the exploration stops at
  `fuel` states ("to the explored bound"). Read an erratum as holding on
  that alphabet before rebinding a log.
