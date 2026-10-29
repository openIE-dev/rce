# Systems: buffers and control loops

Programs of the kind found in operating systems and infrastructure tools.

| Program | What it shows |
|---|---|
| `ring_buffer.rce` | a fixed-size ring buffer that never overwrites when full |
| `reconcile.rce` | a reconcile loop that moves the actual state one step towards the desired state |
| `accounts.rce` | a long-running account service on the host: an unbounded run of bounded steps, where a transfer that fails half way keeps the state |

**A service as a stream of bounded steps.** `accounts.rce` has no main
loop. The host (`rce host serve accounts.rce --log D`, Appendix E) reads
one line of events at a time from stdin, applies each as one step and
answers on its Display sink: `ok`, or `kept SEQ CAUSE` from the `StepKept`
event the host delivers after a kept step (HC-21). The run ends when the
host stops; each step is bounded, because the state holds at most
`MAX_ACCOUNTS` accounts and the one loop (a fee charged to every account)
runs over them, and the host gives every input its own fuel (HC-7a).

A transfer writes the debit before it checks the credit, so a credit over
`CAP` fails in the middle of the step. The event then does not apply to
any region: the debit is not kept, and the `Audit` count of transfers does
not move (6.4, T8). `accounts.cmd` shows a kept transfer; an `example` line
cannot state the state after a kept event (6.9: a kept outcome is the last
step of an example), so the property is checked on the host's log D instead. In
openie-research-experiments `rce/bench/bounded-programs`, 100,000 events
driven one at a time through `rce host serve` gave a median reply time of
4.0 ms in every block of 10,000 (the host's durable append per input is
most of it), and each of the 17,766 kept steps left the state hash
unchanged.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
