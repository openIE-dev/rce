# Systems: buffers and control loops

Programs of the kind found in operating systems and infrastructure tools.

| Program | What it shows |
|---|---|
| `ring_buffer.rce` | a fixed-size ring buffer that never overwrites when full |
| `reconcile.rce` | a reconcile loop that moves the actual state one step towards the desired state |

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
