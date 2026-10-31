# World: simulation

Programs that model a small world that changes step by step. In RCE a
simulation is a `state` that an event moves forward, so every run can be
replayed exactly from its list of events.

| Program | What it shows |
|---|---|
| `cellular_automaton.rce` | a ring of 8 cells updated by elementary rule 30 on each `Step` |
| `queue_sim.rce` | a checkout queue with a fixed capacity; an arrival into a full queue keeps the state |

Limits in the current draft: there is no continuous time and no units of measure,
so physics needs fixed time steps and integrators written by hand.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
