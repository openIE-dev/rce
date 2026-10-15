# Concurrency gap register (Appendix C and E against code and proofs)

Status as of 2026-10-15. Each gap: evidence, severity (S1 unsound or
wrong result, S2 claim not backed, S3 wording or reference), fix, and
state. Commits are on `rce/host`.

| id | gap | evidence | sev | fix | state |
|---|---|---|---|---|---|
| G1 | Threaded lvar `par` gave each branch the fork's fuel and never summed it; the emitters charged sequentially, so reference and targets could disagree at exhaustion | `conc.rs` par_exec before 9030be813; no test of the case | S1 | Sum-checked account in source order; emitters restore fuel on abandoned restart attempts; Lean `RCE.ParFuel` proves the trap point equals the sequential one and work is at most k times F | closed 9030be813, 2aa97399f |
| G2 | LVar threshold reads did not check pairwise incompatibility at run time; put after freeze was not refused | conc.rs lvar code; Kuper and Newton threshold condition | S1 | Handed to the LVar worker (frozen bit, Merge-order thresholds, W-LVAR-1, `LVarQD.lean`) | closed by the LVar worker (0389074f0, 51767447a) |
| G3 | Supervisor intensity had no window; restart was not visible in the log; a crash inside `atomic` was not supervised | conc.rs on_crash before 75487e300 | S1 | Window in logical ticks (`RESTART_WINDOW`), `Restart(...)` log entry, atomic crash supervised; parity on Rust, Python, JS | closed 75487e300 |
| G4 | Restart was derived, not an input: state was not the fold of the log under a changed policy | C.11.11; T3 | S1 | `RestartIn` recorded per crash, replay applies logged decisions, mismatch is `replay diverged` | closed 2aa97399f (reference); emitted runtimes record but do not replay |
| G5 | Budget exhaustion was only a refusal; totality gives no value to act on | C.5; T2 | S2 | `with budget n ... else as e`: atomic block (writes undone on every target), typed `Exhausted{limit, used}`, W-BUD-1 | closed dc411f257 |
| G6 | Sessions: an endpoint used after it was moved is not refused; the checker does not track linearity after `moved` | check06b.rs (~793) | S1 | Linear endpoint tracking plus vectors linking `SessionRec.lean` to the checker | open |
| G7 | Resolver: installing a promoted handler changes δ with no logged height; in-flight steps are not pinned to the old rules | engine.rs promote_escalation_with_spec; book/15 D.3 "do not change δ" | S1 | RESOLVE as a logged event that takes effect at a recorded log height, recording fit and spec versions | open |
| G8 | Appendix D says acceptance and change control "do not change δ" while promotion installs a new δ | book/15 D.3 | S3 | Reword to: not inputs of δ; they govern which δ is installed (G7) | open (with G7) |
| G9 | Section references to C.4.2, C.4.3, C.11.1 and C.6.3 for nondet point at sections that do not exist or say something else | conc.rs, conc06.rs, rt.*, ParSem.lean, ParRefine.lean, book/14 | S3 | Corrected to C.4/T16, C.6/T20, C.11.2, C.5 | closed 2aa97399f |
| G10 | C.14 said the emitters run `par` branches in source order; Rust and Python use threads and JavaScript workers | emit/conc06.rs par_threads, par_workers_js | S3 | C.14 reworded: concurrent where the target allows, observation equals source order | closed 2aa97399f |
| G11 | `pscan` is implemented but has no section in Appendix C | conc.rs header comment | S2 | Specify it in C.11.2 or remove it | open |
| G12 | No Kani harness covers budget charging | proofs/kernel-kani (10 harnesses, none for budgets) | S2 | Budget charge harness in kernel.rs | open |
| G13 | `for ... par` in the reference and the host placement executor compute their reductions separately | conc.rs forpar; host place_par/place_lanes (9ef7e1760) | S2 | Route the reference through the shared fixed tree | open |
| G14 | Program events do not reach a running `fn main` task through the conc.rs scheduler | HC-6; host 2cd4ee0cf | S2 | Feed MainInput events through the scheduler into D | open |
| G15 | HC-6 and HC-12 wording | book/16 | S3 | Owned by the admitter worker, which has rewritten E.2a/E.2b from laws | handed off |
| G16 | One force-push to the worker's own branch `rce/conc` during the 75487e300 push | push script history | process | Line removed; no force-push since | recorded |

Open proof obligations: G6 linearity link to Lean; G12 Kani budget
harness; the core of `RCE.ParSem` has no loops, calls or data
structures, so the full evaluator is linked by generated vectors and
differential tests, not by proof.
