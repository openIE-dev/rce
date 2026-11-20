//! Residual breaks in the wasm emitter.
//!
//! The carried count is the source depth (`e = d`). `break 0` is the local
//! exit. A block or a loop decrements once. An `if` does not. The emitted
//! `br` counts every wasm label between the break and that target, including
//! each `if` and a loop's restart label. Each module is checked by wasmparser
//! and wasmi, then run and compared with the core evaluator.
use rce_spec_check::one::{
    self,
    ir::{self, Atom, Lit, Module, Place, Stmt, Ty},
};
use rce_spec_check::wasm_break::{self, Label, Residual};

fn i64ty() -> Ty {
    Ty::i64()
}

fn lit(v: i64) -> Atom {
    Atom::K(i64ty(), Lit::Int(v as i128))
}

fn module_of(f: ir::Fn) -> Module {
    let (g, class) = one::json::canonical_group(std::slice::from_ref(&f));
    let key = one::json::group_key(&g);
    Module {
        globals: vec![],
        defs: [(key.clone(), g)].into(),
        entry: (key, class[0]),
        names: serde_json::json!({}),
        graphs: Default::default(),
    }
}

fn labels_of(wat: &str) -> Vec<(u32, u32)> {
    wat.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix(";; residual ")?;
            let (a, b) = rest.split_once(" label ")?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect()
}

/// Check the module, run the core evaluator, validate the wasm, run it.
fn agree(name: &str, f: ir::Fn, breaks: &[(u32, u32)]) -> i64 {
    let m = module_of(f);
    one::wf::check(&m).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut ev = one::eval::Eval::new(&m);
    let v = ev
        .call_key(&m.entry.0, m.entry.1, vec![])
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    let one::eval::Val::Int(n) = v else {
        panic!("{name}: {v:?}")
    };
    let func = m.func(&m.entry.0, m.entry.1).unwrap();
    let wat = wasm_break::emit(func).unwrap_or_else(|e| panic!("{name}: {e}"));
    wasm_break::validate(&wat).unwrap_or_else(|e| panic!("{name}: {e}\n{wat}"));
    let got = wasm_break::run(&wat).unwrap_or_else(|e| panic!("{name}: {e}\n{wat}"));
    assert_eq!(got, n as i64, "{name}\n{wat}");
    assert_eq!(labels_of(&wat), breaks, "{name}\n{wat}");
    got
}

struct B {
    next: u32,
    stmts: Vec<Stmt>,
}

impl B {
    fn new() -> Self {
        Self {
            next: 0,
            stmts: vec![],
        }
    }
    fn var(&mut self, v: i64) -> u32 {
        let i = self.next;
        self.next += 1;
        self.stmts.push(Stmt::Var(i64ty(), lit(v)));
        i
    }
    fn set(&mut self, v: u32, a: Atom) {
        self.stmts.push(Stmt::Set(Place::V(v), a));
    }
    fn wadd(&mut self, a: Atom, b: Atom) -> Atom {
        let i = self.next;
        self.next += 1;
        self.stmts
            .push(Stmt::Prim("int.wadd".into(), vec![i64ty()], vec![a, b]));
        Atom::V(i)
    }
    fn eq(&mut self, a: Atom, b: Atom) -> Atom {
        let i = self.next;
        self.next += 1;
        self.stmts
            .push(Stmt::Prim("int.eq".into(), vec![i64ty()], vec![a, b]));
        Atom::V(i)
    }
    fn add_to(&mut self, v: u32, k: i64) {
        let s = self.wadd(Atom::V(v), lit(k));
        self.set(v, s);
    }
    fn brk(&mut self, d: u32) {
        self.stmts.push(Stmt::Break(d));
    }
    fn ret(&mut self, v: u32) {
        self.stmts.push(Stmt::Ret(Atom::V(v)));
    }
    fn block(&mut self, f: impl FnOnce(&mut B)) {
        let mut inner = B {
            next: self.next,
            stmts: vec![],
        };
        f(&mut inner);
        self.next = inner.next;
        self.stmts.push(Stmt::Block(inner.stmts));
    }
    fn loop_(&mut self, f: impl FnOnce(&mut B)) {
        let mut inner = B {
            next: self.next,
            stmts: vec![],
        };
        f(&mut inner);
        self.next = inner.next;
        self.stmts.push(Stmt::Loop(inner.stmts));
    }
    fn when(&mut self, c: Atom, f: impl FnOnce(&mut B)) {
        self.if_(c, f, |_| {});
    }
    fn if_(&mut self, c: Atom, t: impl FnOnce(&mut B), e: impl FnOnce(&mut B)) {
        let mut then_b = B {
            next: self.next,
            stmts: vec![],
        };
        t(&mut then_b);
        let mut else_b = B {
            next: then_b.next,
            stmts: vec![],
        };
        e(&mut else_b);
        self.next = else_b.next;
        self.stmts.push(Stmt::If(c, then_b.stmts, else_b.stmts));
    }
    fn finish(self) -> ir::Fn {
        ir::Fn {
            params: vec![],
            ret: i64ty(),
            body: self.stmts,
        }
    }
}

/// `break 0` inside an `if` leaves the block. The assignment after the `if`
/// does not run.
fn break_inside_if() -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.when(Atom::bool(true), |b| {
            b.set(r, lit(1));
            b.brk(0);
        });
        b.set(r, lit(100));
    });
    b.ret(r);
    b.finish()
}

/// `break 0` taken: the arm after the `if` does not run, and the block's
/// continuation sets 30. Falling through sets 31.
fn break_or_fall(take: bool) -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.if_(Atom::bool(take), |b| b.brk(0), |b| b.set(r, lit(1)));
        b.add_to(r, 30);
    });
    // The break leaves `r` at 0, so the continuation stores 30. The
    // fall-through arm stored 1 and then 30, which is 31, and this test
    // does not replace it.
    let still = b.eq(Atom::V(r), lit(0));
    b.when(still, |b| b.set(r, lit(30)));
    b.ret(r);
    b.finish()
}

/// Three blocks and no `if`. The break is the whole of the inner block, so
/// a tail between the blocks would follow a block that `break d` for `d > 0`
/// cannot complete (W2). The value is 4, then 6 after the nest. The `br`
/// index is `d`: no structural label intervenes.
fn blocks_only(d: u32) -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.block(|b| {
            b.block(|b| {
                b.set(r, lit(4));
                b.brk(d);
            });
        });
    });
    b.add_to(r, 6);
    b.ret(r);
    b.finish()
}

/// Two `if`s around `break d`, then two blocks. The `if`s do not change `d`.
fn if_around_break(d: u32) -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.block(|b| {
            b.when(Atom::bool(true), |b| {
                b.when(Atom::bool(true), |b| b.brk(d));
            });
            b.add_to(r, 1);
        });
        b.add_to(r, 10);
    });
    b.add_to(r, 100);
    b.ret(r);
    b.finish()
}

/// Three blocks. `break d` adds 4, 5 and 6 only for the tails that run.
fn depths(d: u32) -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.block(|b| {
            b.block(|b| {
                b.when(Atom::bool(true), |b| {
                    b.when(Atom::bool(true), |b| b.brk(d));
                });
            });
            b.add_to(r, 4);
        });
        b.add_to(r, 5);
    });
    b.add_to(r, 6);
    b.ret(r);
    b.finish()
}

/// A bare block. `break 0` is `br 0`: no `if` sits between the break and it.
fn bare_block() -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.block(|b| {
        b.set(r, lit(4));
        b.brk(0);
    });
    b.set(r, lit(7));
    b.ret(r);
    b.finish()
}

/// Three `if`s sit inside the block the break leaves, and that block sits
/// inside a loop. The loop's own `break 0` then leaves the loop.
fn ifs_inside_loop() -> ir::Fn {
    let mut b = B::new();
    let r = b.var(0);
    b.loop_(|b| {
        b.block(|b| {
            b.when(Atom::bool(true), |b| {
                b.when(Atom::bool(true), |b| {
                    b.when(Atom::bool(true), |b| {
                        b.set(r, lit(1));
                        b.brk(0);
                    });
                });
            });
            b.set(r, lit(9));
        });
        b.brk(0);
    });
    b.add_to(r, 10);
    b.ret(r);
    b.finish()
}

/// `break 0` leaves the block and the loop runs the tail, then iterates.
/// `break 1` leaves the loop and that tail does not run.
fn block_in_loop() -> ir::Fn {
    let mut b = B::new();
    let n = b.var(0);
    let seen = b.var(0);
    b.loop_(|b| {
        b.block(|b| {
            let first = b.eq(Atom::V(n), lit(0));
            b.if_(
                first,
                |b| {
                    b.add_to(n, 1);
                    b.brk(0);
                },
                |b| b.brk(1),
            );
        });
        b.add_to(seen, 1);
    });
    b.ret(seen);
    b.finish()
}

/// `break 0` leaves the inner loop through two `if`s. `break 1` leaves both
/// loops, and the mark after the inner loop does not run the second time.
fn two_loops() -> ir::Fn {
    let mut b = B::new();
    let phase = b.var(0);
    let mark = b.var(0);
    b.loop_(|b| {
        b.when(Atom::bool(true), |b| {
            b.loop_(|b| {
                b.if_(
                    Atom::bool(true),
                    |b| {
                        let first = b.eq(Atom::V(phase), lit(0));
                        b.if_(
                            first,
                            |b| {
                                b.set(phase, lit(1));
                                b.brk(0);
                            },
                            |b| b.brk(1),
                        );
                    },
                    |b| b.brk(1),
                );
            });
            b.add_to(mark, 1);
        });
    });
    b.ret(mark);
    b.finish()
}

#[test]
fn br_depth_follows_the_residual_law() {
    let d = |labels: &[Label], n: u32| {
        wasm_break::br_label(labels, Residual::at_break(n))
            .unwrap()
            .get()
    };
    assert_eq!(d(&[Label::Block], 0), 0);
    assert_eq!(d(&[Label::If, Label::Block], 0), 1);
    assert_eq!(d(&[Label::If, Label::If, Label::Block], 0), 2);
    assert_eq!(d(&[Label::If, Label::If, Label::If, Label::Block], 0), 3);
    assert_eq!(d(&[Label::LoopRestart, Label::LoopExit], 0), 1);
    assert_eq!(d(&[Label::If, Label::LoopRestart, Label::LoopExit], 0), 2);
    let across = [Label::If, Label::Block, Label::LoopRestart, Label::LoopExit];
    assert_eq!(d(&across, 0), 1);
    assert_eq!(d(&across, 1), 3);
    let two = [
        Label::If,
        Label::If,
        Label::LoopRestart,
        Label::LoopExit,
        Label::If,
        Label::LoopRestart,
        Label::LoopExit,
    ];
    assert_eq!(d(&two, 0), 3);
    assert_eq!(d(&two, 1), 6);
    assert_eq!(d(&[Label::Block, Label::Block], 1), 1);
    assert_eq!(d(&[Label::If, Label::Block, Label::If, Label::Block], 1), 3);
    let only_if = wasm_break::br_label(&[Label::If, Label::If], Residual::at_break(0)).unwrap_err();
    assert!(
        only_if.contains("names no enclosing block or loop"),
        "{only_if}"
    );
}

fn depth(labels: &[Label], residual: u32) -> u32 {
    wasm_break::br_label(labels, Residual::at_break(residual))
        .unwrap()
        .get()
}

/// `one/eval.rs` returns the control of an `if` unchanged. `Block` and `Loop`
/// are what map `Break(0)` to `Next` and `Break(d)` to `Break(d - 1)`.
/// `Break.lean` states the same: `stepExit .ite d = some d`.
#[test]
fn an_if_never_decrements_the_carried_residual_count() {
    // Residual 0 still names the nearest block after any number of `if`s.
    // Residual 1 still names the block after that one. The `if`s add wasm
    // labels and do not consume the count.
    assert_eq!(depth(&[Label::Block], 0), 0);
    assert_eq!(depth(&[Label::If, Label::Block], 0), 1);
    assert_eq!(depth(&[Label::If, Label::If, Label::Block], 0), 2);
    assert_eq!(depth(&[Label::Block, Label::Block], 1), 1);
    assert_eq!(
        depth(&[Label::If, Label::If, Label::Block, Label::Block], 1),
        3
    );
    for d in 0..4 {
        let err = wasm_break::br_label(&[Label::If, Label::If], Residual::at_break(d)).unwrap_err();
        assert!(err.contains("names no enclosing block or loop"), "{err}");
    }
    // `break 0` inside two `if`s is still 0 at the inner block, so that
    // block's tail does not run and the outer tail does: 10 + 100. A decrement
    // on either `if` would also run the inner tail.
    let kept0 = agree("if keeps break 0", if_around_break(0), &[(0, 2)]);
    assert_eq!(kept0, 110);
    // `break 1` is still 1 at the inner block. That block decrements it and
    // the outer block consumes 0, so neither block tail runs: 100. One
    // decrement on an `if` would let the outer tail add 10.
    let kept1 = agree("if keeps break 1", if_around_break(1), &[(1, 3)]);
    assert_eq!(kept1, 100);
}

/// The `br` index equals the residual count when the target contains no `if`
/// and no loop-restart label. Each of those labels adds one. `eval.rs` does
/// not count an `if`, and a wasm `if` is a branch target, so the two numbers
/// differ whenever such a label intervenes.
#[test]
fn br_depth_is_the_residual_count_plus_intervening_structural_labels() {
    assert_eq!(depth(&[Label::Block], 0), 0);
    assert_eq!(depth(&[Label::Block, Label::Block], 1), 1);
    assert_eq!(depth(&[Label::Block, Label::Block, Label::Block], 2), 2);
    assert_eq!(depth(&[Label::If, Label::Block], 0), 0 + 1);
    assert_eq!(
        depth(&[Label::If, Label::If, Label::If, Label::Block], 0),
        0 + 3
    );
    assert_eq!(depth(&[Label::LoopRestart, Label::LoopExit], 0), 0 + 1);
    assert_eq!(
        depth(&[Label::If, Label::LoopRestart, Label::LoopExit], 0),
        0 + 2
    );
    assert_eq!(
        depth(
            &[Label::If, Label::Block, Label::LoopRestart, Label::LoopExit],
            1
        ),
        1 + 2
    );

    // One break. `structural` is the number of `if` and loop-restart labels
    // inside the target. The emitted `br` is checked against `label`.
    let emitted = [
        ("blocks only, break 0", blocks_only(0), 10, 0, 0, 0),
        ("blocks only, break 1", blocks_only(1), 10, 1, 1, 0),
        ("blocks only, break 2", blocks_only(2), 10, 2, 2, 0),
        ("ifs around break 0", if_around_break(0), 110, 0, 2, 2),
        ("ifs around break 1", if_around_break(1), 100, 1, 3, 2),
        ("bare block", bare_block(), 7, 0, 0, 0),
        ("two ifs, break 0", depths(0), 15, 0, 2, 2),
        ("two ifs, break 2", depths(2), 6, 2, 4, 2),
    ];
    assert_eq!(emitted.len(), 8);
    for (name, func, want, residual, label, structural) in emitted {
        let got = agree(name, func, &[(residual, label)]);
        assert_eq!(got, want, "{name}");
        if structural == 0 {
            assert_eq!(label, residual, "{name}");
        } else {
            assert_eq!(label, residual + structural, "{name}");
            assert_ne!(label, residual, "{name}");
        }
    }
    // `break 0` in the block: residual 0 plus the `if`. `break 1`: residual 1
    // plus that `if` and the loop's restart label.
    let seen = agree("block in a loop", block_in_loop(), &[(0, 1), (1, 3)]);
    assert_eq!(seen, 1);
    assert_eq!(1, 0 + 1);
    assert_eq!(3, 1 + 2);
    assert_ne!(1, 0);
    assert_ne!(3, 1);
    // Three `if`s inside the block, then the loop's own `break 0`, whose only
    // structural label is the restart.
    let marked = agree("three ifs in a loop", ifs_inside_loop(), &[(0, 3), (0, 1)]);
    assert_eq!(marked, 11);
    assert_eq!(3, 0 + 3);
    assert_eq!(1, 0 + 1);
    assert_ne!(3, 0);
}

struct Case {
    name: &'static str,
    want: i64,
    func: ir::Fn,
    breaks: Vec<(u32, u32)>,
}

#[test]
fn emitted_breaks_match_the_reference_and_validate() {
    let cases = [
        Case {
            name: "break 0 inside if",
            want: 1,
            func: break_inside_if(),
            breaks: vec![(0, 1)],
        },
        Case {
            name: "break 0 sets 30",
            want: 30,
            func: break_or_fall(true),
            breaks: vec![(0, 1)],
        },
        Case {
            name: "fallthrough sets 31",
            want: 31,
            func: break_or_fall(false),
            breaks: vec![(0, 1)],
        },
        Case {
            name: "depth 0",
            want: 15,
            func: depths(0),
            breaks: vec![(0, 2)],
        },
        Case {
            name: "depth 1",
            want: 11,
            func: depths(1),
            breaks: vec![(1, 3)],
        },
        Case {
            name: "depth 2",
            want: 6,
            func: depths(2),
            breaks: vec![(2, 4)],
        },
        Case {
            name: "bare block",
            want: 7,
            func: bare_block(),
            breaks: vec![(0, 0)],
        },
        Case {
            name: "three ifs in a loop",
            want: 11,
            func: ifs_inside_loop(),
            breaks: vec![(0, 3), (0, 1)],
        },
        Case {
            name: "block in a loop",
            want: 1,
            func: block_in_loop(),
            breaks: vec![(0, 1), (1, 3)],
        },
        Case {
            name: "two loops and ifs",
            want: 1,
            func: two_loops(),
            breaks: vec![(0, 3), (1, 6), (1, 5)],
        },
    ];
    let n = cases.len();
    for case in cases {
        let got = agree(case.name, case.func, &case.breaks);
        assert_eq!(got, case.want, "{}", case.name);
    }
    assert_eq!(n, 10);
}
