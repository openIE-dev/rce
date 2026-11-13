//! Structured control in the wasm emitter.
//!
//! A residual break count and a wasm `br` depth are different numbers.
//! ONE-15 sets the carried count `e` to the source depth `d` (`e = d`, not
//! `e = d + 1`). The originating `break 0` is the local exit and does not
//! decrement `e`. Each block or loop that is left decrements once. An `if`
//! does not. That is what `spec/checker/src/one/eval.rs` does (`Block` and
//! `Loop` subtract one; `If` returns the inner control unchanged) and what
//! W2 counts: a `break d` names an enclosing block or loop.
//!
//! WebAssembly counts every `block`, `loop` and `if` as a `br` label.
//! Branching to a `loop` restarts it, so a loop is emitted as an exit
//! `block` around a `loop`. The exit block is the one residual step for
//! that loop. The restart label is wasm structure only, as an `if` is.
//! The `br` immediate is the number of wasm labels from the break out to
//! its residual target, including that target's label.

/// The carried residual count. At the break it is the source depth.
/// After a block or a loop has been left it is what remains. It is not a
/// wasm label index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residual(u32);

impl Residual {
    /// `e = d` at `break d`.
    pub const fn at_break(depth: u32) -> Self {
        Self(depth)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A wasm `br` label index. Zero is the innermost structured instruction.
/// Every enclosing `block`, `loop` and `if` counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelDepth(u32);

impl LabelDepth {
    const fn zero() -> Self {
        Self(0)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    fn outward(self) -> Self {
        Self(self.0 + 1)
    }
}

/// One wasm structured label that encloses the instruction being emitted.
/// The innermost label is first in the slice [`br_label`] walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    /// An RCE block. Residual 0 branches here, and the continuation runs.
    Block,
    /// The exit block wrapped around an RCE loop. Residual 0 leaves the loop.
    /// The loop is this one residual step.
    LoopExit,
    /// The wasm `loop`, which a branch restarts. Not a residual step.
    LoopRestart,
    /// An RCE `if`. A wasm branch target. The residual count is unchanged.
    If,
}

/// The `br` depth of `residual` under `innermost_first`.
///
/// Residual 0 selects the nearest block or loop exit. Each block or loop
/// exit further out consumes one from the count. An `if` and a loop's
/// restart label add one to the wasm depth and leave the count as it is.
pub fn br_label(innermost_first: &[Label], residual: Residual) -> Result<LabelDepth, String> {
    let mut depth = LabelDepth::zero();
    let mut left = residual;
    for label in innermost_first {
        match label {
            Label::If | Label::LoopRestart => depth = depth.outward(),
            Label::Block | Label::LoopExit => {
                if left.get() == 0 {
                    return Ok(depth);
                }
                left = Residual(left.get() - 1);
                depth = depth.outward();
            }
        }
    }
    Err(format!(
        "break {} names no enclosing block or loop",
        residual.get()
    ))
}

/// An operand. Integer operations are wrapping i64, which is `int.wadd`,
/// `int.wsub` and `int.wmul` at `int[-2^63, 2^63 - 1]`. Comparisons are signed.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Local(u32),
    I64(i64),
    Bool(bool),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

/// A statement of the structured backend. `Break` carries a residual count,
/// not a wasm depth.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Set(u32, Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    Block(Vec<Stmt>),
    Loop(Vec<Stmt>),
    Break(Residual),
    Return(Expr),
}

/// One function. Locals are i64. The first `params` locals are the parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub params: u32,
    pub locals: u32,
    pub body: Vec<Stmt>,
}

/// The text module of `f`, exporting `run`.
pub fn emit_func(f: &Func) -> Result<String, String> {
    if f.locals < f.params {
        return Err("a function declares fewer locals than parameters".into());
    }
    let mut out = Out {
        buf: String::new(),
        labels: vec![],
        indent: 0,
    };
    out.line("(module");
    out.indent = 1;
    let mut head = String::from("(func (export \"run\")");
    for _ in 0..f.params {
        head.push_str(" (param i64)");
    }
    head.push_str(" (result i64)");
    let extra = f.locals - f.params;
    if extra > 0 {
        head.push_str(" (local");
        for _ in 0..extra {
            head.push_str(" i64");
        }
        head.push(')');
    }
    out.line(&head);
    out.indent = 2;
    out.seq(&f.body)?;
    out.line("(unreachable)");
    out.indent = 1;
    out.line(")");
    out.indent = 0;
    out.line(")");
    Ok(out.buf)
}

struct Out {
    buf: String,
    /// Outermost label first. The last label is the innermost.
    labels: Vec<Label>,
    indent: usize,
}

impl Out {
    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("  ");
        }
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    fn seq(&mut self, body: &[Stmt]) -> Result<(), String> {
        for s in body {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Set(i, e) => self.line(&format!("(local.set {i} {})", expr(e))),
            Stmt::If(c, t, e) => {
                self.line(&format!("(if {}", cond(c)));
                self.labels.push(Label::If);
                self.indent += 1;
                self.line("(then");
                self.indent += 1;
                self.seq(t)?;
                self.indent -= 1;
                self.line(")");
                self.line("(else");
                self.indent += 1;
                self.seq(e)?;
                self.indent -= 1;
                self.line(")");
                self.labels.pop();
                self.indent -= 1;
                self.line(")");
            }
            Stmt::Block(b) => {
                self.line("(block");
                self.labels.push(Label::Block);
                self.indent += 1;
                self.seq(b)?;
                self.labels.pop();
                self.indent -= 1;
                self.line(")");
            }
            Stmt::Loop(b) => {
                self.line("(block");
                self.labels.push(Label::LoopExit);
                self.indent += 1;
                self.line("(loop");
                self.labels.push(Label::LoopRestart);
                self.indent += 1;
                self.seq(b)?;
                self.restart()?;
                self.labels.pop();
                self.indent -= 1;
                self.line(")");
                self.labels.pop();
                self.indent -= 1;
                self.line(")");
            }
            Stmt::Break(r) => {
                let innermost_first: Vec<Label> = self.labels.iter().rev().copied().collect();
                let depth = br_label(&innermost_first, *r)?;
                self.line(&format!(";; residual {} label {}", r.get(), depth.get()));
                self.line(&format!("(br {})", depth.get()));
            }
            Stmt::Return(e) => self.line(&format!("(return {})", expr(e))),
        }
        Ok(())
    }

    /// Branch to the enclosing loop's restart label. This is not a residual
    /// break: the loop falls through by starting again.
    fn restart(&mut self) -> Result<(), String> {
        let mut depth = LabelDepth::zero();
        for label in self.labels.iter().rev() {
            if matches!(label, Label::LoopRestart) {
                self.line(&format!("(br {})", depth.get()));
                return Ok(());
            }
            depth = depth.outward();
        }
        Err("a loop has no restart label".into())
    }
}

fn expr(e: &Expr) -> String {
    match e {
        Expr::Local(i) => format!("(local.get {i})"),
        Expr::I64(n) => format!("(i64.const {n})"),
        Expr::Bool(b) => format!("(i64.const {})", i64::from(*b)),
        Expr::Add(a, b) => format!("(i64.add {} {})", expr(a), expr(b)),
        Expr::Sub(a, b) => format!("(i64.sub {} {})", expr(a), expr(b)),
        Expr::Mul(a, b) => format!("(i64.mul {} {})", expr(a), expr(b)),
        Expr::Eq(a, b) => select("i64.eq", a, b),
        Expr::Lt(a, b) => select("i64.lt_s", a, b),
        Expr::Le(a, b) => select("i64.le_s", a, b),
        Expr::Not(a) => format!("(i64.xor {} (i64.const 1))", expr(a)),
    }
}

/// A comparison is i32. The local that holds it is i64, 1 when the test holds.
fn select(op: &str, a: &Expr, b: &Expr) -> String {
    format!(
        "(select (i64.const 1) (i64.const 0) ({op} {} {}))",
        expr(a),
        expr(b)
    )
}

fn cond(e: &Expr) -> String {
    format!("(i32.wrap_i64 {})", expr(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn depth(labels: &[Label], d: u32) -> u32 {
        br_label(labels, Residual::at_break(d)).unwrap().get()
    }

    #[test]
    fn residual_zero_is_the_local_exit_and_an_if_adds_a_label() {
        assert_eq!(depth(&[Label::Block], 0), 0);
        assert_eq!(depth(&[Label::If, Label::Block], 0), 1);
        assert_eq!(depth(&[Label::If, Label::If, Label::Block], 0), 2);
        assert_eq!(
            depth(&[Label::If, Label::If, Label::If, Label::Block], 0),
            3
        );
        assert_eq!(depth(&[Label::LoopRestart, Label::LoopExit], 0), 1);
        assert_eq!(
            depth(&[Label::If, Label::LoopRestart, Label::LoopExit], 0),
            2
        );
    }

    #[test]
    fn a_deeper_break_crosses_blocks_loops_and_ifs() {
        let inner = [Label::If, Label::Block, Label::LoopRestart, Label::LoopExit];
        assert_eq!(depth(&inner, 0), 1);
        assert_eq!(depth(&inner, 1), 3);
        let two = [
            Label::If,
            Label::If,
            Label::LoopRestart,
            Label::LoopExit,
            Label::If,
            Label::LoopRestart,
            Label::LoopExit,
        ];
        assert_eq!(depth(&two, 0), 3);
        assert_eq!(depth(&two, 1), 6);
        assert_eq!(depth(&[Label::Block, Label::Block], 1), 1);
        assert_eq!(
            depth(&[Label::If, Label::Block, Label::If, Label::Block], 1),
            3
        );
    }

    #[test]
    fn an_if_is_not_a_break_target() {
        let err = br_label(&[Label::If, Label::If], Residual::at_break(0)).unwrap_err();
        assert!(err.contains("names no enclosing block or loop"), "{err}");
        let err = br_label(&[Label::Block], Residual::at_break(1)).unwrap_err();
        assert!(err.contains("break 1"), "{err}");
    }

    #[test]
    fn break_zero_inside_an_if_branches_past_the_if() {
        let f = Func {
            params: 0,
            locals: 1,
            body: vec![
                Stmt::Set(0, Expr::I64(0)),
                Stmt::Block(vec![
                    Stmt::If(
                        Expr::Bool(true),
                        vec![
                            Stmt::Set(0, Expr::I64(1)),
                            Stmt::Break(Residual::at_break(0)),
                        ],
                        vec![],
                    ),
                    Stmt::Set(0, Expr::I64(9)),
                ]),
                Stmt::Return(Expr::Local(0)),
            ],
        };
        let wat = emit_func(&f).unwrap();
        assert!(wat.contains(";; residual 0 label 1"), "{wat}");
        assert!(wat.contains("(br 1)"), "{wat}");
        assert!(!wat.contains("(br 0)"), "{wat}");
    }
}
