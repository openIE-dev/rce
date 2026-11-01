//! `rce emit verilog FILE` (spec 10.5; the proposal
//! `spec/proposals/emit-verilog.md`, EV-1 to EV-8): the step function δ of
//! a module as one synthesizable Verilog-2005 module (IEEE 1364-2005).
//!
//! The module holds the state in registers, reset to the initial state s0,
//! reads one event per clock from an event bus (a tag, a payload and a valid
//! strobe), and computes δ in one combinational block: every loop is
//! unrolled to its constant bound, every function call is inlined, and every
//! integer operation is computed exactly at a width that holds its result
//! and then checked against the range of its type (6.3 N2), so an overflow
//! keeps the state in the circuit exactly where the reference evaluator
//! keeps it. The outcome (applied, or kept with the declared error or the
//! trap and its value) and the emitted values are registered outputs.
//!
//! A program outside the subset of EV-1 is not printed: the result is
//! `requires change for target verilog: REASON, line N`.
//!
//! The first comment lines carry the interface as one line of JSON
//! (`rce-hdl/1`): the bus layout, the state layout, the trap and error
//! tables and the program's example events. `scripts/verilog-bench.py`
//! reads it to drive a simulator and to print the trace of every example
//! from the register dumps (EV-6).
use crate::ast::*;
use crate::interp::{Flow, Interp, V};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

/// The most Verilog statements the step logic of one module may unroll to
/// (EV-4, the size ceiling of `emit-targets.md` Tier 3).
pub const CEILING: usize = 20_000;
/// The most values one step may emit (EV-3).
pub const MAX_EMITS: usize = 64;

/// Why a program is outside the subset, and the line.
struct Refuse {
    why: String,
    line: usize,
}

type R<T> = Result<T, Refuse>;

fn no<T>(why: impl Into<String>, line: usize) -> R<T> {
    Err(Refuse { why: why.into(), line })
}

/// The Verilog module of `m`, a module that `rce check` admits, or
/// `requires change for target verilog: REASON, line N`.
pub fn emit(m: &Module) -> Result<String, String> {
    let provenance = format!(
        "// rce-program {} · emitted by rce commit {}",
        crate::apidiff::program_hash(std::slice::from_ref(m)),
        crate::GIT_COMMIT,
    );
    match build(m) {
        Ok(body) => Ok(format!("{provenance}\n{body}")),
        Err(r) => Err(format!("requires change for target verilog: {}, line {}", r.why, r.line)),
    }
}

// ---------------------------------------------------------------- types

/// A type the circuit holds (EV-1): a fixed-width integer, a bool, a kind
/// whose variants carry no fields, or a fixed-size array of these.
#[derive(Clone, PartialEq, Debug)]
enum HT {
    Int(&'static str),
    Bool,
    Kind(String),
    Arr(Box<HT>, usize),
}

/// The width in bits of an integer type (6.3); `lit` is held as i64.
fn int_bits(t: &str) -> u32 {
    match t {
        "i8" | "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        _ => 64,
    }
}

/// The range a value with this type name is checked against (`lit` as i64).
fn range(t: &str) -> (i128, i128) {
    crate::types::int_range(if t == "lit" { "i64" } else { t })
}

/// The signed width that holds `v`.
fn swidth(v: i128) -> u32 {
    if v >= 0 { 129 - v.leading_zeros() } else { 129 - (!v).leading_zeros() }
}

/// The signed width that holds every value of the type.
fn carrier(t: &str) -> u32 {
    let (lo, hi) = range(t);
    swidth(lo).max(swidth(hi))
}

/// Bits to number `n` distinct values, at least 1.
fn bits_for(n: usize) -> u32 {
    let mut b = 1;
    while (1usize << b) < n {
        b += 1;
    }
    b
}

/// A signed Verilog literal and its width.
fn lit(v: i128) -> (String, u32) {
    let w = swidth(v).max(2);
    if v >= 0 { (format!("{w}'sd{v}"), w) } else { (format!("(-{}'sd{})", w + 1, v.unsigned_abs()), w + 1) }
}

fn join_tag(a: &'static str, b: &'static str) -> &'static str {
    if a == "lit" { b } else { a }
}

// --------------------------------------------------------------- values

/// A value during compilation: a Verilog expression and what is known of
/// it. An integer carries the type name the reference evaluator's value
/// would carry (`lit` for an untyped literal, 6.3), a signed width that
/// holds it, and its value when it is a constant.
#[derive(Clone, Debug)]
enum Val {
    Int { e: String, tag: &'static str, w: u32, k: Option<i128> },
    Bool { e: String, k: Option<bool> },
    Kind { kind: String, e: String, k: Option<usize> },
    Arr(Vec<Val>),
}

fn int_k(v: i128, tag: &'static str) -> Val {
    let (e, w) = lit(v);
    Val::Int { e, tag, w, k: Some(v) }
}

fn bool_k(b: bool) -> Val {
    Val::Bool { e: if b { "1'b1".into() } else { "1'b0".into() }, k: Some(b) }
}

fn bool_e(e: String) -> Val {
    Val::Bool { e, k: None }
}

impl Val {
    fn shape(&self) -> String {
        match self {
            Val::Int { tag, .. } => format!("an integer ({tag})"),
            Val::Bool { .. } => "a bool".into(),
            Val::Kind { kind, .. } => format!("a {kind}"),
            Val::Arr(xs) => format!("an array of {}", xs.len()),
        }
    }
}

/// How a trap reads in D (`trap(TEXT)`); the code on `out_trap` is its
/// position in this table plus one.
#[derive(Clone, PartialEq, Debug)]
enum Trap {
    /// `overflow: V is outside T`, V on `out_trap_value`
    Overflow(&'static str),
    /// `index K out of range`, K on `out_trap_value`
    Index,
    /// a fixed text
    Text(String),
}

/// One emitted value's kind on `out_emit_type`.
#[derive(Clone, PartialEq, Debug)]
enum EmitTy {
    Int,
    Bool,
    Kind(String),
}

#[derive(Clone)]
struct Local {
    name: String,
    v: Val,
}

#[derive(Clone, Copy, Default)]
struct J {
    /// may set a flag of `return`, `break` or `continue`
    may: bool,
    /// always leaves the block (a jump, a `fail` or a trap)
    must: bool,
}

impl std::ops::BitOr for J {
    type Output = J;
    fn bitor(self, o: J) -> J {
        J { may: self.may || o.may, must: self.must && o.must }
    }
}

struct Frame {
    /// the flag `return` sets, in a function
    ret: Option<String>,
    /// the local that holds the result, and its declared type
    res: Option<(usize, HT)>,
    /// the `break` and `continue` flags of the loops open in this frame
    loops: Vec<(String, String)>,
}

#[derive(Clone)]
struct Snap {
    locals: Vec<Local>,
    emits: (usize, usize),
}

struct Reg {
    name: String,
    w: u32,
    signed: bool,
    note: String,
}

/// Where a function body's tail value goes.
#[derive(Clone)]
struct Tail {
    res: usize,
    ty: HT,
}

struct Cx<'m> {
    m: &'m Module,
    it: Interp<'m>,
    kinds: BTreeMap<String, Vec<String>>,
    variant: HashMap<String, (String, usize)>,
    events: Vec<(String, Vec<(String, HT)>)>,
    errors: Vec<String>,
    fns: HashMap<String, &'m FnDecl>,
    consts: HashSet<String>,
    regs: Vec<Reg>,
    reg_ix: HashMap<String, usize>,
    body: Vec<String>,
    ind: usize,
    scopes: Vec<HashMap<String, usize>>,
    locals: Vec<Local>,
    frames: Vec<Frame>,
    traps: Vec<Trap>,
    trap_w: u32,
    emit_tys: Vec<EmitTy>,
    emits: (usize, usize),
    max_emits: usize,
    emit_w: u32,
    line: usize,
    calls: Vec<String>,
}

// ------------------------------------------------------------ the module

fn build(m: &Module) -> R<String> {
    if let Some(e) = crate::v09::unemittable(m) {
        return no(e, 0);
    }
    let mut cx = Cx::new(m)?;
    cx.admit_items()?;
    let regions: Vec<(String, Vec<(String, HT)>, usize)> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::State { name, fields, line, .. } => Some((name, fields, *line)),
            _ => None,
        })
        .map(|(n, fs, line)| {
            let mut out = vec![];
            for f in fs {
                out.push((f.name.clone(), cx.ht(&f.ty, f.line)?));
            }
            Ok((n.clone(), out, line))
        })
        .collect::<R<_>>()?;
    if regions.is_empty() {
        return no("a module without a `state` region has no step function to print", 0);
    }
    // the initial state s0, from the reference evaluator
    let s0 = match cx.it.initial_state() {
        Ok(s) => s,
        Err(f) => return no(format!("the initial state does not evaluate ({})", crate::interp::why(f)), 0),
    };
    let tag_w = bits_for(cx.events.len());
    // the event bus
    let mut payload_w = 1;
    let mut ev_json = vec![];
    for (name, fs) in &cx.events {
        let mut off = 0;
        let mut fj = vec![];
        for (f, t) in fs {
            let w = cx.ht_bits(t);
            fj.push(json!({"name": f, "type": cx.ht_json(t), "offset": off, "width": w}));
            off += w;
        }
        payload_w = payload_w.max(off);
        ev_json.push(json!({"name": name, "fields": fj}));
    }
    // the state registers
    let mut st_off = 0;
    let mut reg_json = vec![];
    let mut sregs: Vec<Vec<(String, HT, u32)>> = vec![];
    for (ri, (rn, fs, _)) in regions.iter().enumerate() {
        let mut fj = vec![];
        let mut rr = vec![];
        for (fi, (f, t)) in fs.iter().enumerate() {
            let w = cx.ht_bits(t);
            fj.push(json!({"name": f, "type": cx.ht_json(t), "offset": st_off, "width": w}));
            rr.push((format!("s{ri}_{fi}"), t.clone(), w));
            st_off += w;
        }
        sregs.push(rr);
        reg_json.push(json!({"name": rn, "fields": fj}));
    }
    let state_w = st_off.max(1);
    let k = cx.reg_named("k", 1, false, "kept: the event does not apply");
    let kerr = cx.reg_named("kerr", 1, false, "the declared error of a `fail`, plus one");
    let ktrap = cx.reg_named("ktrap", 1, false, "the trap, plus one");
    let kval = cx.reg_named("kval", 2, true, "the trap's value");
    let ecnt = cx.reg_named("ecnt", 1, false, "values emitted");
    let _ = (k, kerr, ktrap, kval, ecnt);
    // δ, region by region in declaration order (6.4)
    for (ri, (rn, fs, _)) in regions.iter().enumerate() {
        let sel = cx.reg_named(&format!("sel{ri}"), 1, false, &format!("a step of {rn} was taken"));
        let steps: Vec<&Item> = m.items.iter().filter(|i| matches!(i, Item::Step { region, .. } if region == rn)).collect();
        for st in steps {
            let Item::Step { pat, guard, body, line, deadline, .. } = st else { unreachable!() };
            if deadline.is_some() {
                return no("a step with a `deadline`", *line);
            }
            cx.line = *line;
            cx.out(format!("// step {rn} on {} (line {line})", show_pat(pat)));
            let fields: Vec<(String, HT, String)> = fs.iter().enumerate().map(|(fi, (f, t))| (f.clone(), t.clone(), format!("s{ri}_{fi}"))).collect();
            let gate = format!("!k && !{sel}");
            cx.gate(&gate, |cx| cx.step(pat, guard.as_ref(), body, &fields, &sel, ri, *line))?;
            if cx.body.len() > CEILING {
                return no(format!("the step logic unrolls to more than {CEILING} statements (the size ceiling of EV-4)"), *line);
            }
        }
    }
    // the tables of the interface
    let err_w = bits_for(cx.errors.len() + 1);
    let trap_w = bits_for(cx.traps.len() + 1);
    let tval_w = cx.trap_w.max(2);
    cx.widen("kerr", err_w);
    cx.widen("ktrap", trap_w);
    cx.widen("kval", tval_w);
    let ne = cx.max_emits;
    let ncw = bits_for(ne + 1);
    cx.widen("ecnt", ncw);
    let etw = bits_for(cx.emit_tys.len());
    let evw = cx.emit_w.max(1);
    for s in 0..ne {
        cx.reg_named(&format!("et{s}"), etw, false, &format!("the kind of emitted value {s}"));
        cx.reg_named(&format!("ev{s}"), evw, false, &format!("emitted value {s}"));
    }
    let traps: Vec<Value> = cx
        .traps
        .iter()
        .map(|t| match t {
            Trap::Overflow(ty) => json!({"kind": "overflow", "type": ty}),
            Trap::Index => json!({"kind": "index"}),
            Trap::Text(s) => json!({"kind": "text", "text": s}),
        })
        .collect();
    let emit_tys: Vec<Value> = cx
        .emit_tys
        .iter()
        .map(|t| match t {
            EmitTy::Int => json!("int"),
            EmitTy::Bool => json!("bool"),
            EmitTy::Kind(k) => json!(k),
        })
        .collect();
    let examples = cx.examples()?;
    let module = format!("rce_{}", regions.iter().map(|r| r.0.as_str()).collect::<Vec<_>>().join("_"));
    let iface = json!({
        "format": "rce-hdl/1",
        "module": module,
        "tag_width": tag_w,
        "payload_width": payload_w,
        "state_width": state_w,
        "error_width": err_w,
        "trap_width": trap_w,
        "trap_value_width": tval_w,
        "emit_slots": ne,
        "emit_count_width": ncw,
        "emit_type_width": etw,
        "emit_value_width": evw,
        "events": ev_json,
        "regions": reg_json,
        "errors": cx.errors,
        "traps": traps,
        "emit_types": emit_tys,
        "kinds": cx.kinds,
        "examples": examples,
    });
    // the text
    let mut t = String::new();
    let p = |t: &mut String, s: &str| {
        t.push_str(s);
        t.push('\n');
    };
    p(&mut t, "// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function");
    p(&mut t, "// of this program as one synthesizable Verilog-2005 module. One event per clock:");
    p(&mut t, "// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)");
    p(&mut t, "// is taken; out_* hold its outcome from that edge, and the state registers change");
    p(&mut t, "// only when out_applied is 1. rst (synchronous, active high) loads the initial state.");
    p(&mut t, &format!("// rce-hdl/1 {iface}"));
    p(&mut t, &format!("module {module} ("));
    let ports = [
        "input  wire clk".to_string(),
        "input  wire rst".into(),
        "input  wire ev_valid".into(),
        format!("input  wire [{}:0] ev_tag", tag_w - 1),
        format!("input  wire [{}:0] ev_payload", payload_w - 1),
        "output reg  out_valid".into(),
        "output reg  out_applied".into(),
        format!("output reg  [{}:0] out_error", err_w - 1),
        format!("output reg  [{}:0] out_trap", trap_w - 1),
        format!("output reg  [{}:0] out_trap_value", tval_w - 1),
        format!("output reg  [{}:0] out_emit_count", ncw - 1),
        format!("output reg  [{}:0] out_emit_type", (ne.max(1) as u32) * etw - 1),
        format!("output reg  [{}:0] out_emit_value", (ne.max(1) as u32) * evw - 1),
        format!("output wire [{}:0] state", state_w - 1),
    ];
    for (i, port) in ports.iter().enumerate() {
        p(&mut t, &format!("    {port}{}", if i + 1 < ports.len() { "," } else { "" }));
    }
    p(&mut t, ");");
    p(&mut t, "    // the state registers, first field at bit 0 of `state`");
    let mut packed = vec![];
    for (ri, (rn, fs, _)) in regions.iter().enumerate() {
        for (fi, (f, ty)) in fs.iter().enumerate() {
            let w = sregs[ri][fi].2;
            p(&mut t, &format!("    reg [{}:0] s{ri}_{fi}; // {rn}.{f}: {}", w - 1, cx.ht_name(ty)));
            p(&mut t, &format!("    reg [{}:0] nx{ri}_{fi};", w - 1));
            packed.push(format!("s{ri}_{fi}"));
        }
    }
    packed.reverse();
    if packed.is_empty() {
        p(&mut t, "    assign state = 1'b0;");
    } else {
        p(&mut t, &format!("    assign state = {{{}}};", packed.join(", ")));
    }
    p(&mut t, "    // the step's working registers");
    for r in &cx.regs {
        let s = if r.signed { "signed " } else { "" };
        let note = if r.note.is_empty() { String::new() } else { format!(" // {}", r.note) };
        p(&mut t, &format!("    reg {s}[{}:0] {};{note}", r.w - 1, r.name));
    }
    p(&mut t, "    always @* begin");
    for (ri, fs) in sregs.iter().enumerate() {
        for fi in 0..fs.len() {
            p(&mut t, &format!("        nx{ri}_{fi} = s{ri}_{fi};"));
        }
    }
    for r in &cx.regs {
        p(&mut t, &format!("        {} = {}'d0;", r.name, r.w));
    }
    for l in &cx.body {
        p(&mut t, &format!("        {l}"));
    }
    p(&mut t, "    end");
    p(&mut t, "    always @(posedge clk) begin");
    p(&mut t, "        if (rst) begin");
    for (ri, fs) in sregs.iter().enumerate() {
        for (fi, (_, ty, w)) in fs.iter().enumerate() {
            let v = s0[ri].1[fi].1.clone();
            let mut bits = vec![];
            cx.value_bits(&v, ty, &mut bits);
            p(&mut t, &format!("            s{ri}_{fi} <= {w}'h{};", hex(&bits)));
        }
    }
    for (port, w) in [
        ("out_valid", 1),
        ("out_applied", 1),
        ("out_error", err_w),
        ("out_trap", trap_w),
        ("out_trap_value", tval_w),
        ("out_emit_count", ncw),
        ("out_emit_type", (ne.max(1) as u32) * etw),
        ("out_emit_value", (ne.max(1) as u32) * evw),
    ] {
        p(&mut t, &format!("            {port} <= {w}'d0;"));
    }
    p(&mut t, "        end else begin");
    p(&mut t, "            out_valid <= ev_valid;");
    p(&mut t, "            if (ev_valid) begin");
    p(&mut t, "                out_applied <= !k;");
    p(&mut t, "                out_error <= kerr;");
    p(&mut t, "                out_trap <= ktrap;");
    p(&mut t, "                out_trap_value <= kval;");
    p(&mut t, &format!("                out_emit_count <= k ? {ncw}'d0 : ecnt;"));
    if ne == 0 {
        p(&mut t, &format!("                out_emit_type <= {etw}'d0;"));
        p(&mut t, &format!("                out_emit_value <= {evw}'d0;"));
    } else {
        let ts: Vec<String> = (0..ne).rev().map(|s| format!("et{s}")).collect();
        let vs: Vec<String> = (0..ne).rev().map(|s| format!("ev{s}")).collect();
        p(&mut t, &format!("                out_emit_type <= {{{}}};", ts.join(", ")));
        p(&mut t, &format!("                out_emit_value <= {{{}}};", vs.join(", ")));
    }
    p(&mut t, "                if (!k) begin");
    for (ri, fs) in sregs.iter().enumerate() {
        for fi in 0..fs.len() {
            p(&mut t, &format!("                    s{ri}_{fi} <= nx{ri}_{fi};"));
        }
    }
    p(&mut t, "                end");
    p(&mut t, "            end");
    p(&mut t, "        end");
    p(&mut t, "    end");
    p(&mut t, "endmodule");
    Ok(t)
}

fn hex(bits: &[bool]) -> String {
    if bits.is_empty() {
        return "0".into();
    }
    let n = bits.len().div_ceil(4);
    let mut s = String::new();
    for d in (0..n).rev() {
        let mut v = 0;
        for b in 0..4 {
            if bits.get(d * 4 + b).copied().unwrap_or(false) {
                v |= 1 << b;
            }
        }
        s.push(std::char::from_digit(v, 16).unwrap());
    }
    s
}

fn show_pat(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".into(),
        Pat::Path(x) => x.join("."),
        Pat::Ctor(x, args) => format!(
            "{}({})",
            x.join("."),
            args.iter()
                .map(|(l, q)| match l {
                    Some(l) => format!("{l}: {}", show_pat(q)),
                    None => show_pat(q),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pat::Lit(e) => match e {
            Expr::Int(v, _) => v.to_string(),
            Expr::Bool(b, _) => b.to_string(),
            Expr::Neg(x, _) => match &**x {
                Expr::Int(v, _) => format!("-{v}"),
                _ => "<literal>".into(),
            },
            _ => "<literal>".into(),
        },
        Pat::Range(..) => "<range>".into(),
        Pat::Tuple(ps) => format!("({})", ps.iter().map(show_pat).collect::<Vec<_>>().join(", ")),
        Pat::Alt(ps) => ps.iter().map(show_pat).collect::<Vec<_>>().join(" | "),
    }
}

fn stmt_word(s: &Stmt) -> &'static str {
    match s {
        Stmt::Send(..) => "send (`<-`)",
        Stmt::Spawn(_) => "spawn",
        Stmt::Select(..) => "select",
        Stmt::Arena(..) => "with arena",
        Stmt::Device(..) => "device",
        Stmt::Yield(..) => "yield",
        Stmt::Comptime(_) => "comptime",
        Stmt::Check(_) => "check",
        _ => "statement",
    }
}

impl<'m> Cx<'m> {
    fn new(m: &'m Module) -> R<Self> {
        let mut kinds = BTreeMap::new();
        let mut variant = HashMap::new();
        let mut events = vec![];
        let mut errors = vec![];
        let mut fns = HashMap::new();
        let mut consts = HashSet::new();
        for i in &m.items {
            match i {
                Item::Kind { name, generics, variants, .. } if generics.is_empty() && variants.iter().all(|v| v.1.is_empty()) => {
                    for (k, (v, _)) in variants.iter().enumerate() {
                        variant.insert(v.clone(), (name.clone(), k));
                    }
                    kinds.insert(name.clone(), variants.iter().map(|v| v.0.clone()).collect::<Vec<_>>());
                }
                Item::Error { name, .. } => errors.push(name.clone()),
                Item::Fn(f) => {
                    fns.insert(f.sig.name.clone(), f);
                }
                Item::Const { name, .. } => {
                    consts.insert(name.clone());
                }
                _ => {}
            }
        }
        let mut cx = Cx {
            m,
            it: Interp::new(m),
            kinds,
            variant,
            events: vec![],
            errors,
            fns,
            consts,
            regs: vec![],
            reg_ix: HashMap::new(),
            body: vec![],
            ind: 0,
            scopes: vec![],
            locals: vec![],
            frames: vec![],
            traps: vec![],
            trap_w: 2,
            emit_tys: vec![],
            emits: (0, 0),
            max_emits: 0,
            emit_w: 1,
            line: 0,
            calls: vec![],
        };
        for i in &m.items {
            if let Item::Event { name, payload, line, interrupt, foreign } = i {
                if interrupt.is_some() {
                    return no("an `interrupt event`", *line);
                }
                if foreign.is_some() {
                    return no("an `extern` event", *line);
                }
                let mut fs = vec![];
                for (f, t) in payload {
                    fs.push((f.clone(), cx.ht(t, *line)?));
                }
                events.push((name.clone(), fs));
            }
        }
        cx.events = events;
        Ok(cx)
    }

    /// EV-1: the items the circuit is made from.
    fn admit_items(&self) -> R<()> {
        for i in &self.m.items {
            let bad = match i {
                Item::Fn(f) if f.sig.name == "main" => Some(("`fn main` (tasks run on the tick runtime, not in a step)", f.sig.line)),
                Item::Actor { line, .. } => Some(("an `actor`", *line)),
                Item::Extern { line, .. } => Some(("an `extern` function", *line)),
                Item::Impl { line, .. } => Some(("an `impl` block", *line)),
                Item::Trait { line, .. } => Some(("a `trait`", *line)),
                Item::Protocol { line, .. } => Some(("a `protocol`", *line)),
                Item::Node { line, .. } => Some(("a `node`", *line)),
                Item::Supervisor { line, .. } => Some(("a `supervisor`", *line)),
                Item::Use(_, _, line) => Some(("a `use` of another module", *line)),
                Item::State { replicated: true, line, .. } => Some(("a `replicated state`", *line)),
                _ => None,
            };
            if let Some((w, l)) = bad {
                return no(w, l);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ types

    fn ht(&self, t: &Ty, line: usize) -> R<HT> {
        match t {
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() => {
                let n = p[0].as_str();
                if let Some(i) = crate::types::INTS.iter().find(|x| **x == n) {
                    return Ok(HT::Int(i));
                }
                if n == "bool" {
                    return Ok(HT::Bool);
                }
                if self.kinds.contains_key(n) {
                    return Ok(HT::Kind(n.to_string()));
                }
                if self.m.items.iter().any(|i| matches!(i, Item::Kind { name, .. } if name == n)) {
                    return no(format!("the kind `{n}`, whose variants carry fields"), line);
                }
                no(format!("the type `{}`: the circuit holds integers, bool, kinds without fields and fixed arrays (EV-1)", crate::interp::ty_str(t)), line)
            }
            Ty::Array(e, n) => {
                let el = self.ht(e, line)?;
                if matches!(el, HT::Arr(..)) {
                    return no("an array of arrays", line);
                }
                let mut it = Interp::new(self.m);
                it.scopes = vec![HashMap::new()];
                match it.eval(n) {
                    Ok(V::Int(k, _)) if (0..=4096).contains(&k) => Ok(HT::Arr(Box::new(el), k as usize)),
                    _ => no("an array whose length is not a constant up to 4096", line),
                }
            }
            _ => no(format!("the type `{}`: the circuit holds integers, bool, kinds without fields and fixed arrays (EV-1)", crate::interp::ty_str(t)), line),
        }
    }

    fn kind_bits(&self, k: &str) -> u32 {
        bits_for(self.kinds[k].len())
    }

    fn ht_bits(&self, t: &HT) -> u32 {
        match t {
            HT::Int(n) => int_bits(n),
            HT::Bool => 1,
            HT::Kind(k) => self.kind_bits(k),
            HT::Arr(e, n) => self.ht_bits(e) * *n as u32,
        }
    }

    fn ht_json(&self, t: &HT) -> Value {
        match t {
            HT::Int(n) => json!(n),
            HT::Bool => json!("bool"),
            HT::Kind(k) => json!({"kind": k}),
            HT::Arr(e, n) => json!({"array": self.ht_json(e), "len": n}),
        }
    }

    fn ht_name(&self, t: &HT) -> String {
        match t {
            HT::Int(n) => n.to_string(),
            HT::Bool => "bool".into(),
            HT::Kind(k) => k.clone(),
            HT::Arr(e, n) => format!("[{}; {n}]", self.ht_name(e)),
        }
    }

    /// The bits of a value of type `t`, least significant first.
    fn value_bits(&self, v: &V, t: &HT, out: &mut Vec<bool>) {
        match (v, t) {
            (V::Int(x, _), HT::Int(n)) => {
                let b = int_bits(n);
                for i in 0..b {
                    out.push((x >> i) & 1 == 1);
                }
            }
            (V::Bool(b), HT::Bool) => out.push(*b),
            (V::Rec(n, _), HT::Kind(k)) => {
                let i = self.kinds[k].iter().position(|x| x == n).unwrap_or(0);
                for b in 0..self.kind_bits(k) {
                    out.push((i >> b) & 1 == 1);
                }
            }
            (V::List(xs), HT::Arr(e, _)) => {
                for x in xs.iter() {
                    self.value_bits(x, e, out);
                }
            }
            _ => {
                for _ in 0..self.ht_bits(t) {
                    out.push(false);
                }
            }
        }
    }

    fn value_json(&self, v: &V) -> Value {
        match v {
            V::Int(x, _) => serde_json::from_str(&x.to_string()).unwrap_or(Value::Null),
            V::Bool(b) => json!(b),
            V::Rec(n, _) => json!(n),
            V::List(xs) => Value::Array(xs.iter().map(|x| self.value_json(x)).collect()),
            _ => Value::Null,
        }
    }

    /// The example events of the program, for the test bench (EV-6).
    fn examples(&self) -> R<Vec<Value>> {
        let mut xs: Vec<&Example> = vec![];
        for i in &self.m.items {
            match i {
                Item::Intent(_, ex, _) => xs.extend(ex.iter()),
                Item::Example(x) => xs.push(x),
                _ => {}
            }
        }
        let mut out = vec![];
        for x in xs {
            let mut it = Interp::new(self.m);
            let mut evs = vec![];
            for (evn, args) in &x.events {
                it.scopes = vec![HashMap::new()];
                let Some(tag) = self.events.iter().position(|e| &e.0 == evn) else {
                    return no(format!("the example event `{evn}`, which is not a declared event"), x.line);
                };
                let v = match it.construct(evn, args) {
                    Ok(Some(v)) => v,
                    _ => return no(format!("the example event `{evn}`, which does not construct"), x.line),
                };
                let V::Rec(_, fs) = &v else { return no("an example event", x.line) };
                evs.push(json!({"tag": tag, "values": fs.iter().map(|(_, fv)| self.value_json(fv)).collect::<Vec<_>>()}));
            }
            out.push(json!({"line": x.line, "events": evs}));
        }
        Ok(out)
    }

    // -------------------------------------------------------- emission

    fn out(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.body.push(format!("{}{s}", "    ".repeat(self.ind)));
    }

    fn reg_named(&mut self, name: &str, w: u32, signed: bool, note: &str) -> String {
        self.reg_ix.insert(name.to_string(), self.regs.len());
        self.regs.push(Reg { name: name.to_string(), w: w.max(1), signed, note: note.to_string() });
        name.to_string()
    }

    fn reg(&mut self, prefix: &str, w: u32, signed: bool, note: &str) -> String {
        let name = format!("{prefix}{}", self.regs.len());
        self.reg_named(&name, w, signed, note)
    }

    fn widen(&mut self, name: &str, w: u32) {
        if let Some(&i) = self.reg_ix.get(name) {
            self.regs[i].w = self.regs[i].w.max(w);
        }
    }

    fn trap_code(&mut self, t: Trap) -> usize {
        match self.traps.iter().position(|x| *x == t) {
            Some(i) => i + 1,
            None => {
                self.traps.push(t);
                self.traps.len()
            }
        }
    }

    /// Keep the state with trap `t` when `cond` holds (and nothing was kept
    /// before), recording `val` (a signed expression of width `w`).
    fn trap(&mut self, cond: Option<&str>, t: Trap, val: Option<(&str, u32)>) {
        let code = self.trap_code(t);
        let c = match cond {
            Some(c) => format!("!k && ({c})"),
            None => "!k".into(),
        };
        let mut set = format!("k = 1'b1; ktrap = {code};");
        if let Some((e, w)) = val {
            self.trap_w = self.trap_w.max(w);
            set.push_str(&format!(" kval = {e};"));
        }
        self.out(format!("if ({c}) begin {set} end"));
    }

    fn trap_text(&mut self, cond: Option<&str>, text: &str) {
        self.trap(cond, Trap::Text(text.to_string()), None);
    }

    fn live(&self) -> String {
        let mut p = vec!["!k".to_string()];
        if let Some(f) = self.frames.last() {
            if let Some(r) = &f.ret {
                p.push(format!("!{r}"));
            }
            if let Some((b, c)) = f.loops.last() {
                p.push(format!("!{b}"));
                p.push(format!("!{c}"));
            }
        }
        p.join(" && ")
    }

    fn snap(&self) -> Snap {
        Snap { locals: self.locals.clone(), emits: self.emits }
    }

    fn restore(&mut self, s: Snap) {
        self.locals = s.locals;
        self.emits = s.emits;
    }

    fn merge_val(a: &Val, b: &Val) -> Option<Val> {
        match (a, b) {
            (Val::Int { e, tag, w, k }, Val::Int { tag: t2, w: w2, k: k2, .. }) if tag == t2 => {
                Some(Val::Int { e: e.clone(), tag, w: (*w).max(*w2), k: if k == k2 { *k } else { None } })
            }
            (Val::Bool { e, k }, Val::Bool { k: k2, .. }) => Some(Val::Bool { e: e.clone(), k: if k == k2 { *k } else { None } }),
            (Val::Kind { kind, e, k }, Val::Kind { kind: k2n, k: k2, .. }) if kind == k2n => {
                Some(Val::Kind { kind: kind.clone(), e: e.clone(), k: if k == k2 { *k } else { None } })
            }
            (Val::Arr(xs), Val::Arr(ys)) if xs.len() == ys.len() => {
                xs.iter().zip(ys).map(|(x, y)| Self::merge_val(x, y)).collect::<Option<Vec<_>>>().map(Val::Arr)
            }
            _ => None,
        }
    }

    /// The state after two paths: `a` (which left the block when `ja.must`)
    /// and `b`; a local whose integer type differs on the two paths requires
    /// change, since its type would not be static.
    fn merge(&mut self, a: Snap, ja: J, b: Snap, jb: J, base: usize) -> R<()> {
        if ja.must && !jb.must {
            self.restore(b);
            return Ok(());
        }
        if jb.must {
            self.restore(a);
            return Ok(());
        }
        let mut out = a.locals.clone();
        for (i, l) in out.iter_mut().enumerate().take(base.min(b.locals.len())) {
            match Self::merge_val(&a.locals[i].v, &b.locals[i].v) {
                Some(v) => l.v = v,
                None => {
                    return no(
                        format!(
                            "`{}` is {} on one path and {} on another, so its type is not static (EV-2)",
                            l.name,
                            a.locals[i].v.shape(),
                            b.locals[i].v.shape()
                        ),
                        self.line,
                    );
                }
            }
        }
        self.locals = out;
        self.emits = (a.emits.0.min(b.emits.0), a.emits.1.max(b.emits.1));
        Ok(())
    }

    /// `if (c) A else B`, merging the two paths.
    fn branch(&mut self, c: &str, a: impl FnOnce(&mut Self) -> R<J>, b: Option<&mut dyn FnMut(&mut Self) -> R<J>>) -> R<J> {
        let base = self.snap();
        let n = base.locals.len();
        self.out(format!("if ({c}) begin"));
        self.ind += 1;
        let ja = a(self)?;
        self.ind -= 1;
        let sa = self.snap();
        self.restore(base.clone());
        let jb = match b {
            Some(f) => {
                self.out("end else begin");
                self.ind += 1;
                let j = f(self)?;
                self.ind -= 1;
                j
            }
            None => J::default(),
        };
        self.out("end");
        let sb = self.snap();
        self.merge(sa, ja, sb, jb, n)?;
        Ok(ja | jb)
    }

    fn gate(&mut self, c: &str, a: impl FnOnce(&mut Self) -> R<J>) -> R<J> {
        self.branch(c, a, None)
    }

    // ---------------------------------------------------------- locals

    fn lookup(&self, n: &str) -> Option<usize> {
        self.scopes.iter().rev().find_map(|s| s.get(n).copied())
    }

    fn bind(&mut self, n: &str, id: usize) {
        self.scopes.last_mut().expect("a scope").insert(n.to_string(), id);
    }

    /// A new local holding `v` (registers of its shape).
    fn new_local(&mut self, name: &str, v: &Val) -> R<usize> {
        let shape = self.fresh(name, v);
        self.locals.push(Local { name: name.to_string(), v: shape });
        let id = self.locals.len() - 1;
        self.store(id, v)?;
        Ok(id)
    }

    fn fresh(&mut self, name: &str, v: &Val) -> Val {
        match v {
            Val::Int { tag, w, .. } => Val::Int { e: self.reg("l", *w, true, name), tag, w: *w, k: None },
            Val::Bool { .. } => Val::Bool { e: self.reg("l", 1, false, name), k: None },
            Val::Kind { kind, .. } => {
                let b = self.kind_bits(kind);
                Val::Kind { kind: kind.clone(), e: self.reg("l", b, false, name), k: None }
            }
            Val::Arr(xs) => Val::Arr(xs.iter().map(|x| self.fresh(name, x)).collect()),
        }
    }

    /// Store `v` into local `id` (the reference evaluator's `set`: the
    /// local takes the value, and with it the value's type name).
    fn store(&mut self, id: usize, v: &Val) -> R<()> {
        let cur = self.locals[id].v.clone();
        let nv = self.store_val(&cur, v, &self.locals[id].name.clone())?;
        self.locals[id].v = nv;
        Ok(())
    }

    fn store_val(&mut self, cur: &Val, v: &Val, name: &str) -> R<Val> {
        match (cur, v) {
            (Val::Int { e: r, .. }, Val::Int { e, tag, w, k }) => {
                self.out(format!("{r} = {e};"));
                self.widen(r, *w);
                Ok(Val::Int { e: r.clone(), tag, w: *w, k: *k })
            }
            (Val::Bool { e: r, .. }, Val::Bool { e, k }) => {
                self.out(format!("{r} = {e};"));
                Ok(Val::Bool { e: r.clone(), k: *k })
            }
            (Val::Kind { kind, e: r, .. }, Val::Kind { kind: k2, e, k }) if kind == k2 => {
                self.out(format!("{r} = {e};"));
                Ok(Val::Kind { kind: kind.clone(), e: r.clone(), k: *k })
            }
            (Val::Arr(xs), Val::Arr(ys)) if xs.len() == ys.len() => {
                let mut out = vec![];
                for (x, y) in xs.iter().zip(ys) {
                    out.push(self.store_val(x, y, name)?);
                }
                Ok(Val::Arr(out))
            }
            _ => no(format!("`{name}` holds {} and is given {}", cur.shape(), v.shape()), self.line),
        }
    }

    /// A temporary holding `v`, so that later stores do not change it.
    fn hold(&mut self, v: &Val) -> R<Val> {
        if matches!(v, Val::Int { k: Some(_), .. } | Val::Bool { k: Some(_), .. } | Val::Kind { k: Some(_), .. }) {
            return Ok(v.clone());
        }
        let id = self.new_local("held value", v)?;
        Ok(self.locals[id].v.clone())
    }

    // -------------------------------------------------------------- steps

    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, pat: &Pat, guard: Option<&Expr>, body: &Body, fields: &[(String, HT, String)], sel: &str, ri: usize, line: usize) -> R<J> {
        let saved = std::mem::take(&mut self.scopes);
        self.frames.push(Frame { ret: None, res: None, loops: vec![] });
        self.scopes = vec![HashMap::new(), HashMap::new()];
        // the fields, as the pre-state holds them
        let mut ids = vec![];
        for (f, t, r) in fields {
            let v = self.from_bits(t, r, 0);
            let id = self.new_local(f, &v)?;
            self.scopes[0].insert(f.clone(), id);
            ids.push(id);
        }
        let r = (|| {
            let pc = self.event_pat(pat, line)?;
            let inner = |cx: &mut Self| -> R<J> {
                let g = match guard {
                    Some(g) => cx.expr_bool(g)?,
                    None => bool_k(true),
                };
                let run = |cx: &mut Self| -> R<J> {
                    cx.out(format!("{sel} = 1'b1;"));
                    let j = match body {
                        Body::Arrow(e) => {
                            let v = cx.expr(e)?;
                            let Some(&id) = ids.first() else { return no("an arrow step on a region without fields", line) };
                            cx.store(id, &v)?;
                            J::default()
                        }
                        Body::Block(b) => cx.block(b, None)?,
                    };
                    // 6.4 step 3: each field checked against its type, in order
                    for (n, (_, t, _)) in fields.iter().enumerate() {
                        let v = cx.locals[ids[n]].v.clone();
                        let v = cx.fit(v, t)?;
                        let r = format!("nx{ri}_{n}");
                        cx.commit(&r, &v, 0);
                    }
                    Ok(j)
                };
                match &g {
                    Val::Bool { k: Some(true), .. } => run(cx),
                    Val::Bool { k: Some(false), .. } => Ok(J::default()),
                    Val::Bool { e, .. } => {
                        let c = format!("!k && {e}");
                        cx.gate(&c, run)
                    }
                    _ => no("a guard that is not a bool", line),
                }
            };
            match &pc {
                Val::Bool { k: Some(true), .. } => inner(self),
                Val::Bool { k: Some(false), .. } => Ok(J::default()),
                Val::Bool { e, .. } => {
                    let e = e.clone();
                    self.gate(&e, inner)
                }
                _ => unreachable!(),
            }
        })();
        self.frames.pop();
        self.scopes = saved;
        r
    }

    /// The next-state register `r` from `v`, at bit `off`.
    fn commit(&mut self, r: &str, v: &Val, off: u32) {
        match v {
            Val::Int { e, tag, .. } => {
                let _ = tag;
                let w = self.val_bits(v);
                self.out(format!("{r}[{}:{off}] = {e};", off + w - 1));
            }
            Val::Bool { e, .. } => self.out(format!("{r}[{off}] = {e};")),
            Val::Kind { kind, e, .. } => {
                let w = self.kind_bits(kind);
                self.out(format!("{r}[{}:{off}] = {e};", off + w - 1));
            }
            Val::Arr(xs) => {
                let mut o = off;
                for x in xs {
                    self.commit(r, x, o);
                    o += self.val_bits(x);
                }
            }
        }
    }

    /// Storage bits of a value already fitted to its declared type.
    fn val_bits(&self, v: &Val) -> u32 {
        match v {
            Val::Int { tag, .. } => int_bits(tag),
            Val::Bool { .. } => 1,
            Val::Kind { kind, .. } => self.kind_bits(kind),
            Val::Arr(xs) => xs.iter().map(|x| self.val_bits(x)).sum(),
        }
    }

    /// The value of type `t` held in `src` from bit `off`.
    fn from_bits(&self, t: &HT, src: &str, off: u32) -> Val {
        match t {
            HT::Int(n) => {
                let b = int_bits(n);
                let s = format!("{src}[{}:{off}]", off + b - 1);
                if n.starts_with('i') {
                    Val::Int { e: format!("$signed({s})"), tag: n, w: b, k: None }
                } else {
                    Val::Int { e: format!("$signed({{1'b0, {s}}})"), tag: n, w: b + 1, k: None }
                }
            }
            HT::Bool => bool_e(format!("{src}[{off}]")),
            HT::Kind(k) => {
                let b = self.kind_bits(k);
                Val::Kind { kind: k.clone(), e: format!("{src}[{}:{off}]", off + b - 1), k: None }
            }
            HT::Arr(e, n) => {
                let b = self.ht_bits(e);
                Val::Arr((0..*n as u32).map(|j| self.from_bits(e, src, off + j * b)).collect())
            }
        }
    }

    /// The condition that the event on the bus matches `p`, binding the
    /// names of the pattern (6.4 step 1).
    fn event_pat(&mut self, p: &Pat, line: usize) -> R<Val> {
        let tag_w = bits_for(self.events.len());
        let ev = |cx: &Self, n: &str| cx.events.iter().position(|e| e.0 == n);
        match p {
            Pat::Wild => Ok(bool_k(true)),
            Pat::Path(x) if x.len() == 1 && ev(self, &x[0]).is_some() => {
                let i = ev(self, &x[0]).unwrap();
                Ok(bool_e(format!("(ev_tag == {tag_w}'d{i})")))
            }
            Pat::Ctor(x, args) if x.len() == 1 && ev(self, &x[0]).is_some() => {
                let i = ev(self, &x[0]).unwrap();
                let fs = self.events[i].1.clone();
                let mut conds = vec![format!("(ev_tag == {tag_w}'d{i})")];
                let mut off = 0;
                let mut offs = vec![];
                for (_, t) in &fs {
                    offs.push(off);
                    off += self.ht_bits(t);
                }
                for (j, (lb, q)) in args.iter().enumerate() {
                    let k = match lb {
                        Some(l) => fs.iter().position(|f| &f.0 == l),
                        None => (j < fs.len()).then_some(j),
                    };
                    let Some(k) = k else { return Ok(bool_k(false)) };
                    let v = self.from_bits(&fs[k].1, "ev_payload", offs[k]);
                    match self.pat_cond(q, &v, line)? {
                        Val::Bool { k: Some(true), .. } => {}
                        Val::Bool { k: Some(false), .. } => return Ok(bool_k(false)),
                        Val::Bool { e, .. } => conds.push(e),
                        _ => unreachable!(),
                    }
                }
                Ok(bool_e(format!("({})", conds.join(" && "))))
            }
            Pat::Alt(ps) => {
                let mut cs = vec![];
                for q in ps {
                    if self.pat_binds(q) {
                        return no("an alternative event pattern that binds a name", line);
                    }
                    match self.event_pat(q, line)? {
                        Val::Bool { k: Some(true), .. } => return Ok(bool_k(true)),
                        Val::Bool { k: Some(false), .. } => {}
                        Val::Bool { e, .. } => cs.push(e),
                        _ => unreachable!(),
                    }
                }
                if cs.is_empty() { Ok(bool_k(false)) } else { Ok(bool_e(format!("({})", cs.join(" || ")))) }
            }
            _ => no(format!("the step pattern `{}`", show_pat(p)), line),
        }
    }

    /// The condition that `v` matches `p` (interp `pmatch`), binding names.
    fn pat_cond(&mut self, p: &Pat, v: &Val, line: usize) -> R<Val> {
        match p {
            Pat::Wild => Ok(bool_k(true)),
            Pat::Path(x) => {
                let n = x.last().unwrap();
                if let Some((kind, i)) = self.variant.get(n).cloned() {
                    return match v {
                        Val::Kind { kind: k2, e, k } if *k2 == kind => Ok(match k {
                            Some(kv) => bool_k(*kv == i),
                            None => bool_e(format!("({e} == {}'d{i})", self.kind_bits(&kind))),
                        }),
                        _ => Ok(bool_k(false)),
                    };
                }
                if x.len() == 1 && !self.events.iter().any(|e| &e.0 == n) && !self.errors.contains(n) {
                    let id = self.new_local(n, v)?;
                    self.bind(n, id);
                    return Ok(bool_k(true));
                }
                Ok(bool_k(false))
            }
            Pat::Lit(e) => {
                let c = self.expr(e)?;
                self.equal(v, &c, line)
            }
            Pat::Range(a, incl, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                let lo = self.cmp(">=", v, &a, line)?;
                let hi = self.cmp(if *incl { "<=" } else { "<" }, v, &b, line)?;
                self.and(&lo, &hi)
            }
            Pat::Alt(ps) => {
                let mut acc = bool_k(false);
                for q in ps {
                    if self.pat_binds(q) {
                        return no("an alternative pattern that binds a name", line);
                    }
                    let c = self.pat_cond(q, v, line)?;
                    acc = self.or(&acc, &c)?;
                }
                Ok(acc)
            }
            _ => no(format!("the pattern `{}`", show_pat(p)), line),
        }
    }

    fn and(&mut self, a: &Val, b: &Val) -> R<Val> {
        Ok(match (a, b) {
            (Val::Bool { k: Some(false), .. }, _) | (_, Val::Bool { k: Some(false), .. }) => bool_k(false),
            (Val::Bool { k: Some(true), .. }, x) | (x, Val::Bool { k: Some(true), .. }) => x.clone(),
            (Val::Bool { e: x, .. }, Val::Bool { e: y, .. }) => bool_e(format!("({x} && {y})")),
            _ => return no("`and` of values that are not bools", self.line),
        })
    }

    fn or(&mut self, a: &Val, b: &Val) -> R<Val> {
        Ok(match (a, b) {
            (Val::Bool { k: Some(true), .. }, _) | (_, Val::Bool { k: Some(true), .. }) => bool_k(true),
            (Val::Bool { k: Some(false), .. }, x) | (x, Val::Bool { k: Some(false), .. }) => x.clone(),
            (Val::Bool { e: x, .. }, Val::Bool { e: y, .. }) => bool_e(format!("({x} || {y})")),
            _ => return no("`or` of values that are not bools", self.line),
        })
    }

    // ------------------------------------------------------- statements

    fn count(&mut self) -> R<()> {
        if self.body.len() > CEILING {
            return no(format!("the step logic unrolls to more than {CEILING} statements (the size ceiling of EV-4)"), self.line);
        }
        Ok(())
    }

    /// A block in its own scope.
    fn block(&mut self, b: &[Stmt], tail: Option<&Tail>) -> R<J> {
        self.scopes.push(HashMap::new());
        let r = self.stmts(b, tail);
        self.scopes.pop();
        r
    }

    fn stmts(&mut self, b: &[Stmt], tail: Option<&Tail>) -> R<J> {
        let mut j = J::default();
        for (i, s) in b.iter().enumerate() {
            if j.must {
                break;
            }
            let t = if i + 1 == b.len() { tail } else { None };
            let js = if j.may {
                let live = self.live();
                self.gate(&live, |cx| cx.stmt(s, t))?
            } else {
                self.stmt(s, t)?
            };
            j = J { may: j.may || js.may, must: js.must };
        }
        Ok(j)
    }

    fn stmt(&mut self, s: &Stmt, tail: Option<&Tail>) -> R<J> {
        self.count()?;
        match s {
            Stmt::Expr(e) => {
                self.line = e.line();
                let v = self.expr(e)?;
                if let Some(t) = tail {
                    self.set_result(t, v)?;
                }
                Ok(J::default())
            }
            Stmt::Pass => Ok(J::default()),
            Stmt::Let(p, ty, e, line) => {
                self.line = *line;
                let v = self.expr(e)?;
                let v = match ty {
                    Some(t) => {
                        let h = self.ht(t, *line)?;
                        self.fit(v, &h)?
                    }
                    None => v,
                };
                match p {
                    Pat::Path(n) if n.len() == 1 && !self.variant.contains_key(&n[0]) => {
                        let id = self.new_local(&n[0], &v)?;
                        self.bind(&n[0], id);
                    }
                    Pat::Wild => {}
                    _ => return no(format!("the `let` pattern `{}`", show_pat(p)), *line),
                }
                Ok(J::default())
            }
            Stmt::Var(n, ty, e, line) => {
                self.line = *line;
                let v = self.expr(e)?;
                let v = match ty {
                    Some(t) => {
                        let h = self.ht(t, *line)?;
                        self.fit(v, &h)?
                    }
                    None => v,
                };
                let id = self.new_local(n, &v)?;
                self.bind(n, id);
                Ok(J::default())
            }
            Stmt::Assign(op, place, rhs) => {
                self.line = place.line();
                self.assign(op, place, rhs)?;
                Ok(J::default())
            }
            Stmt::If(arms, els) => self.if_chain(arms, els.as_deref(), tail),
            Stmt::While(c, b, body) => {
                self.line = c.line();
                self.lp(Some(c), b.as_ref(), body)
            }
            Stmt::Loop(b, body) => self.lp(None, b.as_ref(), body),
            Stmt::For(p, it, body, line) => {
                self.line = *line;
                self.for_(p, it, body)
            }
            Stmt::Match(e, arms, line) => {
                self.line = *line;
                self.match_(e, arms, tail)
            }
            Stmt::Return(e, line) => {
                self.line = *line;
                let Some(f) = self.frames.last() else { return no("`return`", *line) };
                let Some(ret) = f.ret.clone() else { return no("`return` in a step (a step ends at its last statement)", *line) };
                let res = f.res.clone();
                let v = match e {
                    Some(e) => Some(self.expr(e)?),
                    None => None,
                };
                if let (Some((id, ty)), Some(v)) = (res, v) {
                    let v = self.fit(v, &ty)?;
                    self.store(id, &v)?;
                }
                self.out(format!("{ret} = 1'b1;"));
                Ok(J { may: true, must: true })
            }
            Stmt::Break(line) | Stmt::Continue(line) => {
                let Some((b, c)) = self.frames.last().and_then(|f| f.loops.last().cloned()) else {
                    return no("`break` or `continue` outside a loop", *line);
                };
                let f = if matches!(s, Stmt::Break(_)) { b } else { c };
                self.out(format!("{f} = 1'b1;"));
                Ok(J { may: true, must: true })
            }
            Stmt::Emit(e) => {
                self.line = e.line();
                if self.frames.last().is_some_and(|f| f.ret.is_some()) {
                    return no("`emit` inside a function (EV-3: a step's outputs are emitted by the step)", e.line());
                }
                let v = self.expr(e)?;
                self.emit_val(&v)?;
                Ok(J::default())
            }
            Stmt::Fail(e) => {
                self.line = e.line();
                let (n, args): (String, &[Arg]) = match e {
                    Expr::Name(n, _) => (n.clone(), &[]),
                    Expr::Call(f, a, _, _) if matches!(&**f, Expr::Name(..)) => match &**f {
                        Expr::Name(n, _) => (n.clone(), a.as_slice()),
                        _ => unreachable!(),
                    },
                    _ => return no("a `fail` of a value that is not an error constructor", e.line()),
                };
                // interp `construct`: each field evaluated and fitted in
                // order; only the error's name is kept in D
                let decl: Vec<(String, Ty)> = self
                    .m
                    .items
                    .iter()
                    .find_map(|it| match it {
                        Item::Error { name, payload, .. } if *name == n => Some(payload.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                if args.len() != decl.len() {
                    return no(format!("`fail {n}` without every field of the error"), e.line());
                }
                for (i, a) in args.iter().enumerate() {
                    let k = match &a.label {
                        Some(l) => decl.iter().position(|f| &f.0 == l).unwrap_or(i),
                        None => i,
                    };
                    let v = self.expr(&a.e)?;
                    let t = self.ht(&decl[k].1, e.line())?;
                    self.fit(v, &t)?;
                }
                self.fail(&n, e.line())?;
                Ok(J { may: false, must: true })
            }
            other => no(format!("the statement `{}`", stmt_word(other)), self.line),
        }
    }

    fn fail(&mut self, n: &str, line: usize) -> R<()> {
        let Some(i) = self.errors.iter().position(|x| x == n) else { return no(format!("`fail {n}`, which is not a declared error"), line) };
        self.out(format!("if (!k) begin k = 1'b1; kerr = {}; end", i + 1));
        Ok(())
    }

    fn set_result(&mut self, t: &Tail, v: Val) -> R<()> {
        let v = self.fit(v, &t.ty)?;
        self.store(t.res, &v)
    }

    fn if_chain(&mut self, arms: &[(Expr, Vec<Stmt>)], els: Option<&[Stmt]>, tail: Option<&Tail>) -> R<J> {
        let (c, b) = &arms[0];
        self.line = c.line();
        let cv = self.expr_bool(c)?;
        let rest = |cx: &mut Self| -> R<J> {
            if arms.len() > 1 {
                cx.if_chain(&arms[1..], els, tail)
            } else if let Some(e) = els {
                cx.block(e, tail)
            } else {
                Ok(J::default())
            }
        };
        match cv {
            Val::Bool { k: Some(true), .. } => self.block(b, tail),
            Val::Bool { k: Some(false), .. } => rest(self),
            Val::Bool { e, .. } => {
                let has_else = arms.len() > 1 || els.is_some();
                if has_else {
                    let mut r = rest;
                    self.branch(&e, |cx| cx.block(b, tail), Some(&mut r))
                } else {
                    self.gate(&e, |cx| cx.block(b, tail))
                }
            }
            _ => no("an `if` condition that is not a bool", c.line()),
        }
    }

    fn const_int(&mut self, e: &Expr, what: &str) -> R<i128> {
        match self.expr(e)? {
            Val::Int { k: Some(v), .. } => Ok(v),
            _ => no(format!("{what} that is not a constant (EV-4)"), e.line()),
        }
    }

    /// `while c bound n` and `loop bound n` (6.7), unrolled: n iterations,
    /// and the trap "loop bound" when the condition still holds after them.
    fn lp(&mut self, c: Option<&Expr>, bound: Option<&Expr>, body: &[Stmt]) -> R<J> {
        let line = self.line;
        let Some(bound) = bound else { return no("a loop without a constant `bound` (EV-4)", line) };
        let n = self.const_int(bound, "a loop bound")?;
        if !(0..=CEILING as i128).contains(&n) {
            return no(format!("a loop bound of {n}, above the size ceiling of {CEILING} (EV-4)"), line);
        }
        let brk = self.reg("b", 1, false, "break");
        let cont = self.reg("c", 1, false, "continue");
        self.out(format!("{brk} = 1'b0;"));
        self.frames.last_mut().expect("a frame").loops.push((brk.clone(), cont.clone()));
        let mut j = J::default();
        let r = (|| {
            for it in 1..=n + 1 {
                let one = |cx: &mut Self| -> R<J> {
                    let cv = match c {
                        Some(c) => cx.expr_bool(c)?,
                        None => bool_k(true),
                    };
                    let take = |cx: &mut Self| -> R<J> {
                        if it == n + 1 {
                            cx.trap_text(None, "loop bound");
                            return Ok(J { may: false, must: true });
                        }
                        cx.out(format!("{cont} = 1'b0;"));
                        cx.block(body, None)
                    };
                    match cv {
                        Val::Bool { k: Some(true), .. } => take(cx),
                        Val::Bool { k: Some(false), .. } => {
                            cx.out(format!("{brk} = 1'b1;"));
                            Ok(J { may: false, must: false })
                        }
                        Val::Bool { e, .. } => {
                            let mut stop = |cx: &mut Self| -> R<J> {
                                cx.out(format!("{brk} = 1'b1;"));
                                Ok(J::default())
                            };
                            cx.branch(&e, take, Some(&mut stop))
                        }
                        _ => no("a loop condition that is not a bool", line),
                    }
                };
                let ji = if it == 1 {
                    one(self)?
                } else {
                    let ret = self.frames.last().and_then(|f| f.ret.clone());
                    let g = match ret {
                        Some(r) => format!("!k && !{r} && !{brk}"),
                        None => format!("!k && !{brk}"),
                    };
                    self.gate(&g, one)?
                };
                j.may |= ji.may;
                if ji.must && !ji.may {
                    break; // the loop bound trap, or a `fail` every iteration reaches
                }
                if c.is_some_and(|c| matches!(self.peek_bool(c), Some(false))) {
                    break;
                }
                self.count()?;
            }
            Ok(())
        })();
        self.frames.last_mut().expect("a frame").loops.pop();
        r?;
        // only a `return` inside the body leaves the enclosing block
        let ret_may = j.may && self.frames.last().is_some_and(|f| f.ret.is_some());
        Ok(J { may: ret_may, must: false })
    }

    /// A loop condition that is a constant now (no code).
    fn peek_bool(&self, c: &Expr) -> Option<bool> {
        match c {
            Expr::Bool(b, _) => Some(*b),
            _ => None,
        }
    }

    /// `for x in a..b` with constant bounds, or `for x in ARRAY`, unrolled.
    fn for_(&mut self, p: &Pat, it: &Expr, body: &[Stmt]) -> R<J> {
        let line = self.line;
        let items: Vec<Val> = match it {
            Expr::Bin(op, a, b, _) if op == ".." || op == "..=" => {
                let (av, bv) = (self.expr(a)?, self.expr(b)?);
                let (Val::Int { k: Some(x), tag: s, .. }, Val::Int { k: Some(y), tag: t, .. }) = (&av, &bv) else {
                    return no("a `for` range whose bounds are not constants (EV-4)", line);
                };
                let hi = if op == ".." { *y } else { *y + 1 };
                if hi - x > CEILING as i128 {
                    return no(format!("a `for` range of {} items, above the size ceiling of {CEILING} (EV-4)", hi - x), line);
                }
                let tag = join_tag(s, t);
                (*x..hi.max(*x)).map(|i| int_k(i, tag)).collect()
            }
            _ => match self.expr(it)? {
                Val::Arr(xs) => {
                    let mut out = vec![];
                    for x in xs {
                        out.push(self.hold(&x)?);
                    }
                    out
                }
                v => return no(format!("a `for` over {}: the subset iterates constant ranges and arrays (EV-4)", v.shape()), line),
            },
        };
        let brk = self.reg("b", 1, false, "break");
        let cont = self.reg("c", 1, false, "continue");
        self.out(format!("{brk} = 1'b0;"));
        self.frames.last_mut().expect("a frame").loops.push((brk.clone(), cont.clone()));
        let mut j = J::default();
        let r = (|| {
            for (n, x) in items.iter().enumerate() {
                let one = |cx: &mut Self| -> R<J> {
                    cx.scopes.push(HashMap::new());
                    let r = (|| {
                        match p {
                            Pat::Path(v) if v.len() == 1 => {
                                let id = cx.new_local(&v[0], x)?;
                                cx.bind(&v[0], id);
                            }
                            Pat::Wild => {}
                            _ => return no(format!("the `for` pattern `{}`", show_pat(p)), line),
                        }
                        cx.out(format!("{cont} = 1'b0;"));
                        cx.block(body, None)
                    })();
                    cx.scopes.pop();
                    r
                };
                let ji = if n == 0 || !j.may {
                    one(self)?
                } else {
                    let ret = self.frames.last().and_then(|f| f.ret.clone());
                    let g = match ret {
                        Some(r) => format!("!k && !{r} && !{brk}"),
                        None => format!("!k && !{brk}"),
                    };
                    self.gate(&g, one)?
                };
                j.may |= ji.may;
                if ji.must && !ji.may {
                    break;
                }
                self.count()?;
            }
            Ok(())
        })();
        self.frames.last_mut().expect("a frame").loops.pop();
        r?;
        let ret_may = j.may && self.frames.last().is_some_and(|f| f.ret.is_some());
        Ok(J { may: ret_may, must: false })
    }

    fn match_(&mut self, e: &Expr, arms: &[Arm], tail: Option<&Tail>) -> R<J> {
        let v = self.expr(e)?;
        let v = self.hold(&v)?;
        let m = self.reg("m", 1, false, "a match arm was taken");
        self.out(format!("{m} = 1'b0;"));
        let mut j: Option<J> = None;
        let mut all = false;
        for arm in arms {
            self.line = arm.line;
            let one = |cx: &mut Self| -> R<J> {
                cx.scopes.push(HashMap::new());
                let r = (|| {
                    let pc = cx.pat_cond(&arm.pat, &v, arm.line)?;
                    let inner = |cx: &mut Self| -> R<J> {
                        let g = match &arm.guard {
                            Some(g) => cx.expr_bool(g)?,
                            None => bool_k(true),
                        };
                        let run = |cx: &mut Self| -> R<J> {
                            cx.out(format!("{m} = 1'b1;"));
                            cx.block(&arm.body, tail)
                        };
                        match &g {
                            Val::Bool { k: Some(true), .. } => run(cx),
                            Val::Bool { k: Some(false), .. } => Ok(J::default()),
                            Val::Bool { e, .. } => {
                                let c = format!("!k && {e}");
                                cx.gate(&c, run)
                            }
                            _ => no("a match guard that is not a bool", arm.line),
                        }
                    };
                    match &pc {
                        Val::Bool { k: Some(true), .. } => inner(cx),
                        Val::Bool { k: Some(false), .. } => Ok(J::default()),
                        Val::Bool { e, .. } => {
                            let e = e.clone();
                            cx.gate(&e, inner)
                        }
                        _ => unreachable!(),
                    }
                })();
                cx.scopes.pop();
                r
            };
            let g = format!("!k && !{m}");
            let ja = self.gate(&g, one)?;
            j = Some(match j {
                None => ja,
                Some(x) => J { may: x.may || ja.may, must: x.must && ja.must },
            });
            // an arm that always matches ends the match
            if matches!(arm.pat, Pat::Wild) && arm.guard.is_none() {
                all = true;
                break;
            }
            if let Pat::Path(x) = &arm.pat
                && x.len() == 1
                && !self.variant.contains_key(&x[0])
                && arm.guard.is_none()
            {
                all = true;
                break;
            }
        }
        if !all {
            let g = format!("!{m}");
            self.gate(&g, |cx| {
                cx.trap_text(None, "no match arm");
                Ok(J { may: false, must: true })
            })?;
        }
        Ok(J { may: j.is_some_and(|x| x.may), must: false })
    }

    fn emit_val(&mut self, v: &Val) -> R<()> {
        let (t, w) = match v {
            Val::Int { w, .. } => (EmitTy::Int, *w),
            Val::Bool { .. } => (EmitTy::Bool, 1),
            Val::Kind { kind, .. } => (EmitTy::Kind(kind.clone()), self.kind_bits(kind)),
            Val::Arr(_) => return no("`emit` of an array (EV-3: emitted values are scalars)", self.line),
        };
        let code = match self.emit_tys.iter().position(|x| *x == t) {
            Some(i) => i,
            None => {
                self.emit_tys.push(t);
                self.emit_tys.len() - 1
            }
        };
        self.emit_w = self.emit_w.max(w);
        let (lo, hi) = self.emits;
        if hi + 1 > MAX_EMITS {
            return no(format!("a step that can emit more than {MAX_EMITS} values (EV-3)"), self.line);
        }
        let e = match v {
            Val::Int { e, .. } | Val::Bool { e, .. } | Val::Kind { e, .. } => e.clone(),
            Val::Arr(_) => unreachable!(),
        };
        for s in lo..=hi {
            let set = format!("et{s} = {code}; ev{s} = {e};");
            if lo == hi {
                self.out(set);
            } else {
                self.out(format!("if (ecnt == {s}) begin {set} end"));
            }
        }
        self.out("ecnt = ecnt + 1'b1;");
        self.emits = (lo + 1, hi + 1);
        self.max_emits = self.max_emits.max(hi + 1);
        Ok(())
    }

    // -------------------------------------------------------- assignment

    fn assign(&mut self, op: &str, place: &Expr, rhs: &Expr) -> R<()> {
        let line = place.line();
        let r = self.expr(rhs)?;
        match place {
            Expr::Name(n, _) => {
                let Some(id) = self.lookup(n) else { return no(format!("an assignment to `{n}`, which is not a local or a field"), line) };
                let mut v = r;
                if op != "=" {
                    let cur = self.locals[id].v.clone();
                    let Val::Int { tag: t, .. } = cur else { return no(format!("`{op}` on {}", cur.shape()), line) };
                    let x = self.arith(&op[..op.len() - 1], &cur, &v)?;
                    v = self.chk(x, t)?;
                }
                if let (Val::Int { tag: t, .. }, Val::Int { .. }) = (&self.locals[id].v, &v)
                    && *t != "lit"
                {
                    let t = *t;
                    v = self.chk(v, t)?;
                }
                self.store(id, &v)
            }
            Expr::Index(a, idx, _) if idx.len() == 1 => {
                let Expr::Name(n, _) = &**a else { return no("an assignment to a nested place", line) };
                let Some(id) = self.lookup(n) else { return no(format!("an assignment to `{n}`, which is not a local or a field"), line) };
                let Val::Arr(elems) = self.locals[id].v.clone() else { return no(format!("an index into `{n}`, which is not an array"), line) };
                let i = self.expr(&idx[0])?;
                let Val::Int { .. } = i else { return no("an index that is not an integer", line) };
                let i = self.hold(&i)?;
                let mut v = r;
                if op != "=" {
                    let cur = self.read_index(&elems, &i)?;
                    let Val::Int { tag: t, .. } = cur else { return no(format!("`{op}` on {}", cur.shape()), line) };
                    let x = self.arith(&op[..op.len() - 1], &cur, &v)?;
                    v = self.chk(x, t)?;
                } else {
                    self.bounds(&elems, &i);
                }
                // the element's type name, which the stored value takes
                let tags: Vec<&'static str> = elems.iter().filter_map(|x| if let Val::Int { tag, .. } = x { Some(*tag) } else { None }).collect();
                if let (Val::Int { .. }, Some(t)) = (&v, tags.first())
                    && tags.iter().all(|x| x == t)
                    && *t != "lit"
                {
                    let t = *t;
                    v = self.chk(v, t)?;
                } else if matches!(v, Val::Int { .. }) && !tags.iter().all(|x| *x == "lit") {
                    return no(format!("an indexed assignment to `{n}`, whose elements have different integer types"), line);
                }
                let Val::Int { k: ik, e: ie, .. } = &i else { unreachable!() };
                let mut out = elems.clone();
                match ik {
                    Some(kv) => {
                        if (0..elems.len() as i128).contains(kv) {
                            let j = *kv as usize;
                            out[j] = self.store_val(&elems[j], &v, n)?;
                        }
                    }
                    None => {
                        for (j, el) in elems.iter().enumerate() {
                            let (jl, _) = lit(j as i128);
                            let c = format!("{ie} == {jl}");
                            let mut nv = None;
                            let el2 = el.clone();
                            self.gate(&c, |cx| {
                                nv = Some(cx.store_val(&el2, &v, n)?);
                                Ok(J::default())
                            })?;
                            let nv = nv.unwrap();
                            match Self::merge_val(&nv, el) {
                                Some(x) => out[j] = x,
                                None => return no(format!("an indexed assignment to `{n}` that changes an element's integer type"), line),
                            }
                        }
                    }
                }
                self.locals[id].v = Val::Arr(out);
                Ok(())
            }
            _ => no("an assignment to a place other than a local, a field or an array element", line),
        }
    }

    /// Trap "index K out of range" unless `i` indexes `elems`.
    fn bounds(&mut self, elems: &[Val], i: &Val) {
        let Val::Int { e, w, k, .. } = i else { return };
        let n = elems.len() as i128;
        match k {
            Some(kv) => {
                if !(0..n).contains(kv) {
                    let (l, lw) = lit(*kv);
                    self.trap(None, Trap::Index, Some((&l, lw)));
                }
            }
            None => {
                let (nl, _) = lit(n);
                let c = format!("{e} < 2'sd0 || {e} >= {nl}");
                let (e, w) = (e.clone(), *w);
                self.trap(Some(&c), Trap::Index, Some((&e, w)));
            }
        }
    }

    fn read_index(&mut self, elems: &[Val], i: &Val) -> R<Val> {
        self.bounds(elems, i);
        let Val::Int { e, k, .. } = i else { return no("an index that is not an integer", self.line) };
        if let Some(kv) = k {
            return Ok(if (0..elems.len() as i128).contains(kv) {
                elems[*kv as usize].clone()
            } else {
                elems.first().cloned().unwrap_or(int_k(0, "lit"))
            });
        }
        if elems.is_empty() {
            return Ok(int_k(0, "lit"));
        }
        let mut shape = elems[0].clone();
        for x in &elems[1..] {
            shape = match Self::merge_val(&shape, x) {
                Some(s) => s,
                None => return no("an index into an array whose elements have different types", self.line),
            };
        }
        let r = self.fresh("indexed value", &shape);
        let e = e.clone();
        let mut out = r.clone();
        for (j, x) in elems.iter().enumerate() {
            let (jl, _) = lit(j as i128);
            if j == 0 {
                out = self.store_val(&r, x, "indexed value")?;
            } else {
                self.out(format!("if ({e} == {jl}) begin"));
                self.ind += 1;
                self.store_val(&r, x, "indexed value")?;
                self.ind -= 1;
                self.out("end");
            }
        }
        // the shape's widths cover every element
        Ok(match (out, shape) {
            (Val::Int { e, tag, .. }, Val::Int { w, .. }) => Val::Int { e, tag, w, k: None },
            (o, _) => o,
        })
    }

    // -------------------------------------------------------- expressions

    fn expr_bool(&mut self, e: &Expr) -> R<Val> {
        match self.expr(e)? {
            v @ Val::Bool { .. } => Ok(v),
            v => no(format!("{} where a bool is needed", v.shape()), e.line()),
        }
    }

    fn expr(&mut self, e: &Expr) -> R<Val> {
        self.line = e.line();
        let line = e.line();
        match e {
            Expr::Int(v, _) => Ok(int_k(*v, "lit")),
            Expr::Bool(b, _) => Ok(bool_k(*b)),
            Expr::Name(n, _) => self.name(n, line),
            Expr::Neg(x, _) => {
                let v = self.expr(x)?;
                let Val::Int { e, tag, w, k } = v else { return no(format!("`-` on {}", v.shape()), line) };
                if let Some(kv) = k {
                    return self.fold(self.it.arith("-", V::Int(0, "lit"), V::Int(kv, tag)).map(|r| match r {
                        V::Int(r, _) => V::Int(r, tag),
                        o => o,
                    }));
                }
                let t = self.reg("t", w + 1, true, "");
                self.out(format!("{t} = -{e};"));
                self.check_range(Val::Int { e: t, tag, w: w + 1, k: None }, tag)
            }
            Expr::BitNot(x, _) => {
                let v = self.expr(x)?;
                let Val::Int { e, tag, w, k } = v else { return no(format!("`~` on {}", v.shape()), line) };
                let (lo, hi) = range(tag);
                if let Some(kv) = k {
                    return Ok(int_k(if lo == 0 { hi - kv } else { !kv }, tag));
                }
                if lo == 0 {
                    let (h, hw) = lit(hi);
                    let t = self.reg("t", hw.max(w) + 1, true, "");
                    self.out(format!("{t} = {h} - {e};"));
                    Ok(Val::Int { e: t, tag, w: carrier(tag), k: None })
                } else {
                    let t = self.reg("t", w, true, "");
                    self.out(format!("{t} = ~{e};"));
                    Ok(Val::Int { e: t, tag, w, k: None })
                }
            }
            Expr::Not(x, _) => {
                let v = self.expr(x)?;
                match v {
                    Val::Bool { k: Some(b), .. } => Ok(bool_k(!b)),
                    Val::Bool { e, .. } => Ok(bool_e(format!("(!{e})"))),
                    v => no(format!("`not` on {}", v.shape()), line),
                }
            }
            Expr::Bin(op, a, b, _) => self.bin(op, a, b, line),
            Expr::If(c, a, b, _) => {
                let cv = self.expr_bool(c)?;
                match cv {
                    Val::Bool { k: Some(true), .. } => self.expr(a),
                    Val::Bool { k: Some(false), .. } => self.expr(b),
                    Val::Bool { e: ce, .. } => {
                        let cr = self.reg("q", 1, false, "if condition");
                        self.out(format!("{cr} = {ce};"));
                        let mut va = None;
                        let mut vb = None;
                        let mut fb = |cx: &mut Self| -> R<J> {
                            vb = Some(cx.expr(b)?);
                            Ok(J::default())
                        };
                        self.branch(
                            &cr,
                            |cx| {
                                va = Some(cx.expr(a)?);
                                Ok(J::default())
                            },
                            Some(&mut fb),
                        )?;
                        self.select(&cr, &va.unwrap(), &vb.unwrap(), line)
                    }
                    _ => unreachable!(),
                }
            }
            Expr::Cast(x, ty, _) => {
                let v = self.expr(x)?;
                let h = self.ht(ty, line)?;
                let HT::Int(n) = h else { return no(format!("a cast to `{}`", crate::interp::ty_str(ty)), line) };
                match v {
                    Val::Int { .. } => self.chk(v, n),
                    Val::Bool { e, k } => Ok(match k {
                        Some(b) => int_k(b as i128, n),
                        None => Val::Int { e: format!("$signed({{1'b0, {e}}})"), tag: n, w: 2, k: None },
                    }),
                    v => no(format!("a cast of {} to an integer", v.shape()), line),
                }
            }
            Expr::Index(a, idx, _) if idx.len() == 1 => {
                let av = self.expr(a)?;
                let Val::Arr(elems) = av else { return no(format!("an index into {}", av.shape()), line) };
                let i = self.expr(&idx[0])?;
                if !matches!(i, Val::Int { .. }) {
                    return no("an index that is not an integer", line);
                }
                self.read_index(&elems, &i)
            }
            Expr::List(xs, _) => {
                let mut out = vec![];
                for x in xs {
                    out.push(self.expr(x)?);
                }
                Ok(Val::Arr(out))
            }
            Expr::ListRepeat(x, n, _) => {
                let v = self.expr(x)?;
                let n = self.const_int(n, "a repeat count")?;
                if !(0..=4096).contains(&n) {
                    return no("a repeat count above 4096", line);
                }
                let v = self.hold(&v)?;
                Ok(Val::Arr(vec![v; n as usize]))
            }
            Expr::Call(f, args, _, _) => self.call(f, args, line),
            Expr::Try(x, Some(els), _) => self.try_(x, els, line),
            Expr::Comptime(x, _) => self.expr(x),
            Expr::Float(..) => no("a float (EV-1: the circuit holds integers, bool, kinds and arrays)", line),
            Expr::Str(..) => no("text (EV-1: no dynamic text)", line),
            Expr::Map(..) => no("a map (EV-1)", line),
            _ => no("this expression (EV-1)", line),
        }
    }

    fn name(&mut self, n: &str, line: usize) -> R<Val> {
        if let Some(id) = self.lookup(n) {
            return Ok(self.locals[id].v.clone());
        }
        if let Some((kind, i)) = self.variant.get(n).cloned() {
            let b = self.kind_bits(&kind);
            return Ok(Val::Kind { kind, e: format!("{b}'d{i}"), k: Some(i) });
        }
        if self.consts.contains(n) {
            let v = match self.it.const_value(n) {
                Ok(v) => v,
                Err(f) => return no(format!("the constant `{n}` ({})", crate::interp::why(f)), line),
            };
            return self.val_of(&v, line);
        }
        no(format!("the name `{n}` as a value"), line)
    }

    fn val_of(&self, v: &V, line: usize) -> R<Val> {
        Ok(match v {
            V::Int(x, t) => int_k(*x, t),
            V::Bool(b) => bool_k(*b),
            V::Rec(n, fs) if fs.is_empty() && self.variant.contains_key(n) => {
                let (kind, i) = self.variant[n].clone();
                let b = self.kind_bits(&kind);
                Val::Kind { kind, e: format!("{b}'d{i}"), k: Some(i) }
            }
            V::List(xs) => Val::Arr(xs.iter().map(|x| self.val_of(x, line)).collect::<R<_>>()?),
            o => return no(format!("the constant value {} (EV-1)", o.show()), line),
        })
    }

    /// A constant result of the reference evaluator's arithmetic: the value,
    /// or its trap.
    fn fold(&mut self, r: Result<V, Flow>) -> R<Val> {
        match r {
            Ok(v) => self.val_of(&v, self.line),
            Err(Flow::Trap(t)) => {
                self.trap_text(None, &t);
                Ok(int_k(0, "lit"))
            }
            Err(f) => no(format!("a constant that does not evaluate ({})", crate::interp::why(f)), self.line),
        }
    }

    /// `c ? a : b` of two values of one shape.
    fn select(&mut self, c: &str, a: &Val, b: &Val, line: usize) -> R<Val> {
        Ok(match (a, b) {
            (Val::Int { e: x, tag, w, .. }, Val::Int { e: y, tag: t2, w: w2, .. }) if tag == t2 => {
                Val::Int { e: format!("({c} ? {x} : {y})"), tag, w: (*w).max(*w2), k: None }
            }
            (Val::Bool { e: x, .. }, Val::Bool { e: y, .. }) => bool_e(format!("({c} ? {x} : {y})")),
            (Val::Kind { kind, e: x, .. }, Val::Kind { kind: k2, e: y, .. }) if kind == k2 => Val::Kind { kind: kind.clone(), e: format!("({c} ? {x} : {y})"), k: None },
            (Val::Arr(xs), Val::Arr(ys)) if xs.len() == ys.len() => Val::Arr(xs.iter().zip(ys).map(|(x, y)| self.select(c, x, y, line)).collect::<R<_>>()?),
            _ => return no(format!("an `if` expression that is {} or {}, so its type is not static (EV-2)", a.shape(), b.shape()), line),
        })
    }

    fn bin(&mut self, op: &str, a: &Expr, b: &Expr, line: usize) -> R<Val> {
        if op == "and" || op == "or" {
            let av = self.expr_bool(a)?;
            let short = op == "or";
            return match av {
                Val::Bool { k: Some(x), .. } if x == short => Ok(bool_k(short)),
                Val::Bool { k: Some(_), .. } => self.expr_bool(b),
                Val::Bool { e, .. } => {
                    let r = self.reg("q", 1, false, op);
                    self.out(format!("{r} = {e};"));
                    let c = if short { format!("!{r}") } else { r.clone() };
                    self.gate(&c, |cx| {
                        let bv = cx.expr_bool(b)?;
                        let Val::Bool { e, .. } = bv else { unreachable!() };
                        cx.out(format!("{r} = {e};"));
                        Ok(J::default())
                    })?;
                    Ok(bool_e(r))
                }
                _ => unreachable!(),
            };
        }
        let av = self.expr(a)?;
        let av = if expr_may_store(b) { self.hold(&av)? } else { av };
        let bv = self.expr(b)?;
        self.line = line;
        match op {
            "==" => self.equal(&av, &bv, line),
            "!=" => {
                let v = self.equal(&av, &bv, line)?;
                Ok(match v {
                    Val::Bool { k: Some(x), .. } => bool_k(!x),
                    Val::Bool { e, .. } => bool_e(format!("(!{e})")),
                    _ => unreachable!(),
                })
            }
            "<" | "<=" | ">" | ">=" => self.cmp(op, &av, &bv, line),
            "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" => self.arith(op, &av, &bv),
            _ => no(format!("the operator `{op}`"), line),
        }
    }

    /// `==` as the reference evaluator compares values (integers by value,
    /// whatever their type names).
    fn equal(&mut self, a: &Val, b: &Val, line: usize) -> R<Val> {
        Ok(match (a, b) {
            (Val::Int { k: Some(x), .. }, Val::Int { k: Some(y), .. }) => bool_k(x == y),
            (Val::Int { e: x, .. }, Val::Int { e: y, .. }) => bool_e(format!("({x} == {y})")),
            (Val::Bool { k: Some(x), .. }, Val::Bool { k: Some(y), .. }) => bool_k(x == y),
            (Val::Bool { e: x, .. }, Val::Bool { e: y, .. }) => bool_e(format!("({x} == {y})")),
            (Val::Kind { kind, k: Some(x), .. }, Val::Kind { kind: k2, k: Some(y), .. }) if kind == k2 => bool_k(x == y),
            (Val::Kind { kind, e: x, .. }, Val::Kind { kind: k2, e: y, .. }) if kind == k2 => bool_e(format!("({x} == {y})")),
            (Val::Kind { .. }, Val::Kind { .. }) => bool_k(false),
            (Val::Arr(xs), Val::Arr(ys)) => {
                if xs.len() != ys.len() {
                    return Ok(bool_k(false));
                }
                let mut acc = bool_k(true);
                for (x, y) in xs.iter().zip(ys) {
                    let c = self.equal(x, y, line)?;
                    acc = self.and(&acc, &c)?;
                }
                acc
            }
            _ => return no(format!("`==` of {} and {}", a.shape(), b.shape()), line),
        })
    }

    fn cmp(&mut self, op: &str, a: &Val, b: &Val, line: usize) -> R<Val> {
        Ok(match (a, b) {
            (Val::Int { k: Some(x), .. }, Val::Int { k: Some(y), .. }) => bool_k(match op {
                "<" => x < y,
                "<=" => x <= y,
                ">" => x > y,
                _ => x >= y,
            }),
            (Val::Int { e: x, .. }, Val::Int { e: y, .. }) | (Val::Bool { e: x, .. }, Val::Bool { e: y, .. }) => bool_e(format!("({x} {op} {y})")),
            _ => return no(format!("`{op}` on {} and {}", a.shape(), b.shape()), line),
        })
    }

    /// `v` with type name `t` (interp `chk`): checked against t's range
    /// unless it already carries t.
    fn chk(&mut self, v: Val, t: &'static str) -> R<Val> {
        if let Val::Int { tag, .. } = &v
            && *tag == t
            && t != "lit"
        {
            return Ok(v);
        }
        self.check_range(v, t)
    }

    /// `v` checked against the range of `t` (6.3 N2): out of range keeps
    /// the state with "overflow: V is outside T", V the exact value.
    fn check_range(&mut self, v: Val, t: &'static str) -> R<Val> {
        let Val::Int { e, w, k, .. } = v else { return Ok(v) };
        let ct = if t == "lit" { "i64" } else { t };
        let (lo, hi) = range(ct);
        if let Some(x) = k {
            if x < lo || x > hi {
                let (l, lw) = lit(x);
                self.trap(None, Trap::Overflow(ct), Some((&l, lw)));
            }
            return Ok(int_k(x, t));
        }
        let (l, _) = lit(lo);
        let (h, _) = lit(hi);
        let c = format!("{e} < {l} || {e} > {h}");
        self.trap(Some(&c), Trap::Overflow(ct), Some((&e, w)));
        Ok(Val::Int { e, tag: t, w: w.min(carrier(ct)), k: None })
    }

    /// `a OP b` on integers (interp `arith`), exact and then checked.
    fn arith(&mut self, op: &str, a: &Val, b: &Val) -> R<Val> {
        let line = self.line;
        let (Val::Int { e: x, tag: ta, w: wa, k: ka }, Val::Int { e: y, tag: tb, w: wb, k: kb }) = (a, b) else {
            return no(format!("`{op}` on {} and {}", a.shape(), b.shape()), line);
        };
        if let (Some(p), Some(q)) = (ka, kb) {
            let r = self.it.arith(op, V::Int(*p, ta), V::Int(*q, tb));
            return self.fold(r);
        }
        let ty = join_tag(ta, tb);
        let (wa, wb) = (*wa, *wb);
        let bits = int_bits(ty);
        let mut temp = |cx: &mut Self, w: u32, rhs: String| -> String {
            let t = cx.reg("t", w, true, "");
            cx.out(format!("{t} = {rhs};"));
            t
        };
        match op {
            "+" | "-" | "*" => {
                let w = if op == "*" { wa + wb } else { wa.max(wb) + 1 };
                let t = temp(self, w, format!("{x} {op} {y}"));
                self.check_range(Val::Int { e: t, tag: ty, w, k: None }, ty)
            }
            "/" | "%" => {
                let zero = format!("{y} == 2'sd0");
                self.trap_text(Some(&zero), "division by zero");
                let w = wa + 1;
                let q = self.reg("t", w, true, "");
                self.out(format!("if (!({zero})) {q} = {x} / {y};"));
                let qv = self.check_range(Val::Int { e: q, tag: ty, w, k: None }, ty)?;
                if op == "/" {
                    return Ok(qv);
                }
                let r = self.reg("t", wa.max(wb), true, "");
                self.out(format!("if (!({zero})) {r} = {x} % {y};"));
                self.check_range(Val::Int { e: r, tag: ty, w: wa.max(wb), k: None }, ty)
            }
            "&" | "|" | "^" => {
                let w = wa.max(wb);
                let t = temp(self, w, format!("{x} {op} {y}"));
                self.check_range(Val::Int { e: t, tag: ty, w, k: None }, ty)
            }
            "<<" | ">>" => {
                let amount = match kb {
                    Some(q) => {
                        if *q < 0 || *q >= bits as i128 {
                            self.trap_text(None, "shift amount out of range");
                        }
                        (*q).clamp(0, 127).to_string()
                    }
                    None => {
                        let (bl, _) = lit(bits as i128);
                        let bad = format!("{y} < 2'sd0 || {y} >= {bl}");
                        self.trap_text(Some(&bad), "shift amount out of range");
                        let yr = self.reg("t", wb, true, "shift amount");
                        self.out(format!("{yr} = {y};"));
                        format!("{yr}[{}:0]", wb.min(7) - 1)
                    }
                };
                if op == ">>" {
                    let t = temp(self, wa, format!("{x} >>> {amount}"));
                    return self.chk(Val::Int { e: t, tag: ta, w: wa, k: None }, ty);
                }
                let u = self.reg("t", bits, false, "");
                self.out(format!("{u} = {x} << {amount};"));
                let v = if ty.starts_with('i') {
                    Val::Int { e: format!("$signed({u})"), tag: ty, w: bits, k: None }
                } else {
                    Val::Int { e: format!("$signed({{1'b0, {u}}})"), tag: ty, w: bits + 1, k: None }
                };
                if ty == "lit" { self.check_range(v, ty) } else { Ok(v) }
            }
            _ => no(format!("the operator `{op}`"), line),
        }
    }

    /// `x.checked_op(y)` under `try … else E`: the value, or `fail E`.
    fn try_(&mut self, x: &Expr, els: &Expr, line: usize) -> R<Val> {
        let Expr::Call(f, args, _, _) = x else { return no("a `try` whose operand is not a `checked_*` call", line) };
        let Expr::Member(recv, m, _) = &**f else { return no("a `try` whose operand is not a `checked_*` call", line) };
        let op = match m.as_str() {
            "checked_add" => "+",
            "checked_sub" => "-",
            "checked_mul" => "*",
            "checked_div" => "/",
            _ => return no(format!("`try` on `.{m}` (EV-1: options appear only as `try x.checked_*(y) else E`)"), line),
        };
        let Expr::Name(err, _) = els else { return no("a `try … else` whose error is not a declared error name", line) };
        if args.len() != 1 {
            return no(format!("`.{m}` with {} arguments", args.len()), line);
        }
        let a = self.expr(recv)?;
        let a = if expr_may_store(&args[0].e) { self.hold(&a)? } else { a };
        let b = self.expr(&args[0].e)?;
        let (Val::Int { e: xe, tag: ta, w: wa, k: ka }, Val::Int { e: ye, tag: tb, w: wb, k: kb }) = (&a, &b) else {
            return no(format!("`.{m}` on {} and {}", a.shape(), b.shape()), line);
        };
        let ty = join_tag(ta, tb);
        if let (Some(p), Some(q)) = (ka, kb) {
            return match self.it.arith(op, V::Int(*p, ta), V::Int(*q, tb)) {
                Ok(v) => self.val_of(&v, line),
                Err(Flow::Trap(_)) => {
                    self.fail(err, line)?;
                    Ok(int_k(0, ty))
                }
                Err(f) => no(format!("a constant that does not evaluate ({})", crate::interp::why(f)), line),
            };
        }
        let (wa, wb) = (*wa, *wb);
        let w = match op {
            "*" => wa + wb,
            "/" => wa + 1,
            _ => wa.max(wb) + 1,
        };
        let t = self.reg("t", w, true, "");
        let ct = if ty == "lit" { "i64" } else { ty };
        let (lo, hi) = range(ct);
        let (l, _) = lit(lo);
        let (h, _) = lit(hi);
        let bad = if op == "/" {
            self.out(format!("if ({ye} != 2'sd0) {t} = {xe} / {ye};"));
            format!("{ye} == 2'sd0 || {t} < {l} || {t} > {h}")
        } else {
            self.out(format!("{t} = {xe} {op} {ye};"));
            format!("{t} < {l} || {t} > {h}")
        };
        let i = self.errors.iter().position(|x| x == err);
        let Some(i) = i else { return no(format!("`else {err}`, which is not a declared error"), line) };
        self.out(format!("if (!k && ({bad})) begin k = 1'b1; kerr = {}; end", i + 1));
        Ok(Val::Int { e: t, tag: ty, w: w.min(carrier(ct)), k: None })
    }

    fn call(&mut self, f: &Expr, args: &[Arg], line: usize) -> R<Val> {
        match f {
            Expr::Name(n, _) if self.lookup(n).is_none() && self.fns.contains_key(n) => {
                let fd = self.fns[n];
                self.inline(fd, args, line)
            }
            Expr::Member(recv, m, _) => {
                let a = self.expr(recv)?;
                let Val::Int { tag: t, .. } = &a else { return no(format!("the method `.{m}` on {}", a.shape()), line) };
                let t = *t;
                match m.as_str() {
                    "abs" if args.is_empty() => {
                        let Val::Int { e, w, k, .. } = &a else { unreachable!() };
                        if let Some(kv) = k {
                            return self.fold(crate::interp::chk(kv.abs(), t));
                        }
                        let r = self.reg("t", w + 1, true, "");
                        self.out(format!("{r} = ({e} < 2'sd0) ? -{e} : {e};"));
                        self.check_range(Val::Int { e: r, tag: t, w: w + 1, k: None }, t)
                    }
                    "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub" | "saturating_mul" if args.len() == 1 => {
                        let a = if expr_may_store(&args[0].e) { self.hold(&a)? } else { a };
                        let b = self.expr(&args[0].e)?;
                        let (Val::Int { e: x, w: wa, k: ka, .. }, Val::Int { e: y, w: wb, k: kb, .. }) = (&a, &b) else {
                            return no(format!("`.{m}` of {}", b.shape()), line);
                        };
                        let (lo, hi) = crate::types::int_range(t);
                        let op = match &m[m.len() - 3..] {
                            "add" => "+",
                            "sub" => "-",
                            _ => "*",
                        };
                        if let (Some(p), Some(q)) = (ka, kb) {
                            let r = match op {
                                "+" => p + q,
                                "-" => p - q,
                                _ if m.starts_with("wrapping") => p.wrapping_mul(*q),
                                _ => p.checked_mul(*q).unwrap_or(if (*p < 0) != (*q < 0) { i128::MIN } else { i128::MAX }),
                            };
                            let v = if m.starts_with("wrapping") { r.wrapping_sub(lo).rem_euclid(hi - lo + 1) + lo } else { r.clamp(lo, hi) };
                            return Ok(int_k(v, t));
                        }
                        let w = if op == "*" { wa + wb } else { (*wa).max(*wb) + 1 };
                        let r = self.reg("t", w, true, "");
                        self.out(format!("{r} = {x} {op} {y};"));
                        let bits = int_bits(t);
                        if m.starts_with("wrapping") {
                            let e = if t.starts_with('i') { format!("$signed({r}[{}:0])", bits - 1) } else { format!("$signed({{1'b0, {r}[{}:0]}})", bits - 1) };
                            let o = self.reg("t", bits + 1, true, "");
                            self.out(format!("{o} = {e};"));
                            Ok(Val::Int { e: o, tag: t, w: bits + 1, k: None })
                        } else {
                            let (l, _) = lit(lo);
                            let (h, _) = lit(hi);
                            let o = self.reg("t", carrier(t), true, "");
                            self.out(format!("{o} = ({r} < {l}) ? {l} : (({r} > {h}) ? {h} : {r});"));
                            Ok(Val::Int { e: o, tag: t, w: carrier(t), k: None })
                        }
                    }
                    _ => no(format!("the method `.{m}` (EV-1)"), line),
                }
            }
            Expr::Name(n, _) => no(format!("a call of `{n}`, which is not a function of this module (EV-1)"), line),
            _ => no("this call (EV-1)", line),
        }
    }

    /// A call of a function of the module, inlined (interp `call_fn`):
    /// each argument is evaluated and fitted to its parameter's type in
    /// order, the body runs in a scope of its own, and the result is fitted
    /// to the declared result type.
    fn inline(&mut self, fd: &'m FnDecl, args: &[Arg], line: usize) -> R<Val> {
        let name = fd.sig.name.clone();
        if self.calls.contains(&name) {
            return no(format!("the recursive function `{name}` (EV-1: calls are inlined)"), line);
        }
        if !fd.sig.generics.is_empty() {
            return no(format!("the generic function `{name}`"), fd.sig.line);
        }
        if matches!(fd.sig.ret, Some(Ty::Iter(_))) {
            return no(format!("the iterator function `{name}`"), fd.sig.line);
        }
        let mut params = vec![];
        for p in &fd.sig.params {
            match p {
                Param::Named { comptime: false, name: pn, ty, .. } => {
                    if matches!(ty, Ty::Ref(..)) {
                        return no(format!("the reference parameter `{pn}` of `{name}`"), fd.sig.line);
                    }
                    params.push((pn.clone(), self.ht(ty, fd.sig.line)?));
                }
                _ => return no(format!("the parameters of `{name}`"), fd.sig.line),
            }
        }
        if params.len() != args.len() {
            return no(format!("a call of `{name}` with {} arguments", args.len()), line);
        }
        let ret = match &fd.sig.ret {
            None => None,
            Some(t) => Some(self.ht(t, fd.sig.line)?),
        };
        let mut vals = vec![];
        for ((_, t), a) in params.iter().zip(args) {
            let v = self.expr(&a.e)?;
            let v = self.fit(v, t)?;
            vals.push(self.hold(&v)?);
        }
        self.calls.push(name.clone());
        let saved = std::mem::replace(&mut self.scopes, vec![HashMap::new()]);
        let flag = self.reg("r", 1, false, &format!("returned from {name}"));
        self.out(format!("{flag} = 1'b0;"));
        let res = match &ret {
            Some(t) => {
                let z = self.zero(t);
                Some((self.new_local(&format!("result of {name}"), &z)?, t.clone()))
            }
            None => None,
        };
        for ((pn, _), v) in params.iter().zip(&vals) {
            let id = self.new_local(pn, v)?;
            self.bind(pn, id);
        }
        self.frames.push(Frame { ret: Some(flag), res: res.clone(), loops: vec![] });
        let r = (|| {
            match &fd.body {
                Body::Arrow(e) => {
                    let v = self.expr(e)?;
                    if let Some((id, t)) = &res {
                        let v = self.fit(v, t)?;
                        self.store(*id, &v)?;
                    }
                }
                Body::Block(b) => {
                    let tail = res.as_ref().map(|(id, t)| Tail { res: *id, ty: t.clone() });
                    self.block(b, tail.as_ref())?;
                }
            }
            Ok(())
        })();
        self.frames.pop();
        self.scopes = saved;
        self.calls.pop();
        r?;
        Ok(match res {
            Some((id, _)) => self.locals[id].v.clone(),
            None => bool_k(false),
        })
    }

    /// A zero value of `t` (the start of a result register).
    fn zero(&self, t: &HT) -> Val {
        match t {
            HT::Int(n) => int_k(0, n),
            HT::Bool => bool_k(false),
            HT::Kind(k) => Val::Kind { kind: k.clone(), e: format!("{}'d0", self.kind_bits(k)), k: Some(0) },
            HT::Arr(e, n) => Val::Arr(vec![self.zero(e); *n]),
        }
    }

    /// `v` given declared type `t` (interp `fit`).
    fn fit(&mut self, v: Val, t: &HT) -> R<Val> {
        match (v, t) {
            (v @ Val::Int { .. }, HT::Int(n)) => self.chk(v, n),
            (v @ Val::Bool { .. }, HT::Bool) => Ok(v),
            (v @ Val::Kind { .. }, HT::Kind(k)) if matches!(&v, Val::Kind { kind, .. } if kind == k) => Ok(v),
            (Val::Arr(xs), HT::Arr(e, n)) if xs.len() == *n => {
                let mut out = vec![];
                for x in xs {
                    out.push(self.fit(x, e)?);
                }
                Ok(Val::Arr(out))
            }
            (v, t) => no(format!("{} where `{}` is declared", v.shape(), self.ht_name(t)), self.line),
        }
    }
}

/// Whether evaluating `e` can change a register the left operand reads
/// (only a call can: it stores its parameters and result).
fn expr_may_store(e: &Expr) -> bool {
    let mut found = false;
    crate::v09::each_expr(e, false, &mut |x, _| {
        if matches!(x, Expr::Call(..) | Expr::If(..) | Expr::Index(..) | Expr::Try(..)) || matches!(x, Expr::Bin(op, ..) if op == "and" || op == "or") {
            found = true;
        }
    });
    found
}

impl Cx<'_> {
    /// Whether matching `p` binds a name (interp `pmatch`: a one-segment
    /// path that names no variant, event or error).
    fn pat_binds(&self, p: &Pat) -> bool {
        match p {
            Pat::Path(x) => x.len() == 1 && !self.variant.contains_key(&x[0]) && !self.events.iter().any(|e| e.0 == x[0]) && !self.errors.contains(&x[0]),
            Pat::Ctor(_, args) => args.iter().any(|(_, q)| self.pat_binds(q)),
            Pat::Tuple(ps) | Pat::Alt(ps) => ps.iter().any(|q| self.pat_binds(q)),
            _ => false,
        }
    }
}
