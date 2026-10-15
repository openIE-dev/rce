//! Spec D.7: lowering a module function to a Form IR term.
//!
//! A function f(p1, ..., pn) lowers to a term t with
//! t({"p1": v1, ..., "pn": vn}) = f(v1, ..., vn), values in the Form's JSON
//! embedding (interp::form_value: lists and tuples as arrays, structs as
//! objects with sorted keys). Statements are lowered store-passing: a
//! statement is a term from an environment record to the next one. The
//! environment holds the variables in scope and four control fields
//! (`$done`, `$ret`, `$brk`, `$cnt`); every statement is guarded so it is
//! skipped once the function has returned or the loop iteration has been
//! left by `break` or `continue`. Which variables are in scope is known
//! statically, so an update is an `obj` that rebuilds exactly those fields.
//!
//! Loops become `while` terms whose accumulator is the environment;
//! `while c bound n` counts its iterations and traps as the interpreter
//! does (the check comes after the condition holds, 6.7). `for x in e`
//! walks the items of e by index. Integers must be i64 (other widths have
//! other overflow rules); floats are refused. Generic functions lower to
//! one term for every instance: the IR is untyped over JSON, so the term is
//! the monomorphization at every type at once, and vectors check it at i64.
//!
//! Anything else is refused with the line and the construct. A lowering is
//! a claim, not a proof: `rce index --vectors N` records the interpreter's
//! results on generated inputs, and the engine evaluates the term on them
//! (`rce::form::brick::verify`).
use crate::ast::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

type L = Result<Value, String>;

const CTRL: [&str; 4] = ["$brk", "$cnt", "$done", "$ret"];

pub struct Lowerer<'a> {
    pub fns: HashMap<String, &'a FnDecl>,
    structs: HashMap<String, Vec<String>>,
    consts: HashMap<String, Value>,
    module: String,
    node: String,
    depth: usize,
    /// variables of type usize in the function being lowered: an update
    /// that would make one negative traps, as the interpreter's does
    unsigned: HashSet<String>,
    /// lowered functions, by name: a term does not depend on its call site
    memo: HashMap<String, L>,
    /// format 2 definitions: digest of a callee's term → the term
    defs: BTreeMap<String, Value>,
    /// variants of the module's kinds: name → field names in order
    variants: HashMap<String, Vec<String>>,
    /// nesting of `match` statements (names the scrutinee field `$m<k>`)
    matches: usize,
    /// the function being lowered has a result `T ! E`: `fail` and `try`
    /// leave it with the error value `{"$err": e}`
    fails: bool,
    /// names the temporaries of `try` and `catch` (`$try<k>`, `$c<k>`)
    tmps: usize,
    /// the next call may have a result `T ! E` (it is the operand of `try`
    /// or `catch`, or passed on by `return`); taken by that call
    allow_union: bool,
    /// the recursive group (a strongly connected component of the call
    /// graph) of each function that calls itself, directly or not
    groups: HashMap<String, Vec<String>>,
    /// the group being lowered: a call of a member is `rec` of its index
    group: Option<Vec<String>>,
    /// the counters of the group's members declared with `bound n` (6.7):
    /// `$b<i>` is the number of active calls of member i, passed by every
    /// `rec` call (a call from outside the group starts them at 0)
    rec_bound: Vec<String>,
}

/// The functions a body calls by name.
fn callees(d: &FnDecl) -> BTreeSet<String> {
    fn walk(v: &Value, out: &mut BTreeSet<String>) {
        match v {
            Value::Object(m) => {
                if let Some(Value::Array(a)) = m.get("Call")
                    && let Some(Value::Array(n)) = a.first().and_then(|f| f.get("Name"))
                    && let Some(Value::String(g)) = n.first()
                {
                    out.insert(g.clone());
                }
                m.values().for_each(|x| walk(x, out));
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    walk(
        &serde_json::to_value(&d.body).unwrap_or(Value::Null),
        &mut out,
    );
    out
}

/// The recursive groups of a call graph (Tarjan): each strongly connected
/// component of more than one function, or of one that calls itself, with
/// its members in name order.
fn recursive_groups(g: &BTreeMap<String, BTreeSet<String>>) -> HashMap<String, Vec<String>> {
    struct T<'a> {
        g: &'a BTreeMap<String, BTreeSet<String>>,
        idx: HashMap<&'a str, usize>,
        low: HashMap<&'a str, usize>,
        on: HashSet<&'a str>,
        st: Vec<&'a str>,
        n: usize,
        out: HashMap<String, Vec<String>>,
    }
    fn go<'a>(t: &mut T<'a>, v: &'a str) {
        t.idx.insert(v, t.n);
        t.low.insert(v, t.n);
        t.n += 1;
        t.st.push(v);
        t.on.insert(v);
        for w in t.g[v].iter().filter(|w| t.g.contains_key(w.as_str())) {
            let w = w.as_str();
            if !t.idx.contains_key(w) {
                go(t, w);
                let lw = t.low[w];
                let lv = t.low.get_mut(v).unwrap();
                *lv = (*lv).min(lw);
            } else if t.on.contains(w) {
                let iw = t.idx[w];
                let lv = t.low.get_mut(v).unwrap();
                *lv = (*lv).min(iw);
            }
        }
        if t.low[v] == t.idx[v] {
            let mut c = vec![];
            loop {
                let w = t.st.pop().unwrap();
                t.on.remove(w);
                c.push(w.to_string());
                if w == v {
                    break;
                }
            }
            if c.len() > 1 || t.g[v].contains(v) {
                c.sort();
                for m in &c {
                    t.out.insert(m.clone(), c.clone());
                }
            }
        }
    }
    let mut t = T {
        g,
        idx: HashMap::new(),
        low: HashMap::new(),
        on: HashSet::new(),
        st: vec![],
        n: 0,
        out: HashMap::new(),
    };
    for v in g.keys() {
        if !t.idx.contains_key(v.as_str()) {
            go(&mut t, v);
        }
    }
    t.out
}

/// Largest term (bytes of its JSON) a function lowers to, not counting
/// the definitions it calls (format 2: a call is a `ref` to the digest of
/// the callee's term); past this size the function is refused.
pub const MAX_TERM_BYTES: usize = 1 << 20;

fn non_i64_int(t: &Ty) -> Option<String> {
    let bad = ["i8", "i16", "i32", "u8", "u16", "u32", "u64", "f32", "f64"];
    match t {
        Ty::Path(p, a) => {
            if p.len() == 1 && bad.contains(&p[0].as_str()) {
                return Some(p[0].clone());
            }
            a.iter().find_map(non_i64_int)
        }
        Ty::Tuple(v) => v.iter().find_map(non_i64_int),
        Ty::Ref(false, x) => non_i64_int(x),
        Ty::Fn(..) | Ty::Chan(_) | Ty::Iter(_) | Ty::Ref(..) | Ty::Region(..) | Ty::Union(..) => {
            Some(crate::unparse::ty(t))
        }
        Ty::Slice(x) | Ty::Array(x, _) | Ty::Opt(x) => non_i64_int(x),
        Ty::Unit => None,
    }
}

/// The `&mut` parameters of a function, in parameter order (state-passing:
/// its term returns `[result, their final values...]`).
pub fn mut_params(d: &FnDecl) -> Vec<String> {
    d.sig
        .params
        .iter()
        .filter_map(|p| match p {
            Param::Named {
                name,
                ty: Ty::Ref(true, _),
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// A parameter's type with a top-level `&mut` removed.
fn unmut(t: &Ty) -> &Ty {
    match t {
        Ty::Ref(true, x) => x,
        t => t,
    }
}

fn is_usize(t: &Ty) -> bool {
    matches!(t, Ty::Path(p, a) if a.is_empty() && p.len() == 1 && p[0] == "usize")
}

fn c(v: Value) -> Value {
    json!({"t": "const", "value": v})
}

impl<'a> Lowerer<'a> {
    /// `m` is the linked module the function lives in (imports and `#!lib`
    /// items included); `module` names it in node ids; `consts` are the
    /// literal values of its constants (evaluated by the interpreter).
    pub fn new(m: &'a Module, module: &str, consts: HashMap<String, Value>) -> Self {
        let mut fns = HashMap::new();
        let mut structs = HashMap::new();
        let mut variants = HashMap::new();
        for it in &m.items {
            match it {
                Item::Kind { variants: vs, .. } => {
                    for (n, fs) in vs {
                        variants.insert(n.clone(), fs.iter().map(|f| f.0.clone()).collect());
                    }
                }
                Item::Fn(d) => {
                    fns.insert(d.sig.name.clone(), d);
                }
                // an error value is a record of its payload (no `$v`)
                Item::Error { name, payload, .. } => {
                    structs.insert(name.clone(), payload.iter().map(|f| f.0.clone()).collect());
                }
                Item::Struct { name, fields, .. } => {
                    structs.insert(
                        name.clone(),
                        fields.iter().map(|f| f.name.clone()).collect(),
                    );
                }
                _ => {}
            }
        }
        let calls: BTreeMap<String, BTreeSet<String>> =
            fns.iter().map(|(n, d)| (n.clone(), callees(d))).collect();
        Lowerer {
            fns,
            structs,
            consts,
            module: module.to_string(),
            node: String::new(),
            depth: 0,
            unsigned: HashSet::new(),
            memo: HashMap::new(),
            defs: BTreeMap::new(),
            variants,
            matches: 0,
            fails: false,
            tmps: 0,
            allow_union: false,
            groups: recursive_groups(&calls),
            group: None,
            rec_bound: vec![],
        }
    }

    fn n(&self) -> String {
        self.node.clone()
    }
    fn proj(&self, f: &str) -> Value {
        json!({"t": "proj", "node": self.n(), "field": f})
    }
    fn prim(&self, op: &str, args: Vec<Value>) -> Value {
        json!({"t": "prim", "node": self.n(), "op": op, "args": args})
    }
    fn cond(&self, c: Value, t: Value, e: Value) -> Value {
        json!({"t": "cond", "node": self.n(), "cond": c, "then": t, "otherwise": e})
    }
    fn err(&self, m: &str) -> Value {
        json!({"t": "err", "node": self.n(), "message": m})
    }
    fn pipe(a: Value, b: Value) -> Value {
        if a == json!({"t": "pass"}) {
            return b;
        }
        if b == json!({"t": "pass"}) {
            return a;
        }
        json!({"t": "pipe", "first": a, "then": b})
    }
    /// Sequential composition as a balanced tree of pipes, so the depth of
    /// a block's term grows with the log of its length.
    fn seq(mut ts: Vec<Value>) -> Value {
        match ts.len() {
            0 => json!({"t": "pass"}),
            1 => ts.pop().unwrap(),
            n => {
                let b = ts.split_off(n / 2);
                Self::pipe(Self::seq(ts), Self::seq(b))
            }
        }
    }
    fn arith(&self, op: &str, a: Value, b: Value) -> Value {
        json!({"t": "arith", "node": self.n(), "op": op, "a": a, "b": b})
    }
    /// The environment with the fields `upd` replaced; scope `s`.
    fn set(&self, s: &[String], upd: &[(&str, Value)]) -> Value {
        let mut fields = vec![];
        for v in s {
            match upd.iter().find(|(k, _)| k == v) {
                Some((_, e)) => fields.push(json!([v, e])),
                None => fields.push(json!([v, self.proj(v)])),
            }
        }
        json!({"t": "obj", "fields": fields})
    }
    fn skipping(&self) -> Value {
        self.prim(
            "or",
            vec![
                self.prim("or", vec![self.proj("$done"), self.proj("$brk")]),
                self.proj("$cnt"),
            ],
        )
    }

    /// The function as a Form IR program: its term alone (format 1) when
    /// it calls nothing, else format 2 with the definitions it reaches.
    pub fn lower_fn(&mut self, name: &str) -> L {
        let main = self.lower_fn_in(name, &mut vec![])?;
        let mut need = BTreeSet::new();
        let mut todo = vec![main.clone()];
        while let Some(t) = todo.pop() {
            for k in refs(&t) {
                if need.insert(k.clone()) {
                    todo.push(self.defs[&k].clone());
                }
            }
        }
        if need.is_empty() {
            return Ok(main);
        }
        let defs: serde_json::Map<String, Value> = need
            .into_iter()
            .map(|k| (k.clone(), self.defs[&k].clone()))
            .collect();
        Ok(json!({"format": 2, "defs": defs, "main": main}))
    }

    fn lower_fn_in(&mut self, name: &str, stack: &mut Vec<String>) -> L {
        if let Some(r) = self.memo.get(name) {
            return r.clone();
        }
        if let Some(members) = self.groups.get(name).cloned() {
            return self.lower_group(name, members, stack);
        }
        if stack.iter().any(|s| s == name) {
            let l = self.fns.get(name).map(|d| d.sig.line).unwrap_or(0);
            return Err(format!("line {l}: recursive call of {name}"));
        }
        let r = self.lower_fn_uncached(name, stack).and_then(|t| {
            let n = serde_json::to_string(&t)
                .map(|x| x.len())
                .unwrap_or(usize::MAX);
            if n > MAX_TERM_BYTES {
                Err(format!("its term is {n} bytes, over {MAX_TERM_BYTES}"))
            } else {
                Ok(t)
            }
        });
        // a refusal that names a function on the stack depends on the stack
        if !matches!(&r, Err(e) if e.contains("recursive call")) {
            self.memo.insert(name.to_string(), r.clone());
        }
        r
    }

    /// A recursive group as one `fix` of its members' bodies, a call of a
    /// member inside being `rec` of its index; each member is the fix with
    /// its own entry (the same bodies, so one evaluation of the group).
    fn lower_group(&mut self, name: &str, members: Vec<String>, stack: &mut Vec<String>) -> L {
        let r = (|| {
            let saved = self.group.replace(members.clone());
            let mut bodies = vec![];
            let mut err = None;
            for m in &members {
                match self.lower_fn_uncached(m, stack) {
                    Ok(b) => bodies.push(b),
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            self.group = saved;
            if let Some(e) = err {
                return Err(e);
            }
            let size = serde_json::to_string(&bodies)
                .map(|x| x.len())
                .unwrap_or(usize::MAX);
            if size > MAX_TERM_BYTES {
                return Err(format!("its term is {size} bytes, over {MAX_TERM_BYTES}"));
            }
            Ok(bodies)
        })();
        for (i, m) in members.iter().enumerate() {
            let t = r.clone().map(
                |bodies| json!({"t": "fix", "node": self.node_of(m), "bodies": bodies, "entry": i}),
            );
            self.memo.insert(m.clone(), t);
        }
        self.memo[name].clone()
    }

    /// A call of `g`: `rec` inside its own group, else a `ref` to its term.
    fn callee_term(&mut self, g: &str, stack: &mut Vec<String>) -> L {
        if let Some(i) = self
            .group
            .as_ref()
            .and_then(|ms| ms.iter().position(|m| m == g))
        {
            return Ok(json!({"t": "rec", "node": self.n(), "index": i}));
        }
        let callee = self.lower_fn_in(g, stack)?;
        // format 2: the callee once in the table, a ref at the call
        let k = crate::brick::digest(&callee);
        self.defs.entry(k.clone()).or_insert(callee);
        Ok(json!({"t": "ref", "node": self.n(), "def": k}))
    }

    fn lower_fn_uncached(&mut self, name: &str, stack: &mut Vec<String>) -> L {
        let d = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("no function {name}"))?;
        let l = d.sig.line;
        if stack.iter().any(|s| s == name) {
            return Err(format!("line {l}: recursive call of {name}"));
        }
        if d.sig.comptime {
            return Err(format!("line {l}: comptime function"));
        }
        let mut params = vec![];
        for p in &d.sig.params {
            match p {
                Param::Named {
                    name: n,
                    ty,
                    comptime,
                    ..
                } => {
                    if *comptime {
                        return Err(format!("line {l}: comptime parameter {n}"));
                    }
                    if let Some(t) = non_i64_int(unmut(ty)) {
                        return Err(format!(
                            "line {l}: parameter {n}: type {t} is outside the lowered fragment (integers are i64 or usize, no floats)"
                        ));
                    }
                    params.push(n.clone());
                }
                _ => return Err(format!("line {l}: method")),
            }
        }
        let muts = mut_params(d);
        match &d.sig.ret {
            None | Some(Ty::Unit) if muts.is_empty() => {
                return Err(format!("line {l}: no result value"));
            }
            None | Some(Ty::Unit) => {}
            Some(Ty::Union(t, _)) => {
                if let Some(t) = non_i64_int(t) {
                    return Err(format!(
                        "line {l}: result type {t} is outside the lowered fragment"
                    ));
                }
            }
            Some(t) => {
                if let Some(t) = non_i64_int(t) {
                    return Err(format!(
                        "line {l}: result type {t} is outside the lowered fragment"
                    ));
                }
            }
        }
        let nn = self.node_of(name);
        let saved = (std::mem::replace(&mut self.node, nn), self.depth);
        let saved_u = std::mem::take(&mut self.unsigned);
        let saved_f = std::mem::replace(&mut self.fails, matches!(d.sig.ret, Some(Ty::Union(..))));
        for p in &d.sig.params {
            if let Param::Named { name: n, ty, .. } = p
                && is_usize(unmut(ty))
            {
                self.unsigned.insert(n.clone());
            }
        }
        self.depth = 0;
        stack.push(name.to_string());
        // in a group, the counters of its bounded members and this one's bound
        let (counters, bound) = match &self.group {
            Some(ms) => {
                let cs: Vec<(String, bool)> = ms
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| self.fns[m.as_str()].sig.bound.is_some())
                    .map(|(i, m)| (format!("$b{i}"), m == name))
                    .collect();
                (cs, d.sig.bound)
            }
            None => (vec![], None),
        };
        let saved_b = std::mem::replace(
            &mut self.rec_bound,
            counters.iter().map(|c| c.0.clone()).collect(),
        );
        let r = (|| {
            let mut s: Vec<String> = params.clone();
            s.extend(CTRL.iter().map(|x| x.to_string()));
            // `bound n` (6.7): this call is active call number $b<self>,
            // counted from the call that entered the group (which passes no
            // counters); past n it traps
            let mut bfield = vec![];
            for (f, me) in &counters {
                s.push(f.clone());
                let prev = json!({"t": "if_has", "field": f,
                    "then": self.proj(f), "otherwise": c(json!(0))});
                let v = if *me {
                    self.arith("add", prev, c(json!(1)))
                } else {
                    prev
                };
                bfield.push(json!([f, v]));
            }
            let mine = counters.iter().find(|c| c.1).map(|c| c.0.clone());
            let init_fields: Vec<Value> = params
                .iter()
                .map(|p| json!([p, self.proj(p)]))
                .chain(bfield)
                .chain([
                    json!(["$brk", c(json!(false))]),
                    json!(["$cnt", c(json!(false))]),
                    json!(["$done", c(json!(false))]),
                    json!(["$ret", c(Value::Null)]),
                ])
                .collect();
            let mut init = json!({"t": "obj", "fields": init_fields});
            if let (Some(n), Some(f)) = (bound, &mine) {
                let over = self.prim("gt", vec![self.proj(f), c(json!(n))]);
                init = Self::pipe(
                    init,
                    self.cond(
                        over,
                        self.err("trap: recursion bound"),
                        json!({"t": "pass"}),
                    ),
                );
            }
            // state-passing: with &mut parameters the result is
            // [result, final values of the &mut parameters in order]
            let out = |me: &Self, r: Value| {
                if muts.is_empty() {
                    r
                } else {
                    let mut t = me.prim("append", vec![c(json!([])), r]);
                    for m in &muts {
                        t = me.prim("append", vec![t, me.proj(m)]);
                    }
                    t
                }
            };
            match &d.body {
                Body::Arrow(e) => {
                    self.allow_union = self.fails && matches!(e, Expr::Call(..));
                    let r = self.expr(e, &s, stack)?;
                    Ok(Self::pipe(init, out(self, r)))
                }
                Body::Block(ss) => {
                    let b = self.block(ss, &mut s, true, stack)?;
                    let r = out(self, self.proj("$ret"));
                    Ok(Self::pipe(init, Self::pipe(b, r)))
                }
            }
        })();
        stack.pop();
        self.rec_bound = saved_b;
        self.node = saved.0;
        self.depth = saved.1;
        self.unsigned = saved_u;
        self.fails = saved_f;
        r
    }

    fn node_of(&self, f: &str) -> String {
        if self.module.is_empty() {
            f.to_string()
        } else {
            format!("{}.{f}", self.module)
        }
    }

    /// A block; `s` is the scope on entry and is restored on exit.
    fn block(
        &mut self,
        ss: &[Stmt],
        s: &mut Vec<String>,
        tail: bool,
        stack: &mut Vec<String>,
    ) -> L {
        let outer = s.clone();
        let mut ts = vec![];
        for (i, st) in ss.iter().enumerate() {
            let last = tail && i + 1 == ss.len();
            let t = self.stmt(st, s, last, stack)?;
            ts.push(self.cond(self.skipping(), json!({"t": "pass"}), t));
        }
        *s = outer;
        Ok(Self::seq(ts))
    }

    fn bind(&self, s: &mut Vec<String>, x: &str, l: usize) -> Result<(), String> {
        if s.iter().any(|v| v == x) {
            return Err(format!(
                "line {l}: `{x}` is declared again in an inner scope"
            ));
        }
        s.push(x.to_string());
        Ok(())
    }

    fn decl(
        &mut self,
        x: &str,
        ty: &Option<Ty>,
        e: &Expr,
        l: usize,
        s: &mut Vec<String>,
        stack: &mut Vec<String>,
    ) -> L {
        if let Some(t) = ty.as_ref().and_then(non_i64_int) {
            return Err(format!("line {l}: local of type {t}"));
        }
        let mut v = self.expr(e, s, stack)?;
        if ty.as_ref().is_some_and(is_usize) {
            self.unsigned.insert(x.to_string());
        }
        if self.unsigned.contains(x) {
            v = self.unsigned_guard(v);
        }
        self.bind(s, x, l)?;
        Ok(self.set(s, &[(x, v)]))
    }

    /// A call of a function with `&mut` parameters, as a statement, a
    /// `let`/`var` initializer or the right side of `=` to a name: the call
    /// is evaluated once into `$call`, then every borrowed place is assigned
    /// its returned value and the result (index 0) is bound or assigned.
    fn mut_call_stmt(
        &mut self,
        st: &Stmt,
        s: &mut Vec<String>,
        stack: &mut Vec<String>,
    ) -> Result<Option<Value>, String> {
        let (call, bind): (&Expr, Option<Stmt>) = match st {
            Stmt::Expr(e) => (e, None),
            Stmt::Let(p, ty, e, l) => (e, Some(Stmt::Let(p.clone(), ty.clone(), r0(*l), *l))),
            Stmt::Var(x, ty, e, l) => (e, Some(Stmt::Var(x.clone(), ty.clone(), r0(*l), *l))),
            Stmt::Assign(op, place, e) if op == "=" => (
                e,
                Some(Stmt::Assign(op.clone(), place.clone(), r0(e.line()))),
            ),
            _ => return Ok(None),
        };
        let Expr::Call(f, args, l, _) = call else {
            return Ok(None);
        };
        let Expr::Name(g, _) = &**f else {
            return Ok(None);
        };
        let Some(d) = self.fns.get(g).copied() else {
            return Ok(None);
        };
        let muts = mut_params(d);
        if muts.is_empty() {
            return Ok(None);
        }
        let l = *l;
        if matches!(d.sig.ret, Some(Ty::Union(..))) {
            return Err(format!(
                "line {l}: a call of {g} (it takes &mut and its result is T ! E)"
            ));
        }
        let names: Vec<String> = d
            .sig
            .params
            .iter()
            .filter_map(|p| match p {
                Param::Named { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        if names.len() != args.len() {
            return Err(format!("line {l}: call of {g} with these arguments"));
        }
        let mut fields = vec![];
        let mut places: Vec<(String, Expr)> = vec![];
        for (i, a) in args.iter().enumerate() {
            let n = a.label.clone().unwrap_or_else(|| names[i].clone());
            let e = match (&a.e, muts.contains(&n)) {
                (Expr::Borrow(true, place, _), true) => {
                    places.push((n.clone(), (**place).clone()));
                    (**place).clone()
                }
                // a variable that is itself a &mut parameter, passed on
                (Expr::Name(x, _), true) if s.contains(x) => {
                    places.push((n.clone(), a.e.clone()));
                    a.e.clone()
                }
                (_, true) => return Err(format!("line {l}: &mut argument {n} is not a place")),
                (e, false) => e.clone(),
            };
            fields.push(json!([n, self.expr(&e, s, stack)?]));
        }
        let call = self.callee_term(g, stack)?;
        if call["t"] == "rec" {
            for f in &self.rec_bound {
                fields.push(json!([f, self.proj(f)]));
            }
        }
        let t = Self::pipe(json!({"t": "obj", "fields": fields}), call);
        // env + $call
        let mut s2 = s.clone();
        let mut ext: Vec<Value> = s.iter().map(|v| json!([v, self.proj(v)])).collect();
        ext.push(json!(["$call", t]));
        s2.push("$call".into());
        let mut ts = vec![json!({"t": "obj", "fields": ext})];
        for (n, place) in &places {
            let j = 1 + muts.iter().position(|m| m == n).unwrap();
            let back = Stmt::Assign("=".into(), place.clone(), at_call(j, l));
            ts.push(self.stmt(&back, &mut s2, false, stack)?);
        }
        if let Some(b) = bind {
            ts.push(self.stmt(&b, &mut s2, false, stack)?);
        }
        s2.retain(|v| v != "$call");
        ts.push(self.set(&s2, &[]));
        *s = s2;
        Ok(Some(Self::seq(ts)))
    }

    /// `try x [else e]` as the initializer of `let`/`var`, the right side
    /// of an assignment, a statement of its own or the value of `return`:
    /// `x` once into `$try<k>`; an error (or `none`) leaves the function
    /// with `{"$err": ..}` (`e` when given), else the statement runs with
    /// the value. As in the evaluator, `try none` with no `else` traps.
    fn try_stmt(
        &mut self,
        st: &Stmt,
        s: &mut Vec<String>,
        tail: bool,
        stack: &mut Vec<String>,
    ) -> Result<Option<Value>, String> {
        let te = match st {
            Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) | Stmt::Assign(_, _, e) => e,
            Stmt::Expr(e) | Stmt::Return(Some(e), _) => e,
            _ => return Ok(None),
        };
        let Expr::Try(x, els, l) = te else {
            return Ok(None);
        };
        let l = *l;
        if !self.fails {
            return Err(format!(
                "line {l}: try in a function whose result is not T ! E"
            ));
        }
        let ret = match &**x {
            Expr::Call(f, _, _, _) => match &**f {
                Expr::Name(g, _) => self.fns.get(g).map(|d| d.sig.ret.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(ret) = ret else {
            return Err(format!(
                "line {l}: try on an operand that is not a call of a function"
            ));
        };
        let k = format!("$try{}", self.tmps);
        self.tmps += 1;
        self.allow_union = true;
        let v = self.expr(x, s, stack);
        self.allow_union = false;
        let v = v?;
        let mut s2 = s.clone();
        s2.push(k.clone());
        let mut ext: Vec<Value> = s.iter().map(|x| json!([x, self.proj(x)])).collect();
        ext.push(json!([k, v]));
        let name = Expr::Name(k.clone(), l);
        let ok_st = match st {
            Stmt::Let(p, ty, _, l2) => Stmt::Let(p.clone(), ty.clone(), name, *l2),
            Stmt::Var(x, ty, _, l2) => Stmt::Var(x.clone(), ty.clone(), name, *l2),
            Stmt::Assign(op, place, _) => Stmt::Assign(op.clone(), place.clone(), name),
            Stmt::Return(_, l2) => Stmt::Return(Some(name), *l2),
            _ => Stmt::Expr(name),
        };
        let els_v = match els {
            Some(e) => Some(json!({"t": "obj", "fields": [["$err", self.expr(e, &s2, stack)?]]})),
            None => None,
        };
        let leave = |me: &Self, r: Value| me.set(s, &[("$ret", r), ("$done", c(json!(true)))]);
        let (test, bad) = match &ret {
            Some(Ty::Union(..)) => (
                self.prim("has", vec![self.proj(&k), c(json!("$err"))]),
                leave(self, els_v.unwrap_or_else(|| self.proj(&k))),
            ),
            Some(Ty::Opt(_)) => (
                self.prim("eq", vec![self.proj(&k), c(Value::Null)]),
                match els_v {
                    Some(r) => leave(self, r),
                    None => self.err("trap: try on none"),
                },
            ),
            _ => (c(json!(false)), json!({"t": "pass"})),
        };
        let ok = self.stmt(&ok_st, &mut s2, tail, stack)?;
        s2.retain(|x| x != &k);
        let ok = Self::pipe(ok, self.set(&s2, &[]));
        *s = s2;
        let body = if test == c(json!(false)) {
            ok
        } else {
            self.cond(test, bad, ok)
        };
        Ok(Some(Self::pipe(json!({"t": "obj", "fields": ext}), body)))
    }

    fn and(&self, a: Value, b: Value) -> Value {
        if a == c(json!(true)) {
            return b;
        }
        if b == c(json!(true)) {
            return a;
        }
        self.cond(a, b, c(json!(false)))
    }

    /// A pattern against the value `v` (a term on the environment): the
    /// test, and the bindings (name, term). Values are the evaluator's
    /// embedding: a variant is an object with `$v`, `some(x)` is x, `none`
    /// is null.
    fn pat(
        &mut self,
        p: &Pat,
        v: Value,
        s: &[String],
        stack: &mut Vec<String>,
        l: usize,
        out: &mut Vec<(String, Value)>,
    ) -> L {
        let tag = |me: &Self, v: &Value| {
            Self::pipe(
                v.clone(),
                json!({"t": "proj", "node": me.n(), "field": "$v"}),
            )
        };
        match p {
            Pat::Wild => Ok(c(json!(true))),
            Pat::Path(q) => {
                let n = &q[q.len() - 1];
                if self.variants.contains_key(n) {
                    Ok(self.prim("eq", vec![tag(self, &v), c(json!(n))]))
                } else if n == "none" && q.len() == 1 {
                    Ok(self.prim("eq", vec![v, c(Value::Null)]))
                } else if q.len() == 1 {
                    if s.contains(n) || out.iter().any(|(x, _)| x == n) {
                        return Err(format!("line {l}: pattern binding {n} shadows a variable"));
                    }
                    out.push((n.clone(), v));
                    Ok(c(json!(true)))
                } else {
                    Err(format!("line {l}: pattern {}", q.join(".")))
                }
            }
            Pat::Ctor(q, args) => {
                let n = &q[q.len() - 1];
                if n == "some" && q.len() == 1 && args.len() == 1 {
                    let some = self.prim("ne", vec![v.clone(), c(Value::Null)]);
                    let inner = self.pat(&args[0].1, v, s, stack, l, out)?;
                    return Ok(self.and(some, inner));
                }
                let Some(fs) = self.variants.get(n).cloned() else {
                    return Err(format!("line {l}: pattern {n}(..)"));
                };
                let mut t = self.prim("eq", vec![tag(self, &v), c(json!(n))]);
                for (i, (lb, q)) in args.iter().enumerate() {
                    let f = match lb {
                        Some(f) => f.clone(),
                        None => fs
                            .get(i)
                            .cloned()
                            .ok_or_else(|| format!("line {l}: pattern {n}: too many fields"))?,
                    };
                    let fv = Self::pipe(
                        v.clone(),
                        json!({"t": "proj", "node": self.n(), "field": f}),
                    );
                    let ti = self.pat(q, fv, s, stack, l, out)?;
                    t = self.and(t, ti);
                }
                Ok(t)
            }
            Pat::Lit(e) => {
                let x = self.expr(e, s, stack)?;
                Ok(self.prim("eq", vec![v, x]))
            }
            Pat::Range(a, incl, b) => {
                let (x, y) = (self.expr(a, s, stack)?, self.expr(b, s, stack)?);
                let lo = self.prim("ge", vec![v.clone(), x]);
                let hi = self.prim(if *incl { "le" } else { "lt" }, vec![v, y]);
                Ok(self.and(lo, hi))
            }
            Pat::Tuple(ps) => {
                let n = self.prim("size", vec![v.clone()]);
                let mut t = self.prim("eq", vec![n, c(json!(ps.len()))]);
                for (i, q) in ps.iter().enumerate() {
                    let xi = self.prim("at", vec![v.clone(), c(json!(i))]);
                    let ti = self.pat(q, xi, s, stack, l, out)?;
                    t = self.and(t, ti);
                }
                Ok(t)
            }
            Pat::Alt(ps) => {
                let mut t = c(json!(false));
                for q in ps {
                    let mut b = vec![];
                    let ti = self.pat(q, v.clone(), s, stack, l, &mut b)?;
                    if !b.is_empty() {
                        return Err(format!("line {l}: an alternative pattern that binds"));
                    }
                    t = if t == c(json!(false)) {
                        ti
                    } else {
                        self.cond(t, c(json!(true)), ti)
                    };
                }
                Ok(t)
            }
        }
    }

    /// `match e`: the scrutinee once into `$m<k>`, then the arms in order
    /// (pattern, bindings, guard); no arm traps, as in the evaluator.
    fn match_stmt(
        &mut self,
        e: &Expr,
        arms: &[Arm],
        l: usize,
        s: &mut Vec<String>,
        tail: bool,
        stack: &mut Vec<String>,
    ) -> L {
        let m = format!("$m{}", self.matches);
        self.matches += 1;
        let r = (|| {
            let ev = self.expr(e, s, stack)?;
            let mut sm = s.clone();
            sm.push(m.clone());
            let mut ext: Vec<Value> = s.iter().map(|v| json!([v, self.proj(v)])).collect();
            ext.push(json!([m, ev]));
            // arms from the last: each falls through to the next
            let mut next = self.err("trap: no match arm");
            for a in arms.iter().rev() {
                let mut binds = vec![];
                let test = self.pat(&a.pat, self.proj(&m), &sm, stack, a.line, &mut binds)?;
                let mut sa = sm.clone();
                let mut bf: Vec<Value> = sm.iter().map(|v| json!([v, self.proj(v)])).collect();
                for (x, t) in &binds {
                    bf.push(json!([x, t]));
                    sa.push(x.clone());
                }
                let bind = json!({"t": "obj", "fields": bf});
                let body = self.block(&a.body, &mut sa, tail, stack)?;
                // leave with the outer scope (drops $m and the bindings)
                let body = Self::pipe(body, self.set(s, &[]));
                let taken = match &a.guard {
                    None => body,
                    Some(g) => {
                        let gv = self.expr(g, &sa, stack)?;
                        let back = self.set(&sm, &[]);
                        self.cond(gv, body, Self::pipe(back, next.clone()))
                    }
                };
                next = self.cond(test, Self::pipe(bind, taken), next);
            }
            let _ = l;
            Ok(Self::pipe(json!({"t": "obj", "fields": ext}), next))
        })();
        self.matches -= 1;
        r
    }

    fn unsigned_guard(&self, v: Value) -> Value {
        let neg = self.prim("lt", vec![v.clone(), c(json!(0))]);
        self.cond(neg, self.err("trap: overflow: negative usize"), v)
    }

    fn stmt(&mut self, st: &Stmt, s: &mut Vec<String>, tail: bool, stack: &mut Vec<String>) -> L {
        if let Some(t) = self.mut_call_stmt(st, s, stack)? {
            return Ok(t);
        }
        if let Stmt::Match(e, arms, l) = st {
            return self.match_stmt(e, arms, *l, s, tail, stack);
        }
        if let Some(t) = self.try_stmt(st, s, tail, stack)? {
            return Ok(t);
        }
        match st {
            Stmt::Var(x, ty, e, l) => self.decl(x, ty, e, *l, s, stack),
            Stmt::Let(Pat::Path(q), ty, e, l) if q.len() == 1 => {
                self.decl(&q[0], ty, e, *l, s, stack)
            }
            Stmt::Let(Pat::Tuple(ps), ty, e, l) => {
                if ty.as_ref().and_then(non_i64_int).is_some() {
                    return Err(format!("line {l}: local of a narrow integer type"));
                }
                let tmp = format!("$t{}", self.depth);
                let v = self.expr(e, s, stack)?;
                let mut s2 = s.clone();
                s2.push(tmp.clone());
                let mut t = self.set(&s2, &[(&tmp, v)]);
                for (k, p) in ps.iter().enumerate() {
                    match p {
                        Pat::Path(q) if q.len() == 1 => {
                            let g = self.prim("get", vec![self.proj(&tmp), c(json!(k))]);
                            self.bind(&mut s2, &q[0], *l)?;
                            self.bind(s, &q[0], *l)?;
                            t = Self::pipe(t, self.set(&s2, &[(&q[0], g)]));
                        }
                        Pat::Wild => {}
                        _ => return Err(format!("line {l}: tuple pattern")),
                    }
                }
                Ok(Self::pipe(t, self.set(s, &[])))
            }
            Stmt::Let(_, _, _, l) => Err(format!("line {l}: let pattern")),
            Stmt::Assign(op, place, rhs) => {
                let r = self.expr(rhs, s, stack)?;
                let v = if op == "=" {
                    r
                } else {
                    let cur = self.expr(place, s, stack)?;
                    self.binop(&op[..op.len() - 1], cur, r, place.line())?
                };
                match place {
                    Expr::Name(x, l) => {
                        if !s.contains(x) {
                            return Err(format!("line {l}: assignment to `{x}`"));
                        }
                        let v = if self.unsigned.contains(x) {
                            self.unsigned_guard(v)
                        } else {
                            v
                        };
                        Ok(self.set(s, &[(x, v)]))
                    }
                    Expr::Index(b, ix, l) if ix.len() == 1 => match &**b {
                        Expr::Name(x, _) if s.contains(x) => {
                            let i = self.expr(&ix[0], s, stack)?;
                            let nv = self.prim("set", vec![self.proj(x), i, v]);
                            Ok(self.set(s, &[(x, nv)]))
                        }
                        _ => Err(format!("line {l}: assignment to this place")),
                    },
                    e => Err(format!("line {}: assignment to this place", e.line())),
                }
            }
            Stmt::Expr(e) => {
                if let Expr::Call(f, args, l, _) = e
                    && let Expr::Member(r, m, _) = &**f
                    && let Expr::Name(x, _) = &**r
                    && s.contains(x)
                {
                    match (m.as_str(), args.len()) {
                        ("push", 1) => {
                            let v = self.expr(&args[0].e, s, stack)?;
                            return Ok(
                                self.set(s, &[(x, self.prim("append", vec![self.proj(x), v]))])
                            );
                        }
                        ("pop", 0) if !tail => {
                            let len = self.prim("len", vec![self.proj(x)]);
                            let short = self.prim(
                                "slice",
                                vec![
                                    self.proj(x),
                                    c(json!(0)),
                                    self.arith("sub", len.clone(), c(json!(1))),
                                ],
                            );
                            let gt = self.prim("gt", vec![len, c(json!(0))]);
                            let t = self.cond(gt, short, self.err("pop of an empty list"));
                            return Ok(self.set(s, &[(x, t)]));
                        }
                        _ => return Err(format!("line {l}: method {m}")),
                    }
                }
                self.allow_union = tail && self.fails && matches!(e, Expr::Call(..));
                let v = self.expr(e, s, stack)?;
                if tail {
                    Ok(self.set(s, &[("$ret", v), ("$done", c(json!(true)))]))
                } else {
                    let mut s2 = s.clone();
                    s2.push("$_".into());
                    Ok(Self::pipe(self.set(&s2, &[("$_", v)]), self.set(s, &[])))
                }
            }
            Stmt::Return(Some(e), _) => {
                self.allow_union = self.fails && matches!(e, Expr::Call(..));
                let v = self.expr(e, s, stack)?;
                Ok(self.set(s, &[("$ret", v), ("$done", c(json!(true)))]))
            }
            Stmt::Return(None, l) => Err(format!("line {l}: return without a value")),
            Stmt::If(arms, els) => {
                let mut t = match els {
                    Some(b) => self.block(b, s, tail, stack)?,
                    None => json!({"t": "pass"}),
                };
                for (cnd, b) in arms.iter().rev() {
                    let cv = self.expr(cnd, s, stack)?;
                    let bt = self.block(b, s, tail, stack)?;
                    t = self.cond(cv, bt, t);
                }
                Ok(t)
            }
            Stmt::While(cnd, bound, body) => {
                let cv = self.expr(cnd, s, stack)?;
                self.lp(Some(cv), bound.as_ref(), None, body, s, stack)
            }
            Stmt::Loop(bound, body) => self.lp(None, bound.as_ref(), None, body, s, stack),
            Stmt::For(p, it, body, l) => {
                let x = match p {
                    Pat::Path(q) if q.len() == 1 => q[0].clone(),
                    Pat::Wild => format!("$w{}", self.depth + 1),
                    p => return Err(format!("line {l}: for pattern {}", crate::unparse::pat(p))),
                };
                let items = match it {
                    Expr::Bin(op, a, b, _) if op == ".." || op == "..=" => {
                        let (a, b) = (self.expr(a, s, stack)?, self.expr(b, s, stack)?);
                        let b = if op == "..=" {
                            self.arith("add", b, c(json!(1)))
                        } else {
                            b
                        };
                        self.prim("range", vec![a, b])
                    }
                    e => self.expr(e, s, stack)?,
                };
                self.lp(None, None, Some((x, items, *l)), body, s, stack)
            }
            Stmt::Break(_) => Ok(self.set(s, &[("$brk", c(json!(true)))])),
            Stmt::Continue(_) => Ok(self.set(s, &[("$cnt", c(json!(true)))])),
            Stmt::Check(e) => {
                let v = self.expr(e, s, stack)?;
                Ok(self.cond(v, json!({"t": "pass"}), self.err("check is false")))
            }
            Stmt::Pass => Ok(json!({"t": "pass"})),
            Stmt::Fail(e) => {
                if !self.fails {
                    return Err(format!(
                        "line {}: fail in a function whose result is not T ! E",
                        e.line()
                    ));
                }
                let v = self.expr(e, s, stack)?;
                let ev = json!({"t": "obj", "fields": [["$err", v]]});
                Ok(self.set(s, &[("$ret", ev), ("$done", c(json!(true)))]))
            }
            Stmt::Comptime(b) => self.block(b, s, false, stack),
            other => Err(format!(
                "statement {} has no Form IR term",
                stmt_name(other)
            )),
        }
    }

    /// A loop. `cv`: while condition (on the environment); `each`: a for
    /// loop's variable, item list and line.
    fn lp(
        &mut self,
        cv: Option<Value>,
        bound: Option<&Expr>,
        each: Option<(String, Value, usize)>,
        body: &[Stmt],
        s: &mut Vec<String>,
        stack: &mut Vec<String>,
    ) -> L {
        self.depth += 1;
        let d = self.depth;
        let (nv, xs, iv) = (format!("$n{d}"), format!("$xs{d}"), format!("$i{d}"));
        let mut sl = s.clone();
        let mut pre = vec![];
        sl.push(nv.clone());
        pre.push((nv.clone(), c(json!(0))));
        if let Some((_, items, _)) = &each {
            sl.push(xs.clone());
            sl.push(iv.clone());
            pre.push((xs.clone(), items.clone()));
            pre.push((iv.clone(), c(json!(0))));
        }
        let pre_ref: Vec<(&str, Value)> =
            pre.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let init = self.set(&sl, &pre_ref);
        let bound_v = match bound {
            Some(b) => Some(self.expr(b, s, stack)?),
            None => None,
        };
        let stop = self.prim("or", vec![self.proj("$done"), self.proj("$brk")]);
        let go = match (&cv, &each) {
            (Some(cv), _) => self.cond(stop, c(json!(false)), cv.clone()),
            (None, Some(_)) => self.cond(
                stop,
                c(json!(false)),
                self.prim(
                    "lt",
                    vec![self.proj(&iv), self.prim("size", vec![self.proj(&xs)])],
                ),
            ),
            (None, None) => self.prim("not", vec![stop]),
        };
        let n1 = self.arith("add", self.proj(&nv), c(json!(1)));
        let mut sb = sl.clone();
        let mut head = self.set(&sl, &[("$cnt", c(json!(false))), (&nv, n1)]);
        if let Some(b) = &bound_v {
            let over = self.prim("gt", vec![self.proj(&nv), b.clone()]);
            head = Self::pipe(
                head,
                self.cond(over, self.err("trap: loop bound"), json!({"t": "pass"})),
            );
        }
        if let Some((x, _, l)) = &each {
            let item = self.prim("at", vec![self.proj(&xs), self.proj(&iv)]);
            self.bind(&mut sb, x, *l)?;
            let i1 = self.arith("add", self.proj(&iv), c(json!(1)));
            head = Self::pipe(head, self.set(&sb, &[(x, item), (&iv, i1)]));
        }
        let b = self.block(body, &mut sb, false, stack)?;
        // the body's own scope ends here: rebuild to the loop scope
        let iter = Self::pipe(Self::pipe(head, b), self.set(&sl, &[]));
        let acc = self.proj("acc");
        let w = json!({"t": "while", "node": self.n(), "init": {"t": "pass"},
            "cond": Self::pipe(acc.clone(), go), "f": Self::pipe(acc, iter)});
        self.depth -= 1;
        let after = self.set(s, &[("$brk", c(json!(false))), ("$cnt", c(json!(false)))]);
        Ok(Self::pipe(Self::pipe(init, w), after))
    }

    fn binop(&self, op: &str, x: Value, y: Value, l: usize) -> L {
        Ok(match op {
            "+" | "-" | "*" | "/" | "%" => {
                if op == "+" {
                    // integers add, texts and lists concatenate (`plus`)
                    return Ok(self.prim("plus", vec![x, y]));
                }
                let o = match op {
                    "-" => "sub",
                    "*" => "mul",
                    "/" => "div",
                    _ => "rem",
                };
                self.arith(o, x, y)
            }
            "&" | "|" | "^" | "<<" | ">>" => {
                let o = match op {
                    "&" => "band",
                    "|" => "bor",
                    "^" => "bxor",
                    "<<" => "shl",
                    _ => "shr",
                };
                self.prim(o, vec![x, y])
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                let o = match op {
                    "==" => "eq",
                    "!=" => "ne",
                    "<" => "lt",
                    "<=" => "le",
                    ">" => "gt",
                    _ => "ge",
                };
                self.prim(o, vec![x, y])
            }
            "and" => self.cond(x, y, c(json!(false))),
            "or" => self.cond(x, c(json!(true)), y),
            ".." => self.prim("range", vec![x, y]),
            "..=" => self.prim("range", vec![x, self.arith("add", y, c(json!(1)))]),
            _ => return Err(format!("line {l}: operator {op} has no Form IR term")),
        })
    }

    fn expr(&mut self, e: &Expr, s: &[String], stack: &mut Vec<String>) -> L {
        let l = e.line();
        let no = |what: &str| Err(format!("line {l}: {what} has no Form IR term"));
        match e {
            Expr::Int(n, _) => i64::try_from(*n)
                .map(|v| c(json!(v)))
                .map_err(|_| format!("line {l}: integer literal out of i64")),
            Expr::Bool(b, _) => Ok(c(json!(b))),
            Expr::Str(x, _) => Ok(c(json!(x))),
            Expr::None(_) | Expr::Unit(_) => Ok(c(Value::Null)),
            Expr::Name(n, _) => {
                if s.contains(n) {
                    Ok(self.proj(n))
                } else if let Some(v) = self.consts.get(n) {
                    Ok(c(v.clone()))
                } else if self.variants.get(n).is_some_and(|f| f.is_empty()) {
                    Ok(c(json!({"$v": n})))
                } else {
                    no(&format!("name {n}"))
                }
            }
            Expr::Bin(op, a, b, _) => {
                let (x, y) = (self.expr(a, s, stack)?, self.expr(b, s, stack)?);
                self.binop(op, x, y, l)
            }
            Expr::Not(x, _) => {
                let v = self.expr(x, s, stack)?;
                Ok(self.prim("not", vec![v]))
            }
            Expr::Neg(x, _) => {
                if let Expr::Int(n, _) = &**x {
                    return i64::try_from(-*n)
                        .map(|v| c(json!(v)))
                        .map_err(|_| format!("line {l}: integer literal out of i64"));
                }
                let v = self.expr(x, s, stack)?;
                Ok(self.prim("neg", vec![v]))
            }
            Expr::BitNot(x, _) => {
                let v = self.expr(x, s, stack)?;
                Ok(self.prim("bnot", vec![v]))
            }
            Expr::If(cnd, a, b, _) => {
                let (cv, av, bv) = (
                    self.expr(cnd, s, stack)?,
                    self.expr(a, s, stack)?,
                    self.expr(b, s, stack)?,
                );
                Ok(self.cond(cv, av, bv))
            }
            Expr::Comptime(x, _) => self.expr(x, s, stack),
            Expr::List(xs, _) | Expr::Tuple(xs, _) => {
                let mut t = c(json!([]));
                for x in xs {
                    let v = self.expr(x, s, stack)?;
                    t = self.prim("append", vec![t, v]);
                }
                Ok(t)
            }
            Expr::ListRepeat(v, n, _) => {
                let (v, n) = (self.expr(v, s, stack)?, self.expr(n, s, stack)?);
                let f = self.prim(
                    "append",
                    vec![self.proj("acc"), Self::pipe(self.proj("env"), v)],
                );
                Ok(
                    json!({"t": "fold", "node": self.n(), "over": self.prim("range", vec![c(json!(0)), n]), "init": c(json!([])), "f": f}),
                )
            }
            Expr::Index(b, ix, _) if ix.len() == 1 => {
                let (bv, iv) = (self.expr(b, s, stack)?, self.expr(&ix[0], s, stack)?);
                let neg = self.prim("lt", vec![iv.clone(), c(json!(0))]);
                Ok(self.cond(
                    neg,
                    self.err("trap: index out of range"),
                    self.prim("at", vec![bv, iv]),
                ))
            }
            Expr::Member(b, f, _) => {
                let bv = self.expr(b, s, stack)?;
                Ok(Self::pipe(
                    bv,
                    json!({"t": "proj", "node": self.n(), "field": f}),
                ))
            }
            Expr::TupleMember(b, k, _) => {
                let bv = self.expr(b, s, stack)?;
                Ok(self.prim("get", vec![bv, c(json!(k))]))
            }
            Expr::Cast(x, t, _) => match t {
                Ty::Path(p, a) if a.is_empty() && p.len() == 1 && p[0] == "i64" => {
                    self.expr(x, s, stack)
                }
                // an index: a negative value does not fit (6.3)
                Ty::Path(p, a) if a.is_empty() && p.len() == 1 && p[0] == "usize" => {
                    let v = self.expr(x, s, stack)?;
                    let neg = self.prim("lt", vec![v.clone(), c(json!(0))]);
                    Ok(self.cond(neg, self.err("trap: value does not fit usize"), v))
                }
                _ => no("this cast"),
            },
            Expr::Call(f, args, _, _) => self.call(f, args, s, stack, l),
            Expr::Catch(x, bind, h, _) => {
                if !matches!(&**x, Expr::Call(..)) {
                    return no("catch on this operand");
                }
                let k = format!("$c{}", self.tmps);
                self.tmps += 1;
                self.allow_union = true;
                let v = self.expr(x, s, stack);
                self.allow_union = false;
                let v = v?;
                let mut sh: Vec<String> = s.to_vec();
                let mut bf: Vec<Value> = s.iter().map(|x| json!([x, self.proj(x)])).collect();
                if let Some(b) = bind {
                    if s.contains(b) {
                        return Err(format!("line {l}: catch binding {b} shadows a variable"));
                    }
                    let ev = Self::pipe(
                        self.proj(&k),
                        json!({"t": "proj", "node": self.n(), "field": "$err"}),
                    );
                    bf.push(json!([b, ev]));
                    sh.push(b.clone());
                }
                let hv = self.expr(h, &sh, stack)?;
                let mut ext: Vec<Value> = s.iter().map(|x| json!([x, self.proj(x)])).collect();
                ext.push(json!([k, v]));
                let isr = self.prim("has", vec![self.proj(&k), c(json!("$err"))]);
                Ok(Self::pipe(
                    json!({"t": "obj", "fields": ext}),
                    self.cond(
                        isr,
                        Self::pipe(json!({"t": "obj", "fields": bf}), hv),
                        self.proj(&k),
                    ),
                ))
            }
            Expr::Borrow(false, x, _) => self.expr(x, s, stack),
            other => no(expr_name(other)),
        }
    }

    fn call(
        &mut self,
        f: &Expr,
        args: &[Arg],
        s: &[String],
        stack: &mut Vec<String>,
        l: usize,
    ) -> L {
        let no = |what: &str| Err(format!("line {l}: {what} has no Form IR term"));
        let union_ok = std::mem::take(&mut self.allow_union);
        match f {
            Expr::Member(r, m, _) => {
                let rv = self.expr(r, s, stack)?;
                match (m.as_str(), args.len()) {
                    ("len", 0) => Ok(self.prim("size", vec![rv])),
                    ("chars" | "tostr" | "toint" | "upper" | "lower", 0) => {
                        Ok(self.prim(m, vec![rv]))
                    }
                    ("contains" | "startswith" | "endswith" | "split" | "join", 1) => {
                        let a = self.expr(&args[0].e, s, stack)?;
                        Ok(self.prim(m, vec![rv, a]))
                    }
                    _ => no(&format!("method {m}")),
                }
            }
            Expr::Name(g, _) if g == "copy" && args.len() == 1 => self.expr(&args[0].e, s, stack),
            // a variant: its fields and `$v` (the evaluator's embedding)
            Expr::Name(g, _) if self.variants.contains_key(g) && !self.structs.contains_key(g) => {
                let fields = self.variants[g].clone();
                let mut out = vec![json!(["$v", c(json!(g))])];
                for (i, a) in args.iter().enumerate() {
                    let f = match &a.label {
                        Some(f) if fields.contains(f) => f.clone(),
                        None if i < fields.len() => fields[i].clone(),
                        _ => return no(&format!("variant {g} with these arguments")),
                    };
                    out.push(json!([f, self.expr(&a.e, s, stack)?]));
                }
                if out.len() != fields.len() + 1 {
                    return no(&format!("variant {g} with these arguments"));
                }
                Ok(json!({"t": "obj", "fields": out}))
            }
            Expr::Name(g, _) if self.structs.contains_key(g) => {
                let fields = self.structs[g].clone();
                if args.len() != fields.len()
                    || args
                        .iter()
                        .any(|a| a.label.as_ref().is_none_or(|x| !fields.contains(x)))
                {
                    return no(&format!("constructor {g} with these arguments"));
                }
                let mut out = vec![];
                for a in args {
                    out.push(json!([
                        a.label.clone().unwrap(),
                        self.expr(&a.e, s, stack)?
                    ]));
                }
                Ok(json!({"t": "obj", "fields": out}))
            }
            Expr::Name(g, _) if self.fns.contains_key(g) => {
                let d = self.fns[g];
                if matches!(d.sig.ret, Some(Ty::Union(..))) && !union_ok {
                    return no(&format!(
                        "a call of {g} (its result is T ! E) outside try, catch or return"
                    ));
                }
                if !mut_params(d).is_empty() {
                    return no(&format!(
                        "a call of {g} (it takes &mut) inside an expression"
                    ));
                }
                let names: Vec<String> = d
                    .sig
                    .params
                    .iter()
                    .filter_map(|p| {
                        if let Param::Named { name, .. } = p {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if names.len() != args.len()
                    || args
                        .iter()
                        .any(|a| a.label.as_ref().is_some_and(|x| !names.contains(x)))
                {
                    return no(&format!("call of {g} with these arguments"));
                }
                let mut fields = vec![];
                let mut used = HashSet::new();
                for (i, a) in args.iter().enumerate() {
                    let n = a.label.clone().unwrap_or_else(|| names[i].clone());
                    used.insert(n.clone());
                    fields.push(json!([n, self.expr(&a.e, s, stack)?]));
                }
                let call = self.callee_term(g, stack)?;
                if call["t"] == "rec" {
                    for f in &self.rec_bound {
                        fields.push(json!([f, self.proj(f)]));
                    }
                }
                Ok(Self::pipe(json!({"t": "obj", "fields": fields}), call))
            }
            _ => no("this call"),
        }
    }
}

fn stmt_name(s: &Stmt) -> &'static str {
    match s {
        Stmt::Match(..) => "match",
        Stmt::Send(..) => "send",
        Stmt::Emit(_) => "emit",
        Stmt::Fail(_) => "fail",
        Stmt::Spawn(_) => "spawn",
        Stmt::Select(..) => "select",
        Stmt::Arena(..) => "arena",
        Stmt::Device(..) => "device",
        Stmt::Yield(..) => "yield",
        _ => "this statement",
    }
}

fn expr_name(e: &Expr) -> &'static str {
    match e {
        Expr::Float(..) => "a float",
        Expr::SelfRef(_) => "self",
        Expr::ListComp(..) => "a list comprehension",
        Expr::Map(..) => "a map literal",
        Expr::ChanNew(..) => "a channel",
        Expr::Slice(..) => "a slice",
        Expr::Borrow(..) => "a mutable borrow",
        Expr::Recv(..) => "a receive",
        Expr::Catch(..) => "catch",
        Expr::Try(..) => "try",
        Expr::Lambda(..) => "a lambda",
        Expr::Index(..) => "a multi-index",
        _ => "this expression",
    }
}

/// The definition keys a term refers to.
fn refs(t: &Value) -> Vec<String> {
    let mut out = vec![];
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                if m.get("t").and_then(|x| x.as_str()) == Some("ref") {
                    if let Some(k) = m.get("def").and_then(|x| x.as_str()) {
                        out.push(k.to_string());
                    }
                    return;
                }
                m.values().for_each(|x| walk(x, out));
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    walk(t, &mut out);
    out
}

/// `$call[j]`: item j of a state-passing call's result.
fn at_call(j: usize, l: usize) -> Expr {
    Expr::Index(
        Box::new(Expr::Name("$call".into(), l)),
        vec![Expr::Int(j as i128, l)],
        l,
    )
}

fn r0(l: usize) -> Expr {
    at_call(0, l)
}
