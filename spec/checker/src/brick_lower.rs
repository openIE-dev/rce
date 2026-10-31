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
//! other overflow rules); floats are not lowered. Generic functions lower to
//! one term for every instance: the IR is untyped over JSON, so the term is
//! the monomorphization at every type at once, and vectors check it at i64.
//!
//! Anything else is not lowered; the error names the line and the construct. A lowering is
//! a claim, not a proof: `rce index vectors N` records the interpreter's
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
    /// fields with their declared types: structs, error payloads and
    /// variants, by name
    sfields: HashMap<String, Vec<(String, Ty)>>,
    /// type aliases without a predicate, resolved as the interpreter does
    aliases: HashMap<String, Ty>,
    /// declared types of the constants
    const_tys: HashMap<String, Ty>,
    /// the static types of the locals of the function being lowered
    /// (declared, or of the initializer); an int's type is its width tag
    vty: HashMap<String, Ty>,
    /// untyped locals (tag `lit`) that are also assigned a value of a
    /// narrow width somewhere: their tag depends on the path taken
    amb: HashSet<String>,
    /// the body of the function being lowered (for `name_assigns`)
    body_now: &'a [Stmt],
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
/// the callee's term); past this size the function is not lowered.
pub const MAX_TERM_BYTES: usize = 1 << 20;

fn non_i64_int(t: &Ty) -> Option<String> {
    // every int width lowers (see `arith_at`); floats do not
    let bad = ["f32", "f64"];
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

/// The variable a place (`x`, `x.f`, `x[i]`, nested) is rooted at.
fn place_root(e: &Expr) -> Option<&str> {
    match e {
        Expr::Name(x, _) => Some(x),
        Expr::Member(b, _, _) => place_root(b),
        Expr::Index(b, ix, _) if ix.len() == 1 => place_root(b),
        _ => None,
    }
}

fn c(v: Value) -> Value {
    json!({"t": "const", "value": v})
}

impl<'a> Lowerer<'a> {
    /// `m` is the linked module the function lives in (the items of the
    /// modules it imports included); `module` names it in node ids; `consts` are the
    /// literal values of its constants (evaluated by the interpreter).
    pub fn new(m: &'a Module, module: &str, consts: HashMap<String, Value>) -> Self {
        let mut fns = HashMap::new();
        let mut structs = HashMap::new();
        let mut sfields: HashMap<String, Vec<(String, Ty)>> = HashMap::new();
        let mut aliases = HashMap::new();
        let mut const_tys = HashMap::new();
        let mut variants = HashMap::new();
        for it in &m.items {
            match it {
                Item::Kind { variants: vs, .. } => {
                    for (n, fs) in vs {
                        variants.insert(n.clone(), fs.iter().map(|f| f.0.clone()).collect());
                        sfields.insert(n.clone(), fs.clone());
                    }
                }
                Item::Fn(d) => {
                    fns.insert(d.sig.name.clone(), d);
                }
                Item::TypeAlias { name, ty, pred: None, generics, .. } if generics.is_empty() => {
                    aliases.insert(name.clone(), ty.clone());
                }
                Item::Const { name, ty: Some(t), .. } => {
                    const_tys.insert(name.clone(), t.clone());
                }
                // an error value is a record of its payload (no `$v`)
                Item::Error { name, payload, .. } => {
                    structs.insert(name.clone(), payload.iter().map(|f| f.0.clone()).collect());
                    sfields.insert(name.clone(), payload.clone());
                }
                Item::Struct { name, fields, .. } => {
                    sfields.insert(
                        name.clone(),
                        fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
                    );
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
            sfields,
            aliases,
            const_tys,
            vty: HashMap::new(),
            amb: HashSet::new(),
            body_now: &[],
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
    /// The environment record over scope `s` built field by field: the
    /// form for updates that introduce a key the record does not have yet.
    fn set_new(&self, s: &[String], upd: &[(&str, Value)]) -> Value {
        let mut fields = vec![];
        for v in s {
            match upd.iter().find(|(k, _)| k == v) {
                Some((_, e)) => fields.push(json!([v, e])),
                None => fields.push(json!([v, self.proj(v)])),
            }
        }
        json!({"t": "obj", "fields": fields})
    }
    /// Every updated key must already be a field of the record.
    fn set(&self, s: &[String], upd: &[(&str, Value)]) -> Value {
        // updating variables already in scope: `set` of those fields on the
        // environment (its field set is unchanged), so a statement's term
        // does not grow with the number of variables in scope
        if !upd.is_empty() && upd.iter().all(|(k, _)| s.iter().any(|v| v == k)) {
            let mut t = json!({"t": "pass"});
            for (k, e) in upd {
                t = self.prim("set", vec![t, c(json!(k)), e.clone()]);
            }
            return t;
        }
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
        let r = self.lower_fn_uncached(name, stack).map(|t| {
            let nn = self.node_of(name);
            let saved = std::mem::replace(&mut self.node, nn);
            let t = self.share(t);
            self.node = saved;
            t
        });
        let r = r.and_then(|t| {
            let n = serde_json::to_string(&t)
                .map(|x| x.len())
                .unwrap_or(usize::MAX);
            if n > MAX_TERM_BYTES {
                Err(format!("its term is {n} bytes, over {MAX_TERM_BYTES}"))
            } else {
                Ok(t)
            }
        });
        // an error that names a function on the stack depends on the stack
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
            let nn = self.node_of(name);
            let saved_n = std::mem::replace(&mut self.node, nn);
            let bodies: Vec<Value> = bodies.into_iter().map(|b| self.share(b)).collect();
            self.node = saved_n;
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
        let saved_vty = std::mem::take(&mut self.vty);
        let saved_amb = std::mem::take(&mut self.amb);
        let saved_body = std::mem::replace(
            &mut self.body_now,
            match &d.body {
                Body::Block(ss) => &ss[..],
                _ => &[],
            },
        );
        for p in &d.sig.params {
            if let Param::Named { name: n, ty, .. } = p {
                self.vty.insert(n.clone(), unmut(ty).clone());
            }
        }
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
            // arguments are fitted to the parameter types (the call's `fit`)
            let init_fields: Vec<Value> = params
                .iter()
                .map(|p| json!([p, self.fit(self.proj(p), &self.vty[p].clone(), false)]))
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
                    let r = self.fit_ret(r, &d.sig.ret);
                    Ok(Self::pipe(init, out(self, r)))
                }
                Body::Block(ss) => {
                    let b = self.block(ss, &mut s, true, stack)?;
                    let r = out(self, self.fit_ret(self.proj("$ret"), &d.sig.ret));
                    Ok(Self::pipe(init, Self::pipe(b, r)))
                }
            }
        })();
        stack.pop();
        self.rec_bound = saved_b;
        self.node = saved.0;
        self.depth = saved.1;
        self.unsigned = saved_u;
        self.vty = saved_vty;
        self.amb = saved_amb;
        self.body_now = saved_body;
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
        match ty {
            Some(t) => {
                v = self.fit(v, t, false);
                self.vty.insert(x.to_string(), t.clone());
            }
            None => {
                let tag = self.itag(e);
                match self.sty(e) {
                    Some(t) => {
                        self.vty.insert(x.to_string(), t);
                    }
                    None if tag == ITag::Lit => {
                        self.vty.insert(x.to_string(), Ty::Path(vec!["lit".into()], vec![]));
                    }
                    None => {
                        self.vty.remove(x);
                    }
                }
                self.amb.remove(x);
                if tag == ITag::Amb {
                    self.amb.insert(x.to_string());
                }
                // an untyped local (tag lit) also assigned a narrow value
                // takes that value's tag (6.3): its width depends on the path
                if tag == ITag::Lit {
                    let mut asg = vec![];
                    name_assigns(self.body_now, &mut asg);
                    if asg.iter().any(|(n, r)| {
                        n == x && matches!(self.itag(r), ITag::W(w) if w != "i64") || n == x && self.itag(r) == ITag::Amb
                    }) {
                        self.amb.insert(x.to_string());
                    }
                }
            }
        }
        let had = s.iter().any(|v| v == x);
        self.bind(s, x, l)?;
        Ok(if had { self.set(s, &[(x, v)]) } else { self.set_new(s, &[(x, v)]) })
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
        // an untyped binding of the result takes the callee's result type
        let ret_ty = if d.sig.generics.is_empty() { d.sig.ret.clone() } else { None };
        let bind = match bind {
            Some(Stmt::Let(p, None, e, l2)) if ret_ty.is_some() => Some(Stmt::Let(p, ret_ty.clone(), e, l2)),
            Some(Stmt::Var(x, None, e, l2)) if ret_ty.is_some() => Some(Stmt::Var(x, ret_ty.clone(), e, l2)),
            b => b,
        };
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

    /// `place = v` as an update of the place's root variable: the root
    /// and its new value. `b[i] = v` is `b = set(b, i, v)` and `b.f = v` is
    /// `b = set(b, "f", v)` (a struct keeps its field set), applied down to
    /// the root. A field of type usize gets the guard a usize variable
    /// gets; a field whose name has another narrow type in some struct of
    /// the module is not lowered (its overflow rule is not the lowered one).
    fn place_update(
        &mut self,
        place: &Expr,
        v: Value,
        s: &mut Vec<String>,
        stack: &mut Vec<String>,
    ) -> Result<(String, Value), String> {
        match place {
            Expr::Name(x, l) => {
                if !s.contains(x) {
                    return Err(format!("line {l}: assignment to `{x}`"));
                }
                // `x = v` keeps x's width (6.3): fitted to x's type
                let v = match self.vty.get(x).cloned() {
                    Some(t) if iname(&t).is_some_and(|n| n != "lit") => self.fit(v, &t, true),
                    _ => v,
                };
                Ok((x.clone(), v))
            }
            Expr::Index(b, ix, l) if ix.len() == 1 => {
                if place_root(b).is_none() {
                    return Err(format!("line {l}: assignment to this place"));
                }
                let i = self.expr(&ix[0], s, stack)?;
                let v = match self.sty(b).and_then(|t| self.elem_ty(&t)) {
                    Some(t) => self.fit(v, &t, true),
                    None => v,
                };
                let cur = self.expr(b, s, stack)?;
                let nb = self.prim("set", vec![cur, i, v]);
                self.place_update(b, nb, s, stack)
            }
            Expr::Member(b, f, l) => {
                if place_root(b).is_none() {
                    return Err(format!("line {l}: assignment to this place"));
                }
                // the field keeps its width: fitted to its declared type
                let v = match self.sty(place) {
                    Some(t) => self.fit(v, &t, true),
                    None => {
                        let narrow = self.sfields.values().flatten().any(|(n, t)| {
                            n == f && iname(&self.resolve(t)).is_some_and(|w| w != "i64")
                        });
                        if narrow {
                            return Err(format!("line {l}: field {f} of a value whose struct is not known"));
                        }
                        v
                    }
                };
                let cur = self.expr(b, s, stack)?;
                let nb = self.prim("set", vec![cur, c(json!(f)), v]);
                self.place_update(b, nb, s, stack)
            }
            e => Err(format!("line {}: assignment to this place", e.line())),
        }
    }

    /// The expression has type usize, as far as the lowerer can see
    /// without types: a usize variable, `.len()`, a cast to usize, a call
    /// of a function whose result is usize, and arithmetic over those.
    fn is_unsigned(&self, e: &Expr) -> bool {
        match e {
            Expr::Name(x, _) => self.unsigned.contains(x),
            Expr::Cast(_, t, _) => is_usize(t),
            Expr::Call(f, args, _, _) => match &**f {
                Expr::Member(_, m, _) => (m == "len" || m == "len_bytes") && args.is_empty(),
                Expr::Name(g, _) => self
                    .fns
                    .get(g)
                    .and_then(|d| d.sig.ret.as_ref())
                    .is_some_and(is_usize),
                _ => false,
            },
            Expr::Bin(op, a, b, _) if matches!(op.as_str(), "+" | "-" | "*" | "/" | "%") => {
                self.is_unsigned(a) || self.is_unsigned(b)
            }
            _ => false,
        }
    }

    fn unsigned_guard(&self, v: Value) -> Value {
        let neg = self.prim("lt", vec![v.clone(), c(json!(0))]);
        self.cond(neg, self.err("trap: overflow: negative usize"), v)
    }

    fn stmt(&mut self, st: &Stmt, s: &mut Vec<String>, tail: bool, stack: &mut Vec<String>) -> L {
        // `let _ = e` evaluates `e` and drops it: the same term as the statement `e`
        if let Stmt::Let(Pat::Wild, None, e, _) = st {
            return self.stmt(&Stmt::Expr(e.clone()), s, tail, stack);
        }
        // `let x = p.pop()`: x takes the last item (or none), then p pops
        if let Some((peek, pop)) = pop_split(st) {
            let a = self.stmt(&peek, s, false, stack)?;
            let b = self.stmt(&pop, s, false, stack)?;
            return Ok(Self::seq(vec![a, b]));
        }
        let hoisted = match self.hoist_stmt(st)? {
            Some(h) => Some(h),
            None => self.hoist_try(st)?,
        };
        if let Some((lets, rest)) = hoisted {
            // a hoisted try may leave the function: what follows it is
            // skipped then, as the statements of a block are
            let mut ts = vec![];
            for (i, lt) in lets.iter().chain(std::iter::once(&rest)).enumerate() {
                let t = self.stmt(lt, s, i == lets.len() && tail, stack)?;
                ts.push(if i == 0 { t } else { self.cond(self.skipping(), json!({"t": "pass"}), t) });
            }
            return Ok(Self::seq(ts));
        }
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
                let mut t = self.set_new(&s2, &[(&tmp, v)]);
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
                    self.arith_at(&op[..op.len() - 1], place, rhs, cur, r, place.line())?
                };
                let (x, nv) = self.place_update(place, v, s, stack)?;
                Ok(self.set(s, &[(&x, nv)]))
            }
            Stmt::Expr(e) => {
                if let Expr::Call(f, args, l, _) = e
                    && let Expr::Member(r, m, _) = &**f
                    && place_root(r).is_some_and(|x| s.contains(&x.to_string()))
                {
                    let cur = self.expr(r, s, stack)?;
                    match (m.as_str(), args.len()) {
                        ("push", 1) => {
                            let v = self.expr(&args[0].e, s, stack)?;
                            let nv = self.prim("append", vec![cur, v]);
                            let (x, nr) = self.place_update(r, nv, s, stack)?;
                            return Ok(self.set(s, &[(&x, nr)]));
                        }
                        // pop: the last item (none on an empty list, which
                        // stays empty), 6.4
                        ("pop", 0) => {
                            let len = self.prim("len", vec![cur.clone()]);
                            let short = self.prim(
                                "slice",
                                vec![
                                    cur.clone(),
                                    c(json!(0)),
                                    self.arith("sub", len.clone(), c(json!(1))),
                                ],
                            );
                            let (x, nr) = self.place_update(r, short, s, stack)?;
                            if !tail {
                                return Ok(self.set(s, &[(&x, nr)]));
                            }
                            let gt = self.prim("gt", vec![len.clone(), c(json!(0))]);
                            let last = self.prim("at", vec![cur.clone(), self.arith("sub", len, c(json!(1)))]);
                            let v = self.cond(gt, last, c(Value::Null));
                            return Ok(self.set(s, &[("$ret", v), (&x, nr), ("$done", c(json!(true)))]));
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
                    Ok(Self::pipe(self.set_new(&s2, &[("$_", v)]), self.set_new(s, &[])))
                }
            }
            Stmt::Return(Some(e), _) => {
                self.allow_union = self.fails && matches!(e, Expr::Call(..));
                let v = self.expr(e, s, stack)?;
                Ok(self.set(s, &[("$ret", v), ("$done", c(json!(true)))]))
            }
            Stmt::Return(None, _) => Ok(self.set(s, &[("$ret", c(Value::Null)), ("$done", c(json!(true)))])),
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
                // the loop variable's type: the range bounds' width, or
                // the element type of the collection
                let xt = match it {
                    Expr::Bin(op, a, b, _) if op == ".." || op == "..=" => {
                        match join_tag(self.itag(a), self.itag(b)) {
                            ITag::W(w) => Some(Ty::Path(vec![w.into()], vec![])),
                            ITag::Lit => Some(Ty::Path(vec!["lit".into()], vec![])),
                            _ => None,
                        }
                    }
                    e => self.sty(e).and_then(|t| self.elem_ty(&t)),
                };
                match xt {
                    Some(t) => {
                        self.vty.insert(x.clone(), t);
                    }
                    None => {
                        self.vty.remove(&x);
                    }
                }
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
        let init = self.set_new(&sl, &pre_ref);
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
        let mut head = self.set_new(&sl, &[("$cnt", c(json!(false))), (&nv, n1)]);
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
            head = Self::pipe(head, self.set_new(&sb, &[(x, item), (&iv, i1)]));
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
                self.arith_at(op, a, b, x, y, l)
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
                match self.itag(x) {
                    ITag::Amb => Err(format!("line {l}: the width of this negation depends on the path taken")),
                    ITag::W(w) if w == "u64" || w == "usize" => Ok(self.arith("usub", c(json!(0)), v)),
                    ITag::W(w) if w != "i64" => Ok(self.fit(
                        self.prim("neg", vec![v]),
                        &Ty::Path(vec![w.into()], vec![]),
                        true,
                    )),
                    _ => Ok(self.prim("neg", vec![v])),
                }
            }
            Expr::BitNot(x, _) => {
                let v = self.expr(x, s, stack)?;
                // unsigned: max - v; signed: two's complement
                match self.itag(x) {
                    ITag::Amb => Err(format!("line {l}: the width of this ~ depends on the path taken")),
                    ITag::W(w) if w == "u64" || w == "usize" => {
                        Ok(self.arith("usub", int_const(u64::MAX as i128), v))
                    }
                    ITag::W(w) if w.starts_with('u') => {
                        Ok(self.arith("sub", int_const(int_range(w).1), v))
                    }
                    _ => Ok(self.prim("bnot", vec![v])),
                }
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
            // `x[a:b]` (6.3): bounds from the end when negative, then
            // 0 <= a <= b <= len or a trap; a text slices its UTF-8 bytes
            // on char boundaries
            Expr::Slice(x, a, b, _) => {
                let op = match self.sty(x).map(|t| self.resolve(&t)) {
                    Some(Ty::Path(p, _)) if p.len() == 1 && p[0] == "text" => "bslice",
                    Some(t) if self.elem_ty(&t).is_some() => "slice",
                    _ => return no("a slice of a value of unknown type"),
                };
                let mut fs = vec![json!(["$sx", self.expr(x, s, stack)?])];
                if let Some(a) = a {
                    fs.push(json!(["$sa", self.expr(a, s, stack)?]));
                }
                if let Some(b) = b {
                    fs.push(json!(["$sb", self.expr(b, s, stack)?]));
                }
                let n = self.prim("size", vec![self.proj("$sx")]);
                let norm = |me: &Self, k: Value| {
                    let kk = me.cond(
                        me.prim("lt", vec![k.clone(), c(json!(0))]),
                        me.arith("add", n.clone(), k.clone()),
                        k,
                    );
                    // the check sees the slice too: its bound is the size of $sx
                    Self::pipe(
                        json!({"t": "obj", "fields": [["$sx", me.proj("$sx")], ["$sk", kk]]}),
                        me.cond(
                            me.prim("or", vec![
                                me.prim("lt", vec![me.proj("$sk"), c(json!(0))]),
                                me.prim("gt", vec![me.proj("$sk"), n.clone()]),
                            ]),
                            me.err("trap: slice index out of range"),
                            me.proj("$sk"),
                        ),
                    )
                };
                let lo = if a.is_some() { norm(self, self.proj("$sa")) } else { c(json!(0)) };
                let hi = if b.is_some() { norm(self, self.proj("$sb")) } else { n.clone() };
                let bounds = json!({"t": "obj", "fields": [["$sx", self.proj("$sx")], ["lo", lo], ["hi", hi]]});
                let body = self.cond(
                    self.prim("gt", vec![self.proj("lo"), self.proj("hi")]),
                    self.err("trap: slice bounds reversed"),
                    self.prim(op, vec![self.proj("$sx"), self.proj("lo"), self.proj("hi")]),
                );
                Ok(Self::pipe(json!({"t": "obj", "fields": fs}), Self::pipe(bounds, body)))
            }
            Expr::TupleMember(b, k, _) => {
                let bv = self.expr(b, s, stack)?;
                Ok(self.prim("get", vec![bv, c(json!(k))]))
            }
            // `e as T` for an int T: the value checked against T's range
            // (6.3; a bool is 0 or 1); floats are not lowered
            Expr::Cast(x, t, _) => match iname(&self.resolve(t)) {
                Some(_) => {
                    let mut v = self.expr(x, s, stack)?;
                    if self.sty(x).is_some_and(|t| matches!(&t, Ty::Path(p, a) if a.is_empty() && p.len() == 1 && p[0] == "bool")) {
                        v = self.cond(v, c(json!(1)), c(json!(0)));
                    } else if self.sty(x).is_some_and(|t| non_i64_int(&t).is_some()) {
                        return no("this cast");
                    }
                    Ok(self.fit(v, t, true))
                }
                None => no("this cast"),
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
                    ("len" | "len_bytes", 0) => Ok(self.prim("size", vec![rv])),
                    // options embed as the value or null
                    ("issome", 0) => Ok(self.prim("ne", vec![rv, c(Value::Null)])),
                    ("isnone", 0) => Ok(self.prim("eq", vec![rv, c(Value::Null)])),
                    // `o.or(d)`: both evaluated (arguments are), d when none
                    ("or", 1) => {
                        let dv = self.expr(&args[0].e, s, stack)?;
                        let both = json!({"t": "obj", "fields": [["o", rv], ["d", dv]]});
                        let pick = self.cond(
                            self.prim("eq", vec![self.proj("o"), c(Value::Null)]),
                            self.proj("d"),
                            self.proj("o"),
                        );
                        Ok(Self::pipe(both, pick))
                    }
                    // the language's strip trims Unicode White_Space
                    ("strip", 0) => Ok(self.prim("ustrip", vec![rv])),
                    ("isdigit", 0) => Ok(self.prim("isdigit", vec![rv])),
                    // `xs.map(fn(x: T) => e)`: e on each item, the outer
                    // variables read from the map's `env`
                    ("map", 1) => {
                        let Expr::Lambda(ps, body, _) = &args[0].e else {
                            return no("map of a value that is not a lambda");
                        };
                        let [Param::Named { name: x, ty, .. }] = &ps[..] else {
                            return no("a lambda with these parameters");
                        };
                        if s.contains(x) {
                            return no(&format!("a lambda parameter {x} that shadows a variable"));
                        }
                        let mut fields: Vec<Value> = s
                            .iter()
                            .map(|v| json!([v, Self::pipe(self.proj("env"), self.proj(v))]))
                            .collect();
                        fields.push(json!([x, self.proj("x")]));
                        let mut s2: Vec<String> = s.to_vec();
                        s2.push(x.clone());
                        let had = self.vty.insert(x.clone(), ty.clone());
                        let b = self.expr(body, &s2, stack);
                        match had {
                            Some(t) => {
                                self.vty.insert(x.clone(), t);
                            }
                            None => {
                                self.vty.remove(x);
                            }
                        }
                        let f = Self::pipe(json!({"t": "obj", "fields": fields}), b?);
                        Ok(json!({"t": "map", "node": self.n(), "over": rv, "f": f}))
                    }
                    ("chars" | "tostr" | "toint" | "upper" | "lower", 0) => {
                        Ok(self.prim(m, vec![rv]))
                    }
                    ("contains" | "startswith" | "endswith" | "split", 1) => {
                        let a = self.expr(&args[0].e, s, stack)?;
                        Ok(self.prim(m, vec![rv, a]))
                    }
                    // `sep.join(xs)`: the receiver is the separator, the
                    // prim takes the list first
                    ("join", 1) => {
                        let a = self.expr(&args[0].e, s, stack)?;
                        Ok(self.prim("join", vec![a, rv]))
                    }
                    _ => no(&format!("method {m}")),
                }
            }
            // `some(x)` embeds as x (none is null), `Box(x)` as x
            Expr::Name(g, _) if (g == "copy" || g == "some" || g == "Box") && args.len() == 1 => {
                self.expr(&args[0].e, s, stack)
            }
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
                    let mut v = self.expr(&a.e, s, stack)?;
                    if let Some((_, t)) = self.sfields.get(g).and_then(|fs| fs.iter().find(|x| x.0 == f)) {
                        v = self.fit(v, &t.clone(), false);
                    }
                    out.push(json!([f, v]));
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
                let ftys = self.sfields.get(g).cloned().unwrap_or_default();
                for a in args {
                    let f = a.label.clone().unwrap();
                    let mut v = self.expr(&a.e, s, stack)?;
                    // fields are fitted to their declared types
                    if let Some((_, t)) = ftys.iter().find(|x| x.0 == f) {
                        v = self.fit(v, t, false);
                    }
                    out.push(json!([f, v]));
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

// ---------------------------------------------------------------------
// Integer widths (6.3). The interpreter tags every int with its width:
// a literal is `lit`; a value takes its declared type where it is fitted
// (parameters, typed locals, struct fields, results, casts); an operation
// takes the width of its operands (`lit` yields to the other); `x = v`
// keeps x's width; a result outside the width traps. The lowerer computes
// the same tag statically and lowers each operation at its width: i64
// (and lit) as before, u64 and usize by the checked u64 operators, the
// widths of 32 bits and less in i64 (exact there) then range-checked.
// A tag that depends on the path taken is not lowered.

/// The width tag of an int expression, as far as the lowerer sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ITag {
    Lit,
    W(&'static str),
    /// `lit` on one path and a narrow width on another
    Amb,
    Unk,
}

const INT_NAMES: [&str; 10] = [
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize", "lit",
];

fn iname(t: &Ty) -> Option<&'static str> {
    match t {
        Ty::Path(p, a) if p.len() == 1 && a.is_empty() => {
            INT_NAMES.iter().find(|x| **x == p[0]).copied()
        }
        Ty::Ref(_, x) | Ty::Region(x, _) => iname(x),
        _ => None,
    }
}

fn int_range(n: &str) -> (i128, i128) {
    match n {
        "i8" => (-128, 127),
        "i16" => (-32768, 32767),
        "i32" => (i32::MIN as i128, i32::MAX as i128),
        "u8" => (0, 255),
        "u16" => (0, 65535),
        "u32" => (0, u32::MAX as i128),
        "u64" | "usize" => (0, u64::MAX as i128),
        _ => (i64::MIN as i128, i64::MAX as i128),
    }
}

fn int_const(x: i128) -> Value {
    match i64::try_from(x) {
        Ok(i) => c(json!(i)),
        Err(_) => c(json!(x as u64)),
    }
}

fn join_tag(a: ITag, b: ITag) -> ITag {
    use ITag::*;
    match (a, b) {
        (Lit, x) | (x, Lit) => x,
        (W(x), _) => W(x),
        (_, W(y)) => W(y),
        (Unk, _) | (_, Unk) => Unk,
        (Amb, Amb) => Amb,
    }
}

/// Every `x = e` to a plain name in a body, nested blocks included.
fn name_assigns(ss: &[Stmt], out: &mut Vec<(String, Expr)>) {
    for st in ss {
        match st {
            Stmt::Assign(op, Expr::Name(x, _), e) if op == "=" => out.push((x.clone(), e.clone())),
            Stmt::If(arms, els) => {
                for (_, b) in arms {
                    name_assigns(b, out);
                }
                if let Some(b) = els {
                    name_assigns(b, out);
                }
            }
            Stmt::While(_, _, b) | Stmt::Loop(_, b) | Stmt::For(_, _, b, _) | Stmt::Comptime(b) => {
                name_assigns(b, out)
            }
            Stmt::Match(_, arms, _) => {
                for a in arms {
                    name_assigns(&a.body, out);
                }
            }
            _ => {}
        }
    }
}

impl<'a> Lowerer<'a> {
    fn resolve(&self, t: &Ty) -> Ty {
        let mut t = t.clone();
        for _ in 0..8 {
            t = match t {
                Ty::Ref(_, x) | Ty::Region(x, _) => *x,
                Ty::Path(ref p, ref a) if p.len() == 1 && a.is_empty() && self.aliases.contains_key(&p[0]) => {
                    self.aliases[&p[0]].clone()
                }
                _ => return t,
            };
        }
        t
    }

    /// The element type of a list, array or slice type.
    fn elem_ty(&self, t: &Ty) -> Option<Ty> {
        match self.resolve(t) {
            Ty::Path(p, a) if p.len() == 1 && p[0] == "List" && a.len() == 1 => Some(a[0].clone()),
            Ty::Slice(x) | Ty::Array(x, _) => Some(*x),
            _ => None,
        }
    }

    /// The static type of an expression, where the lowerer can tell it.
    fn sty(&self, e: &Expr) -> Option<Ty> {
        match e {
            Expr::Name(x, _) => self
                .vty
                .get(x)
                .or_else(|| self.const_tys.get(x))
                .map(|t| self.resolve(t)),
            Expr::Member(b, f, _) => match self.resolve(&self.sty(b)?) {
                Ty::Path(p, a) if p.len() == 1 && a.is_empty() => self
                    .sfields
                    .get(&p[0])?
                    .iter()
                    .find(|x| &x.0 == f)
                    .map(|x| self.resolve(&x.1)),
                _ => None,
            },
            Expr::Index(b, ix, _) if ix.len() == 1 => {
                self.elem_ty(&self.sty(b)?).map(|t| self.resolve(&t))
            }
            Expr::Cast(_, t, _) => Some(self.resolve(t)),
            Expr::Borrow(_, x, _) => self.sty(x),
            // an int operation: its width tag
            Expr::Bin(op, ..)
                if matches!(op.as_str(), "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>") =>
            {
                match self.itag(e) {
                    ITag::W(w) => Some(Ty::Path(vec![w.into()], vec![])),
                    _ => None,
                }
            }
            Expr::Neg(..) | Expr::BitNot(..) | Expr::If(..) => match self.itag(e) {
                ITag::W(w) => Some(Ty::Path(vec![w.into()], vec![])),
                _ => None,
            },
            Expr::Call(f, args, _, _) => match &**f {
                Expr::Member(_, m, _) if (m == "len" || m == "len_bytes") && args.is_empty() => {
                    Some(Ty::Path(vec!["usize".into()], vec![]))
                }
                Expr::Name(g, _) if self.sfields.contains_key(g) && self.structs.contains_key(g) => {
                    Some(Ty::Path(vec![g.clone()], vec![]))
                }
                Expr::Name(g, _) => {
                    let d = self.fns.get(g)?;
                    if !d.sig.generics.is_empty() {
                        return None;
                    }
                    match d.sig.ret.as_ref()? {
                        Ty::Union(ok, _) => Some(self.resolve(ok)),
                        t => Some(self.resolve(t)),
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn itag(&self, e: &Expr) -> ITag {
        match e {
            Expr::Int(..) => ITag::Lit,
            Expr::Neg(x, _) | Expr::BitNot(x, _) => self.itag(x),
            Expr::Name(x, _) if self.amb.contains(x) => ITag::Amb,
            Expr::Bin(op, a, b, _)
                if matches!(op.as_str(), "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>") =>
            {
                join_tag(self.itag(a), self.itag(b))
            }
            Expr::If(_, a, b, _) => {
                let (x, y) = (self.itag(a), self.itag(b));
                if x == y {
                    x
                } else if matches!((x, y), (ITag::Lit, ITag::W("i64")) | (ITag::W("i64"), ITag::Lit)) {
                    // lit and i64 have the same range: no difference
                    ITag::W("i64")
                } else if matches!((x, y), (ITag::Unk, _) | (_, ITag::Unk)) {
                    ITag::Unk
                } else {
                    ITag::Amb
                }
            }
            _ => match self.sty(e).as_ref().and_then(iname) {
                Some("lit") => ITag::Lit,
                Some(n) => ITag::W(n),
                None => ITag::Unk,
            },
        }
    }

    /// A term true on its input when the input fits `t` (the interpreter's
    /// `fit`: ints in range, recursively through lists, options, tuples
    /// and struct fields); None when every value of the embedding fits.
    fn fits(&self, t: &Ty, depth: usize) -> Option<Value> {
        if depth > 4 {
            return None;
        }
        let t = self.resolve(t);
        let pass = json!({"t": "pass"});
        if let Some(n) = iname(&t) {
            if n == "i64" || n == "lit" {
                // a u64 above i64::MAX does not fit i64
                return Some(self.prim("le", vec![pass, int_const(i64::MAX as i128)]));
            }
            let (lo, hi) = int_range(n);
            return Some(self.and(
                self.prim("ge", vec![pass.clone(), int_const(lo)]),
                self.prim("le", vec![pass, int_const(hi)]),
            ));
        }
        match &t {
            Ty::Opt(x) => {
                let k = self.fits(x, depth + 1)?;
                // none fits; the inner check runs only on a present value
                // (or is strict and would test the fields of none)
                Some(self.cond(
                    self.prim("eq", vec![pass, c(Value::Null)]),
                    c(json!(true)),
                    k,
                ))
            }
            Ty::Tuple(ts) => {
                let mut acc: Option<Value> = None;
                for (i, x) in ts.iter().enumerate() {
                    if let Some(k) = self.fits(x, depth + 1) {
                        let ki = Self::pipe(self.prim("get", vec![pass.clone(), c(json!(i))]), k);
                        acc = Some(match acc {
                            None => ki,
                            Some(a) => self.and(a, ki),
                        });
                    }
                }
                acc
            }
            Ty::Path(p, a) if p.len() == 1 && a.is_empty() && self.sfields.contains_key(&p[0]) => {
                let mut acc: Option<Value> = None;
                for (f, x) in self.sfields[&p[0]].clone() {
                    if let Some(k) = self.fits(&x, depth + 1) {
                        let kf = Self::pipe(self.proj(&f), k);
                        acc = Some(match acc {
                            None => kf,
                            Some(a) => self.and(a, kf),
                        });
                    }
                }
                acc
            }
            _ => {
                let x = self.elem_ty(&t)?;
                let k = self.fits(&x, depth + 1)?;
                let each = json!({"t": "map", "node": self.n(), "over": pass,
                    "f": Self::pipe(self.proj("x"), k)});
                Some(self.prim("all", vec![each]))
            }
        }
    }

    /// `v` fitted to `t`: v, or the trap of a value outside t. Only the
    /// narrow widths are checked where the source has the declared type
    /// anyway (`strict` false); casts and assignments check every width.
    fn fit(&self, v: Value, t: &Ty, strict: bool) -> Value {
        let t = self.resolve(t);
        if !strict && matches!(iname(&t), Some("i64") | Some("lit")) {
            return v;
        }
        match self.fits(&t, 0) {
            None => v,
            Some(k) => Self::pipe(
                v,
                self.cond(k, json!({"t": "pass"}), self.err("trap: overflow: value outside its type")),
            ),
        }
    }

    /// A function result fitted to its declared type (the ok side of
    /// `T ! E`; an error value passes).
    fn fit_ret(&self, v: Value, ret: &Option<Ty>) -> Value {
        match ret {
            Some(Ty::Union(ok, _)) => match self.fits(ok, 0) {
                None => v,
                Some(k) => {
                    let is_err = self.prim("has", vec![json!({"t": "pass"}), c(json!("$err"))]);
                    let ok_k = self.cond(k, json!({"t": "pass"}), self.err("trap: overflow: value outside its type"));
                    Self::pipe(v, self.cond(is_err, json!({"t": "pass"}), ok_k))
                }
            },
            Some(t) => self.fit(v, t, false),
            None => v,
        }
    }

    /// `x op y` for the arithmetic and bit operators at the width of the
    /// operands' tag.
    fn arith_at(&self, op: &str, a: &Expr, b: &Expr, x: Value, y: Value, l: usize) -> L {
        let tag = join_tag(self.itag(a), self.itag(b));
        let arith_op = matches!(op, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>");
        if !arith_op {
            return self.binop(op, x, y, l);
        }
        let w = match tag {
            ITag::Lit | ITag::Unk | ITag::W("i64") => return self.binop(op, x, y, l),
            ITag::Amb => {
                return Err(format!(
                    "line {l}: the width of this integer operation depends on the path taken"
                ))
            }
            ITag::W(w) => w,
        };
        if op == "+" && (self.sty(a).is_some_and(|t| iname(&t).is_none())) {
            // texts and lists
            return self.binop(op, x, y, l);
        }
        let (lo, hi) = int_range(w);
        let bits: i64 = match w {
            "i8" | "u8" => 8,
            "i16" | "u16" => 16,
            "i32" | "u32" => 32,
            _ => 64,
        };
        let wide = w == "u64" || w == "usize";
        let shift_ok = |me: &Self, y: Value, t: Value| {
            let bad = me.prim(
                "or",
                vec![
                    me.prim("lt", vec![y.clone(), c(json!(0))]),
                    me.prim("ge", vec![y, c(json!(bits))]),
                ],
            );
            me.cond(bad, me.err("trap: shift amount out of range"), t)
        };
        if wide {
            return Ok(match op {
                "+" => self.arith("uadd", x, y),
                "-" => self.arith("usub", x, y),
                "*" => self.arith("umul", x, y),
                "/" => self.arith("udiv", x, y),
                "%" => self.arith("urem", x, y),
                // a negative literal operand can leave the u64 range
                "&" | "|" | "^" => self.fit(
                    self.binop(op, x, y, l)?,
                    &Ty::Path(vec![w.into()], vec![]),
                    true,
                ),
                "<<" => self.prim("ushl", vec![x, y]),
                _ => self.prim("ushr", vec![x, y]),
            });
        }
        let in_range = |me: &Self, v: Value| me.fit(v, &Ty::Path(vec![w.into()], vec![]), true);
        Ok(match op {
            "&" | "|" | "^" => in_range(self, self.binop(op, x, y, l)?),
            "<<" => {
                // (x << y) mod 2^bits, read back as signed for i8..i32
                let mask = (1i64 << bits) - 1;
                let r = self.prim("band", vec![self.prim("shl", vec![x, y.clone()]), c(json!(mask))]);
                let r = if lo < 0 {
                    Self::pipe(
                        r,
                        self.cond(
                            self.prim("ge", vec![json!({"t": "pass"}), c(json!(1i64 << (bits - 1)))]),
                            self.arith("sub", json!({"t": "pass"}), c(json!(1i64 << bits))),
                            json!({"t": "pass"}),
                        ),
                    )
                } else {
                    r
                };
                shift_ok(self, y, r)
            }
            ">>" => {
                let r = self.prim("shr", vec![x, y.clone()]);
                shift_ok(self, y, r)
            }
            _ => {
                let _ = hi;
                in_range(self, self.binop(op, x, y, l)?)
            }
        })
    }
}

// ---------------------------------------------------------------------
// A call with `&mut` arguments inside an expression is evaluated before
// the statement, into `let $h<k> = call`, then the statement uses `$h<k>`.
// This keeps the interpreter's order exactly when nothing the statement
// evaluates besides the calls mentions a variable the calls mutate (the
// calls stay in their own left-to-right, innermost-first order), and no
// call sits where it might not run (an `if` branch, the right side of
// `and`/`or`, a `catch` handler, `try ... else`, a lambda).

/// The expression variants whose children are all evaluated, in order,
/// before the node itself; the indices of children that may not run.
fn hoist_shape(variant: &str) -> Option<&'static [usize]> {
    Some(match variant {
        "Call" | "Not" | "Neg" | "BitNot" | "Member" | "Index" | "TupleMember" | "Cast"
        | "List" | "Tuple" | "Borrow" => &[],
        "If" => &[1, 2],
        "Catch" => &[2],
        "Try" => &[1],
        _ => return None,
    })
}

impl<'a> Lowerer<'a> {
    fn json_mut_call(&self, v: &Value) -> Option<String> {
        let call = v.get("Call")?.as_array()?;
        let g = call.first()?.get("Name")?.as_array()?.first()?.as_str()?;
        let d = self.fns.get(g)?;
        (!mut_params(d).is_empty()).then(|| g.to_string())
    }

    fn json_has_mut_call(&self, v: &Value) -> bool {
        if self.json_mut_call(v).is_some() {
            return true;
        }
        match v {
            Value::Array(xs) => xs.iter().any(|x| self.json_has_mut_call(x)),
            Value::Object(m) => m.values().any(|x| self.json_has_mut_call(x)),
            _ => false,
        }
    }

    /// Replaces the &mut calls under `v` (an expression, serialized) by
    /// `$h<k>` names, collecting the calls in evaluation order.
    fn hoist_json(&mut self, v: &mut Value, top: bool, out: &mut Vec<Value>) -> Result<(), String> {
        let Some((variant, body)) = v.as_object_mut().and_then(|m| {
            if m.len() == 1 {
                m.iter_mut().next().map(|(k, b)| (k.clone(), b))
            } else {
                None
            }
        }) else {
            return Ok(());
        };
        if variant == "Bin" {
            let op = body.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string();
            let lazy: &[usize] = if op == "and" || op == "or" { &[2] } else { &[] };
            let kids = body.as_array_mut().unwrap();
            for (i, k) in kids.iter_mut().enumerate() {
                if lazy.contains(&i) {
                    if self.json_has_mut_call(k) {
                        return Err("a call with &mut on the right of and/or".into());
                    }
                } else {
                    self.hoist_json(k, false, out)?;
                }
            }
        } else if let Some(lazy) = hoist_shape(&variant) {
            if let Some(kids) = body.as_array_mut() {
                for (i, k) in kids.iter_mut().enumerate() {
                    if lazy.contains(&i) {
                        if self.json_has_mut_call(k) {
                            return Err("a call with &mut where it may not run".into());
                        }
                    } else {
                        self.hoist_kids(k, out)?;
                    }
                }
            }
        } else if self.json_has_mut_call(body) {
            return Err(format!("a call with &mut inside {variant}"));
        }
        if !top && self.json_mut_call(v).is_some() {
            self.tmps += 1;
            let name = format!("$h{}", self.tmps);
            let call = std::mem::replace(v, json!({"Name": [name, 0]}));
            out.push(json!([name, call]));
        }
        Ok(())
    }

    /// The children of a node: expressions, or lists of them (call
    /// arguments are `{label, e}`).
    fn hoist_kids(&mut self, k: &mut Value, out: &mut Vec<Value>) -> Result<(), String> {
        match k {
            Value::Array(xs) => {
                for x in xs.iter_mut() {
                    self.hoist_kids(x, out)?;
                }
                Ok(())
            }
            Value::Object(m) if m.contains_key("e") && m.contains_key("label") => {
                self.hoist_json(m.get_mut("e").unwrap(), false, out)
            }
            Value::Object(_) => self.hoist_json(k, false, out),
            _ => Ok(()),
        }
    }

    /// `st` with its nested &mut calls taken out: the `let`s to run first
    /// and the rewritten statement; None when there is nothing to take out.
    fn hoist_stmt(&mut self, st: &Stmt) -> Result<Option<(Vec<Stmt>, Stmt)>, String> {
        let mut sv = serde_json::to_value(st).map_err(|e| e.to_string())?;
        if !self.json_has_mut_call(&sv) {
            return Ok(None);
        }
        let (variant, body) = match sv.as_object_mut().and_then(|m| m.iter_mut().next()) {
            Some((k, b)) => (k.clone(), b),
            None => return Ok(None),
        };
        // which child is the expression, and whether its top may stay a
        // call (the statement forms `mut_call_stmt` lowers)
        let (ix, top_ok) = match variant.as_str() {
            "Expr" => (None, true),
            "Let" | "Var" => (Some(2), true),
            "Assign" => (Some(2), true),
            "Return" => (Some(0), false),
            "Match" => (Some(0), false),
            "If" => (None, false),
            _ => return Ok(None),
        };
        let mut out = vec![];
        let r = match (variant.as_str(), ix) {
            ("Expr", _) => self.hoist_json(body, top_ok, &mut out),
            ("If", _) => {
                // only the first condition runs unconditionally
                let arms = &mut body[0];
                match arms.get_mut(0).and_then(|a| a.get_mut(0)) {
                    Some(c0) => self.hoist_json(c0, false, &mut out),
                    None => Ok(()),
                }
            }
            (_, Some(i)) => match body.get_mut(i) {
                Some(e) if !e.is_null() => self.hoist_json(e, top_ok, &mut out),
                _ => Ok(()),
            },
            _ => Ok(()),
        };
        let l = st_line(st);
        if let Err(m) = r {
            return Err(format!("line {l}: {m}"));
        }
        if out.is_empty() {
            return Ok(None);
        }
        // nothing else in the statement may mention what the calls mutate
        let mut roots = vec![];
        for o in &out {
            let call: Expr = serde_json::from_value(o[1].clone()).map_err(|e| e.to_string())?;
            if let Expr::Call(f, args, _, _) = &call
                && let Expr::Name(g, _) = &**f
            {
                let d = self.fns[g.as_str()];
                let muts = mut_params(d);
                let names: Vec<String> = d.sig.params.iter().filter_map(|p| match p {
                    Param::Named { name, .. } => Some(name.clone()),
                    _ => None,
                }).collect();
                for (i, a) in args.iter().enumerate() {
                    let n = a.label.clone().or_else(|| names.get(i).cloned()).unwrap_or_default();
                    if muts.contains(&n) {
                        let p = match &a.e {
                            Expr::Borrow(true, p, _) => &**p,
                            e => e,
                        };
                        if let Some(r) = place_root(p) {
                            roots.push(r.to_string());
                        }
                    }
                }
            }
        }
        let mut mentioned = vec![];
        json_names(&sv, &mut mentioned);
        if let Some(r) = roots.iter().find(|r| mentioned.contains(r)) {
            return Err(format!(
                "line {l}: a call with &mut {r} in a statement that also reads {r}"
            ));
        }
        let rest: Stmt = serde_json::from_value(sv).map_err(|e| e.to_string())?;
        let lets = out
            .into_iter()
            .map(|o| {
                let call: Expr = serde_json::from_value(o[1].clone()).unwrap();
                Stmt::Var(o[0].as_str().unwrap().to_string(), None, call, l)
            })
            .collect();
        Ok(Some((lets, rest)))
    }
}

fn json_names(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            if let Some(Value::Array(a)) = m.get("Name")
                && let Some(Value::String(n)) = a.first()
            {
                out.push(n.clone());
            }
            for x in m.values() {
                json_names(x, out);
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| json_names(x, out)),
        _ => {}
    }
}

fn st_line(st: &Stmt) -> usize {
    match st {
        Stmt::Let(_, _, _, l) | Stmt::Var(_, _, _, l) | Stmt::Return(_, l) | Stmt::Match(_, _, l) => *l,
        Stmt::Expr(e) | Stmt::Assign(_, _, e) => e.line(),
        Stmt::If(arms, _) => arms.first().map(|a| a.0.line()).unwrap_or(0),
        _ => 0,
    }
}

// `try x` inside an expression is taken out the same way, into
// `let $h<k> = try x` before the statement, when what the statement
// evaluates besides the `try`s cannot fail (names, literals, fields,
// constructors, `+` of texts or lists): then leaving early with the error
// or running on is decided at the same point as in the interpreter.

impl<'a> Lowerer<'a> {
    /// The expression cannot trap or leave the function (try nodes are
    /// checked by the caller).
    fn total(&self, e: &Expr) -> bool {
        match e {
            Expr::Int(..) | Expr::Bool(..) | Expr::Str(..) | Expr::None(_) | Expr::Unit(_) => true,
            Expr::Name(..) => true,
            Expr::Try(..) => true,
            Expr::Member(b, _, _) | Expr::Borrow(_, b, _) => self.total(b),
            Expr::List(xs, _) | Expr::Tuple(xs, _) => xs.iter().all(|x| self.total(x)),
            Expr::Bin(op, a, b, _) if op == "+" => {
                let seq = |t: Option<Ty>| {
                    t.map(|t| self.resolve(&t)).is_some_and(|t| {
                        matches!(&t, Ty::Path(p, _) if p.len() == 1 && (p[0] == "text" || p[0] == "List"))
                    })
                };
                (seq(self.sty(a)) || seq(self.sty(b))) && self.total(a) && self.total(b)
            }
            Expr::Call(f, args, _, _) => match &**f {
                Expr::Name(g, _)
                    if g == "some" || g == "copy" || g == "Box" || self.structs.contains_key(g) || self.variants.contains_key(g) =>
                {
                    // a constructor fits its fields: total when none is narrow
                    let narrow = self.sfields.get(g).is_some_and(|fs| fs.iter().any(|(_, t)| self.fits(t, 0).is_some() && iname(&self.resolve(t)).is_some_and(|w| w != "i64")));
                    !narrow && args.iter().all(|a| self.total(&a.e))
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn take_trys(&mut self, e: &mut Expr, out: &mut Vec<(String, Expr)>) {
        match e {
            Expr::Try(..) => {
                self.tmps += 1;
                let name = format!("$h{}", self.tmps);
                let t = std::mem::replace(e, Expr::Name(name.clone(), 0));
                out.push((name, t));
            }
            Expr::Member(b, _, _) | Expr::Borrow(_, b, _) => self.take_trys(b, out),
            Expr::List(xs, _) | Expr::Tuple(xs, _) => xs.iter_mut().for_each(|x| self.take_trys(x, out)),
            Expr::Bin(_, a, b, _) => {
                self.take_trys(a, out);
                self.take_trys(b, out);
            }
            Expr::Call(_, args, _, _) => args.iter_mut().for_each(|a| self.take_trys(&mut a.e, out)),
            _ => {}
        }
    }

    fn hoist_try(&mut self, st: &Stmt) -> Result<Option<(Vec<Stmt>, Stmt)>, String> {
        let e = match st {
            Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) | Stmt::Assign(_, _, e) => e,
            Stmt::Expr(e) | Stmt::Return(Some(e), _) => e,
            _ => return Ok(None),
        };
        // try_stmt takes a try at the top
        if matches!(e, Expr::Try(..)) || !serde_json::to_string(e).unwrap_or_default().contains("\"Try\"") {
            return Ok(None);
        }
        if !self.total(e) {
            return Ok(None);
        }
        if let Stmt::Assign(_, place, _) = st
            && !self.total(place)
        {
            return Ok(None);
        }
        let mut e2 = e.clone();
        let mut out = vec![];
        self.take_trys(&mut e2, &mut out);
        if out.is_empty() {
            return Ok(None);
        }
        let l = st_line(st);
        let rest = match st {
            Stmt::Let(p, t, _, l) => Stmt::Let(p.clone(), t.clone(), e2, *l),
            Stmt::Var(x, t, _, l) => Stmt::Var(x.clone(), t.clone(), e2, *l),
            Stmt::Assign(op, p, _) => Stmt::Assign(op.clone(), p.clone(), e2),
            Stmt::Return(_, l) => Stmt::Return(Some(e2), *l),
            _ => Stmt::Expr(e2),
        };
        let lets = out.into_iter().map(|(n, t)| Stmt::Var(n, None, t, l)).collect();
        Ok(Some((lets, rest)))
    }
}

/// `let/var x = p.pop()` and `y = p.pop()` (y not p's root) as the binding
/// of `if p.len() > 0 then some(p[p.len() - 1]) else none`, then `p.pop()`.
fn pop_split(st: &Stmt) -> Option<(Stmt, Stmt)> {
    let e = match st {
        Stmt::Let(_, _, e, _) | Stmt::Var(_, _, e, _) => e,
        Stmt::Assign(op, _, e) if op == "=" => e,
        _ => return None,
    };
    let Expr::Call(f, args, l, b) = e else { return None };
    let Expr::Member(p, m, _) = &**f else { return None };
    if m != "pop" || !args.is_empty() {
        return None;
    }
    let root = place_root(p)?;
    if let Stmt::Assign(_, place, _) = st
        && place_root(place).is_none_or(|r| r == root)
    {
        return None;
    }
    let l = *l;
    let len = Expr::Call(Box::new(Expr::Member(p.clone(), "len".into(), l)), vec![], l, false);
    let last = Expr::Index(
        p.clone(),
        vec![Expr::Bin("-".into(), Box::new(len.clone()), Box::new(Expr::Int(1, l)), l)],
        l,
    );
    let peek = Expr::If(
        Box::new(Expr::Bin(">".into(), Box::new(len), Box::new(Expr::Int(0, l)), l)),
        Box::new(Expr::Call(Box::new(Expr::Name("some".into(), l)), vec![Arg { label: None, e: last }], l, false)),
        Box::new(Expr::None(l)),
        l,
    );
    let bind = match st {
        Stmt::Let(q, t, _, l2) => Stmt::Let(q.clone(), t.clone(), peek, *l2),
        Stmt::Var(x, t, _, l2) => Stmt::Var(x.clone(), t.clone(), peek, *l2),
        Stmt::Assign(op, q, _) => Stmt::Assign(op.clone(), q.clone(), peek),
        _ => return None,
    };
    Some((bind, Stmt::Expr(Expr::Call(f.clone(), vec![], l, *b))))
}

// ---------------------------------------------------------------------
// Sharing (format 2): a subterm that occurs more than once in a lowered
// term is a pure function of its input, so it moves to the definition
// table once and each occurrence becomes a `ref` to it. Store-passing
// lowering repeats the same environment rebuilds and guards for every
// statement; sharing them keeps a term proportional to its distinct
// parts instead of capping it. A subterm holding `rec` stays in place
// (it names the enclosing fix).

const SHARE_MIN_BYTES: usize = 160;

struct Scan {
    count: usize,
    size: usize,
    rec: bool,
}

fn is_term(v: &Value) -> bool {
    v.get("t").is_some_and(|t| t.is_string())
}

impl<'a> Lowerer<'a> {
    fn share(&mut self, t: Value) -> Value {
        let mut tab = HashMap::new();
        let (t, ..) = self.sh(t, None, &mut tab);
        if !tab.values().any(|s| s.count > 1 && s.size >= SHARE_MIN_BYTES && !s.rec) {
            return t;
        }
        let mut unused = HashMap::new();
        self.sh(t, Some(&tab), &mut unused).0
    }

    /// One bottom-up pass: the node's hash, serialized size and whether it
    /// holds `rec`, computed on the original; without `tab` the term nodes
    /// are counted into `acc`, with it the shared ones become refs.
    fn sh(
        &mut self,
        v: Value,
        tab: Option<&HashMap<u64, Scan>>,
        acc: &mut HashMap<u64, Scan>,
    ) -> (Value, u64, usize, bool) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let term = is_term(&v);
        let (nv, size, rec) = match v {
            Value::Object(m) => {
                let mut size = 2;
                let mut rec = m.get("t").and_then(|t| t.as_str()) == Some("rec");
                let mut out = serde_json::Map::new();
                for (k, x) in m {
                    k.hash(&mut h);
                    if k == "value" && term {
                        let st = x.to_string();
                        st.hash(&mut h);
                        size += st.len() + k.len() + 4;
                        out.insert(k, x);
                        continue;
                    }
                    let (y, hx, sx, rx) = self.sh(x, tab, acc);
                    hx.hash(&mut h);
                    size += sx + k.len() + 4;
                    rec |= rx;
                    out.insert(k, y);
                }
                (Value::Object(out), size, rec)
            }
            Value::Array(xs) => {
                let mut size = 2;
                let mut rec = false;
                let mut out = Vec::with_capacity(xs.len());
                for x in xs {
                    let (y, hx, sx, rx) = self.sh(x, tab, acc);
                    hx.hash(&mut h);
                    size += sx + 1;
                    rec |= rx;
                    out.push(y);
                }
                (Value::Array(out), size, rec)
            }
            x => {
                let st = x.to_string();
                st.hash(&mut h);
                let n = st.len();
                (x, n, false)
            }
        };
        let hv = h.finish();
        if !term {
            return (nv, hv, size, rec);
        }
        match tab {
            None => {
                acc.entry(hv).or_insert(Scan { count: 0, size, rec }).count += 1;
                (nv, hv, size, rec)
            }
            Some(t) => match t.get(&hv) {
                Some(sc) if sc.count > 1 && sc.size >= SHARE_MIN_BYTES && !sc.rec => {
                    let k = crate::brick::digest(&nv);
                    self.defs.entry(k.clone()).or_insert(nv);
                    (json!({"t": "ref", "node": self.n(), "def": k}), hv, size, rec)
                }
                _ => (nv, hv, size, rec),
            },
        }
    }
}
