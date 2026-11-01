//! Laws (spec 6.11): properties of every reachable state, or of every step
//! taken from one, decided exactly by exhaustive exploration (method D1).
//!
//! D1 explores reach(P) breadth first from s₀ over the complete alphabet
//! (every instance of every declared event, in canonical order), to at most
//! `LAW_FUEL` states, and evaluates every law at each new state (`always`)
//! or at each step (`every step`). A violation is exact at any depth: the
//! sequence that reaches it exists, and it is printed as an `example` line.
//! A closed exploration with no violation proves the law for every event
//! sequence of every length (T63, `spec/lean/RCE/Law.lean`). Anything else
//! is `undecided`, which requires change (W-LAW-1) unless a person attests
//! the law with an attestation bound to the hashes of the definitions the
//! law covers (`apidiff::law_cover`).
//!
//! The other methods of `spec/proposals/laws.md` (an inductive invariant, a
//! Lean proof) are proposed and not implemented.
use crate::ast::*;
use crate::interp::{Interp, Region, V, state_json, why};
use std::collections::{HashMap, VecDeque};

/// The state bound of D1, a constant of the language (6.11), beside
/// LOOP_LIMIT and VALUE_LIMIT: admission depends only on the program text.
pub const LAW_FUEL: usize = 1_000_000;

/// The most instances one event contributes to the alphabet (6.11), as in
/// `rce dim` (10.11). An event with more is incomplete.
pub const ALPHABET_MAX: usize = 4096;

/// The integer types whose values D1 lists in full (at most 4,096 values).
fn small_int(t: &str) -> Option<(i128, i128)> {
    match t {
        "u8" => Some((0, 255)),
        "i8" => Some((-128, 127)),
        _ => None,
    }
}

/// `old(f)` in the outcome of an `every step` law is held as the name
/// `old(f)`, which no source name can be, bound to field f before the step.
pub fn hold_old(e: &mut Expr) {
    if let Expr::Call(f, a, l, _) = e
        && matches!(&**f, Expr::Name(n, _) if n == "old")
        && a.len() == 1
        && a[0].label.is_none()
        && let Expr::Name(x, _) = &a[0].e
    {
        *e = Expr::Name(format!("old({x})"), *l);
        return;
    }
    walk_mut(e, &mut hold_old);
}

/// Apply `f` to each direct subexpression of `e`.
fn walk_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    use Expr as E;
    match e {
        E::Name(..) | E::SelfRef(_) | E::Int(..) | E::Float(..) | E::Str(..) | E::Bool(..) | E::None(_) | E::Unit(_) => {}
        E::Tuple(v, _) | E::List(v, _) => v.iter_mut().for_each(f),
        E::ListRepeat(a, b, _) => {
            f(a);
            f(b);
        }
        E::ListComp(a, _, b, c, _) => {
            f(a);
            f(b);
            if let Some(c) = c {
                f(c);
            }
        }
        E::Map(v, _) => {
            for (k, x) in v {
                f(k);
                f(x);
            }
        }
        E::ChanNew(_, c, _) => {
            if let Some(c) = c {
                f(c);
            }
        }
        E::Call(g, a, _, _) => {
            f(g);
            for x in a {
                f(&mut x.e);
            }
        }
        E::Index(x, ix, _) => {
            f(x);
            ix.iter_mut().for_each(f);
        }
        E::Slice(x, lo, hi, _) => {
            f(x);
            if let Some(lo) = lo {
                f(lo);
            }
            if let Some(hi) = hi {
                f(hi);
            }
        }
        E::Member(x, _, _) | E::TupleMember(x, _, _) | E::Neg(x, _) | E::BitNot(x, _) | E::Borrow(_, x, _) | E::Recv(x, _) | E::Comptime(x, _) | E::Cast(x, _, _) | E::Not(x, _) | E::Lambda(_, x, _) => f(x),
        E::Bin(_, a, b, _) | E::Catch(a, _, b, _) => {
            f(a);
            f(b);
        }
        E::Try(a, b, _) => {
            f(a);
            if let Some(b) = b {
                f(b);
            }
        }
        E::If(c, a, b, _) => {
            f(c);
            f(a);
            f(b);
        }
    }
}

/// Every law of a module, top level and inside intents, in text order.
pub fn laws(m: &Module) -> Vec<&Law> {
    m.items.iter().filter_map(|it| if let Item::Law(l) = it { Some(l) } else { None }).collect()
}

/// One event instance of the alphabet: the event, its arguments as written
/// in an `example` line, and its value.
struct Inst {
    name: String,
    args: Vec<Arg>,
    v: V,
}

/// The listable values of a payload field type (6.11, complete alphabet),
/// in canonical order, or the reason the type is not listable.
fn values(t: &Ty, m: &Module, depth: usize) -> Result<Vec<Expr>, String> {
    if depth > 8 {
        return Err("a type nested too deep to list".into());
    }
    match t {
        Ty::Unit => Ok(vec![Expr::Unit(0)]),
        Ty::Opt(inner) => {
            let mut v = vec![Expr::None(0)];
            for e in values(inner, m, depth + 1)? {
                v.push(Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e }], 0, false));
            }
            Ok(v)
        }
        Ty::Path(p, args) if args.is_empty() && p.len() == 1 => {
            let n = p[0].as_str();
            if n == "bool" {
                return Ok(vec![Expr::Bool(false, 0), Expr::Bool(true, 0)]);
            }
            if let Some((lo, hi)) = small_int(n) {
                return Ok((lo..=hi).map(|i| Expr::Int(i, 0)).collect());
            }
            for it in &m.items {
                match it {
                    // a refined type lists its base type's values; a value
                    // outside the predicate traps at construction and is no event
                    Item::TypeAlias { name, generics, ty, .. } if name == n && generics.is_empty() => return values(ty, m, depth + 1),
                    Item::Kind { name, generics, variants, .. } if name == n && generics.is_empty() => {
                        let mut out = vec![];
                        for (vn, fs) in variants {
                            if fs.is_empty() {
                                out.push(Expr::Name(vn.clone(), 0));
                                continue;
                            }
                            for row in product(fs, m, depth)? {
                                out.push(Expr::Call(Box::new(Expr::Name(vn.clone(), 0)), row, 0, false));
                            }
                        }
                        if out.len() > ALPHABET_MAX {
                            return Err(format!("kind {n} has more than {ALPHABET_MAX} values"));
                        }
                        return Ok(out);
                    }
                    _ => {}
                }
            }
            Err(crate::unparse::ty(t))
        }
        _ => Err(crate::unparse::ty(t)),
    }
}

/// Every labelled argument row of a payload, in canonical order.
fn product(fs: &[(String, Ty)], m: &Module, depth: usize) -> Result<Vec<Vec<Arg>>, String> {
    let mut rows: Vec<Vec<Arg>> = vec![vec![]];
    for (f, t) in fs {
        let col = values(t, m, depth + 1)?;
        if rows.len().saturating_mul(col.len()) > ALPHABET_MAX {
            return Err(format!("more than {ALPHABET_MAX} instances"));
        }
        let mut next = vec![];
        for r in &rows {
            for e in &col {
                let mut r2 = r.clone();
                r2.push(Arg { label: Some(f.clone()), e: e.clone() });
                next.push(r2);
            }
        }
        rows = next;
    }
    Ok(rows)
}

/// The alphabet D1 explores: every instance of every event whose payload is
/// listable, in declaration order; and, for each event that is not, why
/// (`Deposit.amount (u64)`).
fn alphabet(m: &Module, it: &mut Interp) -> (Vec<Inst>, Vec<String>) {
    let mut out = vec![];
    let mut incomplete = vec![];
    for i in &m.items {
        let Item::Event { name, payload, .. } = i else { continue };
        let mut rows = vec![vec![]];
        let mut bad = None;
        if !payload.is_empty() {
            match product(payload, m, 0) {
                Ok(r) => rows = r,
                Err(why) => {
                    // name the first field that is not listable
                    let f = payload.iter().find(|(_, t)| values(t, m, 0).is_err());
                    bad = Some(match f {
                        Some((f, t)) => format!("{name}.{f} ({})", crate::unparse::ty(t)),
                        None => format!("{name} ({why})"),
                    });
                }
            }
        }
        if let Some(b) = bad {
            incomplete.push(b);
            continue;
        }
        for args in rows {
            it.scopes = vec![HashMap::new()];
            it.refuel();
            // a value outside a refined type traps at construction: no event
            if let Ok(Some(v)) = it.construct(name, &args) {
                out.push(Inst { name: name.clone(), args, v });
            }
        }
    }
    (out, incomplete)
}

/// The verdict of one law (6.11).
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// closed exploration, no violation: `states` states, `steps` steps
    Holds { states: usize, steps: usize },
    /// a violation after `after` events, the `example` line that reaches it,
    /// what failed, and how many states were explored
    Fails { after: usize, example: String, what: String, explored: usize },
    /// no violation, and the exploration did not decide the law
    Undecided { why: String },
    /// undecided, and attested by `who` on `date` with the current cover
    Attested { who: String, date: String, why: String },
}

/// One decided law.
#[derive(Debug, Clone)]
pub struct Decided {
    pub name: String,
    pub line: usize,
    pub verdict: Verdict,
    /// instances in the alphabet, and whether it is complete
    pub alphabet: usize,
    pub complete: bool,
    /// the cover hash an attestation must carry (`apidiff::law_cover`)
    pub cover: String,
    /// an attestation that names an older cover: lapsed
    pub lapsed: Option<String>,
    /// an `attest` line that is not `WHO;YYYY-MM-DD;COVER`
    pub malformed: Vec<String>,
}

/// An attestation `WHO;YYYY-MM-DD;COVER` (the `ui.rules` form with the cover
/// hash added), split, or `None` when malformed.
pub fn attestation(a: &str) -> Option<(String, String, String)> {
    let p: Vec<&str> = a.split(';').map(str::trim).collect();
    if p.len() != 3 || p[0].is_empty() {
        return None;
    }
    let d: Vec<&str> = p[1].split('-').collect();
    let date_ok = d.len() == 3 && [4, 2, 2].iter().zip(&d).all(|(n, x)| x.len() == *n && x.bytes().all(|b| b.is_ascii_digit()));
    let hash_ok = p[2].len() == 64 && p[2].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !date_ok || !hash_ok {
        return None;
    }
    Some((p[0].to_string(), p[1].to_string(), p[2].to_string()))
}

/// The fields of a state as one scope, every region flattened, as for the
/// `-> e` of an example (6.9).
fn scope_of(s: &[Region]) -> HashMap<String, V> {
    s.iter().flat_map(|r| r.1.iter().map(|(n, v, _)| (n.clone(), v.clone()))).collect()
}

/// A value written as an expression an `example` line can state, when it
/// has one.
fn lit(v: &V) -> Option<Expr> {
    Some(match v {
        V::Int(i, _) => Expr::Int(*i, 0),
        V::Bool(b) => Expr::Bool(*b, 0),
        V::Text(s) => Expr::Str(s.to_string(), 0),
        V::Unit => Expr::Unit(0),
        V::Opt(None) => Expr::None(0),
        V::Opt(Some(x)) => Expr::Call(Box::new(Expr::Name("some".into(), 0)), vec![Arg { label: None, e: lit(x)? }], 0, false),
        V::Rec(n, fs) if fs.is_empty() => Expr::Name(n.clone(), 0),
        V::Rec(n, fs) => Expr::Call(
            Box::new(Expr::Name(n.clone(), 0)),
            fs.iter().map(|(f, x)| Some(Arg { label: Some(f.clone()), e: lit(x)? })).collect::<Option<Vec<_>>>()?,
            0,
            false,
        ),
        V::List(xs) => Expr::List(xs.iter().map(lit).collect::<Option<Vec<_>>>()?, 0),
        _ => return None,
    })
}

/// The outcome of an `example` line that states the state `s` as it is: a
/// single field by its value, else the conjunction of every field that has
/// a literal (`true` when none has).
fn state_outcome(s: &[Region]) -> Expr {
    let fs: Vec<(&String, &V)> = s.iter().flat_map(|r| r.1.iter().map(|(n, v, _)| (n, v))).collect();
    if fs.len() == 1
        && let Some(e) = lit(fs[0].1)
    {
        return e;
    }
    // written as text (`x == 1 and won == false`): `==` binds tighter than
    // `and` (3.8), so the line needs no parentheses; the expression is only
    // ever printed
    let eqs: Vec<String> = fs.iter().filter_map(|(n, v)| lit(v).map(|e| format!("{n} == {}", crate::unparse::expr(&e)))).collect();
    if eqs.is_empty() { Expr::Bool(true, 0) } else { Expr::Name(eqs.join(" and "), 0) }
}

/// What one law says about one state or step: `None` when it holds,
/// otherwise what failed and the outcome of the `example` line that states
/// what the program does there.
type Check = Option<(String, Option<String>, Option<Expr>)>;

fn check_always(it: &mut Interp, p: &Expr, s: &[Region]) -> Check {
    it.scopes = vec![scope_of(s)];
    it.refuel();
    match it.eval(p).and_then(|v| it.truthy(v)) {
        Ok(true) => None,
        Ok(false) => {
            let fs: Vec<&(String, V, Ty)> = s.iter().flat_map(|r| r.1.iter()).collect();
            // `-> not p` holds now; a single field compares by value (6.9)
            let out = match (&fs[..], p) {
                ([(_, v @ V::Bool(_), _)], _) => lit(v).unwrap_or(Expr::Bool(true, 0)),
                (_, Expr::Not(x, _)) => (**x).clone(),
                _ => Expr::Not(Box::new(p.clone()), 0),
            };
            Some(("the claim is false".into(), None, Some(out)))
        }
        Err(f) => Some((format!("evaluating the claim: {}", why(f)), None, Some(state_outcome(s)))),
    }
}

#[allow(clippy::too_many_arguments)]
fn check_every(
    it: &mut Interp,
    pat: &Option<Pat>,
    guard: &Option<Expr>,
    outcome: &LawOutcome,
    s: &[Region],
    ev: &V,
    r: &Result<(Vec<Region>, Vec<V>), String>,
) -> Check {
    it.scopes = vec![scope_of(s), HashMap::new()];
    it.refuel();
    if let Some(p) = pat {
        match it.pmatch(p, ev) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(f) => return Some((format!("matching the pattern: {}", why(f)), None, None)),
        }
    }
    let binds = it.scopes.last().cloned().unwrap_or_default();
    if let Some(g) = guard {
        match it.eval(g).and_then(|v| it.truthy(v)) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(f) => return Some((format!("evaluating the guard: {}", why(f)), None, None)),
        }
    }
    // what the program does at this step, as an example states it
    let now: (Option<String>, Option<Expr>) = match r {
        Ok((s2, _)) => (None, Some(state_outcome(s2))),
        // the reason as `Example::kept` holds it (`Quit`, `trap(…)`)
        Err(k) => (Some(k.clone()), None),
    };
    let bad = |what: String| Some((what, now.0.clone(), now.1.clone()));
    match (outcome, r) {
        (LawOutcome::Kept(None), Err(_)) => None,
        (LawOutcome::Kept(None), Ok(_)) => bad("the event applies".into()),
        (LawOutcome::Kept(Some(x)), Err(k)) if k == x => None,
        (LawOutcome::Kept(Some(_)), Err(k)) => bad(format!("the event is kept: {k}")),
        (LawOutcome::Kept(Some(_)), Ok(_)) => bad("the event applies".into()),
        (LawOutcome::Applied, Ok(_)) => None,
        (LawOutcome::Applied, Err(k)) => bad(format!("the event is kept: {k}")),
        (LawOutcome::Expr(o), _) => {
            // a kept step leaves the state unchanged (T8)
            let post: &[Region] = match r {
                Ok((s2, _)) => s2,
                Err(_) => s,
            };
            let mut sc = scope_of(post);
            for (n, v) in scope_of(s) {
                sc.insert(format!("old({n})"), v);
            }
            it.scopes = vec![sc, binds];
            it.refuel();
            match it.eval(o).and_then(|v| it.truthy(v)) {
                Ok(true) => None,
                Ok(false) => bad("the outcome is false after the step".into()),
                Err(f) => bad(format!("evaluating the outcome: {}", why(f))),
            }
        }
    }
}

/// Decide every law of a type-checked module by D1 (6.11).
pub fn decide(m: &Module) -> Vec<Decided> {
    decide_with(m, LAW_FUEL)
}

/// `decide` with another state bound (tests; admission uses `LAW_FUEL`).
pub fn decide_with(m: &Module, fuel: usize) -> Vec<Decided> {
    let ls = laws(m);
    if ls.is_empty() {
        return vec![];
    }
    let mut it = Interp::new(m);
    let (alpha, incomplete) = alphabet(m, &mut it);
    let k = alpha.len();
    let complete = incomplete.is_empty();
    // per law: the first violation (path end, what, outcome)
    struct Found {
        node: usize,
        ev: Option<usize>,
        what: String,
        kept: Option<String>,
        value: Option<Expr>,
    }
    let mut found: Vec<Option<Found>> = ls.iter().map(|_| None).collect();
    let mut parent: Vec<Option<(usize, usize)>> = vec![];
    let mut depth: Vec<usize> = vec![];
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(usize, Vec<Region>)> = VecDeque::new();
    let mut bound = false;
    let mut steps = 0usize;
    let mut init_err = None;
    it.refuel();
    match it.initial_state() {
        Ok(s0) => {
            index.insert(state_json(&s0, m), 0);
            parent.push(None);
            depth.push(0);
            for (j, l) in ls.iter().enumerate() {
                if let LawClaim::Always(p) = &l.claim
                    && let Some((what, kept, value)) = check_always(&mut it, p, &s0)
                {
                    found[j] = Some(Found { node: 0, ev: None, what, kept, value });
                }
            }
            queue.push_back((0, s0));
        }
        Err(f) => init_err = Some(why(f)),
    }
    while let Some((i, s)) = queue.pop_front() {
        if found.iter().all(Option::is_some) {
            break;
        }
        for (e, inst) in alpha.iter().enumerate() {
            it.scopes = vec![HashMap::new()];
            it.refuel();
            let r = it.delta(&s, &inst.v);
            steps += 1;
            for (j, l) in ls.iter().enumerate() {
                if found[j].is_some() {
                    continue;
                }
                if let LawClaim::Every { pat, guard, outcome } = &l.claim
                    && let Some((what, kept, value)) = check_every(&mut it, pat, guard, outcome, &s, &inst.v, &r)
                {
                    found[j] = Some(Found { node: i, ev: Some(e), what, kept, value });
                }
            }
            let Ok((s2, _)) = r else { continue }; // T8: a kept step leaves the state unchanged
            let key = state_json(&s2, m);
            if index.contains_key(&key) {
                continue;
            }
            if index.len() >= fuel {
                bound = true;
                continue;
            }
            let j2 = index.len();
            index.insert(key, j2);
            parent.push(Some((i, e)));
            depth.push(depth[i] + 1);
            for (j, l) in ls.iter().enumerate() {
                if found[j].is_none()
                    && let LawClaim::Always(p) = &l.claim
                    && let Some((what, kept, value)) = check_always(&mut it, p, &s2)
                {
                    found[j] = Some(Found { node: j2, ev: None, what, kept, value });
                }
            }
            queue.push_back((j2, s2));
        }
    }
    let closed = queue.is_empty() && !bound;
    let path = |mut n: usize| {
        let mut v = vec![];
        while let Some((p, e)) = parent[n] {
            v.push(e);
            n = p;
        }
        v.reverse();
        v
    };
    let states = index.len();
    let mut out = vec![];
    for (j, l) in ls.iter().enumerate() {
        let cover = crate::apidiff::law_cover(m, l);
        let mut malformed = vec![];
        let mut valid = vec![];
        for a in &l.attest {
            match attestation(a) {
                Some(x) => valid.push(x),
                None => malformed.push(a.clone()),
            }
        }
        let verdict = if let Some(e) = &init_err {
            Verdict::Undecided { why: format!("the initial state traps: {e}") }
        } else if let Some(f) = &found[j] {
            let mut evs = path(f.node);
            if let Some(e) = f.ev {
                evs.push(e);
            }
            let x = Example {
                events: evs.iter().map(|&e| (alpha[e].name.clone(), alpha[e].args.clone())).collect(),
                kept: f.kept.clone(),
                value: f.value.clone(),
                line: 0,
            };
            Verdict::Fails { after: evs.len(), example: crate::unparse::example(&x), what: f.what.clone(), explored: states }
        } else if closed && complete {
            Verdict::Holds { states, steps }
        } else {
            let mut why = vec![];
            if !complete {
                why.push(format!("alphabet incomplete: {}", incomplete.join(", ")));
            }
            if bound {
                why.push(format!("bound reached at {} states (LAW_FUEL)", group(fuel)));
            }
            Verdict::Undecided { why: why.join("; ") }
        };
        let mut lapsed = None;
        let verdict = match verdict {
            // an attestation never turns a `fails` into anything else
            Verdict::Undecided { why } => match valid.iter().find(|(_, _, c)| *c == cover) {
                Some((who, date, _)) => Verdict::Attested { who: who.clone(), date: date.clone(), why },
                None => {
                    if let Some((who, date, _)) = valid.first() {
                        lapsed = Some(format!("{who} on {date}"));
                    }
                    Verdict::Undecided { why }
                }
            },
            v => v,
        };
        out.push(Decided { name: l.name.clone(), line: l.line, verdict, alphabet: k, complete, cover, lapsed, malformed });
    }
    out
}

/// 1000000 as `1,000,000`.
pub fn group(n: usize) -> String {
    let s = n.to_string();
    let mut o = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            o.push(',');
        }
        o.push(c);
    }
    o
}

fn alpha_text(d: &Decided) -> String {
    format!("alphabet {} {}", d.alphabet, if d.complete { "complete" } else { "incomplete" })
}

/// The W-LAW-1 reports of a module (7, 6.11): one per law that fails, is
/// undecided and not attested, or has a malformed attestation. A failing
/// law's report ends with the counterexample on its own line, indented to
/// paste into an `intent`.
pub fn gate(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    for d in decide(m) {
        for a in &d.malformed {
            errs.push(format!("line {}: W-LAW-1 law {}: the attestation {} is not WHO;YYYY-MM-DD;COVER", d.line, d.name, crate::unparse::text(a)));
        }
        match &d.verdict {
            Verdict::Holds { .. } | Verdict::Attested { .. } => {}
            Verdict::Fails { after, example, what, explored } => errs.push(format!(
                "line {}: W-LAW-1 law {} fails {} (exhaustive, {}, {} states explored): {what}\n    {example}",
                d.line,
                d.name,
                if *after == 0 { "at the initial state".to_string() } else { format!("after {after} event{}", if *after == 1 { "" } else { "s" }) },
                alpha_text(&d),
                group(*explored),
            )),
            Verdict::Undecided { why } => errs.push(format!(
                "line {}: W-LAW-1 law {} undecided (exhaustive): {why}{}; a person who decides that it holds attests it with: attest \"WHO;YYYY-MM-DD;{}\"",
                d.line,
                d.name,
                d.lapsed.as_ref().map(|w| format!("; the attestation by {w} lapsed: a definition it covers changed")).unwrap_or_default(),
                d.cover
            )),
        }
    }
    errs
}

/// The report of `rce law FILE` (6.11): one line per law, with the verdict,
/// the method and the counts; a failing law's counterexample follows it.
pub fn report(ds: &[Decided]) -> String {
    let w = ds.iter().map(|d| d.name.len()).max().unwrap_or(0);
    let mut o = String::new();
    for d in ds {
        let (v, rest) = match &d.verdict {
            Verdict::Holds { states, steps } => (
                "holds",
                format!("exhaustive  {} states, {} steps, alphabet {} (complete), closed", group(*states), group(*steps), d.alphabet),
            ),
            Verdict::Fails { after, explored, what, .. } => (
                "fails",
                format!("exhaustive  after {after} event{}; {} states explored; {what}", if *after == 1 { "" } else { "s" }, group(*explored)),
            ),
            Verdict::Undecided { why } => (
                "undecided",
                format!(
                    "exhaustive  {why}{}; attest with \"WHO;YYYY-MM-DD;{}\"",
                    d.lapsed.as_ref().map(|w| format!("; the attestation by {w} lapsed")).unwrap_or_default(),
                    d.cover
                ),
            ),
            Verdict::Attested { who, date, why } => ("attested", format!("attestation  by {who} on {date} (exhaustive: {why})")),
        };
        o.push_str(&format!("law {:w$}  {v:9}  {rest}\n", d.name));
        if let Verdict::Fails { example, .. } = &d.verdict {
            o.push_str(&format!("    {example}\n"));
        }
        for a in &d.malformed {
            o.push_str(&format!("    the attestation {} is not WHO;YYYY-MM-DD;COVER\n", crate::unparse::text(a)));
        }
    }
    o
}

/// The exit status of `rce law` (10.1): 0 when every law holds or is
/// attested; 3 when the only obstacle is the bound; 1 otherwise.
pub fn status(ds: &[Decided]) -> i32 {
    let mut s = 0;
    for d in ds {
        if !d.malformed.is_empty() {
            return 1;
        }
        match &d.verdict {
            Verdict::Holds { .. } | Verdict::Attested { .. } => {}
            Verdict::Fails { .. } => return 1,
            Verdict::Undecided { .. } if !d.complete => return 1,
            Verdict::Undecided { .. } => s = 3,
        }
    }
    s
}

/// The law rows of `rce change OLD NEW` (6.11): each law of OLD decided on
/// NEW (CC-L1), and each law added, removed or edited (CC-L2). The second
/// value is the reason the change is a meaning change, when it is one: a law
/// of OLD that fails on NEW, or a law removed or edited. A law added to NEW
/// is listed and is no meaning change by itself (an erratum when δ and λ are
/// unchanged).
pub fn change_rows(old: &Module, new: &Module) -> (Vec<String>, Option<String>) {
    let (lo, ln) = (laws(old), laws(new));
    let mut rows = vec![];
    let mut why = None;
    if lo.is_empty() && ln.is_empty() {
        return (rows, why);
    }
    // OLD's laws, as written, over NEW's definitions
    let mut graft = new.clone();
    graft.items.retain(|i| !matches!(i, Item::Law(_)));
    graft.items.extend(lo.iter().map(|l| Item::Law(Law { in_intent: false, ..(*l).clone() })));
    let on_new = decide(&graft);
    for (l, d) in lo.iter().zip(&on_new) {
        let edit = match ln.iter().find(|x| x.name == l.name) {
            None => "removed",
            Some(x) if crate::unparse::law_claim(&x.claim) != crate::unparse::law_claim(&l.claim) => "edited",
            Some(_) => "unchanged",
        };
        let on = match &d.verdict {
            Verdict::Holds { .. } => "holds on NEW".to_string(),
            Verdict::Fails { after, .. } => format!("fails on NEW after {after} event{}", if *after == 1 { "" } else { "s" }),
            Verdict::Undecided { why } => format!("undecided on NEW ({why})"),
            Verdict::Attested { .. } => "undecided on NEW, attested".to_string(),
        };
        rows.push(format!("{} {}  {edit}; the law of OLD {on}", if edit == "unchanged" { "=" } else if edit == "removed" { "-" } else { "~" }, l.name));
        if let Verdict::Fails { example, .. } = &d.verdict {
            rows.push(format!("    {example}"));
            if why.is_none() {
                why = Some(format!("NEW breaks law {} of OLD (CC-L1)", l.name));
            }
        }
        if edit != "unchanged" && why.is_none() {
            why = Some(format!("law {} {edit} (CC-L2)", l.name));
        }
    }
    for x in &ln {
        if !lo.iter().any(|l| l.name == x.name) {
            rows.push(format!("+ {}  added", x.name));
        }
    }
    (rows, why)
}
