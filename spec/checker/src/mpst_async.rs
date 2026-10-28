//! Multiparty global types with full merge and asynchronous, order-preserving
//! delivery (T28a, `RCE.SessionAsync`; full merge after Hou, Yoshida and Kuhn,
//! arXiv:2402.16741, Def. 3.1; queues after Honda, Yoshida and Carbone, POPL
//! 2008, Sec. 2). `proj` is the checker's copy of the Lean projection, linked
//! by `tests/lean_session_async.rs`. `explore` runs the asynchronous
//! semantics (`SessionAsync.AStep`) from a configuration and reports the
//! first reachable configuration that is neither final nor able to step, or
//! whose receiver cannot take its queue head (a stuck configuration); `SessionAsync.deadlock_free` says
//! there is none when every role projects.
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum G {
    Done,
    /// `p` sends to `q` one of the offered labels (`ha`: label 1, `hb`: label 0)
    Comm(usize, usize, bool, bool, Box<G>, Box<G>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum L {
    Done,
    Snd(usize, bool, bool, Box<L>, Box<L>),
    Rcv(usize, bool, bool, Box<L>, Box<L>),
}

/// no role talks to itself, and a choice offers a label
pub fn wf(g: &G) -> bool {
    match g {
        G::Done => true,
        G::Comm(p, q, ha, hb, a, b) => p != q && (*ha || *hb) && (!*ha || wf(a)) && (!*hb || wf(b)),
    }
}

/// full merge: sends agree, receives take the union of labels
pub fn merge(x: &L, y: &L) -> Option<L> {
    match (x, y) {
        (L::Done, L::Done) => Some(L::Done),
        (L::Snd(q, ha, hb, a, b), L::Snd(q2, ha2, hb2, a2, b2))
            if q == q2 && ha == ha2 && hb == hb2 =>
        {
            let u = if *ha { merge(a, a2)? } else { L::Done };
            let v = if *hb { merge(b, b2)? } else { L::Done };
            Some(L::Snd(*q, *ha, *hb, Box::new(u), Box::new(v)))
        }
        (L::Rcv(p, ha, hb, a, b), L::Rcv(p2, ha2, hb2, a2, b2)) if p == p2 => {
            let one = |h: bool, h2: bool, a: &L, a2: &L| match (h, h2) {
                (true, true) => merge(a, a2),
                (true, false) => Some(a.clone()),
                (false, true) => Some(a2.clone()),
                (false, false) => Some(L::Done),
            };
            let u = one(*ha, *ha2, a, a2)?;
            let v = one(*hb, *hb2, b, b2)?;
            Some(L::Rcv(
                *p,
                *ha || *ha2,
                *hb || *hb2,
                Box::new(u),
                Box::new(v),
            ))
        }
        _ => None,
    }
}

pub fn proj(g: &G, r: usize) -> Option<L> {
    match g {
        G::Done => Some(L::Done),
        G::Comm(p, q, ha, hb, a, b) => {
            let x = if *ha { proj(a, r)? } else { L::Done };
            let y = if *hb { proj(b, r)? } else { L::Done };
            if r == *q {
                Some(L::Rcv(*p, *ha, *hb, Box::new(x), Box::new(y)))
            } else if r == *p {
                Some(L::Snd(*q, *ha, *hb, Box::new(x), Box::new(y)))
            } else {
                match (ha, hb) {
                    (true, true) => merge(&x, &y),
                    (true, false) => Some(x),
                    (false, true) => Some(y),
                    (false, false) => None,
                }
            }
        }
    }
}

/// a configuration: local types and one FIFO queue per ordered pair of roles
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cfg {
    pub c: Vec<L>,
    pub q: Vec<Vec<VecDeque<bool>>>,
}

fn pick<T>(d: bool, a: T, b: T) -> T {
    if d { a } else { b }
}

/// the asynchronous steps of `SessionAsync.AStep`
pub fn steps(k: &Cfg) -> Vec<Cfg> {
    let mut out = vec![];
    for (r, l) in k.c.iter().enumerate() {
        match l {
            L::Snd(q, ha, hb, a, b) => {
                for d in [true, false] {
                    if pick(d, *ha, *hb) {
                        let mut n = k.clone();
                        n.c[r] = pick(d, a, b).as_ref().clone();
                        n.q[r][*q].push_back(d);
                        out.push(n);
                    }
                }
            }
            L::Rcv(p, ha, hb, a, b) => {
                if let Some(&d) = k.q[*p][r].front() {
                    if pick(d, *ha, *hb) {
                        let mut n = k.clone();
                        n.c[r] = pick(d, a, b).as_ref().clone();
                        n.q[*p][r].pop_front();
                        out.push(n);
                    }
                }
            }
            L::Done => {}
        }
    }
    out
}

pub fn is_final(k: &Cfg) -> bool {
    k.c.iter().all(|l| *l == L::Done) && k.q.iter().flatten().all(|q| q.is_empty())
}

/// a receiver whose queue head it does not offer
pub fn receiver_stuck(k: &Cfg) -> bool {
    k.c.iter().enumerate().any(|(r, l)| match l {
        L::Rcv(p, ha, hb, ..) => k.q[*p][r].front().is_some_and(|&d| !pick(d, *ha, *hb)),
        _ => false,
    })
}

/// explores every reachable configuration (up to `limit` of them); returns
/// the number explored, or the first stuck configuration
pub fn explore(start: Cfg, limit: usize) -> Result<usize, Cfg> {
    let mut seen = HashSet::new();
    let mut todo = vec![start];
    while let Some(k) = todo.pop() {
        if !seen.insert(k.clone()) {
            continue;
        }
        let next = steps(&k);
        if (next.is_empty() && !is_final(&k)) || receiver_stuck(&k) {
            return Err(k);
        }
        if seen.len() >= limit {
            break;
        }
        todo.extend(next);
    }
    Ok(seen.len())
}

/// the start configuration: the given local types, empty queues
pub fn start(c: Vec<L>) -> Cfg {
    let n = c.len();
    Cfg {
        c,
        q: vec![vec![VecDeque::new(); n]; n],
    }
}

fn flag(t: &str) -> Option<bool> {
    match t {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

/// prefix form used by the Lean vectors
pub fn parse_g(s: &str) -> Option<G> {
    fn go<'a>(t: &mut impl Iterator<Item = &'a str>) -> Option<G> {
        Some(match t.next()? {
            "e" => G::Done,
            "c" => {
                let (p, q) = (t.next()?.parse().ok()?, t.next()?.parse().ok()?);
                let (ha, hb) = (flag(t.next()?)?, flag(t.next()?)?);
                G::Comm(p, q, ha, hb, Box::new(go(t)?), Box::new(go(t)?))
            }
            _ => return None,
        })
    }
    let mut t = s.split_whitespace();
    let g = go(&mut t)?;
    t.next().is_none().then_some(g)
}

pub fn show_l(l: &L) -> String {
    let f = |b: &bool| if *b { "1" } else { "0" };
    match l {
        L::Done => "e".into(),
        L::Snd(q, ha, hb, a, b) => format!("s {q} {} {} {} {}", f(ha), f(hb), show_l(a), show_l(b)),
        L::Rcv(p, ha, hb, a, b) => format!("r {p} {} {} {} {}", f(ha), f(hb), show_l(a), show_l(b)),
    }
}
