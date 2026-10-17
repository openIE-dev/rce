//! Multiparty global types with projection (T28g, `RCE.SessionGlobal`;
//! Honda, Yoshida and Carbone, POPL 2008, plain merge). `proj` is the
//! checker's copy of the Lean projection, linked by
//! `tests/lean_session_global.rs`. A global type every role projects from is
//! deadlock free (`SessionGlobal.deadlock_free`); one of messages only gets
//! priorities from its event order (`SessionGlobal.global_prio`), which
//! `msgs_net` turns into a `session_prio::Net` so it composes with the other
//! channels of a program.
use crate::session_prio::Net;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum G {
    Done,
    Msg(usize, usize, Box<G>),
    Br(usize, usize, Box<G>, Box<G>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L {
    Done,
    Send(usize, Box<L>),
    Recv(usize, Box<L>),
    Sel(usize, Box<L>, Box<L>),
    Bra(usize, Box<L>, Box<L>),
}

/// no role talks to itself
pub fn wf(g: &G) -> bool {
    match g {
        G::Done => true,
        G::Msg(p, q, k) => p != q && wf(k),
        G::Br(p, q, a, b) => p != q && wf(a) && wf(b),
    }
}

pub fn proj(g: &G, r: usize) -> Option<L> {
    match g {
        G::Done => Some(L::Done),
        G::Msg(p, q, k) => {
            let x = Box::new(proj(k, r)?);
            Some(if r == *p { L::Send(*q, x) } else if r == *q { L::Recv(*p, x) } else { *x })
        }
        G::Br(p, q, a, b) => {
            let (x, y) = (proj(a, r)?, proj(b, r)?);
            if r == *p {
                Some(L::Sel(*q, Box::new(x), Box::new(y)))
            } else if r == *q {
                Some(L::Bra(*p, Box::new(x), Box::new(y)))
            } else if x == y {
                Some(x)
            } else {
                None
            }
        }
    }
}

/// the messages of a global type without branches, in order
pub fn msgs(g: &G) -> Option<Vec<(usize, usize)>> {
    let (mut v, mut g) = (vec![], g);
    loop {
        match g {
            G::Done => return Some(v),
            G::Msg(p, q, k) => {
                v.push((*p, *q));
                g = k;
            }
            G::Br(..) => return None,
        }
    }
}

/// `SessionGlobal.netOf`: message `i` on channel `i`, at priority `i`
pub fn msgs_net(evs: &[(usize, usize)], roles: usize) -> Net {
    let mut procs = vec![vec![]; roles];
    for (i, (p, q)) in evs.iter().enumerate() {
        procs[*p].push((i, 0));
        procs[*q].push((i, 0));
    }
    Net { procs, ends: evs.to_vec(), done: vec![(1, 1); evs.len()] }
}

/// prefix form used by the Lean vectors
pub fn parse_g(s: &str) -> Option<G> {
    fn go<'a>(t: &mut impl Iterator<Item = &'a str>) -> Option<G> {
        let n = |t: &mut dyn Iterator<Item = &'a str>| t.next()?.parse().ok();
        Some(match t.next()? {
            "e" => G::Done,
            "m" => G::Msg(n(t)?, n(t)?, Box::new(go(t)?)),
            "b" => G::Br(n(t)?, n(t)?, Box::new(go(t)?), Box::new(go(t)?)),
            _ => return None,
        })
    }
    let mut t = s.split_whitespace();
    let g = go(&mut t)?;
    t.next().is_none().then_some(g)
}

pub fn show_l(l: &L) -> String {
    match l {
        L::Done => "e".into(),
        L::Send(q, k) => format!("s {q} {}", show_l(k)),
        L::Recv(p, k) => format!("r {p} {}", show_l(k)),
        L::Sel(q, a, b) => format!("S {q} {} {}", show_l(a), show_l(b)),
        L::Bra(p, a, b) => format!("B {p} {} {}", show_l(a), show_l(b)),
    }
}
