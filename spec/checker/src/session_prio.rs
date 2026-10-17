//! W-SES-5 by priorities (T28p, `spec/lean/RCE/SessionPrio.lean`).
//!
//! A program whose session graph is not a tree (a spawn passes two
//! endpoints, or a body acts on a channel while it still holds both ends)
//! is still accepted when its exchanges can be given priorities that every
//! process follows in increasing order (Kobayashi, CONCUR 2006; Padovani,
//! CSL-LICS 2014). `net_of` reads the program as a network: `main` and each
//! process it spawns (recursively) is a process; the `t`-th exchange on a
//! session channel is the action `(c, t)`, the same for both ends; a spawn
//! is a one-step channel between the spawner (at the spawn) and the new
//! process (its first action). `decide` accepts when each channel has two
//! distinct holders that do the same number of exchanges on it and the
//! order "right before, in some process" on actions has no cycle; then
//! longest-path priorities increase along every process, which is
//! `SessionPrio.chk`, and `SessionPrio.decideP_sound` gives deadlock
//! freedom for every reachable state, cycles in the session graph included.
//!
//! The reading covers straight-line bodies: channels made with
//! `chan[session P]` where `P` is a sequence of `send` / `recv`, sends,
//! receives and spawns at the top level of a body, endpoints passed to a
//! spawn before their holder used them. A program outside it (an endpoint
//! under a branch or loop, moved by `let`, sent as a value, passed to a
//! call, a protocol with choice or loop) is not read, and W-SES-5's tree
//! rule stands. Exchanges are rendezvous; channel buffers are not modelled
//! (they only let a send finish earlier).
use crate::ast::*;
use std::collections::{BTreeMap, HashMap};

pub type Act = (usize, usize);

#[derive(Debug, Default, Clone)]
pub struct Net {
    /// each process's actions in order (process 0 is `main`)
    pub procs: Vec<Vec<Act>>,
    /// holders of the two ends of each channel
    pub ends: Vec<(usize, usize)>,
    /// exchanges done by each end of each channel
    pub done: Vec<(usize, usize)>,
}

/// accept when the holders differ, both ends do the same exchanges and the
/// action order is acyclic (`SessionPrio.decideP`)
pub fn decide(n: &Net) -> bool {
    if n.ends.iter().any(|(a, b)| a == b) || n.done.iter().any(|(a, b)| a != b) {
        return false;
    }
    let mut succ: BTreeMap<Act, Vec<Act>> = BTreeMap::new();
    let mut indeg: BTreeMap<Act, usize> = BTreeMap::new();
    for l in &n.procs {
        for x in l {
            indeg.entry(*x).or_default();
            succ.entry(*x).or_default();
        }
        for w in l.windows(2) {
            succ.get_mut(&w[0]).unwrap().push(w[1]);
            *indeg.get_mut(&w[1]).unwrap() += 1;
        }
    }
    let mut ready: Vec<Act> = indeg.iter().filter(|(_, d)| **d == 0).map(|(a, _)| *a).collect();
    let mut seen = 0;
    while let Some(a) = ready.pop() {
        seen += 1;
        for b in &succ[&a] {
            let d = indeg.get_mut(b).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push(*b);
            }
        }
    }
    seen == indeg.len()
}

fn mentions(s: &impl serde::Serialize, names: &HashMap<String, (usize, bool)>) -> bool {
    let j = serde_json::to_string(s).unwrap_or_default();
    names.keys().any(|x| j.contains(&format!("{{\"Name\":[{},", serde_json::to_string(x).unwrap())))
}

struct Rd<'a> {
    fns: HashMap<&'a str, &'a FnDecl>,
    protos: &'a HashMap<String, Vec<ProtoStep>>,
    net: Net,
}

fn session_proto(t: &Ty) -> Option<&str> {
    match t {
        Ty::Path(q, a) if q.len() == 1 && q[0] == "session" => match a.first() {
            Some(Ty::Path(p, _)) if p.len() == 1 => Some(&p[0]),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> Rd<'a> {
    fn straight(&self, p: &str) -> bool {
        self.protos
            .get(p)
            .is_some_and(|s| s.iter().all(|x| matches!(x, ProtoStep::Send(_) | ProtoStep::Recv(_))))
    }

    fn act(&mut self, me: usize, env: &HashMap<String, (usize, bool)>, x: &str) -> Option<()> {
        let &(c, side) = env.get(x)?;
        let k = if side { &mut self.net.done[c].0 } else { &mut self.net.done[c].1 };
        let t = *k;
        *k += 1;
        self.net.procs[me].push((c, t));
        Some(())
    }

    fn body(&mut self, me: usize, f: &'a FnDecl, mut env: HashMap<String, (usize, bool)>) -> Option<()> {
        let Body::Block(stmts) = &f.body else {
            return if mentions(&f.body, &env) { None } else { Some(()) };
        };
        for s in stmts {
            match s {
                Stmt::Let(Pat::Tuple(ps), _, Expr::ChanNew(t, cap, _), _) if session_proto(t).is_some() => {
                    let p = session_proto(t)?;
                    if !self.straight(p) || ps.len() != 2 || cap.as_ref().is_some_and(|c| mentions(c, &env)) {
                        return None;
                    }
                    let (Pat::Path(x), Pat::Path(y)) = (&ps[0], &ps[1]) else { return None };
                    let c = self.net.ends.len();
                    self.net.ends.push((me, me));
                    self.net.done.push((0, 0));
                    env.insert(x[0].clone(), (c, true));
                    env.insert(y[0].clone(), (c, false));
                }
                Stmt::Send(Expr::Name(x, _), v) if env.contains_key(x) => {
                    if mentions(v, &env) {
                        return None;
                    }
                    self.act(me, &env, x)?;
                }
                Stmt::Let(_, _, Expr::Recv(r, _), _) | Stmt::Expr(Expr::Recv(r, _))
                    if matches!(&**r, Expr::Name(x, _) if env.contains_key(x)) =>
                {
                    let Expr::Name(x, _) = &**r else { unreachable!() };
                    self.act(me, &env, x)?;
                }
                Stmt::Spawn(Expr::Call(callee, args, _, _)) => {
                    let g = match &**callee {
                        Expr::Name(fname, _) => self.fns.get(fname.as_str()).copied(),
                        _ => None,
                    };
                    let Some(g) = g else {
                        // a spawn that is not of a function of this module:
                        // fine as long as it takes no endpoint
                        if mentions(s, &env) {
                            return None;
                        }
                        continue;
                    };
                    if self.net.procs.len() >= 64 || g.sig.params.len() != args.len() {
                        return None;
                    }
                    let new = self.net.procs.len();
                    self.net.procs.push(vec![]);
                    let sc = self.net.ends.len();
                    self.net.ends.push((me, new));
                    self.net.done.push((1, 1));
                    self.net.procs[me].push((sc, 0));
                    self.net.procs[new].push((sc, 0));
                    let mut genv = HashMap::new();
                    for (a, p) in args.iter().zip(&g.sig.params) {
                        match (&a.e, p) {
                            (Expr::Name(x, _), Param::Named { name, ty, .. }) if env.contains_key(x) => {
                                session_proto(ty)?;
                                let (c, side) = env.remove(x)?;
                                let (used, holder) = if side {
                                    (self.net.done[c].0, &mut self.net.ends[c].0)
                                } else {
                                    (self.net.done[c].1, &mut self.net.ends[c].1)
                                };
                                if used > 0 || *holder != me {
                                    return None;
                                }
                                *holder = new;
                                genv.insert(name.clone(), (c, side));
                            }
                            (e, _) if mentions(e, &env) => return None,
                            _ => {}
                        }
                    }
                    self.body(new, g, genv)?;
                }
                s if mentions(s, &env) => return None,
                _ => {}
            }
        }
        Some(())
    }
}

/// the network of a program in the straight-line reading, from `main`
pub fn net_of(m: &Module, protos: &HashMap<String, Vec<ProtoStep>>) -> Option<Net> {
    let mut fns = HashMap::new();
    for it in &m.items {
        if let Item::Fn(f) = it {
            fns.insert(f.sig.name.as_str(), f);
        }
    }
    let main = *fns.get("main")?;
    let mut rd = Rd {
        fns,
        protos,
        net: Net {
            procs: vec![vec![]],
            ..Default::default()
        },
    };
    rd.body(0, main, HashMap::new())?;
    Some(rd.net)
}

/// W-SES-5 holds by priorities
pub fn discharges(m: &Module, protos: &HashMap<String, Vec<ProtoStep>>) -> bool {
    net_of(m, protos).is_some_and(|n| decide(&n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_and_reversed_ring() {
        let ends = vec![(0, 1), (1, 2), (2, 0)];
        let ok = Net {
            procs: vec![vec![(0, 0), (2, 0)], vec![(0, 0), (1, 0)], vec![(1, 0), (2, 0)]],
            ends: ends.clone(),
            done: vec![(1, 1); 3],
        };
        assert!(decide(&ok));
        let bad = Net {
            procs: vec![vec![(2, 0), (0, 0)], vec![(0, 0), (1, 0)], vec![(1, 0), (2, 0)]],
            ends,
            done: vec![(1, 1); 3],
        };
        assert!(!decide(&bad));
    }
}
