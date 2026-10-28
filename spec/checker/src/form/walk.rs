//! A walk over the regions of a function, with the types of every source
//! (FG5) and the identity of values (§7): a value keeps its identity along
//! a capture into a Branch arm and along the returned operand of a
//! read-return port.
use super::catalogue::{self, Mode};
use super::ir::*;
use super::types::{Tx, node_outs};

/// One region on the path from a function's body to the region walked.
pub struct Fr<'a> {
    pub body: &'a Body,
    /// the Branch or Iterate that owns this region; none for a function's body
    pub owner: Option<&'a Node>,
    /// the arm's number, or the Iterate region's (0 test, 1 body)
    pub arm: u32,
    pub ins: Vec<Option<BTy>>,
    pub outs: Vec<Option<Vec<BTy>>>,
}

pub struct Cx<'a> {
    pub tx: Tx<'a>,
    pub group: &'a str,
    pub fi: u32,
    pub f: &'a Func,
    pub frames: Vec<Fr<'a>>,
    pub path: Vec<String>,
}

pub trait Visit<'a> {
    fn region(&mut self, _cx: &Cx<'a>) -> Result<(), String> {
        Ok(())
    }
    fn node(&mut self, _cx: &Cx<'a>, _i: usize) -> Result<(), String> {
        Ok(())
    }
}

impl<'a> Fr<'a> {
    /// The Branch this region is an arm of.
    pub fn arm_of(&self) -> Option<&'a Node> {
        self.owner.filter(|n| matches!(n.attrs, Attr::Branch(..)))
    }
}

impl<'a> Cx<'a> {
    pub fn depth(&self) -> usize {
        self.frames.len() - 1
    }
    pub fn top(&self) -> &Fr<'a> {
        self.frames.last().unwrap()
    }
    pub fn path(&self) -> String {
        self.path.join(" ")
    }

    /// The source a value starts at: captures and read-return ports followed back.
    pub fn trace(&self, mut d: usize, mut s: Src) -> (usize, Src) {
        loop {
            match s {
                Src::In(j) => match self.frames[d].arm_of() {
                    Some(b) if d > 0 => match b.ins.get(1 + j as usize) {
                        Some(x) => {
                            s = *x;
                            d -= 1;
                        }
                        None => return (d, s),
                    },
                    _ => return (d, s),
                },
                Src::Out(i, p) => {
                    let Some(n) = self.frames[d].body.nodes.get(i as usize) else { return (d, s) };
                    let Attr::Prim(name, _, _) = &n.attrs else { return (d, s) };
                    let Some(e) = catalogue::entry(name) else { return (d, s) };
                    match e.results.get(p as usize) {
                        Some((Some(k), _)) if e.operands.get(*k).is_some_and(|o| o.0 == Mode::ReadReturn) => match n.ins.get(*k) {
                            Some(x) => s = *x,
                            None => return (d, s),
                        },
                        _ => return (d, s),
                    }
                }
            }
        }
    }

    /// The node a traced source is an output of.
    pub fn node_at(&self, d: usize, s: &Src) -> Option<&'a Node> {
        match s {
            Src::Out(i, _) => self.frames[d].body.nodes.get(*i as usize),
            _ => None,
        }
    }

    /// The integer literal a source holds: the output of a Bind node, followed back.
    pub fn lit(&self, d: usize, s: &Src) -> Option<i128> {
        let (d, s) = self.trace(d, *s);
        match &self.node_at(d, &s)?.attrs {
            Attr::Lit(_, Lit::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn src_ty(&self, d: usize, s: &Src) -> Option<BTy> {
        let fr = &self.frames[d];
        match s {
            Src::In(j) => fr.ins.get(*j as usize).cloned().flatten(),
            Src::Out(i, p) => fr.outs.get(*i as usize)?.as_ref()?.get(*p as usize).cloned(),
        }
    }

    pub fn ty(&self, s: &Src) -> Option<BTy> {
        self.src_ty(self.depth(), s)
    }

    /// The input types of each region of a node of the region walked.
    pub fn region_ins(&self, n: &Node) -> Vec<Vec<Option<BTy>>> {
        match &n.attrs {
            Attr::Branch(..) => {
                let caps: Vec<Option<BTy>> = n.ins.iter().skip(1).map(|s| self.ty(s)).collect();
                n.regions.iter().map(|_| caps.clone()).collect()
            }
            Attr::Count(ts) | Attr::Measure(_, _, ts) => n.regions.iter().map(|_| ts.iter().cloned().map(Some).collect()).collect(),
            _ => n.regions.iter().map(|_| vec![]).collect(),
        }
    }
}

fn enter<'a>(cx: &mut Cx<'a>, body: &'a Body, owner: Option<&'a Node>, arm: u32, ins: Vec<Option<BTy>>, v: &mut dyn Visit<'a>) -> Result<(), String> {
    cx.frames.push(Fr { body, owner, arm, ins, outs: vec![] });
    let d = cx.depth();
    let mut outs = vec![];
    for n in &body.nodes {
        let lits: Vec<Option<i128>> = n.ins.iter().map(|s| cx.lit(d, s)).collect();
        outs.push(node_outs(&cx.tx, cx.group, n, &lits));
    }
    cx.frames[d].outs = outs;
    v.region(cx)?;
    for (i, n) in body.nodes.iter().enumerate() {
        cx.path.push(format!("node {i}"));
        v.node(cx, i)?;
        let rins = cx.region_ins(n);
        for (k, (r, ri)) in n.regions.iter().zip(rins).enumerate() {
            let seg = match n.attrs {
                Attr::Branch(..) => format!("arm {k}"),
                _ => if k == 0 { "test".to_string() } else { "body".to_string() },
            };
            cx.path.push(seg);
            enter(cx, r, Some(n), k as u32, ri, v)?;
            cx.path.pop();
        }
        cx.path.pop();
    }
    cx.frames.pop();
    Ok(())
}

/// Walk every region of a function, outermost first, each node before the
/// regions it owns.
pub fn walk<'a>(m: &'a Module, group: &'a str, fi: u32, f: &'a Func, v: &mut dyn Visit<'a>) -> Result<(), (String, String)> {
    let mut cx = Cx { tx: Tx::new(m), group, fi, f, frames: vec![], path: vec![] };
    let ins = f.ins.iter().cloned().map(Some).collect();
    match enter(&mut cx, &f.body, None, 0, ins, v) {
        Ok(()) => Ok(()),
        Err(e) => Err((cx.path(), e)),
    }
}
