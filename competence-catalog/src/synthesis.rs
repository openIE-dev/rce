//! A fresh public synthesis engine — G2 beyond extraction.
//!
//! Where extraction finds a single basis operation that covers a spec,
//! synthesis *composes* basis operations into new code that covers a spec no
//! single operation does. This is bounded, deterministic, replay-verified
//! composition search: for a single-argument spec `A → C`, search pairs
//! `g : A → B`, `f : B → C` and keep `f ∘ g` iff it reproduces every probe.
//! The synthesized program is ordinary Form, admitted by the same replay gate
//! as any competence — synthesis proposes, the gate disposes.
//!
//! This is a public, from-scratch build (no private code). It is bounded
//! two-operation composition, not learned synthesis; the search is exhaustive
//! over type-compatible pairs within a budget, and every candidate is checked.

use std::collections::HashMap;

use serde_json::Value;

use crate::runtime::RuntimeBridge;
use crate::semantic::Probe;

/// A synthesized composition: new Form code built from basis operations.
pub struct Composition {
    /// The composed expression, e.g. `math.sqrt(math.abs(x))`.
    pub expr: String,
    /// The operations composed, innermost first.
    pub ops: Vec<String>,
    /// The caller Form program that computes the composition.
    pub program: String,
}

/// Bounded composition synthesis over the basis library.
pub struct Synthesizer {
    bridge: RuntimeBridge,
    /// operation -> (module, name, input_type, output_type) for single-arg ops.
    unary: HashMap<String, (String, String, String, String)>,
    /// The search budget (candidate pairs tried).
    pub budget: usize,
}

/// Split a single-argument shape `(A)->C` into (A, C), or None if not unary.
fn unary_shape(shape: &str) -> Option<(String, String)> {
    let (args, ret) = shape.split_once("->")?;
    let args = args.trim();
    let inner = args.strip_prefix('(')?.strip_suffix(')')?;
    if inner.contains(',') {
        return None; // multi-arg: out of scope for bounded unary composition
    }
    let a = inner.trim().to_string();
    if a.is_empty() {
        return None;
    }
    Some((a, ret.trim().to_string()))
}

impl Synthesizer {
    /// A synthesizer over a signature table (operation, module, name, shape).
    pub fn new(bridge: RuntimeBridge, entries: &[(String, String, String, String)]) -> Self {
        let mut unary = HashMap::new();
        for (op, m, n, sh) in entries {
            if let Some((a, c)) = unary_shape(sh) {
                unary.insert(op.clone(), (m.clone(), n.clone(), a, c));
            }
        }
        Synthesizer { bridge, unary, budget: 4000 }
    }

    /// The default initializer for a type (for the state field).
    fn default_for(t: &str) -> Option<String> {
        if t.starts_with('i') || t.starts_with('u') || t == "usize" || t == "isize" {
            Some("0".into())
        } else if t == "f32" || t == "f64" {
            Some("0.0".into())
        } else if t == "bool" {
            Some("false".into())
        } else if t == "text" {
            Some("\"\"".into())
        } else {
            None
        }
    }

    /// Synthesize a composition covering the spec, or `None`. Searches
    /// type-compatible `f ∘ g` pairs within the budget; every candidate is
    /// replay-verified against all probes before it is returned.
    pub fn synthesize(&self, shape: &str, probes: &[Probe]) -> Option<Composition> {
        let (a, c) = unary_shape(shape)?;
        let c_init = Self::default_for(&c)?;
        let inner: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, ai, _))| *ai == a).collect();
        let outer: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, _, co))| *co == c).collect();
        self.search(&a, &c, &c_init, probes, inner, outer)
    }

    /// Learned synthesis: the same bounded search, but candidate operations are
    /// tried in order of hindsight reuse — experienced parts first. The search
    /// learns which compositions work without ever skipping the replay gate.
    pub fn synthesize_learned(
        &self,
        shape: &str,
        probes: &[Probe],
        h: &crate::learning::Hindsight,
    ) -> Option<Composition> {
        let (a, c) = unary_shape(shape)?;
        let c_init = Self::default_for(&c)?;
        let mut inner: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, ai, _))| *ai == a).collect();
        let mut outer: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, _, co))| *co == c).collect();
        inner.sort_by(|x, y| h.reuse(y.0).cmp(&h.reuse(x.0)));
        outer.sort_by(|x, y| h.reuse(y.0).cmp(&h.reuse(x.0)));
        self.search(&a, &c, &c_init, probes, inner, outer)
    }

    /// Deeper learned synthesis: search three-operation compositions `f∘h∘g`
    /// (value-typed unary ops), bounded by the budget and replay-verified. This
    /// extends competence construction beyond pairs — deeper pipelines the
    /// 2-op search cannot reach.
    pub fn synthesize_deep(&self, shape: &str, probes: &[Probe]) -> Option<Composition> {
        let (a, d) = unary_shape(shape)?;
        let d_init = Self::default_for(&d)?;
        let inner: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, ai, _))| *ai == a).collect();
        let outer: Vec<(&String, &(String, String, String, String))> =
            self.unary.iter().filter(|(_, (_, _, _, co))| *co == d).collect();
        let mut tried = 0;
        for (gop, (gm, gn, _, gb)) in &inner {
            let mids: Vec<(&String, &(String, String, String, String))> =
                self.unary.iter().filter(|(_, (_, _, hi, _))| hi == gb).collect();
            for (hop, (hm, hn, _, hc)) in &mids {
                for (fop, (fm, fn_, fi, _)) in &outer {
                    if fi != hc {
                        continue;
                    }
                    tried += 1;
                    if tried > self.budget {
                        return None;
                    }
                    let gcall = format!("{}.{}", gm.rsplit('.').next().unwrap_or(gm), gn);
                    let hcall = format!("{}.{}", hm.rsplit('.').next().unwrap_or(hm), hn);
                    let fcall = format!("{}.{}", fm.rsplit('.').next().unwrap_or(fm), fn_);
                    let body = format!("{fcall}({hcall}({gcall}(a)))");
                    let program = format!(
                        "module synth\n\nuse {gm}\nuse {hm}\nuse {fm}\n\nstate Out\n    v: {d} = {d_init}\n\n\
                         event Calc(x: {a})\n\nstep Out on Calc(x: a)\n    v = {body}\n"
                    );
                    let covers = probes.iter().all(|p| {
                        let first = match p.args.as_array().and_then(|arr| arr.first()) {
                            Some(f) => f,
                            None => return false,
                        };
                        let event = format!("Calc(x: {first})");
                        self.bridge.run_caller("synth", &program, &event)
                            .map(|got| got == p.expected).unwrap_or(false)
                    });
                    if covers {
                        return Some(Composition {
                            expr: body,
                            ops: vec![gop.to_string(), hop.to_string(), fop.to_string()],
                            program,
                        });
                    }
                }
            }
        }
        None
    }

    fn search(
        &self,
        a: &str,
        c: &str,
        c_init: &str,
        probes: &[Probe],
        inner: Vec<(&String, &(String, String, String, String))>,
        outer: Vec<(&String, &(String, String, String, String))>,
    ) -> Option<Composition> {
        let mut tried = 0;
        for (gop, (gm, gn, _, gb)) in &inner {
            for (fop, (fm, fn_, fi, _)) in &outer {
                if fi != gb {
                    continue; // intermediate types must match
                }
                tried += 1;
                if tried > self.budget {
                    return None;
                }
                let gcall = format!("{}.{}", gm.rsplit('.').next().unwrap_or(gm), gn);
                let fcall = format!("{}.{}", fm.rsplit('.').next().unwrap_or(fm), fn_);
                // Reference-ness is read from the types: the spec input `a` and
                // the outer op's input `fi` carry `&` when by-reference.
                let iarg = if a.starts_with('&') { "&a" } else { "a" };
                let gexpr = format!("{gcall}({iarg})");
                let oarg = if fi.starts_with('&') { format!("&{gexpr}") } else { gexpr.clone() };
                let body = format!("{fcall}({oarg})");
                let program = format!(
                    "module synth\n\nuse {gm}\nuse {fm}\n\nstate Out\n    v: {c} = {c_init}\n\n\
                     event Calc(x: {a})\n\nstep Out on Calc(x: a)\n    v = {body}\n"
                );
                let covers = probes.iter().all(|p| {
                    let first = match p.args.as_array().and_then(|arr| arr.first()) {
                        Some(f) => f,
                        None => return false,
                    };
                    let event = format!("Calc(x: {first})");
                    self.bridge.run_caller("synth", &program, &event)
                        .map(|got| got == p.expected).unwrap_or(false)
                });
                if covers {
                    return Some(Composition {
                        expr: body,
                        ops: vec![gop.to_string(), fop.to_string()],
                        program,
                    });
                }
            }
        }
        None
    }
}
