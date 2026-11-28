//! The composition-hierarchy backend — atoms → elements → molecules → galaxy.
//!
//! The ecosystem completes as a composition algebra, each level built from the one below and
//! priced in joules (the Law of Coding):
//!
//! | Level | What it is | Grounded in |
//! |---|---|---|
//! | **Atom** | an irreducible fabric primitive, priced + placeable | `placement::Prim`, `costdata`, `fabric` |
//! | **Element** | a universal construct: atom composition + emission reach | `structure::CONSTRUCTS`, the LUTs |
//! | **Molecule** | an admitted, content-addressed Form (a competence) | `program` replay gate, `holdout` key |
//! | **Galaxy** | the cross-project federation of molecules (reuse economy) | content-addressed CRDT union |
//!
//! Atoms are the finite destination (a handful of fabric ops); elements are the reusable typed
//! primitives every language emits; molecules are programs admitted by replay; the galaxy is the
//! shared knowledgebase where a molecule admitted in one project is served at the lookup floor in
//! every other. Adding capacity at any level is adding a row, not rewriting the others.

use std::collections::{BTreeMap, BTreeSet};

use crate::costdata::instruction_energy;
use crate::fabric::builtin_fabrics;
use crate::holdout::behavior_key;
use crate::lut_emit::builtin_luts;
use crate::placement::Prim;
use crate::program::{generative_probes, lift_program, replay_admit};
use crate::structure::{overall, CONSTRUCTS};

/// An atom: an irreducible fabric primitive, its calibrated energy prior, and how many fabrics
/// can execute it. The finite destination of code.
pub struct Atom {
    pub prim: Prim,
    pub name: &'static str,
    pub energy_pj: f64,
    pub fabrics: usize,
}

/// The atom set: the five fabric primitives, priced (Horowitz-form est prior) and placeable
/// across the whole heterogeneous fabric table.
pub fn atoms() -> Vec<Atom> {
    let nfab = builtin_fabrics().len();
    let prims = [
        (Prim::Add, "add"),
        (Prim::Mul, "mul"),
        (Prim::Fma, "fma"),
        (Prim::Load, "load"),
        (Prim::Store, "store"),
    ];
    prims
        .iter()
        .map(|(p, name)| Atom { prim: *p, name, energy_pj: instruction_energy(*p), fabrics: nfab })
        .collect()
}

/// The atom composition of a construct (the fabric work an element lowers to). Structural /
/// control constructs carry no direct fabric primitive (priced via dispatch), hence empty.
pub fn construct_atoms(c: &str) -> Vec<Prim> {
    match c {
        "add" | "sub" | "incdec" | "compound_assign" => vec![Prim::Add],
        "mul" | "div" | "mod" | "pow" => vec![Prim::Mul],
        "index" | "member" | "deref" | "ref" | "addressof" => vec![Prim::Load],
        "let" | "assign" | "update" => vec![Prim::Store],
        _ => vec![],
    }
}

/// An element: a universal construct, its atom composition, and its emission reach (the number
/// of languages whose LUT emits it). The reusable typed primitive.
pub struct Element {
    pub name: &'static str,
    pub atoms: Vec<Prim>,
    pub languages: usize,
}

/// The element set: the 131-construct taxonomy, each with its atom composition and how many
/// languages emit it.
pub fn elements() -> Vec<Element> {
    let luts = builtin_luts();
    CONSTRUCTS
        .iter()
        .map(|c| Element {
            name: c,
            atoms: construct_atoms(c),
            languages: luts.iter().filter(|l| l.template(c).is_some()).count(),
        })
        .collect()
}

/// A molecule: an admitted, content-addressed Form. `key` is the name-free behavior key (same
/// behavior = same molecule, regardless of spelling); `admitted` is the replay-gate verdict over
/// the generative battery; `probes` is the battery size that admitted it.
pub struct Molecule {
    pub key: String,
    pub admitted: bool,
    pub probes: usize,
}

/// Admit a source as a molecule: lift to Form, run the generative battery, replay-gate it, and
/// content-address it by behavior. A weak/impostor form is refused (not admitted).
pub fn admit_molecule(src: &str) -> Option<Molecule> {
    let p = lift_program(src).ok()?;
    let battery = generative_probes(&p);
    if battery.is_empty() { return None; }
    let admitted = replay_admit(&p, &battery);
    let key = behavior_key(src)?;
    Some(Molecule { key, admitted, probes: battery.len() })
}

/// A galaxy: the cross-project federation of molecules. Each project contributes admitted
/// molecule keys; the union is content-addressed, so the same competence is shared at the lookup
/// floor everywhere. The join is a CRDT union (idempotent, commutative, associative).
#[derive(Default, Clone)]
pub struct Galaxy {
    pub projects: BTreeMap<String, BTreeSet<String>>,
}

impl Galaxy {
    pub fn new() -> Self { Galaxy::default() }

    /// Admit a source into a project; returns the molecule key if admitted.
    pub fn admit(&mut self, project: &str, src: &str) -> Option<String> {
        let m = admit_molecule(src)?;
        if !m.admitted { return None; }
        self.projects.entry(project.to_string()).or_default().insert(m.key.clone());
        Some(m.key)
    }

    /// CRDT join: idempotent union of two galaxies (same key = same competence, shared).
    pub fn join(&mut self, other: &Galaxy) {
        for (proj, keys) in &other.projects {
            self.projects.entry(proj.clone()).or_default().extend(keys.iter().cloned());
        }
    }

    /// Distinct molecules across all projects (the galaxy's reach).
    pub fn reach(&self) -> usize {
        let mut all: BTreeSet<&String> = BTreeSet::new();
        for keys in self.projects.values() { all.extend(keys.iter()); }
        all.len()
    }

    /// The union of all molecule keys (the shared competence set).
    pub fn union(&self) -> BTreeSet<String> {
        let mut all = BTreeSet::new();
        for keys in self.projects.values() { all.extend(keys.iter().cloned()); }
        all
    }

    /// Reuse hit-rate: held-out requests served by the galaxy union (name-free behavior key).
    pub fn reuse(&self, requests: &[&str]) -> (usize, usize) {
        let union = self.union();
        let hits = requests
            .iter()
            .filter(|s| behavior_key(s).map(|k| union.contains(&k)).unwrap_or(false))
            .count();
        (hits, requests.len())
    }
}

/// The measured completion report across the four levels.
pub struct Report {
    pub atoms: usize,
    pub fabrics: usize,
    pub elements: usize,
    pub emission_coverage: f64,
    pub elements_with_reach: usize,
}

/// Measure the hierarchy: atom count, fabric count, element count, mean emission coverage, and
/// how many elements at least one language emits.
pub fn report() -> Report {
    let els = elements();
    Report {
        atoms: atoms().len(),
        fabrics: builtin_fabrics().len(),
        elements: CONSTRUCTS.len(),
        emission_coverage: overall(),
        elements_with_reach: els.iter().filter(|e| e.languages > 0).count(),
    }
}