//! The composition-hierarchy backend: atoms -> elements -> molecules -> galaxy.

use competence_catalog::matter::{self, Galaxy};
use competence_catalog::placement::Prim;

#[test]
fn atoms_are_the_priced_fabric_floor() {
    let a = matter::atoms();
    assert_eq!(a.len(), 5, "the finite fabric destination");
    assert!(a.iter().all(|x| x.fabrics == 22), "every atom places on every fabric");
    let e = |n: &str| a.iter().find(|x| x.name == n).unwrap().energy_pj;
    assert!(e("fma") < e("mul") + e("add"), "a fused atom beats the split (Horowitz prior)");
    assert!(e("load") > e("add"), "b >> a: movement dominates a flop");
}

#[test]
fn elements_are_the_emitted_taxonomy() {
    let els = matter::elements();
    assert_eq!(els.len(), 131);
    assert!(els.iter().all(|x| x.languages >= 1), "every construct emits in >=1 language");
    let r = matter::report();
    assert_eq!(r.elements, 131);
    assert_eq!(r.emission_coverage, 1.0);
    assert_eq!(r.elements_with_reach, 131);
    assert_eq!(matter::construct_atoms("add"), vec![Prim::Add]);
    assert_eq!(matter::construct_atoms("index"), vec![Prim::Load]);
    assert!(matter::construct_atoms("while").is_empty(), "control constructs carry no fabric atom");
}

#[test]
fn molecules_admit_by_replay_and_key_by_behavior() {
    let m = matter::admit_molecule("return a+b;").unwrap();
    assert!(m.admitted);
    assert_eq!(m.probes, 36, "2 free vars -> 6x6 generative battery");
    let paraphrase = matter::admit_molecule("return b+a;").unwrap();
    assert_eq!(m.key, paraphrase.key, "same behavior = same molecule (name-free)");
    let novel = matter::admit_molecule("return a*b;").unwrap();
    assert_ne!(m.key, novel.key, "different behavior = different molecule");
}

#[test]
fn galaxy_is_a_crdt_reuse_economy() {
    let mut g1 = Galaxy::new();
    g1.admit("A", "return a+b;");
    let mut g2 = Galaxy::new();
    g2.admit("B", "return b+a;"); // same competence as A, different spelling
    g2.admit("B", "return a*b;"); // novel
    assert_eq!(g1.reach(), 1);
    g1.join(&g2);
    assert_eq!(g1.reach(), 2, "add shared across projects, mul novel: union = 2");
    assert_eq!(g1.reuse(&["return a+b;", "return a*b;", "return a-b;"]), (2, 3));
    // idempotent join
    let r = g1.reach();
    g1.join(&g2);
    assert_eq!(g1.reach(), r, "CRDT join is idempotent");
    // commutative reach
    let mut g3 = Galaxy::new();
    g3.admit("A", "return a+b;");
    g3.join(&g2);
    assert_eq!(g3.reach(), g1.reach(), "join reach is commutative");
}