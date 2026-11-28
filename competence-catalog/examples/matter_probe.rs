//! Print the atoms -> elements -> molecules -> galaxy completion report and laws.

use competence_catalog::matter;

fn main() {
    let r = matter::report();
    println!(
        "atoms={} fabrics={} elements={} emission_coverage={:.3} elements_with_reach={}",
        r.atoms, r.fabrics, r.elements, r.emission_coverage, r.elements_with_reach
    );
    for a in matter::atoms() {
        println!("  atom {:5} energy_pj={:5} fabrics={}", a.name, a.energy_pj, a.fabrics);
    }
    let m = matter::admit_molecule("return a+b;").unwrap();
    println!("molecule admitted={} probes={} key={}…", m.admitted, m.probes, &m.key[..12]);

    let mut g1 = matter::Galaxy::new();
    g1.admit("projA", "return a+b;");
    let mut g2 = matter::Galaxy::new();
    g2.admit("projB", "return b+a;"); // same behavior as projA
    g2.admit("projB", "return a*b;"); // novel
    let before = g1.reach();
    g1.join(&g2);
    println!("galaxy reach before={} after={} (A:add, B:add+mul)", before, g1.reach());
    let (h, n) = g1.reuse(&["return a+b;", "return a*b;", "return a-b;"]);
    println!("galaxy reuse={h}/{n}");
    let r1 = g1.reach();
    g1.join(&g2);
    println!("idempotent join: {} == {}", r1, g1.reach());
}