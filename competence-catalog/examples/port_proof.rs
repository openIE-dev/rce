//! Print the no-pay conformity/port artifacts so they can be inspected, not asserted.

fn main() {
    let ts = vec![
        competence_catalog::rtos::Task { name: "ctrl", period_us: 2000, wcet_us: 500 },
        competence_catalog::rtos::Task { name: "plan", period_us: 4000, wcet_us: 500 },
    ];
    let (conf, c) = competence_catalog::rtos::zephyr_project(&ts).unwrap();
    println!("=== prj.conf ===\n{conf}");
    println!("=== main.c ===\n{c}");
    println!("=== seL4 manifest ===\n{}", competence_catalog::rtos::sel4_manifest(&ts).unwrap());
    let mut t = competence_catalog::trust::TrustLayer::new();
    t.record("safety-update", true, "HwShunt");
    println!("=== module A (excerpt) ===");
    let f = t.module_a_file();
    println!("route: {}", f["route"]);
    println!("declaration: {}", f["declaration"]);
    println!("=== local capabilities ===\n{:?}", competence_catalog::fabric::local_capabilities());
}
