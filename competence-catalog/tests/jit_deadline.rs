use competence_catalog::jit::DeadlineJit;

#[test]
fn cold_serve_interprets_now_and_queues_compile() {
    let mut j = DeadlineJit::new();
    let (out, native) = j.serve("2+3").unwrap();
    assert_eq!(out, "5");
    assert!(!native);
    assert_eq!(j.jit.compiles, 0); // no compile spike inside the tick
    assert_eq!(j.pending_len(), 1);
}

#[test]
fn maintenance_compiles_offloop_then_serves_native() {
    let mut j = DeadlineJit::new();
    j.serve("2+3").unwrap();
    assert!(j.maintenance().unwrap());
    assert_eq!(j.offloop_compiles, 1);
    assert_eq!(j.pending_len(), 0);
    let (out, native) = j.serve("2+3").unwrap();
    assert_eq!(out, "5");
    assert!(native);
    assert!(!j.maintenance().unwrap());
}

#[test]
fn duplicate_cold_serves_queue_once() {
    let mut j = DeadlineJit::new();
    j.serve("4+4").unwrap();
    j.serve("4+4").unwrap();
    assert_eq!(j.pending_len(), 1);
}
