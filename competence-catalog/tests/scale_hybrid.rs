use competence_catalog::scale::{accuracy, hybrid_plan, train_native};
use competence_catalog::train::{receipt_corpus, Scorer};

#[test]
fn hybrid_plan_spreads_shards_beyond_cpu() {
    let rows = vec![5000usize, 5000, 5000, 5000];
    let plan = hybrid_plan(&rows, 100_000.0, 134_217_728).unwrap();
    assert_eq!(plan.len(), 4);
    assert!(plan.iter().any(|a| a.fabric != "cpu-arm64" && a.fabric != "cpu-x86"), "{plan:?}");
}

#[test]
fn hybrid_plan_refuses_overload() {
    let rows = vec![1_000_000usize, 1_000_000, 1_000_000];
    assert!(hybrid_plan(&rows, 10.0, 134_217_728).is_none());
}

#[test]
fn hybrid_plan_is_deterministic() {
    let rows = vec![1000usize, 2000, 3000];
    let a = hybrid_plan(&rows, 100_000.0, 134_217_728).unwrap();
    let b = hybrid_plan(&rows, 100_000.0, 134_217_728).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.fabric, y.fabric);
    }
}

#[test]
fn native_blas_trainer_matches_scalar_on_owned_hardware() {
    let corpus = receipt_corpus(4000);
    let hold = &corpus[3000..];
    let train = &corpus[..3000];
    let t0 = std::time::Instant::now();
    let native = train_native(train, 150, 0.5).expect("native Accelerate kernel");
    let native_us = t0.elapsed().as_micros();
    let t1 = std::time::Instant::now();
    let scalar = Scorer::train(train, 150, 0.5);
    let scalar_us = t1.elapsed().as_micros();
    let an = accuracy(&native, hold);
    let as_ = accuracy(&scalar, hold);
    println!("native(Accelerate/AMX, incl. compile) acc {an:.4} in {native_us}us; scalar acc {as_:.4} in {scalar_us}us");
    assert!(an > 0.9, "native {an}");
    assert!((an - as_).abs() < 0.05, "native {an} vs scalar {as_}");
}
