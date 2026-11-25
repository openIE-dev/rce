use competence_catalog::fabric::{builtin_fabrics, memory_kib, place, place_within};

#[test]
fn memory_bound_device_excludes_big_fabrics() {
    let fs = builtin_fabrics();
    let need = 1_048_576; // 1 GiB act
    let cap = 8_388_608; // 8 GiB device (HW3-class)
    let f = place_within(&fs, need, cap).expect("some fabric fits");
    assert!(memory_kib(f.name) >= need && memory_kib(f.name) <= cap);
    assert_ne!(f.name, "gpu-nvidia-ptx");
    assert_ne!(f.name, "cpu-x86");
}

#[test]
fn thor_class_budget_matches_unconstrained_placement() {
    let fs = builtin_fabrics();
    let unconstrained = place(&fs).unwrap().name;
    let within = place_within(&fs, 1_048_576, 134_217_728).unwrap().name;
    assert_eq!(unconstrained, within);
}

#[test]
fn impossible_need_refuses() {
    let fs = builtin_fabrics();
    assert!(place_within(&fs, 1_099_511_627_776, 1_099_511_627_776).is_none());
}

#[test]
fn groq_sram_class_is_small() {
    assert_eq!(memory_kib("groq-lpu"), 235_520);
    let fs = builtin_fabrics();
    let f = place_within(&fs, 1024, 235_520).unwrap();
    assert!(memory_kib(f.name) <= 235_520);
}
