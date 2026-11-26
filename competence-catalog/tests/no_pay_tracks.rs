use competence_catalog::fabric::local_capabilities;
use competence_catalog::rtos::{sel4_manifest, zephyr_project, Task};
use competence_catalog::trust::TrustLayer;

fn ts() -> Vec<Task> {
    vec![
        Task { name: "ctrl", period_us: 2000, wcet_us: 500 },
        Task { name: "plan", period_us: 4000, wcet_us: 500 },
    ]
}

#[test]
fn module_a_file_is_self_assessment_with_frozen_log() {
    let mut t = TrustLayer::new();
    t.record("safety-update", true, "HwShunt");
    let f = t.module_a_file();
    assert_eq!(f["schema"], "rce-module-a/1");
    assert!(f["route"].as_str().unwrap().contains("module A"));
    assert_eq!(f["frozen_admission_log"].as_array().unwrap().len(), 1);
    assert!(f["declaration"].as_str().unwrap().contains("self-declaration"));
}

#[test]
fn slsa_statement_binds_subject_digest() {
    let t = TrustLayer::new();
    let s = t.slsa_statement("abcd1234");
    assert_eq!(s["_type"], "https://in-toto.io/Statement/v1");
    assert_eq!(s["subject"][0]["digest"]["sha256"], "abcd1234");
    assert_eq!(s["predicateType"], "https://slsa.dev/provenance/v1");
}

#[test]
fn zephyr_project_has_rate_monotonic_priorities() {
    let (conf, c) = zephyr_project(&ts()).unwrap();
    assert!(conf.contains("CONFIG_MULTITHREADING=y"));
    assert!(c.contains("ctrl_entry"));
    // ctrl (shorter period) gets priority 0, plan priority 1
    assert!(c.contains("ctrl_entry, NULL, NULL, NULL, 0,"));
    assert!(c.contains("plan_entry, NULL, NULL, NULL, 1,"));
    // definitions precede main
    assert!(c.find("K_THREAD_STACK_DEFINE").unwrap() < c.find("int main").unwrap());
}

#[test]
fn sel4_manifest_admission_checked() {
    let m = sel4_manifest(&ts()).unwrap();
    assert!(m.contains("seL4"));
    assert!(m.contains("\"priority\":0"));
    let bad = vec![Task { name: "a", period_us: 2, wcet_us: 2 }, Task { name: "b", period_us: 2, wcet_us: 2 }];
    assert!(sel4_manifest(&bad).is_none());
    assert!(zephyr_project(&bad).is_none());
}

#[test]
fn local_capability_lookup_finds_real_fabrics() {
    let caps = local_capabilities();
    assert!(caps.contains(&"wasm"));
    assert!(caps.contains(&"cpu-arm64") || caps.contains(&"cpu-x86"));
    // on this Mac, Metal and Accelerate frameworks exist
    assert!(caps.contains(&"gpu-apple-metal"));
    assert!(caps.contains(&"apple-amx"));
}
