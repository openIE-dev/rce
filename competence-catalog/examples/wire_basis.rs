//! Wire the real 75-module stdlib basis into the competence catalog: seed all
//! 1280 public functions as intent-keyed competence, then resolve novel intents
//! to G0 hits and execute the owned stdlib operations through the RCE runtime.

use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::{basis_gate, Catalog};
use serde::Deserialize;

#[derive(Deserialize)]
struct Entry {
    operation: String,
    shape: String,
}

fn main() {
    let index = std::fs::read_to_string("../stdlib-basis-index.json").expect("basis index");
    let entries: Vec<Entry> = serde_json::from_str(&index).expect("parse index");
    let basis: Vec<(String, String)> =
        entries.into_iter().map(|e| (e.operation, e.shape)).collect();

    let mut cat = Catalog::with_gate(basis_gate()).expect("catalog");
    let seeded = cat.seed_basis_index(&basis);
    println!("Seeded the real basis: {seeded} competences (|H_corpus|)");

    let bridge = RuntimeBridge::new(
        "../rce/spec/checker/target/release/rce",
        "../bridge-test/wire",
    );

    // G0 hit + real execution: hash.fnv1a32 on novel bytes "hi".
    if let Some(h) = cat.hit_intent("hash.fnv1a32", "(&[u8])->u32") {
        let r = bridge.call_hash_u8("hash", "fnv1a32", &[104, 105]).unwrap();
        println!("G0 hit {h} -> fnv1a32(\"hi\") = {r}  (executed at the interpreter floor)");
    }
    // col.search.binary_search
    if let Some(h) = cat.hit_intent("col.search.binary_search", "(&[i64],i64)->i64") {
        let r = bridge.call_binary_search(&[1, 3, 5, 7, 9, 11], 7).unwrap();
        println!("G0 hit {h} -> binary_search([1,3,5,7,9,11], 7) = {r}  (expect 3)");
    }
    // text.glob.glob_match
    if let Some(h) = cat.hit_intent("text.glob.glob_match", "(&text,&text)->bool") {
        let r = bridge.call_glob("*.txt", "file.txt").unwrap();
        println!("G0 hit {h} -> glob_match(\"*.txt\", \"file.txt\") = {r}  (expect true)");
    }
    // A novel operation outside the basis -> miss (escalation).
    let miss = cat.hit_intent("nonexistent.op", "()->()");
    println!("novel operation outside basis -> hit? {miss:?} (expect None)");

    let m = cat.meter();
    println!(
        "meter: catalog_size={} hits={} misses={} reuse_rate={:.3} cost_per_act={:.1}µJ proposer_calls={}",
        m.catalog_size, m.hits, m.misses, m.reuse_rate, m.cost_per_act_uj, m.proposer_calls
    );
}