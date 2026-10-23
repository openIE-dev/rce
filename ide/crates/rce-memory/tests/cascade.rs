use rce_memory::cascade::{Gear, recall};
use rce_memory::conformance::nf;
use rce_memory::engine::{MemEngine, MemoryEngine};
use rce_memory::fact::{AsOf, Provenance};
use rce_memory::form::FormIndex;
use rce_memory::vector::VectorIndex;

const EX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../spec/examples/"
);

fn ix() -> FormIndex {
    let p: Vec<(String, String)> = ["04_bank_account.rce", "39_structural_recursion.rce"]
        .iter()
        .map(|n| {
            (
                n.to_string(),
                std::fs::read_to_string(format!("{EX}{n}")).unwrap(),
            )
        })
        .collect();
    FormIndex::build(&p)
}

fn facts() -> Vec<rce_memory::fact::Fact> {
    let mut e = MemEngine::default();
    let mut a = nf("tree_sum", "complexity", "linear in tree size", 0);
    a.text = "proved by structural recursion".into();
    e.store(&a, 1).unwrap();
    let mut b = nf("deploy", "policy", "friday freeze", 0);
    b.text = "no releases on fridays".into();
    e.store(&b, 1).unwrap();
    e.as_of(AsOf::now(10)).unwrap()
}

struct Toy;
impl VectorIndex for Toy {
    fn name(&self) -> &str {
        "toy"
    }
    fn search(&self, q: &str, _k: usize) -> Option<Vec<(String, f32)>> {
        (q == "semantic money").then(|| vec![("def:Account".into(), 0.9)])
    }
}

#[test]
fn exact_gear_first() {
    let r = recall(&ix(), &facts(), None, "tree_sum", 10);
    assert_eq!(r.found, Some(Gear::Exact));
    assert_eq!(r.tried.len(), 1);
    assert!(r.hits.iter().any(|h| h.key == "def:tree_sum"));
    assert!(
        r.hits
            .iter()
            .any(|h| h.fact.as_ref().is_some_and(|f| f.predicate == "complexity"))
    );
}

#[test]
fn graph_gear_walks_one_hop() {
    let r = recall(&ix(), &facts(), None, "who calls tree_sum?", 20);
    assert_eq!(r.found, Some(Gear::Graph));
    assert_eq!(
        r.tried.iter().map(|t| t.gear).collect::<Vec<_>>(),
        [Gear::Exact, Gear::Graph]
    );
    let caller = r.hits.iter().find(|h| h.key == "def:sample_sum").unwrap();
    assert!(
        caller
            .via
            .as_deref()
            .unwrap()
            .contains("caller of tree_sum")
    );
    assert!(r.hits.iter().any(|h| h.key == "def:Tree"));
    assert!(
        r.hits.iter().any(|h| h.fact.is_some()),
        "facts about walked defs"
    );
}

#[test]
fn keyword_gear_after_graph() {
    let r = recall(&ix(), &facts(), None, "releases fridays", 5);
    assert_eq!(r.found, Some(Gear::Keyword));
    assert_eq!(
        r.tried.iter().map(|t| t.gear).collect::<Vec<_>>(),
        [Gear::Exact, Gear::Graph, Gear::Keyword]
    );
    assert_eq!(r.hits[0].fact.as_ref().unwrap().subject, "deploy");
}

#[test]
fn vector_gear_only_when_installed() {
    let r = recall(&ix(), &facts(), None, "semantic money", 5);
    assert!(r.escalate && r.found.is_none());
    assert_eq!(r.tried.len(), 3, "no vector gear without an index");
    let r = recall(&ix(), &facts(), Some(&Toy), "semantic money", 5);
    assert_eq!(r.found, Some(Gear::Vector));
    assert_eq!(r.hits[0].key, "def:Account");
}

#[test]
fn not_found_escalates() {
    let r = recall(&ix(), &facts(), Some(&Toy), "zzqx", 5);
    assert!(r.escalate);
    assert!(r.hits.is_empty());
    assert_eq!(r.tried.last().unwrap().gear, Gear::Vector);
    assert!(r.touched() > 0);
}

#[test]
fn model_facts_can_be_filtered_by_caller() {
    let mut f = facts();
    for x in &mut f {
        x.provenance = Provenance::Model;
    }
    let kept: Vec<_> = f
        .into_iter()
        .filter(|x| x.provenance != Provenance::Model)
        .collect();
    let r = recall(&ix(), &kept, None, "deploy", 5);
    assert_ne!(r.found, Some(Gear::Exact));
}

#[test]
fn bm25_tokenizer_splits_identifiers() {
    let t = rce_memory::bm25::tokenize("checked_sub treeSum");
    for w in ["checked_sub", "checked", "sub", "treesum", "tree", "sum"] {
        assert!(t.contains(&w.to_string()), "{w} in {t:?}");
    }
}
