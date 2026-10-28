use rce_memory::form::{FormIndex, IndexCache, form_hash};

const EX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../spec/examples/"
);

fn ex(name: &str) -> (String, String) {
    (
        name.to_string(),
        std::fs::read_to_string(format!("{EX}{name}")).unwrap(),
    )
}

fn prog() -> Vec<(String, String)> {
    vec![
        ex("04_bank_account.rce"),
        ex("39_structural_recursion.rce"),
        ex("01_counter.rce"),
    ]
}

fn idx(ix: &FormIndex, n: &str) -> usize {
    *ix.find(n)
        .first()
        .unwrap_or_else(|| panic!("no def {n}: {:?}", ix.defs))
}

#[test]
fn checker_admits_and_defs_are_found() {
    let ix = FormIndex::build(&prog());
    assert!(
        ix.verdicts.values().all(Option::is_none),
        "{:?}",
        ix.verdicts
    );
    for (n, k) in [
        ("Account", "state"),
        ("Deposit", "event"),
        ("Insufficient", "error"),
        ("Withdrawn", "struct"),
        ("Tree", "kind"),
        ("tree_sum", "fn"),
        ("Counter", "state"),
    ] {
        let d = &ix.defs[idx(&ix, n)];
        assert_eq!(d.kind, k, "{n}");
    }
    let ts = &ix.defs[idx(&ix, "tree_sum")];
    assert_eq!(
        (ts.file.as_str(), ts.line),
        ("39_structural_recursion.rce", 7)
    );
}

#[test]
fn refs_and_call_edges() {
    let ix = FormIndex::build(&prog());
    let sum = idx(&ix, "sample_sum");
    let ts = idx(&ix, "tree_sum");
    let tree = idx(&ix, "Tree");
    assert!(
        ix.out_edges(sum, true).contains(&ts),
        "sample_sum calls tree_sum"
    );
    assert!(ix.in_edges(ts, true).contains(&sum));
    assert!(
        ix.out_edges(ts, false).contains(&tree),
        "tree_sum uses Tree"
    );
    // self-recursion is not an edge
    assert!(!ix.out_edges(ts, true).contains(&ts));
    // a step references its region, event and error
    let ins = idx(&ix, "Insufficient");
    let users: Vec<&str> = ix
        .in_edges(ins, false)
        .iter()
        .map(|&i| ix.defs[i].kind.as_str())
        .collect();
    assert!(
        users.contains(&"step") && users.contains(&"intent"),
        "{users:?}"
    );
}

#[test]
fn fit_file_is_reported_not_indexed_wrongly() {
    let mut p = prog();
    p.push(("bad.rce".into(), "fn f( -> \n".into()));
    let ix = FormIndex::build(&p);
    assert!(ix.verdicts["bad.rce"].is_some());
    assert!(ix.defs.iter().all(|d| d.file != "bad.rce"));
}

#[test]
fn hash_is_order_independent_and_content_sensitive() {
    let mut p = prog();
    let h = form_hash(&p);
    p.reverse();
    assert_eq!(form_hash(&p), h);
    p[0].1.push_str("\n# edit\n");
    assert_ne!(form_hash(&p), h);
}

#[test]
fn cache_is_keyed_by_form_hash() {
    let d = tempfile::tempdir().unwrap();
    let c = IndexCache::new(d.path()).unwrap();
    let p = prog();
    let (a, hit) = c.load(&p).unwrap();
    assert!(!hit);
    let (b, hit) = c.load(&p).unwrap();
    assert!(hit);
    assert_eq!(a, b);
    assert_eq!(a, FormIndex::build(&p), "cache equals a fresh rebuild");
    let mut q = p.clone();
    q.truncate(1);
    let (c2, hit) = c.load(&q).unwrap();
    assert!(!hit);
    assert_ne!(c2.form_hash, a.form_hash);
    // a corrupt entry is rebuilt, not trusted
    std::fs::write(d.path().join(format!("{}.json", a.form_hash)), "{nope").unwrap();
    let (e, hit) = c.load(&p).unwrap();
    assert!(!hit);
    assert_eq!(e, a);
}

#[test]
fn repo_map_ranks_and_respects_budget() {
    let ix = FormIndex::build(&prog());
    let (all, picked) = ix.repo_map(&[], 100_000);
    assert_eq!(picked.len(), ix.defs.len());
    assert!(all.contains("39_structural_recursion.rce:\n"));
    let (small, few) = ix.repo_map(&[], 120);
    assert!(few.len() < picked.len() && !few.is_empty());
    assert!(small.len() < all.len());
    // focusing on sample_sum ranks what it reaches above unrelated defs
    let sum = idx(&ix, "sample_sum");
    let pr = ix.pagerank(&[sum]);
    let ts = idx(&ix, "tree_sum");
    let counter = idx(&ix, "Counter");
    assert!(pr[ts] > pr[counter]);
    let total: f64 = pr.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "rank mass {total}");
}
