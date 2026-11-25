use competence_catalog::train::{receipt_corpus, train_on_corpus};

#[test]
fn corpus_is_scaled_and_both_classes_present() {
    let c = receipt_corpus(2000);
    assert_eq!(c.len(), 2000);
    assert!(c.iter().any(|(_, y)| *y));
    assert!(c.iter().any(|(_, y)| !*y));
}

#[test]
fn trained_proposer_calibrates_on_holdout() {
    let (_, acc) = train_on_corpus(2000, 200, 0.5);
    println!("holdout accuracy on 400 receipts: {acc}");
    assert!(acc > 0.9, "holdout accuracy {acc}");
}

#[test]
fn training_is_deterministic() {
    let (_, a1) = train_on_corpus(1000, 100, 0.5);
    let (_, a2) = train_on_corpus(1000, 100, 0.5);
    assert_eq!(a1, a2);
}
