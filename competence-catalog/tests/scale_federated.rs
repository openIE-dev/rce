use competence_catalog::scale::{accuracy, federated_train, shard};
use competence_catalog::train::{receipt_corpus, Scorer};

#[test]
fn federated_over_four_shards_matches_centralized_at_20k_receipts() {
    let corpus = receipt_corpus(20_000);
    let hold = &corpus[18_000..];
    let train = &corpus[..18_000];
    let shards = shard(train, 4);
    let fed = federated_train(&shards, 150, 0.5);
    let cent = Scorer::train(train, 150, 0.5);
    let af = accuracy(&fed, hold);
    let ac = accuracy(&cent, hold);
    println!("federated acc {af:.4} centralized acc {ac:.4} (20k receipts, CPU)");
    assert!(af > 0.9, "federated {af}");
    assert!((af - ac).abs() < 0.05, "fed {af} vs cent {ac}");
}

#[test]
fn federated_is_deterministic() {
    let corpus = receipt_corpus(4000);
    let shards = shard(&corpus, 4);
    let a = federated_train(&shards, 80, 0.5);
    let b = federated_train(&shards, 80, 0.5);
    assert_eq!(a.w, b.w);
    assert_eq!(a.b, b.b);
}

#[test]
fn shard_partitions_cover_the_corpus() {
    let corpus = receipt_corpus(1000);
    let sh = shard(&corpus, 4);
    let total: usize = sh.iter().map(|s| s.len()).sum();
    assert_eq!(total, 1000);
    assert_eq!(sh.len(), 4);
}
