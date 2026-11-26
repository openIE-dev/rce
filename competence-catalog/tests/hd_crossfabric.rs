use competence_catalog::hd::{cuda_source, predict_native, Elm, HdModel};
use competence_catalog::train::receipt_corpus;

fn acc<F: Fn(&[f64]) -> bool>(f: F, hold: &[(Vec<f64>, bool)]) -> f64 {
    let c = hold.iter().filter(|(x, y)| f(x) == *y).count();
    c as f64 / hold.len() as f64
}

#[test]
fn hd_one_shot_learning_is_accurate_without_backprop() {
    let corpus = receipt_corpus(4000);
    let t0 = std::time::Instant::now();
    let m = HdModel::train(&corpus[..3000]);
    let us = t0.elapsed().as_micros();
    let a = acc(|x| m.predict(x), &corpus[3000..]);
    println!("HD one-shot acc {a:.4}, train {us}us (single pass, no epochs)");
    assert!(a > 0.8, "HD {a}");
}

#[test]
fn elm_closed_form_readout_is_accurate() {
    let corpus = receipt_corpus(4000);
    let m = Elm::train(&corpus[..3000], 64, 1.0);
    let a = acc(|x| m.predict(x), &corpus[3000..]);
    println!("ELM closed-form acc {a:.4}");
    assert!(a > 0.8, "ELM {a}");
}

#[test]
fn hd_same_math_on_second_fabric_matches() {
    let corpus = receipt_corpus(500);
    let m = HdModel::train(&corpus[..400]);
    for (x, _) in &corpus[400..410] {
        let rust = m.predict(x);
        let native = predict_native(&m, x).expect("native HD kernel");
        assert_eq!(rust, native);
    }
}

#[test]
fn cuda_kernel_emitted_as_data_for_cloud_gpu() {
    let s = cuda_source();
    assert!(s.contains("__global__"));
    assert!(s.contains("hd_encode"));
}

#[test]
fn hd_training_is_deterministic() {
    let corpus = receipt_corpus(300);
    let a = HdModel::train(&corpus);
    let b = HdModel::train(&corpus);
    assert_eq!(a.prototypes, b.prototypes);
}
