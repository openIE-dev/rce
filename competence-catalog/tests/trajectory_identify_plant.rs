use competence_catalog::trajectory::{identify_plant, Tick, Trajectory};
use competence_catalog::twin::{simulate, Plant};

fn ep(acts: &[f64], rec: &[f64]) -> Trajectory {
    Trajectory {
        source: "rig".into(),
        ticks: acts.iter().zip(rec.iter()).map(|(a, o)| Tick { obs: *o, act: *a }).collect(),
    }
}

#[test]
fn identifies_the_plant_that_explains_the_corpus() {
    let true_plant = Plant { dt: 0.1 };
    let a1 = vec![1.0, 1.0];
    let a2 = vec![2.0, 2.0];
    let eps = vec![ep(&a1, &simulate(&true_plant, &a1)), ep(&a2, &simulate(&true_plant, &a2))];
    let (plant, r) = identify_plant(&eps, &[0.05, 0.1, 0.2], 1e-9).unwrap();
    assert_eq!(plant.dt, 0.1);
    assert_eq!(r.admitted, 2);
}

#[test]
fn no_explaining_plant_returns_none() {
    let true_plant = Plant { dt: 0.1 };
    let acts = vec![1.0, 1.0];
    let eps = vec![ep(&acts, &simulate(&true_plant, &acts))];
    assert!(identify_plant(&eps, &[0.05, 0.2], 1e-9).is_none());
}
