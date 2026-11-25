use competence_catalog::value::{rank, value, Candidate};

#[test]
fn lookup_beats_proposer_at_equal_benefit() {
    let r = rank(&[
        Candidate { id: "novel".into(), benefit: 10.0, joules: 5_000_000.0 },
        Candidate { id: "covered".into(), benefit: 10.0, joules: 1.0 },
    ])
    .unwrap();
    assert_eq!(r[0].id, "covered");
    assert_eq!(r[0].value, 10.0);
    assert!(r[0].value > r[1].value);
}

#[test]
fn higher_benefit_wins_only_per_joule() {
    let r = rank(&[
        Candidate { id: "big".into(), benefit: 100.0, joules: 5_000_000.0 },
        Candidate { id: "small".into(), benefit: 1.0, joules: 1.0 },
    ])
    .unwrap();
    assert_eq!(r[0].id, "small");
}

#[test]
fn ties_break_deterministically_by_id() {
    let a = Candidate { id: "b".into(), benefit: 2.0, joules: 2.0 };
    let b = Candidate { id: "a".into(), benefit: 3.0, joules: 3.0 };
    let r1 = rank(&[a.clone(), b.clone()]).unwrap();
    let r2 = rank(&[b, a]).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r1[0].id, "a");
}

#[test]
fn refuses_invalid_boundary_inputs() {
    assert!(value(-1.0, 1.0).is_err());
    assert!(value(f64::NAN, 1.0).is_err());
    assert!(value(1.0, 0.0).is_err());
    assert!(value(1.0, -2.0).is_err());
    assert!(value(1.0, f64::INFINITY).is_err());
    assert!(rank(&[Candidate { id: "x".into(), benefit: 1.0, joules: 0.0 }]).is_err());
}

#[test]
fn zero_benefit_is_allowed_and_ranks_last() {
    let r = rank(&[
        Candidate { id: "useful".into(), benefit: 1.0, joules: 1.0 },
        Candidate { id: "pointless".into(), benefit: 0.0, joules: 1.0 },
    ])
    .unwrap();
    assert_eq!(r[1].id, "pointless");
    assert_eq!(r[1].value, 0.0);
}
