use ide_receipt::chain::read_jsonl;
use ide_receipt::map::{from_mei, from_mol};
use ide_receipt::*;
use serde_json::json;

fn draft(sim: Option<f64>, meas: Option<Measured>) -> Draft {
    let mut d = Draft::new(ActionKind::Check, "checker", policy_hash(&json!({"p": 1})));
    d.e_sim_j = sim;
    d.e_meas = meas;
    d.verdict = Verdict::Pass;
    d
}

#[test]
fn chain_signs_links_and_verifies() {
    let mut l = Ledger::new(generate_key().unwrap());
    let a = l
        .append(draft(
            Some(1.0),
            Some(Measured {
                joules: 1.5,
                source: MeterSource::Rapl,
            }),
        ))
        .unwrap();
    let b = l.append(draft(None, None)).unwrap();
    assert_eq!(a.body.prev_hash, GENESIS);
    assert_eq!(b.body.prev_hash, a.hash);
    assert_eq!(a.body.delta_j, Some(0.5));
    assert_eq!(b.body.delta_j, None);
    assert_eq!(a.signature.len(), 128);
    verify_chain(&[a, b], &l.verifying_key(), GENESIS).unwrap();
}

#[test]
fn remote_estimate_has_no_delta() {
    let mut l = Ledger::new(generate_key().unwrap());
    let r = l
        .append(draft(
            Some(2.0),
            Some(Measured {
                joules: 3.0,
                source: MeterSource::EstimatedRemote,
            }),
        ))
        .unwrap();
    assert_eq!(r.body.delta_j, None);
}

#[test]
fn tamper_is_detected() {
    let mut l = Ledger::new(generate_key().unwrap());
    let a = l.append(draft(Some(1.0), None)).unwrap();
    let b = l.append(draft(Some(2.0), None)).unwrap();
    let vk = l.verifying_key();
    let mut t = a.clone();
    t.body.draft.e_sim_j = Some(0.1);
    assert_eq!(
        verify_chain(&[t, b.clone()], &vk, GENESIS),
        Err(VerifyError::Hash(0))
    );
    // re-sign the tampered body with another key: signer mismatch
    let mut t2 = a.body.clone();
    t2.draft.e_sim_j = Some(0.1);
    let forged = Receipt::sign(t2, &generate_key().unwrap());
    assert_eq!(
        verify_chain(&[forged, b.clone()], &vk, GENESIS),
        Err(VerifyError::Signer(0))
    );
    // drop a receipt: link breaks
    assert_eq!(
        verify_chain(std::slice::from_ref(&b), &vk, GENESIS),
        Err(VerifyError::Link(1))
    );
}

#[test]
fn file_ledger_roundtrips_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("r.jsonl");
    let key = generate_key().unwrap();
    {
        let mut l = Ledger::open(&p, key.clone()).unwrap();
        l.append(draft(
            Some(0.25),
            Some(Measured {
                joules: 0.3,
                source: MeterSource::Nvml,
            }),
        ))
        .unwrap();
        l.append(draft(None, None)).unwrap();
    }
    let mut l = Ledger::open(&p, key.clone()).unwrap();
    let c = l.append(draft(Some(1.0), None)).unwrap();
    assert_eq!(c.body.seq, 2);
    let rs = read_jsonl(&p).unwrap();
    assert_eq!(rs.len(), 3);
    verify_chain(&rs, &key.verifying_key(), GENESIS).unwrap();
    // The existing chain does not verify under another key, so opening fails.
    assert!(Ledger::open(&p, generate_key().unwrap()).is_err());
}

#[test]
fn delta_must_match_body() {
    let key = generate_key().unwrap();
    let body = Body {
        schema: SCHEMA.into(),
        seq: 0,
        draft: draft(
            Some(1.0),
            Some(Measured {
                joules: 2.0,
                source: MeterSource::Rapl,
            }),
        ),
        delta_j: Some(5.0),
        prev_hash: GENESIS.into(),
    };
    let r = Receipt::sign(body, &key);
    assert_eq!(
        verify_chain(&[r], &key.verifying_key(), GENESIS),
        Err(VerifyError::Delta(0))
    );
}

#[test]
fn maps_mol_receipt() {
    let v = json!({
        "id": "x", "schema": "mol.receipt.v1", "cascade_answered": "model",
        "estimated_j": 0.8, "estimate_kind": "analytical", "measured_j": 1.1,
        "measure_source": "rapl", "energy_honesty": "measured",
        "board_synth_claimed": false, "timestamp": "2026-10-12T17:00:00Z", "rationale": ""
    });
    let d = from_mol(&v, "p").unwrap();
    assert_eq!(d.action, ActionKind::ModelCall);
    assert_eq!(d.gear, "mol.model");
    assert_eq!(d.e_sim_j, Some(0.8));
    assert_eq!(
        d.e_meas,
        Some(Measured {
            joules: 1.1,
            source: MeterSource::Rapl
        })
    );
    // An SMC measurement has no tag here: dropped, not relabelled.
    let mut v2 = v.clone();
    v2["measure_source"] = json!("smc");
    let d2 = from_mol(&v2, "p").unwrap();
    assert_eq!(d2.e_meas, None);
    assert_eq!(d2.detail["unmapped"].as_array().unwrap().len(), 1);
    assert!(from_mol(&json!({"schema": "other"}), "p").is_err());
}

#[test]
fn maps_mei_receipt() {
    let modelled = json!({"receipt_id": "r", "op": "run", "input_hash": "a", "output_hash": "b",
        "joules": 0.5, "substrate": "stub-cpu", "tier": "L1", "jouledb_tier": "Formula",
        "measured_vs_modelled": "Modelled", "timestamp_ns": 1});
    let d = from_mei(&modelled, "p").unwrap();
    assert_eq!(d.e_sim_j, Some(0.5));
    assert_eq!(d.e_meas, None);
    assert_eq!(d.gear, "mei.L1");
    let mut m = modelled.clone();
    m["measured_vs_modelled"] = json!("Measured");
    m["substrate"] = json!("linux-rapl");
    assert_eq!(
        from_mei(&m, "p").unwrap().e_meas,
        Some(Measured {
            joules: 0.5,
            source: MeterSource::Rapl
        })
    );
    m["substrate"] = json!("cpu");
    assert_eq!(from_mei(&m, "p").unwrap().e_meas, None);
}
