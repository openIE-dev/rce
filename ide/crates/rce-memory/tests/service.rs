//! Receipts for every memory action, through the service.
use ide_energy::attrib::Sample;
use ide_energy::meter::{Meter, MeterError};
use ide_energy::sim::{LinearSim, Simulator};
use ide_receipt::{ActionKind, GENESIS, Ledger, MeterSource, SigningKey, Verdict, verify_chain};
use rce_memory::engine::{MemEngine, MemoryEngine};
use rce_memory::fact::Provenance;
use rce_memory::form::IndexCache;
use rce_memory::notes::Notes;
use rce_memory::service::*;

const EX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../spec/examples/"
);

fn prog() -> Vec<(String, String)> {
    ["04_bank_account.rce", "39_structural_recursion.rce"]
        .iter()
        .map(|n| {
            (
                n.to_string(),
                std::fs::read_to_string(format!("{EX}{n}")).unwrap(),
            )
        })
        .collect()
}

struct Fake(f64);
impl Meter for Fake {
    fn source(&self) -> &'static str {
        "rapl"
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        self.0 += 0.5;
        Ok(Sample {
            t_ns: 0,
            joules: self.0,
        })
    }
}

fn mem(d: &std::path::Path) -> Memory<MemEngine> {
    Memory::new(
        MemEngine::default(),
        IndexCache::new(&d.join(".rce-ide/index")).unwrap(),
        Notes::new(d, "notes").unwrap(),
    )
}

fn rem(s: &str, p: &str, o: &str) -> Remember {
    Remember {
        subject: s.into(),
        predicate: p.into(),
        object: o.into(),
        text: String::new(),
        multi: false,
        valid_from: None,
        valid_to: None,
        provenance: Provenance::Inferred,
        note: false,
    }
}

fn q(s: &str) -> RecallQuery {
    RecallQuery {
        query: s.into(),
        valid_at: None,
        known_at: None,
        exclude_model: false,
        limit: 10,
    }
}

#[test]
fn every_action_writes_a_verified_receipt() {
    let d = tempfile::tempdir().unwrap();
    let mut m = mem(d.path());
    let mut ledger = Ledger::new(SigningKey::from_bytes(&[7; 32]));
    let sim = LinearSim::default();
    let p = prog();
    let mut rs = vec![];
    {
        let mut ctx = ReceiptCtx {
            ledger: &mut ledger,
            sim: &sim,
            meter: None,
            policy_hash: "ph",
        };
        let (plan, r) = m
            .remember(&mut ctx, &p, &rem("tree_sum", "complexity", "linear"), 100)
            .unwrap();
        let f = m.engine.fetch(plan.new_id.unwrap()).unwrap();
        assert_eq!(
            f.receipt_id.as_deref(),
            Some(r.hash.as_str()),
            "fact carries the receipt id"
        );
        assert_eq!(
            f.form_hash,
            rce_memory::form::form_hash(&p),
            "pinned to the Form"
        );
        assert_eq!(
            r.body.draft.action,
            ActionKind::Other("memory.remember".into())
        );
        assert_eq!(r.body.draft.verdict, Verdict::Pass);
        assert_eq!(
            r.body.draft.e_sim_j, None,
            "uncalibrated: no invented joules"
        );
        assert_eq!(r.body.draft.e_meas, None, "no meter: no measurement");
        rs.push(r);
        let (rc, r) = m.recall(&mut ctx, &p, &q("tree_sum"), 110).unwrap();
        assert!(!rc.hits.is_empty());
        assert_eq!(r.body.draft.gear, "memory.exact");
        assert_eq!(r.body.draft.detail["detail"]["index_cache_hit"], false);
        rs.push(r);
        let (_, r) = m.recall(&mut ctx, &p, &q("tree_sum"), 111).unwrap();
        assert_eq!(
            r.body.draft.detail["detail"]["index_cache_hit"], true,
            "second recall uses the cache"
        );
        rs.push(r);
        let (_, r) = m.recall(&mut ctx, &p, &q("zzqx"), 112).unwrap();
        assert_eq!(r.body.draft.gear, "memory.not_found");
        assert_eq!(r.body.draft.detail["detail"]["escalate"], true);
        rs.push(r);
        let (text, names, r) = m
            .repo_map(&mut ctx, &p, &["sample_sum".into()], 400)
            .unwrap();
        assert!(!text.is_empty() && !names.is_empty());
        rs.push(r);
        let id = m.engine.list_live().unwrap()[0].id;
        let (_, r) = m.forget(&mut ctx, id, None, 120).unwrap();
        assert_eq!(
            r.body.draft.action,
            ActionKind::Other("memory.forget".into())
        );
        rs.push(r);
        let (_, r) = m.forget(&mut ctx, id, None, 121).unwrap();
        assert!(
            matches!(r.body.draft.verdict, Verdict::Rejected { .. }),
            "second forget not performed, still receipted"
        );
        rs.push(r);
    }
    verify_chain(&rs, &ledger.verifying_key(), GENESIS).unwrap();
}

#[test]
fn calibrated_sim_and_meter_fill_both_energies() {
    let d = tempfile::tempdir().unwrap();
    let mut m = mem(d.path());
    let mut ledger = Ledger::new(SigningKey::from_bytes(&[9; 32]));
    let mut sim = LinearSim::default();
    let samples: Vec<_> = (1..20u64)
        .map(|n| (features("exact", n * 10), 0.001 * (n * 10) as f64 + 0.01))
        .collect();
    sim.calibrate(&samples).unwrap();
    assert!(sim.predict(&features("exact", 50)).is_some());
    let meter = ide_energy::shared::SharedMeter::new(Box::new(Fake(0.0)));
    let mut ctx = ReceiptCtx {
        ledger: &mut ledger,
        sim: &sim,
        meter: Some(&meter),
        policy_hash: "ph",
    };
    let (_, r) = m.recall(&mut ctx, &prog(), &q("tree_sum"), 1).unwrap();
    let e_sim = r.body.draft.e_sim_j.expect("calibrated gear predicts");
    let meas = r.body.draft.e_meas.expect("meter measured");
    assert_eq!(meas.source, MeterSource::Rapl);
    assert!((meas.joules - 0.5).abs() < 1e-9);
    assert!((r.body.delta_j.unwrap() - (0.5 - e_sim)).abs() < 1e-9);
    // a gear with no calibration stays None even with a meter
    let (_, r) = m.recall(&mut ctx, &prog(), &q("zzqx"), 2).unwrap();
    assert_eq!(r.body.draft.e_sim_j, None);
    assert!(r.body.draft.e_meas.is_some());
}

#[test]
fn remember_as_note_writes_markdown_and_forget_marks_it() {
    let d = tempfile::tempdir().unwrap();
    let mut m = mem(d.path());
    let mut ledger = Ledger::new(SigningKey::from_bytes(&[3; 32]));
    let sim = LinearSim::default();
    let mut ctx = ReceiptCtx {
        ledger: &mut ledger,
        sim: &sim,
        meter: None,
        policy_hash: "ph",
    };
    let mut r = rem("Account", "decision", "Keep the balance on an overdraft");
    r.note = true;
    r.text = "Regulators require it.\n".into();
    let (plan, rc) = m.remember(&mut ctx, &prog(), &r, 1_000).unwrap();
    let path = d.path().join("notes/account-decision.md");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains(&format!("receipt: {}", rc.hash)), "{text}");
    // the note round-trips through sync without a change
    let (rep, _) = m.sync_notes(&mut ctx, &prog(), 2_000).unwrap();
    assert_eq!(rep.unchanged, 1, "{rep:?}");
    // a hand edit is visible to the next recall (files win, synced in-line)
    std::fs::write(
        &path,
        text.replace("Regulators require it.", "Policy 7 requires it."),
    )
    .unwrap();
    let (rcl, _) = m
        .recall(&mut ctx, &prog(), &q("Policy requires"), 3_000)
        .unwrap();
    assert!(
        rcl.hits
            .iter()
            .any(|h| h.fact.as_ref().is_some_and(|f| f.text.contains("Policy 7"))),
        "{rcl:?}"
    );
    let live = m.engine.list_live().unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].lineage, plan.new_id.unwrap());
    let (_, _) = m.forget(&mut ctx, live[0].id, None, 4_000).unwrap();
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("status: forgotten")
    );
    assert!(path.exists(), "files are never deleted");
    let (rep, _) = m.sync_notes(&mut ctx, &prog(), 5_000).unwrap();
    assert!(rep.imported.is_empty() && rep.expired.is_empty(), "{rep:?}");
}

#[test]
fn exclude_model_and_as_of_through_service() {
    let d = tempfile::tempdir().unwrap();
    let mut m = mem(d.path());
    let mut ledger = Ledger::new(SigningKey::from_bytes(&[4; 32]));
    let sim = LinearSim::default();
    let mut ctx = ReceiptCtx {
        ledger: &mut ledger,
        sim: &sim,
        meter: None,
        policy_hash: "ph",
    };
    let mut a = rem("deploy", "policy", "friday freeze");
    a.provenance = Provenance::Model;
    m.remember(&mut ctx, &prog(), &a, 100).unwrap();
    let mut x = q("deploy");
    x.exclude_model = true;
    let (r, _) = m.recall(&mut ctx, &prog(), &x, 200).unwrap();
    assert!(r.escalate, "model fact filtered out");
    let mut y = q("deploy");
    y.known_at = Some(50);
    let (r, _) = m.recall(&mut ctx, &prog(), &y, 200).unwrap();
    assert!(r.escalate, "not yet known at 50");
    let (r, _) = m.recall(&mut ctx, &prog(), &q("deploy"), 200).unwrap();
    assert!(!r.escalate);
}

#[test]
fn redb_default_layout_opens() {
    let d = tempfile::tempdir().unwrap();
    let m = Memory::open_default(d.path()).unwrap();
    assert_eq!(m.engine.health().unwrap().backend, "redb");
    assert!(d.path().join(".rce-ide/memory.redb").exists());
}
