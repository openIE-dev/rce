//! MEI agent end to end against a mock HTTP server. No live keys, no network
//! beyond 127.0.0.1.
use ide_energy::sim::LinearSim;
use ide_receipt::chain::read_jsonl;
use ide_receipt::{GENESIS, Ledger, MeterSource, generate_key, verify_chain};
use mei_agent::agent::Status;
use mei_agent::catalog::GearTier;
use mei_agent::http::SystemOne;
use mei_agent::*;
use rce_mcp::Server;
use rce_mcp::items::Edit;
use rce_mcp::ws::Workspace;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

#[derive(Debug, Clone)]
struct Seen {
    path: String,
    auth: Option<String>,
    body: Value,
}

/// A mock server: answers each request with the next reply queued for its path.
struct Mock {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn mock(replies: Vec<(&'static str, u16, Value)>) -> Mock {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    let queue = Arc::new(Mutex::new(replies));
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { break };
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            let (mut len, mut auth) = (0usize, None);
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                let h = h.trim_end();
                if h.is_empty() {
                    break;
                }
                let (k, v) = h.split_once(':').unwrap();
                match k.to_ascii_lowercase().as_str() {
                    "content-length" => len = v.trim().parse().unwrap(),
                    "authorization" => auth = Some(v.trim().to_string()),
                    _ => {}
                }
            }
            let mut b = vec![0; len];
            r.read_exact(&mut b).unwrap();
            s2.lock().unwrap().push(Seen {
                path: path.clone(),
                auth,
                body: serde_json::from_slice(&b).unwrap_or(Value::Null),
            });
            let mut q = queue.lock().unwrap();
            let i = q.iter().position(|(p, _, _)| path.ends_with(p));
            let (code, body) = match i {
                Some(i) => {
                    let (_, c, b) = q.remove(i);
                    (c, b.to_string())
                }
                None => (404, "{}".into()),
            };
            let _ = write!(
                c,
                "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    Mock { url, seen }
}

fn chat_reply(edits: Value, tin: u64, tout: u64) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": json!({"edits": edits}).to_string()}}],
           "usage": {"prompt_tokens": tin, "completion_tokens": tout}})
}

fn setup(
    cfg: AgentConfig,
) -> (
    tempfile::TempDir,
    Agent,
    std::path::PathBuf,
    ide_receipt::VerifyingKey,
) {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("counter.rce"), COUNTER).unwrap();
    let lp = t.path().join("r.jsonl");
    let key = generate_key().unwrap();
    let vk = key.verifying_key();
    let server = Server::new(
        Workspace::new(&root).unwrap(),
        Ledger::open(&lp, key).unwrap(),
        LinearSim::default(),
        None,
    );
    (t, Agent::new(server, cfg), lp, vk)
}

fn reset_task(acceptance: bool) -> Task {
    Task {
        path: "counter.rce".into(),
        goal: "Add a Reset event that sets the counter to 0".into(),
        acceptance: acceptance.then(|| Acceptance {
            items: vec!["event Reset".into(), "step Counter on Reset".into()],
            min_examples: 2,
        }),
    }
}

fn good_edits() -> Value {
    json!([{"op": "insert_item", "after": "event Dec", "text": "event Reset"},
           {"op": "insert_item", "text": "step Counter on Reset -> 0"}])
}

fn cfg(budget: f64) -> AgentConfig {
    AgentConfig {
        budget_j: budget,
        timeout: Duration::from_secs(10),
        ..Default::default()
    }
}

#[test]
fn g0_accepts_when_already_done() {
    let (_t, mut a, lp, vk) = setup(cfg(10.0));
    let task = Task {
        path: "counter.rce".into(),
        goal: "nothing to do".into(),
        acceptance: Some(Acceptance {
            items: vec!["state Counter".into()],
            min_examples: 2,
        }),
    };
    let o = a.run(&task).unwrap();
    assert_eq!(o.status, Status::Accepted { gear: 0 });
    assert_eq!(o.gears_tried, [0]);
    let rs = read_jsonl(&lp).unwrap();
    verify_chain(&rs, &vk, GENESIS).unwrap();
    assert_eq!(rs.len(), o.receipts.len());
    assert_eq!(rs.len(), 3); // rce_check, rce_test, turn.accept
}

#[test]
fn g1_rule_fixes_it() {
    let mut c = cfg(10.0);
    c.rules = vec![
        Rule {
            when: "reset event".into(),
            edits: serde_json::from_value(good_edits()).unwrap(),
        },
        Rule {
            when: "add a reset".into(),
            edits: serde_json::from_value(good_edits()).unwrap(),
        },
    ];
    let (t, mut a, lp, vk) = setup(c);
    let o = a.run(&reset_task(true)).unwrap();
    assert_eq!(o.status, Status::Accepted { gear: 1 });
    assert_eq!(o.gears_tried, [0, 1]);
    let src = std::fs::read_to_string(t.path().join("ws/counter.rce")).unwrap();
    assert!(src.contains("event Reset") && src.contains("on Reset -> 0"));
    verify_chain(&read_jsonl(&lp).unwrap(), &vk, GENESIS).unwrap();
}

#[test]
fn climbs_to_g4_only_on_reject_or_not_done() {
    let m = mock(vec![
        (
            "/v1/systemone",
            200,
            json!({"model": "kev-latest", "answers": {"done": {"noul": 0.08, "confidence": 0.9}}, "usage": {"input_tokens": 120, "output_tokens": 0}}),
        ),
        // G3 proposes a step on an undeclared event: the checker refuses it.
        (
            "/local/v1/chat/completions",
            200,
            chat_reply(
                json!([{"op": "insert_item", "text": "step Counter on Reset -> 0"}]),
                900,
                40,
            ),
        ),
        // G4 fixes it; the System-One then says done.
        (
            "/remote/v1/chat/completions",
            200,
            chat_reply(good_edits(), 1100, 60),
        ),
        (
            "/v1/systemone",
            200,
            json!({"model": "kev-latest", "answers": {"done": {"noul": 0.97}}, "usage": {"input_tokens": 130}}),
        ),
    ]);
    let key_env = "MEI_TEST_KEY_CLIMB";
    // SAFETY: test-only variable with a unique name; no other thread reads it.
    unsafe { std::env::set_var(key_env, "test-key-not-live") };
    let mut c = cfg(1000.0);
    c.systemone = Some(SystemOne {
        base_url: m.url.clone(),
        model: Some("kev-latest".into()),
        key_env: None,
        timeout: Duration::from_secs(10),
    });
    c.catalog = Catalog {
        providers: vec![
            Provider {
                provider: "local-vllm".into(),
                base_url: format!("{}/local/v1", m.url),
                model_id: "dots3-note".into(),
                key_env: None,
                gear: GearTier::G3,
                price_in_per_mtok: None,
                price_out_per_mtok: None,
                energy_prior_j_per_token: None,
            },
            Provider {
                provider: "frontier-b".into(),
                base_url: format!("{}/remote2/v1", m.url),
                model_id: "big-b".into(),
                key_env: Some(key_env.into()),
                gear: GearTier::G4,
                price_in_per_mtok: Some(9.0),
                price_out_per_mtok: Some(30.0),
                energy_prior_j_per_token: None,
            },
            Provider {
                provider: "frontier-a".into(),
                base_url: format!("{}/remote/v1", m.url),
                model_id: "big-a".into(),
                key_env: Some(key_env.into()),
                gear: GearTier::G4,
                price_in_per_mtok: Some(3.0),
                price_out_per_mtok: Some(15.0),
                energy_prior_j_per_token: Some(0.01),
            },
        ],
    };
    let (t, mut a, lp, vk) = setup(c);
    let o = a.run(&reset_task(false)).unwrap();
    assert_eq!(o.status, Status::Accepted { gear: 4 }, "{o:?}");
    assert_eq!(o.gears_tried, [0, 1, 2, 3, 4]);
    let seen = m.seen.lock().unwrap().clone();
    let paths: Vec<&str> = seen.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/v1/systemone",
            "/local/v1/chat/completions",
            "/remote/v1/chat/completions",
            "/v1/systemone"
        ]
    );
    assert_eq!(seen[1].auth, None); // the local server gets no key
    assert_eq!(seen[2].auth.as_deref(), Some("Bearer test-key-not-live"));
    assert_eq!(seen[2].body["model"], "big-a"); // lowest energy prior wins
    assert_eq!(seen[0].body["questions"]["done"]["type"], "noul");
    let src = std::fs::read_to_string(t.path().join("ws/counter.rce")).unwrap();
    assert!(src.contains("event Reset"));
    let rs = read_jsonl(&lp).unwrap();
    verify_chain(&rs, &vk, GENESIS).unwrap();
    // The remote call is charged only with its catalog prior, tagged as an estimate.
    let remote: Vec<_> = rs
        .iter()
        .filter(|r| r.body.draft.model.as_deref() == Some("big-a"))
        .collect();
    assert_eq!(remote.len(), 1);
    let m4 = remote[0].body.draft.e_meas.unwrap();
    assert_eq!(m4.source, MeterSource::EstimatedRemote);
    assert!((m4.joules - 0.01 * 1160.0).abs() < 1e-9);
    assert_eq!(remote[0].body.delta_j, None);
    assert_eq!(remote[0].body.draft.tokens_in, 1100);
    // The local call had no meter on this box and no calibration: nothing invented.
    let local: Vec<_> = rs
        .iter()
        .filter(|r| r.body.draft.model.as_deref() == Some("dots3-note"))
        .collect();
    assert_eq!(local[0].body.draft.e_meas, None);
    assert_eq!(local[0].body.draft.e_sim_j, None);
    assert!((o.charged_j - 11.6).abs() < 1e-9);
    // Exactly one checker refusal (G3) on the way up.
    assert_eq!(
        rs.iter()
            .filter(|r| matches!(r.body.draft.verdict, ide_receipt::Verdict::Refused { .. }))
            .count(),
        1
    );
    // The key value is never written into a receipt.
    assert!(
        !std::fs::read_to_string(&lp)
            .unwrap()
            .contains("test-key-not-live")
    );
}

#[test]
fn budget_stops_the_climb() {
    let m = mock(vec![]);
    let mut c = cfg(1.0);
    c.gear_floor_j = [0.0, 5.0, 5.0, 50.0, 500.0];
    c.systemone = Some(SystemOne {
        base_url: m.url.clone(),
        model: None,
        key_env: None,
        timeout: Duration::from_secs(5),
    });
    let (_t, mut a, _lp, _vk) = setup(c);
    let o = a.run(&reset_task(true)).unwrap();
    match &o.status {
        Status::Failed { gear: 0, reason } => assert!(reason.contains("budget"), "{reason}"),
        s => panic!("{s:?}"),
    }
    assert!(m.seen.lock().unwrap().is_empty());
}

#[test]
fn missing_key_sends_nothing() {
    let m = mock(vec![]);
    let mut c = cfg(100.0);
    c.catalog = Catalog {
        providers: vec![Provider {
            provider: "frontier".into(),
            base_url: format!("{}/remote/v1", m.url),
            model_id: "big".into(),
            key_env: Some("MEI_TEST_KEY_DEFINITELY_UNSET".into()),
            gear: GearTier::G4,
            price_in_per_mtok: None,
            price_out_per_mtok: None,
            energy_prior_j_per_token: None,
        }],
    };
    let (_t, mut a, lp, _vk) = setup(c);
    let o = a.run(&reset_task(true)).unwrap();
    assert!(matches!(o.status, Status::Failed { gear: 4, .. }), "{o:?}");
    assert!(m.seen.lock().unwrap().is_empty());
    assert!(
        std::fs::read_to_string(&lp)
            .unwrap()
            .contains("not set; no request sent")
    );
}

#[test]
fn chart_guards() {
    let ch = mei_agent::chart::turn_chart();
    let mut a = mei_agent::chart::actor(&ch);
    assert!(!a.send("decide")); // nothing to decide in plan
    assert!(a.send("start") && a.is_active("propose") && a.is_active("gear0"));
    assert!(a.child().is_some());
    assert!(a.send("proposed") && a.send("set_done") && a.send("decide"));
    assert!(a.is_active("accept"));
    // Reject at gear0 with the budget spent: fail, never escalate.
    let mut b = mei_agent::chart::actor(&ch);
    assert!(
        b.send("start")
            && b.send("proposed")
            && b.send("set_reject")
            && b.send("budget_spent")
            && b.send("decide")
    );
    assert!(b.is_active("failed") && b.is_active("gear0"));
    // Walk to gear4 on rejects; at gear4 a reject fails.
    let mut c = mei_agent::chart::actor(&ch);
    assert!(c.send("start"));
    for g in 0..4 {
        assert!(
            c.send("proposed")
                && c.send("set_reject")
                && c.send("decide")
                && c.is_active("escalate")
        );
        assert!(
            c.send("next") && c.is_active(mei_agent::chart::GEARS[g + 1]) && c.is_active("v_none")
        );
    }
    assert!(
        c.send("proposed") && c.send("set_not_done") && c.send("decide") && c.is_active("failed")
    );
}

#[test]
fn edits_parse_from_fenced_reply() {
    let e = mei_agent::agent::parse_edits(
        "```json\n{\"edits\":[{\"op\":\"delete_item\",\"name\":\"event Dec\"}]}\n```",
    )
    .unwrap();
    assert_eq!(
        e,
        [Edit::DeleteItem {
            name: "event Dec".into()
        }]
    );
    assert!(mei_agent::agent::parse_edits("sure! here you go").is_err());
}

/// The TLS stack initializes with the pure-Rust provider. Connecting to a
/// closed local TLS port fails as a transport error, not a provider panic.
#[test]
fn https_client_builds_with_rustcrypto() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    let r = mei_agent::http::chat(
        &format!("https://127.0.0.1:{port}/v1"),
        None,
        "m",
        "s",
        "u",
        Duration::from_secs(2),
    );
    assert!(
        matches!(r, Err(mei_agent::http::HttpError::Transport(_))),
        "{r:?}"
    );
}
