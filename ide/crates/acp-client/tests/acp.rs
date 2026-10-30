// The ACP client runs outside agents only inside the native jail, which is
// Linux-only (Landlock + seccomp); elsewhere it does not run, by design.
#![cfg(target_os = "linux")]
//! acp-client against the mock ACP agent binary, with the real rce-mcp.
use acp_client::{AcpClient, AcpConfig, client::AcpError, g4::AcpG4};
use ide_receipt::{
    ActionKind, GENESIS, Ledger, Verdict, chain::read_jsonl, generate_key, verify_chain,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

/// The rce-mcp binary, built into this test's target dir once.
fn rce_mcp() -> &'static Path {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = std::env::current_exe().unwrap();
        let profile_dir = exe.parent().unwrap().parent().unwrap().to_path_buf(); // target/debug
        let target = profile_dir.parent().unwrap();
        let st = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .args([
                "build",
                "-q",
                "-p",
                "rce-mcp",
                "--bin",
                "rce-mcp",
                "--target-dir",
            ])
            .arg(target)
            .status()
            .unwrap();
        assert!(st.success());
        profile_dir.join("rce-mcp")
    })
}

struct Env {
    _t: tempfile::TempDir,
    ws: PathBuf,
    ledger: PathBuf,
    vk: ide_receipt::VerifyingKey,
}

fn env() -> (Env, AcpConfig, Ledger) {
    let t = tempfile::tempdir().unwrap();
    let ws = t.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("counter.rce"), COUNTER).unwrap();
    let ledger = t.path().join("ide.jsonl");
    let key = generate_key().unwrap();
    let vk = key.verifying_key();
    let l = Ledger::open(&ledger, key).unwrap();
    let mut c = AcpConfig::new(env!("CARGO_BIN_EXE_mock-acp-agent"), &ws, rce_mcp());
    c.turn_timeout = Duration::from_secs(30);
    (
        Env {
            _t: t,
            ws,
            ledger,
            vk,
        },
        c,
        l,
    )
}

const EDIT: &str = r#"mock:mcp propose_edit {"path":"counter.rce","edits":[{"op":"insert_item","after":"event Dec","text":"event Reset"},{"op":"insert_item","text":"step Counter on Reset -> 0"}]}"#;

#[test]
fn edits_go_through_rce_mcp_and_each_turn_is_signed() {
    let (e, c, l) = env();
    let mut a = AcpClient::start(c, l).unwrap();
    assert_eq!(a.agent_info["name"], "mock-acp-agent");
    let r = a.prompt(&format!("add reset\n{EDIT}")).unwrap();
    assert_eq!(r.stop_reason, "end_turn");
    assert_eq!(r.permissions_granted, ["propose_edit"]);
    assert_eq!(r.mediated, ["counter.rce"]);
    assert!(r.unmediated.is_empty() && r.methods_outside_grant.is_empty());
    assert_eq!(r.tool_calls[0].3, "completed");
    assert!(r.agent_text.contains("done") && r.usage.is_some());
    assert!(
        std::fs::read_to_string(e.ws.join("counter.rce"))
            .unwrap()
            .contains("event Reset")
    );
    // rce-mcp's own ledger has the checker-gated edit
    let sub = read_jsonl(&a.mcp_ledger).unwrap();
    assert!(
        sub.iter()
            .any(|x| x.body.draft.action == ActionKind::ProposeEdit
                && x.body.draft.verdict == Verdict::Pass)
    );
    assert_eq!(
        r.mcp_ledger_head.as_deref(),
        Some(sub.last().unwrap().hash.as_str())
    );
    // second turn, nothing to do
    let r2 = a.prompt("just talk").unwrap();
    assert!(r2.mediated.is_empty());
    let rs = read_jsonl(&e.ledger).unwrap();
    assert_eq!(rs.len(), 2);
    verify_chain(&rs, &e.vk, GENESIS).unwrap();
    assert_eq!(
        rs[0].body.draft.action,
        ActionKind::Other("acp.turn".into())
    );
    assert_eq!(
        rs[0].body.draft.detail["turn"]["mediated"],
        json!(["counter.rce"])
    );
    assert_eq!(
        rs[0].body.draft.e_sim_j, None,
        "uncalibrated: no invented number"
    );
    assert!(
        rs[0].body.draft.detail["features"]["net_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn fs_terminal_and_foreign_tools_are_outside_the_grant() {
    let (e, c, l) = env();
    let mut a = AcpClient::start(c, l).unwrap();
    let p = e.ws.join("counter.rce").display().to_string();
    let r = a
        .prompt(&format!(
            "mock:fs_write {p}\nmock:fs_read {p}\nmock:terminal\nmock:ask Bash execute\n\
             mock:ask mcp__rce__propose_edit execute\nmock:ask propose_edit edit\nmock:ask mcp__other__propose_edit edit\n\
             mock:ask mcp__rce__rm_rf edit"
        ))
        .unwrap();
    assert_eq!(
        r.methods_outside_grant,
        ["fs/write_text_file", "fs/read_text_file", "terminal/create"]
    );
    assert!(
        r.agent_text.contains("fs_write error=-32601")
            && r.agent_text.contains("fs_read error=-32601")
    );
    assert!(r.permissions_granted.is_empty());
    assert_eq!(r.permissions_denied.len(), 5, "{:?}", r.permissions_denied);
    assert_eq!(std::fs::read_to_string(&p).unwrap(), COUNTER);
}

#[test]
fn writes_outside_rce_mcp_are_reverted_and_flagged() {
    let (e, c, l) = env();
    let mut a = AcpClient::start(c, l).unwrap();
    let r = a
        .prompt("mock:rogue counter.rce garbage\nmock:rogue new.txt hi")
        .unwrap();
    assert_eq!(r.unmediated, ["counter.rce", "new.txt"]);
    assert_eq!(r.reverted, r.unmediated);
    assert_eq!(
        std::fs::read_to_string(e.ws.join("counter.rce")).unwrap(),
        COUNTER
    );
    assert!(!e.ws.join("new.txt").exists());
    let rs = read_jsonl(&e.ledger).unwrap();
    assert!(matches!(rs[0].body.draft.verdict, Verdict::Rejected { .. }));
}

#[test]
fn other_protocol_versions_are_not_spoken() {
    let (_e, mut c, l) = env();
    c.env.push(("MOCK_ACP_VERSION".into(), "2".into()));
    match AcpClient::start(c, l) {
        Err(AcpError::Version(v)) => assert_eq!(v, json!(2)),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("accepted v2"),
    }
}

#[test]
fn timeout_cancels_the_turn() {
    let (_e, mut c, l) = env();
    c.turn_timeout = Duration::from_millis(300);
    let mut a = AcpClient::start(c, l).unwrap();
    let r = a.prompt("mock:hang").unwrap();
    assert_eq!(r.stop_reason, "cancelled");
}

#[test]
fn acp_agent_as_mei_g4() {
    use ide_energy::sim::LinearSim;
    use mei_agent::agent::Status;
    use mei_agent::{Acceptance, Agent, AgentConfig, Task};
    use rce_mcp::{Server, ws::Workspace};
    let (e, c, l) = env();
    let acp = AcpClient::start(c, l).unwrap();
    let mei_ledger = e.ws.parent().unwrap().join("mei.jsonl");
    let server = Server::new(
        Workspace::new(&e.ws).unwrap(),
        Ledger::open(&mei_ledger, generate_key().unwrap()).unwrap(),
        LinearSim::default(),
        None,
    );
    let cfg = AgentConfig {
        budget_j: 10.0,
        external_g4: Some(Box::new(AcpG4(acp))),
        ..Default::default()
    };
    let mut agent = Agent::new(server, cfg);
    let task = Task {
        path: "counter.rce".into(),
        goal: format!("Add a Reset event\n{EDIT}"),
        acceptance: Some(Acceptance {
            items: vec!["event Reset".into(), "step Counter on Reset".into()],
            min_examples: 2,
        }),
    };
    let o = agent.run(&task).unwrap();
    assert_eq!(o.status, Status::Accepted { gear: 4 }, "{o:?}");
    assert_eq!(o.gears_tried, [0, 1, 2, 3, 4]);
    let mei = read_jsonl(&mei_ledger).unwrap();
    let g4 = mei
        .iter()
        .find(|r| r.body.draft.gear == "g4" && r.body.draft.action == ActionKind::ModelCall)
        .unwrap();
    assert_eq!(g4.body.draft.model.as_deref(), Some("acp:mock-acp-agent"));
    let turn = &g4.body.draft.detail["turn_receipt"];
    let ide = read_jsonl(&e.ledger).unwrap();
    assert_eq!(turn, &json!(ide[0].hash));
}

/// A file outside the workspace (a sibling of it, so the same filesystem
/// and permissions; only the jail can stop the read).
fn secret(e: &Env) -> PathBuf {
    let p = e.ws.parent().unwrap().join("secret.txt");
    std::fs::write(&p, "not for agents").unwrap();
    p
}

#[test]
fn jailed_agent_cannot_read_write_or_connect_outside_the_workspace() {
    let (e, c, l) = env();
    let s = secret(&e);
    let home = std::env::var("HOME").unwrap_or("/root".into());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut a = AcpClient::start(c, l).unwrap();
    assert_eq!(a.jail["backend"], "landlock+seccomp");
    let r = a
        .prompt(&format!(
            "mock:read {}\nmock:read /etc/passwd\nmock:read {home}/.bashrc\nmock:read {}/counter.rce\nmock:write_abs {}/planted.txt x\nmock:connect {port}\n{EDIT}",
            s.display(),
            e.ws.display(),
            e.ws.parent().unwrap().display(),
        ))
        .unwrap();
    let t = &r.agent_text;
    // EACCES (13) for files; EACCES/EPERM for the connect.
    assert!(t.starts_with("read err 13 read err 13 read err "), "{t}");
    assert!(t.contains(&format!("read ok {} ", COUNTER.len())), "{t}");
    assert!(t.contains("write_abs err 13 "), "{t}");
    assert!(
        t.contains("connect err 13 ") || t.contains("connect err 1 "),
        "{t}"
    );
    assert!(!e.ws.parent().unwrap().join("planted.txt").exists());
    // The rce-mcp it started inherits the jail and still works in the workspace.
    assert_eq!(r.permissions_granted, ["propose_edit"]);
    assert!(
        std::fs::read_to_string(e.ws.join("counter.rce"))
            .unwrap()
            .contains("event Reset")
    );
    let rs = read_jsonl(&e.ledger).unwrap();
    let d = &rs.last().unwrap().body.draft.detail;
    assert_eq!(d["jail"]["backend"], "landlock+seccomp");
    assert!(d["jail"]["rw"][0].as_str().unwrap().ends_with("/ws"));
    a.shutdown();
}

#[test]
fn unjailed_control_reads_the_same_file() {
    // The control case: without the jail the same read succeeds, so the
    // jailed failure above is the jail's doing.
    let (e, mut c, l) = env();
    let s = secret(&e);
    c.jail = None;
    let mut a = AcpClient::start(c, l).unwrap();
    assert!(a.jail.is_null());
    let r = a.prompt(&format!("mock:read {}", s.display())).unwrap();
    assert!(r.agent_text.starts_with("read ok 14 "), "{}", r.agent_text);
    a.shutdown();
}

#[test]
fn jail_grants_declared_ports_only() {
    let (_e, mut c, l) = env();
    let ok = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let no = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let (pok, pno) = (
        ok.local_addr().unwrap().port(),
        no.local_addr().unwrap().port(),
    );
    c.jail.as_mut().unwrap().tcp_connect = vec![pok];
    let mut a = AcpClient::start(c, l).unwrap();
    let r = a
        .prompt(&format!("mock:connect {pok}\nmock:connect {pno}"))
        .unwrap();
    assert!(
        r.agent_text.starts_with("connect ok 0 connect err 13"),
        "{}",
        r.agent_text
    );
    a.shutdown();
}

/// An HTTP server that answers whether it got the supervisor credential.
fn echo_http() -> u16 {
    use std::io::{BufRead, BufReader, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for mut s in l.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut head = String::new();
            let mut line = String::new();
            while r.read_line(&mut line).unwrap_or(0) > 2 {
                head += &line;
                line.clear();
            }
            let b = if head.contains("Authorization: Bearer supervisor-secret") {
                "cred=yes"
            } else {
                "cred=no"
            };
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
                b.len()
            );
        }
    });
    port
}

#[test]
fn jailed_agent_uses_mcp_over_http_and_the_network_proxy() {
    use ide_sandbox::net::{Approvals, Credentials, Decision, Fixed, NetManifest, NetProxy};
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    let (e, mut c, l) = env();
    let api = echo_http();
    let other = echo_http();
    let key = format!("http://127.0.0.1:{api}");
    let mut creds = Credentials::default();
    creds.set(&key, "Authorization", "Bearer supervisor-secret");
    let np = Arc::new(Mutex::new(NetProxy::new(
        NetManifest::parse(&format!(
            r#"{{"net":[{{"host":"127.0.0.1","port":{api},"proto":"http","reason":"model api"}}]}}"#
        ))
        .unwrap(),
        Approvals::default(),
        Box::new(Fixed(BTreeMap::from([(key.clone(), true)]))),
        creds,
    )));
    c.mcp_transport = acp_client::client::McpTransport::Http;
    c.mcp_args = vec!["--no-memory".into()];
    c.net = Some(np.clone());
    let mut a = AcpClient::start(c, l).unwrap();
    assert!(
        a.jail["mcp"]
            .as_str()
            .unwrap()
            .starts_with("http http://127.0.0.1:")
    );
    let mport: u16 = a.jail["mcp"]
        .as_str()
        .unwrap()
        .rsplit(':')
        .next()
        .unwrap()
        .split('/')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let pport: u16 = a.jail["proxy"]
        .as_str()
        .unwrap()
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let mut tcp: Vec<u64> = a.jail["tcp_connect"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    tcp.sort();
    let mut want = vec![mport as u64, pport as u64];
    want.sort();
    assert_eq!(tcp, want, "only the MCP and proxy ports");
    let r = a
        .prompt(&format!(
            "mock:fetch http://127.0.0.1:{api}/v1/models\nmock:fetch http://127.0.0.1:{other}/\nmock:connect {api}\n{EDIT}"
        ))
        .unwrap();
    let t = &r.agent_text;
    // Through the proxy: allowed, with the supervisor's credential (the
    // agent never saw it); undeclared: 403; direct: not performed, the jail denies it.
    assert!(t.starts_with("fetch 200 cred=yes "), "{t}");
    assert!(t.contains("fetch 403 "), "{t}");
    assert!(
        t.contains("connect err 13 ") || t.contains("connect err 1 "),
        "{t}"
    );
    // MCP over HTTP from inside the jail edits the workspace with receipts.
    assert_eq!(r.permissions_granted, ["propose_edit"]);
    assert!(!r.mcp_receipts.is_empty());
    assert!(
        std::fs::read_to_string(e.ws.join("counter.rce"))
            .unwrap()
            .contains("event Reset")
    );
    assert_eq!(r.net.len(), 2);
    assert!(r.net[0].credential_injected && r.net[0].decision == Decision::Allowed);
    assert_eq!(r.net[1].decision, Decision::Undeclared);
    let rs = read_jsonl(&e.ledger).unwrap();
    let d = &rs.last().unwrap().body.draft.detail;
    assert_eq!(d["turn"]["net"].as_array().unwrap().len(), 2);
    assert!(!d.to_string().contains("supervisor-secret"));
    a.shutdown();
}
