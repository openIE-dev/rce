//! W1c adapters: file effects under a granted root and a real PTY, each
//! request answered once and logged, and replay with adapters disconnected
//! reaching the same state (HC-3, HC-5, HC-8).
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::path::Path;

fn prog(n: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host").join(n)).unwrap()
}
fn log_text(st: &MemStore) -> String {
    st.lines.iter().map(|l| format!("{l}\n")).collect()
}

#[test]
fn file_effects_stay_under_the_granted_root() {
    let src = prog("files.rce");
    let m = parse(&src).unwrap();
    let base = std::env::temp_dir().join(format!("rce-fx-files-{}", std::process::id()));
    let root = base.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(base.join("secret.txt"), "no").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(base.join("secret.txt"), root.join("link")).unwrap();
    let ads: Vec<Box<dyn Adapter>> = rce_spec_check::adapters::file_adapters(&root).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    h.admit(&src, r#"Save(path: "a.txt", body: "hello\nworld"), Save(path: "b.txt", body: ""), Load(path: "a.txt"), Ls(path: "")"#).unwrap();
    let ok_state = format!("{:?}", h.state());
    assert!(ok_state.contains("hello\\nworld") && ok_state.contains("written\", Int(2"), "{ok_state}");
    for bad in ["../secret.txt", "/etc/hosts", "link", ""] {
        h.admit(&src, &format!("Load(path: \"{bad}\")")).unwrap();
    }
    let live = h.state_hash().to_string();
    let s = format!("{:?}", h.state());
    drop(h);
    assert!(s.contains("errors\", Int(4"), "{s}");
    assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "hello\nworld");
    let text = log_text(&st);
    assert!(!text.contains("\"no\""), "the secret never reaches D");
    let (_, h2, _) = replay(&m, &text).unwrap();
    assert_eq!(h2, live);
    let _ = std::fs::remove_dir_all(&base);
}

#[cfg(unix)]
#[test]
fn pty_session_echoes_and_replays() {
    let src = prog("pty.rce");
    let m = parse(&src).unwrap();
    let ads: Vec<Box<dyn Adapter>> = rce_spec_check::adapters::pty::pty_adapters().into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    h.admit(&src, r#"Start, Send(data: "ping\n")"#).unwrap();
    let mut seen = false;
    for _ in 0..100 {
        h.admit(&src, "Poll").unwrap();
        let s = format!("{:?}", h.state());
        // the terminal echoes the input and cat writes it back
        if s.matches("ping").count() >= 2 {
            seen = true;
            break;
        }
    }
    assert!(seen, "{:?}", h.state());
    h.admit(&src, "Stop").unwrap();
    let s = format!("{:?}", h.state());
    assert!(s.contains("closed\", Bool(true)") && s.contains("errors\", Int(0"), "{s}");
    let live = h.state_hash().to_string();
    drop(h);
    let (_, h2, _) = replay(&m, &log_text(&st)).unwrap();
    assert_eq!(h2, live);
}
