//! HC-29: frames are derived from D, and a renderer's confirmation is an
//! input. A program emits `Frame(view)` after a step; `rce host run` and
//! `serve` write each frame to the sink granted by `frames FILE`; `rce host
//! replay frames` re-derives every frame from D with every adapter
//! disconnected, so `ui.rules` is checked on exactly what a session showed;
//! `presented FILE` turns a renderer's confirmation into `host.Presented`
//! and the bound `Presented(seq, digest)`.
use rce_spec_check::frame;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn prog() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/frames.rce")
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-hc29-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rce(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(args).env("RCE_STATUS", "off").stdin(Stdio::null()).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// The sink lines `rce host replay frames` prints, as the sink holds them.
fn replayed_sink(out: &str) -> String {
    out.lines().filter_map(|l| l.strip_prefix("frame ")).map(|l| format!("{l}\n")).collect()
}

/// A program's frames replay byte-identically: the sink written live over
/// two runs on one D equals the sink lines replay re-derives from D alone,
/// and each view digest equals the one `ui.view.view_digest` gives (the
/// program writes it in a `Display` beside each frame).
#[test]
fn frames_replay_byte_identically() {
    let d = dir("bytes");
    let (sink, log) = (d.join("frames.jsonl"), d.join("door.log"));
    let (c, out1, err) = rce(&["host", "run", "frames", s(&sink), s(&prog()), s(&log), "Focus, Open, Tick, Tick"]);
    assert_eq!(c, 0, "{out1}{err}");
    let (c, out2, err) = rce(&["host", "run", "frames", s(&sink), s(&prog()), s(&log), "Dim, Close"]);
    assert_eq!(c, 0, "{out2}{err}");
    let live = std::fs::read_to_string(&sink).unwrap();
    assert_eq!(live.lines().count(), 4, "{live}");
    let (c, rep, err) = rce(&["host", "replay", "frames", s(&prog()), s(&log)]);
    assert_eq!(c, 0, "{rep}{err}");
    assert!(rep.starts_with("replay ok"), "{rep}");
    assert_eq!(replayed_sink(&rep), live, "replay must re-derive the live sink byte for byte");
    // the host's digests are ui.view's: the program printed view.view_digest
    let refs: Vec<String> = format!("{out1}{out2}").lines().filter_map(|l| l.trim().strip_prefix("emit Display(data: \"")).map(|x| x[..64].to_string()).collect();
    let fs: Vec<frame::Frame> = live.lines().map(|l| frame::from_line(l).unwrap()).collect();
    assert_eq!(fs.iter().map(|f| f.view_digest.clone()).collect::<Vec<_>>(), refs);
    // the first frame starts the trace; the ops of the later ones are view.diff's
    assert_eq!(fs[0].ev, "focus toggle");
    let v: Vec<serde_json::Value> = live.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(v[0]["ops"].is_null());
    assert!(v[2]["ops"].as_array().unwrap().iter().any(|o| o["op"] == "set" && o["name"] == "fg" && o["value"] == "#777777"), "{}", v[2]);
    // replay has no adapter: the same D replays with no sink at all
    let (c, plain, _) = rce(&["host", "replay", s(&prog()), s(&log)]);
    assert_eq!(c, 0);
    assert!(!plain.contains("frame {"), "{plain}");
    let _ = std::fs::remove_dir_all(&d);
}

/// `ui.rules` run on the replayed frames gives the same report as on the
/// live ones: the report of the sink's frames, as one trace, equals the one
/// `rce host replay frames` prints, and the dimmed frame fails 1.4.3.
#[test]
fn ui_rules_report_is_the_same_on_live_and_replayed_frames() {
    let d = dir("rules");
    let (sink, log) = (d.join("frames.jsonl"), d.join("door.log"));
    let (c, out, err) = rce(&["host", "run", "frames", s(&sink), s(&prog()), s(&log), "Focus, Open, Dim"]);
    assert_eq!(c, 0, "{out}{err}");
    let live: Vec<frame::Frame> = std::fs::read_to_string(&sink).unwrap().lines().map(|l| frame::from_line(l).unwrap()).collect();
    let r = frame::rules_report(&live).unwrap();
    assert_eq!(r.verdicts.len(), 55, "{:?}", r.verdicts);
    assert!(r.verdicts.iter().any(|v| v.starts_with("1.4.3 fails")), "{:?}", r.verdicts);
    assert!(r.findings.iter().any(|(i, t)| *i == 2 && t.starts_with("1.4.3 fails")), "{:?}", r.findings);
    assert!(!r.findings.iter().any(|(i, _)| *i < 2), "the first two frames meet every rule: {:?}", r.findings);
    let (c, rep, err) = rce(&["host", "replay", "frames", s(&prog()), s(&log)]);
    assert_eq!(c, 0, "{rep}{err}");
    let mut want = String::new();
    for (i, l) in std::fs::read_to_string(&sink).unwrap().lines().enumerate() {
        want += &format!("frame {l}\n");
        for (_, t) in r.findings.iter().filter(|(k, _)| *k == i) {
            want += &format!("  finding {t}\n");
        }
    }
    want += &format!("ui.rules: {} frames, {} findings\n", live.len(), r.findings.len());
    for (_, t) in r.findings.iter().filter(|(k, _)| *k >= live.len()) {
        want += &format!("  finding {t}\n");
    }
    for v in &r.verdicts {
        want += &format!("  {v}\n");
    }
    assert!(rep.contains(&want), "replay:\n{rep}\nlive:\n{want}");
    let _ = std::fs::remove_dir_all(&d);
}

/// Without `ev` and `ms` the host derives them: `start` for the first frame
/// of D, `tick` after a `Tick`, the input's name otherwise, and 16 ms per
/// `Tick` entry since the previous frame (HC-14, HC-29).
#[test]
fn step_words_and_times_are_derived_from_d() {
    let d = dir("clock");
    let src = "module plain\nuse ui.view\n\nstruct Frame\n    view: View\n\nstate S\n    n: i64 = 0\n\nevent Tick\nevent Go\n\nstep S on Tick\n    n += 1\n    if n % 2 == 0\n        emit Frame(view: view.view_new(\"div\", \"\"))\nstep S on Go\n    emit Frame(view: view.view_new(\"main\", \"\"))\n";
    let p = d.join("plain.rce");
    std::fs::write(&p, src).unwrap();
    let (sink, log) = (d.join("f.jsonl"), d.join("plain.log"));
    let (c, out, err) = rce(&["host", "run", "frames", s(&sink), s(&p), s(&log), "Go, Tick, Tick, Tick, Tick, Go"]);
    assert_eq!(c, 0, "{out}{err}");
    let fs: Vec<frame::Frame> = std::fs::read_to_string(&sink).unwrap().lines().map(|l| frame::from_line(l).unwrap()).collect();
    let got: Vec<(String, i64)> = fs.iter().map(|f| (f.ev.clone(), f.ms)).collect();
    assert_eq!(got, vec![("start".into(), 0), ("tick".into(), 32), ("tick".into(), 32), ("Go".into(), 0)]);
    let (c, rep, _) = rce(&["host", "replay", "frames", s(&p), s(&log)]);
    assert_eq!(c, 0);
    assert_eq!(replayed_sink(&rep), std::fs::read_to_string(&sink).unwrap());
    let _ = std::fs::remove_dir_all(&d);
}

/// The confirmation round-trips: `Host::presented` appends
/// `host.Presented` {digest, seq} and delivers `Presented(seq, digest)`;
/// replay re-derives the state that counted it; a confirmation that names
/// no frame of D appends nothing; and a D whose `host.Presented` names no
/// frame does not replay.
#[test]
fn presented_confirmation_round_trips() {
    use rce_spec_check::host::{Entry, Host, replay_report};
    let d = dir("ack");
    let log = d.join("door.log");
    let src = std::fs::read_to_string(prog()).unwrap();
    let m = rce_spec_check::program(&src).unwrap();
    let (seq, dig) = {
        let mut h = Host::open_file_with(&m, &log, vec![]).unwrap();
        let a = h.admit(&src, "Focus, Open").unwrap();
        let f = a.iter().flat_map(|x| x.frames.iter()).last().unwrap().clone().unwrap();
        let before = h.len();
        assert!(h.presented(f.seq, "00").is_err());
        assert!(h.presented(f.seq + 1, &f.frame_digest).is_err());
        assert_eq!(h.len(), before, "a refused confirmation appends nothing");
        let got = h.presented(f.seq, &f.frame_digest).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].event, format!("Presented(seq: {}, digest: \"{}\")", f.seq, f.frame_digest));
        (f.seq, f.frame_digest)
    };
    let text = std::fs::read_to_string(&log).unwrap();
    let es: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap()).collect();
    let i = es.iter().position(|e| e.kind() == "host.Presented").unwrap();
    let p: serde_json::Value = serde_json::from_str(&es[i].event).unwrap();
    assert_eq!(p["payload"], serde_json::json!({"digest": dig, "seq": seq}));
    assert_eq!(es[i + 1].kind(), "Presented");
    let r = replay_report(&m, &text).unwrap();
    let n = r.state.iter().flat_map(|(_, fs)| fs.iter()).find(|(f, _, _)| f == "presented").map(|(_, v, _)| v.show());
    assert_eq!(n.as_deref(), Some("1"));
    // the same confirmation, chained into a D whose frame it does not name
    let mut forged: Vec<String> = vec![];
    let (mut prev, mut hash) = (rce_spec_check::interp::D_GENESIS.to_string(), String::new());
    for e in es.iter().take(1) {
        forged.push(e.line());
        prev = e.hash.clone();
        hash = e.state_hash.clone();
    }
    let bad = Entry::new(&m, 1, es[i].event.clone(), &prev, &hash);
    forged.push(bad.line());
    let err = replay_report(&m, &(forged.join("\n") + "\n")).unwrap_err();
    assert!(err.contains("host.Presented names no frame"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

/// `rce host serve frames FILE presented FILE`: a renderer reads the sink,
/// appends `SEQ FRAME_DIGEST` for each frame it showed, and the host loop
/// records each as `host.Presented` and the bound `Presented`; the session
/// replays with the count.
#[cfg(unix)]
#[test]
fn serve_reads_a_renderers_confirmations() {
    let d = dir("serve");
    let (sink, acks, log) = (d.join("frames.jsonl"), d.join("acks"), d.join("door.log"));
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve", "frames", s(&sink), "presented", s(&acks), s(&prog()), s(&log)])
        .env("RCE_STATUS", "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    writeln!(i, "Focus").unwrap();
    writeln!(i, "Open").unwrap();
    let until = |what: &str, f: &dyn Fn() -> bool| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !f() {
            assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    };
    until("two frames", &|| std::fs::read_to_string(&sink).is_ok_and(|t| t.lines().count() == 2));
    let fs: Vec<frame::Frame> = std::fs::read_to_string(&sink).unwrap().lines().map(|l| frame::from_line(l).unwrap()).collect();
    let mut a = std::fs::OpenOptions::new().append(true).open(&acks).unwrap();
    for f in &fs {
        writeln!(a, "{} {}", f.seq, f.frame_digest).unwrap();
    }
    writeln!(a, "not a confirmation").unwrap();
    drop(a);
    until("two confirmations in D", &|| std::fs::read_to_string(&log).is_ok_and(|t| t.matches("\"type\":\"Presented\"").count() == 2));
    drop(i);
    let o = c.wait_with_output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stderr).contains("is not SEQ FRAME_DIGEST"));
    let (code, rep, _) = rce(&["host", "replay", "frames", s(&prog()), s(&log)]);
    assert_eq!(code, 0, "{rep}");
    assert!(rep.contains("presented = 2"), "{rep}");
    assert_eq!(replayed_sink(&rep), std::fs::read_to_string(&sink).unwrap());
    let _ = std::fs::remove_dir_all(&d);
}
