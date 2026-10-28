//! Headless tests of the view models, the photon view, the scene, the
//! language-server transports, and (when a GPU adapter exists, including
//! a software one) the offscreen wgpu render.
use rce_ide::app::{Options, open};
use rce_ide::lsp::Lsp;
use rce_ide::scene;
use rce_ide::view::{self, Queue};
use rce_ide::vm::{Action, Ide, ProposalStatus};
use std::path::PathBuf;

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

fn ws(memory: bool) -> (tempfile::TempDir, Ide) {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("counter.rce"), COUNTER).unwrap();
    let ide = open(&Options {
        root,
        file: "counter.rce".into(),
        lsp: None,
        memory,
    })
    .unwrap();
    (t, ide)
}

fn edits(v: serde_json::Value) -> Vec<rce_mcp::items::Edit> {
    serde_json::from_value(v).unwrap()
}

fn good() -> Vec<rce_mcp::items::Edit> {
    edits(
        serde_json::json!([{"op": "insert_item", "after": "event Dec", "text": "event Reset"},
                              {"op": "insert_item", "text": "step Counter on Reset -> 0"}]),
    )
}

#[test]
fn editor_shows_language_server_diagnostics() {
    let (_t, mut ide) = ws(false);
    assert!(ide.editor.diagnostics.is_empty());
    assert!(ide.editor.server.starts_with("rce"));
    let bad = COUNTER.replace(
        "step Counter on Inc -> n + 1",
        "step Counter on Nope -> n + 1",
    );
    ide.apply(Action::SetText(bad.clone())).unwrap();
    assert!(!ide.editor.diagnostics.is_empty());
    let line = bad.lines().position(|l| l.contains("Nope")).unwrap() + 1;
    assert!(
        ide.editor.diagnostics.iter().any(|d| d.line == line),
        "{:?}",
        ide.editor.diagnostics
    );
    let lines = ide.editor.lines();
    assert!(!lines[line - 1].2.is_empty());
    // Typing never writes the file.
    assert_eq!(
        std::fs::read_to_string(ide.server.ws.root.join("counter.rce")).unwrap(),
        COUNTER
    );
}

#[test]
fn stdio_and_in_process_language_server_agree() {
    let bad = COUNTER.replace(
        "step Counter on Inc -> n + 1",
        "step Counter on Nope -> n + 1",
    );
    let mut a = Lsp::stdio(&PathBuf::from(env!("CARGO_BIN_EXE_rce-lsp")), &[]).unwrap();
    let mut b = Lsp::in_process();
    assert!(a.is_stdio() && !b.is_stdio());
    assert_eq!(a.server_name, b.server_name);
    assert_eq!(a.open("file:///x.rce", COUNTER), vec![]);
    assert_eq!(b.open("file:///x.rce", COUNTER), vec![]);
    let da = a.change("file:///x.rce", &bad);
    assert!(!da.is_empty());
    assert_eq!(da, b.change("file:///x.rce", &bad));
}

#[test]
fn proposals_accept_decline_and_fit_with_receipts() {
    let (_t, mut ide) = ws(false);
    let ok = ide.propose("counter.rce", "mei g1", "add Reset", good());
    let bad = ide.propose("counter.rce", "mei g3", "break it", edits(serde_json::json!([{"op": "replace_item", "name": "step Counter on Inc", "text": "step Counter on Nope -> n"}])));
    let no = ide.propose(
        "counter.rce",
        "acp",
        "delete Dec",
        edits(serde_json::json!([{"op": "delete_item", "name": "event Dec"}])),
    );
    assert_eq!(ide.pending().len(), 3);
    ide.apply(Action::Accept(ok)).unwrap();
    assert!(matches!(
        ide.proposals[0].status,
        ProposalStatus::Accepted { .. }
    ));
    assert!(
        ide.editor.text.contains("event Reset"),
        "editor reloads after accept"
    );
    ide.apply(Action::Accept(bad)).unwrap();
    let ProposalStatus::Fit {
        diagnostics,
        receipt,
    } = &ide.proposals[1].status
    else {
        panic!("{:?}", ide.proposals[1].status)
    };
    assert!(!diagnostics.is_empty() && receipt.is_some());
    assert!(
        !std::fs::read_to_string(ide.server.ws.root.join("counter.rce"))
            .unwrap()
            .contains("Nope")
    );
    ide.apply(Action::Decline(no)).unwrap();
    assert!(matches!(
        ide.proposals[2].status,
        ProposalStatus::Declined { .. }
    ));
    assert!(
        ide.apply(Action::Accept(no)).is_err(),
        "not pending any more"
    );
    // Receipts panel: three actions, chain verified, no invented energy.
    let r = &ide.receipts;
    assert_eq!(r.rows.len(), 3);
    assert_eq!(r.chain_ok, Some(true));
    assert_eq!(r.rows[2].action, "proposal.decline");
    assert!(
        r.rows
            .iter()
            .all(|x| x.e_sim_j.is_none() && x.delta_j.is_none()),
        "uncalibrated simulator"
    );
    assert_eq!(r.totals.with_sim, 0);
    assert_eq!(
        r.totals.with_hw,
        r.rows.iter().filter(|x| x.e_meas_j.is_some()).count()
    );
}

#[test]
fn taps_on_the_photon_view_drive_the_models() {
    let (_t, mut ide) = ws(false);
    let id = ide.propose("counter.rce", "mei g1", "add Reset", good());
    let q = Queue::default();
    let root = view::root(&ide.panels(), &q, 1400.0, 900.0);
    let ctx = view::paint(&root, 1400.0, 900.0);
    let (x, y) = view::find(&ctx, &format!("accept {id}")).expect("accept button laid out");
    assert!(view::click(&ctx, x, y));
    assert_eq!(*q.borrow(), vec![Action::Accept(id)]);
    assert!(!view::click(&ctx, -5.0, -5.0));
    let acts: Vec<Action> = q.borrow_mut().drain(..).collect();
    for a in acts {
        ide.apply(a).unwrap();
    }
    assert!(matches!(
        ide.proposals[0].status,
        ProposalStatus::Accepted { .. }
    ));
    // Re-render: the accept button is gone, the receipt row is there.
    let ctx = view::paint(&view::root(&ide.panels(), &q, 1400.0, 900.0), 1400.0, 900.0);
    assert!(view::find(&ctx, &format!("accept {id}")).is_none());
    let s = scene::build(&ctx, 1400.0, 900.0);
    assert!(
        s.texts
            .iter()
            .any(|t| t.text.starts_with("accepted, receipt"))
    );
    assert!(
        s.texts.iter().any(|t| t.text.contains("propose_edit")),
        "receipt row"
    );
}

#[test]
fn memory_recall_panel() {
    let (_t, mut ide) = ws(true);
    ide.apply(Action::Recall("Counter".into())).unwrap();
    assert_eq!(ide.memory.query, "Counter");
    assert!(ide.memory.error.is_none(), "{:?}", ide.memory.error);
    assert!(!ide.memory.hits.is_empty(), "{:?}", ide.memory);
    assert!(!ide.receipts.rows.is_empty(), "recall writes a receipt");
}

#[test]
fn network_endpoints_are_approved_in_the_ide() {
    let (_t, mut ide) = ws(false);
    std::fs::write(
        ide.server.ws.root.join("rce.json"),
        r#"{"net":[{"host":"api.example.com","port":443,"proto":"tcp","reason":"fetch rates"}]}"#,
    )
    .unwrap();
    ide.apply(Action::Refresh).unwrap();
    assert_eq!(ide.net.pending.len(), 1);
    let q = Queue::default();
    let ctx = view::paint(&view::root(&ide.panels(), &q, 1400.0, 900.0), 1400.0, 900.0);
    let (x, y) = view::find(&ctx, "allow tcp://api.example.com:443").unwrap();
    view::click(&ctx, x, y);
    let acts: Vec<Action> = q.borrow_mut().drain(..).collect();
    for a in acts {
        ide.apply(a).unwrap();
    }
    assert!(ide.net.pending.is_empty());
    assert_eq!(
        ide.net.decided,
        vec![("tcp://api.example.com:443".to_string(), true)]
    );
    // The sandbox proxy reads the same decision.
    let a = ide_sandbox::net::Approvals::load(&ide.approvals_path);
    assert_eq!(a.decided.get("tcp://api.example.com:443"), Some(&true));
}

#[test]
fn scene_clips_and_orders() {
    let mut ctx = photon::RenderCtx::new();
    let r = |x, y, w, h| photon::LayoutRect::new(x, y, w, h);
    ctx.push_clip(&r(10.0, 10.0, 50.0, 50.0));
    ctx.fill_rect(&r(0.0, 0.0, 100.0, 100.0), photon::Color::WHITE);
    ctx.fill_rect(&r(200.0, 200.0, 5.0, 5.0), photon::Color::WHITE); // fully clipped
    ctx.fill_text(&r(12.0, 12.0, 40.0, 14.0), "hi", 12.0, photon::Color::BLACK);
    ctx.pop_clip();
    let s = scene::build(&ctx, 300.0, 300.0);
    assert_eq!(s.quads.len(), 1);
    assert_eq!(
        s.quads[0].clip,
        scene::R {
            x: 10.0,
            y: 10.0,
            w: 50.0,
            h: 50.0
        }
    );
    assert_eq!(s.texts.len(), 1);
    assert!(s.quads[0].order < s.texts[0].order);
}

#[cfg(feature = "gpu")]
#[test]
fn offscreen_render_or_honest_skip() {
    let mut r = match rce_ide::gpu::Renderer::headless() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP offscreen render: {e}");
            return;
        }
    };
    eprintln!(
        "adapter: {} ({:?}, {:?})",
        r.adapter_info.name, r.adapter_info.backend, r.adapter_info.device_type
    );
    let (_t, ide) = ws(false);
    let q = Queue::default();
    let (w, h) = (800u32, 500u32);
    let ctx = view::paint(
        &view::root(&ide.panels(), &q, w as f32, h as f32),
        w as f32,
        h as f32,
    );
    let s = scene::build(&ctx, w as f32, h as f32);
    let (px, st) = r.render_rgba(&s, w, h).unwrap();
    assert_eq!(px.len(), (w * h * 4) as usize);
    assert!(st.glyphs > 100, "{st:?}");
    let at =
        |x: u32, y: u32| px[((y * w + x) * 4) as usize..((y * w + x) * 4 + 3) as usize].to_vec();
    // The editor panel's empty lower area is the PANEL colour (sRGB bytes).
    let panel = view::PANEL.to_array();
    let p = at(20, h - 10);
    for (c, b) in p.iter().zip(panel) {
        assert!((*c as f32 / 255.0 - b).abs() < 0.02, "{p:?} vs {panel:?}");
    }
    // Text pixels exist in the editor panel.
    let bright = (0..h / 2)
        .flat_map(|y| (0..w / 2).map(move |x| (x, y)))
        .filter(|&(x, y)| at(x, y)[0] > 180)
        .count();
    assert!(bright > 200, "{bright} bright pixels");
}

#[test]
fn editing_rediagnoses_and_saves_only_through_an_accepted_proposal() {
    use rce_ide::vm::{EditOp, Motion};
    let (_t, mut ide) = ws(false);
    ide.set_rows(8);
    ide.apply(Action::Edit(EditOp::Move {
        to: Motion::DocEnd,
        select: false,
    }))
    .unwrap();
    assert_eq!(ide.editor.view.lines.len(), 8);
    assert!(ide.editor.view.first > 0, "scrolled to the cursor");
    for op in [
        EditOp::Newline,
        EditOp::Insert("event Reset".into()),
        EditOp::Newline,
        EditOp::Insert("step Counter on Rset -> 0".into()),
    ] {
        ide.apply(Action::Edit(op)).unwrap();
    }
    assert!(ide.editor.dirty);
    assert!(!ide.editor.diagnostics.is_empty(), "checker flags the typo");
    // Fix the typo by selecting the word and typing over it.
    for _ in 0..2 {
        ide.apply(Action::Edit(EditOp::Move {
            to: Motion::WordLeft,
            select: false,
        }))
        .unwrap();
    }
    ide.apply(Action::Edit(EditOp::Move {
        to: Motion::WordRight,
        select: true,
    }))
    .unwrap();
    ide.apply(Action::Edit(EditOp::Insert("Reset".into())))
        .unwrap();
    assert!(
        ide.editor.diagnostics.is_empty(),
        "{:?}",
        ide.editor.diagnostics
    );
    let disk = ide.server.ws.root.join("counter.rce");
    assert_eq!(std::fs::read_to_string(&disk).unwrap(), COUNTER);
    ide.apply(Action::Save).unwrap();
    let id = ide.pending()[0].id;
    ide.apply(Action::Accept(id)).unwrap();
    assert!(
        std::fs::read_to_string(&disk)
            .unwrap()
            .contains("event Reset\nstep Counter on Reset -> 0")
    );
    assert!(!ide.editor.dirty);
    // Undo still works across the reload, back to the buffer before it.
    ide.apply(Action::Edit(EditOp::Undo)).unwrap();
    assert!(ide.editor.dirty);
}

#[test]
fn clicks_in_the_editor_map_to_line_and_column() {
    let (_t, mut ide) = ws(false);
    let m = view::Metrics::default();
    let q = Queue::default();
    ide.set_rows(view::editor_rows(900.0, &m));
    let root = view::root_with(&ide.panels(), &q, &m, 1400.0, 900.0);
    let ctx = view::paint(&root, 1400.0, 900.0);
    let a = ctx
        .annotations
        .iter()
        .find(|a| a.label.as_deref() == Some("editor-text"))
        .unwrap()
        .rect;
    let x = a.x + 5.0 * m.char_w + 8.0 + 3.0 * m.char_w + 1.0;
    let y = a.y + 4.0 * m.line_h + 2.0;
    assert_eq!(view::editor_hit(&ctx, &ide.editor, &m, x, y), Some((4, 3)));
    assert_eq!(view::editor_hit(&ctx, &ide.editor, &m, 1300.0, 10.0), None);
}

#[test]
fn demo_populates_every_panel_from_real_services() {
    let (_t, mut ide) = ws(true);
    rce_ide::demo::populate(&mut ide, "Counter").unwrap();
    assert!(ide.receipts.rows.len() >= 7);
    assert_eq!(ide.receipts.chain_ok, Some(true));
    assert_eq!(ide.proposals.len(), 4);
    assert!(!ide.memory.hits.is_empty());
    assert_eq!(ide.net.pending.len(), 1);
    assert_eq!(ide.net.decided.len(), 1);
    assert!(ide.editor.dirty && !ide.editor.diagnostics.is_empty());
    assert_eq!(ide.buffer.selected_text().as_deref(), Some("Incr"));
}

/// A clipboard shared with the test (stands in for the OS one).
#[derive(Clone, Default)]
struct Shared(std::rc::Rc<std::cell::RefCell<Option<String>>>);
impl rce_ide::clip::Clipboard for Shared {
    fn get(&mut self) -> Option<String> {
        self.0.borrow().clone()
    }
    fn set(&mut self, t: &str) {
        *self.0.borrow_mut() = Some(t.into());
    }
    fn name(&self) -> &'static str {
        "test"
    }
}

#[test]
fn copy_cut_paste_go_through_the_clipboard() {
    use rce_ide::vm::{EditOp, Motion};
    let (_t, mut ide) = ws(false);
    let cb = Shared::default();
    ide.clip = Box::new(cb.clone());
    let ed = |ide: &mut rce_ide::vm::Ide, op| ide.apply(Action::Edit(op)).unwrap();
    // Copy with no selection leaves the clipboard alone.
    *cb.0.borrow_mut() = Some("from another app".into());
    ed(&mut ide, EditOp::Copy);
    assert_eq!(cb.0.borrow().as_deref(), Some("from another app"));
    // Paste takes what another app put there.
    ed(
        &mut ide,
        EditOp::Move {
            to: Motion::DocStart,
            select: false,
        },
    );
    ed(&mut ide, EditOp::Paste(None));
    assert!(ide.buffer.text().starts_with("from another app"));
    // Select it, cut: the system clipboard has it and the buffer doesn't.
    ed(
        &mut ide,
        EditOp::Move {
            to: Motion::DocStart,
            select: true,
        },
    );
    ed(&mut ide, EditOp::Cut);
    assert_eq!(cb.0.borrow().as_deref(), Some("from another app"));
    assert!(!ide.buffer.text().starts_with("from another app"));
    // Copy a line.
    ed(
        &mut ide,
        EditOp::Move {
            to: Motion::LineEnd,
            select: true,
        },
    );
    ed(&mut ide, EditOp::Copy);
    let line = ide.buffer.text().lines().next().unwrap().to_string();
    assert_eq!(cb.0.borrow().as_deref(), Some(line.as_str()));
}

#[test]
fn system_clipboard_falls_back_without_a_display() {
    use rce_ide::clip::{Clipboard, System};
    // Headless: the in-process fallback; with a display (this box has an
    // X server): the real X11 clipboard via x11rb. Both round-trip.
    let mut c = System::new();
    c.set("rce clip test");
    assert_eq!(c.get().as_deref(), Some("rce clip test"), "{}", c.name());
}

#[test]
fn calibrate_runs_the_real_workloads_under_a_fake_meter() {
    use ide_energy::calib::{Plan, load_sim};
    use ide_energy::meter::Meter;
    use ide_energy::sim::Simulator;
    let t = tempfile::tempdir().unwrap();
    let out = t.path().join("calibration.json");
    let mut meters: Vec<Box<dyn Meter>> = vec![Box::new(rce_ide::calibrate::FakeMeter::new(10.0))];
    let plan = Plan {
        reps: 2,
        settle: std::time::Duration::ZERO,
    };
    let (c, files) = rce_ide::calibrate::run(&mut meters, &plan, &out).unwrap();
    assert_eq!(c.workloads.len(), 10);
    assert_eq!(c.workloads[0], "empty");
    let m = &c.meters[0];
    assert_eq!(m.meter, "fake");
    assert_eq!(m.runs.len(), 20);
    // Real checker features: passes and nodes vary independently.
    let terms = &m.calibrated_terms["checker/-"];
    assert!(
        terms.contains(&"form_nodes".to_string()) && terms.contains(&"checker_passes".to_string()),
        "{terms:?}"
    );
    // 10 W over wall time: every window measured, none negative. (Wall
    // time under a parallel test run is too noisy to assert a fit quality;
    // coefficient recovery is tested in ide-energy with exact fakes.)
    assert!(
        m.runs
            .iter()
            .all(|r| r.joules >= 0.0 && r.joules.is_finite())
    );
    assert!(m.r2.is_finite());
    assert_eq!(files.len(), 2);
    let sim = load_sim(&out).unwrap();
    let f = &m
        .runs
        .iter()
        .find(|r| r.workload == "check large x80")
        .unwrap()
        .features;
    assert!(f.checker_passes >= 80 && f.form_nodes > 0);
    assert!(sim.predict(f).is_some());
}

#[test]
fn calibrate_command_with_a_fake_meter_writes_the_file() {
    let t = tempfile::tempdir().unwrap();
    let out = t.path().join("c.json");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce-ide"))
        .args([
            "telemetry",
            "calibrate",
            "--reps",
            "1",
            "--fake-meter",
            "5",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    let so = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "{so}{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(so.contains("fake: R²") && so.contains("checker/-:"), "{so}");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(v["meter"], "fake");
    assert!(v["fits"]["checker/-"]["coef"].is_array());
}
