//! Offscreen screenshots of the 3D graph mode, the hardware layer and the
//! telemetry dashboard, from a real demo workspace:
//!
//! - the Form: RCE spec examples, checked by the reference checker, grown
//!   over four real versions (one of them an MEI proposal accepted through
//!   rce-mcp, via `demo::populate`);
//! - receipts: the ledger rce-mcp wrote for those actions;
//! - facts: written through the `remember` tool (one contradicted);
//! - hardware: discovered on this machine; telemetry: probed on this machine.
//!
//! This machine has no power meter, so the receipts carry no joules. The
//! energy overlay uses SYNTHETIC actions, marked as such on screen and in
//! the data (`synthetic: true`), and the dashboard adds SYNTHETIC rows next
//! to the real probe statuses.
//!
//! `cargo run -p rce-ide --example graph3d_shots -- OUT_DIR`
use rce_graph3d::dash::SourceRow;
use rce_graph3d::layout::Mode;
use rce_graph3d::native::Sampler;
use rce_graph3d::overlay::Action as Act;
use rce_graph3d::render::Renderer3d;
use rce_ide::gpu::Renderer;
use rce_ide::graph_panel::{self, GraphView};
use rce_ide::scene::R;
use serde_json::json;

fn main() -> Result<(), String> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "screens".into());
    let ex = std::env::var("RCE_EXAMPLES")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../spec/examples").into());
    let rd =
        |f: &str| std::fs::read_to_string(format!("{ex}/{f}")).map_err(|e| format!("{f}: {e}"));
    let dir = tempfile_dir()?;
    std::fs::write(dir.join("counter.rce"), rd("01_counter.rce")?).map_err(|e| e.to_string())?;
    let o = rce_ide::app::Options {
        root: dir.clone(),
        file: "counter.rce".into(),
        lsp: None,
        memory: true,
    };
    let mut ide = rce_ide::app::open(&o)?;
    let now = rce_memory::now_ms;
    let mut hist = vec![];
    let t0 = now();
    hist.push((t0, "v1 counter".to_string(), ide.server.ws.program()));
    std::thread::sleep(std::time::Duration::from_millis(30));
    std::fs::write(dir.join("bank.rce"), rd("04_bank_account.rce")?).map_err(|e| e.to_string())?;
    hist.push((now(), "v2 +bank".into(), ide.server.ws.program()));
    let call = |ide: &mut rce_ide::vm::Ide, tool: &str, a: serde_json::Value| {
        ide.server.call(tool, &a).map(|_| ()).map_err(|(_, m)| m)
    };
    call(
        &mut ide,
        "remember",
        json!({"subject": "Counter", "predicate": "owner", "object": "mei", "provenance": "inferred"}),
    )?;
    rce_ide::demo::populate(&mut ide, "Counter")?;
    hist.push((now(), "v3 MEI: +Reset".into(), ide.server.ws.program()));
    std::thread::sleep(std::time::Duration::from_millis(30));
    std::fs::write(dir.join("traffic.rce"), rd("17_traffic_light.rce")?)
        .map_err(|e| e.to_string())?;
    std::fs::write(dir.join("guarded.rce"), rd("30_state_machine_guarded.rce")?)
        .map_err(|e| e.to_string())?;
    call(
        &mut ide,
        "remember",
        json!({"subject": "Counter", "predicate": "owner", "object": "david", "provenance": "extracted"}),
    )?;
    call(&mut ide, "rce_check", json!({}))?;
    hist.push((
        now(),
        "v4 +traffic +guarded".into(),
        ide.server.ws.program(),
    ));

    // Live telemetry: every source this machine has, sampled for ~1 s.
    let mut s = Sampler::open();
    for _ in 0..10 {
        s.tick();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let mut tl = graph_panel::timeline_from_ide(&ide, &hist, &s.rows, s.cpu_util);
    let real_receipts = tl.actions.len();
    // SYNTHETIC energy (no meter here): per item, E_sim and E_meas on the CPU package, one anomaly.
    let pkg = tl
        .hardware
        .nodes
        .iter()
        .find(|n| n.kind == rce_graph3d::Kind::Package)
        .map(|n| n.id.clone());
    let gpu = tl
        .hardware
        .nodes
        .iter()
        .find(|n| n.kind == rce_graph3d::Kind::Gpu)
        .map(|n| n.id.clone());
    let (a, b) = (t0, now());
    let plan: &[(&[&str], f64, f64, bool)] = &[
        (&["Inc"], 0.8, 0.85, false),
        (&["Dec", "Underflow"], 1.1, 1.0, false),
        (&["Counter"], 2.4, 2.6, false),
        (&["Deposit", "Account"], 1.6, 1.7, false),
        (&["Withdraw", "Account", "Insufficient"], 2.0, 2.1, false),
        (&["Reset"], 0.6, 4.9, true), // anomaly: measured far above predicted
        (&["Inc", "Counter"], 0.9, 0.95, false),
        (&["Tick", "Light"], 1.3, 1.25, true),
        (&["Door", "Open", "Close"], 1.8, 1.9, false),
        (&["Lock", "Unlock"], 1.2, 1.15, false),
    ];
    for (i, (names, sim, meas, use_gpu)) in plan.iter().enumerate() {
        let dev = if *use_gpu {
            gpu.clone().or(pkg.clone())
        } else {
            pkg.clone()
        };
        tl.actions.push(Act {
            seq: 10_000 + i as u64,
            t_ms: a + (b - a) * (i as i64 + 1) / (plan.len() as i64 + 1),
            action: "synthetic".into(),
            e_sim_j: Some(*sim),
            e_meas_j: Some(*meas),
            source: Some(
                if *use_gpu && gpu.is_some() {
                    "nvml"
                } else {
                    "rapl"
                }
                .into(),
            ),
            delta_j: Some(meas - sim),
            device: dev,
            names: names.iter().map(|s| s.to_string()).collect(),
            synthetic: true,
        });
    }
    eprintln!(
        "graph3d_shots: {} versions, {} real receipts, {} facts, {} hardware nodes, {} sources",
        tl.versions.len(),
        real_receipts,
        tl.facts.len(),
        tl.hardware.nodes.len(),
        s.rows.len()
    );
    let mut r = Renderer::headless()?;
    let mut r3 = Renderer3d::new(&r.device, r.format);
    let (w, h) = (1600u32, 1000u32);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let note = "energy: SYNTHETIC (no meter here) · Form, receipts, facts: real";

    // 1. The code graph over time, with the energy overlay.
    let mut v = GraphView::new(tl.clone(), false);
    v.title = "Form graph".into();
    v.note = note.into();
    v.k_sigma = 2.0;
    v.scrub(1.0);
    v.cam.rotate(0.3, 0.1);
    v.select_label("Counter");
    v.set_cursor("counter.rce", ide.buffer.visible().cursor.0 + 1);
    let px = v.render_rgba(&mut r, &mut r3, w, h)?;
    rce_ide::png::write(&format!("{out}/graph3d.png"), w, h, &px)?;
    // Scrubbed back to v2, layered by time, for the record.
    let mut vt = GraphView::new(tl.clone(), false);
    vt.note = note.into();
    vt.set_mode(Mode::ByTime);
    vt.scrub(0.4);
    let px = vt.render_rgba(&mut r, &mut r3, w, h)?;
    rce_ide::png::write(&format!("{out}/graph3d-time.png"), w, h, &px)?;

    // 2. The hardware layer: code above, this machine's topology below, code -> hardware edges.
    let mut hv = GraphView::new(tl.clone(), true);
    hv.title = format!(
        "Code + hardware ({})",
        if cfg!(target_os = "macos") {
            "sysctl"
        } else {
            "sysfs"
        }
    );
    hv.note = "hardware: discovered here · energy: SYNTHETIC".into();
    hv.k_sigma = 2.0;
    hv.style.max_labels = 60;
    hv.cam.rotate(0.0, 0.25);
    if let Some(p) = &pkg {
        let lab = hv
            .snap
            .graph
            .nodes
            .iter()
            .find(|n| &n.id == p)
            .map(|n| n.label.clone())
            .unwrap_or_default();
        hv.select_label(&lab);
    }
    let px = hv.render_rgba(&mut r, &mut r3, w, h)?;
    rce_ide::png::write(&format!("{out}/hardware.png"), w, h, &px)?;

    // 3. The telemetry dashboard: every probed source with its real status, plus SYNTHETIC rows.
    let mut rows = s.rows.clone();
    let syn = |tag: &str, dom: &str, base: f64| {
        let series: Vec<(i64, f64)> = (0..40)
            .map(|i| {
                (
                    i * 250,
                    base + (i as f64 * 0.7).sin() * base * 0.25 + if i == 31 { base } else { 0.0 },
                )
            })
            .collect();
        SourceRow {
            tag: format!("demo-{tag}"),
            domain: dom.into(),
            detail: "SYNTHETIC series for the screenshot; not a measurement".into(),
            status: "available".into(),
            resolution: "250 ms".into(),
            accuracy: "none: synthetic".into(),
            joules: rce_graph3d::dash::integrate(&series),
            series,
            synthetic: true,
        }
    };
    rows.push(syn("rapl", "cpu", 18.0));
    rows.push(syn("nvml", "gpu", 42.0));
    let area = R {
        x: 0.0,
        y: 0.0,
        w: w as f32,
        h: h as f32,
    };
    let sc = graph_panel::dashboard_scene(
        &rows,
        area,
        w,
        h,
        "real probe results on this machine; rows marked SYNTHETIC are demo series",
    );
    let (px, _) = r.render_rgba(&sc, w, h)?;
    rce_ide::png::write(&format!("{out}/telemetry.png"), w, h, &px)?;
    let model = json!({"timeline": tl, "telemetry": rows});
    std::fs::write(
        format!("{out}/graph3d-model.json"),
        serde_json::to_string(&model).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "graph3d_shots: wrote {out}/graph3d.png, graph3d-time.png, hardware.png, telemetry.png on {} ({:?})",
        r.adapter_info.name, r.adapter_info.backend
    );
    Ok(())
}

fn tempfile_dir() -> Result<std::path::PathBuf, String> {
    let d = std::env::temp_dir().join(format!("rce-graph3d-demo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}
