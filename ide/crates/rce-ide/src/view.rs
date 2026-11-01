//! The IDE as a photon `ViewNode` tree, plus input dispatch.
use crate::model::{
    Action, EditorVm, Focus, MemoryVm, NetVm, Panels, Proposal, ProposalStatus, ReceiptsVm,
};
use photon::AccessibilityRole;
use photon::layout::{Alignment2D, HAlignment, HStack, VAlignment, VStack, ZStack};
use photon::{Color, LeafKind, RenderCtx, TextLeaf, ViewExt, ViewNode, render_node};
use std::cell::RefCell;
use std::rc::Rc;

/// Actions produced by taps, drained by the host each frame.
pub type Queue = Rc<RefCell<Vec<Action>>>;

pub const BG: Color = Color::rgb(0.11, 0.12, 0.14);
pub const PANEL: Color = Color::rgb(0.15, 0.16, 0.19);
pub const FG: Color = Color::rgb(0.86, 0.87, 0.89);
pub const DIM: Color = Color::rgb(0.55, 0.57, 0.62);
pub const ERR: Color = Color::rgb(0.94, 0.38, 0.36);
pub const OK: Color = Color::rgb(0.40, 0.78, 0.48);
/// Panel titles, the caret and the save and recall buttons. At 8 bits
/// (#5c94f5) it has 4.86 : 1 on `PANEL` as text (WCAG 1.4.3, 4.5 : 1).
pub const ACCENT: Color = Color::rgb(0.36, 0.58, 0.96);
/// Button labels: dark text on the coloured fills (`ACCENT`, `OK`, `ERR`),
/// each at least 5.1 : 1; white on those fills was 2.1 to 3.3 : 1.
pub const ON_FILL: Color = BG;
/// The keyboard focus outline: 2 px, 14.5 : 1 on `PANEL` (WCAG 2.4.7).
pub const FOCUS_RING: Color = Color::WHITE;

pub fn text(s: impl Into<String>, size: f32, color: Color) -> ViewNode {
    let mut t = TextLeaf::new(s);
    t.font_size = size;
    t.color = color;
    let w = t.content.chars().count() as f32 * size * MONO;
    ViewNode::Leaf(LeafKind::Text(t)).frame(Some(w), Some(size * 1.4))
}

/// Advance of the UI's monospace font per em (photon's layout guesses
/// 0.55; DejaVu Sans Mono and most monospace faces are ~0.6).
pub const MONO: f32 = 0.602;

fn vstack(children: Vec<ViewNode>, spacing: f32) -> ViewNode {
    ViewNode::Container {
        layout: Box::new(
            VStack::new()
                .spacing(spacing)
                .alignment(HAlignment::Leading),
        ),
        children,
    }
}

fn hstack(children: Vec<ViewNode>, spacing: f32) -> ViewNode {
    ViewNode::Container {
        layout: Box::new(HStack::new().spacing(spacing).alignment(VAlignment::Top)),
        children,
    }
}

/// A button named by its label. `focus` is the label of the button that
/// has keyboard focus, if any; that one is drawn with `FOCUS_RING`.
fn button(label: &str, color: Color, q: &Queue, a: Action, focus: Option<&str>) -> ViewNode {
    let q = q.clone();
    let b = text(label, 13.0, ON_FILL)
        .padding_xy(8.0, 3.0)
        .background_color(color)
        .corner_radius(4.0);
    let b = if focus == Some(label) {
        b.border(2.0, FOCUS_RING)
    } else {
        b
    };
    b.accessibility_label(label.to_string())
        .accessibility_role(AccessibilityRole::Button)
        .on_tap(move || q.borrow_mut().push(a.clone()))
}

/// Every button of the window with its action, in the order the view lays
/// them out: save, then each pending proposal's accept and reject, recall,
/// then each pending endpoint's allow and deny. `Focus::Control(n)` is the
/// n-th of these.
pub fn controls(p: &Panels<'_>) -> Vec<(String, Action)> {
    let mut c = vec![];
    if p.editor.dirty {
        c.push(("save as proposal".to_string(), Action::Save));
    }
    for x in p.proposals {
        if x.status == ProposalStatus::Pending {
            c.push((format!("accept {}", x.id), Action::Accept(x.id)));
            c.push((format!("reject {}", x.id), Action::Reject(x.id)));
        }
    }
    c.push(("recall".to_string(), Action::Recall(p.memory.query.clone())));
    for (key, _) in &p.net.pending {
        for allow in [true, false] {
            let word = if allow { "allow" } else { "deny" };
            c.push((
                format!("{word} {key}"),
                Action::Net {
                    key: key.clone(),
                    allow,
                },
            ));
        }
    }
    c
}

/// The focus after one step forward (Tab, F6) or back (Shift+Tab,
/// Shift+F6): the editor, the memory query, then each control in turn.
pub fn focus_step(p: &Panels<'_>, back: bool) -> Focus {
    let n = controls(p).len() as u32;
    // Stops 0 (editor), 1 (memory query), 2.. (controls).
    let stops = n + 2;
    let at = match p.focus {
        Focus::Editor => 0,
        Focus::Memory => 1,
        Focus::Control(i) => 2 + i.min(n.saturating_sub(1)),
    };
    let to = if back {
        (at + stops - 1) % stops
    } else {
        (at + 1) % stops
    };
    match to {
        0 => Focus::Editor,
        1 => Focus::Memory,
        i => Focus::Control(i - 2),
    }
}

/// The action of the control that has keyboard focus (Enter or Space).
pub fn focused_action(p: &Panels<'_>) -> Option<Action> {
    match p.focus {
        Focus::Control(i) => controls(p).into_iter().nth(i as usize).map(|c| c.1),
        _ => None,
    }
}

fn focused_label(p: &Panels<'_>) -> Option<String> {
    match p.focus {
        Focus::Control(i) => controls(p).into_iter().nth(i as usize).map(|c| c.0),
        _ => None,
    }
}

fn panel(title: &str, body: Vec<ViewNode>, w: f32, h: f32) -> ViewNode {
    let mut c = vec![text(title, 15.0, ACCENT)];
    c.extend(body);
    // Content at its natural height, pinned top-leading in a fixed panel.
    let content = vstack(c, 4.0).padding_all(8.0);
    ViewNode::Container {
        layout: Box::new(
            ZStack::new().alignment(Alignment2D::new(HAlignment::Leading, VAlignment::Top)),
        ),
        children: vec![content],
    }
    .frame(Some(w), Some(h))
    .background_color(PANEL)
    .clip(photon::ClipShape::Rect)
}

fn j(x: Option<f64>) -> String {
    x.map_or("—".into(), |v| format!("{v:.3} J"))
}

/// Text metrics of the editor font. The host measures the real monospace
/// advance with its renderer; the default matches photon's layout estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub char_w: f32,
    pub line_h: f32,
    pub size: f32,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            char_w: 13.0 * 0.6,
            line_h: 18.0,
            size: 13.0,
        }
    }
}

pub const SEL: Color = Color::rgba(0.33, 0.55, 0.95, 0.35);
const GUTTER_CHARS: f32 = 5.0;
/// Panel chrome above and below the text area (title, path, status, up to
/// three diagnostic lines).
const EDITOR_CHROME: f32 = 8.0 * 2.0 + 20.0 + 16.0 + 18.0 + 3.0 * 16.0 + 6.0 * 4.0;

fn gap(w: f32, h: f32) -> ViewNode {
    ViewNode::Leaf(LeafKind::Spacer { min_length: 0.0 }).frame(Some(w.max(0.0)), Some(h))
}

fn quad(w: f32, h: f32, c: Color) -> ViewNode {
    gap(w, h).background_color(c)
}

fn zstack(children: Vec<ViewNode>) -> ViewNode {
    ViewNode::Container {
        layout: Box::new(
            ZStack::new().alignment(Alignment2D::new(HAlignment::Leading, VAlignment::Top)),
        ),
        children,
    }
}

/// How many text lines fit in an editor panel of height `h`.
pub fn editor_rows(h: f32, m: &Metrics) -> usize {
    (((h - EDITOR_CHROME) / m.line_h).floor() as usize).max(1)
}

pub fn editor(
    e: &EditorVm,
    focused: bool,
    q: &Queue,
    m: &Metrics,
    w: f32,
    h: f32,
    focus: Option<&str>,
) -> ViewNode {
    let v = &e.view;
    let head = text(
        format!(
            "{}{}   [{}]",
            e.path,
            if e.dirty { " •" } else { "" },
            e.server
        ),
        12.0,
        DIM,
    );
    let gw = GUTTER_CHARS * m.char_w + 8.0;
    let tw = (w - 16.0 - gw).max(0.0);
    let mut rows = vec![];
    for (i, l) in v.lines.iter().enumerate() {
        let n = v.first + i + 1;
        let bad = e.diagnostics.iter().any(|d| d.line == n);
        let mut layers = vec![];
        for &(_, a, b) in v.selection.iter().filter(|s| s.0 + 1 == n) {
            layers.push(hstack(
                vec![
                    gap(a as f32 * m.char_w, m.line_h),
                    quad((b - a) as f32 * m.char_w, m.line_h, SEL),
                ],
                0.0,
            ));
        }
        let mut t = TextLeaf::new(l.clone());
        t.font_size = m.size;
        // ERR on the selection colour is 2.7 : 1, so a selected line with a
        // diagnostic is drawn in FG (6.4 : 1); its line number stays ERR.
        let selected = v.selection.iter().any(|s| s.0 + 1 == n && s.2 > s.1);
        t.color = if bad && !selected { ERR } else { FG };
        layers.push(ViewNode::Leaf(LeafKind::Text(t)));
        if v.cursor.0 + 1 == n {
            let x = v.cursor.1 as f32 * m.char_w;
            if let Some(p) = &v.preedit {
                // IME composition at the caret, underlined.
                let pw = p.chars().count() as f32 * m.char_w;
                layers.push(hstack(
                    vec![gap(x, m.line_h), quad(pw, m.line_h, PANEL)],
                    0.0,
                ));
                layers.push(hstack(
                    vec![gap(x, 1.0), text(p.clone(), m.size, Color::WHITE)],
                    0.0,
                ));
                layers.push(hstack(
                    vec![gap(x, m.line_h - 2.0), quad(pw, 1.5, ACCENT)],
                    0.0,
                ));
            } else if focused {
                layers.push(hstack(
                    vec![gap(x, m.line_h), quad(2.0, m.line_h, ACCENT)],
                    0.0,
                ));
            }
        }
        rows.push(hstack(
            vec![
                text(format!("{n:>4}"), m.size, if bad { ERR } else { DIM })
                    .frame(Some(gw), Some(m.line_h)),
                zstack(layers).frame(Some(tw), Some(m.line_h)),
            ],
            0.0,
        ));
    }
    let q2 = q.clone();
    let area = vstack(rows, 0.0)
        .frame(Some(w - 16.0), Some(v.lines.len().max(1) as f32 * m.line_h))
        .accessibility_label("editor-text".to_string())
        .on_tap(move || q2.borrow_mut().push(Action::Focus(Focus::Editor)));
    let pos = format!(
        "Ln {}, Col {} · {} lines{}{}{}",
        v.cursor.0 + 1,
        v.cursor.1 + 1,
        v.total_lines,
        match v.selection.iter().map(|s| s.2 - s.1).sum::<usize>() {
            0 => String::new(),
            n => format!(" · {n} selected"),
        },
        if e.can_undo { " · undo" } else { "" },
        if e.can_redo { " · redo" } else { "" },
    );
    let status = if e.diagnostics.is_empty() {
        text(format!("rce check: ok   {pos}"), 12.0, OK)
    } else {
        text(
            format!("rce check: {} diagnostic(s)   {pos}", e.diagnostics.len()),
            12.0,
            ERR,
        )
    };
    let mut body = vec![head, area, status];
    for d in e.diagnostics.iter().take(3) {
        body.push(text(format!("L{}: {}", d.line, d.message), 12.0, ERR));
    }
    let save = if e.dirty {
        vec![button("save as proposal", ACCENT, q, Action::Save, focus)]
    } else {
        vec![]
    };
    body.splice(1..1, save);
    panel("Editor", body, w, h)
}

/// Map a point in the editor's text area to (line, display column).
pub fn editor_hit(
    ctx: &RenderCtx,
    e: &EditorVm,
    m: &Metrics,
    x: f32,
    y: f32,
) -> Option<(usize, usize)> {
    let a = ctx
        .annotations
        .iter()
        .find(|a| a.label.as_deref() == Some("editor-text"))?;
    let r = a.rect;
    if x < r.x || y < r.y || x >= r.x + r.width || y >= r.y + r.height {
        return None;
    }
    let gw = GUTTER_CHARS * m.char_w + 8.0;
    let line = e.view.first + ((y - r.y) / m.line_h) as usize;
    let col = ((x - r.x - gw) / m.char_w).round().max(0.0) as usize;
    Some((line, col))
}

pub fn proposals(ps: &[Proposal], q: &Queue, w: f32, h: f32, focus: Option<&str>) -> ViewNode {
    let mut rows = vec![];
    if ps.is_empty() {
        rows.push(text("no proposals", 12.0, DIM));
    }
    for p in ps {
        let head = text(
            format!(
                "#{} {} — {} ({} edit(s), {})",
                p.id,
                p.path,
                p.summary,
                p.edits.len(),
                p.origin
            ),
            13.0,
            FG,
        );
        let tail = match &p.status {
            ProposalStatus::Pending => hstack(
                vec![
                    button(
                        &format!("accept {}", p.id),
                        OK,
                        q,
                        Action::Accept(p.id),
                        focus,
                    ),
                    button(
                        &format!("reject {}", p.id),
                        ERR,
                        q,
                        Action::Reject(p.id),
                        focus,
                    ),
                ],
                6.0,
            ),
            ProposalStatus::Accepted { receipt } => text(
                format!("accepted, receipt {}", &receipt[..receipt.len().min(12)]),
                12.0,
                OK,
            ),
            ProposalStatus::RequiresChange { diagnostics, .. } => text(
                format!("requires change (rce check): {}", diagnostics.join("; ")),
                12.0,
                ERR,
            ),
            ProposalStatus::Rejected { receipt } => text(
                format!("rejected, receipt {}", &receipt[..receipt.len().min(12)]),
                12.0,
                DIM,
            ),
        };
        rows.push(vstack(vec![head, tail], 2.0));
    }
    panel("Proposals", rows, w, h)
}

pub fn receipts(r: &ReceiptsVm, w: f32, h: f32) -> ViewNode {
    let t = &r.totals;
    let mut rows = vec![
        text(
            format!(
                "{} actions · E_sim {:.3} J ({} predicted) · E_meas {:.3} J ({} hardware) · est. remote {:.3} J · Δ {:.3} J ({}) · chain {}",
                t.actions,
                t.e_sim_j,
                t.with_sim,
                t.e_meas_hw_j,
                t.with_hw,
                t.e_est_remote_j,
                t.delta_j,
                t.with_delta,
                match r.chain_ok {
                    Some(true) => "verified",
                    Some(false) => "BROKEN",
                    None => "—",
                }
            ),
            12.0,
            if r.chain_ok == Some(false) { ERR } else { DIM },
        ),
        text(
            "seq  action           gear     E_sim        E_meas (source)          Δ          verdict",
            12.0,
            DIM,
        ),
    ];
    for x in r.rows.iter().rev().take(40) {
        let meas = match (&x.e_meas_j, &x.source) {
            (Some(m), Some(s)) => format!("{m:.3} J ({s})"),
            _ => "—".into(),
        };
        rows.push(text(
            format!(
                "{:<4} {:<16} {:<8} {:<12} {:<24} {:<10} {}",
                x.seq,
                x.action,
                x.gear,
                j(x.e_sim_j),
                meas,
                j(x.delta_j),
                x.verdict
            ),
            12.0,
            if x.verdict == "pass" || x.verdict == "notrun" {
                FG
            } else {
                ERR
            },
        ));
    }
    panel("Receipts and energy", rows, w, h)
}

pub fn memory(
    m: &MemoryVm,
    focused: bool,
    q: &Queue,
    w: f32,
    h: f32,
    focus: Option<&str>,
) -> ViewNode {
    let qf = q.clone();
    let mut rows = vec![hstack(
        vec![
            text(
                format!(
                    "query: {}{}",
                    if m.query.is_empty() && !focused {
                        "—"
                    } else {
                        &m.query
                    },
                    if focused { "▏" } else { "" }
                ),
                13.0,
                if focused { Color::WHITE } else { FG },
            )
            .padding_xy(4.0, 1.0)
            .background_color(if focused { BG } else { PANEL })
            // The field's edge: DIM (4.6 : 1 on PANEL, WCAG 1.4.11), and the
            // focus outline when it has keyboard focus (2.4.7).
            .border(
                if focused { 2.0 } else { 1.0 },
                if focused { FOCUS_RING } else { DIM },
            )
            .accessibility_label("memory-query".to_string())
            .accessibility_role(AccessibilityRole::TextBox)
            .on_tap(move || qf.borrow_mut().push(Action::Focus(Focus::Memory))),
            button(
                "recall",
                ACCENT,
                q,
                Action::Recall(m.query.clone()),
                focus,
            ),
        ],
        8.0,
    )];
    if let Some(e) = &m.error {
        rows.push(text(e.clone(), 12.0, ERR));
    }
    if let Some(g) = &m.gear {
        rows.push(text(format!("answered by: {g}"), 12.0, DIM));
    }
    if m.escalate {
        rows.push(text("no cheap gear answered (escalate)", 12.0, DIM));
    }
    for hit in &m.hits {
        rows.push(text(
            format!("{:.2}  {}  {}", hit.score, hit.key, hit.detail),
            12.0,
            FG,
        ));
    }
    panel("Memory recall", rows, w, h)
}

pub fn network(n: &NetVm, q: &Queue, w: f32, h: f32, focus: Option<&str>) -> ViewNode {
    let mut rows = vec![];
    if n.pending.is_empty() && n.decided.is_empty() {
        rows.push(text("no declared endpoints", 12.0, DIM));
    }
    for (key, reason) in &n.pending {
        rows.push(vstack(
            vec![
                text(
                    format!(
                        "program wants {key} — {}",
                        if reason.is_empty() {
                            "no reason given"
                        } else {
                            reason
                        }
                    ),
                    13.0,
                    FG,
                ),
                hstack(
                    vec![
                        button(
                            &format!("allow {key}"),
                            OK,
                            q,
                            Action::Net {
                                key: key.clone(),
                                allow: true,
                            },
                            focus,
                        ),
                        button(
                            &format!("deny {key}"),
                            ERR,
                            q,
                            Action::Net {
                                key: key.clone(),
                                allow: false,
                            },
                            focus,
                        ),
                    ],
                    6.0,
                ),
            ],
            2.0,
        ));
    }
    for (k, a) in &n.decided {
        rows.push(text(
            format!("{k}: {}", if *a { "allowed" } else { "denied" }),
            12.0,
            if *a { OK } else { DIM },
        ));
    }
    panel("Network", rows, w, h)
}

/// The whole window.
pub fn root(p: &Panels<'_>, q: &Queue, w: f32, h: f32) -> ViewNode {
    root_with(p, q, &Metrics::default(), w, h)
}

pub fn root_with(p: &Panels<'_>, q: &Queue, m: &Metrics, w: f32, h: f32) -> ViewNode {
    let lw = (w * 0.5).floor();
    let rw = w - lw - 6.0;
    let fl = focused_label(p);
    let f = fl.as_deref();
    let right = vstack(
        vec![
            proposals(p.proposals, q, rw, h * 0.22, f),
            receipts(p.receipts, rw, h * 0.34),
            memory(p.memory, p.focus == Focus::Memory, q, rw, h * 0.22, f),
            network(p.net, q, rw, h * 0.22 - 18.0, f),
        ],
        6.0,
    );
    hstack(
        vec![
            editor(p.editor, p.focus == Focus::Editor, q, m, lw, h, f),
            right,
        ],
        6.0,
    )
    .frame(Some(w), Some(h))
    .background_color(BG)
}

/// Lay out and paint a tree into a fresh `RenderCtx`.
pub fn paint(node: &ViewNode, w: f32, h: f32) -> RenderCtx {
    let mut ctx = RenderCtx::new();
    render_node(node, photon::Rect::new(0.0, 0.0, w, h), &mut ctx);
    ctx
}

/// Dispatch a click: the topmost tap region containing the point.
pub fn click(ctx: &RenderCtx, x: f32, y: f32) -> bool {
    for r in ctx.hit_regions.iter().rev() {
        let b = r.rect;
        if x >= b.x
            && y >= b.y
            && x < b.x + b.width
            && y < b.y + b.height
            && let photon::HitKind::Tap(f) = &r.kind
        {
            f();
            return true;
        }
    }
    false
}

/// Centre of the element with this accessibility label.
pub fn find(ctx: &RenderCtx, label: &str) -> Option<(f32, f32)> {
    ctx.annotations
        .iter()
        .find(|a| a.label.as_deref() == Some(label))
        .map(|a| {
            (
                a.rect.x + a.rect.width / 2.0,
                a.rect.y + a.rect.height / 2.0,
            )
        })
}
