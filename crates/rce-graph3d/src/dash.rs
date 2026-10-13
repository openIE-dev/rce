//! 2D primitives for the panels around the 3D view: the telemetry
//! dashboard, labels, legend and the time scrubber. The host draws them
//! with its own text stack (rce-ide: Parley + swash).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Prim {
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    },
    Text {
        x: f32,
        y: f32,
        size: f32,
        text: String,
        color: [f32; 4],
    },
}

pub const FG: [f32; 4] = [0.86, 0.88, 0.92, 1.0];
pub const DIM: [f32; 4] = [0.55, 0.58, 0.65, 1.0];
pub const BG: [f32; 4] = [0.09, 0.10, 0.12, 1.0];
pub const CARD: [f32; 4] = [0.13, 0.14, 0.17, 1.0];
pub const OK: [f32; 4] = [0.45, 0.85, 0.55, 1.0];
pub const WARN: [f32; 4] = [1.0, 0.75, 0.3, 1.0];
pub const BAD: [f32; 4] = [1.0, 0.45, 0.45, 1.0];

pub fn rect(x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> Prim {
    Prim::Rect { x, y, w, h, color }
}
pub fn text(x: f32, y: f32, size: f32, t: impl Into<String>, color: [f32; 4]) -> Prim {
    Prim::Text {
        x,
        y,
        size,
        text: t.into(),
        color,
    }
}

/// One energy source as the dashboard shows it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceRow {
    pub tag: String,
    pub domain: String,
    pub detail: String,
    /// `available`, or the reason it is not (`needs setup: ...`, `unsupported: ...`, `stale: ...`).
    pub status: String,
    pub resolution: String,
    pub accuracy: String,
    /// (t_ms, watts) samples; empty when the source is not read.
    pub series: Vec<(i64, f64)>,
    /// Joules integrated over the series window.
    pub joules: Option<f64>,
    pub synthetic: bool,
}

impl SourceRow {
    pub fn available(&self) -> bool {
        self.status == "available"
    }
    pub fn watts(&self) -> Option<f64> {
        self.series.last().map(|s| s.1)
    }
}

/// Integrate (t_ms, W) by the trapezoid rule, in joules.
pub fn integrate(s: &[(i64, f64)]) -> Option<f64> {
    if s.len() < 2 {
        return None;
    }
    Some(
        s.windows(2)
            .map(|w| (w[1].0 - w[0].0) as f64 / 1000.0 * (w[0].1 + w[1].1) / 2.0)
            .sum(),
    )
}

/// The dashboard: one card per source. A source without samples shows its
/// status, never a number.
pub fn dashboard(rows: &[SourceRow], x: f32, y: f32, w: f32, h: f32, note: &str) -> Vec<Prim> {
    let mut out = vec![rect(x, y, w, h, BG)];
    out.push(text(x + 16.0, y + 12.0, 20.0, "Telemetry", FG));
    out.push(text(x + 140.0, y + 17.0, 13.0, note, WARN));
    let cols = if w > 900.0 { 2 } else { 1 };
    let cw = (w - 16.0 * (cols as f32 + 1.0)) / cols as f32;
    let ch = 108.0;
    for (i, r) in rows.iter().enumerate() {
        let cx = x + 16.0 + (i % cols) as f32 * (cw + 16.0);
        let cy = y + 48.0 + (i / cols) as f32 * (ch + 12.0);
        if cy + ch > y + h {
            out.push(text(
                cx,
                cy,
                13.0,
                format!("+{} more sources", rows.len() - i),
                DIM,
            ));
            break;
        }
        out.push(rect(cx, cy, cw, ch, CARD));
        let st_col = if r.available() {
            OK
        } else if r.status.starts_with("needs setup") {
            WARN
        } else {
            BAD
        };
        out.push(rect(cx, cy, 4.0, ch, st_col));
        out.push(text(
            cx + 14.0,
            cy + 8.0,
            15.0,
            format!("{}  ·  {}", r.tag, r.domain),
            FG,
        ));
        if r.synthetic {
            out.push(text(cx + cw - 96.0, cy + 9.0, 12.0, "SYNTHETIC", WARN));
        }
        out.push(text(
            cx + 14.0,
            cy + 28.0,
            12.0,
            crate::graph::trunc(&r.detail, 70),
            DIM,
        ));
        let spark = (cx + 14.0, cy + 48.0, cw * 0.55, 40.0);
        match (r.watts(), r.series.len() >= 2) {
            (Some(wt), true) => {
                let max = r.series.iter().map(|s| s.1).fold(1e-9, f64::max);
                let n = r.series.len();
                let bw = spark.2 / n as f32;
                for (k, s) in r.series.iter().enumerate() {
                    let bh = (s.1 / max) as f32 * spark.3;
                    out.push(rect(
                        spark.0 + k as f32 * bw,
                        spark.1 + spark.3 - bh,
                        (bw - 1.0).max(1.0),
                        bh.max(1.0),
                        [0.35, 0.65, 1.0, 1.0],
                    ));
                }
                let tx = spark.0 + spark.2 + 16.0;
                out.push(text(tx, cy + 46.0, 18.0, format!("{wt:.2} W"), FG));
                let j = r
                    .joules
                    .map(|j| format!("{j:.1} J over window"))
                    .unwrap_or_else(|| "J: window too short".into());
                out.push(text(tx, cy + 70.0, 12.0, j, DIM));
            }
            _ => {
                out.push(text(
                    spark.0,
                    cy + 52.0,
                    15.0,
                    crate::graph::trunc(&r.status, 80),
                    st_col,
                ));
                out.push(text(
                    spark.0,
                    cy + 72.0,
                    12.0,
                    "no reading shown: not measured",
                    DIM,
                ));
            }
        }
        out.push(text(
            cx + 14.0,
            cy + ch - 18.0,
            11.0,
            format!("accuracy: {}", crate::graph::trunc(&r.accuracy, 90)),
            DIM,
        ));
    }
    out
}
