//! Flatten photon draw commands into what the GPU host draws: solid quads
//! and text runs, each with a resolved clip rectangle and opacity.
//! Pure; tested headless.
use photon::{DrawCommand, RenderCtx};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct R {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl R {
    pub fn intersect(self, o: R) -> R {
        let x0 = self.x.max(o.x);
        let y0 = self.y.max(o.y);
        let x1 = (self.x + self.w).min(o.x + o.w);
        let y1 = (self.y + self.h).min(o.y + o.h);
        R {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        }
    }
    pub fn is_empty(self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
    pub rect: R,
    pub color: [f32; 4],
    pub clip: R,
    /// Paint order across quads and texts.
    pub order: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    pub rect: R,
    pub text: String,
    pub size: f32,
    pub color: [f32; 4],
    pub clip: R,
    pub order: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    pub quads: Vec<Quad>,
    pub texts: Vec<TextRun>,
    /// Commands the host does not draw (images, shadows, filters), counted.
    pub skipped: usize,
}

pub fn build(ctx: &RenderCtx, w: f32, h: f32) -> Scene {
    let full = R {
        x: 0.0,
        y: 0.0,
        w,
        h,
    };
    let mut clips = vec![full];
    let mut alpha = vec![1.0f32];
    let mut tr = vec![(0.0f32, 0.0f32)];
    let mut s = Scene::default();
    let mut n = 0usize;
    let dy = -ctx.scroll_offset_y;
    for c in &ctx.commands {
        let clip = *clips.last().unwrap_or(&full);
        let a = *alpha.last().unwrap_or(&1.0);
        let (tx, ty) = *tr.last().unwrap_or(&(0.0, 0.0));
        let rr = |r: &photon::LayoutRect| R {
            x: r.x + tx,
            y: r.y + ty + dy,
            w: r.width,
            h: r.height,
        };
        let col = |c: &photon::Color| {
            let v = c.to_array();
            [v[0], v[1], v[2], v[3] * a]
        };
        match c {
            DrawCommand::FillRect { rect, color } => {
                let r = rr(rect);
                if !r.intersect(clip).is_empty() {
                    n += 1;
                    s.quads.push(Quad {
                        rect: r,
                        color: col(color),
                        clip,
                        order: n,
                    });
                }
            }
            DrawCommand::StrokeRect { rect, width, color } => {
                let r = rr(rect);
                let cl = col(color);
                for e in [
                    R {
                        x: r.x,
                        y: r.y,
                        w: r.w,
                        h: *width,
                    },
                    R {
                        x: r.x,
                        y: r.y + r.h - width,
                        w: r.w,
                        h: *width,
                    },
                    R {
                        x: r.x,
                        y: r.y,
                        w: *width,
                        h: r.h,
                    },
                    R {
                        x: r.x + r.w - width,
                        y: r.y,
                        w: *width,
                        h: r.h,
                    },
                ] {
                    if !e.intersect(clip).is_empty() {
                        n += 1;
                        s.quads.push(Quad {
                            rect: e,
                            color: cl,
                            clip,
                            order: n,
                        });
                    }
                }
            }
            DrawCommand::Text {
                rect,
                text,
                font_size,
                color,
            } => {
                let r = rr(rect);
                if !r.intersect(clip).is_empty() {
                    n += 1;
                    s.texts.push(TextRun {
                        rect: r,
                        text: text.clone(),
                        size: *font_size,
                        color: col(color),
                        clip,
                        order: n,
                    });
                }
            }
            DrawCommand::PushClip { rect } | DrawCommand::PushClipShape { rect, .. } => {
                clips.push(rr(rect).intersect(clip))
            }
            DrawCommand::PopClip | DrawCommand::PopClipShape => {
                if clips.len() > 1 {
                    clips.pop();
                }
            }
            DrawCommand::PushOpacity { alpha: x } => alpha.push(a * x),
            DrawCommand::PopOpacity => {
                if alpha.len() > 1 {
                    alpha.pop();
                }
            }
            DrawCommand::PushTransform { transform } => {
                // Translation only; scale/rotation are not drawn by this host.
                let m = transform.0;
                tr.push((tx + m[4], ty + m[5]));
            }
            DrawCommand::PopTransform => {
                if tr.len() > 1 {
                    tr.pop();
                }
            }
            _ => s.skipped += 1,
        }
    }
    s
}
