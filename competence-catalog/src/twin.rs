//! Model-twin admission seed — sim2real's tractable core: a controller is admitted
//! only if its recorded ticks reproduce a deterministic plant model within tol.
//! Not a physics simulator (that edge stays open, registered); a double-integrator
//! plant is the spectest, exactly as replay gates competence. Mujin's live twin is
//! the peer; the delta is that here the twin's verdict is a gate, not a demo.

/// A deterministic double-integrator plant (position/velocity), fixed dt.
#[derive(Clone, Copy, Debug)]
pub struct Plant {
    pub dt: f64,
}

/// Simulate the plant under an action stream; returns the position trace.
pub fn simulate(p: &Plant, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += a * p.dt;
            x += v * p.dt;
            x
        })
        .collect()
}

/// Max abs divergence between a recorded trace and the twin's trace; None if lengths differ.
pub fn divergence(recorded: &[f64], sim: &[f64]) -> Option<f64> {
    if recorded.len() != sim.len() {
        return Option::None;
    }
    recorded
        .iter()
        .zip(sim.iter())
        .map(|(r, s)| (r - s).abs())
        .fold(Option::Some(0.0f64), |acc, d| match acc {
            Option::None => Option::None,
            Option::Some(m) => {
                if d.is_finite() { Option::Some(m.max(d)) } else { Option::None }
            }
        })
}

/// Gate: admit the controller iff its recorded trace reproduces the twin within tol.
pub fn admit_controller(p: &Plant, acts: &[f64], recorded: &[f64], tol: f64) -> bool {
    if !tol.is_finite() || tol < 0.0 || !p.dt.is_finite() || p.dt <= 0.0 {
        return false;
    }
    match divergence(recorded, &simulate(p, acts)) {
        Option::Some(d) => d <= tol,
        Option::None => false,
    }
}

/// Spring-damper contact force while the plant presses past a surface at `wall`:
/// F = k*penetration + c*approach-velocity. A force-model seed toward ISO/TS 15066
/// (collaborative force/pressure limits); full pressure/area mapping stays open.
pub fn contact_force(p: &Plant, acts: &[f64], wall: f64, k: f64, c: f64) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += a * p.dt;
            x += v * p.dt;
            let pen = (x - wall).max(0.0);
            if pen > 0.0 { k * pen + c * v.max(0.0) } else { 0.0 }
        })
        .collect()
}

/// Force-limit gate: every contact sample finite and <= limit (Newtons).
pub fn force_limit_ok(forces: &[f64], limit_n: f64) -> bool {
    limit_n.is_finite() && limit_n >= 0.0 && forces.iter().all(|f| f.is_finite() && *f <= limit_n)
}

/// Contact pressure = force / contact area (Pa) — the second half of the ISO/TS 15066
/// force/pressure pair. None if the area is not a positive finite number.
pub fn contact_pressure(forces: &[f64], area_m2: f64) -> Option<Vec<f64>> {
    if !area_m2.is_finite() || area_m2 <= 0.0 {
        return Option::None;
    }
    Option::Some(forces.iter().map(|f| f / area_m2).collect())
}

/// Pressure-limit gate: every sample finite and <= limit (Pascals).
pub fn pressure_limit_ok(pressures: &[f64], limit_pa: f64) -> bool {
    limit_pa.is_finite() && limit_pa >= 0.0 && pressures.iter().all(|p| p.is_finite() && *p <= limit_pa)
}

/// A richer plant for physics-fidelity sim2real: double integrator + gravity + linear drag
/// + a floor at `wall` with coefficient of restitution. Still a seed — full rigid-body /
/// multi-contact solving remains the open frontier.
#[derive(Clone, Copy, Debug)]
pub struct Plant2 {
    pub dt: f64,
    pub gravity: f64,
    pub drag: f64,
    pub wall: f64,
    pub restitution: f64,
}

/// Simulate Plant2; returns the position trace. Contact clamps to the floor and reflects
/// velocity by the restitution coefficient.
pub fn simulate2(p: &Plant2, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            let acc = a - p.gravity - p.drag * v;
            v += acc * p.dt;
            x += v * p.dt;
            if x < p.wall {
                x = p.wall;
                v = -v * p.restitution;
            }
            x
        })
        .collect()
}

/// A two-boundary contact plant (floor + ceiling) with per-contact restitution and tangential
/// friction. A multi-contact seed; full rigid-body / arbitrary contact manifolds stay open.
#[derive(Clone, Copy, Debug)]
pub struct Plant3 {
    pub dt: f64,
    pub gravity: f64,
    pub lo: f64,
    pub hi: f64,
    pub restitution: f64,
    pub friction: f64,
}

/// Simulate Plant3; the trace stays within [lo, hi]; contact reflects velocity by restitution
/// and bleeds it by friction.
pub fn simulate3(p: &Plant3, acts: &[f64]) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    acts.iter()
        .map(|a| {
            v += (a - p.gravity) * p.dt;
            x += v * p.dt;
            if x < p.lo {
                x = p.lo;
                v = -v * p.restitution;
                v *= 1.0 - p.friction;
            }
            if x > p.hi {
                x = p.hi;
                v = -v * p.restitution;
                v *= 1.0 - p.friction;
            }
            x
        })
        .collect()
}

/// A 2D rigid body (circle) in a static box: gravity, per-wall restitution, tangential
/// friction that induces spin. A single-body rigid-contact seed; multi-body / arbitrary
/// manifolds stay open.
#[derive(Clone, Copy, Debug)]
pub struct Body2 {
    pub dt: f64,
    pub g: f64,
    pub radius: f64,
    pub lo: (f64, f64),
    pub hi: (f64, f64),
    pub rest: f64,
    pub fric: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct State2 {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub spin: f64,
}

/// Simulate the body under a per-step force sequence; contacts clamp to the box, reflect
/// the normal velocity by restitution, bleed tangential velocity and add spin by friction.
pub fn simulate_body(b: &Body2, s0: &State2, forces: &[(f64, f64)]) -> Vec<State2> {
    let mut s = *s0;
    let mut out = Vec::new();
    for (fx, fy) in forces {
        s.vx += fx * b.dt;
        s.vy += (fy - b.g) * b.dt;
        s.x += s.vx * b.dt;
        s.y += s.vy * b.dt;
        if s.x - b.radius < b.lo.0 {
            s.x = b.lo.0 + b.radius;
            s.vx = -s.vx * b.rest;
            s.spin += s.vy * b.fric;
            s.vy *= 1.0 - b.fric;
        }
        if s.x + b.radius > b.hi.0 {
            s.x = b.hi.0 - b.radius;
            s.vx = -s.vx * b.rest;
            s.spin -= s.vy * b.fric;
            s.vy *= 1.0 - b.fric;
        }
        if s.y - b.radius < b.lo.1 {
            s.y = b.lo.1 + b.radius;
            s.vy = -s.vy * b.rest;
            s.spin += s.vx * b.fric;
            s.vx *= 1.0 - b.fric;
        }
        if s.y + b.radius > b.hi.1 {
            s.y = b.hi.1 - b.radius;
            s.vy = -s.vy * b.rest;
            s.spin -= s.vx * b.fric;
            s.vx *= 1.0 - b.fric;
        }
        out.push(s);
    }
    out
}

/// A multi-body world: N circles with mass, pairwise impulse contact (restitution) and
/// box bounds. A multi-contact seed; arbitrary manifolds / rotation stay open.
#[derive(Clone, Copy, Debug)]
pub struct WorldN {
    pub dt: f64,
    pub g: f64,
    pub rest: f64,
    pub lo: (f64, f64),
    pub hi: (f64, f64),
}

#[derive(Clone, Copy, Debug)]
pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
    pub m: f64,
    /// Orientation (rad) and angular velocity (rad/s) — rigid rotation.
    pub theta: f64,
    pub omega: f64,
}

impl Ball {
    /// A non-rotating ball (convenience for earlier tests).
    pub fn new(x: f64, y: f64, vx: f64, vy: f64, r: f64, m: f64) -> Ball {
        Ball { x, y, vx, vy, r, m, theta: 0.0, omega: 0.0 }
    }
}

/// An oriented box obstacle (arbitrary contact manifold seed): closest-point contact in the
/// box's local frame, so the normal follows the box's orientation, not the world axes.
#[derive(Clone, Copy, Debug)]
pub struct Obb {
    pub cx: f64,
    pub cy: f64,
    pub hx: f64,
    pub hy: f64,
    pub angle: f64,
}

/// Resolve a ball vs an oriented box: push out along the local closest-point normal and
/// reflect the normal velocity by restitution; friction spins the ball.
pub fn collide_obb(b: &mut Ball, o: &Obb, rest: f64, fric: f64) {
    let (s, c) = (o.angle.sin(), o.angle.cos());
    let dx = b.x - o.cx;
    let dy = b.y - o.cy;
    // world -> local
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;
    let qx = lx.clamp(-o.hx, o.hx);
    let qy = ly.clamp(-o.hy, o.hy);
    let mut nx = lx - qx;
    let mut ny = ly - qy;
    let dist = (nx * nx + ny * ny).sqrt();
    if dist >= b.r || dist < 1e-12 {
        return;
    }
    nx /= dist;
    ny /= dist;
    // local -> world normal
    let wnx = nx * c - ny * s;
    let wny = nx * s + ny * c;
    let pen = b.r - dist;
    b.x += wnx * pen;
    b.y += wny * pen;
    let vn = b.vx * wnx + b.vy * wny;
    if vn < 0.0 {
        b.vx -= (1.0 + rest) * vn * wnx;
        b.vy -= (1.0 + rest) * vn * wny;
        // tangential friction -> torque
        let tx = -wny;
        let ty = wnx;
        let vt = b.vx * tx + b.vy * ty;
        b.omega -= fric * vt / b.r;
        b.vx -= fric * vt * tx;
        b.vy -= fric * vt * ty;
    }
}

/// Simulate N bodies: integrate, resolve pairwise contacts by impulse, clamp to bounds.
pub fn simulate_n(w: &WorldN, s0: &[Ball], steps: usize) -> Vec<Vec<Ball>> {
    let mut s = s0.to_vec();
    let mut out = Vec::new();
    for _ in 0..steps {
        for b in &mut s {
            b.vy -= w.g * w.dt;
            b.x += b.vx * w.dt;
            b.y += b.vy * w.dt;
            b.theta += b.omega * w.dt;
        }
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                let dx = s[j].x - s[i].x;
                let dy = s[j].y - s[i].y;
                let dist = (dx * dx + dy * dy).sqrt();
                let min = s[i].r + s[j].r;
                if dist < min && dist > 1e-12 {
                    let nx = dx / dist;
                    let ny = dy / dist;
                    let overlap = (min - dist) / 2.0;
                    s[i].x -= nx * overlap;
                    s[i].y -= ny * overlap;
                    s[j].x += nx * overlap;
                    s[j].y += ny * overlap;
                    let rvx = s[j].vx - s[i].vx;
                    let rvy = s[j].vy - s[i].vy;
                    let vn = rvx * nx + rvy * ny;
                    if vn < 0.0 {
                        let mi = s[i].m;
                        let mj = s[j].m;
                        let imp = -(1.0 + w.rest) * vn / (1.0 / mi + 1.0 / mj);
                        s[i].vx -= imp * nx / mi;
                        s[i].vy -= imp * ny / mi;
                        s[j].vx += imp * nx / mj;
                        s[j].vy += imp * ny / mj;
                        // tangential friction induces opposite spin on the two bodies
                        let tx = -ny;
                        let ty = nx;
                        let vt = rvx * tx + rvy * ty;
                        s[i].omega -= 0.1 * vt / s[i].r;
                        s[j].omega += 0.1 * vt / s[j].r;
                    }
                }
            }
        }
        for b in &mut s {
            if b.x - b.r < w.lo.0 { b.x = w.lo.0 + b.r; b.vx = -b.vx * w.rest; }
            if b.x + b.r > w.hi.0 { b.x = w.hi.0 - b.r; b.vx = -b.vx * w.rest; }
            if b.y - b.r < w.lo.1 { b.y = w.lo.1 + b.r; b.vy = -b.vy * w.rest; }
            if b.y + b.r > w.hi.1 { b.y = w.hi.1 - b.r; b.vy = -b.vy * w.rest; }
        }
        out.push(s.clone());
    }
    out
}

/// A convex polygon body (arbitrary contact manifold seed): vertices in world space.
#[derive(Clone, Debug)]
pub struct Poly {
    pub v: Vec<(f64, f64)>,
    pub m: f64,
    pub vx: f64,
    pub vy: f64,
    pub theta: f64,
    pub omega: f64,
}

fn poly_edges(p: &Poly) -> Vec<((f64, f64), (f64, f64))> {
    (0..p.v.len()).map(|i| (p.v[i], p.v[(i + 1) % p.v.len()])).collect()
}

fn closest_on_segment(px: f64, py: f64, a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let abx = b.0 - a.0;
    let aby = b.1 - a.1;
    let len2 = abx * abx + aby * aby;
    let t = if len2 > 0.0 { ((px - a.0) * abx + (py - a.1) * aby) / len2 } else { 0.0 }.clamp(0.0, 1.0);
    (a.0 + t * abx, a.1 + t * aby)
}

fn inside_convex(px: f64, py: f64, p: &Poly) -> bool {
    let mut sign = 0.0;
    for (a, b) in poly_edges(p) {
        let cross = (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0);
        if cross.abs() > 1e-12 {
            if sign == 0.0 {
                sign = cross;
            } else if sign * cross < 0.0 {
                return false;
            }
        }
    }
    true
}

/// Circle vs convex polygon: closest-point contact, push out along the boundary normal,
/// reflect by restitution, friction induces spin. Handles the center-inside case.
pub fn circle_poly(b: &mut Ball, p: &Poly, rest: f64, fric: f64) {
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for (a, c) in poly_edges(p) {
        let q = closest_on_segment(b.x, b.y, a, c);
        let d = ((b.x - q.0) * (b.x - q.0) + (b.y - q.1) * (b.y - q.1)).sqrt();
        if d < best.0 {
            best = (d, q.0, q.1);
        }
    }
    let (dist, qx, qy) = best;
    let inside = inside_convex(b.x, b.y, p);
    if !inside && dist >= b.r {
        return;
    }
    let (mut nx, mut ny) = if dist > 1e-12 { ((b.x - qx) / dist, (b.y - qy) / dist) } else { (0.0, 1.0) };
    if inside {
        nx = -nx;
        ny = -ny;
    }
    let pen = if inside { b.r + dist } else { b.r - dist };
    b.x += nx * pen;
    b.y += ny * pen;
    let vn = b.vx * nx + b.vy * ny;
    if vn < 0.0 {
        b.vx -= (1.0 + rest) * vn * nx;
        b.vy -= (1.0 + rest) * vn * ny;
        let (tx, ty) = (-ny, nx);
        let vt = b.vx * tx + b.vy * ty;
        b.omega -= fric * vt / b.r;
        b.vx -= fric * vt * tx;
        b.vy -= fric * vt * ty;
    }
}

/// Separating-axis overlap of two convex polygons: minimum translation vector (nx, ny, depth)
/// pushing b away from a, or None if separated.
pub fn sat_overlap(a: &Poly, b: &Poly) -> Option<(f64, f64, f64)> {
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for p in [a, b] {
        for (e0, e1) in poly_edges(p) {
            let (mut nx, mut ny) = (-(e1.1 - e0.1), e1.0 - e0.0);
            let len = (nx * nx + ny * ny).sqrt();
            if len < 1e-12 {
                continue;
            }
            nx /= len;
            ny /= len;
            let proj = |q: &Poly| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for (x, y) in &q.v {
                    let d = x * nx + y * ny;
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
                (lo, hi)
            };
            let (la, ha) = proj(a);
            let (lb, hb) = proj(b);
            let overlap = ha.min(hb) - la.max(lb);
            if overlap <= 0.0 {
                return Option::None;
            }
            if overlap < best.0 {
                let ca = (a.v.iter().map(|v| v.0).sum::<f64>() / a.v.len() as f64, a.v.iter().map(|v| v.1).sum::<f64>() / a.v.len() as f64);
                let cb = (b.v.iter().map(|v| v.0).sum::<f64>() / b.v.len() as f64, b.v.iter().map(|v| v.1).sum::<f64>() / b.v.len() as f64);
                let dir = (cb.0 - ca.0) * nx + (cb.1 - ca.1) * ny;
                let (nx, ny) = if dir < 0.0 { (-nx, -ny) } else { (nx, ny) };
                best = (overlap, nx, ny);
            }
        }
    }
    Option::Some((best.1, best.2, best.0))
}

/// Resolve two convex polygons by the SAT minimum translation vector + normal impulse.
pub fn resolve_poly(a: &mut Poly, b: &mut Poly, rest: f64) {
    if let Some((nx, ny, depth)) = sat_overlap(a, b) {
        let total = a.m + b.m;
        let share_a = b.m / total;
        let share_b = a.m / total;
        for p in a.v.iter_mut() {
            p.0 -= nx * depth * share_a;
            p.1 -= ny * depth * share_a;
        }
        for p in b.v.iter_mut() {
            p.0 += nx * depth * share_b;
            p.1 += ny * depth * share_b;
        }
        let rvx = b.vx - a.vx;
        let rvy = b.vy - a.vy;
        let vn = rvx * nx + rvy * ny;
        if vn < 0.0 {
            let imp = -(1.0 + rest) * vn / (1.0 / a.m + 1.0 / b.m);
            a.vx -= imp * nx / a.m;
            a.vy -= imp * ny / a.m;
            b.vx += imp * nx / b.m;
            b.vy += imp * ny / b.m;
            let (tx, ty) = (-ny, nx);
            let vt = rvx * tx + rvy * ty;
            a.omega -= 0.1 * vt;
            b.omega += 0.1 * vt;
        }
    }
}

fn cross2(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn point_in_tri(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(neg && pos)
}

/// Ear-clipping triangulation of a simple (possibly concave) polygon into convex triangles.
pub fn triangulate(v: &[(f64, f64)]) -> Vec<Poly> {
    let mut pts: Vec<(f64, f64)> = v.to_vec();
    let area: f64 = (0..pts.len())
        .map(|i| {
            let j = (i + 1) % pts.len();
            pts[i].0 * pts[j].1 - pts[j].0 * pts[i].1
        })
        .sum();
    if area < 0.0 {
        pts.reverse();
    }
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    let mut out = Vec::new();
    let mk = |a: usize, b: usize, c: usize| Poly {
        v: vec![pts[a], pts[b], pts[c]],
        m: 1.0, vx: 0.0, vy: 0.0, theta: 0.0, omega: 0.0,
    };
    while idx.len() > 3 {
        let mut ear = Option::None;
        for i in 0..idx.len() {
            let a = idx[(i + idx.len() - 1) % idx.len()];
            let b = idx[i];
            let c = idx[(i + 1) % idx.len()];
            if cross2(pts[a], pts[b], pts[c]) <= 1e-12 {
                continue;
            }
            let mut contains = false;
            for &j in &idx {
                if j == a || j == b || j == c {
                    continue;
                }
                if point_in_tri(pts[j], pts[a], pts[b], pts[c]) {
                    contains = true;
                    break;
                }
            }
            if !contains {
                ear = Some((i, a, b, c));
                break;
            }
        }
        match ear {
            Some((i, a, b, c)) => {
                out.push(mk(a, b, c));
                idx.remove(i);
            }
            Option::None => break,
        }
    }
    if idx.len() == 3 {
        out.push(mk(idx[0], idx[1], idx[2]));
    }
    out
}

/// Circle vs a concave polygon: resolve against each convex triangle of its decomposition.
pub fn circle_concave(b: &mut Ball, verts: &[(f64, f64)], rest: f64, fric: f64) {
    for tri in triangulate(verts) {
        circle_poly(b, &tri, rest, fric);
    }
}

/// A triangle mesh (non-polygon contact manifold): a soup of convex triangles.
#[derive(Clone, Debug)]
pub struct Mesh {
    pub tris: Vec<[(f64, f64); 3]>,
}

fn tri_poly(t: &[(f64, f64); 3]) -> Poly {
    Poly { v: t.to_vec(), m: 1.0, vx: 0.0, vy: 0.0, theta: 0.0, omega: 0.0 }
}

/// Boundary edges of a mesh (edges used by exactly one triangle) — internal triangle edges
/// are not contact surfaces.
fn mesh_boundary(m: &Mesh) -> Vec<((f64, f64), (f64, f64))> {
    let mut count: std::collections::BTreeMap<(u64, u64, u64, u64), usize> = std::collections::BTreeMap::new();
    let key = |a: (f64, f64), b: (f64, f64)| {
        let ka = (a.0.to_bits(), a.1.to_bits());
        let kb = (b.0.to_bits(), b.1.to_bits());
        if ka <= kb { (ka.0, ka.1, kb.0, kb.1) } else { (kb.0, kb.1, ka.0, ka.1) }
    };
    let mut edges: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for t in &m.tris {
        for i in 0..3 {
            let a = t[i];
            let c = t[(i + 1) % 3];
            *count.entry(key(a, c)).or_insert(0) += 1;
            edges.push((a, c));
        }
    }
    edges.into_iter().filter(|(a, c)| count[&key(*a, *c)] == 1).collect()
}

fn point_in_mesh(px: f64, py: f64, boundary: &[((f64, f64), (f64, f64))]) -> bool {
    let mut ins = false;
    for (a, c) in boundary {
        if (a.1 > py) != (c.1 > py) {
            let xin = a.0 + (py - a.1) * (c.0 - a.0) / (c.1 - a.1);
            if px < xin {
                ins = !ins;
            }
        }
    }
    ins
}

/// Circle vs mesh: contact only on the mesh's outer boundary (not internal triangle edges).
pub fn circle_mesh(b: &mut Ball, m: &Mesh, rest: f64, fric: f64) {
    let boundary = mesh_boundary(m);
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for (a, c) in &boundary {
        let q = closest_on_segment(b.x, b.y, *a, *c);
        let d = ((b.x - q.0) * (b.x - q.0) + (b.y - q.1) * (b.y - q.1)).sqrt();
        if d < best.0 {
            best = (d, q.0, q.1);
        }
    }
    let (dist, qx, qy) = best;
    let inside = point_in_mesh(b.x, b.y, &boundary);
    if !inside && dist >= b.r {
        return;
    }
    let (mut nx, mut ny) = if dist > 1e-12 { ((b.x - qx) / dist, (b.y - qy) / dist) } else { (0.0, 1.0) };
    if inside {
        nx = -nx;
        ny = -ny;
    }
    let pen = if inside { b.r + dist } else { b.r - dist };
    b.x += nx * pen;
    b.y += ny * pen;
    let vn = b.vx * nx + b.vy * ny;
    if vn < 0.0 {
        b.vx -= (1.0 + rest) * vn * nx;
        b.vy -= (1.0 + rest) * vn * ny;
        let (tx, ty) = (-ny, nx);
        let vt = b.vx * tx + b.vy * ty;
        b.omega -= fric * vt / b.r;
        b.vx -= fric * vt * tx;
        b.vy -= fric * vt * ty;
    }
}

/// First overlapping triangle pair between two meshes (SAT), as an MTV.
pub fn mesh_overlap(a: &Mesh, b: &Mesh) -> Option<(f64, f64, f64)> {
    for ta in &a.tris {
        for tb in &b.tris {
            if let Some(o) = sat_overlap(&tri_poly(ta), &tri_poly(tb)) {
                return Some(o);
            }
        }
    }
    Option::None
}

/// Separate two meshes: iteratively apply the first overlapping triangle pair's MTV until
/// no triangle pair overlaps (bounded iterations).
pub fn resolve_mesh(a: &mut Mesh, b: &mut Mesh, _rest: f64) {
    for _ in 0..16 {
        let Some((nx, ny, depth)) = mesh_overlap(a, b) else { break };
        for t in a.tris.iter_mut() {
            for p in t.iter_mut() {
                p.0 -= nx * (depth / 2.0 + 1e-9);
                p.1 -= ny * (depth / 2.0 + 1e-9);
            }
        }
        for t in b.tris.iter_mut() {
            for p in t.iter_mut() {
                p.0 += nx * (depth / 2.0 + 1e-9);
                p.1 += ny * (depth / 2.0 + 1e-9);
            }
        }
    }
}

/// One stateful step of Plant2 (for closed-loop / hardware-in-the-loop harnesses).
pub fn step2(p: &Plant2, x: f64, v: f64, act: f64) -> (f64, f64) {
    let acc = act - p.gravity - p.drag * v;
    let mut nv = v + acc * p.dt;
    let mut nx = x + nv * p.dt;
    if nx < p.wall {
        nx = p.wall;
        nv = -nv * p.restitution;
    }
    (nx, nv)
}

/// Real-to-sim over the richer parameter set: pick the candidate plant whose trace best
/// explains the recording; None when no candidate is within tol.
pub fn identify2(acts: &[f64], recorded: &[f64], candidates: &[Plant2], tol: f64) -> Option<(Plant2, f64)> {
    if !tol.is_finite() || tol < 0.0 {
        return Option::None;
    }
    let mut best: Option<(Plant2, f64)> = Option::None;
    for p in candidates {
        if !p.dt.is_finite() || p.dt <= 0.0 {
            continue;
        }
        let d = match divergence(recorded, &simulate2(p, acts)) {
            Option::Some(d) => d,
            Option::None => continue,
        };
        let better = match best {
            Option::None => true,
            Option::Some((_, bd)) => d < bd,
        };
        if better {
            best = Option::Some((*p, d));
        }
    }
    match best {
        Option::Some((p, d)) if d <= tol => Option::Some((p, d)),
        _ => Option::None,
    }
}

/// Real-to-sim seed (system identification): pick the candidate dt whose simulated trace
/// best explains the recorded trace; return (dt, divergence) only if within tol.
/// None when no candidate explains the data — the sim2real gap stays visible, not hidden.
pub fn identify(acts: &[f64], recorded: &[f64], candidates: &[f64], tol: f64) -> Option<(f64, f64)> {
    if !tol.is_finite() || tol < 0.0 {
        return Option::None;
    }
    let mut best: Option<(f64, f64)> = Option::None;
    for dt in candidates {
        if !dt.is_finite() || *dt <= 0.0 {
            continue;
        }
        let d = match divergence(recorded, &simulate(&Plant { dt: *dt }, acts)) {
            Option::Some(d) => d,
            Option::None => continue,
        };
        let better = match best {
            Option::None => true,
            Option::Some((_, bd)) => d < bd,
        };
        if better {
            best = Option::Some((*dt, d));
        }
    }
    match best {
        Option::Some((dt, d)) if d <= tol => Option::Some((dt, d)),
        _ => Option::None,
    }
}
