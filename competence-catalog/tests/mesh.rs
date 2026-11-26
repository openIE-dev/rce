use competence_catalog::twin::{circle_mesh, mesh_overlap, resolve_mesh, Ball, Mesh};

fn square_mesh(cx: f64, cy: f64, h: f64) -> Mesh {
    Mesh {
        tris: vec![
            [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h)],
            [(cx - h, cy - h), (cx + h, cy + h), (cx - h, cy + h)],
        ],
    }
}

#[test]
fn circle_vs_mesh_resolves_outside() {
    let m = square_mesh(0.0, 0.0, 1.0);
    let mut b = Ball::new(0.0, 0.0, 0.0, 0.0, 0.3, 1.0);
    circle_mesh(&mut b, &m, 0.5, 0.0);
    // center was inside; after resolution it must be outside the square
    assert!(b.x.abs() > 1.0 || b.y.abs() > 1.0, "still inside: {b:?}");
}

#[test]
fn mesh_overlap_detects_and_separates() {
    let a = square_mesh(0.0, 0.0, 1.0);
    let b = square_mesh(1.5, 0.0, 1.0);
    assert!(mesh_overlap(&a, &b).is_some());
    let c = square_mesh(5.0, 0.0, 1.0);
    assert!(mesh_overlap(&a, &c).is_none());
}

#[test]
fn resolve_mesh_removes_overlap() {
    let mut a = square_mesh(0.0, 0.0, 1.0);
    let mut b = square_mesh(1.5, 0.0, 1.0);
    resolve_mesh(&mut a, &mut b, 0.5);
    assert!(mesh_overlap(&a, &b).is_none());
}
