//! Overlapping actions: receipts + idle == meter total (law C3 accounting).
use ide_energy::attrib::Sample;
use ide_energy::meter::{Meter, MeterError};
use ide_energy::shared::SharedMeter;
use std::sync::{Arc, Mutex};

/// A scripted counter: each sample() returns the next (t, joules).
struct Script(Arc<Mutex<Vec<(u64, f64)>>>);
impl Meter for Script {
    fn source(&self) -> &'static str { "rapl" }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let mut v = self.0.lock().unwrap();
        if v.is_empty() { return Err(MeterError::Read("script", "exhausted".into())); }
        let (t, j) = v.remove(0);
        Ok(Sample { t_ns: t, joules: j })
    }
}

fn meter(points: &[(u64, f64)]) -> SharedMeter {
    SharedMeter::new(Box::new(Script(Arc::new(Mutex::new(points.to_vec())))))
}

#[test]
fn overlapping_actions_are_not_double_counted() {
    // A: [0,10], B: [5,20], idle 20..30 via tick. Counter: 0,10,20,40,45.
    let m = meter(&[(0, 0.0), (5, 10.0), (10, 20.0), (20, 40.0), (30, 45.0)]);
    let a = m.begin().unwrap(); // t0
    let b = m.begin().unwrap(); // t5
    let ja = m.end(a).unwrap(); // t10: [0,5] all A = 10, [5,10] split = 5
    let jb = m.end(b).unwrap(); // t20: [5,10] half = 5, [10,20] all B = 20
    m.tick().unwrap(); // t30: idle 5
    assert!((ja - 15.0).abs() < 1e-9, "a={ja}");
    assert!((jb - 25.0).abs() < 1e-9, "b={jb}");
    let bk = m.books();
    assert!((bk.idle - 5.0).abs() < 1e-9);
    assert!((bk.booked + bk.idle - bk.total).abs() < 1e-9, "{bk:?}");
    // The old per-window difference would have booked 20 + 30 = 50 > 45.
    assert!(ja + jb < 20.0 + 30.0);
}

#[test]
fn many_random_overlaps_balance() {
    // Deterministic LCG; 200 begin/end/tick operations, monotone counter.
    let mut x: u64 = 12345;
    let mut next = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); x >> 33 };
    let mut pts = vec![];
    let (mut t, mut j) = (0u64, 0.0f64);
    for _ in 0..400 { t += 1 + next() % 50; j += (next() % 1000) as f64 / 10.0; pts.push((t, j)); }
    let m = meter(&pts);
    let mut open = vec![];
    let mut sum = 0.0;
    for _ in 0..200 {
        match next() % 3 {
            0 => open.push(m.begin().unwrap()),
            1 if !open.is_empty() => { let k = (next() as usize) % open.len(); sum += m.end(open.remove(k)).unwrap(); }
            _ => m.tick().unwrap(),
        }
    }
    for s in open.drain(..) { sum += m.end(s).unwrap(); }
    let bk = m.books();
    assert!((bk.booked - sum).abs() < 1e-6);
    assert!((bk.booked + bk.idle - bk.total).abs() < 1e-6, "{bk:?}");
}

#[test]
fn unreadable_meter_is_not_zero() {
    let m = meter(&[(0, 0.0)]);
    let a = m.begin().unwrap();
    assert_eq!(m.end(a), None);
}
