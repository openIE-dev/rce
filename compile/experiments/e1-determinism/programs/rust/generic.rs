use std::collections::BTreeMap;

trait Area {
    fn area(&self) -> f64;
}

struct Rect(f64, f64);
struct Circle(f64);

impl Area for Rect {
    fn area(&self) -> f64 {
        self.0 * self.1
    }
}

impl Area for Circle {
    fn area(&self) -> f64 {
        3.0 * self.0 * self.0
    }
}

fn total<T: Area>(xs: &[T]) -> f64 {
    xs.iter().map(|x| x.area()).sum()
}

fn main() {
    let rects = [Rect(1.0, 2.0), Rect(3.0, 4.0)];
    let circles = [Circle(1.0), Circle(2.0)];
    let mut m = BTreeMap::new();
    m.insert("rects", total(&rects));
    m.insert("circles", total(&circles));
    for (k, v) in &m {
        println!("{k} {v}");
    }
}
