// Live reference for lib/os/clock.rce: reads "s1 n1 s2 n2 k" lines, prints results.
use std::io::BufRead;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
fn o(d: Option<Duration>) -> String { match d { Some(x) => format!("{}.{}", x.as_secs(), x.subsec_nanos()), None => "none".into() } }
fn t(x: Option<SystemTime>) -> String {
    match x { None => "none".into(), Some(st) => match st.duration_since(UNIX_EPOCH) {
        Ok(d) => format!("{}.{}", d.as_secs() as i128, d.subsec_nanos()),
        Err(e) => { let d = e.duration(); let (s, n) = (-(d.as_secs() as i128), d.subsec_nanos() as i128);
                    let (s, n) = if n > 0 { (s - 1, 1_000_000_000 - n) } else { (s, n) }; format!("{}.{}", s, n) } } } }
fn main() {
    for line in std::io::stdin().lock().lines() {
        let l = line.unwrap();
        let v: Vec<u64> = l.split(' ').map(|x| x.parse().unwrap()).collect();
        let a = Duration::new(v[0], v[1] as u32);
        let b = Duration::new(v[2], v[3] as u32);
        let k = v[4] as u32;
        let f = [format!("{:?}", a), format!("{:.0?}", a), format!("{:.1?}", a), format!("{:.2?}", a), format!("{:.3?}", a),
                 format!("{:.5?}", a), format!("{:.9?}", a), format!("{:.12?}", a), format!("{:9?}", a), format!("{:>9?}", a),
                 format!("{:^9?}", a), format!("{:<12.3?}", a), format!("{:>14.1?}", a), format!("{:^11.0?}", a),
                 format!("{:+?}", a), format!("{:+.2?}", a), format!("{:>+10.1?}", a)];
        // SystemTime points: epoch + a and epoch - b (both fit i64 secs when s < 2^62)
        let pa = UNIX_EPOCH.checked_add(a);
        let pb = UNIX_EPOCH.checked_sub(b);
        let since = match (pa, pb) { (Some(x), Some(y)) => match x.duration_since(y) { Ok(d) => format!("ok {}", o(Some(d))), Err(e) => format!("err {}", o(Some(e.duration()))) }, _ => "na".into() };
        let since2 = match (pa, pb) { (Some(x), Some(y)) => match y.duration_since(x) { Ok(d) => format!("ok {}", o(Some(d))), Err(e) => format!("err {}", o(Some(e.duration()))) }, _ => "na".into() };
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", l,
            o(a.checked_add(b)), o(a.checked_sub(b)), o(Some(a.abs_diff(b))), o(a.checked_mul(k)), o(a.checked_div(k)),
            t(pa.and_then(|x| x.checked_add(b))), t(pa.and_then(|x| x.checked_sub(b))), t(pb.and_then(|x| x.checked_sub(a))),
            since, since2, f.join("\x1f"), t(pa), t(pb));
    }
}
