#![allow(dead_code)]
mod parse;
mod canon;
use std::sync::atomic::{AtomicBool, Ordering};
#[derive(Debug, PartialOrd, PartialEq, Hash, Clone, Eq)]
pub(crate) enum InternalEvent {
    Event(crossterm::event::Event),
    CursorPosition(u16, u16),
    KeyboardEnhancementFlags(crossterm::event::KeyboardEnhancementFlags),
    PrimaryDeviceAttributes,
}
pub static RAW: AtomicBool = AtomicBool::new(false);
pub fn raw() -> bool { RAW.load(Ordering::Relaxed) }
struct R(u64);
impl R {
    fn n(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn b(&mut self, k: u64) -> u64 { self.n() % k }
    fn pick<'a>(&mut self, v: &[&'a [u8]]) -> &'a [u8] { v[self.b(v.len() as u64) as usize] }
}
fn num(r: &mut R) -> Vec<u8> {
    let v: u64 = match r.b(8) { 0 => r.b(10), 1 => r.b(40), 2 => r.b(300), 3 => 57344 + r.b(120), 4 => r.b(70000), 5 => 1 + r.b(256), 6 => r.b(1 << 20), _ => r.b(140) };
    if r.b(20) == 0 { return b"x".to_vec(); }
    if r.b(25) == 0 { return Vec::new(); }
    v.to_string().into_bytes()
}
fn params(r: &mut R) -> Vec<u8> {
    let mut o = Vec::new();
    let k = r.b(5);
    for i in 0..k {
        if i > 0 { o.push(b';'); }
        o.extend(num(r));
        if r.b(5) == 0 { o.push(b':'); o.extend(num(r)); }
    }
    o
}
fn piece(r: &mut R) -> Vec<u8> {
    let mut o = Vec::new();
    match r.b(16) {
        0 => o.extend(r.pick(&[b"a", b"Z", b"\r", b"\n", b"\t", b"\x7f", b"\x00", b"\x01", b"\x1a", b"\x1c", b"\x1f", b"\x1b", b" ", b"~", b"1"])),
        1 => { o.push(0x1b); o.push(r.b(128) as u8); }
        2 => { o.extend(b"\x1bO"); o.push(r.pick(&[b"A", b"B", b"C", b"D", b"H", b"F", b"P", b"Q", b"R", b"S", b"x", b"1"])[0]); }
        3 | 4 | 5 => { o.extend(b"\x1b["); o.extend(params(r)); o.push(r.pick(&[b"~", b"u", b"A", b"B", b"C", b"D", b"F", b"H", b"P", b"Q", b"R", b"S", b"M", b"Z", b"I", b"O", b"m", b"c", b"q", b"@", b"z"])[0]); }
        6 => { o.extend(b"\x1b[<"); o.extend(params(r)); o.push(if r.b(2) == 0 { b'M' } else { b'm' }); }
        7 => { o.extend(b"\x1b[M"); for _ in 0..3 { o.push(r.b(256) as u8); } }
        8 => { o.extend(b"\x1b[?"); o.extend(params(r)); o.push(r.pick(&[b"u", b"c", b"x"])[0]); }
        9 => { o.extend(b"\x1b[200~"); for _ in 0..r.b(6) { o.push(r.pick(&[b"a", b"\x1b", b"[", b"\xc3", b"\xa9", b"\xff", b"\n"])[0]); } if r.b(4) != 0 { o.extend(b"\x1b[201~"); } }
        10 => { let c = char::from_u32(r.b(0x110000) as u32).unwrap_or('\u{fffd}'); let mut b = [0; 4]; o.extend(c.encode_utf8(&mut b).as_bytes()); }
        11 => { for _ in 0..1 + r.b(3) { o.push((0x80 + r.b(128)) as u8); } }
        12 => { o.extend(b"\x1b[["); o.push(r.pick(&[b"A", b"E", b"F"])[0]); }
        13 => { o.extend(b"\x1b\x1b"); }
        14 => { let c = char::from_u32(0x80 + r.b(0x800) as u32).unwrap_or('a'); let mut b = [0; 4]; let s = c.encode_utf8(&mut b).as_bytes().to_vec(); o.extend(&s[..1 + r.b(s.len() as u64) as usize]); }
        _ => { o.extend(b"\x1b["); o.extend(num(r)); o.push(b';'); o.extend(num(r)); o.push(r.pick(&[b"R", b"M", b"~", b"u"])[0]); }
    }
    o
}
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
fn main() {
    // "one HEX AVAIL RAW": a single parse_event call, for derived checks
    if std::env::args().nth(1).as_deref() == Some("one") {
        let a: Vec<String> = std::env::args().collect();
        let b: Vec<u8> = (0..a[2].len()).step_by(2).map(|i| u8::from_str_radix(&a[2][i..i + 2], 16).unwrap()).collect();
        RAW.store(a[4] == "1", Ordering::Relaxed);
        println!("{}", canon::res(&parse::parse_event(&b, a[3] == "1")));
        return;
    }
    let mut r = R(0x4d46_7569_696e_7075);
    let n_one: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(20000);
    let n_stream: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(5000);
    for _ in 0..n_one {
        let b = piece(&mut r);
        let avail = r.b(2) == 0;
        let rw = r.b(4) == 0;
        RAW.store(rw, Ordering::Relaxed);
        let out = std::panic::catch_unwind(|| canon::res(&parse::parse_event(&b, avail))).unwrap_or("panic".into());
        println!("e\t{}\t{}\t{}\t{}", hex(&b), avail as u8, rw as u8, out);
    }
    for _ in 0..n_stream {
        let mut b = Vec::new();
        for _ in 0..1 + r.b(5) { b.extend(piece(&mut r)); }
        let rw = r.b(4) == 0;
        RAW.store(rw, Ordering::Relaxed);
        // crossterm's Parser::advance with more = false at the end of the read
        let mut buf: Vec<u8> = Vec::new();
        let mut evs = Vec::new();
        for (i, &x) in b.iter().enumerate() {
            let more = i + 1 < b.len();
            buf.push(x);
            match std::panic::catch_unwind(|| parse::parse_event(&buf, more)) {
                Ok(Ok(Some(e))) => { evs.push(canon::ev(&e)); buf.clear(); }
                Ok(Ok(None)) => {}
                Ok(Err(_)) => buf.clear(),
                Err(_) => { evs.push("panic".into()); buf.clear(); }
            }
        }
        println!("s\t{}\t{}\t{}\t{}", hex(&b), rw as u8, evs.join(";"), hex(&buf));
    }
}
