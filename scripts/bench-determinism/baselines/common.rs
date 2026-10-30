//! Shared by the hand-written baselines of scripts/bench-determinism.py:
//! read a stream (one event per line, example syntax), the repetition
//! count, and print the last state as the positional canonical JSON that
//! the log D hashes (6.1, T7), so the script can check that the baseline
//! reaches the state the checker reaches. Built with `--cfg bench_log`, a
//! baseline also keeps the log D of 6.1 natively: per step the positional
//! JSON of the event and of the state, state_hash = SHA-256 of the state,
//! hash = SHA-256 of {event, prev_hash, seq, state_hash}, the entry kept in
//! memory; it prints the head hash, which must equal the checker's.
#![allow(dead_code)]

pub struct Ev {
    pub name: String,
    pub args: Vec<(String, String)>,
}

impl Ev {
    pub fn get(&self, k: &str) -> &str {
        &self.args.iter().find(|(n, _)| n == k).unwrap_or_else(|| panic!("{} has no {k}", self.name)).1
    }
    pub fn int(&self, k: &str) -> i64 {
        self.get(k).parse().unwrap()
    }
    pub fn real(&self, k: &str) -> f64 {
        self.get(k).parse().unwrap()
    }
    pub fn text(&self, k: &str) -> String {
        self.get(k).trim_matches('"').to_string()
    }
}

/// (events, reps) from argv: STREAM REPS.
pub fn input() -> (Vec<Ev>, usize) {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let evs = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let l = l.trim();
            match l.split_once('(') {
                None => Ev { name: l.to_string(), args: vec![] },
                Some((n, rest)) => Ev {
                    name: n.to_string(),
                    args: rest
                        .trim_end_matches(')')
                        .split(", ")
                        .map(|kv| {
                            let (k, v) = kv.split_once(": ").unwrap();
                            (k.to_string(), v.to_string())
                        })
                        .collect(),
                },
            }
        })
        .collect();
    (evs, a[2].parse().unwrap())
}

/// The canonical JSON of a float (6.1.1 rule 3), as the checker writes it.
pub fn fj(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let e = format!("{:e}", x.abs());
    let (m, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let a = x.abs();
    if (1e-4..1e16).contains(&a) {
        let p = ex + 1;
        let body = if p <= 0 {
            format!("0.{}{}", "0".repeat((-p) as usize), digits)
        } else if p as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(p as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..p as usize], &digits[p as usize..])
        };
        format!("{sign}{body}")
    } else {
        format!("{sign}{m}e{ex}")
    }
}

/// JSON string (the checker's `jstr`, for the plain keys of these programs).
pub fn js(s: &str) -> String {
    format!("\"{s}\"")
}

// The portable SHA-256 of the emitted Rust runtime (spec 10.5), copied so the
// baseline hashes as emitted code does (no SHA instructions).
/// SHA-256 (FIPS 180-4) of a byte string as 64 lowercase hex digits (5.15 `sha256`).
pub fn sha256_hex(m: &[u8]) -> String {
    const K: [u32; 64] = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut d = m.to_vec();
    d.push(0x80);
    while d.len() % 64 != 56 { d.push(0); }
    d.extend_from_slice(&((m.len() as u64).wrapping_mul(8)).to_be_bytes());
    for c in d.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut cc, mut dd, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let t1 = hh.wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)).wrapping_add((e & f) ^ (!e & g)).wrapping_add(K[i]).wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)).wrapping_add((a & b) ^ (a & cc) ^ (b & cc));
            hh = g; g = f; f = e; e = dd.wrapping_add(t1); dd = cc; cc = b; b = a; a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, cc, dd, e, f, g, hh]) { *x = x.wrapping_add(y); }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

pub struct Chain {
    pub prev: String,
    pub seq: usize,
    pub store: Vec<String>,
}

impl Chain {
    pub fn new() -> Chain {
        Chain { prev: "0".repeat(64), seq: 0, store: vec![] }
    }
    /// Append one entry: `event` and `state` are the positional JSON (T7).
    pub fn add(&mut self, event: &str, state: &str) {
        let sh = sha256_hex(state.as_bytes());
        let body = format!("{{\"event\":{event},\"prev_hash\":\"{}\",\"seq\":{},\"state_hash\":\"{sh}\"}}", self.prev, self.seq);
        let h = sha256_hex(body.as_bytes());
        self.store.push(body);
        self.prev = h;
        self.seq += 1;
    }
}
