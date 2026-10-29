//! HC-25: the HTTP adapter's parser (`adapters::http::parse_request`) has
//! one meaning with `net.http.parse_message(b, true, false, false)`, the
//! library parser written in RCE that `stdlib/tests/net/http_checks.rce`
//! checks against a model and h11. Both run here on every string of those
//! checks and on 10,000 seeded mutations of them; the RCE side runs in the
//! reference interpreter, as `check` lines whose expected text is the Rust
//! parser's result. The rules only the host applies are listed in
//! `limits_give_their_statuses`: the limits and deadlines of HC-25, 505 for
//! a version other than 1.x.
use rce_spec_check::adapters::http::{BODY_MAX, FIELDS_MAX, HEAD_MAX, LINE_MAX, Parsed, Parser, Progress, parse_request};
use std::path::Path;

/// The string literals of every `check` line of http_checks.rce, unescaped.
fn vectors() -> Vec<Vec<u8>> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../stdlib/tests/net/http_checks.rce");
    let text = std::fs::read_to_string(p).unwrap();
    let mut out: Vec<Vec<u8>> = vec![];
    for line in text.lines().filter(|l| l.starts_with("check ")) {
        let cs: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            if cs[i] != '"' {
                i += 1;
                continue;
            }
            let mut s = String::new();
            i += 1;
            while i < cs.len() && cs[i] != '"' {
                if cs[i] == '\\' {
                    i += 1;
                    match cs[i] {
                        'r' => s.push('\r'),
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'u' => {
                            let end = (i..cs.len()).find(|&k| cs[k] == '}').unwrap();
                            let hex: String = cs[i + 2..end].iter().collect();
                            s.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
                            i = end;
                        }
                        c => s.push(c),
                    }
                } else {
                    s.push(cs[i]);
                }
                i += 1;
            }
            i += 1;
            if !out.contains(&s.as_bytes().to_vec()) {
                out.push(s.into_bytes());
            }
        }
    }
    out
}

fn pairs(ns: &[String], vs: &[String]) -> String {
    ns.iter().zip(vs).map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(";")
}

fn show(p: &Parsed, used: usize) -> String {
    let nums: Vec<String> = p.body.iter().map(|b| b.to_string()).collect();
    format!("{}|{}|{}.{}|{}|{}|{}|{used}", p.method, p.target, p.major, p.minor, pairs(&p.names, &p.values), nums.join(","), pairs(&p.trailer_names, &p.trailer_values))
}

/// The Rust parser's result in the probe's notation, or `None` for a rule
/// only the host applies (a limit, or 505).
fn expected(b: &[u8]) -> Option<String> {
    match parse_request(b) {
        Ok(None) => Some("INCOMPLETE".into()),
        Ok(Some((p, used))) => Some(show(&p, used)),
        Err((400 | 501, why)) => Some(format!("ERR:{why}")),
        Err((413 | 414 | 431 | 505, _)) => None,
        Err((s, w)) => panic!("unexpected status {s} {w}"),
    }
}

const PROBE: &str = r#"module http_probe

use net.http

fn pairs(ns: &[text], vs: &[text]) -> text
    var ps: List[text] = []
    for i in 0..ns.len()
        ps.push(ns[i] + "=" + vs[i])
    return ";".join(ps)

fn nums(b: &[i64]) -> text
    var ps: List[text] = []
    for x in b
        ps.push(x.tostr())
    return ",".join(ps)

fn failed(why: text) -> http.Message = http.Message(request: true, method: "", target: "", major: 0, minor: 0, status: 0, reason: why, names: [], values: [], body: [], trailer_names: [], trailer_values: [], used: -1)

fn probe(raw: List[i64]) -> text
    let m = http.parse_message(&raw, true, false, false) catch |e| failed(e.why)
    if m.used < 0
        return if m.reason.startswith("incomplete") then "INCOMPLETE" else "ERR:" + m.reason
    return m.method + "|" + m.target + "|" + m.major.tostr() + "." + m.minor.tostr() + "|" + pairs(&m.names, &m.values) + "|" + nums(&m.body) + "|" + pairs(&m.trailer_names, &m.trailer_values) + "|" + m.used.tostr()
"#;

fn rce_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (' '..='~').contains(&c) => o.push(c),
            c => o.push_str(&format!("\\u{{{:x}}}", c as u32)),
        }
    }
    o.push('"');
    o
}

/// Run net.http on every input in the reference interpreter and require the
/// Rust parser's result; returns how many inputs a host-only rule decided.
fn agree(inputs: &[Vec<u8>]) -> usize {
    let mut host_only = 0;
    let mut shards: Vec<String> = vec![];
    let mut cur = String::from(PROBE);
    let mut n = 0;
    for b in inputs {
        let Some(want) = expected(b) else {
            host_only += 1;
            continue;
        };
        let list: Vec<String> = b.iter().map(|x| x.to_string()).collect();
        cur.push_str(&format!("\ncheck probe([{}]) == {}\n", list.join(", "), rce_str(&want)));
        n += 1;
        // one comptime fuel budget per module: shards of 250 checks
        if n % 250 == 0 {
            shards.push(std::mem::replace(&mut cur, String::from(PROBE)));
        }
    }
    shards.push(cur);
    let fails: Vec<String> = std::thread::scope(|s| {
        let hs: Vec<_> = shards
            .chunks(shards.len().div_ceil(4).max(1))
            .map(|part| {
                s.spawn(move || {
                    let mut bad = vec![];
                    for (k, src) in part.iter().enumerate() {
                        for (_, v) in rce_spec_check::check_program(&[(format!("probe{k}.rce"), src.clone())], false) {
                            if let Err(e) = v {
                                bad.push(e);
                            }
                        }
                    }
                    bad
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    assert!(fails.is_empty(), "net.http and the host's parser differ:\n{}", fails.join("\n"));
    host_only
}

#[test]
fn vectors_agree_with_net_http() {
    let vs = vectors();
    assert!(vs.len() >= 121, "{} strings in http_checks.rce", vs.len());
    let host_only = agree(&vs);
    assert_eq!(host_only, 0, "no vector reaches a host-only rule");
}

/// A fixed, seeded generator (xorshift64*), so the corpus is the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutations(n: usize) -> Vec<Vec<u8>> {
    const BYTES: &[u8] = b"\r\n\t :;,0123456789aAzZ-/.\x00\x01\x7f\x80\xff\"Hh";
    let base = vectors();
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = vec![];
    while out.len() < n {
        let mut b = base[r.below(base.len())].clone();
        for _ in 0..1 + r.below(3) {
            let at = r.below(b.len() + 1);
            let x = if r.below(2) == 0 { BYTES[r.below(BYTES.len())] } else { r.below(256) as u8 };
            match r.below(4) {
                0 if at < b.len() => b[at] = x,
                1 => b.insert(at, x),
                2 if at < b.len() => {
                    b.remove(at);
                }
                _ => b.truncate(at),
            }
        }
        out.push(b);
    }
    out
}

#[test]
fn mutations_agree_with_net_http() {
    let ms = mutations(10_000);
    let host_only = agree(&ms);
    // a mutated version digit gives 505, which only the host applies
    assert!(host_only < 500, "{host_only} mutations reached a host-only rule");
}

/// The connection threads feed a growing buffer to one parser; the result
/// must be the result of parsing the whole buffer, whatever the split.
#[test]
fn incremental_feed_equals_whole_buffer() {
    let mut inputs = vectors();
    inputs.extend(mutations(2_000));
    for b in &inputs {
        let whole = Parser::default().feed(b);
        let mut p = Parser::default();
        let mut got = Progress::Incomplete;
        for k in 1..=b.len() {
            got = p.feed(&b[..k]);
            if got != Progress::Incomplete {
                break;
            }
        }
        assert_eq!(got, whole, "{:?}", String::from_utf8_lossy(b));
    }
}

#[test]
fn limits_give_their_statuses() {
    let status = |b: &[u8]| -> (u16, String) {
        match parse_request(b) {
            Err(e) => e,
            Ok(r) => panic!("not refused: {r:?}"),
        }
    };
    // request line over LINE_MAX: 414, before the line has ended
    let long = format!("GET /{} HTTP/1.1\r\n", "a".repeat(LINE_MAX));
    assert_eq!(status(long.as_bytes()), (414, format!("request line over {LINE_MAX} bytes")));
    assert_eq!(status(&long.as_bytes()[..LINE_MAX + 10]).0, 414);
    // fields over HEAD_MAX: 431
    let mut big = String::from("GET / HTTP/1.1\r\nHost: x\r\n");
    while big.len() <= HEAD_MAX {
        big.push_str(&format!("X-Pad: {}\r\n", "p".repeat(1000)));
    }
    assert_eq!(status(big.as_bytes()), (431, format!("field section over {HEAD_MAX} bytes")));
    // more than FIELDS_MAX fields: 431
    let mut many = String::from("GET / HTTP/1.1\r\nHost: x\r\n");
    for i in 0..FIELDS_MAX {
        many.push_str(&format!("X-{i}: v\r\n"));
    }
    many.push_str("\r\n");
    assert_eq!(status(many.as_bytes()), (431, format!("over {FIELDS_MAX} fields")));
    // a body over BODY_MAX: 413 from the head alone, by length or by chunk size
    let cl = format!("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n", BODY_MAX + 1);
    assert_eq!(status(cl.as_bytes()), (413, format!("body over {BODY_MAX} bytes")));
    let ok = format!("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: {BODY_MAX}\r\n\r\n");
    assert_eq!(parse_request(ok.as_bytes()), Ok(None), "exactly BODY_MAX waits for the body");
    let ch = format!("POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n", BODY_MAX + 1);
    assert_eq!(status(ch.as_bytes()).0, 413);
    // a transfer coding the host does not implement: 501; not a final chunked: 400
    assert_eq!(status(b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"), (501, "unsupported transfer coding".into()));
    assert_eq!(status(b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip\r\n\r\n"), (400, "transfer coding".into()));
    // a well-formed version other than 1.x: 505 (net.http parses it)
    assert_eq!(status(b"GET / HTTP/2.0\r\n\r\n").0, 505);
    assert_eq!(status(b"GET / HTTP/0.9\r\n\r\n").0, 505);
    // RFC 9112 3.2, in both parsers: no Host in HTTP/1.1, or two Host fields
    assert_eq!(status(b"GET / HTTP/1.1\r\n\r\n"), (400, "Host".into()));
    assert_eq!(status(b"GET / HTTP/1.0\r\nHost: a\r\nHOST: b\r\n\r\n"), (400, "Host".into()));
    let (p, used) = parse_request(b"GET / HTTP/1.0\r\n\r\n").unwrap().unwrap();
    assert!(p.close && used == 18, "HTTP/1.0 without Host is admitted and closes");
    let (p, _) = parse_request(b"HEAD / HTTP/1.1\r\nHost: a\r\nConnection: Close\r\n\r\n").unwrap().unwrap();
    assert!(p.head && p.close);
    let (p, _) = parse_request(b"GET / HTTP/1.0\r\nHost: a\r\nConnection: keep-alive\r\n\r\n").unwrap().unwrap();
    assert!(!p.close);
    assert_eq!(parse_request(b"POST / HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\n"), Ok(None), "a body still to come");
    let mut q = Parser::default();
    assert_eq!(q.feed(b"POST / HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\n"), Progress::Incomplete);
    assert!(q.head_done() && q.expect_continue());
}
