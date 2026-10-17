//! Long text and list values evaluate in linear time (os.pkg's official
//! semver vector builds a 5.6M-character pre-release): reading a variable,
//! indexing it and pushing onto it must not copy the whole buffer. Fuel is
//! unchanged by the copy-on-write representation: the least fuel limit
//! under which the probe admits is the one the copying evaluator needed.
//! Its own test binary: it sets the process-wide `interp::FUEL_LIMIT`.
use rce_spec_check::interp::{FUEL_LIMIT, admit};
use rce_spec_check::parse::parse;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Instant;

fn probe(n: usize) -> String {
    format!(
        "module p\n\
fn rep(n: i64) -> text\n    var o: List[text] = []\n    var i: i64 = 0\n    while i < n bound 16777216\n        o.push(\"1a\")\n        i += 1\n    return \"\".join(o)\n\
fn count(t: &text) -> i64\n    let cs = t.chars()\n    var k: i64 = 0\n    var j: usize = 0\n    while j < cs.len() bound 16777216\n        if cs[j] == \"1\" and t[j] == 49\n            k += 1\n        j += 1\n    var xs: List[i64] = [1]\n    while xs.len() < 50 bound 64\n        xs.push(xs[xs.len() - 1] * 2 % 1000)\n    let u = t.upper() + \"x\"\n    xs.pop()\n    return k + xs.len() as i64 + u.len() as i64\n\
fn go() -> i64\n    let s = rep({n})\n    return count(&s)\n\
check go() == {}\n",
        3 * n + 50
    )
}

fn admits(src: &str) -> bool {
    admit(&parse(src).unwrap()).is_empty()
}

#[test]
fn long_text_is_linear_and_fuel_is_unchanged() {
    let lim = FUEL_LIMIT.load(SeqCst);
    // fuel parity: the least admitting limit of the small probe, as the
    // copying evaluator (rce/host 0e5ee828d) computed it
    let src = probe(3000);
    let (mut lo, mut hi) = (1u64, lim);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        FUEL_LIMIT.store(mid, SeqCst);
        if admits(&src) { hi = mid } else { lo = mid + 1 }
    }
    FUEL_LIMIT.store(lim, SeqCst);
    println!("least fuel {lo}");
    assert_eq!(lo, LEAST_FUEL_3000, "fuel accounting changed");
    // linear time: an 800 000-byte text (the copying evaluator took hours).
    // N9 (interp::SIZE_LIMIT = 1 000 000) traps any list or text built above
    // that size, and the probe builds the text, its upper-cased copy and one
    // more byte, so the probe stays under the limit by construction.
    let t = Instant::now();
    assert!(admits(&probe(400_000)));
    assert!(t.elapsed().as_secs() < 30, "800k-byte probe took {:?}", t.elapsed());
}

const LEAST_FUEL_3000: u64 = 150_899;
