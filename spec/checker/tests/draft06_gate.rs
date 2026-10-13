//! The draft 0.6 concurrency syntax is behind the `conc06` cargo feature:
//! the default checker refuses it with a message naming the feature, and
//! still accepts the same words as member names.
use rce_spec_check::parse::parse;

const PAR: &str = "fn main()\n    var a: i64 = 0\n    var b: i64 = 0\n    par\n        branch one\n            a = 1\n        branch two\n            b = 2\n    emit a + b\n";

#[test]
fn par_needs_the_feature() {
    let r = parse(PAR);
    if cfg!(feature = "conc06") {
        assert!(r.is_ok(), "{r:?}");
    } else {
        let e = r.unwrap_err();
        assert!(e.contains("conc06"), "{e}");
    }
}
