//! The runtime functions of text written in the core (src/one/rtt.rs),
//! against what the reference interpreter computes for the same methods
//! (interp `builtin`, whose code each expectation below restates), on
//! texts with multi-byte characters, the white space of 6.3 N7 and line
//! endings.
use rce_spec_check::interp::is_ws;
use rce_spec_check::one::{self, build::*, ir::*, rt, rtt};

const TEXTS: &[&str] = &[
    "", " ", "a", "  a b  ", "\u{3000}x\u{2028}", "héllo wörld", "a\r\nb\n\nc\r", "a\n", "\n", "\r\n", "abc", "a,b,,c", ",", ",,",
    "日本語", "\u{85}x\u{85}", "tab\there", "x\u{200b}y", "AbC xYz", "élan", "a\u{a0}b", "\u{1680}", "1234", "12a", " 7 ", "a\rb",
];
const SEPS: &[&str] = &["", ",", ",,", "ab", "é", "\n", "語", "xyz"];

/// Run `body` on a fresh entry function and return what the module prints.
fn run(build: impl FnOnce(&mut Fns, &mut FB)) -> String {
    let mut f = Fns::default();
    let entry = f.declare("entry", vec![], Ty::unit());
    let mut b = FB::new(vec![], Ty::unit());
    build(&mut f, &mut b);
    let u = b.unit().unwrap();
    b.ret(u);
    f.define(entry, b.finish());
    let m = f.finish(entry).unwrap();
    one::wf::check(&m).unwrap_or_else(|e| panic!("not well formed: {e}"));
    let r = one::eval::run(&m).unwrap_or_else(|e| panic!("{e}"));
    assert!(r.halt.is_none(), "halted: {:?}", r.halt);
    String::from_utf8(r.out).unwrap()
}

/// Print a text as `len:bytes|`.
fn put(f: &mut Fns, b: &mut FB, t: Atom) {
    let d = rt::dec(f).unwrap();
    let n = b.p("bytes.len", vec![], vec![t.clone()]).unwrap();
    let ds = b.call(sym(d), Ty::Bytes, vec![n]);
    let l = b.cats(vec![ds, Atom::bytes(b":"), t, Atom::bytes(b"|")]).unwrap();
    b.out(l);
}

fn want_text(t: &str) -> String {
    format!("{}:{t}|", t.len())
}

/// Print each text of a list, then a newline.
fn put_list(f: &mut Fns, b: &mut FB, l: Atom) {
    let n = b.p("list.len", vec![Ty::Bytes], vec![l.clone()]).unwrap();
    let d = rt::dec(f).unwrap();
    b.for_n(n, |b, i| {
        let x = b.list_at(l.clone(), i)?;
        let k = b.p("bytes.len", vec![], vec![x.clone()])?;
        let ds = b.call(sym(d), Ty::Bytes, vec![k]);
        let s = b.cats(vec![ds, Atom::bytes(b":"), x, Atom::bytes(b"|")])?;
        b.out(s);
        Ok(())
    })
    .unwrap();
    b.drop(l).unwrap();
    b.out(Atom::bytes(b"\n"));
}

fn want_list<'a>(xs: impl Iterator<Item = &'a str>) -> String {
    let mut s: String = xs.map(want_text).collect();
    s.push('\n');
    s
}

fn text_fn(make: fn(&mut Fns) -> usize, want: fn(&str) -> String) {
    let got = run(|f, b| {
        let id = make(f);
        for t in TEXTS {
            let r = b.call(sym(id), Ty::Bytes, vec![Atom::bytes(t.as_bytes())]);
            put(f, b, r);
        }
    });
    let w: String = TEXTS.iter().map(|t| want_text(&want(t))).collect();
    assert_eq!(got, w);
}

#[test]
fn strip_case_reverse() {
    text_fn(|f| rtt::strip(f).unwrap(), |t| t.trim_matches(is_ws).to_string());
    text_fn(|f| rtt::case(f, true).unwrap(), |t| t.to_ascii_uppercase());
    text_fn(|f| rtt::case(f, false).unwrap(), |t| t.to_ascii_lowercase());
    text_fn(|f| rtt::reverse(f).unwrap(), |t| t.chars().rev().collect());
}

fn list_fn(make: fn(&mut Fns) -> usize, want: fn(&str) -> Vec<String>) {
    let got = run(|f, b| {
        let id = make(f);
        for t in TEXTS {
            let r = b.call(sym(id), Ty::list(Ty::Bytes), vec![Atom::bytes(t.as_bytes())]);
            put_list(f, b, r);
        }
    });
    let w: String = TEXTS.iter().map(|t| want_list(want(t).iter().map(|x| x.as_str()))).collect();
    assert_eq!(got, w);
}

#[test]
fn chars_words_lines() {
    list_fn(|f| rtt::chars(f).unwrap(), |t| t.chars().map(|c| c.to_string()).collect());
    list_fn(|f| rtt::words(f).unwrap(), |t| t.split(is_ws).filter(|w| !w.is_empty()).map(String::from).collect());
    list_fn(|f| rtt::lines(f).unwrap(), |t| t.lines().map(String::from).collect());
}

#[test]
fn split_and_search() {
    let got = run(|f, b| {
        let sp = rtt::split(f).unwrap();
        let co = rtt::contains(f).unwrap();
        let sw = rtt::affix(f, true).unwrap();
        let ew = rtt::affix(f, false).unwrap();
        for t in TEXTS {
            for d in SEPS {
                let (ta, da) = (Atom::bytes(t.as_bytes()), Atom::bytes(d.as_bytes()));
                let r = b.call(sym(sp), Ty::list(Ty::Bytes), vec![ta.clone(), da.clone()]);
                put_list(f, b, r);
                for g in [co, sw, ew] {
                    let r = b.call(sym(g), Ty::Bool, vec![ta.clone(), da.clone()]);
                    let s = b.var(Ty::Bytes, Atom::bytes(b"F"));
                    b.when(r, |b| {
                        b.set(s, Atom::bytes(b"T"));
                        Ok(())
                    })
                    .unwrap();
                    let sv = b.take(s);
                    b.out(sv);
                }
                b.out(Atom::bytes(b"\n"));
            }
        }
    });
    let mut w = String::new();
    for t in TEXTS {
        for d in SEPS {
            w += &want_list(t.split(d));
            for r in [t.contains(d), t.starts_with(d), t.ends_with(d)] {
                w += if r { "T" } else { "F" };
            }
            w += "\n";
        }
    }
    assert_eq!(got, w);
}

#[test]
fn classes() {
    let got = run(|f, b| {
        let ids: Vec<usize> = ["isdigit", "isalpha", "isspace"].iter().map(|c| rtt::is_class(f, c).unwrap()).collect();
        for t in TEXTS {
            for id in &ids {
                let r = b.call(sym(*id), Ty::Bool, vec![Atom::bytes(t.as_bytes())]);
                let s = b.var(Ty::Bytes, Atom::bytes(b"F"));
                b.when(r, |b| {
                    b.set(s, Atom::bytes(b"T"));
                    Ok(())
                })
                .unwrap();
                let sv = b.take(s);
                b.out(sv);
            }
        }
    });
    let mut w = String::new();
    for t in TEXTS {
        let ne = !t.is_empty();
        w += if ne && t.chars().all(|c| c.is_ascii_digit()) { "T" } else { "F" };
        w += if ne && t.chars().all(|c| c.is_ascii_alphabetic()) { "T" } else { "F" };
        w += if ne && t.chars().all(|c| c.is_ascii_whitespace()) { "T" } else { "F" };
    }
    assert_eq!(got, w);
}

/// Print the value of a call returning an integer, or the text of its trap.
fn put_caught(f: &mut Fns, b: &mut FB, id: usize, ret: Ty, args: Vec<Atom>, show: fn(&mut Fns, &mut FB, Atom)) {
    let c = b.catch(sym(id), ret.clone(), args);
    let ct = b.ty(&c);
    let tag = b.p("sum.tag", vec![ct.clone()], vec![c.clone()]).unwrap();
    let ok = b.int_cmp("int.eq", tag, lit(0)).unwrap();
    let mut fns = std::mem::take(f);
    b.if_(ok, |b| {
        let v = b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(0)])?;
        show(&mut fns, b, v);
        Ok(())
    }, |b| {
        let tag = b.p("sum.tag", vec![ct.clone()], vec![c.clone()])?;
        let one = b.int_cmp("int.eq", tag, lit(1))?;
        b.guard_halt_unless(one, "internal: tag")?;
        let h = b.p("sum.get", vec![ct.clone()], vec![c.clone(), lit(1)])?;
        let ht = Ty::halt();
        let t = b.p("sum.tag", vec![ht.clone()], vec![h.clone()])?;
        let z = b.int_cmp("int.eq", t, lit(0))?;
        b.guard_halt_unless(z, "a declared error")?;
        let m = b.p("sum.get", vec![ht], vec![h, lit(0)])?;
        b.out(Atom::bytes(b"trap "));
        b.out(m);
        Ok(())
    })
    .unwrap();
    *f = fns;
    b.out(Atom::bytes(b"\n"));
}

fn show_int(f: &mut Fns, b: &mut FB, v: Atom) {
    let d = rt::dec(f).unwrap();
    let s = b.call(sym(d), Ty::Bytes, vec![v]);
    b.out(s);
}

#[test]
fn toint() {
    let ins: Vec<String> = [
        "0", "-0", "+5", " 42 ", "-", "+", "", "12a", "9223372036854775807", "9223372036854775808", "-9223372036854775808",
        "-9223372036854775809", "170141183460469231731687303715884105727", "170141183460469231731687303715884105728",
        "-170141183460469231731687303715884105728", "-170141183460469231731687303715884105729", "0000000000000000000000000000000000000000000001",
        "\u{3000}7\u{3000}", "1_000", "--1", "+-1", "007", "-00000000000000000000000000000000000000000000000",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let got = run(|f, b| {
        let id = rtt::toint(f).unwrap();
        for s in &ins {
            put_caught(f, b, id, Ty::i64(), vec![Atom::bytes(s.as_bytes())], show_int);
        }
    });
    let mut w = String::new();
    for s in &ins {
        w += &match s.trim_matches(is_ws).parse::<i128>() {
            Err(_) => format!("trap not an integer: {s:?}"),
            Ok(v) if v < i64::MIN as i128 || v > i64::MAX as i128 => format!("trap overflow: {v} is outside i64"),
            Ok(v) => v.to_string(),
        };
        w += "\n";
    }
    assert_eq!(got, w);
}

fn show_text(f: &mut Fns, b: &mut FB, v: Atom) {
    put(f, b, v);
}

#[test]
fn utf8_of_bytes() {
    let ins: Vec<Vec<i128>> = vec![
        vec![],
        vec![104, 105],
        vec![0xc3, 0xa9],
        vec![0xe6, 0x97, 0xa5],
        vec![0xf0, 0x9f, 0x98, 0x80],
        vec![0xc0, 0x80],
        vec![0xe0, 0x80, 0x80],
        vec![0xed, 0xa0, 0x80],
        vec![0xf4, 0x90, 0x80, 0x80],
        vec![0xf5, 0x80, 0x80, 0x80],
        vec![0x80],
        vec![0xc3],
        vec![104, 256],
        vec![-1],
        vec![0x7f, 0xdf, 0xbf, 0xef, 0xbf, 0xbf, 0xf4, 0x8f, 0xbf, 0xbf],
    ];
    let et = Ty::Int(-1, 256);
    let got = run(|f, b| {
        let id = rtt::utf8(f, et.clone()).unwrap();
        for xs in &ins {
            let mut l = b.p("list.new", vec![et.clone()], vec![]).unwrap();
            for x in xs {
                l = b.p("list.push", vec![et.clone()], vec![l, Atom::K(Ty::Int(*x, *x), Lit::Int(*x))]).unwrap();
            }
            put_caught(f, b, id, Ty::Bytes, vec![l], show_text);
        }
    });
    let mut w = String::new();
    for xs in &ins {
        let r = if xs.iter().any(|x| !(0..=255).contains(x)) {
            "trap utf8: byte outside 0..255".to_string()
        } else {
            match String::from_utf8(xs.iter().map(|x| *x as u8).collect()) {
                Ok(s) => want_text(&s),
                Err(_) => "trap utf8: invalid UTF-8".to_string(),
            }
        };
        w += &r;
        w += "\n";
    }
    assert_eq!(got, w);
}

#[test]
fn join_and_boundary() {
    let got = run(|f, b| {
        let j = rtt::join(f).unwrap();
        let bd = rtt::boundary(f).unwrap();
        for sep in SEPS {
            let mut l = b.p("list.new", vec![Ty::Bytes], vec![]).unwrap();
            for t in ["a", "", "é", "bc"] {
                l = b.p("list.push", vec![Ty::Bytes], vec![l, Atom::bytes(t.as_bytes())]).unwrap();
            }
            let r = b.call(sym(j), Ty::Bytes, vec![Atom::bytes(sep.as_bytes()), l]);
            put(f, b, r);
        }
        for t in TEXTS {
            for i in 0..=t.len() as i128 + 1 {
                let r = b.call(sym(bd), Ty::Bool, vec![Atom::bytes(t.as_bytes()), Atom::K(Ty::Int(i, i), Lit::Int(i))]);
                let s = b.var(Ty::Bytes, Atom::bytes(b"F"));
                b.when(r, |b| {
                    b.set(s, Atom::bytes(b"T"));
                    Ok(())
                })
                .unwrap();
                let sv = b.take(s);
                b.out(sv);
            }
        }
    });
    let mut w = String::new();
    for sep in SEPS {
        w += &want_text(&["a", "", "é", "bc"].join(sep));
    }
    for t in TEXTS {
        for i in 0..=t.len() + 1 {
            w += if t.is_char_boundary(i) { "T" } else { "F" };
        }
    }
    assert_eq!(got, w);
}

#[test]
fn context_encoders() {
    let ctxs = ["Html", "Url", "Sql", "Json", "ShellArg", "Header"];
    let got = run(|f, b| {
        let ids: Vec<usize> = ctxs.iter().map(|c| rtt::encode(f, c).unwrap()).collect();
        for t in TEXTS.iter().chain(["<script>x</script>", "O'Brien", "a b&c", "\"q\"\\", "\u{1}"].iter()) {
            for id in &ids {
                put_caught(f, b, *id, Ty::Bytes, vec![Atom::bytes(t.as_bytes())], show_text);
            }
        }
    });
    let mut w = String::new();
    for t in TEXTS.iter().chain(["<script>x</script>", "O'Brien", "a b&c", "\"q\"\\", "\u{1}"].iter()) {
        for c in ctxs {
            if c == "Header" && (t.contains('\r') || t.contains('\n')) {
                w += "trap Header.encode: a header value contains CR or LF (F.5.7)\n";
            } else {
                w += &want_text(&rce_spec_check::index08::encode(c, t));
                w += "\n";
            }
        }
    }
    assert_eq!(got, w);
}

/// `json_esc_size` against interp `json_size` of the same text.
#[test]
fn json_escape_size() {
    let mut ts: Vec<String> = TEXTS.iter().map(|t| t.to_string()).collect();
    ts.extend(["\"q\"", "a\\b", "\u{0}\u{1f}\u{7f}", "\t\r\n", "\u{1b}[0m"].map(String::from));
    let got = run(|f, b| {
        let id = rtt::json_esc_size(f).unwrap();
        let d = rt::dec(f).unwrap();
        for t in &ts {
            let r = b.call(sym(id), Ty::Int(0, i64::MAX as i128), vec![Atom::bytes(t.as_bytes())]);
            let s = b.call(sym(d), Ty::Bytes, vec![r]);
            let l = b.cat(s, Atom::bytes(b"\n")).unwrap();
            b.out(l);
        }
    });
    let want: String = ts.iter().map(|t| format!("{}\n", rce_spec_check::interp::json_size(&rce_spec_check::interp::V::Text(t.as_str().into())))).collect();
    assert_eq!(got, want);
}
