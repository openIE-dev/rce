//! 6.3 N7, Unicode pin: no text operation of the runtime reads the Unicode
//! tables of the toolchain that built it. Text is scalar values in UTF-8
//! (version-independent); case and class operations are ASCII; the
//! whitespace of `strip`, `words` and `toint` is the White_Space set of
//! Unicode 18.0.0, enumerated in `interp::WHITE_SPACE` and in the emitted
//! runtimes. Display writes as `\u{h}` exactly the code points listed in
//! `spec/data/display-escape.json` (`interp::display_escaped`), the set
//! Rust 1.98.1's `{:?}` escapes, enumerated once. A Rust or Python upgrade
//! with new Unicode tables therefore cannot change a result.
use std::process::Command;

fn run(src: &str, name: &str) -> String {
    let p = std::env::temp_dir().join(format!("rce-unicode-pin-{}-{name}.rce", std::process::id()));
    std::fs::write(&p, src).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).arg("run").arg(&p).output().unwrap();
    let _ = std::fs::remove_file(&p);
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn white_space_is_the_pinned_set() {
    // 25 code points; U+180E left White_Space in Unicode 6.3 and U+FEFF and
    // U+200B were never in it, so they are content, not whitespace
    assert_eq!(rce_spec_check::interp::WHITE_SPACE.len(), 25);
    for c in rce_spec_check::interp::WHITE_SPACE {
        assert!(c.is_whitespace(), "{c:?}");
    }
    for c in ['\u{180e}', '\u{feff}', '\u{200b}', '\u{1c}', '\u{1f}'] {
        assert!(!rce_spec_check::interp::is_ws(c), "{c:?}");
    }
}

#[test]
fn strip_words_toint_use_the_pinned_set() {
    let src = "fn main()\n    emit \"\\u{3000}\\u{a0}a\\u{2009}b\\u{180e}c\\u{85}\".strip()\n    emit \"\\u{3000}a\\u{feff}b\\u{a0}c\".words().len().tostr()\n    emit \"\\u{2003}42\\u{2028}\".toint().tostr()\n";
    let out = run(src, "ws");
    assert!(out.contains(r#"emit "a\u{2009}b\u{180e}c""#), "{out}");
    assert!(out.contains("emit \"2\""), "{out}");
    assert!(out.contains("emit \"42\""), "{out}");
}

#[test]
fn runtime_sources_call_no_unicode_table() {
    // std functions whose answer depends on the toolchain's Unicode version
    let banned = [
        ".trim()", "trim_start()", "trim_end()", "split_whitespace", "char::is_whitespace",
        ".is_whitespace()", ".is_alphabetic()", ".is_alphanumeric()", ".is_numeric()",
        ".is_uppercase()", ".is_lowercase()", ".to_uppercase()", ".to_lowercase()",
        ".isspace()", ".isalpha()", ".isdigit()", ".isupper()", ".islower()", ".strip()",
        ".split()", ".upper()", ".lower()", ".casefold()", "unicodedata", ".trim(", "\\s",
        "toLocale", "Intl.", ".normalize(", "Character.is", "isWhitespace", ".toUpperCase()",
        ".toLowerCase()",
    ];
    // the two JavaScript helpers apply toUpperCase/toLowerCase to [a-z]/[A-Z] only
    let allowed = ["const aupper = (s) => s.replace(/[a-z]/g", "const alower = (s) => s.replace(/[A-Z]/g", "toString(16).toUpperCase()",
        // decimal.Decimal.normalize strips trailing zeros of a number; it is not Unicode normalization
        "decimal.Decimal(repr(abs(x))).normalize()"];
    let mut bad = Vec::new();
    for f in ["src/interp.rs", "src/emit/rt.rs", "src/emit/rt.py", "src/emit/rt.js", "src/emit/rt.scala"] {
        let text = std::fs::read_to_string(f).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap().split(" # ").next().unwrap();
            if allowed.iter().any(|a| code.contains(a)) || code.trim_start().starts_with('#') {
                continue;
            }
            for b in banned {
                if code.contains(b) {
                    bad.push(format!("{f}:{}: `{b}`: {}", n + 1, line.trim()));
                }
            }
        }
    }
    // the emitted Rust runtime writes text by `rdebug` from the data, never
    // by `{:?}` of a str; `{:?}` of a float is digits, and `dbg_v` and
    // `dbg_flow` mirror the reference's derived Debug of the reason of a kept step
    let rt = std::fs::read_to_string("src/emit/rt.rs").unwrap();
    let mut in_dbg = false;
    for (n, line) in rt.lines().enumerate() {
        if line.starts_with("pub fn dbg_v(") || line.starts_with("pub fn dbg_flow(") {
            in_dbg = true;
        } else if line.starts_with('}') {
            in_dbg = false;
        }
        let code = line.split("//").next().unwrap();
        if !in_dbg && code.replace("{f:?}", "").contains(":?}") {
            bad.push(format!("src/emit/rt.rs:{}: `:?}}`: {}", n + 1, line.trim()));
        }
    }
    assert!(bad.is_empty(), "calls that read the toolchain's Unicode tables:\n{}", bad.join("\n"));
}

/// The display escape set is data: `spec/data/display-escape.json`, read by
/// `interp::show_text` and by the runtime written in the core. It was
/// enumerated from the `{:?}` of a str of the pinned toolchain, and on that
/// toolchain display of every code point, alone and between two letters,
/// is that `{:?}` byte for byte, so no trace changed when the set became
/// data. A toolchain whose tables differ fails here instead of changing a
/// display: moving the pin is a change of the specification (6.3 N7).
#[test]
fn display_escape_set_is_the_pinned_data() {
    let rs = rce_spec_check::interp::display_escaped();
    assert_eq!(rs.len(), 911);
    for w in rs.windows(2) {
        assert!(w[0].0 <= w[0].1 && w[0].1 < w[1].0, "ranges ascending and disjoint: {w:?}");
    }
    let d: serde_json::Value = serde_json::from_str(rce_spec_check::interp::DISPLAY_ESCAPE_JSON).unwrap();
    assert_eq!(d["format"], "rce-display-escape/1");
    assert_eq!(d["unicode"], "17.0.0");
    let mut differ = vec![];
    for cp in 0..0x110000u32 {
        let Some(c) = char::from_u32(cp) else { continue };
        for t in [c.to_string(), format!("a{c}b")] {
            let (got, want) = (rce_spec_check::interp::show_text(&t), format!("{t:?}"));
            if got != want && differ.len() < 10 {
                differ.push(format!("U+{cp:04X}: data {got}, toolchain {want}"));
            }
        }
    }
    assert!(differ.is_empty(), "display differs from the pinned toolchain's {{:?}}:\n{}", differ.join("\n"));
}
