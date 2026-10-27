//! 6.3 N7, Unicode pin: no text operation of the runtime reads the Unicode
//! tables of the toolchain that built it. Text is scalar values in UTF-8
//! (version-independent); case and class operations are ASCII; the
//! whitespace of `strip`, `words` and `toint` is the White_Space set of
//! Unicode 18.0.0, enumerated in `interp::WHITE_SPACE` and in the emitted
//! runtimes. A Rust or Python upgrade with new Unicode tables therefore
//! cannot change a result.
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
    // U+2009 and U+180E are content, displayed as themselves (10.5)
    assert!(out.contains("emit \"a\u{2009}b\u{180e}c\""), "{out}");
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
    assert!(bad.is_empty(), "calls that read the toolchain's Unicode tables:\n{}", bad.join("\n"));
}
