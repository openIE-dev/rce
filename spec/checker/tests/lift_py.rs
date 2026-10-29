//! The Python lift (spec/lift/python): each committed program lifted by
//! lift_py.py from a Python fixture is accepted by `rce check`, and
//! `rce run` writes exactly the standard output that CPython 3.13.16
//! wrote for the source (fixtures/pos/NAME.stdout, recorded from the
//! oracle, never from the lift). spec/lift/python/test_lift_py.py checks,
//! with Python, that the lift still gives these files byte for byte and
//! that the negative fixtures require change with their rules.
use rce_spec_check::cli;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

/// The text of a string literal as `rce run` shows an emitted value.
fn decode(lit: &str) -> String {
    let s = lit.strip_prefix('"').and_then(|s| s.strip_suffix('"')).expect("an emitted text");
    let (mut o, mut it) = (String::new(), s.chars().peekable());
    while let Some(c) = it.next() {
        if c != '\\' {
            o.push(c);
            continue;
        }
        match it.next().unwrap() {
            '0' => o.push('\0'),
            't' => o.push('\t'),
            'n' => o.push('\n'),
            'r' => o.push('\r'),
            '"' => o.push('"'),
            '\\' => o.push('\\'),
            'u' => {
                assert_eq!(it.next(), Some('{'));
                let h: String = it.by_ref().take_while(|&c| c != '}').collect();
                o.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap());
            }
            x => panic!("unknown escape \\{x}"),
        }
    }
    o
}

#[test]
fn lifted_python_fixtures_print_what_cpython_printed() {
    for name in ["ints", "lists", "text"] {
        let lifted = format!("../lift/python/fixtures/pos/{name}.lifted.rce");
        let (code, out) = rce(&["check", &lifted]);
        assert_eq!(code, 0, "{name}: {out}");
        let (code, out) = rce(&["run", &lifted]);
        assert_eq!(code, 0, "{name}: {out}");
        let mut stdout = String::new();
        for line in out.lines() {
            let lit = line.strip_prefix("emit ").unwrap_or_else(|| panic!("{name}: {line}"));
            stdout.push_str(&decode(lit));
        }
        let want = std::fs::read(format!("../lift/python/fixtures/pos/{name}.stdout")).unwrap();
        assert_eq!(stdout.as_bytes(), &want[..], "{name}: the lifted program's standard output is CPython's");
    }
}
