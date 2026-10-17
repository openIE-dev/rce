//! Version 0.8 indexed values (spec 5.16; CORE-PROPERTIES.md P1; Lean
//! RCE/Label.lean): the encoders agree across the reference evaluator and
//! the Rust, Python and JavaScript runtimes; W-CTX-1, W-SEC-1, W-CAP-1,
//! W-CAP-2 and the reserved names refuse; and the gate: none of it exists
//! in a 0.5, 0.6 or 0.7 module, whose meaning is unchanged.
use rce_spec_check::cli;
use std::process::Command;

const ENC: &str = "# rce-version: 0.8
fn main()
    let s = \"<a href='x'>&\\\"é\\n\\t\\\\ q/?=</a>\"
    emit (Html.lit(\"<p>\") + Html.encode(s) + Html.lit(\"</p>\")).raw()
    emit (Url.lit(\"/s?q=\") + Url.encode(s)).raw()
    emit (Sql.lit(\"WHERE n = \") + Sql.encode(s)).raw()
    emit (Json.lit(\"{\\\"v\\\": \") + Json.encode(s) + Json.lit(\"}\")).raw()
";

fn dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rce-indexed-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn run(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let c = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (c, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}
fn emits(s: &str) -> Vec<String> {
    s.lines().filter(|l| l.contains("emit")).map(|l| l.split_once("emit").unwrap().1.trim().to_string()).collect()
}
#[allow(dead_code)]
fn want() -> Vec<String> {
    let s = "<a href='x'>&\"é\n\t\\ q/?=</a>";
    use rce_spec_check::index08::encode;
    vec![
        format!("<p>{}</p>", encode("Html", s)),
        format!("/s?q={}", encode("Url", s)),
        format!("WHERE n = {}", encode("Sql", s)),
        format!("{{\"v\": {}}}", encode("Json", s)),
    ]
}

#[test]
fn encoders_are_the_spec_tables() {
    use rce_spec_check::index08::encode;
    assert_eq!(encode("Html", "<a b=\"c\">'&"), "&lt;a b=&quot;c&quot;&gt;&#39;&amp;");
    assert_eq!(encode("Url", "a b/é~"), "a%20b%2F%C3%A9~");
    assert_eq!(encode("Sql", "O'Brien"), "'O''Brien'");
    assert_eq!(encode("Json", "a\"\\\n\u{1}"), "\"a\\\"\\\\\\n\\u0001\"");
    // no structural metacharacter of a context survives its encoder (Label.lean)
    for c in ['<', '>', '"', '\''] {
        assert!(!encode("Html", "x<>\"'y").contains(c));
    }
}

#[test]
fn reference_and_every_runtime_agree_on_the_encoders() {
    let d = dir();
    let f = d.join("enc.rce");
    std::fs::write(&f, ENC).unwrap();
    let (c, o, e) = run(&["run", f.to_str().unwrap()]);
    assert_eq!(c, 0, "{o}{e}");
    let reference = emits(&o);
    assert_eq!(reference.len(), 4, "{o}");
    for (target, ext, tool) in [("rust", "rs", "rustc"), ("python", "py", "python3"), ("js", "mjs", "node")] {
        if Command::new(tool).arg("--version").output().is_err() {
            continue;
        }
        let (c, out, err) = run(&["emit", target, f.to_str().unwrap()]);
        assert_eq!(c, 0, "{err}");
        let p = d.join(format!("i_{target}.{ext}"));
        std::fs::write(&p, out).unwrap();
        let o = if target == "rust" {
            let exe = d.join("i_rust");
            let c = Command::new("rustc").args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&p).output().unwrap();
            assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
            Command::new(&exe).arg("--run").output().unwrap()
        } else {
            Command::new(tool).arg(&p).arg("--run").output().unwrap()
        };
        let s = String::from_utf8_lossy(&o.stdout).into_owned();
        assert_eq!(emits(&s), reference, "{target}: {s}{}", String::from_utf8_lossy(&o.stderr));
    }
}

fn refuses(src: &str, rule: &str) {
    let f = dir().join("bad.rce");
    std::fs::write(&f, format!("# rce-version: 0.8\n{src}")).unwrap();
    let (c, o, e) = run(&["check", f.to_str().unwrap()]);
    assert_ne!(c, 0, "accepted:\n{src}");
    assert!(format!("{o}{e}").contains(rule), "{rule} expected for\n{src}\n{o}{e}");
}

#[test]
fn injection_secrets_and_missing_authority_are_refused() {
    let sql = "state P\n    q: Text[Sql] = Sql.lit(\"\")\nevent E(b: text)\n";
    refuses(&format!("{sql}step P on E(b: x) -> Sql.lit(\"x = \") + x\n"), "W-CTX-1");
    refuses(&format!("{sql}step P on E(b: x) -> Sql.lit(x)\n"), "W-CTX-1");
    refuses(&format!("{sql}step P on E(b: x) -> Html.encode(x)\n"), "W-TY-1");
    refuses("state P\n    q: Text[Html] = Html.lit(\"\")\nevent E(b: Text[Html])\nstep P on E(b: x) -> x\n", "W-CTX-1");
    refuses("state P\n    q: Text[Css] = Html.lit(\"\")\nevent E(b: text)\nstep P on E(b: _) -> q\n", "W-CTX-1");
    let sec = "state V\n    n: u32 = 0\nevent E(p: Secret[text])\n";
    refuses(&format!("{sec}step V on E(p: p)\n    emit p\n"), "W-SEC-1");
    refuses(&format!("{sec}step V on E(p: p)\n    if p == \"hunter2\"\n        n += 1\n"), "W-TY-1");
    refuses(&format!("{sec}step V on E(p: p)\n    n = p.len()\n"), "W-SEC-1");
    refuses("struct S\n    p: Secret[text]?\nstate V\n    n: u32 = 0\nevent E(p: Secret[text])\nstep V on E(p: p)\n    emit S(p: some(p))\n", "W-SEC-1");
    refuses("error Leak(p: Secret[text])\nstate V\n    n: u32 = 0\nevent E\nstep V on E -> n\n", "W-SEC-1");
    let cap = "struct Root\nstate P\n    limit: Guarded[u32, Root] = 1\n";
    refuses(&format!("{cap}event E(n: u32)\nstep P on E(n: n) -> n\n"), "W-CAP-2");
    refuses(&format!("{cap}event E(c: Cap[Root], n: u32)\nevent F(n: u32)\nstep P on E(c: _, n: n) | F(n: n) -> n\n"), "W-CAP-2");
    refuses("struct Root\nstate P\n    c: Cap[Root]? = none\nevent E\nstep P on E -> none\n", "W-CAP-1");
    refuses("state P\n    c: Cap[Nope]? = none\nevent E\nstep P on E -> none\n", "W-CAP-1");
    refuses("struct Html\n    a: u32\nfn main()\n    emit 1\n", "W-NAME-2");
}

#[test]
fn guarded_reads_and_capability_writes_are_accepted() {
    let f = dir().join("ok.rce");
    std::fs::write(&f, "# rce-version: 0.8\nstruct Root\nstate P\n    limit: Guarded[u8, Root] = 1\n    n: u32 = 0\nevent E(k: u32)\nevent G(c: Cap[Root], v: u8)\nstep P on E(k: k)\n    n = n + k\nstep P on G(c: _, v: v)\n    limit = v\n").unwrap();
    let (c, o, e) = run(&["run", f.to_str().unwrap(), "G(c: (), v: 7), E(k: 2)"]);
    assert_eq!(c, 0, "{o}{e}");
    assert!(o.contains("limit = 7") && o.contains("n = 2"), "{o}");
    // Guarded keeps its base range (6.3): 300 does not fit u8
    let (c, o, _) = run(&["run", f.to_str().unwrap(), "G(c: (), v: 300)"]);
    assert_ne!(c, 0, "{o}");
}

#[test]
fn html_is_one_reserved_type_in_every_module() {
    // 5.16 is in the language: `Html` cannot be declared, and `Html.lit` is read, whatever the label
    for v in ["", "# rce-version: 0.6\n", "# rce-version: 0.7\n"] {
        let f = dir().join("old.rce");
        std::fs::write(&f, format!("{v}struct Html\n    b: u32\nfn main()\n    emit 1\n")).unwrap();
        assert_ne!(run(&["check", f.to_str().unwrap()]).0, 0, "{v}");
        std::fs::write(&f, format!("{v}fn main()\n    emit Html.lit(\"x\").raw()\n")).unwrap();
        let (c, _, e) = run(&["check", f.to_str().unwrap()]);
        assert_eq!(c, 0, "{v}: {e}");
    }
}
