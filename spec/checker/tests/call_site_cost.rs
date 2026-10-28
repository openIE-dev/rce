//! C.15: the declared version of the module that contains a call
//! expression governs the cost of a standard library method call. A sort
//! written in a 0.7 module is size-scaled (c_call + n*ceil(log2 n)) even
//! when a 0.5 or 0.6 module calls the function around it, and a sort
//! written in a 0.6 module costs c_call even when a 0.7 module calls it.
//! The budget boundary is the same in the reference evaluator and in the
//! Rust, Python and JavaScript emitters, on the linked program.
use rce_spec_check::{check_program, emit, link, parse::parse};
use std::process::Command;

fn lib(v: &str) -> String {
    format!(
        "# rce-version: {v}\nmodule costs.lib\n\npub fn sorted(n: i64) -> i64\n    var xs: List[i64] = [5, 3, 9, 1, 7, 2, 8, 4]\n    xs.sort()\n    return n\n"
    )
}

/// The calling module: a budget around the imported call, a check that
/// holds exactly when the budget does not trap. `v` "0.5" means no
/// declaration (a 0.5 module has no `with budget`, so the 0.5 case calls
/// through a 0.6 wrapper instead; see `caller_version_does_not_matter`).
fn main_src(v: &str, b: i128) -> String {
    format!(
        "# rce-version: {v}\nuse costs.lib\n\nfn go() -> i64\n    var r: i64 = 0\n    with budget {b}\n        r = lib.sorted(1)\n    return r\n\ncheck go() == 1\n"
    )
}

fn reference_ok(main: &str, libsrc: &str) -> bool {
    let files = vec![("main.rce".to_string(), main.to_string()), ("lib.rce".to_string(), libsrc.to_string())];
    let r = check_program(&files, false);
    assert!(r[1].1.is_ok(), "{:?}", r[1]);
    match &r[0].1 {
        Ok(()) => true,
        Err(e) => {
            assert!(e.contains("budget"), "unexpected kept outcome: {e}");
            false
        }
    }
}

/// smallest budget at which the check holds, by the reference evaluator
fn boundary(mainv: &str, libv: &str) -> i128 {
    let l = lib(libv);
    let (mut lo, mut hi) = (0i128, 2000i128);
    assert!(reference_ok(&main_src(mainv, hi), &l));
    while lo < hi {
        let mid = (lo + hi) / 2;
        if reference_ok(&main_src(mainv, mid), &l) { hi = mid } else { lo = mid + 1 }
    }
    lo
}

fn linked(main: &str, libsrc: &str) -> rce_spec_check::ast::Module {
    let m = parse(main).unwrap();
    let l = parse(libsrc).unwrap();
    let k = link::link(&m, &[&m, &l]);
    assert!(k.errs.is_empty(), "{:?}", k.errs);
    k.module
}

fn run_target(m: &rce_spec_check::ast::Module, target: &str, stem: &str) -> Option<String> {
    let lang = emit::Lang::parse(target).unwrap();
    let code = emit::emit(m, lang).unwrap();
    let (ext, tool) = match target {
        "python" => ("py", "python3"),
        "js" => ("mjs", "node"),
        _ => ("rs", "rustc"),
    };
    if Command::new(tool).arg("--version").output().is_err() {
        return None;
    }
    let d = std::env::temp_dir().join(format!("rce-csc-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(format!("{stem}.{ext}"));
    std::fs::write(&p, code).unwrap();
    let o = if target == "rust" {
        let exe = d.join(stem);
        let c = Command::new("rustc").args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&p).output().unwrap();
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        Command::new(&exe).output().unwrap()
    } else {
        Command::new(tool).arg(&p).output().unwrap()
    };
    Some(String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
}

#[test]
fn every_module_charges_the_size_scaled_method_cost() {
    // one language: the label of the caller or the callee changes nothing
    let b = boundary("0.6", "0.6");
    for (mv, lv) in [("0.6", "0.7"), ("0.7", "0.7"), ("0.7", "0.6")] {
        assert_eq!(boundary(mv, lv), b, "main {mv}, lib {lv}");
    }
}

#[test]
fn emitters_agree_on_the_mixed_boundary() {
    for (mv, lv) in [("0.6", "0.7"), ("0.7", "0.6"), ("0.7", "0.7"), ("0.6", "0.6")] {
        let b = boundary(mv, lv);
        let l = lib(lv);
        for (budget, holds) in [(b - 1, false), (b, true)] {
            let m = linked(&main_src(mv, budget), &l);
            for t in ["rust", "python", "js"] {
                let stem = format!("m{}{}_{budget}_{t}", &mv[2..], &lv[2..]);
                if let Some(out) = run_target(&m, t, &stem) {
                    assert_eq!(!out.contains("budget"), holds, "{t}: main {mv}, lib {lv}, budget {budget}: {out}");
                }
            }
        }
    }
}

#[test]
fn caller_version_does_not_matter_for_a_05_caller() {
    // A 0.5 module has no `with budget`; it calls a 0.6 wrapper that holds
    // the budget, and the wrapper calls the 0.7 sort. The cost is the 0.7 one.
    let wrap = |b: i128| format!(
        "# rce-version: 0.6\nmodule costs.wrap\nuse costs.lib\n\npub fn go() -> i64\n    var r: i64 = 0\n    with budget {b}\n        r = lib.sorted(1)\n    return r\n"
    );
    let main05 = "use costs.wrap\n\ncheck wrap.go() == 1\n";
    let ok = |b: i128| {
        let files = vec![
            ("main.rce".to_string(), main05.to_string()),
            ("wrap.rce".to_string(), wrap(b)),
            ("lib.rce".to_string(), lib("0.7")),
        ];
        let r = check_program(&files, false);
        r[0].1.is_ok()
    };
    let b67 = boundary("0.6", "0.7");
    assert!(ok(b67) && !ok(b67 - 1), "a 0.5 caller sees the 0.7 cost at the same boundary {b67}");
}
