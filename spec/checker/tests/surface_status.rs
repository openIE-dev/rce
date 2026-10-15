//! SURFACE.md is informative (spec 0, document status): it says so, and
//! every RCE program in it is accepted by the reference checker and its
//! examples pass, so it cannot drift from the normative language.
use rce_spec_check::cli;

fn rce(args: &[String]) -> (i32, String) {
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(args, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

#[test]
fn surface_is_informative_and_its_programs_check() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let s = std::fs::read_to_string(root.join("SURFACE.md")).unwrap();
    assert!(s.contains("**Status: informative.**"));
    let blocks: Vec<&str> = s.split("```").skip(1).step_by(2).collect();
    let progs: Vec<&str> = blocks.into_iter().filter(|b| b.contains("\nstate ") || b.contains("\nstep ")).collect();
    assert!(!progs.is_empty());
    let d = std::env::temp_dir().join(format!("rce-surface-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    for (i, p) in progs.iter().enumerate() {
        let f = d.join(format!("s{i}.rce"));
        std::fs::write(&f, p.trim_start_matches('\n')).unwrap();
        let f = f.to_string_lossy().into_owned();
        for cmd in ["check", "test"] {
            let (code, out) = rce(&[cmd.into(), f.clone()]);
            assert_eq!(code, 0, "SURFACE.md program {i}: rce {cmd}: {out}");
        }
    }
}
