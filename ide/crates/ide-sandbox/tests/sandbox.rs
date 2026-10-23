#[cfg(target_os = "linux")]
use ide_sandbox::jail::platform_jail;
use ide_sandbox::jail::{Jail, JailError, JailPolicy, NoJail};
use ide_sandbox::wasi::{FuelLimit, Sandbox, guest_path};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const COUNTER: &str =
    include_str!("../../../../spec/examples/01_counter.rce");

/// Build crates/rce-guest for wasm32-wasip1 once, in its own target dir.
fn guest() -> &'static [u8] {
    static W: OnceLock<Vec<u8>> = OnceLock::new();
    W.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tdir = root.join("target/guest");
        let st = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .current_dir(&root)
            .args([
                "build",
                "-q",
                "-p",
                "rce-guest",
                "--release",
                "--target",
                "wasm32-wasip1",
                "--target-dir",
            ])
            .arg(&tdir)
            .status()
            .expect("cargo");
        assert!(st.success(), "guest build failed");
        std::fs::read(tdir.join("wasm32-wasip1/release/rce-guest.wasm")).unwrap()
    })
}

fn ws() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let w = t.path().join("ws");
    std::fs::create_dir_all(&w).unwrap();
    std::fs::write(w.join("counter.rce"), COUNTER).unwrap();
    std::fs::write(w.join("bad.rce"), "step Counter on Inc -> n + 1\n").unwrap();
    std::fs::write(t.path().join("secret.rce"), COUNTER).unwrap();
    (t, w)
}

const FUEL: FuelLimit = FuelLimit::Fuel(2_000_000_000);

#[test]
fn check_test_run_in_wasi() {
    let sb = Sandbox::new(guest()).unwrap();
    let (_t, w) = ws();
    let r = sb.check(&w, "counter.rce", FUEL).unwrap();
    assert_eq!(r.exit_code, Some(0), "{r:?}");
    assert!(r.stdout.starts_with("ok /ws/counter.rce"), "{r:?}");
    assert!(r.fuel_used > 0 && r.fuel_used <= r.fuel_limit);
    assert_eq!(r.features().wasi_fuel, r.fuel_used);
    let r = sb.test(&w, "counter.rce", FUEL).unwrap();
    assert!(r.stdout.contains("(2 examples, 0 checks)"), "{r:?}");
    let r = sb.check(&w, "bad.rce", FUEL).unwrap();
    assert_eq!(r.exit_code, Some(1), "{r:?}");
    let r = sb
        .run_program(&w, "counter.rce", Some("Inc, Inc, Dec, Dec, Dec"), FUEL)
        .unwrap();
    assert_eq!(r.exit_code, Some(1)); // the third Dec is refused: Underflow
    assert!(r.stdout.contains("refused"), "{}", r.stdout);
    // Same program, same input: same fuel (deterministic metering).
    let a = sb.check(&w, "counter.rce", FUEL).unwrap().fuel_used;
    let b = sb.check(&w, "counter.rce", FUEL).unwrap().fuel_used;
    assert_eq!(a, b);
}

#[test]
fn fuel_runs_out() {
    let sb = Sandbox::new(guest()).unwrap();
    let (_t, w) = ws();
    let r = sb
        .check(&w, "counter.rce", FuelLimit::Fuel(10_000))
        .unwrap();
    assert!(r.out_of_fuel, "{r:?}");
    assert_eq!(r.exit_code, None);
    // wasmi charges a block before running it; a block it cannot pay for is not run.
    assert!(r.fuel_used <= 10_000);
}

#[test]
fn fuel_from_joules() {
    assert_eq!(
        FuelLimit::Joules {
            remaining_j: 2.0,
            joules_per_fuel: 0.25
        }
        .fuel()
        .unwrap(),
        8
    );
    assert!(
        FuelLimit::Joules {
            remaining_j: 0.0,
            joules_per_fuel: 1e-9
        }
        .fuel()
        .is_err()
    );
    assert!(
        FuelLimit::Joules {
            remaining_j: 1.0,
            joules_per_fuel: 0.0
        }
        .fuel()
        .is_err()
    );
    assert!(FuelLimit::Fuel(0).fuel().is_err());
}

#[test]
fn only_the_workspace_is_visible() {
    let sb = Sandbox::new(guest()).unwrap();
    let (_t, w) = ws();
    assert!(guest_path("../secret.rce").is_err());
    assert!(guest_path("/etc/passwd").is_err());
    for p in ["/ws/../secret.rce", "/etc/passwd", "secret.rce"] {
        let r = sb.run(&w, &["check", p], FUEL).unwrap();
        assert_eq!(r.exit_code, Some(2), "{p}: {r:?}");
    }
    // The preopen is read-only and the guest has no environment.
    assert!(!w.join("counter.rce.tmp").exists());
}

#[test]
fn no_jail_refuses() {
    let p = JailPolicy::default();
    assert!(matches!(
        NoJail.run(Command::new("true"), &p),
        Err(JailError::Unavailable(_))
    ));
}

#[cfg(target_os = "linux")]
#[test]
fn linux_jail_enforces_or_fails_closed() {
    let t = tempfile::tempdir().unwrap();
    let inside = t.path().join("in");
    let outside = t.path().join("out");
    std::fs::create_dir_all(&inside).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("s"), "secret").unwrap();
    let pol = JailPolicy {
        rw: vec![inside.clone()],
        ro: ["/usr", "/bin", "/lib", "/lib64", "/etc"]
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .collect(),
        allow_net: false,
        tcp_connect: vec![],
    };
    let j = platform_jail();
    match j.probe(&pol) {
        Err(e) => {
            eprintln!("jail unavailable here: {e}");
            assert!(
                j.run(Command::new("true"), &pol).is_err(),
                "must fail closed"
            );
        }
        Ok(()) => {
            let sh = |s: &str| {
                let mut c = Command::new("/bin/sh");
                c.args(["-c", s]);
                j.run(c, &pol).unwrap()
            };
            let w = sh(&format!(
                "echo hi > {}/f && cat {}/f",
                inside.display(),
                inside.display()
            ));
            assert!(w.status.success(), "{w:?}");
            assert_eq!(String::from_utf8_lossy(&w.stdout), "hi\n");
            let r = sh(&format!("cat {}/s", outside.display()));
            assert!(!r.status.success(), "read outside the policy must fail");
            let r = sh(&format!("echo x > {}/s2", outside.display()));
            assert!(!r.status.success() && !outside.join("s2").exists());
            // Network: python/bash may be absent; use the shell's /dev/tcp only if bash exists.
            if Path::new("/bin/bash").exists() {
                let mut c = Command::new("/bin/bash");
                c.args(["-c", "exec 3<>/dev/tcp/127.0.0.1/9"]);
                let o = j.run(c, &pol).unwrap();
                assert!(!o.status.success());
                let err = String::from_utf8_lossy(&o.stderr);
                assert!(
                    err.contains("not permitted") || err.contains("denied"),
                    "{err}"
                );
            }
        }
    }
}
