//! W6 E1: the apps and RCEdb are byte-identical on every commit. For each
//! program, the reference trace (`rce trace --json` over its examples)
//! equals the committed golden in spec/traces/apps/, and every emitted
//! target available here (Rust, Python, JS; Scala when scala-cli is
//! present) prints exactly the reference observations. Regenerate the
//! goldens with `RCE_BLESS_TRACES=1 cargo test --release --test
//! golden_apps`, in a commit of their own that says why
//! (scripts/check-golden-commits.sh enforces this in CI).
//! The programs in crates/rce/examples/ are not RCE source (they are in
//! the earlier `spec` notation) and have no goldens.
use rce_spec_check::cli::trace;
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

const APPS: [&str; 3] = ["apps/terminal.rce", "apps/chat.rce", "rcedb/rcedb.rce"];

fn rce() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn stem(p: &str) -> String {
    Path::new(p).file_stem().unwrap().to_string_lossy().into_owned()
}

#[test]
fn app_traces_match_their_goldens() {
    let dir = rce().join("spec/traces/apps");
    let bless = std::env::var_os("RCE_BLESS_TRACES").is_some();
    if bless {
        std::fs::create_dir_all(&dir).unwrap();
    }
    for a in APPS {
        let p = rce().join(a);
        let t = trace(&p.to_string_lossy(), &std::fs::read_to_string(&p).unwrap(), None).unwrap() + "\n";
        let g = dir.join(format!("{}.json", stem(a)));
        if bless {
            std::fs::write(&g, &t).unwrap();
        }
        let want = std::fs::read_to_string(&g).unwrap_or_else(|e| panic!("{}: {e} (bless with RCE_BLESS_TRACES=1)", g.display()));
        assert!(t == want, "{a}: trace differs from {} (byte for byte)", g.display());
    }
}

fn run(lang: Lang, t: &str, dir: &Path, stem: &str, code: &str) -> Result<String, String> {
    let src = dir.join(format!("{stem}.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join(stem);
            let c = Command::new(t).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&src).output().map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(format!("rustc: {}", String::from_utf8_lossy(&c.stderr)));
            }
            Command::new(&exe).output()
        }
        Lang::Scala => Command::new(t)
            .args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
            .arg(&src)
            .output(),
        _ => Command::new(t).arg(&src).output(),
    }
    .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("exit {:?}: {}", out.status.code(), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}

#[test]
fn every_target_prints_the_reference_observations_for_the_apps() {
    let dir = std::env::temp_dir().join(format!("rce-golden-apps-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut checked = 0;
    std::thread::scope(|s| {
        let mut hs = vec![];
        for lang in targets::covered(&targets::ALL, targets::oracle) {
            let t = targets::oracle(lang);
            for a in APPS {
                let (dir, t) = (&dir, t.clone());
                hs.push(s.spawn(move || {
                    let m = with_probes(&parse(&std::fs::read_to_string(rce().join(a)).unwrap()).unwrap());
                    let want = observe(&m);
                    let code = emit(&m, lang).unwrap_or_else(|e| panic!("{a}: {lang:?} refused: {e}"));
                    let got = run(lang, &t, dir, &format!("{}_{:?}", stem(a), lang).to_lowercase(), &code).unwrap_or_else(|e| panic!("{a} {lang:?}: {e}"));
                    let got: Vec<&str> = got.lines().collect();
                    if let Some(i) = (0..want.len().max(got.len())).find(|&i| want.get(i).map(|s| s.as_str()) != got.get(i).copied()) {
                        panic!("{a} {lang:?}: line {}: want {:?}, got {:?}", i + 1, want.get(i), got.get(i));
                    }
                    assert!(!want.is_empty(), "{a}: no observations");
                }));
            }
        }
        for h in hs {
            h.join().unwrap();
            checked += 1;
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("{checked} (app, target) pairs agree with the reference");
    assert!(checked >= APPS.len(), "at least one target ran");
}
