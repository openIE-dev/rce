//! Hash-pinned builds (F.5.3, S5): W-PIN-1 and signed advisories.
use rce_spec_check::manifest::{self, canonical};
use ring::signature::{Ed25519KeyPair, KeyPair};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-manifest-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const LIB: &str = "# rce-version: 0.9\nmodule lib\n\npub fn one() -> i64\n    return 1\n";
const APP: &str = "# rce-version: 0.9\nmodule app\n\nuse lib\n\nfn main()\n    emit lib.one()\n";

fn manifest_for(d: &std::path::Path, extra: serde_json::Value) -> String {
    std::fs::write(d.join("lib.rce"), LIB).unwrap();
    std::fs::write(d.join("app.rce"), APP).unwrap();
    let mut m = json!({
        "dependencies": [],
        "formatter": "rce-fmt 1",
        "language": "0.9",
        "modules": [
            {"module": "app", "path": "app.rce", "sha256": manifest::module_hash(APP).unwrap()},
            {"module": "lib", "path": "lib.rce", "sha256": manifest::module_hash(LIB).unwrap()}
        ]
    });
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        m[k] = v;
    }
    canonical(&m)
}

#[test]
fn a_pinned_manifest_is_accepted() {
    let d = tmp("ok");
    let t = manifest_for(&d, json!({}));
    let pinned = manifest::check(&t, &d).unwrap();
    assert_eq!(pinned.len(), 2);
}

#[test]
fn a_module_whose_hash_differs_is_refused_and_formatting_never_changes_a_hash() {
    let d = tmp("hash");
    let t = manifest_for(&d, json!({}));
    // reformatting (extra blank lines, a comment) does not change the hash
    std::fs::write(d.join("lib.rce"), LIB.replace("pub fn", "# note\n\n\npub fn")).unwrap();
    assert!(manifest::check(&t, &d).is_ok());
    // a change of meaning does
    std::fs::write(d.join("lib.rce"), LIB.replace("return 1", "return 2")).unwrap();
    let e = manifest::check(&t, &d).unwrap_err().join("; ");
    assert!(e.contains("W-PIN-1") && e.contains("the manifest pins"), "{e}");
}

#[test]
fn unknown_keys_globs_relative_paths_and_unpinned_dependencies_are_refused() {
    let d = tmp("bad");
    let good = manifest_for(&d, json!({}));
    for (why, extra) in [
        ("unknown key", json!({"features": ["x"]})),
        ("glob", json!({"modules": [{"module": "app", "path": "*.rce", "sha256": "0".repeat(64)}]})),
        ("relative path", json!({"modules": [{"module": "app", "path": "../app.rce", "sha256": "0".repeat(64)}]})),
        ("unpinned dependency", json!({"dependencies": [{"name": "x", "sha256": "latest"}]})),
        ("wrong language", json!({"language": "0.8"})),
    ] {
        let t = manifest_for(&d, extra);
        let e = manifest::check(&t, &d).unwrap_err().join("; ");
        assert!(e.contains("W-PIN-1"), "{why}: {e}");
    }
    // not canonical: whitespace
    let pretty = serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&good).unwrap()).unwrap();
    assert!(manifest::check(&pretty, &d).unwrap_err().join(";").contains("canonical"));
    // an import that the manifest does not pin
    std::fs::write(d.join("app.rce"), APP.replace("use lib", "use lib\nuse other")).unwrap();
    let t = good.replace(&manifest::module_hash(APP).unwrap(), &manifest::module_hash(&APP.replace("use lib", "use lib\nuse other")).unwrap());
    let e = manifest::check(&t, &d).unwrap_err().join("; ");
    assert!(e.contains("imports `other`"), "{e}");
}

#[test]
fn an_import_cycle_is_refused() {
    let d = tmp("cycle");
    let a = "# rce-version: 0.9\nmodule a\n\nuse b\n\npub fn f() -> i64\n    return 1\n";
    let b = "# rce-version: 0.9\nmodule b\n\nuse a\n\npub fn g() -> i64\n    return 2\n";
    std::fs::write(d.join("a.rce"), a).unwrap();
    std::fs::write(d.join("b.rce"), b).unwrap();
    let t = canonical(&json!({
        "dependencies": [], "formatter": "rce-fmt 1", "language": "0.9",
        "modules": [
            {"module": "a", "path": "a.rce", "sha256": manifest::module_hash(a).unwrap()},
            {"module": "b", "path": "b.rce", "sha256": manifest::module_hash(b).unwrap()}
        ]
    }));
    let e = manifest::check(&t, &d).unwrap_err().join("; ");
    assert!(e.contains("import cycle"), "{e}");
}

fn signed(seed: u8, adv: serde_json::Value) -> (serde_json::Value, [u8; 32]) {
    let kp = Ed25519KeyPair::from_seed_unchecked(&[seed; 32]).unwrap();
    let digest = {
        use sha2::{Digest, Sha256};
        Sha256::digest(canonical(&adv).as_bytes())
    };
    let sig: String = kp.sign(&digest).as_ref().iter().map(|b| format!("{b:02x}")).collect();
    let mut pk = [0u8; 32];
    pk.copy_from_slice(kp.public_key().as_ref());
    (json!({"advisory": adv, "signature": sig}), pk)
}

#[test]
fn signed_advisories_verify_and_mark_pinned_hashes_unsafe() {
    let d = tmp("adv");
    let t = manifest_for(&d, json!({}));
    let pinned = manifest::check(&t, &d).unwrap();
    let h = pinned["lib"].clone();
    let (a1, pk) = signed(7, json!({"kind": "advisory", "hash": h, "reason": "wrong result", "issued_by": "maint", "seq": 1}));
    let keys: BTreeMap<String, [u8; 32]> = [("maint".to_string(), pk)].into();
    let adv = manifest::verify_advisories(&canonical(&json!([a1.clone()])), &keys).unwrap();
    let u = manifest::unsafe_list(&t, &pinned, &adv);
    assert_eq!(u.len(), 1);
    assert!(u[0].contains("module lib") && u[0].contains("wrong result"), "{u:?}");
    // the advisory does not change what the hash means: the manifest still checks
    assert!(manifest::check(&t, &d).is_ok());
    // a tampered reason, an untrusted issuer and a replayed seq are refused
    let mut bad = a1.clone();
    bad["advisory"]["reason"] = json!("fine actually");
    assert!(manifest::verify_advisories(&canonical(&json!([bad])), &keys).unwrap_err().join(";").contains("bad signature"));
    assert!(manifest::verify_advisories(&canonical(&json!([a1.clone()])), &BTreeMap::new()).unwrap_err().join(";").contains("not trusted"));
    let e = manifest::verify_advisories(&canonical(&json!([a1.clone(), a1])), &keys).unwrap_err().join(";");
    assert!(e.contains("does not increase"), "{e}");
}

#[test]
fn the_cli_checks_a_manifest() {
    let d = tmp("cli");
    let t = manifest_for(&d, json!({}));
    std::fs::write(d.join("rce.manifest.json"), &t).unwrap();
    let mut out = vec![];
    let mut err = vec![];
    let code = rce_spec_check::cli::main(&["manifest-check".into(), d.join("rce.manifest.json").to_string_lossy().into_owned()], &mut &b""[..], &mut out, &mut err);
    let o = String::from_utf8(out).unwrap();
    assert_eq!(code, 0, "{o}");
    assert!(o.contains("2 modules pinned, 0 unsafe"), "{o}");
}
