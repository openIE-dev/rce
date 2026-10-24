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

#[test]
fn a_generated_manifest_is_accepted_and_pins_every_module() {
    let d = tmp("gen");
    std::fs::create_dir_all(d.join("sub")).unwrap();
    std::fs::write(d.join("lib.rce"), LIB).unwrap();
    std::fs::write(d.join("sub/app.rce"), APP).unwrap();
    let text = manifest::generate(&d).unwrap();
    assert_eq!(text, canonical(&serde_json::from_str::<serde_json::Value>(&text).unwrap()));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["modules"][0]["module"], "app");
    assert_eq!(v["modules"][0]["path"], "sub/app.rce");
    assert_eq!(v["modules"][1]["module"], "lib");
    assert_eq!(v["modules"][1]["sha256"], manifest::module_hash(LIB).unwrap());
    let pinned = manifest::check(&text, &d).unwrap();
    assert_eq!(pinned.len(), 2);
}

const USES_LIB: &str = "module app\n\nuse enc\n\nfn f() -> text\n    let b = [104, 105]\n    return enc.b64_encode(&b, false, true)\n";

#[test]
fn the_built_in_library_is_the_pinned_library() {
    // every embedded module hashes to its entry in stdlib/lib/MANIFEST.json
    // and the manifest lists nothing the checker does not carry (XF-3)
    assert_eq!(manifest::stdlib_unpinned(), Vec::<String>::new());
    assert_eq!(manifest::stdlib_hash().len(), 64);
}

#[test]
fn a_program_that_uses_the_library_pins_it_as_the_dependency_stdlib() {
    let d = tmp("stdlib-dep");
    std::fs::write(d.join("app.rce"), USES_LIB).unwrap();
    let text = manifest::generate(&d).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["dependencies"], json!([{"name": "stdlib", "sha256": manifest::stdlib_hash()}]));
    assert!(manifest::check(&text, &d).is_ok(), "{:?}", manifest::check(&text, &d));
    // without the dependency the import is unpinned
    let mut bare = v.clone();
    bare["dependencies"] = json!([]);
    let errs = manifest::check(&canonical(&bare), &d).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("imports `enc`, which the manifest does not pin")), "{errs:?}");
}

#[test]
fn a_wrong_library_hash_and_a_dependency_named_after_a_library_module_are_refused() {
    let d = tmp("stdlib-wrong");
    std::fs::write(d.join("app.rce"), USES_LIB).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&manifest::generate(&d).unwrap()).unwrap();
    v["dependencies"] = json!([{"name": "stdlib", "sha256": "0".repeat(64)}]);
    let errs = manifest::check(&canonical(&v), &d).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("dependency `stdlib` pins") && e.contains("the built-in library of this checker is")), "{errs:?}");
    // a hash that only has the right format no longer passes for a library module
    v["dependencies"] = json!([{"name": "enc", "sha256": "0".repeat(64)}]);
    let errs = manifest::check(&canonical(&v), &d).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("dependency `enc` names a module of the built-in library")), "{errs:?}");
}

#[test]
fn a_lib_directive_is_a_use_of_the_library_and_is_pinned_the_same_way() {
    let d = tmp("stdlib-directive");
    std::fs::write(d.join("app.rce"), "module app\n#!lib crypto\n\nfn f() -> List[i64]\n    return sha256([104, 105])\n").unwrap();
    let text = manifest::generate(&d).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["dependencies"], json!([{"name": "stdlib", "sha256": manifest::stdlib_hash()}]));
    assert!(manifest::check(&text, &d).is_ok());
    let mut bare = v.clone();
    bare["dependencies"] = json!([]);
    let errs = manifest::check(&canonical(&bare), &d).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("uses the built-in `crypto` through `#!lib crypto`, which the manifest does not pin")), "{errs:?}");
}

#[test]
fn a_lib_directive_is_never_an_import_edge_to_a_file_of_the_directory() {
    // a local module named crypto, unrelated to the library, and a module
    // that names the built-in crypto by directive: the directive still needs
    // the stdlib pin, and no import cycle is reported through the local file
    let d = tmp("stdlib-directive-local");
    std::fs::write(d.join("crypto.rce"), "module crypto\n#!lib crypto\n\npub fn mine() -> i64 = 1\n").unwrap();
    std::fs::write(d.join("app.rce"), "module app\n#!lib crypto\n\nfn f() -> List[i64]\n    return sha256([104, 105])\n").unwrap();
    let text = manifest::generate(&d).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["dependencies"], json!([{"name": "stdlib", "sha256": manifest::stdlib_hash()}]));
    let r = manifest::check(&text, &d);
    assert!(r.as_ref().map_or_else(|e| !e.iter().any(|x| x.contains("cycle")), |_| true), "{r:?}");
}

#[test]
fn the_library_hash_is_over_the_canonical_json_of_its_manifest() {
    let t = rce_spec_check::stdlib::manifest_text();
    let v: serde_json::Value = serde_json::from_str(t).unwrap();
    assert_eq!(manifest::stdlib_hash(), manifest::sha256_hex(canonical(&v).as_bytes()));
    // the file itself is canonical, so the hash is also the hash of its text
    assert_eq!(canonical(&v), t);
}

#[test]
fn a_dependency_named_after_a_library_namespace_is_refused_with_its_reason() {
    let d = tmp("stdlib-namespace");
    std::fs::write(d.join("app.rce"), USES_LIB).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&manifest::generate(&d).unwrap()).unwrap();
    let deps = v["dependencies"].as_array().unwrap().clone();
    let mut with_ns = deps.clone();
    with_ns.push(json!({"name": "data", "sha256": "1".repeat(64)}));
    v["dependencies"] = serde_json::Value::Array(with_ns);
    let errs = manifest::check(&canonical(&v), &d).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("dependency `data` names a namespace of the built-in library")), "{errs:?}");
}
