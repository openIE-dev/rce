//! Hash-pinned builds (spec F.5.3, S5): the manifest, W-PIN-1, and signed
//! advisories. A build is a function of its pinned inputs: the manifest is the
//! whole answer, there is no resolver.
//!
//! Manifest (canonical JSON, 6.1: keys sorted, no insignificant whitespace, no
//! trailing newline, one final line feed is ignored; a file that is not canonical is refused, so
//! a manifest has one spelling):
//!
//! ```text
//! {"dependencies":[{"name":"stdlib","sha256":"<64 hex>"}],
//!  "formatter":"rce-fmt 1",
//!  "language":"0.9",
//!  "modules":[{"module":"app","path":"app.rce","sha256":"<64 hex>"}],
//!  "reads":[{"path":"data/table.txt","sha256":"<64 hex>"}]}
//! ```
//!
//! `reads` is optional (compile-time file reads are pinned imports). The hash
//! of a module is the SHA-256 of its canonical unparse (`rce hash` text of the
//! module, headers included), so formatting never changes it. Nothing here
//! proves T54; it assumes SHA-256 is collision resistant.
use crate::ast::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Canonical JSON text of a value (keys sorted by the map, no whitespace).
pub fn canonical(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// The hash a manifest pins for a module: SHA-256 of its canonical unparse.
pub fn module_hash(src: &str) -> Result<String, String> {
    let m = crate::parse::parse_surface(src)?;
    Ok(sha256_hex(crate::unparse::unparse(&m).as_bytes()))
}

/// The hash by which a program pins the built-in library (4.1, F.5.3): the
/// SHA-256 of the canonical text of the library's own manifest, which pins
/// every module of the library by its hash. A manifest names it as the
/// dependency `stdlib`.
pub fn stdlib_hash() -> String {
    sha256_hex(crate::stdlib::manifest_text().as_bytes())
}

/// The embedded library against its embedded manifest: every module listed
/// with the hash of its text, and nothing else (XF-3). Empty when the library
/// this checker carries is exactly the pinned library.
pub fn stdlib_unpinned() -> Vec<String> {
    let mut errs = vec![];
    let v: Value = match serde_json::from_str(crate::stdlib::manifest_text()) {
        Ok(v) => v,
        Err(e) => return vec![format!("W-PIN-1 the built-in library's manifest is not JSON: {e}")],
    };
    let mut listed = BTreeSet::new();
    for e in v["modules"].as_array().into_iter().flatten() {
        let (name, want) = (e["module"].as_str().unwrap_or(""), e["sha256"].as_str().unwrap_or(""));
        listed.insert(name.to_string());
        match crate::stdlib::source(name).map(module_hash) {
            None => pin(&mut errs, format!("the built-in library's manifest lists `{name}`, which the library does not carry")),
            Some(Err(x)) => pin(&mut errs, format!("built-in module `{name}` does not parse: {x}")),
            Some(Ok(h)) if h != want => pin(&mut errs, format!("built-in module `{name}` hashes to {h}, its manifest pins {want}")),
            Some(Ok(_)) => {}
        }
    }
    for n in crate::stdlib::names() {
        if !listed.contains(n) {
            pin(&mut errs, format!("built-in module `{n}` is not in the library's manifest"));
        }
    }
    errs
}

fn bad_path(p: &str) -> Option<&'static str> {
    if p.is_empty() {
        Some("an empty path")
    } else if p.contains(['*', '?', '[', ']', '{', '}']) {
        Some("a glob")
    } else if p.starts_with('/') || p.contains('\\') || p.split('/').any(|s| s == ".." || s == "." || s.is_empty()) {
        Some("an absolute or relative path (a path is inside the manifest's directory, with no `.` or `..` segment)")
    } else {
        None
    }
}

fn pin(e: &mut Vec<String>, m: String) {
    e.push(format!("W-PIN-1 {m}"));
}

fn entries<'a>(v: &'a Value, key: &str, fields: &[&str], errs: &mut Vec<String>) -> Vec<&'a serde_json::Map<String, Value>> {
    let mut out = vec![];
    match v.get(key) {
        None => {}
        Some(Value::Array(a)) => {
            for x in a {
                match x.as_object() {
                    Some(o) => {
                        for k in o.keys() {
                            if !fields.contains(&k.as_str()) {
                                pin(errs, format!("`{key}` entry has the unknown key `{k}`: unknown keys, features and build flags are refused"));
                            }
                        }
                        for f in fields {
                            if !o.get(*f).is_some_and(|x| x.is_string()) {
                                pin(errs, format!("`{key}` entry has no string `{f}`"));
                            }
                        }
                        out.push(o);
                    }
                    None => pin(errs, format!("`{key}` holds an entry that is not an object")),
                }
            }
        }
        Some(_) => pin(errs, format!("`{key}` is not a list")),
    }
    out
}

/// The imports of a module source: `use a.b` names the module `a.b`.
fn imports(src: &str) -> Vec<String> {
    match crate::parse::parse_surface(src) {
        Ok(m) => m
            .items
            .iter()
            .filter_map(|i| if let Item::Use(p, _, _) = i { Some(p.join(".")) } else { None })
            .collect(),
        Err(_) => vec![],
    }
}

/// W-PIN-1 over a manifest and the files it names (relative to `root`).
/// `Err` is the list of findings; `Ok` is the pinned module hashes.
/// Write the manifest of a directory (F.5.3): every `.rce` file under `root`,
/// recursively, in sorted order, as a module pinned by its hash; the output
/// is canonical JSON and `check` accepts it at `root/MANIFEST.json`. Imports
/// between the listed modules need no dependency entry; an import of the
/// built-in library (4.1) adds the one dependency `stdlib`, pinned by the
/// hash of the library's manifest. Any other import of a module outside the
/// directory is reported by `check`, never guessed here (there is no resolver).
pub fn generate(root: &Path) -> Result<String, String> {
    let mut files = vec![];
    fn walk(d: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
        let mut es: Vec<_> = std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        es.sort();
        for p in es {
            if p.is_dir() {
                walk(&p, out)?;
            } else if p.extension().is_some_and(|x| x == "rce") {
                out.push(p);
            }
        }
        Ok(())
    }
    walk(root, &mut files)?;
    let mut mods = vec![];
    let mut imps = vec![];
    for f in &files {
        let src = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let m = crate::parse::parse(&src).map_err(|e| format!("{}: {e}", f.display()))?;
        let rel = f.strip_prefix(root).map_err(|e| e.to_string())?.to_string_lossy().replace('\\', "/");
        let name = m.path.as_ref().map(|p| p.join(".")).unwrap_or_else(|| rel.trim_end_matches(".rce").replace('/', "."));
        mods.push(serde_json::json!({"module": name, "path": rel, "sha256": module_hash(&src)?}));
        imps.extend(imports(&src));
    }
    mods.sort_by(|a, b| a["module"].as_str().cmp(&b["module"].as_str()));
    // an import of the built-in library (4.1) pins the library by the hash
    // of its manifest; an import of a module of the directory needs nothing
    let own: BTreeSet<&str> = mods.iter().filter_map(|m| m["module"].as_str()).collect();
    let uses_lib = imps.iter().any(|i| !own.contains(i.as_str()) && crate::stdlib::has(i));
    let deps = if uses_lib { serde_json::json!([{"name": "stdlib", "sha256": stdlib_hash()}]) } else { serde_json::json!([]) };
    Ok(canonical(&serde_json::json!({"dependencies": deps, "formatter": "rce-fmt 1", "language": "0.9", "modules": mods})))
}

pub fn check(text: &str, root: &Path) -> Result<BTreeMap<String, String>, Vec<String>> {
    let mut errs: Vec<String> = vec![];
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return Err(vec![format!("W-PIN-1 the manifest is not JSON: {e}")]),
    };
    if canonical(&v) != text {
        pin(&mut errs, "the manifest is not in canonical JSON (6.1): keys sorted, no whitespace, no trailing newline".into());
    }
    let Some(top) = v.as_object() else {
        return Err(vec!["W-PIN-1 the manifest is not an object".into()]);
    };
    for k in top.keys() {
        if !["dependencies", "formatter", "language", "modules", "reads"].contains(&k.as_str()) {
            pin(&mut errs, format!("unknown key `{k}`: unknown keys, features, build flags, globs and inheritance are refused"));
        }
    }
    if top.get("language").and_then(|x| x.as_str()) != Some("0.9") {
        pin(&mut errs, "`language` must be the string \"0.9\"".into());
    }
    if !top.get("formatter").is_some_and(|x| x.as_str().is_some_and(|s| !s.is_empty())) {
        pin(&mut errs, "`formatter` (the pinned formatter version) is required".into());
    }
    let mods = entries(&v, "modules", &["module", "path", "sha256"], &mut errs);
    let deps = entries(&v, "dependencies", &["name", "sha256"], &mut errs);
    let reads = entries(&v, "reads", &["path", "sha256"], &mut errs);
    if !top.contains_key("modules") || !top.contains_key("dependencies") {
        pin(&mut errs, "`modules` and `dependencies` are required (a list may be empty)".into());
    }
    for d in &deps {
        let h = d.get("sha256").and_then(|x| x.as_str()).unwrap_or("");
        let n = d.get("name").and_then(|x| x.as_str()).unwrap_or("?");
        if !is_hash(h) {
            pin(&mut errs, format!("dependency `{n}` is not pinned by a SHA-256 (64 lowercase hex digits)"));
        } else if n == "stdlib" {
            // the built-in library is pinned by its manifest's hash, and the
            // hash is checked against the library this checker carries
            let lib = stdlib_hash();
            if h != lib {
                pin(&mut errs, format!("dependency `stdlib` pins {h}; the built-in library of this checker is {lib} (stdlib/lib/MANIFEST.json)"));
            }
        } else if crate::stdlib::has(n) || crate::stdlib::names().iter().any(|m| m.split('.').next() == Some(n)) {
            pin(&mut errs, format!("dependency `{n}` names a module of the built-in library; the library is pinned as the one dependency `stdlib`, by the hash of its manifest"));
        }
    }
    let mut names = BTreeSet::new();
    let mut pinned = BTreeMap::new();
    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for e in &mods {
        let (name, path, want) = (
            e.get("module").and_then(|x| x.as_str()).unwrap_or(""),
            e.get("path").and_then(|x| x.as_str()).unwrap_or(""),
            e.get("sha256").and_then(|x| x.as_str()).unwrap_or(""),
        );
        if !names.insert(name.to_string()) {
            pin(&mut errs, format!("module `{name}` is listed twice"));
        }
        if let Some(why) = bad_path(path) {
            pin(&mut errs, format!("module `{name}`: `{path}` is {why}"));
            continue;
        }
        if !is_hash(want) {
            pin(&mut errs, format!("module `{name}` is not pinned by a SHA-256"));
            continue;
        }
        match std::fs::read_to_string(root.join(path)) {
            Err(_) => pin(&mut errs, format!("module `{name}`: cannot read `{path}`")),
            Ok(src) => match module_hash(&src) {
                Err(e) => pin(&mut errs, format!("module `{name}`: `{path}` does not parse: {e}")),
                Ok(h) => {
                    if h != want {
                        pin(&mut errs, format!("module `{name}`: `{path}` hashes to {h}, the manifest pins {want}"));
                    }
                    pinned.insert(name.to_string(), h);
                    graph.insert(name.to_string(), imports(&src));
                }
            },
        }
    }
    for e in &reads {
        let (path, want) = (e.get("path").and_then(|x| x.as_str()).unwrap_or(""), e.get("sha256").and_then(|x| x.as_str()).unwrap_or(""));
        if let Some(why) = bad_path(path) {
            pin(&mut errs, format!("read `{path}` is {why}"));
        } else {
            match std::fs::read(root.join(path)) {
                Err(_) => pin(&mut errs, format!("read `{path}` cannot be read")),
                Ok(b) => {
                    if sha256_hex(&b) != want {
                        pin(&mut errs, format!("read `{path}` hashes to {}, the manifest pins {want}", sha256_hex(&b)));
                    }
                }
            }
        }
    }
    // an import is a pinned module of the manifest; there is no resolver
    let dep_names: BTreeSet<&str> = deps.iter().filter_map(|d| d.get("name").and_then(|x| x.as_str())).collect();
    for (m, imps) in &graph {
        for i in imps {
            let built_in = dep_names.contains("stdlib") && crate::stdlib::has(i);
            if !names.contains(i) && !built_in && !dep_names.contains(i.as_str()) && !dep_names.contains(i.split('.').next().unwrap_or("")) {
                pin(&mut errs, format!("module `{m}` imports `{i}`, which the manifest does not pin by hash"));
            }
        }
    }
    // import cycles (W-MOD-2)
    fn cyc(n: &str, g: &BTreeMap<String, Vec<String>>, st: &mut BTreeMap<String, u8>) -> bool {
        match st.get(n) {
            Some(1) => return true,
            Some(2) => return false,
            _ => {}
        }
        st.insert(n.into(), 1);
        for m in g.get(n).into_iter().flatten() {
            if g.contains_key(m) && cyc(m, g, st) {
                return true;
            }
        }
        st.insert(n.into(), 2);
        false
    }
    let mut st = BTreeMap::new();
    for n in graph.keys() {
        if cyc(n, &graph, &mut st) {
            pin(&mut errs, format!("import cycle through `{n}` (W-MOD-2)"));
            break;
        }
    }
    if errs.is_empty() { Ok(pinned) } else { Err(errs) }
}

/// A signed advisory (F.5.3): withdrawal of a hash. It never changes what a
/// hash means, only whether a tool warns on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Advisory {
    pub hash: String,
    pub reason: String,
    pub issued_by: String,
    pub seq: u64,
}

/// Verify a list of signed advisories: `[{"advisory": {kind, hash, reason,
/// issued_by, seq}, "signature": "<128 hex>"}, ...]`. Each is signed with
/// Ed25519 (RFC 8032, deterministic) over the SHA-256 of the canonical JSON of
/// the `advisory` object, by the key trusted for `issued_by`. `seq` of one
/// issuer strictly increases through the list (a replayed or reordered advisory
/// is refused).
pub fn verify_advisories(text: &str, keys: &BTreeMap<String, [u8; 32]>) -> Result<Vec<Advisory>, Vec<String>> {
    let v: Value = serde_json::from_str(text).map_err(|e| vec![format!("W-PIN-1 the advisories are not JSON: {e}")])?;
    let Some(list) = v.as_array() else { return Err(vec!["W-PIN-1 the advisories are not a list".into()]) };
    let mut errs = vec![];
    let mut out = vec![];
    let mut last: BTreeMap<String, u64> = BTreeMap::new();
    for (i, e) in list.iter().enumerate() {
        let adv = e.get("advisory");
        let sig = e.get("signature").and_then(|s| s.as_str()).unwrap_or("");
        let Some(a) = adv.and_then(|a| a.as_object()) else {
            errs.push(format!("W-PIN-1 advisory {i} has no `advisory` object"));
            continue;
        };
        let get = |k: &str| a.get(k).and_then(|x| x.as_str()).map(String::from);
        let (Some(hash), Some(reason), Some(by), Some(seq)) = (get("hash"), get("reason"), get("issued_by"), a.get("seq").and_then(|x| x.as_u64())) else {
            errs.push(format!("W-PIN-1 advisory {i} lacks `hash`, `reason`, `issued_by` or `seq`"));
            continue;
        };
        if get("kind").as_deref() != Some("advisory") || a.len() != 5 || !is_hash(&hash) {
            errs.push(format!("W-PIN-1 advisory {i} is not exactly {{kind: \"advisory\", hash, reason, issued_by, seq}} with a SHA-256 hash"));
            continue;
        }
        let Some(key) = keys.get(&by) else {
            errs.push(format!("W-PIN-1 advisory {i} is issued by `{by}`, whose key is not trusted"));
            continue;
        };
        let digest = Sha256::digest(canonical(adv.unwrap()).as_bytes());
        let sigb: Vec<u8> = (0..sig.len() / 2).filter_map(|j| u8::from_str_radix(sig.get(2 * j..2 * j + 2)?, 16).ok()).collect();
        let ok = sig.len() == 128
            && sigb.len() == 64
            && ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, key).verify(&digest, &sigb).is_ok();
        if !ok {
            errs.push(format!("W-PIN-1 advisory {i} (hash {hash}) has a bad signature"));
            continue;
        }
        if last.get(&by).is_some_and(|l| seq <= *l) {
            errs.push(format!("W-PIN-1 advisory {i}: `seq` {seq} of `{by}` does not increase"));
            continue;
        }
        last.insert(by.clone(), seq);
        out.push(Advisory { hash, reason, issued_by: by, seq });
    }
    if errs.is_empty() { Ok(out) } else { Err(errs) }
}

/// The pinned hashes of a manifest that carry an advisory (the lockfile's
/// unsafe list): module hashes and dependency hashes. A tool warns on them.
pub fn unsafe_list(manifest_text: &str, pinned: &BTreeMap<String, String>, adv: &[Advisory]) -> Vec<String> {
    let mut all: BTreeMap<String, String> = pinned.iter().map(|(k, v)| (v.clone(), format!("module {k}"))).collect();
    if let Ok(v) = serde_json::from_str::<Value>(manifest_text)
        && let Some(ds) = v.get("dependencies").and_then(|d| d.as_array())
    {
        for d in ds {
            if let (Some(n), Some(h)) = (d.get("name").and_then(|x| x.as_str()), d.get("sha256").and_then(|x| x.as_str())) {
                all.insert(h.to_string(), format!("dependency {n}"));
            }
        }
    }
    adv.iter()
        .filter_map(|a| all.get(&a.hash).map(|what| format!("{} ({what}): {} [issued by {}, seq {}]", a.hash, a.reason, a.issued_by, a.seq)))
        .collect()
}
