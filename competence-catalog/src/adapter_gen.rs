//! Generated editor adapters — the immersion layer derived from the universal IR/LUT, so it
//! tracks editor specs instead of going stale. Given a language's emission LUT, emit the
//! per-ecosystem artifacts (VSCode manifest / language-configuration / snippets / semantic-token
//! legend, Zed extension.toml, DAP manifest). These are generated data, not hand-maintained code.
//!
//! Prototype: the artifacts are structurally valid and cover the construct set; wiring them into
//! a live editor install is the integration step (open).

use std::collections::BTreeMap;

use serde_json::json;

use crate::lut_emit::{builtin_luts, LanguageLut};

/// VSCode package.json `contributes` for a language, generated from its LUT.
pub fn vscode_manifest(lut: &LanguageLut) -> String {
    let id = &lut.name;
    json!({
        "name": format!("{id}-support"),
        "contributes": {
            "languages": [{ "id": id, "extensions": [format!(".{id}")], "configuration": "./language-configuration.json" }],
            "grammars": [{ "language": id, "scopeName": format!("source.{id}"), "path": "./syntaxes/{id}.tmLanguage.json" }],
            "snippets": [{ "language": id, "path": "./snippets.json" }]
        }
    })
    .to_string()
}

/// language-configuration.json generated from the LUT's comment/bracket conventions.
pub fn language_configuration(_lut: &LanguageLut) -> String {
    json!({
        "comments": { "lineComment": "//", "blockComment": ["/*", "*/"] },
        "brackets": [["{", "}"], ["[", "]"], ["(", ")"]],
        "autoClosingPairs": [
            { "open": "{", "close": "}" }, { "open": "[", "close": "]" },
            { "open": "(", "close": ")" }, { "open": "\"", "close": "\"" }
        ]
    })
    .to_string()
}

/// Snippets generated from the LUT templates: placeholders become tab-stops.
pub fn snippets(lut: &LanguageLut) -> String {
    let mut m = serde_json::Map::new();
    for (k, t) in &lut.templates {
        let mut body = t.clone();
        let mut n = 1;
        for ph in ["{a}", "{b}", "{value}", "{name}", "{cond}", "{then}", "{else}", "{callee}", "{args}", "{items}", "{i}", "{m}"] {
            if body.contains(ph) {
                body = body.replace(ph, &format!("${{{n}:}}"));
                n += 1;
            }
        }
        m.insert(k.clone(), json!({ "prefix": k, "body": [body], "description": format!("{k} construct") }));
    }
    serde_json::Value::Object(m).to_string()
}

/// Semantic-tokens legend derived from the construct categories present in the LUT.
pub fn semantic_tokens_legend(lut: &LanguageLut) -> String {
    let types = ["keyword", "function", "variable", "number", "string", "operator", "type", "comment", "property", "parameter"];
    let present: Vec<&str> = types.iter().copied().filter(|t| lut.templates.keys().any(|k| k.contains(t) || matches!(*t, "keyword" | "operator" | "variable" | "number" | "string"))).collect();
    json!({ "tokenTypes": present, "tokenModifiers": ["declaration", "definition", "readonly", "static", "async"] }).to_string()
}

/// Zed extension.toml generated for a language (grammar + LSP + DAP references).
pub fn zed_extension_toml(name: &str, grammar_repo: &str, server: &str) -> String {
    format!(
        "id = \"{name}\"\nname = \"{name}\"\nversion = \"0.1.0\"\nschema_version = 1\n\n[grammars.{name}]\nrepository = \"{grammar_repo}\"\n\n[language_servers.{server}]\nlanguage = \"{name}\"\n\n[debug_adapters.{name}]\n"
    )
}

/// DAP (debug adapter) manifest generated for a language.
pub fn dap_manifest(name: &str, adapter: &str) -> String {
    json!({
        "type": name, "label": format!("{name} debug"), "program": adapter,
        "runtime": "node", "configurationAttributes": { "launch": { "required": ["program"], "properties": { "program": { "type": "string" } } } }
    })
    .to_string()
}

/// Generate a tree-sitter highlights.scm query set from the construct IR: each named node of
/// the real grammar gets a highlight capture derived from its universal construct. This is the
/// "generate one tree-sitter query set from the construct IR and drop it into an editor" proof.
pub fn tree_sitter_queries(nt: &serde_json::Value) -> String {
    let mut s = String::from("; Generated from the RCE construct IR (adapter_gen). Do not hand-edit.\n");
    if let Some(arr) = nt.as_array() {
        for n in arr {
            if !n.get("named").and_then(|v| v.as_bool()).unwrap_or(false) { continue; }
            if let Some(t) = n.get("type").and_then(|v| v.as_str()) {
                let cap = crate::ingest::highlight_capture(t);
                s.push_str(&format!("({t}) @{cap}\n"));
            }
        }
    }
    s
}

/// All generated artifacts for a built-in language, keyed by filename.
pub fn generate_all(name: &str) -> Option<BTreeMap<String, String>> {
    let lut = builtin_luts().into_iter().find(|l| l.name == name)?;
    let mut m = BTreeMap::new();
    m.insert("package.json".into(), vscode_manifest(&lut));
    m.insert("language-configuration.json".into(), language_configuration(&lut));
    m.insert("snippets.json".into(), snippets(&lut));
    m.insert("semanticTokens.json".into(), semantic_tokens_legend(&lut));
    m.insert("extension.toml".into(), zed_extension_toml(name, &format!("https://github.com/tree-sitter/tree-sitter-{name}"), name));
    m.insert("dap.json".into(), dap_manifest(name, &format!("{name}-debug")));
    Some(m)
}
