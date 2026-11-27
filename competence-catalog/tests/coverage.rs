//! Structure-first coverage falsifiers: the (construct × language) matrix is filled to 100%
//! structurally, higher constructs render, and grammar ingestion derives coverage (not a random walk).

use competence_catalog::ingest::constructs_from_node_types;
use competence_catalog::lut_emit::{builtin_luts, emit_expr};
use competence_catalog::structure::{coverage, gaps, overall, CONSTRUCTS};
use serde_json::json;

#[test]
fn coverage_matrix_is_complete() {
    assert_eq!(CONSTRUCTS.len(), 113);
    for l in builtin_luts() {
        assert!(gaps(&l).is_empty(), "{} missing {:?}", l.name, gaps(&l));
        assert_eq!(coverage(&l), 1.0);
    }
    assert_eq!(overall(), 1.0);
}

#[test]
fn emission_covers_every_ingested_language() {
    let names: std::collections::BTreeSet<String> =
        builtin_luts().iter().map(|l| l.name.clone()).collect();
    let measured = [
        "rust", "python", "go", "javascript", "java", "c", "cpp", "c-sharp", "ruby", "swift",
        "kotlin", "scala", "haskell", "julia", "lua", "zig", "ocaml", "r", "dart", "elixir",
        "erlang", "clojure", "scheme", "nim", "crystal", "d", "fortran", "php", "ada", "sml",
        "purescript", "mojo", "cangjie", "wenyan", "elm", "racket", "commonlisp", "elisp",
        "gleam", "moonbit", "rescript", "reason", "nix", "fsharp", "idris", "agda", "lean",
        "fstar", "perl", "coq", "dhall", "unison", "nadeshiko", "ezhil", "alusus",
        "koka", "grain", "roc", "v", "odin",
        "bash", "typescript", "html", "css", "json", "yaml", "toml", "dockerfile", "cmake",
        "make", "powershell", "tlaplus", "markdown", "graphql", "prolog",
        "cuda", "glsl", "objc", "hcl", "tcl", "vim",
        "pony", "starlark", "hare", "hlsl", "cairo", "squirrel", "smali",
        "firrtl", "ispc", "ron", "puppet", "xcompose", "ssh-config",
        "cobol", "pli", "rpgle", "pascal", "basic", "verilog", "thrift", "scss", "slang", "capnp",
        "sql", "solidity",
    ];
    for k in measured {
        let emit = match k { "javascript" => "js", "c-sharp" => "csharp", "objc" => "objectivec", _ => k };
        assert!(names.contains(emit), "ingested language {k} has no emission LUT row");
    }
}

#[test]
fn higher_constructs_render() {
    let rust = builtin_luts().into_iter().find(|l| l.name == "rust").unwrap();
    assert!(emit_expr("f(1, 2)", &rust).is_ok());
    assert!(emit_expr("[1, 2, 3]", &rust).is_ok());
    assert!(emit_expr("a[0]", &rust).is_ok());
    assert!(emit_expr("a.b", &rust).is_ok());
    let clj = builtin_luts().into_iter().find(|l| l.name == "clojure").unwrap();
    assert!(emit_expr("f(1, 2)", &clj).is_ok());
}

#[test]
fn ingest_derives_constructs_from_grammar() {
    let nt = json!([
        {"type":"integer","named":true},
        {"type":"identifier","named":true},
        {"type":"binary_expression","named":true},
        {"type":"if_expression","named":true},
        {"type":"call_expression","named":true},
        {"type":"subscript_expression","named":true},
        {"type":"attribute","named":true},
        {"type":"list","named":true},
        {"type":"string","named":true}
    ]);
    let cs = constructs_from_node_types(&nt);
    for c in ["int_lit", "var", "add", "if_expr", "call", "index", "member", "list", "str_lit"] {
        assert!(cs.contains(&c), "missing {c}");
    }
}

#[test]
fn ingest_real_rce_grammar() {
    let paths = ["../rce/tree-sitter/src/node-types.json", "../tree-sitter/src/node-types.json"];
    let Some(p) = paths.iter().find(|p| std::path::Path::new(p).exists()) else {
        println!("rce grammar not present; skipping");
        return;
    };
    let txt = std::fs::read_to_string(p).unwrap();
    let nt: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let cs = constructs_from_node_types(&nt);
    assert!(!cs.is_empty());
    for c in ["var", "call", "if_expr", "list", "index", "member"] {
        assert!(cs.contains(&c), "RCE grammar missing {c}");
    }
}