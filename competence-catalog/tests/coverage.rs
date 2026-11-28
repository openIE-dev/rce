//! Structure-first coverage falsifiers: the (construct × language) matrix is filled to 100%
//! structurally, higher constructs render, and grammar ingestion derives coverage (not a random walk).

use competence_catalog::ingest::constructs_from_node_types;
use competence_catalog::lut_emit::{builtin_luts, emit_expr};
use competence_catalog::structure::{coverage, gaps, overall, CONSTRUCTS};
use serde_json::json;

#[test]
fn c_subset_lift_replays() {
    use competence_catalog::lift_rust::lift_c;
    use competence_catalog::program::{replay_admit, Env};
    let p = lift_c("int add(int a, int b) { return a + b; } int r = add(1, 2); return r;").unwrap();
    assert!(replay_admit(&p, &[(Env::new(), 3)]), "C fn + decl + call replays to 3");
    let w = lift_c("int x = 0; while (x < 3) { x = x + 1; } return x;").unwrap();
    let mut env = Env::new();
    let v = competence_catalog::program::eval_stmt(&w, &mut env);
    assert_eq!(v.unwrap(), Some(3), "C while replays to 3");
}

#[test]
fn flowg_core_kernel_ingests_fully() {
    // best-case lift: flowg's core compute kernel maps 1:1 onto the universal taxonomy
    let (m, t) = competence_catalog::flowg_ingest::core_coverage();
    assert_eq!((m, t), (40, 40), "flowg core kernel + near frontier lifts 100%");
    assert!(competence_catalog::flowg_ingest::near_frontier().is_empty());
    assert!(!competence_catalog::flowg_ingest::frontier_groups().is_empty());
}

#[test]
fn logged_placement_keeps_replay_bit_identical() {
    use competence_catalog::placement::{PlacementLog, builtin_targets};
    let mut log = PlacementLog::default();
    let t = builtin_targets();
    let first = log.place("muladd", &t).unwrap();
    // hardware/costs change (drop the cheap tensor backend); replay must reuse the logged backend
    let changed: Vec<_> = t.into_iter().filter(|x| x.name != "amx-tensor").collect();
    let replayed = log.place("muladd", &changed).unwrap();
    assert_eq!(first, replayed, "logged placement: replay reuses backend, bit-identical");
    assert_eq!(log.replay("muladd"), Some(first.as_str()));
}

#[test]
fn two_spellings_one_canonical_identity() {
    // The convergence claim: different surface spellings of one behaviour collapse to one
    // canonical identity (decidable polynomial fragment). Identity is name-free.
    let a = "(a+b)*(a-b)";
    let b = "a*a - b*b";
    assert_eq!(competence_catalog::symbolic::symbolically_equivalent(a, b), Some(true));
    assert_eq!(
        competence_catalog::symbolic::canonical_poly(a).unwrap(),
        competence_catalog::symbolic::canonical_poly(b).unwrap(),
        "two spellings, one canonical key"
    );
}

#[test]
fn db_as_competence_sql_lift() {
    let s = competence_catalog::sql::schema("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);");
    let t = &s["tables"][0];
    assert_eq!(t["name"], "USERS");
    assert_eq!(t["columns"].as_array().unwrap().len(), 3);
    assert_eq!(t["columns"][0]["name"], "id");
    let q = competence_catalog::sql::query("SELECT name, age FROM users WHERE age > 18;");
    assert_eq!(q["projection"].as_array().unwrap().len(), 2);
    assert_eq!(q["source"], "USERS");
    assert!(q["filter"].as_str().unwrap().contains("AGE > 18"));
    // exposed via MCP
    let m = competence_catalog::mcp::call("sql_schema", &serde_json::json!({ "src": "CREATE TABLE t (a INT);" })).unwrap();
    assert_eq!(m["tables"][0]["name"], "T");
}

#[test]
fn python_subset_lift_replays() {
    use competence_catalog::lift_rust::lift_python;
    use competence_catalog::program::{replay_admit, Env};
    let p = lift_python("def add(a, b):\n    return a + b\nx = add(1, 2)\nreturn x\n").unwrap();
    let mut e = Env::new();
    let v = competence_catalog::program::eval_stmt(&p, &mut e);
    assert_eq!(v.unwrap(), Some(3), "Python def + call replays to 3");
    let w = lift_python("x = 0\nwhile x < 3:\n    x = x + 1\nreturn x\n").unwrap();
    assert!(replay_admit(&w, &[(Env::new(), 3)]), "Python while replays to 3");
    let i = lift_python("x = 5\nif x > 3:\n    return 1\nelse:\n    return 0\n").unwrap();
    assert!(replay_admit(&i, &[(Env::new(), 1)]), "Python if/else replays to 1");
}

#[test]
fn behavior_statechart_and_mcp_surface() {
    use competence_catalog::lift_rust::lift_rust;
    let p = lift_rust("let x = 0; while x < 3 { x = x + 1; } return x;").unwrap();
    let sc = competence_catalog::behavior::statechart(&p);
    assert!(sc["states"].as_array().unwrap().len() >= 3);
    assert!(sc["transitions"].as_array().unwrap().iter().any(|t| t["event"] == "loop"));
    assert!(sc["transitions"].as_array().unwrap().iter().any(|t| t["event"] == "repeat"));
    // MCP dispatcher: deterministic understanding tools
    let args = serde_json::json!({ "src": "let x = 1; return x + 1;" });
    let cg = competence_catalog::mcp::call("call_graph", &args).unwrap();
    assert!(cg["nodes"].as_array().is_some());
    let ui = competence_catalog::mcp::call("ui_spec", &args).unwrap();
    assert!(ui["catalog"].as_array().is_some());
    let ad = competence_catalog::mcp::call("admit", &serde_json::json!({ "src": "return 2;", "probes": [ { "env": {}, "result": 2 } ] })).unwrap();
    assert_eq!(ad["admitted"], true);
    assert!(competence_catalog::mcp::tools_list()["tools"].as_array().unwrap().len() >= 6);
}

#[test]
fn xref_call_graph_from_admitted_form() {
    use competence_catalog::lift_rust::lift_rust;
    use competence_catalog::xref::{xref, call_graph};
    let p = lift_rust("fn add(a, b) { return a + b; } let r = add(1, 2); return r;").unwrap();
    let x = xref(&p);
    assert!(x.calls.contains(&("top".to_string(), "add".to_string())));
    assert!(x.defs.contains(&"r".to_string()) && x.defs.contains(&"add".to_string()));
    assert!(x.uses.iter().any(|(n, c)| n == "r" && *c >= 1));
    let g = call_graph(&p);
    assert!(g["edges"].as_array().unwrap().iter().any(|e| e["from"] == "top" && e["to"] == "add"));
}

#[test]
fn rust_subset_lift_round_trips_and_replays() {
    use competence_catalog::lift_rust::lift_rust;
    use competence_catalog::program::{replay_admit, Env};
    let f = lift_rust("fn add(a, b) { return a + b; } let r = add(1, 2); return r;").unwrap();
    assert!(replay_admit(&f, &[(Env::new(), 3)]), "fn + let-call + return replays to 3");
    let p = lift_rust("let x = 1; return x + 1;").unwrap();
    assert!(replay_admit(&p, &[(Env::new(), 2)]));
    // emission is the inverse: render then re-lift still replays
    let lut = competence_catalog::lut_emit::builtin_luts().into_iter().find(|l| l.name == "rust").unwrap();
    let code = competence_catalog::lut_emit::render_stmt(&p, &lut).unwrap();
    let back = lift_rust(&code).unwrap();
    assert!(replay_admit(&back, &[(Env::new(), 2)]));
}

#[test]
fn genui_constrained_catalog_and_renderer_gate() {
    let spec = competence_catalog::genui::ui_spec("let x = 1; return x + 1;").unwrap();
    assert!(competence_catalog::genui::validate(&spec).is_ok());
    let v: serde_json::Value = serde_json::from_str(&spec).unwrap();
    assert!(v["catalog"].as_array().unwrap().len() >= 8);
    // a renderer must refuse any component outside the admitted catalog
    let bad = spec.replace("\"bind\"", "\"raw_html\"");
    assert!(competence_catalog::genui::validate(&bad).is_err());
}

#[test]
fn standards_emission_from_ir() {
    let m = competence_catalog::adapter_gen::generate_all("rust").expect("rust lut");
    for k in ["AGENTS.md", "mcp.json", "agent.json", "SKILL.md"] {
        assert!(m.contains_key(k), "missing {k}");
    }
    assert!(m["AGENTS.md"].contains("replay-gated"));
    let mcp: serde_json::Value = serde_json::from_str(&m["mcp.json"]).unwrap();
    assert!(mcp["tools"].as_array().unwrap().len() >= 3);
    let acp: serde_json::Value = serde_json::from_str(&m["agent.json"]).unwrap();
    assert_eq!(acp["protocol"], "ACP");
    assert!(m["SKILL.md"].contains("## add"));
}

#[test]
fn generative_property_probes_catch_weak_spec_impostor() {
    use competence_catalog::program::{lift_program, generative_probes, replay_admit, Env};
    let good = lift_program("return x + 1;").unwrap();
    let battery = generative_probes(&good);
    assert!(battery.len() >= 6, "property battery over free input");
    assert!(replay_admit(&good, &battery));
    // A constant form matches the single weak probe (x=0 -> 1) but fails the battery.
    let weak = vec![(Env::new(), 1)];
    let impostor = lift_program("return 1;").unwrap();
    assert!(replay_admit(&impostor, &weak), "weak spec is underdetermined");
    assert!(!replay_admit(&impostor, &battery), "generative battery refuses the impostor");
}

#[test]
fn coverage_matrix_is_complete() {
    assert_eq!(CONSTRUCTS.len(), 131);
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
        "sql", "solidity", "latex", "typst",
        "groovy", "erlang", "matlab", "vhdl", "proto", "forth", "smalltalk", "octave",
        "bqn", "uiua", "wolfram",
        "maple", "maxima", "gap", "sage", "macaulay2",
        "goose", "jai", "c3", "vale", "ante", "magma", "carbon", "austral", "pari",
        "vala", "haxe", "gdscript", "fennel", "janet", "inko", "koto", "nu", "fish", "luau",
        "teal", "systemverilog", "dafny", "codeql", "cython", "clean", "gren", "faust", "chuck",
        "supercollider", "gnuplot", "sas", "picat", "pkl", "kcl", "nickel", "cue", "rego",
        "noir", "circom", "tact", "clarity", "mlir", "wit", "usd", "openscad", "zeek",
        "sourcepawn", "sqf", "brightscript", "vba", "qml",
        "alloy", "arduino", "apex", "al", "magik", "p", "ohm", "penrose", "ink", "vcl",
        "cfengine", "json5", "kdl", "ini", "devicetree", "dbml", "just", "meson", "batch",
        "sed", "restructuredtext", "svelte", "vue", "templ", "liquid", "jinja", "twig",
        "caddyfile", "editorconfig", "git-commit", "go-template", "heex", "liquidsoap", "m68k",
        "nginx", "plum", "ros-interface", "soong", "typespec", "vento", "wikitext", "topaz",
        "wasm", "boogie", "scilab", "stan", "asciidoc",
        "m3u", "blade", "dash", "desktop", "diff", "djot", "git-config", "handlebars", "hurl",
        "idl", "lilypond", "mcfunction", "razor", "rbs", "robots", "sh", "slim", "tql",
        "typoscript", "unified-diff", "zsh", "cfml", "cfscript", "pod", "soql", "sosl", "tptp",
        "xml", "dtd", "wing", "rpmspec", "slint",
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