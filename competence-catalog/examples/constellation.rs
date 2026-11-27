//! Build the language constellation from the exhaustive curation + real tree-sitter grammars.
//! Usage: cargo run --example constellation -- [out.json]
//! Reads /tmp/nt-<key>.json for each measured language; writes the graph JSON.

use std::collections::BTreeMap;

use competence_catalog::constellation::{self, UniverseEntry};

const MEASURED: &[(&str, &str)] = &[
    ("rust", "Rust"), ("python", "Python"), ("go", "Go"), ("javascript", "JavaScript"),
    ("java", "Java"), ("c", "C"), ("cpp", "C++"), ("c-sharp", "C#"), ("ruby", "Ruby"),
    ("swift", "Swift"), ("kotlin", "Kotlin"), ("scala", "Scala"), ("haskell", "Haskell"),
    ("julia", "Julia"), ("lua", "Lua"), ("zig", "Zig"), ("ocaml", "OCaml"), ("r", "R"),
    ("dart", "Dart"), ("elixir", "Elixir"), ("erlang", "Erlang"), ("clojure", "Clojure"),
    ("scheme", "Scheme"), ("nim", "Nim"), ("crystal", "Crystal"), ("d", "D"),
    ("fortran", "Fortran"), ("php", "PHP"), ("ada", "Ada"), ("sml", "Standard ML"),
    ("purescript", "PureScript"), ("mojo", "Mojo"), ("cangjie", "Cangjie"), ("wenyan", "Wenyan"),
    ("elm", "Elm"), ("racket", "Racket"), ("commonlisp", "Common Lisp"), ("elisp", "Emacs Lisp"),
    ("gleam", "Gleam"), ("moonbit", "MoonBit"), ("rescript", "ReScript"), ("reason", "Reason"),
    ("nix", "Nix"), ("fsharp", "F#"), ("idris", "Idris"), ("agda", "Agda"), ("lean", "Lean"),
    ("fstar", "F*"), ("perl", "Perl"), ("coq", "Coq"), ("dhall", "Dhall"), ("unison", "Unison"),
    ("nadeshiko", "Nadeshiko"), ("ezhil", "Ezhil"), ("alusus", "Alusus"),
    ("koka", "Koka"), ("grain", "Grain"), ("roc", "Roc"), ("v", "V"), ("odin", "Odin"),
    ("bash", "Bash"), ("typescript", "TypeScript"), ("html", "HTML"), ("css", "CSS"),
    ("json", "JSON"), ("yaml", "YAML"), ("toml", "TOML"), ("dockerfile", "Dockerfile"),
    ("cmake", "CMake"), ("make", "Make"), ("powershell", "PowerShell"), ("tlaplus", "TLA+"),
    ("markdown", "Markdown"), ("graphql", "GraphQL"), ("prolog", "Prolog"),
    ("cuda", "CUDA"), ("glsl", "GLSL"), ("objc", "Objective-C"), ("hcl", "HCL"),
    ("tcl", "Tcl"), ("vim", "Vim script"),
    ("pony", "Pony"), ("starlark", "Starlark"), ("hare", "Hare"),
    ("hlsl", "HLSL"), ("cairo", "Cairo"), ("squirrel", "Squirrel"), ("smali", "Smali"),
    ("firrtl", "FIRRTL"), ("ispc", "ISPC"), ("ron", "RON"), ("puppet", "Puppet"),
    ("xcompose", "Xcompose"), ("ssh-config", "SSH Config"),
];

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/constellation.json".into());
    let mut universe = constellation::universe();
    let mut metrics: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut missing = Vec::new();
    for (key, name) in MEASURED {
        let path = format!("/tmp/nt-{key}.json");
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                let v: serde_json::Value = serde_json::from_str(&s).expect("node-types json");
                metrics.insert(key.to_string(), v);
            }
            Err(_) => missing.push(*key),
        }
        // ensure the measured language is a node even if absent from linguist (e.g. Wenyan)
        if !universe.iter().any(|e| e.key == *key) {
            universe.push(UniverseEntry {
                name: name.to_string(), key: key.to_string(), kind: "programming".into(),
                group: String::new(), extensions: Vec::new(), aliases: Vec::new(),
                language_id: None, color: String::new(), measured: true,
            });
        }
    }
    let graph = constellation::build(&universe, &metrics);
    let json = serde_json::to_string_pretty(&graph).expect("serialize");
    std::fs::write(&out, &json).expect("write graph");
    println!("languages={} measured={} nodes={} edges={} -> {}",
        graph.language_count, graph.measured_count, graph.nodes.len(), graph.edges.len(), out);
    if !missing.is_empty() {
        println!("missing grammars: {missing:?}");
    }
    // sample Law frames
    for n in graph.nodes.iter().filter(|n| n.measured).take(0) {
        let _ = n;
    }
    for key in ["rust", "haskell", "moonbit", "wenyan", "lean"] {
        if let Some(n) = graph.nodes.iter().find(|n| n.id == *key) {
            if let (Some(g), Some(l)) = (&n.grammar, &n.law) {
                println!("{}: N={} C={} H={:.3} red={:.3} surf={:.2} | {}",
                    n.label, g.named_nodes, g.constructs, g.h_struct_bits, g.redundancy,
                    g.surface_per_construct, l.statement);
            }
        }
    }
}
