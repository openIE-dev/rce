//! The language constellation — a knowledge graph + information-theoretic profile for EVERY
//! language in the exhaustive curation (GitHub Linguist `languages.yml`, cross-referenced with
//! the tree-sitter grammar directory). Each node carries the **Law of Coding** frame:
//! `I = H_task - H_corpus -> 0`, priced in joules.
//!
//! Honesty contract: a language is `measured` only when a real tree-sitter `node-types.json`
//! was ingested and its named-node coverage computed. Its information profile (structural
//! entropy, redundancy, surface-forms-per-construct) is grammar-derived, never fabricated.
//! Languages without a public grammar are present as nodes with `measured=false` and no
//! invented numbers — the graph knows *of* them without pretending to *know* them. This is
//! what makes "one IR for all languages" a checked claim rather than a slogan: the IR's job
//! per language is exactly to collapse that language's measured surface redundancy.

use serde::{Deserialize, Serialize};

use crate::ingest;
use crate::lut_emit::builtin_luts;

/// G0 lookup floor and proposer cost (µJ), the Law-of-Coding price anchors.
pub const LOOKUP_UJ: f64 = 1.0;
pub const PROPOSER_UJ: f64 = 5_000_000.0;

/// One language as listed in the exhaustive curation (linguist).
#[derive(Debug, Clone, Deserialize)]
pub struct UniverseEntry {
    pub name: String,
    #[serde(default)]
    pub key: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub group: String,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub language_id: Option<i64>,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub measured: bool,
}

/// Grammar-derived information profile (only for measured languages).
#[derive(Debug, Clone, Serialize)]
pub struct GrammarProfile {
    pub named_nodes: usize,
    pub mapped: usize,
    pub coverage: f64,
    pub constructs: usize,
    /// Shannon entropy of the construct histogram, bits (structural, not corpus).
    pub h_struct_bits: f64,
    /// 1 - H/log2(C): the surface variety that is syntactic sugar, not free information.
    pub redundancy: f64,
    /// N/C: mean surface forms per universal construct — the distance the one-IR LUT bridges.
    pub surface_per_construct: f64,
}

/// The Law of Coding frame mapped to one language.
#[derive(Debug, Clone, Serialize)]
pub struct LawFrame {
    /// H_corpus structural proxy (bits) — the grammar's construct spread.
    pub h_corpus_struct: f64,
    /// Surface redundancy the IR collapses at admission (one-time).
    pub surface_redundancy: f64,
    /// Steady-state per-act cost once competence is admitted (µJ) — the lookup floor.
    pub lookup_uj: f64,
    /// Cold proposer cost (µJ) — what the law drives toward zero reuse of.
    pub proposer_uj: f64,
    /// proposer/lookup ratio — the energy the law saves per reused act.
    pub ratio: f64,
    /// Whether an emission LUT row exists for this language.
    pub emission_row: bool,
    /// Plain-language statement of the law for this language.
    pub statement: String,
}

/// A constellation node (language, or a family/paradigm hub).
#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub kind: String, // "language" | "family" | "paradigm"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub paradigms: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    pub measured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<GrammarProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub law: Option<LawFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// A typed edge.
#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: String, // "family" | "paradigm" | "influence"
}

#[derive(Debug, Clone, Serialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub measured_count: usize,
    pub language_count: usize,
}

/// Curated identity for the major languages (region/year/paradigms). Well-documented public
/// facts; the long tail keeps only its linguist family and an honest empty paradigm list.
fn identity(name: &str) -> (Vec<&'static str>, Option<&'static str>, Option<i32>) {
    let p = |v: &[&'static str]| v.to_vec();
    match name {
        "Rust" => (p(&["systems", "functional", "imperative"]), Some("US/intl"), Some(2010)),
        "Python" => (p(&["imperative", "object-oriented", "functional"]), Some("Netherlands"), Some(1991)),
        "Go" => (p(&["imperative", "concurrent"]), Some("US"), Some(2009)),
        "JavaScript" => (p(&["imperative", "functional", "event-driven"]), Some("US"), Some(1995)),
        "Java" => (p(&["object-oriented", "imperative"]), Some("US"), Some(1995)),
        "C" => (p(&["imperative", "systems"]), Some("US"), Some(1972)),
        "C++" => (p(&["systems", "object-oriented", "generic"]), Some("UK/Denmark"), Some(1985)),
        "C#" => (p(&["object-oriented", "functional", "imperative"]), Some("US"), Some(2000)),
        "Ruby" => (p(&["object-oriented", "functional", "imperative"]), Some("Japan"), Some(1995)),
        "Swift" => (p(&["imperative", "functional", "object-oriented"]), Some("US"), Some(2014)),
        "Kotlin" => (p(&["object-oriented", "functional"]), Some("Russia/Czech"), Some(2011)),
        "Scala" => (p(&["functional", "object-oriented"]), Some("Switzerland"), Some(2004)),
        "Haskell" => (p(&["functional", "lazy", "pure"]), Some("UK/US intl"), Some(1990)),
        "Julia" => (p(&["functional", "imperative", "scientific"]), Some("US"), Some(2012)),
        "Lua" => (p(&["imperative", "functional", "scripting"]), Some("Brazil"), Some(1993)),
        "Zig" => (p(&["systems", "imperative"]), Some("US"), Some(2016)),
        "OCaml" => (p(&["functional", "imperative", "object-oriented"]), Some("France"), Some(1996)),
        "R" => (p(&["functional", "scientific", "imperative"]), Some("NZ/Canada"), Some(1993)),
        "Dart" => (p(&["object-oriented", "functional", "imperative"]), Some("Denmark"), Some(2011)),
        "Elixir" => (p(&["functional", "concurrent", "immutable"]), Some("Brazil/US"), Some(2011)),
        "Erlang" => (p(&["functional", "concurrent", "immutable"]), Some("Sweden"), Some(1986)),
        "Clojure" => (p(&["functional", "lisp", "immutable"]), Some("US"), Some(2007)),
        "Scheme" => (p(&["functional", "lisp", "minimal"]), Some("US"), Some(1975)),
        "Nim" => (p(&["systems", "imperative", "functional"]), Some("Germany"), Some(2008)),
        "Crystal" => (p(&["object-oriented", "systems", "compiled"]), Some("Argentina"), Some(2014)),
        "D" => (p(&["systems", "object-oriented", "imperative"]), Some("US"), Some(2001)),
        "Fortran" => (p(&["imperative", "scientific", "array"]), Some("US"), Some(1957)),
        "PHP" => (p(&["imperative", "object-oriented", "scripting"]), Some("Denmark/Canada"), Some(1995)),
        "Ada" => (p(&["imperative", "systems", "real-time", "strong-typed"]), Some("France/US"), Some(1980)),
        "Standard ML" => (p(&["functional", "module", "strong-typed"]), Some("UK/US"), Some(1983)),
        "PureScript" => (p(&["functional", "pure", "strong-typed"]), Some("UK"), Some(2013)),
        "Mojo" => (p(&["systems", "scientific", "python-superset"]), Some("US"), Some(2023)),
        "Cangjie" => (p(&["object-oriented", "functional", "systems"]), Some("China"), Some(2024)),
        "Wenyan" => (p(&["esoteric", "classical-chinese", "imperative"]), Some("China"), Some(2019)),
        "Elm" => (p(&["functional", "pure", "reactive"]), Some("US"), Some(2012)),
        "Racket" => (p(&["functional", "lisp", "language-oriented"]), Some("US"), Some(1995)),
        "Common Lisp" => (p(&["functional", "lisp", "multi-paradigm"]), Some("US"), Some(1984)),
        "Emacs Lisp" => (p(&["functional", "lisp", "scripting"]), Some("US"), Some(1985)),
        "Gleam" => (p(&["functional", "immutable", "concurrent", "strong-typed"]), Some("UK"), Some(2016)),
        "MoonBit" => (p(&["functional", "systems", "strong-typed"]), Some("China"), Some(2023)),
        "ReScript" => (p(&["functional", "strong-typed", "compiled-to-js"]), Some("China/intl"), Some(2020)),
        "Reason" => (p(&["functional", "strong-typed"]), Some("US/China"), Some(2016)),
        "Nix" => (p(&["functional", "pure", "declarative", "lazy"]), Some("Netherlands"), Some(2003)),
        "F#" => (p(&["functional", "object-oriented", "strong-typed"]), Some("UK/US"), Some(2005)),
        "Idris" => (p(&["functional", "dependent-types", "pure"]), Some("UK"), Some(2007)),
        "Agda" => (p(&["functional", "dependent-types", "proof-assistant"]), Some("Sweden"), Some(2007)),
        "Lean" => (p(&["functional", "dependent-types", "proof-assistant"]), Some("US"), Some(2013)),
        "F*" => (p(&["functional", "dependent-types", "verification"]), Some("US/France"), Some(2011)),
        "TypeScript" => (p(&["object-oriented", "functional", "strong-typed"]), Some("US"), Some(2012)),
        "Coq" => (p(&["functional", "dependent-types", "proof-assistant"]), Some("France"), Some(1989)),
        "Prolog" => (p(&["logic", "declarative"]), Some("France"), Some(1972)),
        "APL" => (p(&["array", "functional"]), Some("US"), Some(1966)),
        "Lisp" => (p(&["functional", "lisp"]), Some("US"), Some(1958)),
        "Smalltalk" => (p(&["object-oriented", "pure"]), Some("US"), Some(1972)),
        "Pascal" => (p(&["imperative", "structured", "strong-typed"]), Some("Switzerland"), Some(1970)),
        "Perl" => (p(&["imperative", "scripting", "text-processing"]), Some("US"), Some(1987)),
        "Dhall" => (p(&["functional", "pure", "total", "config"]), Some("US"), Some(2016)),
        "Unison" => (p(&["functional", "pure", "content-addressed"]), Some("US"), Some(2019)),
        "Nadeshiko" => (p(&["scripting", "japanese", "imperative"]), Some("Japan"), Some(2004)),
        "Ezhil" => (p(&["scripting", "tamil", "imperative"]), Some("India"), Some(2007)),
        "Alusus" => (p(&["systems", "arabic", "compiled"]), Some("West Asia"), Some(2018)),
        "Koka" => (p(&["functional", "effect-typed", "algebraic-effects"]), Some("US/NZ"), Some(2012)),
        "Grain" => (p(&["functional", "wasm", "strong-typed"]), Some("US"), Some(2017)),
        "Roc" => (p(&["functional", "pure", "platform-oriented"]), Some("US"), Some(2018)),
        "V" => (p(&["systems", "compiled", "simple"]), Some("Ukraine"), Some(2019)),
        "Odin" => (p(&["systems", "imperative", "data-oriented"]), Some("US"), Some(2016)),
        "Bash" => (p(&["imperative", "shell", "scripting"]), Some("US"), Some(1989)),
        "TypeScript" => (p(&["object-oriented", "functional", "strong-typed"]), Some("US"), Some(2012)),
        "HTML" => (p(&["markup", "declarative"]), Some("CH/UK"), Some(1993)),
        "CSS" => (p(&["stylesheet", "declarative"]), Some("CH/US"), Some(1996)),
        "JSON" => (p(&["data", "declarative"]), Some("US"), Some(2001)),
        "YAML" => (p(&["data", "declarative"]), Some("US"), Some(2001)),
        "TOML" => (p(&["data", "config"]), Some("US"), Some(2013)),
        "Dockerfile" => (p(&["build", "declarative"]), Some("US"), Some(2013)),
        "CMake" => (p(&["build", "declarative"]), Some("US"), Some(2000)),
        "Make" => (p(&["build", "declarative"]), Some("US"), Some(1976)),
        "PowerShell" => (p(&["shell", "scripting", "object-pipeline"]), Some("US"), Some(2006)),
        "TLA+" => (p(&["spec", "formal", "temporal-logic"]), Some("US"), Some(1999)),
        "Markdown" => (p(&["markup", "lightweight"]), Some("US"), Some(2004)),
        "GraphQL" => (p(&["query", "declarative"]), Some("US"), Some(2015)),
        "Prolog" => (p(&["logic", "declarative"]), Some("France"), Some(1972)),
        "CUDA" => (p(&["parallel", "C++", "GPU"]), Some("US"), Some(2007)),
        "GLSL" => (p(&["shading", "C-like", "GPU"]), Some("US"), Some(2004)),
        "Objective-C" => (p(&["OOP", "C", "dynamic"]), Some("US"), Some(1984)),
        "HCL" => (p(&["config", "declarative"]), Some("US"), Some(2014)),
        "Tcl" => (p(&["scripting", "command"]), Some("US"), Some(1988)),
        "Vim script" => (p(&["scripting", "editor"]), Some("Netherlands"), Some(1991)),
        "Odin" => (p(&["systems", "imperative"]), Some("Norway"), Some(2016)),
        "Pony" => (p(&["actor", "OOP", "capability"]), Some("UK"), Some(2015)),
        "Starlark" => (p(&["config", "scripting"]), Some("US"), Some(2018)),
        "Hare" => (p(&["systems", "imperative"]), Some("US"), Some(2022)),
        "HLSL" => (p(&["shading", "C-like", "GPU"]), Some("US"), Some(2002)),
        "Cairo" => (p(&["systems", "proof", "ZK"]), Some("Israel"), Some(2020)),
        "Squirrel" => (p(&["OOP", "scripting"]), Some("Italy"), Some(2003)),
        "Smali" => (p(&["bytecode", "assembly"]), Some("US"), Some(2010)),
        "FIRRTL" => (p(&["hardware", "IR"]), Some("US"), Some(2016)),
        "ISPC" => (p(&["parallel", "C", "SPMD"]), Some("US"), Some(2011)),
        "RON" => (p(&["config", "data"]), Some("US"), Some(2017)),
        "Puppet" => (p(&["config", "declarative"]), Some("US"), Some(2005)),
        "Xcompose" => (p(&["config", "input"]), Some("US"), Some(1990)),
        "SSH Config" => (p(&["config"]), Some("US"), Some(1995)),
        "COBOL" => (p(&["procedural", "business", "legacy"]), Some("US"), Some(1959)),
        "PL/I" => (p(&["procedural", "legacy"]), Some("US"), Some(1964)),
        "RPG" => (p(&["procedural", "business", "legacy"]), Some("US"), Some(1959)),
        "Pascal" => (p(&["procedural", "structured"]), Some("Switzerland"), Some(1970)),
        "BASIC" => (p(&["procedural", "beginner"]), Some("US"), Some(1964)),
        "Verilog" => (p(&["hardware", "HDL"]), Some("US"), Some(1984)),
        "Thrift" => (p(&["IDL", "serialization"]), Some("US"), Some(2007)),
        "SCSS" => (p(&["styling", "CSS"]), Some("US"), Some(2006)),
        "Slang" => (p(&["shading", "GPU"]), Some("US"), Some(2023)),
        "Cap'n Proto" => (p(&["IDL", "serialization"]), Some("US"), Some(2013)),
        "SQL" => (p(&["query", "declarative"]), Some("US"), Some(1974)),
        "Solidity" => (p(&["smart-contract", "OOP", "C-like"]), Some("Ethereum"), Some(2014)),
        "LaTeX" => (p(&["typesetting", "markup", "declarative"]), Some("US"), Some(1984)),
        "Typst" => (p(&["typesetting", "markup", "scripting"]), Some("Germany"), Some(2023)),
        "Groovy" => (p(&["JVM", "OOP", "scripting"]), Some("US"), Some(2003)),
        "MATLAB" => (p(&["scientific", "array", "numeric"]), Some("US"), Some(1984)),
        "VHDL" => (p(&["hardware", "HDL"]), Some("US"), Some(1987)),
        "Protocol Buffers" => (p(&["IDL", "serialization"]), Some("US"), Some(2008)),
        "Forth" => (p(&["stack", "concatenative"]), Some("US"), Some(1970)),
        "Smalltalk" => (p(&["OOP", "live", "message-passing"]), Some("US"), Some(1972)),
        "Octave" => (p(&["scientific", "array", "numeric"]), Some("US"), Some(1988)),
        "BQN" => (p(&["array", "tacit", "functional"]), Some("US"), Some(2020)),
        "Uiua" => (p(&["array", "stack", "tacit"]), Some("US"), Some(2023)),
        "Wolfram Language" => (p(&["symbolic", "multi-paradigm", "rule-based"]), Some("US"), Some(1988)),
        "Maple" => (p(&["symbolic", "computer-algebra"]), Some("Canada"), Some(1982)),
        "Maxima" => (p(&["symbolic", "computer-algebra"]), Some("US"), Some(1968)),
        "GAP" => (p(&["computational", "group-theory"]), Some("Germany"), Some(1986)),
        "SageMath" => (p(&["symbolic", "computer-algebra", "Python"]), Some("US"), Some(2005)),
        "Macaulay2" => (p(&["symbolic", "algebraic-geometry"]), Some("US"), Some(1993)),
        "Goose" => (p(&["systems", "concurrent", "experimental"]), Some("Netherlands"), Some(2024)),
        "Jai" => (p(&["systems", "game", "metaprogramming"]), Some("US"), Some(2014)),
        "C3" => (p(&["systems", "C-like"]), Some("Sweden"), Some(2019)),
        "Vale" => (p(&["systems", "memory-safe", "generational"]), Some("US"), Some(2020)),
        "Ante" => (p(&["systems", "functional", "linear-types"]), Some("US"), Some(2020)),
        "Magma" => (p(&["symbolic", "computer-algebra", "algebra"]), Some("Australia"), Some(1993)),
        "Carbon" => (p(&["systems", "C++-successor", "generic"]), Some("US"), Some(2022)),
        "Austral" => (p(&["systems", "linear-types", "capability"]), Some("Australia"), Some(2020)),
        "PARI/GP" => (p(&["computer-algebra", "number-theory"]), Some("France"), Some(1985)),
        _ => (Vec::new(), None, None),
    }
}

/// Well-documented influence edges (from -> to), by linguist name.
const INFLUENCES: &[(&str, &str)] = &[
    ("C", "C++"), ("C", "Java"), ("C", "C#"), ("C", "Go"), ("C", "Rust"), ("C", "Zig"),
    ("C++", "Java"), ("C++", "Rust"), ("Java", "Kotlin"), ("Java", "Scala"), ("Java", "C#"),
    ("JavaScript", "TypeScript"), ("Haskell", "Rust"), ("Haskell", "Elm"), ("Haskell", "PureScript"),
    ("Haskell", "Gleam"), ("Haskell", "Idris"), ("Haskell", "Agda"), ("Haskell", "Nix"),
    ("OCaml", "Rust"), ("OCaml", "F#"), ("OCaml", "ReScript"), ("OCaml", "Reason"),
    ("Standard ML", "OCaml"), ("Standard ML", "Haskell"), ("Lisp", "Scheme"), ("Lisp", "Common Lisp"),
    ("Scheme", "Racket"), ("Scheme", "Clojure"), ("Erlang", "Elixir"), ("Erlang", "Gleam"),
    ("Ruby", "Elixir"), ("Ruby", "Crystal"), ("Python", "Mojo"), ("Python", "Nim"), ("Python", "Julia"),
    ("Smalltalk", "Ruby"), ("Smalltalk", "Swift"), ("Pascal", "Ada"), ("Coq", "Lean"),
    ("Agda", "Idris"), ("APL", "J"), ("OCaml", "F*"), ("Rust", "MoonBit"), ("OCaml", "MoonBit"),
    ("Kotlin", "Cangjie"), ("Swift", "Cangjie"), ("Rust", "Cangjie"), ("Haskell", "Dhall"),
    ("Haskell", "Unison"), ("Erlang", "Unison"),
];

/// A curated language family (lineage) for clustering the constellation. Linguist `group`
/// only marks dialects, so most languages need a real family. The long tail falls into honest
/// broad buckets rather than one undifferentiated pile.
fn family_of(name: &str, group: &str) -> String {
    let f = match name {
        // Fine-grained subgroups (less density than the old coarse buckets).
        "C" | "C++" | "C#" | "Objective-C" | "Objective-C++" | "D" => "C family",
        "Rust" | "Go" | "Ada" | "Zig" | "Odin" | "Hare" | "Jai" | "C3" | "Vale" | "Ante"
        | "Carbon" | "Goose" | "Nim" | "V" | "Pony" | "Austral" | "Forth" => "Modern systems",
        "SystemVerilog" | "Verilog" | "VHDL" | "FIRRTL" | "Bluespec" => "Hardware HDL",
        "CUDA" | "GLSL" | "HLSL" | "ISPC" | "Slang" => "GPU / shading",
        "Solidity" | "Cairo" | "Move" => "Smart contract",
        "Java" | "Kotlin" | "Scala" | "Groovy" | "JRuby" | "AspectJ" | "Smali" => "JVM",
        "OCaml" | "Standard ML" | "SML" | "F#" | "Reason" | "ReScript" | "Lazy ML" | "Miranda"
        | "Clean" => "ML family",
        "Haskell" | "Elm" | "PureScript" | "Grain" | "Roc" | "Koka" => "Haskell family",
        "Idris" | "Agda" | "Lean" | "Coq" | "F*" | "Isabelle" | "HOL" | "Twelf" | "Beluga"
        | "ATS" | "Dependent ML" => "Proof assistants",
        "Lisp" | "Common Lisp" | "Emacs Lisp" | "NewLISP" | "Hy" | "Guile" => "Common Lisp family",
        "Scheme" | "Racket" | "Fennel" | "Janet" => "Scheme family",
        "Clojure" => "Clojure family",
        "Erlang" | "Elixir" | "Gleam" | "LFE" | "Alpaca" | "Hamler" => "BEAM / concurrent",
        "JavaScript" | "TypeScript" | "CoffeeScript" | "LiveScript" | "Dart" | "ActionScript"
        | "ECMAScript" | "Flow" => "JavaScript family",
        "Python" | "Mojo" | "Cython" | "Starlark" | "Boo" | "Cobra" => "Python family",
        "Ruby" | "Crystal" | "Mirah" | "Opal" => "Ruby family",
        "Fortran" | "R" | "Julia" | "MATLAB" | "Octave" | "SAS" | "Stata" | "LabVIEW" | "Gnuplot"
        | "APL" | "J" | "K" | "Q" | "BQN" | "Uiua" => "Numeric / array",
        "Mathematica" | "Wolfram Language" | "Maple" | "Maxima" | "GAP" | "SageMath"
        | "Macaulay2" | "Magma" | "PARI/GP" => "Computer algebra",
        "Prolog" | "Mercury" | "Datalog" | "ASP" | "Answer Set Programming" | "Logtalk"
        | "TLA+" | "PlusCal" | "Alloy" => "Logic",
        "Shell" | "Bash" | "Zsh" | "Fish" | "PowerShell" | "Tcsh" | "Awk" | "Sed" => "Shell",
        "Perl" | "Raku" | "Perl 6" | "Tcl" | "Expect" | "Vim script" | "Squirrel" => "Scripting",
        "SQL" | "PLSQL" | "TSQL" | "PLpgSQL" | "SQLPL" | "HiveQL" | "Spark SQL" => "SQL",
        "Nix" | "Dhall" | "Unison" | "HashiCorp Configuration Language" | "HCL" | "Jsonnet"
        | "CUE" | "Puppet" | "RON" | "Xcompose" | "SSH Config" => "Config / functional",
        "Wenyan" | "Brainfuck" | "Whitespace" | "Malbolge" | "Befunge" | "Piet" | "Shakespeare"
        | "Chef" | "LOLCODE" | "Ook!" | "Chicken" | "FALSE" | "INTERCAL" => "Esoteric",
        "Smalltalk" | "Self" | "Newspeak" | "Pharo" | "Squeak" => "Smalltalk",
        "COBOL" | "PL/I" | "RPG" | "ALGOL" | "JOVIAL" => "Legacy / business",
        "Pascal" | "Delphi" | "Object Pascal" | "Modula-2" | "Modula-3" | "Oberon" | "BASIC" => "Pascal / structured",
        "Thrift" | "Cap'n Proto" | "Protocol Buffers" | "Avro" => "IDL / serialization",
        "SCSS" | "Sass" | "Less" | "Stylus" => "styling",
        "Swift" => "Apple / Swift",
        "PHP" | "Hack" | "Zephir" => "PHP family",
        "Lua" | "MoonScript" | "Terra" => "Lua family",
        "MoonBit" | "Cangjie" | "Nadeshiko" => "East Asian new",
        "Ezhil" => "South Asian",
        "Alusus" => "West Asian",
        "HTML" | "CSS" | "XML" | "Markdown" | "LaTeX" | "Typst" => "markup",
        "JSON" | "YAML" | "TOML" | "INI" | "CSV" => "data / config",
        "Dockerfile" | "CMake" | "Make" | "Meson" | "Bazel" | "Ninja" => "build",
        "GraphQL" | "SPARQL" | "CodeQL" | "Rego" => "query",
        // Survey-driven expansion families (27 Nov 2026).
        "Vala" => "C family",
        "Inko" => "Modern systems",
        "Haxe" => "Cross-target",
        "GDScript" | "SQF" | "SourcePawn" | "BrightScript" => "Game / domain scripting",
        "Koto" => "Scripting",
        "Nushell" => "Shell",
        "Luau" | "Teal" => "Lua family",
        "Dafny" => "Proof assistants",
        "Gren" => "Haskell family",
        "Faust" | "ChucK" | "SuperCollider" => "Audio / music",
        "Picat" => "Logic",
        "Pkl" | "KCL" | "Nickel" => "Config / functional",
        "Noir" | "Circom" | "Tact" | "Clarity" => "Smart contract",
        "MLIR" | "WIT" | "WebAssembly" => "IR / interface",
        "USD" | "OpenSCAD" => "Domain / modeling",
        "Zeek" => "Security / network",
        "VBA" => "Legacy / business",
        "QML" => "UI / declarative",
        // Second survey batch families.
        "Alloy" => "Logic",
        "Arduino" => "C family",
        "Apex" => "C family",
        "AL" => "Legacy / business",
        "Magik" => "Scripting",
        "P" => "Modern systems",
        "Ohm" => "IR / interface",
        "Penrose" | "Ink" => "Domain / modeling",
        "VCL" | "CFEngine" => "Config / functional",
        "JSON5" | "KDL" | "INI" | "DeviceTree" | "DBML" => "data / config",
        "Just" | "Meson" | "Batch" => "build",
        "Sed" => "Shell",
        "reStructuredText" | "Templ" | "Liquid" | "Jinja" | "Twig" => "markup",
        "Svelte" | "Vue" => "JavaScript family",
        // Third survey batch families.
        "Caddyfile" | "Nginx" => "Config / functional",
        "EditorConfig" | "Git Commit" => "data / config",
        "Go Template" | "HTML+EEX" | "Vento" | "Wikitext" | "Topaz" => "markup",
        "Liquidsoap" => "Audio / music",
        "Motorola 68K Assembly" => "Assembly",
        "Plumlang" => "Modern systems",
        "ROS Interface" | "TypeSpec" => "IDL / serialization",
        "Soong" => "build",
        _ => "",
    };
    if !f.is_empty() {
        return f.to_string();
    }
    if !group.is_empty() {
        return group.to_string();
    }
    "other".to_string()
}

fn slug(s: &str) -> String {
    // Encode meaningful symbols so C / C++ / C# / F# / F* stay distinct nodes.
    let t = s.to_lowercase().replace('#', "sharp").replace('+', "plus").replace('*', "star");
    let mut out = String::new();
    let mut dash = false;
    for c in t.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Build the constellation. `metrics` maps a language key to its real node-types value
/// (only measured languages are present); every universe entry becomes a node regardless.
pub fn build(universe: &[UniverseEntry], metrics: &std::collections::BTreeMap<String, serde_json::Value>) -> Graph {
    let lut_names: std::collections::BTreeSet<String> =
        builtin_luts().iter().map(|l| l.name.to_lowercase()).collect();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut families: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut paradigms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut measured_count = 0;
    let mut language_count = 0;
    let mut name_to_id: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();

    for e in universe {
        let id = slug(&e.name);
        name_to_id.insert(e.name.clone(), id.clone());
        let (par, region, year) = identity(&e.name);
        let family = family_of(&e.name, &e.group);
        let is_lang = e.kind == "programming" || e.measured;
        if is_lang {
            language_count += 1;
        }
        let (grammar, law) = if let Some(nt) = metrics.get(&e.key) {
            let (n, mapped, c, h, red, surf) = ingest::structural_metrics(nt);
            let cov = if n == 0 { 0.0 } else { mapped as f64 / n as f64 };
            let gp = GrammarProfile {
                named_nodes: n, mapped, coverage: cov, constructs: c,
                h_struct_bits: h, redundancy: red, surface_per_construct: surf,
            };
            let emission_row = lut_names.contains(&e.key.to_lowercase());
            let lf = LawFrame {
                h_corpus_struct: h,
                surface_redundancy: red,
                lookup_uj: LOOKUP_UJ,
                proposer_uj: PROPOSER_UJ,
                ratio: PROPOSER_UJ / LOOKUP_UJ,
                emission_row,
                statement: format!(
                    "I = H_task - H_corpus -> 0. {} surface forms collapse to {} universal constructs \
                     (redundancy {:.2}); once admitted, each act costs the {} µJ lookup floor, not the \
                     {:.0} µJ proposer.",
                    n, c, red, LOOKUP_UJ, PROPOSER_UJ
                ),
            };
            measured_count += 1;
            (Some(gp), Some(lf))
        } else {
            (None, None)
        };
        if grammar.is_some() {
            // measured flag from real metrics, not the curation guess
        }
        let measured = grammar.is_some();
        nodes.push(Node {
            id: id.clone(), label: e.name.clone(),
            kind: if is_lang { "language".into() } else { e.kind.clone() },
            family: Some(family.clone()),
            paradigms: par.iter().map(|s| s.to_string()).collect(),
            region: region.map(|s| s.to_string()), year,
            measured, grammar, law,
            color: if e.color.is_empty() { None } else { Some(e.color.clone()) },
        });
        if is_lang {
            families.insert(family.clone());
            edges.push(Edge { from: id.clone(), to: format!("family:{}", slug(&family)), kind: "family".into() });
            for p in &par {
                paradigms.insert(p.to_string());
                edges.push(Edge { from: id.clone(), to: format!("paradigm:{}", slug(p)), kind: "paradigm".into() });
            }
        }
    }

    for f in &families {
        nodes.push(Node {
            id: format!("family:{}", slug(f)), label: f.clone(), kind: "family".into(),
            family: None, paradigms: Vec::new(), region: None, year: None,
            measured: false, grammar: None, law: None, color: None,
        });
    }
    for p in &paradigms {
        nodes.push(Node {
            id: format!("paradigm:{}", slug(p)), label: p.clone(), kind: "paradigm".into(),
            family: None, paradigms: Vec::new(), region: None, year: None,
            measured: false, grammar: None, law: None, color: None,
        });
    }
    for (a, b) in INFLUENCES {
        if let (Some(fa), Some(fb)) = (name_to_id.get(*a), name_to_id.get(*b)) {
            edges.push(Edge { from: fa.clone(), to: fb.clone(), kind: "influence".into() });
        }
    }

    Graph { nodes, edges, measured_count, language_count }
}

/// Load the embedded exhaustive curation (linguist universe).
pub fn universe() -> Vec<UniverseEntry> {
    serde_json::from_str(include_str!("../data/universe.json")).expect("universe.json parses")
}
