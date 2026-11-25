//! Planetary unsiloing — across the board: every language has an origin; one competence serves all.
//!
//! The four classes (functional/web/systems/math-HPC) are a lens, not the limit. Every language was
//! made for a reason — its origin is its nuance. They grew in siloes fenced by arbitrary moats. RCE
//! unsilos them: a competence is written once as a Form, verified once, and projected to every
//! surface, each keeping its idiom. The hardware↔software↔user cycle is the true three-body problem
//! (information theory × material science × software development); at its center is **user benefit**
//! — correct competence delivered per joule. Unsiloing maximizes that center by letting every
//! origin contribute to, and draw from, one amorphous verified substrate.

use crate::fabric::{builtin_fabrics, place};
use crate::lut_emit::{builtin_luts, emit_expr};

/// A programming class (a lens on the whole): idiom + representative surface + natural fabric.
pub struct Class {
    pub name: &'static str,
    pub nuance: &'static str,
    pub lang: &'static str,
    pub fabric: &'static str,
}

pub fn classes() -> Vec<Class> {
    vec![
        Class { name: "functional", nuance: "equational reasoning, purity, composition, immutable data",
            lang: "haskell", fabric: "cpu-arm64" },
        Class { name: "web", nuance: "ubiquity, async/event-driven, browser/edge reach",
            lang: "js", fabric: "wasm" },
        Class { name: "systems", nuance: "control, zero-cost abstraction, memory & concurrency",
            lang: "rust", fabric: "cpu-x86" },
        Class { name: "math-hpc", nuance: "numeric density, tensors, massive parallelism",
            lang: "c", fabric: "gpu-nvidia-ptx" },
    ]
}

/// Why each language was made — its origin, the nuance it contributes to the whole.
pub fn origins() -> Vec<(&'static str, &'static str)> {
    vec![
        ("c", "portable systems, close to the machine"),
        ("cpp", "zero-cost abstraction over C"),
        ("rust", "memory safety without garbage collection"),
        ("go", "simple concurrent cloud services"),
        ("zig", "simple, explicit systems code"),
        ("python", "readability and glue"),
        ("ruby", "developer happiness"),
        ("perl", "text processing glue"),
        ("php", "server-side web templating"),
        ("lua", "embeddable scripting"),
        ("js", "the language of the web"),
        ("typescript", "typed web at scale"),
        ("java", "write once, run on the JVM"),
        ("kotlin", "pragmatic JVM/Android"),
        ("scala", "FP + OO on the JVM"),
        ("swift", "safe, fast Apple systems"),
        ("csharp", "productive managed systems"),
        ("haskell", "purity, laziness, equational reasoning"),
        ("ocaml", "pragmatic functional + objects"),
        ("fsharp", "functional-first on .NET"),
        ("elm", "reliable functional web UI"),
        ("clojure", "Lisp, immutable data on the JVM"),
        ("scheme", "minimal Lisp, language as substrate"),
        ("racket", "Lisp for making languages"),
        ("commonlisp", "symbolic computing, the programmable program"),
        ("erlang", "fault-tolerant telecom concurrency"),
        ("elixir", "productive BEAM concurrency"),
        ("r", "statistics and data analysis"),
        ("julia", "numeric/HPC without the two-language problem"),
        ("fortran", "array numerics for science"),
        ("matlab", "matrix engineering"),
        ("sql", "declarative data queries"),
        ("prolog", "logic and relations"),
        ("ada", "safe real-time / embedded"),
        ("vb", "approachable business automation"),
        ("dart", "client-optimized UI"),
        ("nim", "Pythonic systems efficiency"),
        ("crystal", "Ruby syntax, native speed"),
        ("d", "systems with modern ergonomics"),
        ("groovy", "dynamic JVM scripting"),
        ("objectivec", "Smalltalk messaging on C"),
        ("solidity", "deterministic contract execution"),
        ("move", "resource-safe digital assets"),
        ("cangjie", "native intelligent-era systems (Huawei)"),
        ("moonbit", "fast, safe, verified-friendly (IDEA)"),
        ("purescript", "pure FP compiling to JS"),
    ]
}

fn origin_of(lang: &str) -> &'static str {
    origins().into_iter().find(|(l, _)| *l == lang).map(|(_, o)| o).unwrap_or("a distinct origin")
}

/// A cross-surface projection of one Form.
pub struct Projection {
    pub lang: String,
    pub origin: &'static str,
    pub code: String,
}

/// Project a single Form to EVERY language LUT (across the board), each keeping its idiom/origin.
pub fn project_all(src: &str) -> Result<Vec<Projection>, String> {
    let mut out = Vec::new();
    for lut in builtin_luts() {
        let code = emit_expr(src, &lut)?;
        out.push(Projection { lang: lut.name.clone(), origin: origin_of(&lut.name), code });
    }
    Ok(out)
}

/// The lowest-joule fabric across the heterogeneous set (HPC placement).
pub fn best_fabric() -> Option<&'static str> {
    place(&builtin_fabrics()).map(|f| f.name)
}