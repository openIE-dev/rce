//! The form layer of rce-form.md: decoding of an `rce-form/1` module, its
//! canonical encoding (C1 to C11), keys, and the gate FG1 to FG15, which
//! applies a module (it complies with the Form) or keeps it with the first
//! rule that fails, the group key, the function index and the node path.
//!
//! - `catalogue`: the catalogue of operations (§4), `spec/data/form-catalogue.json`
//! - `ir`: the data of a module and its canonical JSON (§10)
//! - `decode`: strict decoding (C1, FG1)
//! - `canon`: the Merkle digest and the canonical rules C2 to C11, keys
//! - `types`, `walk`: the types of sources and the identity of values (§7, §8)
//! - `gate`: FG1 to FG15 (§11)
//! - `eval`: a Form evaluator for the admission vectors of FG15 (§2, §18)
pub mod canon;
pub mod catalogue;
pub mod decode;
pub mod eval;
pub mod gate;
pub mod ir;
pub mod types;
pub mod walk;

/// Why the gate keeps a module: the first rule that fails, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct Kept {
    /// FG1 to FG15
    pub rule: &'static str,
    pub group: Option<String>,
    pub func: Option<u32>,
    pub path: String,
    pub reason: String,
}

impl Kept {
    /// One line: `kept RULE GROUP FN PATH: REASON`, `-` for a part that does not apply.
    pub fn line(&self) -> String {
        format!(
            "kept {} {} {} {}: {}",
            self.rule,
            self.group.as_deref().unwrap_or("-"),
            self.func.map(|f| f.to_string()).unwrap_or("-".into()),
            if self.path.is_empty() { "-" } else { &self.path },
            self.reason
        )
    }
}

/// The gate's decision on a module.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// it complies with the Form; the module key
    Applied(String),
    Kept(Kept),
}

/// FG1 to FG15 on a module's bytes, and on its admission vectors when given.
pub fn check(bytes: &[u8], vectors: Option<&[u8]>) -> Outcome {
    match gate::gate(bytes, vectors) {
        Ok(_) => Outcome::Applied(crate::interp::d_sha(&String::from_utf8_lossy(bytes))),
        Err(k) => Outcome::Kept(k),
    }
}
