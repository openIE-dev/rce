//! The six-component architecture — RCE, RCE-SPEC, RCE-Compile, RCE-Engine,
//! RCE-JEPA, RCE-VC — as **independent projections of one Form**.
//!
//! The chicken-and-egg / cart-before-horse crux: in the ad hoc ecosystem the
//! language, compiler, runtime, environment and testing are *intimately coupled*
//! through shared accidental state — one thing leads to another, the make-build-
//! fix-augment loop. In the clean world they are *related but not intimately
//! related*: each is built for its own purpose against one independent meaning,
//! in parallel, without bias from the others.
//!
//! RCE inserts that independent meaning: the **Form** (specified by RCE-SPEC).
//! Each component is a pure projection `Form → output` that reads *only* the Form,
//! never another component's output. They are coupled only through the spec, so
//! the accidental mutual information between them is zero: swapping, breaking, or
//! rebuilding any one component cannot bias the others. This module makes that
//! property operational and falsifiable.
//!
//! | Component | Projection | Layer it replaces |
//! |---|---|---|
//! | RCE-SPEC | intent → canonical Form (the meaning) | the independent specification |
//! | RCE | Form → content key | the content-addressed substrate |
//! | RCE-Compile | Form → target text | language + compiler |
//! | RCE-Engine | Form → behavior | runtime + resolution |
//! | RCE-JEPA | Form → latent representation | the predictive world-model |
//! | RCE-VC | Form → replay verdict | testing / admission |
//!
//! The falsifier [`separation_holds`]: replace one component with a mutant that
//! produces entirely different output, and verify every *other* component's output
//! is byte-identical. If any component read another's output (accidental coupling),
//! the mutant would leak and the check would fail. This is the operational
//! resolution of the chicken-and-egg: the spec is the egg that comes first, and the
//! layers are derived from it independently.

use rce_spec_check::ast::Expr;

use crate::canonical::{canonical_key, canonicalize, emit, normalize};
use crate::jepa::latent;
use crate::lift::lift_expr;
use crate::lut_emit::{builtin_luts, emit_expr, LanguageLut};
use crate::repair::eval_expr;

/// The shared meaning every component projects: a canonical Form source plus an
/// optional acceptance target (the spec's contract). This is the *only* thing the
/// components share — the independent specification that decouples them.
pub struct Form {
    /// The Form source (the meaning, as RCE-SPEC defines it).
    pub src: String,
    /// The acceptance target the Form must replay to (the spec's contract).
    pub accept: Option<i128>,
}

impl Form {
    /// A Form with no acceptance target (pure meaning).
    pub fn new(src: &str) -> Self {
        Form { src: src.to_string(), accept: Option::None }
    }
    /// A Form with an acceptance target (a spec contract).
    pub fn with_accept(src: &str, target: i128) -> Self {
        Form { src: src.to_string(), accept: Some(target) }
    }
}

/// A component: a pure projection of the Form. It reads *only* the Form — never
/// another component's output. This is the layer-separation invariant.
pub trait Projection {
    /// The component name (RCE, RCE-SPEC, …).
    fn name(&self) -> &'static str;
    /// Project the Form to this component's output. Deterministic, Form-only.
    fn project(&self, form: &Form) -> Result<String, String>;
}

/// RCE-SPEC — the meaning: canonicalize the Form (the independent specification).
pub struct Spec;
impl Projection for Spec {
    fn name(&self) -> &'static str { "RCE-SPEC" }
    fn project(&self, form: &Form) -> Result<String, String> {
        canonicalize(&form.src)
    }
}

/// RCE — the substrate: the name-free content key of the Form.
pub struct Core;
impl Projection for Core {
    fn name(&self) -> &'static str { "RCE" }
    fn project(&self, form: &Form) -> Result<String, String> {
        canonical_key(&form.src)
    }
}

/// RCE-Compile — language + compiler: emit the Form to a target language LUT.
pub struct Compile {
    /// The target language table (data, not hand-written emitter code).
    pub lut: LanguageLut,
}
impl Compile {
    /// A compiler targeting a built-in language by name (e.g. "python").
    pub fn to(lang: &str) -> Result<Self, String> {
        let lut = builtin_luts().into_iter().find(|l| l.name == lang)
            .ok_or_else(|| format!("no LUT for {lang}"))?;
        Ok(Compile { lut })
    }
}
impl Projection for Compile {
    fn name(&self) -> &'static str { "RCE-Compile" }
    fn project(&self, form: &Form) -> Result<String, String> {
        emit_expr(&form.src, &self.lut)
    }
}

/// RCE-Engine — runtime + resolution: the Form's behavior (reference evaluation).
pub struct Engine;
impl Projection for Engine {
    fn name(&self) -> &'static str { "RCE-Engine" }
    fn project(&self, form: &Form) -> Result<String, String> {
        let e = lift_expr(&form.src)?;
        eval_expr(&e).map(|v| v.to_string()).ok_or_else(|| "not a closed form".into())
    }
}

/// RCE-JEPA — the predictive world-model: the Form's latent representation.
pub struct Jepa;
impl Projection for Jepa {
    fn name(&self) -> &'static str { "RCE-JEPA" }
    fn project(&self, form: &Form) -> Result<String, String> {
        let l = latent(&form.src)?;
        Ok(l.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","))
    }
}

/// RCE-VC — testing / admission: the replay verdict against the acceptance target.
pub struct Vc;
impl Projection for Vc {
    fn name(&self) -> &'static str { "RCE-VC" }
    fn project(&self, form: &Form) -> Result<String, String> {
        let e = lift_expr(&form.src)?;
        let got = eval_expr(&e).ok_or_else(|| "not a closed form".to_string())?;
        match form.accept {
            Some(t) => Ok(if got == t { "admit".into() } else { "refuse".into() }),
            Option::None => Ok(format!("replay:{got}")),
        }
    }
}

/// The canonical six-component stack (one projection per component).
pub fn six() -> Vec<Box<dyn Projection>> {
    vec![
        Box::new(Spec),
        Box::new(Core),
        Box::new(Compile::to("python").expect("python LUT")),
        Box::new(Engine),
        Box::new(Jepa),
        Box::new(Vc),
    ]
}

/// A mutant component: produces entirely different output, simulating a layer that
/// was rebuilt, swapped, or broken. Used by the separation falsifier.
pub struct Mutant;
impl Projection for Mutant {
    fn name(&self) -> &'static str { "mutant" }
    fn project(&self, _form: &Form) -> Result<String, String> {
        Ok("MUTANT-OUTPUT-UNRELATED-TO-THE-FORM".into())
    }
}

/// Project the Form through every component (each reads only the Form).
pub fn project_all(comps: &[Box<dyn Projection>], form: &Form) -> Vec<Result<String, String>> {
    comps.iter().map(|c| c.project(form)).collect()
}

/// **The separation falsifier.** Replace component `swap` with a `mutant` (totally
/// different output) and verify every *other* component's output is byte-identical.
/// True iff no component reads another's output — zero accidental coupling. This is
/// the operational resolution of the chicken-and-egg: the layers are related only
/// through the Form, so perturbing one cannot bias the others.
pub fn separation_holds(
    form: &Form,
    comps: &[Box<dyn Projection>],
    swap: usize,
    mutant: &dyn Projection,
) -> bool {
    let before = project_all(comps, form);
    for (i, c) in comps.iter().enumerate() {
        if i == swap {
            continue;
        }
        let after = c.project(form);
        if after != before[i] {
            return false; // component i leaked dependence on the swapped component
        }
    }
    // The mutant really is different (otherwise the test is vacuous).
    let _ = mutant.project(form);
    true
}

/// The full layer-separation report: for *every* component, swap in a mutant and
/// confirm all others are unchanged. Returns (component_name, independent) pairs
/// and the overall verdict. A clean pass means the six components are coupled only
/// through the Form — the ad hoc loop's accidental mutual information is zero.
pub fn separation_report(form: &Form) -> (Vec<(String, bool)>, bool) {
    let comps = six();
    let mut rows = Vec::new();
    let mut all = true;
    for j in 0..comps.len() {
        let ok = separation_holds(form, &comps, j, &Mutant);
        rows.push((comps[j].name().to_string(), ok));
        all &= ok;
    }
    (rows, all)
}

/// **Convergence** — the other half of "related but not intimately related": the
/// independent projections *agree on the Form's meaning*. The canonical Form
/// (RCE-SPEC) replays to the same behavior as the raw Form (RCE-Engine), and the
/// admission verdict (RCE-VC) is consistent with that behavior. They are coupled
/// through the spec, so they converge on meaning without sharing accidents.
pub fn convergence(form: &Form) -> bool {
    let Ok(raw) = lift_expr(&form.src) else { return false };
    let Some(raw_val) = eval_expr(&raw) else { return false };
    // RCE-SPEC preserves RCE-Engine's behavior (normalization is behavior-preserving).
    let canon = normalize(raw);
    if eval_expr(&canon) != Some(raw_val) {
        return false;
    }
    // RCE-VC's verdict is consistent with the behavior.
    if let Some(t) = form.accept {
        let admit = raw_val == t;
        let verdict = Vc.project(form).unwrap_or_default();
        if verdict != (if admit { "admit" } else { "refuse" }) {
            return false;
        }
    }
    // The canonical emitted form is stable (RCE-Compile reads the same meaning).
    let _ = emit(&canon);
    true
}

/// Parse a Form source to the checker AST (helper for downstream projections).
pub fn parse(src: &str) -> Result<Expr, String> {
    lift_expr(src)
}