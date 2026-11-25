//! The execution bridge: run a real stdlib competence through the RCE runtime.
//!
//! A competence record names a stdlib operation (`math.sin`). To execute it on
//! an argument, the bridge generates a caller Form (`use math`, a state holding
//! the result, an event, a step calling the operation), runs it through the
//! reference checker (`rce run`), and parses the resulting state. The stdlib is
//! built into the checker, so this executes the *real* verified competence at
//! the interpreter floor — zero model calls.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use rcevc_core::{Authority, Checker, Scope, State, Statement, Unit};
use serde_json::{json, Value};

use crate::CompetenceRecord;

/// A bridge to the RCE reference runtime.
#[derive(Clone)]
pub struct RuntimeBridge {
    /// Path to the `rce` checker binary (fork mode).
    pub rce_bin: PathBuf,
    /// Directory for generated caller programs (fork mode).
    pub workdir: PathBuf,
    /// Execute in-process through the linked reference interpreter instead of
    /// forking `rce run`. This removes the process-spawn overhead that dominates
    /// the measured energy of a G0 act (the path to the interpreter floor).
    pub in_process: bool,
}

impl RuntimeBridge {
    /// A bridge using the given checker binary and work directory (fork mode).
    pub fn new(rce_bin: impl Into<PathBuf>, workdir: impl Into<PathBuf>) -> Self {
        RuntimeBridge { rce_bin: rce_bin.into(), workdir: workdir.into(), in_process: false }
    }

    /// A bridge that executes in-process through the linked reference
    /// interpreter (`rce_spec_check::cli::run`) — no process fork per act.
    pub fn new_inprocess() -> Self {
        RuntimeBridge { rce_bin: PathBuf::new(), workdir: PathBuf::new(), in_process: true }
    }

    /// Execute a unary `math` function (f64 -> f64) on `x`, returning the result.
    /// Generates a caller Form, runs it, parses the final state value.
    pub fn call_math_unary(&self, func: &str, x: f64) -> Result<f64, String> {
        let prog = format!(
            "module call_{func}\n\nuse math\n\nstate Out\n    v: f64 = 0.0\n\n\
             event Calc(x: f64)\n\nstep Out on Calc(x: a)\n    v = math.{func}(a)\n"
        );
        let path = self.workdir.join(format!("call_{func}.rce"));
        std::fs::create_dir_all(&self.workdir).map_err(|e| e.to_string())?;
        std::fs::write(&path, prog).map_err(|e| e.to_string())?;

        let out = Command::new(&self.rce_bin)
            .arg("run")
            .arg(&path)
            .arg(format!("Calc(x: {x})"))
            .output()
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            return Err(format!("runtime refused: {}{}", stdout, String::from_utf8_lossy(&out.stderr)));
        }
        parse_state_value(&stdout).ok_or_else(|| format!("no value in output: {stdout}"))
    }

    /// Run an arbitrary caller Form through the reference runtime and return the
    /// final state's `v` field as text. This is the general G0 execution path:
    /// the caller `use`s the basis module and calls the owned operation.
    pub fn run_caller(&self, name: &str, prog: &str, event: &str) -> Result<String, String> {
        if self.in_process {
            // The reference interpreter, linked in — no process fork per act.
            let out = rce_spec_check::cli::run(prog, Some(event))
                .map_err(|e| format!("runtime refused: {e}"))?;
            if out.last_kept {
                let last = out.lines.last().cloned().unwrap_or_default();
                return Err(format!("runtime refused: {last}"));
            }
            let stdout = out.lines.join("\n");
            return parse_v(&stdout).ok_or_else(|| format!("no value in output: {stdout}"));
        }
        let path = self.workdir.join(format!("{name}.rce"));
        std::fs::create_dir_all(&self.workdir).map_err(|e| e.to_string())?;
        std::fs::write(&path, prog).map_err(|e| e.to_string())?;
        let out = Command::new(&self.rce_bin)
            .arg("run")
            .arg(&path)
            .arg(event)
            .output()
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            return Err(format!("runtime refused: {}{}", stdout, String::from_utf8_lossy(&out.stderr)));
        }
        parse_v(&stdout).ok_or_else(|| format!("no value in output: {stdout}"))
    }

    /// Execute `hash.fnv1a32` (or any `(&[u8]) -> u32` hash) on bytes.
    pub fn call_hash_u8(&self, module: &str, func: &str, bytes: &[i64]) -> Result<String, String> {
        let lit: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
        let prog = format!(
            "module call_{func}\n\nuse {module}\n\nstate Out\n    v: u32 = 0\n\n\
             event Calc(xs: List[u8])\n\nstep Out on Calc(xs: a)\n    v = {module}.{func}(&a)\n"
        );
        self.run_caller(&format!("call_{func}"), &prog, &format!("Calc(xs: [{}])", lit.join(", ")))
    }

    /// Execute `col.search.binary_search` (`(&[i64], i64) -> i64`).
    pub fn call_binary_search(&self, xs: &[i64], target: i64) -> Result<String, String> {
        let lit: Vec<String> = xs.iter().map(|b| b.to_string()).collect();
        let prog = "module call_bs\n\nuse col.search\n\nstate Out\n    v: i64 = 0\n\n\
             event Calc(xs: List[i64], t: i64)\n\nstep Out on Calc(xs: a, t: b)\n    v = search.binary_search(&a, b)\n";
        self.run_caller("call_bs", prog, &format!("Calc(xs: [{}], t: {target})", lit.join(", ")))
    }

    /// Execute `text.glob.glob_match` (`(&text, &text) -> bool`).
    pub fn call_glob(&self, pat: &str, s: &str) -> Result<String, String> {
        let prog = "module call_glob\n\nuse text.glob\n\nstate Out\n    v: bool = false\n\n\
             event Calc(p: text, s: text)\n\nstep Out on Calc(p: a, s: b)\n    v = glob.glob_match(&a, &b)\n";
        self.run_caller("call_glob", prog, &format!("Calc(p: \"{pat}\", s: \"{s}\")"))
    }

    /// The general G0 executor: run any basis operation whose signature can be
    /// marshalled from JSON. `shape` is the normalized intent shape
    /// `(argtypes)->ret` (generics erased to `T0`, `&mut` written without space).
    /// Generates a typed caller Form, runs it through the reference runtime, and
    /// returns the result text. Unsupported shapes (unit/error-union returns,
    /// `&mut`, custom structs, function arguments) return `Err` so the caller
    /// can report them honestly rather than guess.
    pub fn call_general(
        &self,
        module: &str,
        name: &str,
        shape: &str,
        args: &Value,
    ) -> Result<String, String> {
        let (argstr, ret) = split_shape(shape)?;
        let argtypes = split_top_level(&argstr);
        let arr = args.as_array().ok_or_else(|| "args must be a JSON array".to_string())?;
        if arr.len() != argtypes.len() {
            return Err(format!("arity mismatch: shape wants {}, got {}", argtypes.len(), arr.len()));
        }

        let mut event_bindings: Vec<String> = Vec::new();
        let mut step_patterns: Vec<String> = Vec::new();
        let mut call_args: Vec<String> = Vec::new();
        let mut event_lits: Vec<String> = Vec::new();
        let mut var_decls: Vec<String> = Vec::new();
        let mut mut_types: Vec<String> = Vec::new();
        let mut subst: HashMap<String, String> = HashMap::new();
        for (i, (raw, val)) in argtypes.iter().zip(arr.iter()).enumerate() {
            let (declared_base, rk) = strip_ref(raw);
            let base = resolve_type(&declared_base, val);
            let (ety, lit) = if let Some(cl) = marshal_collection(&base, val) {
                cl
            } else {
                let ety = event_type_of(&base).ok_or_else(|| format!("argument {i} type {base} not marshalable"))?;
                let lit = marshal_lit(&base, val).ok_or_else(|| format!("argument {i} value not marshalable as {base}"))?;
                (ety, lit)
            };
            // Unify the declared (generic) type against the concrete type so a
            // generic return (T0, K, List[T1]…) instantiates consistently.
            unify_generics(&declared_base, &ety, &mut subst);
            let binder = format!("a{i}");
            event_bindings.push(format!("p{i}: {ety}"));
            step_patterns.push(format!("p{i}: {binder}"));
            event_lits.push(format!("p{i}: {lit}"));
            match rk {
                "mut" => {
                    // Stateful: bind a mutable var, pass &mut, read it back below.
                    var_decls.push(format!("var m{i}: {ety} = {binder}"));
                    call_args.push(format!("&mut m{i}"));
                    mut_types.push(ety);
                }
                "ref" => call_args.push(format!("&{binder}")),
                _ => call_args.push(binder),
            }
        }

        let ret = apply_subst(&ret, &subst);
        let prefix = module.rsplit('.').next().unwrap_or(module);
        let call_expr = format!("{prefix}.{name}({})", call_args.join(", "));
        let event_sig = if event_bindings.is_empty() { "Calc".to_string() } else { format!("Calc({})", event_bindings.join(", ")) };
        let step_sig = if step_patterns.is_empty() { "Calc".to_string() } else { format!("Calc({})", step_patterns.join(", ")) };
        let event_lit = if event_lits.is_empty() { "Calc".to_string() } else { format!("Calc({})", event_lits.join(", ")) };

        let (ret_ty, ret_init, body) = if mut_types.is_empty() {
            // Pure path: the state holds the call's result.
            let (rt, ri) = state_init(&ret).ok_or_else(|| format!("return type {ret} not supported"))?;
            (rt, ri, format!("    v = {call_expr}\n"))
        } else if ret == "()" {
            // Unit return: the result is the mutated argument, read back.
            let (rt, ri) = state_init(&mut_types[0]).ok_or_else(|| format!("mut type {} not supported", mut_types[0]))?;
            let mut b = String::new();
            for vd in &var_decls { b.push_str(&format!("    {vd}\n")); }
            b.push_str(&format!("    {call_expr}\n"));
            b.push_str("    v = m0\n");
            (rt, ri, b)
        } else {
            // Value return with &mut args: declare the vars, hold the result.
            let (rt, ri) = state_init(&ret).ok_or_else(|| format!("return type {ret} not supported"))?;
            let mut b = String::new();
            for vd in &var_decls { b.push_str(&format!("    {vd}\n")); }
            b.push_str(&format!("    v = {call_expr}\n"));
            (rt, ri, b)
        };

        let prog = format!(
            "module call_{name}\n\nuse {module}\n\nstate Out\n    v: {ret_ty} = {ret_init}\n\n\
             event {event_sig}\n\nstep Out on {step_sig}\n{body}"
        );
        self.run_caller(&format!("call_{name}"), &prog, &event_lit)
    }
}

/// Infer a concrete scalar type from a JSON value (for an erased generic).
fn infer_scalar(val: &Value) -> String {
    if val.is_boolean() { "bool".into() }
    else if val.is_string() { "text".into() }
    else if val.is_f64() { "f64".into() }
    else if val.is_array() { "List[i64]".into() }
    else { "i64".into() }
}

/// Recursively resolve erased generics (`T`, `T0`, `List[T]`) to concrete types
/// from the JSON value, so a generic operation instantiates consistently.
fn resolve_type(base: &str, val: &Value) -> String {
    // A type variable is a single uppercase letter optionally followed by digits
    // (T, K, V, T0, K1…); multi-letter names (Big, Mat, Dur) are structs.
    let mut cs = base.chars();
    let is_gen = matches!(cs.next(), Some(c) if c.is_ascii_uppercase()) && cs.all(|c| c.is_ascii_digit());
    if is_gen {
        return infer_scalar(val);
    }
    if let Some(inner) = base.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        let ev = val.as_array().and_then(|a| a.first()).unwrap_or(val);
        return format!("List[{}]", resolve_type(inner, ev));
    }
    if let Some(inner) = base.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let ev = val.as_array().and_then(|a| a.first()).unwrap_or(val);
        return format!("[{}]", resolve_type(inner, ev));
    }
    base.to_string()
}

/// Whether a type is a generic variable (single uppercase letter + optional digits).
fn is_generic_var(t: &str) -> bool {
    let mut cs = t.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_uppercase()) && cs.all(|c| c.is_ascii_digit())
}

/// Split a parameterized type into (constructor, inner): `Deque[T0]` → ("Deque","T0"),
/// `[T]` → ("","T"), `i64` → ("i64","").
fn split_param(t: &str) -> (String, String) {
    if let Some(open) = t.find('[') {
        if t.ends_with(']') {
            // A slice `[T]` and `List[T]` share the List constructor for unification.
            let name = if open == 0 { "List".to_string() } else { t[..open].to_string() };
            return (name, t[open + 1..t.len() - 1].to_string());
        }
    }
    (t.to_string(), String::new())
}

/// Unify a declared (generic) type against a concrete type, recording generic→
/// concrete substitutions so a generic return instantiates consistently.
fn unify_generics(declared: &str, concrete: &str, subst: &mut HashMap<String, String>) {
    if is_generic_var(declared) {
        subst.entry(declared.to_string()).or_insert_with(|| concrete.to_string());
        return;
    }
    let (dname, dinner) = split_param(declared);
    let (cname, cinner) = split_param(concrete);
    if dname == cname && !dinner.is_empty() {
        let ds = split_top_level(&dinner);
        let cs = split_top_level(&cinner);
        for (dx, cx) in ds.iter().zip(cs.iter()) {
            unify_generics(dx, cx, subst);
        }
    }
}

/// Apply generic substitutions to a type (recursing into parameters and `?`).
fn apply_subst(t: &str, subst: &HashMap<String, String>) -> String {
    if is_generic_var(t) {
        return subst.get(t).cloned().unwrap_or_else(|| t.to_string());
    }
    if let Some(inner) = t.strip_suffix('?') {
        return format!("{}?", apply_subst(inner, subst));
    }
    let (name, inner) = split_param(t);
    if !inner.is_empty() {
        let parts: Vec<String> = split_top_level(&inner).iter().map(|p| apply_subst(p, subst)).collect();
        if t.starts_with('[') {
            return format!("[{}]", parts.join(", "));
        }
        return format!("{}[{}]", name, parts.join(", "));
    }
    t.to_string()
}

/// Split a normalized shape `(a,b)->r` into (argstring, return).
fn split_shape(shape: &str) -> Result<(String, String), String> {
    let shape = shape.trim();
    let open = shape.find('(').ok_or_else(|| format!("bad shape {shape}"))?;
    // find the matching close paren for the argument list
    let mut depth = 0;
    let mut close = None;
    for (i, ch) in shape.char_indices().skip(open) {
        match ch {
            '(' => depth += 1,
            ')' => { depth -= 1; if depth == 0 { close = Some(i); break; } }
            _ => {}
        }
    }
    let close = close.ok_or_else(|| format!("unbalanced shape {shape}"))?;
    let args = shape[open + 1..close].to_string();
    let rest = shape[close + 1..].trim();
    let ret = rest.strip_prefix("->").ok_or_else(|| format!("shape {shape} has no return"))?.trim().to_string();
    // Strip a recursion-bound suffix (e.g. `List[i64]bound1000000`).
    let ret = match ret.find("bound") {
        Some(i) => ret[..i].to_string(),
        None => ret,
    };
    Ok((args, ret))
}

/// Split top-level commas respecting [..] and (..) nesting.
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '[' | '(' => { depth += 1; cur.push(ch); }
            ']' | ')' => { depth -= 1; cur.push(ch); }
            ',' if depth == 0 => { out.push(cur.trim().to_string()); cur = String::new(); }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out.into_iter().filter(|x| !x.is_empty()).collect()
}

/// Strip a reference prefix: `&mut T` -> (T, "mut"), `&T` -> (T, "ref"), T -> (T, "val").
fn strip_ref(ty: &str) -> (String, &'static str) {
    if let Some(rest) = ty.strip_prefix("&mut") {
        (rest.to_string(), "mut")
    } else if let Some(rest) = ty.strip_prefix('&') {
        (rest.to_string(), "ref")
    } else {
        (ty.to_string(), "val")
    }
}

const INT_TYPES: &[&str] = &["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize"];

/// A marshalable stdlib struct: named fields, each a simple type or another
/// marshalable struct. Extracted from the stdlib definitions (all-simple fields).
#[derive(Clone)]
struct StructDef {
    fields: Vec<(String, String)>,
}

/// The struct registry, keyed by both local and module-qualified name.
fn struct_registry() -> &'static HashMap<String, StructDef> {
    static REG: OnceLock<HashMap<String, StructDef>> = OnceLock::new();
    REG.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("./struct-registry.json"))
            .expect("struct registry parses");
        let mut m = HashMap::new();
        if let Some(obj) = v.as_object() {
            for (name, def) in obj {
                let fields: Vec<(String, String)> = def["fields"].as_object()
                    .map(|fs| fs.iter().map(|(k, t)| (k.clone(), t.as_str().unwrap_or("").to_string())).collect())
                    .unwrap_or_default();
                let sd = StructDef { fields };
                m.insert(name.clone(), sd.clone());
                if let Some(q) = def["qualified"].as_str() {
                    m.insert(q.to_string(), sd);
                }
            }
        }
        m
    })
}

/// Resolve a shape type to a marshalable struct, returning its in-scope local
/// name (the name `use <module>` brings into scope) and its definition.
fn lookup_struct(base: &str) -> Option<(String, &'static StructDef)> {
    let reg = struct_registry();
    if let Some(sd) = reg.get(base) {
        let local = base.rsplit('.').next().unwrap_or(base).to_string();
        return Some((local, sd));
    }
    None
}

/// The Form event type for a base argument type, or None if not marshalable.
/// A slice `[T]` is carried as `List[T]` (the caller passes it by reference).
fn event_type_of(base: &str) -> Option<String> {
    if INT_TYPES.contains(&base) || base == "f32" || base == "f64" || base == "bool" || base == "text" {
        return Some(base.to_string());
    }
    if let Some(inner) = base.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        return event_type_of(inner).map(|t| format!("List[{t}]"));
    }
    if let Some(inner) = base.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return event_type_of(inner).map(|t| format!("List[{t}]"));
    }
    if let Some((local, _)) = lookup_struct(base) {
        return Some(local);
    }
    // Defunctionalized closures: a kind tag (e.g. col.func.Op) is first-order
    // data the combinator applies via an interpreter (Reynolds/Danvy).
    if base == "Op" {
        return Some("Op".into());
    }
    if base == "PParser" {
        return Some("PParser".into());
    }
    if let Some((local, _)) = lookup_kind(base) {
        return Some(local);
    }
    None
}

/// Marshal a recursive defunctionalized parser term (text.parse.PParser): a
/// nullary tag (`PDigit`, `PInt`) or a constructor with fields (`PChar(want:)`,
/// `PSeq(a:,b:)`, `PAlt`, `PMany(p:)`, `PFixed(v:)`). JSON: a string for nullary,
/// `{Ctor: {fields}}` for constructors; nested parsers recurse.
fn marshal_pparser(val: &Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        return Some(s.to_string()); // PDigit / PInt
    }
    let obj = val.as_object()?;
    let (ctor, fields) = obj.iter().next()?;
    match ctor.as_str() {
        "PChar" => {
            let w = fields.get("want")?.as_str()?;
            Some(format!("PChar(want: \"{w}\")"))
        }
        "PFixed" => {
            let v = fields.get("v")?.as_i64()?;
            Some(format!("PFixed(v: {v})"))
        }
        "PSeq" | "PAlt" => {
            let a = marshal_pparser(fields.get("a")?)?;
            let b = marshal_pparser(fields.get("b")?)?;
            Some(format!("{ctor}(a: {a}, b: {b})"))
        }
        "PMany" => {
            let p = marshal_pparser(fields.get("p")?)?;
            Some(format!("PMany(p: {p})"))
        }
        _ => None,
    }
}

/// A recursive sum type (kind): constructors, each with named typed fields.
#[derive(Clone)]
struct KindDef {
    ctors: HashMap<String, Vec<(String, String)>>,
}

/// The kind registry (PList, Jv, JsonDoc, Cbor, V, Val, Data, Body, …), keyed by
/// local and qualified name. Extracted from the stdlib kind definitions.
fn kind_registry() -> &'static HashMap<String, KindDef> {
    static REG: OnceLock<HashMap<String, KindDef>> = OnceLock::new();
    REG.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("./kind-registry.json"))
            .expect("kind registry parses");
        let mut m = HashMap::new();
        if let Some(obj) = v.as_object() {
            for (name, def) in obj {
                let mut ctors = HashMap::new();
                if let Some(cs) = def["ctors"].as_object() {
                    for (ctor, fields) in cs {
                        let fs: Vec<(String, String)> = fields.as_object()
                            .map(|f| f.iter().map(|(k, t)| (k.clone(), t.as_str().unwrap_or("").to_string())).collect())
                            .unwrap_or_default();
                        ctors.insert(ctor.clone(), fs);
                    }
                }
                let kd = KindDef { ctors };
                m.insert(name.clone(), kd.clone());
                if let Some(q) = def["qualified"].as_str() {
                    m.insert(q.to_string(), kd);
                }
            }
        }
        m
    })
}

/// Resolve a shape type to a registered kind, returning its local name.
fn lookup_kind(base: &str) -> Option<(String, &'static KindDef)> {
    let reg = kind_registry();
    if let Some(kd) = reg.get(base) {
        return Some((base.rsplit('.').next().unwrap_or(base).to_string(), kd));
    }
    None
}

/// Marshal a recursive kind term from JSON: a string names a nullary constructor
/// (`Nil`, `JsNull`); `{Ctor: {fields}}` builds a constructor, recursing into
/// fields (which may themselves be kinds or lists of kinds).
fn marshal_kind(base: &str, val: &Value) -> Option<String> {
    let (_, kd) = lookup_kind(base)?;
    if let Some(s) = val.as_str() {
        let fields = kd.ctors.get(s)?;
        if fields.is_empty() {
            return Some(s.to_string());
        }
        return None;
    }
    let obj = val.as_object()?;
    let (ctor, fields) = obj.iter().next()?;
    let ctor_fields = kd.ctors.get(ctor)?;
    let mut parts = Vec::new();
    for (fname, ftype) in ctor_fields {
        let fv = fields.get(fname)?;
        let lit = marshal_field(ftype, fv)?;
        parts.push(format!("{fname}: {lit}"));
    }
    Some(format!("{ctor}({})", parts.join(", ")))
}

/// Marshal a kind constructor field by its type: a list, a nested kind, or a
/// scalar/struct literal.
fn marshal_field(ftype: &str, val: &Value) -> Option<String> {
    if let Some(inner) = ftype.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        let arr = val.as_array()?;
        let mut lits = Vec::new();
        for v in arr {
            lits.push(marshal_field(inner, v)?);
        }
        return Some(format!("[{}]", lits.join(", ")));
    }
    if lookup_kind(ftype).is_some() {
        return marshal_kind(ftype, val);
    }
    marshal_lit(ftype, val)
}

/// Marshal a JSON value to a Form literal of the given base type.
fn marshal_lit(base: &str, val: &Value) -> Option<String> {
    if INT_TYPES.contains(&base) {
        if let Some(n) = val.as_i64() { return Some(n.to_string()); }
        if let Some(n) = val.as_u64() { return Some(n.to_string()); }
        return None;
    }
    if base == "f32" || base == "f64" {
        let x = val.as_f64()?;
        let s = format!("{x}");
        return Some(if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") });
    }
    if base == "bool" {
        return Some(val.as_bool()?.to_string());
    }
    if base == "text" {
        let s = val.as_str()?;
        let esc = s.replace('\\', "\\\\").replace('"', "\\\"");
        return Some(format!("\"{esc}\""));
    }
    let inner = base.strip_prefix("List[").and_then(|s| s.strip_suffix(']'))
        .or_else(|| base.strip_prefix('[').and_then(|s| s.strip_suffix(']')));
    if let Some(inner) = inner {
        let arr = val.as_array()?;
        let mut lits = Vec::new();
        for v in arr {
            lits.push(marshal_lit(inner, v)?);
        }
        return Some(format!("[{}]", lits.join(", ")));
    }
    // A defunctionalized Op tag marshals as its constructor name (e.g. Abs);
    // checked before the struct registry, which also lists Op.
    if base == "Op" {
        return val.as_str().map(|s| s.to_string());
    }
    if base == "PParser" {
        return marshal_pparser(val);
    }
    if lookup_kind(base).is_some() {
        return marshal_kind(base, val);
    }
    if let Some((local, def)) = lookup_struct(base) {
        let obj = val.as_object()?;
        let mut parts = Vec::new();
        for (fname, ftype) in &def.fields {
            let fv = obj.get(fname)?;
            let lit = marshal_lit(ftype, fv)?;
            parts.push(format!("{fname}: {lit}"));
        }
        return Some(format!("{local}({})", parts.join(", ")));
    }
    None
}

/// The state field type and initializer for a return type, or None if unsupported.
fn state_init(ret: &str) -> Option<(String, String)> {
    if ret == "()" {
        return None; // unit returns carry no value
    }
    // Error union `Ok ! Err`: the state holds the union, defaulted to the ok
    // value; the error type is brought into scope by `use <module>`.
    if let Some((ok, err)) = ret.split_once('!') {
        let (ok_ty, ok_init) = state_init(ok)?;
        return Some((format!("{ok_ty} ! {err}"), ok_init));
    }
    if let Some(inner) = ret.strip_suffix('?') {
        let t = event_type_of(inner)?;
        return Some((format!("Option[{t}]"), "none".into()));
    }
    if INT_TYPES.contains(&ret) {
        return Some((ret.to_string(), "0".into()));
    }
    if ret == "f32" || ret == "f64" {
        return Some((ret.to_string(), "0.0".into()));
    }
    if ret == "bool" {
        return Some(("bool".into(), "false".into()));
    }
    if ret == "text" {
        return Some(("text".into(), "\"\"".into()));
    }
    if let Some(inner) = ret.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        let t = event_type_of(inner)?;
        return Some((format!("List[{t}]"), "[]".into()));
    }
    if let Some((local, def)) = lookup_struct(ret) {
        let mut parts = Vec::new();
        for (fname, ftype) in &def.fields {
            let (_, init) = state_init(ftype)?;
            parts.push(format!("{fname}: {init}"));
        }
        return Some((local.clone(), format!("{local}({})", parts.join(", "))));
    }
    // Instantiated generic collections (the element types are already resolved).
    if ret.starts_with("Deque[") {
        return Some((ret.to_string(), "Deque(front: [], back: [])".into()));
    }
    if ret.starts_with("OrdMap[") {
        return Some((ret.to_string(), "OrdMap(keys: [], vals: [])".into()));
    }
    if ret.starts_with("HMap[") {
        return Some((ret.to_string(), "HMap(keys: [], vals: [], slots: [])".into()));
    }
    None
}

/// Infer a list's element type from its first JSON element.
fn infer_elem(arr: &Value) -> Option<String> {
    Some(infer_scalar(arr.as_array()?.first()?))
}

/// Marshal the generic collections whose layout is plain lists: Deque[T]
/// (`front`/`back`) and OrdMap[K,V] (`keys`/`vals`). Returns (event type, literal).
/// HMap is excluded — its `slots` hash table must be computed, not marshalled.
fn marshal_collection(base: &str, val: &Value) -> Option<(String, String)> {
    let name = base.split('[').next().unwrap_or("");
    match name {
        "Deque" => {
            let (front, back) = if let Some(o) = val.as_object() {
                (o.get("front")?.clone(), o.get("back").cloned().unwrap_or_else(|| json!([])))
            } else if val.is_array() {
                (val.clone(), json!([]))
            } else {
                return None;
            };
            let elem = infer_elem(&front).or_else(|| infer_elem(&back)).unwrap_or_else(|| "i64".into());
            let fl = marshal_lit(&format!("List[{elem}]"), &front)?;
            let bl = marshal_lit(&format!("List[{elem}]"), &back)?;
            Some((format!("Deque[{elem}]"), format!("Deque(front: {fl}, back: {bl})")))
        }
        "OrdMap" => {
            let o = val.as_object()?;
            let keys = o.get("keys")?;
            let vals = o.get("vals")?;
            let k = infer_elem(keys).unwrap_or_else(|| "i64".into());
            let v = infer_elem(vals).unwrap_or_else(|| "i64".into());
            let kl = marshal_lit(&format!("List[{k}]"), keys)?;
            let vl = marshal_lit(&format!("List[{v}]"), vals)?;
            Some((format!("OrdMap[{k}, {v}]"), format!("OrdMap(keys: {kl}, vals: {vl})")))
        }
        // An empty HMap; hm_insert rebuilds the `slots` hash table, so a chain of
        // inserts from empty is valid. (A populated HMap needs computed slots.)
        "HMap" => Some(("HMap[i64]".into(), "HMap(keys: [], vals: [], slots: [])".into())),
        _ => None,
    }
}

/// Parse the final state's `v = <value>` field as text (any type).
pub fn parse_v(stdout: &str) -> Option<String> {
    for line in stdout.lines().rev() {
        if let Some(idx) = line.find("v = ") {
            let rest = &line[idx + 4..];
            let end = rest.find(" }").unwrap_or(rest.len());
            return Some(rest[..end].trim().to_string());
        }
    }
    None
}

/// Parse the final `Out { v = <value> }` line of `rce run` output.
fn parse_state_value(stdout: &str) -> Option<f64> {
    for line in stdout.lines().rev() {
        if let Some(idx) = line.find("v = ") {
            let rest = &line[idx + 4..];
            let end = rest.find(" }").unwrap_or(rest.len());
            return rest[..end].trim().parse::<f64>().ok();
        }
    }
    None
}

/// Verify a competence by replay: run the caller and check it matches a known
/// vector. This is the gate's evidence (Authority::Replay) for a real record.
pub fn verify_math_unary(bridge: &RuntimeBridge, func: &str, x: f64, expected: f64) -> bool {
    match bridge.call_math_unary(func, x) {
        Ok(got) => got == expected,
        Err(_) => false,
    }
}

/// The workdir helper for tests/examples.
pub fn ensure_dir(p: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(p)
}

/// A gate checker that verifies a real competence by replay through the RCE
/// reference runtime: it re-runs the record's stdlib operation on the probe
/// input and compares the declared spectest output. Authority is `Replay`;
/// the trusted base is the reference runtime + the stdlib built into it.
pub struct RuntimeChecker {
    /// The runtime used to re-execute the spectest.
    pub bridge: RuntimeBridge,
}

impl Checker for RuntimeChecker {
    fn statement(&self) -> Statement {
        Statement {
            name: "rce-runtime-spectest".into(),
            property_class: "the stdlib operation reproduces its spectest via the reference runtime".into(),
            linking_assumption: "re-executed by the RCE reference checker on the admitted post-state".into(),
            trusted_base: "rce reference runtime + stdlib built into the checker".into(),
            authority: Authority::Replay,
            scope: Scope::Unit,
            version: "1".into(),
        }
    }

    fn check_unit(&self, _s: &State, _slot: &str, u: &Unit) -> Result<(), String> {
        let r = CompetenceRecord::decode(&u.body).ok_or_else(|| "malformed record".to_string())?;
        let x: f64 = r.probe_input.parse().map_err(|_| "probe_input not f64".to_string())?;
        let expected: f64 = r.probe_output.parse().map_err(|_| "probe_output not f64".to_string())?;
        let got = self.bridge.call_math_unary(&r.handler_kind, x)?;
        if got != expected {
            return Err(format!(
                "spectest failed: {}({}) = {}, declared {}",
                r.handler_kind, x, got, expected
            ));
        }
        Ok(())
    }
}

/// Per-record replay admission for *any* tractable operation. A crystallized
/// record names an operation (`handler_kind`), a spectest input (`probe_input`,
/// a positional JSON argument array) and a declared output (`probe_output`).
/// The checker re-executes the operation through the general runtime bridge and
/// compares: a record whose handler does not reproduce its declared spectest is
/// refused, not ranked — the negative-transfer guard. Authority is `Replay`;
/// the trusted base is the reference runtime + the stdlib built into it.
pub struct GeneralReplayChecker {
    /// The runtime used to re-execute the spectest.
    pub bridge: RuntimeBridge,
    /// operation -> (module, name, shape): the signature table to marshal against.
    pub signatures: HashMap<String, (String, String, String)>,
}

impl GeneralReplayChecker {
    /// Build a replay checker from a basis index (operation, module, name, shape).
    pub fn from_index(bridge: RuntimeBridge, entries: &[(String, String, String, String)]) -> Self {
        let signatures = entries.iter()
            .map(|(op, m, n, sh)| (op.clone(), (m.clone(), n.clone(), sh.clone())))
            .collect();
        GeneralReplayChecker { bridge, signatures }
    }
}

impl Checker for GeneralReplayChecker {
    fn statement(&self) -> Statement {
        Statement {
            name: "rce-general-replay-spectest".into(),
            property_class: "the crystallized operation reproduces its declared spectest via the reference runtime".into(),
            linking_assumption: "re-executed by the RCE reference checker on the admitted post-state".into(),
            trusted_base: "rce reference runtime + stdlib built into the checker".into(),
            authority: Authority::Replay,
            scope: Scope::Unit,
            version: "1".into(),
        }
    }

    fn check_unit(&self, _s: &State, _slot: &str, u: &Unit) -> Result<(), String> {
        let r = CompetenceRecord::decode(&u.body).ok_or_else(|| "malformed record".to_string())?;
        let (module, name, shape) = self.signatures.get(&r.handler_kind)
            .ok_or_else(|| format!("no signature for {:?}", r.handler_kind))?;
        let args: Value = serde_json::from_str(&r.probe_input)
            .map_err(|_| "probe_input not a JSON argument array".to_string())?;
        let got = self.bridge.call_general(module, name, shape, &args)?;
        if got != r.probe_output {
            return Err(format!(
                "spectest failed: {}({}) = {}, declared {}",
                r.handler_kind, r.probe_input, got, r.probe_output
            ));
        }
        Ok(())
    }
}

/// A shape-appropriate default JSON value for a base type, mirroring the marshal
/// acceptance, so a coverage sweep can exercise every marshalable operation.
/// `depth` bounds recursion on recursive kinds/structs (a cons cell must not
/// default to an infinite term); nullary constructors are preferred.
fn default_val(base: &str) -> Option<Value> {
    default_val_at(base, 0)
}

fn default_val_at(base: &str, depth: u32) -> Option<Value> {
    if INT_TYPES.contains(&base) { return Some(json!(2)); }
    if base == "f32" || base == "f64" { return Some(json!(2.0)); }
    if base == "bool" { return Some(json!(true)); }
    if base == "text" { return Some(json!("a")); }
    if is_generic_var(base) { return Some(json!(2)); }
    if let Some(inner) = base.strip_prefix("List[").and_then(|s| s.strip_suffix(']'))
        .or_else(|| base.strip_prefix('[').and_then(|s| s.strip_suffix(']'))) {
        if depth > 3 { return Some(json!([])); }
        let e = default_val_at(inner, depth + 1)?;
        return Some(json!([e.clone(), e]));
    }
    let name = base.split('[').next().unwrap_or("");
    match name {
        "Deque" => {
            let e = json!(2);
            return Some(json!({"front": [e.clone(), e], "back": []}));
        }
        "OrdMap" => {
            return Some(json!({"keys": [2, 3], "vals": [2, 3]}));
        }
        _ => {}
    }
    if base == "Op" { return Some(json!("Abs")); }
    if base == "PParser" { return Some(json!("PDigit")); }
    if let Some((_, def)) = lookup_kind(base) {
        // Prefer a nullary constructor (terminates recursion); else recurse bounded.
        if let Some((ctor, fields)) = def.ctors.iter().find(|(_, f)| f.is_empty()) {
            return Some(json!(ctor.clone()));
        }
        if depth > 3 { return None; }
        let (ctor, fields) = def.ctors.iter().next()?;
        let mut o = serde_json::Map::new();
        for (fname, ftype) in fields { o.insert(fname.clone(), default_val_at(ftype, depth + 1)?); }
        return Some(json!({ ctor.clone(): Value::Object(o) }));
    }
    if let Some((_, def)) = lookup_struct(base) {
        if depth > 3 { return None; }
        let mut o = serde_json::Map::new();
        for (fname, ftype) in &def.fields { o.insert(fname.clone(), default_val_at(ftype, depth + 1)?); }
        return Some(Value::Object(o));
    }
    None
}

/// Default positional arguments for a whole shape, or None if any arg is unmarshalable.
fn default_args_for_shape(shape: &str) -> Option<Value> {
    let (argstr, _ret) = split_shape(shape).ok()?;
    let argtypes = split_top_level(&argstr);
    let mut arr = Vec::new();
    for raw in &argtypes {
        let (base, _rk) = strip_ref(raw);
        arr.push(default_val(&base)?);
    }
    Some(Value::Array(arr))
}

impl RuntimeBridge {
    /// Sweep the whole basis: generate shape-appropriate args and execute each op.
    /// Returns (covered, returned_value, trapped_legitimately). An op is *covered*
    /// if it marshals and runs (a value or a real trap); the rest are refused with
    /// a reason. This is the live executor-coverage measurement (the ≥90% bar).
    pub fn sweep_coverage(&self, entries: &HashMap<String, (String, String, String)>) -> (usize, usize, usize) {
        let (mut covered, mut value, mut trap) = (0, 0, 0);
        for (_op, (module, name, shape)) in entries {
            let Some(args) = default_args_for_shape(shape) else { continue };
            match self.call_general(module, name, shape, &args) {
                Ok(_) => { covered += 1; value += 1; }
                Err(e) if e.starts_with("runtime refused") => { covered += 1; trap += 1; }
                Err(_) => {}
            }
        }
        (covered, value, trap)
    }
}