//! Production JIT (seed): content-addressed native compile-cache + native execution.
//!
//! A baseline JIT: emit the Form to C (LUT), compile once with the system toolchain at -O2 to a
//! native executable, cache it by the name-free content key, and run the native binary. Recurring
//! competence compiles once and then executes native at the floor — the compile energy is paid once
//! and amortized, exactly the Law of Coding. This is a compile-cache / AOT-JIT hybrid; a tracing /
//! optimizing JIT (inline caches, tiering, deopt) is the open production frontier.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use crate::canonical::canonical_key;
use crate::lut_emit::{builtin_luts, emit_expr};

pub struct Jit {
    cache: HashMap<String, PathBuf>,
    dir: PathBuf,
    pub compiles: u64,
    pub hits: u64,
    hot: HashMap<String, u64>,
    pub threshold: u64,
}

impl Jit {
    pub fn new() -> Self {
        // Per-instance dir: parallel tests/instances must not race on the same exe path.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("rce-jit-cache-{n}"));
        let _ = std::fs::create_dir_all(&dir);
        Jit { cache: HashMap::new(), dir, compiles: 0, hits: 0, hot: HashMap::new(), threshold: 3 }
    }

    /// Tiered execution: interpret while cold, compile to native once a Form is hot. This is the
    /// seed of a tracing/tiering JIT — pay compile energy only for recurring (hot) competence.
    pub fn run_tiered(&mut self, src: &str) -> Result<String, String> {
        let key = canonical_key(src)?;
        let n = self.hot.entry(key.clone()).or_insert(0);
        *n += 1;
        if *n <= self.threshold && !self.cache.contains_key(&key) {
            // cold tier: interpret at the floor, no compile cost
            let e = crate::lift::lift_expr(src)?;
            return crate::repair::eval_expr(&e).map(|v| v.to_string())
                .ok_or_else(|| "not a closed form".into());
        }
        self.run(src)
    }

    /// Compile (once, cached) and run the Form natively; return the printed result.
    pub fn run(&mut self, src: &str) -> Result<String, String> {
        let key = canonical_key(src)?;
        let exe = if let Some(p) = self.cache.get(&key) {
            self.hits += 1;
            p.clone()
        } else {
            let p = self.compile(src, &key)?;
            self.compiles += 1;
            self.cache.insert(key, p.clone());
            p
        };
        let out = Command::new(&exe).output().map_err(|e| format!("run: {e}"))?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    fn compile(&self, src: &str, key: &str) -> Result<PathBuf, String> {
        let c_lut = builtin_luts().into_iter().find(|l| l.name == "c")
            .ok_or_else(|| "no C LUT".to_string())?;
        let expr = emit_expr(src, &c_lut)?;
        let code = format!(
            "#include <stdio.h>\nint main(void){{ printf(\"%lld\\n\", (long long)({expr})); return 0; }}\n"
        );
        let cpath = self.dir.join(format!("{key}.c"));
        let exe = self.dir.join(key);
        std::fs::write(&cpath, code).map_err(|e| format!("write: {e}"))?;
        let st = Command::new("clang").arg("-O2").arg(&cpath).arg("-o").arg(&exe)
            .output().map_err(|e| format!("clang: {e}"))?;
        if !st.status.success() {
            return Err(format!("compile failed: {}", String::from_utf8_lossy(&st.stderr)));
        }
        Ok(exe)
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

/// Deadline-aware tiering: a cold Form is interpreted NOW (inside the control loop's
/// deadline) and its compile is queued for a maintenance slot OFF the hot path. A
/// compile spike never blocks a control tick — the ros2_control catch-up failure mode
/// answered by construction.
pub struct DeadlineJit {
    pub jit: Jit,
    pending: Vec<String>,
    pub offloop_compiles: u64,
}

impl DeadlineJit {
    pub fn new() -> Self {
        DeadlineJit { jit: Jit::new(), pending: Vec::new(), offloop_compiles: 0 }
    }

    /// Serve one act: cached native if hot, else interpret now and queue the compile.
    /// Returns (result, served_native). Never compiles inside this call.
    pub fn serve(&mut self, src: &str) -> Result<(String, bool), String> {
        let key = canonical_key(src)?;
        if self.jit.cache.contains_key(&key) {
            let out = self.jit.run(src)?;
            return Ok((out, true));
        }
        let e = crate::lift::lift_expr(src)?;
        let v = crate::repair::eval_expr(&e)
            .map(|v| v.to_string())
            .ok_or_else(|| "not a closed form".to_string())?;
        if !self.pending.iter().any(|p| p == src) {
            self.pending.push(src.to_string());
        }
        Ok((v, false))
    }

    /// One maintenance slot off the hot path: compile at most one queued Form.
    pub fn maintenance(&mut self) -> Result<bool, String> {
        match self.pending.pop() {
            Some(src) => {
                self.jit.run(&src)?;
                self.offloop_compiles += 1;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

impl Default for DeadlineJit {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracing-JIT seed: trace hot subtrees, compile each distinct hot subtree to native ONCE
/// (content-keyed guard), reuse the native helper afterwards; deopt to the interpreter when
/// a guard/compile fails. Compile energy is paid per distinct hot subtree, amortized over
/// every occurrence — the tracing/tiering frontier in seed form.
pub struct TracingJit {
    pub jit: Jit,
    hot: HashMap<String, u64>,
    pub threshold: u64,
    pub deopts: u64,
    pub subtree_compiles: u64,
}

impl TracingJit {
    pub fn new() -> Self {
        TracingJit { jit: Jit::new(), hot: HashMap::new(), threshold: 2, deopts: 0, subtree_compiles: 0 }
    }

    /// Run a Form; returns (result, used_native). Guard misses deopt, never wrong answers.
    pub fn run(&mut self, src: &str) -> Result<(String, bool), String> {
        let e = crate::lift::lift_expr(src)?;
        let (mut native, mut deopt) = (false, false);
        let v = self.eval_traced(&e, &mut native, &mut deopt);
        if deopt {
            self.deopts += 1;
        }
        match v {
            Some(v) => Ok((v.to_string(), native)),
            None => Err("not a closed form".into()),
        }
    }

    fn eval_traced(&mut self, e: &rce_spec_check::ast::Expr, native: &mut bool, deopt: &mut bool) -> Option<i128> {
        let rendered = crate::canonical::emit(e);
        let key = match canonical_key(&rendered) {
            Ok(k) => k,
            Err(_) => {
                *deopt = true;
                return crate::repair::eval_expr(e);
            }
        };
        let n = self.hot.entry(key.clone()).or_insert(0);
        *n += 1;
        if *n >= self.threshold {
            if self.jit.cache.contains_key(&key) {
                if let Ok(out) = self.jit.run(&rendered) {
                    if let Ok(v) = out.parse::<i128>() {
                        *native = true;
                        return Some(v);
                    }
                }
                *deopt = true;
                return crate::repair::eval_expr(e);
            }
            return match self.jit.run(&rendered) {
                Ok(out) => {
                    self.subtree_compiles += 1;
                    match out.parse::<i128>() {
                        Ok(v) => {
                            *native = true;
                            Some(v)
                        }
                        Err(_) => {
                            *deopt = true;
                            crate::repair::eval_expr(e)
                        }
                    }
                }
                Err(_) => {
                    *deopt = true;
                    crate::repair::eval_expr(e)
                }
            };
        }
        use rce_spec_check::ast::Expr::*;
        match e {
            Int(v, _) => Some(*v),
            Neg(a, _) => Some(-self.eval_traced(a, native, deopt)?),
            Bin(op, a, b, _) => {
                let x = self.eval_traced(a, native, deopt)?;
                let y = self.eval_traced(b, native, deopt)?;
                match op.as_str() {
                    "+" => Some(x + y),
                    "-" => Some(x - y),
                    "*" => Some(x * y),
                    "/" => if y != 0 { Some(x / y) } else { Option::None },
                    "%" => if y != 0 { Some(x % y) } else { Option::None },
                    _ => crate::repair::eval_expr(e),
                }
            }
            _ => crate::repair::eval_expr(e),
        }
    }
}

impl Default for TracingJit {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed runtime value for type-feedback specialization.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Int(i128),
    Real(f64),
    Bool(bool),
}

fn eval_val(e: &rce_spec_check::ast::Expr, env: &std::collections::BTreeMap<String, Val>) -> Option<Val> {
    use rce_spec_check::ast::Expr::*;
    match e {
        Int(v, _) => Some(Val::Int(*v)),
        Name(n, _) => env.get(n).cloned(),
        Neg(a, _) => match eval_val(a, env)? {
            Val::Int(v) => Some(Val::Int(-v)),
            Val::Real(v) => Some(Val::Real(-v)),
            _ => Option::None,
        },
        Bin(op, a, b, _) => {
            let x = eval_val(a, env)?;
            let y = eval_val(b, env)?;
            let real = matches!(x, Val::Real(_)) || matches!(y, Val::Real(_));
            let xf = match &x { Val::Int(v) => *v as f64, Val::Real(v) => *v, Val::Bool(b) => *b as i128 as f64 };
            let yf = match &y { Val::Int(v) => *v as f64, Val::Real(v) => *v, Val::Bool(b) => *b as i128 as f64 };
            let xi = match &x { Val::Int(v) => *v, Val::Bool(b) => *b as i128, _ => 0 };
            let yi = match &y { Val::Int(v) => *v, Val::Bool(b) => *b as i128, _ => 0 };
            match op.as_str() {
                "+" | "-" | "*" | "/" | "%" => {
                    if real {
                        let r = match op.as_str() {
                            "+" => xf + yf, "-" => xf - yf, "*" => xf * yf,
                            "/" => if yf != 0.0 { xf / yf } else { return Option::None },
                            _ => if yf != 0.0 { xf % yf } else { return Option::None },
                        };
                        Some(Val::Real(r))
                    } else {
                        let r = match op.as_str() {
                            "+" => xi + yi, "-" => xi - yi, "*" => xi * yi,
                            "/" => if yi != 0 { xi / yi } else { return Option::None },
                            _ => if yi != 0 { xi % yi } else { return Option::None },
                        };
                        Some(Val::Int(r))
                    }
                }
                "==" => Some(Val::Bool(xf == yf)),
                "!=" => Some(Val::Bool(xf != yf)),
                "<" => Some(Val::Bool(xf < yf)),
                ">" => Some(Val::Bool(xf > yf)),
                "<=" => Some(Val::Bool(xf <= yf)),
                ">=" => Some(Val::Bool(xf >= yf)),
                "&&" => Some(Val::Bool(xf != 0.0 && yf != 0.0)),
                "||" => Some(Val::Bool(xf != 0.0 || yf != 0.0)),
                _ => Option::None,
            }
        }
        _ => Option::None,
    }
}

/// Type-feedback optimizing JIT: observe argument types per call site; after a threshold,
/// compile a native kernel specialized to that signature; a later call with a DIFFERENT
/// signature deopts to the interpreter (never a wrong answer).
pub struct OptJit {
    pub jit: Jit,
    hits: HashMap<String, u64>,
    spec: HashMap<String, std::path::PathBuf>,
    pub deopts: u64,
    pub specs: u64,
    pub threshold: u64,
    dir: std::path::PathBuf,
}

impl OptJit {
    pub fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("rce-opt-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        OptJit { jit: Jit::new(), hits: HashMap::new(), spec: HashMap::new(), deopts: 0, specs: 0, threshold: 2, dir }
    }

    fn sig(args: &[(String, Val)]) -> String {
        args.iter()
            .map(|(n, v)| format!("{}:{}", n, match v { Val::Int(_) => "i64", Val::Real(_) => "f64", Val::Bool(_) => "bool" }))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn run(&mut self, src: &str, args: &[(String, Val)]) -> Result<(String, bool), String> {
        let e = crate::lift::lift_expr(src)?;
        let base = crate::canonical::canonical_key(src)?;
        let sig = Self::sig(args);
        let key = format!("{base}|{sig}");
        let n = self.hits.entry(key.clone()).or_insert(0);
        *n += 1;
        if *n >= self.threshold {
            let exe = if let Some(p) = self.spec.get(&key) {
                p.clone()
            } else {
                let p = self.compile(&e, args, &key)?;
                self.specs += 1;
                self.spec.insert(key.clone(), p.clone());
                p
            };
            let mut cmd = std::process::Command::new(&exe);
            for (_, v) in args {
                cmd.arg(match v { Val::Int(i) => i.to_string(), Val::Real(f) => f.to_string(), Val::Bool(b) => (*b as i128).to_string() });
            }
            let out = cmd.output().map_err(|e| e.to_string())?;
            if !out.status.success() {
                return Err(String::from_utf8_lossy(&out.stderr).into_owned());
            }
            return Ok((String::from_utf8_lossy(&out.stdout).trim().to_string(), true));
        }
        // cold path: interpret; if a specialization exists for another signature, this is a deopt
        if self.spec.keys().any(|k| k.starts_with(&format!("{base}|")) && k != &key) {
            self.deopts += 1;
        }
        let env: std::collections::BTreeMap<String, Val> = args.iter().cloned().collect();
        match eval_val(&e, &env) {
            Some(Val::Int(i)) => Ok((i.to_string(), false)),
            Some(Val::Real(f)) => Ok((f.to_string(), false)),
            Some(Val::Bool(b)) => Ok(((b as i128).to_string(), false)),
            Option::None => Err("not a closed form under args".into()),
        }
    }

    fn compile(&self, e: &rce_spec_check::ast::Expr, args: &[(String, Val)], key: &str) -> Result<std::path::PathBuf, String> {
        let real = args.iter().any(|(_, v)| matches!(v, Val::Real(_)));
        let ty = if real { "double" } else { "long long" };
        let parse = if real { "atof" } else { "atoll" };
        let mut c = String::from("#include <stdio.h>\n#include <stdlib.h>\nint main(int argc, char** argv){\n");
        for (i, (n, _)) in args.iter().enumerate() {
            c.push_str(&format!("  {ty} {n} = ({ty}){parse}(argv[{}]);\n", i + 1));
        }
        let body = crate::canonical::emit(e);
        c.push_str(&format!("  printf(\"%.17g\\n\", (double)({body}));\n  return 0;\n}}\n"));
        let safe: String = key.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
        let csrc = self.dir.join(format!("{safe}.c"));
        let exe = self.dir.join(safe);
        std::fs::write(&csrc, c).map_err(|e| e.to_string())?;
        let cc = std::process::Command::new("clang").arg("-O2").arg(&csrc).arg("-o").arg(&exe).output().map_err(|e| e.to_string())?;
        if !cc.status.success() {
            return Err(String::from_utf8_lossy(&cc.stderr).into_owned());
        }
        Ok(exe)
    }
}

impl Default for OptJit {
    fn default() -> Self {
        Self::new()
    }
}