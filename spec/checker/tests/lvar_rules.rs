//! The link from the checker's LVar rules to the Lean model: the rule table
//! `conc::LVAR_RULES` and the certificates W-CRDT-1 derives are written out
//! as lean/RCE/LVarRules.lean, which `lake build` elaborates (a row naming a
//! missing declaration, or a certificate that does not denote a lawful
//! join, fails the build). This test keeps the committed file equal to the
//! generated one and checks each trap string against every runtime.
#![cfg(feature = "conc06")]
use rce_spec_check::conc::{LVAR_RULES, merge_certs, lvar_rules_lean};
use rce_spec_check::parse::parse;

pub const PROGRAMS: &[&str] = &[
    "struct Hw\n    hi: i64\n    seen: bool\n\nimpl Merge for Hw\n    fn merge(self, other: Hw) -> Hw\n        Hw(hi: self.hi.max(other.hi), seen: self.seen or other.seen)\n    fn bottom() -> Hw\n        Hw(hi: 0, seen: false)\n",
    "struct Lo\n    lo: i64\n    all: bool\n\nimpl Merge for Lo\n    fn merge(self, other: Lo) -> Lo\n        Lo(lo: self.lo.min(other.lo), all: self.all and other.all)\n    fn bottom() -> Lo\n        Lo(lo: 1000, all: true)\n\nstruct Pair\n    a: Lo\n    n: i64\n\nimpl Merge for Pair\n    fn merge(self, other: Pair) -> Pair\n        Pair(a: self.a.merge(other.a), n: self.n.max(other.n))\n    fn bottom() -> Pair\n        Pair(a: Lo(lo: 1000, all: true), n: 0)\n",
    "struct Ver\n    ts: i64\n    hi: i64\n    ok: bool\n\nimpl Merge for Ver\n    fn merge(self, other: Ver) -> Ver\n        if self.ts > other.ts\n            self\n        elif other.ts > self.ts\n            other\n        else\n            Ver(ts: self.ts, hi: self.hi.max(other.hi), ok: self.ok or other.ok)\n    fn bottom() -> Ver\n        Ver(ts: 0, hi: 0, ok: false)\n",
    "struct Reg\n    seen: Set[i64]\n    common: Set[i64]\n    hi: Map[text, i64]\n    tags: Map[text, Set[i64]]\n\nimpl Merge for Reg\n    fn merge(self, other: Reg) -> Reg\n        Reg(seen: self.seen.union(other.seen), common: self.common.inter(other.common), hi: self.hi.merge(other.hi), tags: self.tags.merge(other.tags))\n    fn bottom() -> Reg\n        Reg(seen: set_of([]), common: set_of([1]), hi: {\"z\": 0}, tags: {\"z\": set_of([])})\n",
    "struct Hi\n    v: i64\n\nimpl Merge for Hi\n    fn merge(self, other: Hi) -> Hi\n        if other.v > self.v\n            return other\n        self\n    fn bottom() -> Hi\n        Hi(v: 0)\n",
];

pub fn certs() -> Vec<(String, String)> {
    let mut out = vec![];
    for p in PROGRAMS {
        let m = parse(p).unwrap_or_else(|e| panic!("{e}\n{p}"));
        let cs = merge_certs(&m);
        let mut names: Vec<_> = cs.keys().cloned().collect();
        names.sort();
        for n in names {
            let env = |t: &str| cs.get(t).cloned();
            out.push((n.clone(), cs[&n].to_lean(&env)));
        }
    }
    out
}

fn lean_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lean/RCE/LVarRules.lean")
}

#[test]
fn every_test_program_is_certified() {
    let c = certs();
    let names: Vec<_> = c.iter().map(|x| x.0.as_str()).collect();
    assert_eq!(names, ["Hw", "Lo", "Pair", "Ver", "Reg", "Hi"], "{c:?}");
    assert_eq!(c[0].1, "(.prod .maxI .orB)");
    assert_eq!(c[2].1, "(.prod (.prod .minI .andB) .maxI)");
    assert_eq!(c[3].1, "(.lex (.prod .maxI .orB))");
    assert_eq!(c[4].1, "(.prod .union (.prod .inter (.prod (.map .maxI) (.map .union))))");
    assert_eq!(c[5].1, ".maxI");
}

#[test]
fn lean_rule_table_is_current() {
    let want = lvar_rules_lean(&certs());
    let p = lean_path();
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&p, &want).unwrap();
    }
    let have = std::fs::read_to_string(&p).unwrap_or_default();
    assert_eq!(have, want, "lean/RCE/LVarRules.lean is stale: rerun with RCE_BLESS=1, then lake build");
    let root = std::fs::read_to_string(lean_path().parent().unwrap().parent().unwrap().join("RCE.lean")).unwrap();
    assert!(root.contains("import RCE.LVarRules"), "RCE.lean must import RCE.LVarRules so lake build checks it");
}

#[test]
fn every_trap_is_raised_by_every_runtime() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let srcs = ["conc.rs", "emit/rt.rs", "emit/rt.py", "emit/rt.js", "emit/rt.scala"];
    for (rule, trap, _) in LVAR_RULES {
        let Some(t) = trap else { continue };
        for f in srcs {
            let s = std::fs::read_to_string(dir.join(f)).unwrap();
            assert!(s.contains(&format!("\"{t}\"")), "{f} does not raise `{t}` ({rule})");
        }
    }
}
