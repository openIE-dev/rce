//! The retrieval cascade: cheap gears first, stop at the first that answers.
//!
//! 1. `exact`: a Form def or fact subject/object equal to the query.
//! 2. `graph`: query tokens that name Form defs, plus their one-hop
//!    callers, callees and references, plus facts about those defs.
//! 3. `keyword`: BM25 over facts and defs.
//! 4. `vector`: only when a [`VectorIndex`] is installed and answers.
//! 5. not found: `escalate = true`, so MEI moves to a model gear.
//!
//! Every gear reports how many rows it touched; that is the telemetry
//! feature behind E_sim for the recall receipt.
use crate::bm25::{Bm25, tokenize};
use crate::fact::Fact;
use crate::form::{Def, FormIndex};
use crate::vector::VectorIndex;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gear {
    Exact,
    Graph,
    Keyword,
    Vector,
}

impl Gear {
    pub fn as_str(self) -> &'static str {
        match self {
            Gear::Exact => "exact",
            Gear::Graph => "graph",
            Gear::Keyword => "keyword",
            Gear::Vector => "vector",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub key: String,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub def: Option<Def>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact: Option<Fact>,
    /// For graph hits: how the def relates to the matched name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Trace {
    pub gear: Gear,
    pub touched: u64,
    pub hits: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recall {
    pub found: Option<Gear>,
    pub hits: Vec<Hit>,
    pub tried: Vec<Trace>,
    /// True when no gear answered: the caller should escalate.
    pub escalate: bool,
}

impl Recall {
    pub fn touched(&self) -> u64 {
        self.tried.iter().map(|t| t.touched).sum()
    }
}

fn def_hit(ix: &FormIndex, i: usize, score: f64, via: Option<String>) -> Hit {
    Hit {
        key: format!("def:{}", ix.defs[i].name),
        score,
        def: Some(ix.defs[i].clone()),
        fact: None,
        via,
    }
}

fn fact_hit(f: &Fact, score: f64) -> Hit {
    Hit {
        key: format!("fact:{}", f.id),
        score,
        def: None,
        fact: Some(f.clone()),
        via: None,
    }
}

/// Run the cascade. `facts` are already filtered by the caller's
/// bitemporal query and provenance policy.
pub fn recall(
    ix: &FormIndex,
    facts: &[Fact],
    vector: Option<&dyn VectorIndex>,
    query: &str,
    limit: usize,
) -> Recall {
    let q = query.trim();
    let limit = limit.max(1);
    let mut tried = vec![];
    let done = |gear, hits: Vec<Hit>, tried: Vec<Trace>| Recall {
        found: Some(gear),
        hits,
        tried,
        escalate: false,
    };

    // 1. exact
    let mut hits = vec![];
    for i in ix.find(q) {
        hits.push(def_hit(ix, i, 1.0, None));
    }
    for f in facts.iter().filter(|f| f.subject == q || f.object == q) {
        hits.push(fact_hit(f, 1.0));
    }
    hits.truncate(limit);
    tried.push(Trace {
        gear: Gear::Exact,
        touched: (ix.defs.len() + facts.len()) as u64,
        hits: hits.len(),
    });
    if !hits.is_empty() {
        return done(Gear::Exact, hits, tried);
    }

    // 2. graph walk, one hop from defs named in the query
    let mut touched = 0u64;
    let mut seeds = BTreeSet::new();
    for w in q.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
        if w.is_empty() {
            continue;
        }
        touched += ix.defs.len() as u64;
        seeds.extend(ix.find(w));
    }
    let mut seen = BTreeSet::new();
    for &s in &seeds {
        if seen.insert(s) {
            hits.push(def_hit(ix, s, 1.0, Some("named".into())));
        }
        for (rel, list) in [
            ("callee", ix.out_edges(s, true)),
            ("uses", ix.out_edges(s, false)),
            ("caller", ix.in_edges(s, true)),
            ("used_by", ix.in_edges(s, false)),
        ] {
            touched += ix.refs.len() as u64;
            for t in list {
                if seen.insert(t) {
                    hits.push(def_hit(
                        ix,
                        t,
                        0.5,
                        Some(format!("{rel} of {}", ix.defs[s].name)),
                    ));
                }
            }
        }
    }
    let names: BTreeSet<&str> = seen.iter().map(|&i| ix.defs[i].name.as_str()).collect();
    if !names.is_empty() {
        touched += facts.len() as u64;
        for f in facts.iter().filter(|f| names.contains(f.subject.as_str())) {
            hits.push(fact_hit(f, 0.75));
        }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.key.cmp(&b.key)));
    hits.truncate(limit);
    tried.push(Trace {
        gear: Gear::Graph,
        touched,
        hits: hits.len(),
    });
    if !hits.is_empty() {
        return done(Gear::Graph, hits, tried);
    }

    // 3. BM25 keyword
    if !tokenize(q).is_empty() {
        let docs = facts
            .iter()
            .map(|f| {
                (
                    format!("fact:{}", f.id),
                    format!("{} {} {} {}", f.subject, f.predicate, f.object, f.text),
                )
            })
            .chain(ix.defs.iter().map(|d| {
                (
                    format!("def:{}", d.name),
                    format!("{} {} {}", d.name, d.kind, d.file),
                )
            }));
        let bm = Bm25::new(docs);
        for (k, s) in bm.search(q, limit) {
            hits.push(resolve(ix, facts, &k, s));
        }
        tried.push(Trace {
            gear: Gear::Keyword,
            touched: bm.len() as u64,
            hits: hits.len(),
        });
        if !hits.is_empty() {
            return done(Gear::Keyword, hits, tried);
        }
    }

    // 4. vector, only if installed
    if let Some(v) = vector {
        let res = v.search(q, limit);
        if let Some(res) = res {
            for (k, s) in res {
                let h = resolve(ix, facts, &k, s as f64);
                if h.def.is_some() || h.fact.is_some() {
                    hits.push(h);
                }
            }
        }
        tried.push(Trace {
            gear: Gear::Vector,
            touched: hits.len() as u64,
            hits: hits.len(),
        });
        if !hits.is_empty() {
            return done(Gear::Vector, hits, tried);
        }
    }

    Recall {
        found: None,
        hits,
        tried,
        escalate: true,
    }
}

fn resolve(ix: &FormIndex, facts: &[Fact], key: &str, score: f64) -> Hit {
    if let Some(id) = key
        .strip_prefix("fact:")
        .and_then(|s| s.parse::<u64>().ok())
        && let Some(f) = facts.iter().find(|f| f.id == id)
    {
        return fact_hit(f, score);
    }
    if let Some(n) = key.strip_prefix("def:")
        && let Some(i) = ix.defs.iter().position(|d| d.name == n)
    {
        return def_hit(ix, i, score, None);
    }
    Hit {
        key: key.into(),
        score,
        def: None,
        fact: None,
        via: None,
    }
}
