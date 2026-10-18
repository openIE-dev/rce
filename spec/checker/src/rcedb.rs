//! RCEdb v1 (W3): a serverless, embedded, in-process database whose
//! storage and queries are the RCE module `crates/rce/rcedb/rcedb.rce`,
//! run on the RCE engine over the host's durable log D (Appendix E.8).
//!
//! Rules (R1-R8, reconstructed from Appendix E.8 and E.3; see
//! crates/rce/REMAINING-PLAN.md):
//! - R1 D is the only primary artefact (HS-1); a database is a log file.
//! - R2 Tables, rows and edges are RCE state, changed only by events
//!   admitted to D (HC-1); a refused write leaves state unchanged (T8).
//! - R3 Writes are durable by group confirm: one barrier per batch, and the
//!   durable mark moves only through the bound `Synced` (HS-2, HS-6).
//! - R4 Queries are RCE steps evaluated on state without appending to D;
//!   a step that would change state is not a query (HS-8).
//! - R5 Results are in canonical key order, never unspecified (HS-8, 6.7).
//! - R6 As-of reads are the same query over a prefix fold of D (HS-9).
//! - R7 Snapshots, indexes and every emitted store are caches or
//!   projections of D, checked by the state hash they carry (HS-5, HS-7).
//! - R8 Foreign engines (SQLite, Postgres, DuckDB, a graph store) are
//!   emit targets only: their durability or ordering never reaches δ.
use crate::ast::{Arg, Expr, Module};
use crate::host::{self, Host};
use crate::interp::V;
use crate::parse::parse;
use std::path::{Path, PathBuf};

/// The RCEdb v0 module.
pub const SRC: &str = include_str!("../../../rcedb/rcedb.rce");

/// A target for [`Db::emit`] (R8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dialect {
    Sqlite,
    Postgres,
    DuckDb,
    /// openCypher `MERGE` statements for a property-graph store.
    Cypher,
}

pub struct Db {
    path: PathBuf,
    h: Host<'static>,
    m: &'static Module,
}

fn module() -> Result<&'static Module, String> {
    use std::sync::OnceLock;
    static M: OnceLock<Result<Module, String>> = OnceLock::new();
    M.get_or_init(|| parse(SRC).map_err(|e| format!("rcedb.rce: {e}"))).as_ref().map_err(|e| e.clone())
}

fn args(kv: &[(&str, &str)]) -> Vec<Arg> {
    kv.iter().map(|(k, v)| Arg { label: Some((*k).into()), e: Expr::Str((*v).into(), 0) }).collect()
}

fn field<'v>(fs: &'v [(String, V)], n: &str) -> &'v str {
    fs.iter().find_map(|(f, v)| match v {
        V::Text(t) if f == n => Some(t.as_str()),
        _ => None,
    }).unwrap_or("")
}

/// One row: (table, key, row text).
pub type Row = (String, String, String);
/// One edge: (src, label, dst).
pub type Edge = (String, String, String);

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}
fn sql_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
fn cy_str(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

impl Db {
    /// Open (or create) the database stored as the log at `path` (R1):
    /// verify on open, cut a torn tail, refold (HS-4, HC-8).
    pub fn open(path: &Path) -> Result<Db, String> {
        let m = module()?;
        let h = Host::open_file(m, path)?;
        Ok(Db { path: path.to_path_buf(), h, m })
    }

    fn write(&mut self, ev: &str, kv: &[(&str, &str)]) -> Result<(), String> {
        let out = self.h.admit_event(ev, args(kv))?;
        match &out[0].outcome {
            Ok(_) => Ok(()),
            Err(r) => Err(r.clone()),
        }
    }

    pub fn insert(&mut self, table: &str, key: &str, row: &str) -> Result<(), String> {
        self.write("Insert", &[("table", table), ("key", key), ("row", row)])
    }
    pub fn delete(&mut self, table: &str, key: &str) -> Result<(), String> {
        self.write("Delete", &[("table", table), ("key", key)])
    }
    pub fn link(&mut self, src: &str, label: &str, dst: &str) -> Result<(), String> {
        self.write("Link", &[("src", src), ("lbl", label), ("dst", dst)])
    }
    pub fn unlink(&mut self, src: &str, label: &str, dst: &str) -> Result<(), String> {
        self.write("Unlink", &[("src", src), ("lbl", label), ("dst", dst)])
    }

    /// Give `table` a typed schema, "name:type,..." with type text, i64 or
    /// bool; refused once the table has rows. Every later row of the table
    /// is its values in column order separated by tabs, and a row that does
    /// not match is refused (BadRow).
    pub fn schema(&mut self, table: &str, cols: &str) -> Result<(), String> {
        self.write("Schema", &[("table", table), ("cols", cols)])
    }
    /// A secondary index on a schema column: a fold of D kept in state,
    /// backfilled from the rows already there (R7).
    pub fn index(&mut self, table: &str, col: &str) -> Result<(), String> {
        self.write("Index", &[("table", table), ("col", col)])
    }
    /// Rows of `table` whose `col` equals `value`, in key order; through the
    /// index when there is one, else by scan, with the same answer.
    pub fn lookup(&mut self, table: &str, col: &str, value: &str) -> Result<Vec<Row>, String> {
        Ok(Db::rows_of(self.h.query("Lookup", args(&[("table", table), ("col", col), ("value", value)]))?).0)
    }
    /// Each table: (name, rows, schema or "", indexed columns comma-joined).
    pub fn describe(&mut self) -> Result<Vec<(String, u64, String, String)>, String> {
        let mut v = vec![];
        for o in self.h.query("Describe", vec![])? {
            if let V::Rec(n, fs) = o {
                if n == "Table" {
                    let rows = fs.iter().find_map(|(f, v)| match v {
                        V::Int(i, _) if f == "rows" => Some(*i as u64),
                        _ => None,
                    }).unwrap_or(0);
                    v.push((field(&fs, "table").into(), rows, field(&fs, "cols").into(), field(&fs, "indexes").into()));
                }
            }
        }
        Ok(v)
    }

    fn rows_of(outs: Vec<V>) -> (Vec<Row>, Vec<Edge>) {
        let (mut rows, mut edges) = (vec![], vec![]);
        for o in outs {
            match o {
                V::Rec(n, fs) if n == "Row" => rows.push((field(&fs, "table").into(), field(&fs, "key").into(), field(&fs, "row").into())),
                V::Rec(n, fs) if n == "Edge" => edges.push((field(&fs, "src").into(), field(&fs, "lbl").into(), field(&fs, "dst").into())),
                _ => {}
            }
        }
        (rows, edges)
    }

    /// Rows of `table` with key >= `start`, in canonical key order (R4, R5).
    pub fn scan(&mut self, table: &str, start: &str) -> Result<Vec<Row>, String> {
        Ok(Db::rows_of(self.h.query("Scan", args(&[("table", table), ("start", start)]))?).0)
    }
    /// Outgoing edges of `src`, in canonical order.
    pub fn edges(&mut self, src: &str) -> Result<Vec<Edge>, String> {
        Ok(Db::rows_of(self.h.query("Edges", args(&[("src", src)]))?).1)
    }
    /// Every row and edge, in canonical order.
    pub fn dump(&mut self) -> Result<(Vec<Row>, Vec<Edge>), String> {
        Ok(Db::rows_of(self.h.query("Dump", vec![])?))
    }

    /// The number of entries of D and the state hash: the provenance every
    /// projection carries (R7).
    pub fn seq(&self) -> u64 {
        self.h.len()
    }
    pub fn state_hash(&self) -> &str {
        self.h.state_hash()
    }
    pub fn durable(&self) -> Option<u64> {
        self.h.durable
    }

    /// The database as of entry k of D (R6), read only.
    pub fn as_of(&self, k: usize) -> Result<AsOf, String> {
        let text = std::fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        Ok(AsOf { h: host::as_of(self.m, &text, k)? })
    }

    /// The current state as a script for a foreign engine (R8): one
    /// transaction, rows and edges in canonical order, headed by the seq
    /// and state hash it projects.
    pub fn emit(&mut self, d: Dialect) -> Result<String, String> {
        let (rows, edges) = self.dump()?;
        Ok(emit(d, &rows, &edges, self.seq(), self.state_hash()))
    }
}

/// A read-only past state (R6).
pub struct AsOf {
    h: Host<'static>,
}

impl AsOf {
    pub fn scan(&mut self, table: &str, start: &str) -> Result<Vec<Row>, String> {
        Ok(Db::rows_of(self.h.query("Scan", args(&[("table", table), ("start", start)]))?).0)
    }
    pub fn state_hash(&self) -> &str {
        self.h.state_hash()
    }
}

/// Render rows and edges for a dialect.
pub fn emit(d: Dialect, rows: &[Row], edges: &[Edge], seq: u64, state_hash: &str) -> String {
    let mut s = String::new();
    let c = if d == Dialect::Cypher { "//" } else { "--" };
    s.push_str(&format!("{c} RCEdb projection of D: entries {seq}, state_hash {state_hash}\n"));
    if d == Dialect::Cypher {
        for (t, k, r) in rows {
            s.push_str(&format!("MERGE (n:Row {{table: {}, key: {}}}) SET n.row = {};\n", cy_str(t), cy_str(k), cy_str(r)));
        }
        for (a, l, b) in edges {
            s.push_str(&format!("MERGE (a:Node {{id: {}}}) MERGE (b:Node {{id: {}}}) MERGE (a)-[:`{}`]->(b);\n", cy_str(a), cy_str(b), l.replace('`', "``")));
        }
        return s;
    }
    let jt = match d {
        Dialect::Postgres => "JSONB",
        Dialect::DuckDb => "JSON",
        _ => "TEXT",
    };
    s.push_str("BEGIN;\n");
    let mut tables: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
    tables.dedup();
    for t in &tables {
        s.push_str(&format!("CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, row {jt} NOT NULL);\n", sql_ident(t)));
    }
    for (t, k, r) in rows {
        s.push_str(&format!("INSERT INTO {} (key, row) VALUES ({}, {});\n", sql_ident(t), sql_str(k), sql_str(r)));
    }
    s.push_str("CREATE TABLE IF NOT EXISTS \"_edges\" (src TEXT NOT NULL, label TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY (src, label, dst));\n");
    for (a, l, b) in edges {
        s.push_str(&format!("INSERT INTO \"_edges\" (src, label, dst) VALUES ({}, {}, {});\n", sql_str(a), sql_str(l), sql_str(b)));
    }
    s.push_str("COMMIT;\n");
    s
}
