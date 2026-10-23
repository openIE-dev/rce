//! Default backend: embedded redb (pure Rust, one file, no server).
//! Rows are JSON in table `facts` (id -> bytes); the id counter lives in
//! `meta`. A plan is one write transaction.
use crate::engine::{Health, MemError, MemoryEngine, Plan, Result, check_plan};
use crate::fact::Fact;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;

const FACTS: TableDefinition<u64, &[u8]> = TableDefinition::new("facts");
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

fn se(e: impl std::fmt::Display) -> MemError {
    MemError::Storage(e.to_string())
}

pub struct RedbEngine {
    db: Database,
}

impl RedbEngine {
    /// Open or create the database file.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path).map_err(se)?;
        let w = db.begin_write().map_err(se)?;
        {
            w.open_table(FACTS).map_err(se)?;
            w.open_table(META).map_err(se)?;
        }
        w.commit().map_err(se)?;
        Ok(RedbEngine { db })
    }
}

fn decode(b: &[u8]) -> Result<Fact> {
    serde_json::from_slice(b).map_err(se)
}

impl MemoryEngine for RedbEngine {
    fn next_id(&mut self) -> Result<u64> {
        let w = self.db.begin_write().map_err(se)?;
        let id = {
            let mut t = w.open_table(META).map_err(se)?;
            let cur = t
                .get("next_id")
                .map_err(se)?
                .map(|v| v.value())
                .unwrap_or(0);
            t.insert("next_id", cur + 1).map_err(se)?;
            cur + 1
        };
        w.commit().map_err(se)?;
        Ok(id)
    }

    fn commit(&mut self, plan: &Plan) -> Result<()> {
        let w = self.db.begin_write().map_err(se)?;
        {
            let mut t = w.open_table(FACTS).map_err(se)?;
            let get = |t: &redb::Table<u64, &[u8]>, id: u64| -> Result<Option<Fact>> {
                match t.get(id).map_err(se)? {
                    Some(v) => Ok(Some(decode(v.value())?)),
                    None => Ok(None),
                }
            };
            check_plan(|id| get(&t, id), plan)?;
            for &(id, at) in &plan.expires {
                let mut f = get(&t, id)?.ok_or(MemError::NotFound(id))?;
                f.expired_at = Some(at);
                let b = serde_json::to_vec(&f).map_err(se)?;
                t.insert(id, b.as_slice()).map_err(se)?;
            }
            for f in &plan.inserts {
                let b = serde_json::to_vec(f).map_err(se)?;
                t.insert(f.id, b.as_slice()).map_err(se)?;
            }
        }
        // Dropping `w` without commit aborts on any error above.
        w.commit().map_err(se)
    }

    fn get(&self, id: u64) -> Result<Option<Fact>> {
        let r = self.db.begin_read().map_err(se)?;
        let t = r.open_table(FACTS).map_err(se)?;
        match t.get(id).map_err(se)? {
            Some(v) => Ok(Some(decode(v.value())?)),
            None => Ok(None),
        }
    }

    fn scan(&self) -> Result<Vec<Fact>> {
        let r = self.db.begin_read().map_err(se)?;
        let t = r.open_table(FACTS).map_err(se)?;
        let mut out = vec![];
        for kv in t.iter().map_err(se)? {
            let (_, v) = kv.map_err(se)?;
            out.push(decode(v.value())?);
        }
        Ok(out)
    }

    fn health(&self) -> Result<Health> {
        let r = self.db.begin_read().map_err(se)?;
        let t = r.open_table(FACTS).map_err(se)?;
        Ok(Health {
            backend: "redb",
            rows: t.len().map_err(se)?,
        })
    }
}
