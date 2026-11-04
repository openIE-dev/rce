//! JouleDB-backed graph store.
//!
//! Persists FunctionGraphs to JouleDB's B-tree storage via a key-value encoding.
//! Keys follow the convention: `__rce__::func::{name}` and `__rce__::prog::{name}`.
//! Values are JSON-serialized FunctionGraph / ProgramGraph.
//!
//! This module requires the `jouledb` feature to be enabled.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::graph::{FunctionGraph, ProgramGraph};
use crate::store::{GraphStore, StoreError};

const FUNC_PREFIX: &str = "__rce__::func::";
const PROG_PREFIX: &str = "__rce__::prog::";

/// JouleDB-backed graph store.
///
/// Uses a simple file-backed key-value store (BTreeMap serialized to a single
/// file) as the bridge to JouleDB. When JouleDB's StorageEngine is available
/// as a workspace dependency, this can be upgraded to use the B-tree directly.
pub struct JouleDbStore {
    path: PathBuf,
    data: BTreeMap<String, String>,
}

impl JouleDbStore {
    /// Open or create a JouleDB graph store at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let data = if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            serde_json::from_str(&contents).unwrap_or_default()
        } else {
            BTreeMap::new()
        };
        Ok(Self { path, data })
    }

    /// Flush the store to disk.
    fn flush(&self) -> Result<(), StoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    /// Get all keys with a given prefix.
    fn keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.data
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k[prefix.len()..].to_string())
            .collect()
    }

    /// Total number of stored graphs (functions + programs).
    pub fn count(&self) -> usize {
        self.data.len()
    }
}

impl GraphStore for JouleDbStore {
    fn put_function(&mut self, graph: &FunctionGraph) -> Result<(), StoreError> {
        let key = format!("{}{}", FUNC_PREFIX, graph.name);
        let json = serde_json::to_string(graph)?;
        self.data.insert(key, json);
        self.flush()
    }

    fn get_function(&self, name: &str) -> Result<FunctionGraph, StoreError> {
        let key = format!("{}{}", FUNC_PREFIX, name);
        let json = self.data.get(&key)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        Ok(serde_json::from_str(json)?)
    }

    fn put_program(&mut self, name: &str, program: &ProgramGraph) -> Result<(), StoreError> {
        let key = format!("{}{}", PROG_PREFIX, name);
        let json = serde_json::to_string(program)?;
        self.data.insert(key, json);
        self.flush()
    }

    fn get_program(&self, name: &str) -> Result<ProgramGraph, StoreError> {
        let key = format!("{}{}", PROG_PREFIX, name);
        let json = self.data.get(&key)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        Ok(serde_json::from_str(json)?)
    }

    fn list_functions(&self) -> Result<Vec<String>, StoreError> {
        Ok(self.keys_with_prefix(FUNC_PREFIX))
    }

    fn list_programs(&self) -> Result<Vec<String>, StoreError> {
        Ok(self.keys_with_prefix(PROG_PREFIX))
    }

    fn delete_function(&mut self, name: &str) -> Result<(), StoreError> {
        let key = format!("{}{}", FUNC_PREFIX, name);
        self.data.remove(&key);
        self.flush()
    }

    fn delete_program(&mut self, name: &str) -> Result<(), StoreError> {
        let key = format!("{}{}", PROG_PREFIX, name);
        self.data.remove(&key);
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::*;

    fn sample_func() -> FunctionGraph {
        FunctionGraph {
            name: "add".to_string(),
            nodes: vec![
                GraphNode::Param(Param { name: "a".into(), data_type: "i64".into() }),
                GraphNode::Param(Param { name: "b".into(), data_type: "i64".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "i64".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Copy },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Copy },
            ],
            output_node: 2,
            return_type: "i64".to_string(),
        }
    }

    fn sample_program() -> ProgramGraph {
        ProgramGraph {
            functions: vec![sample_func()],
            entry: "add".to_string(),
            allows: Default::default(),
        }
    }

    #[test]
    fn jouledb_store_function_roundtrip() {
        let path = std::env::temp_dir().join("rce_jouledb_test.json");
        let _ = std::fs::remove_file(&path);

        let mut store = JouleDbStore::open(&path).unwrap();
        let func = sample_func();
        store.put_function(&func).unwrap();

        let loaded = store.get_function("add").unwrap();
        assert_eq!(loaded.name, "add");
        assert_eq!(loaded.nodes.len(), 3);

        let names = store.list_functions().unwrap();
        assert_eq!(names, vec!["add"]);

        store.delete_function("add").unwrap();
        assert!(store.get_function("add").is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jouledb_store_program_roundtrip() {
        let path = std::env::temp_dir().join("rce_jouledb_prog_test.json");
        let _ = std::fs::remove_file(&path);

        let mut store = JouleDbStore::open(&path).unwrap();
        let prog = sample_program();
        store.put_program("myapp", &prog).unwrap();

        let loaded = store.get_program("myapp").unwrap();
        assert_eq!(loaded.entry, "add");
        assert_eq!(loaded.functions.len(), 1);

        let progs = store.list_programs().unwrap();
        assert_eq!(progs, vec!["myapp"]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jouledb_store_persistence() {
        let path = std::env::temp_dir().join("rce_jouledb_persist_test.json");
        let _ = std::fs::remove_file(&path);

        // Write
        {
            let mut store = JouleDbStore::open(&path).unwrap();
            store.put_function(&sample_func()).unwrap();
            store.put_program("test", &sample_program()).unwrap();
        }

        // Read (new instance)
        {
            let store = JouleDbStore::open(&path).unwrap();
            let func = store.get_function("add").unwrap();
            assert_eq!(func.name, "add");
            let prog = store.get_program("test").unwrap();
            assert_eq!(prog.entry, "add");
        }

        let _ = std::fs::remove_file(&path);
    }
}
