//! Persistence layer for FunctionGraphs.
//!
//! Provides serialization/deserialization of `FunctionGraph` and `ProgramGraph`
//! to JSON (file-based or in-memory). This is the foundation layer —
//! JouleDB-native storage can build on top via the `Store` trait.

use std::path::Path;

use crate::graph::{FunctionGraph, ProgramGraph};

/// Errors during graph storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("graph not found: {0}")]
    NotFound(String),
}

/// Trait for FunctionGraph storage backends.
pub trait GraphStore {
    /// Store a function graph by name.
    fn put_function(&mut self, graph: &FunctionGraph) -> Result<(), StoreError>;
    /// Load a function graph by name.
    fn get_function(&self, name: &str) -> Result<FunctionGraph, StoreError>;
    /// Store a full program graph.
    fn put_program(&mut self, name: &str, program: &ProgramGraph) -> Result<(), StoreError>;
    /// Load a program graph by name.
    fn get_program(&self, name: &str) -> Result<ProgramGraph, StoreError>;
    /// List all stored function names.
    fn list_functions(&self) -> Result<Vec<String>, StoreError>;
    /// List all stored program names.
    fn list_programs(&self) -> Result<Vec<String>, StoreError>;
    /// Delete a function graph by name.
    fn delete_function(&mut self, name: &str) -> Result<(), StoreError>;
    /// Delete a program graph by name.
    fn delete_program(&mut self, name: &str) -> Result<(), StoreError>;
}

// ── In-memory store ─────────────────────────────────────────────────

/// In-memory graph store backed by HashMaps.
#[derive(Default)]
pub struct MemoryStore {
    functions: std::collections::HashMap<String, FunctionGraph>,
    programs: std::collections::HashMap<String, ProgramGraph>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphStore for MemoryStore {
    fn put_function(&mut self, graph: &FunctionGraph) -> Result<(), StoreError> {
        self.functions.insert(graph.name.clone(), graph.clone());
        Ok(())
    }

    fn get_function(&self, name: &str) -> Result<FunctionGraph, StoreError> {
        self.functions.get(name).cloned().ok_or_else(|| StoreError::NotFound(name.to_string()))
    }

    fn put_program(&mut self, name: &str, program: &ProgramGraph) -> Result<(), StoreError> {
        self.programs.insert(name.to_string(), program.clone());
        Ok(())
    }

    fn get_program(&self, name: &str) -> Result<ProgramGraph, StoreError> {
        self.programs.get(name).cloned().ok_or_else(|| StoreError::NotFound(name.to_string()))
    }

    fn list_functions(&self) -> Result<Vec<String>, StoreError> {
        Ok(self.functions.keys().cloned().collect())
    }

    fn list_programs(&self) -> Result<Vec<String>, StoreError> {
        Ok(self.programs.keys().cloned().collect())
    }

    fn delete_function(&mut self, name: &str) -> Result<(), StoreError> {
        self.functions.remove(name);
        Ok(())
    }

    fn delete_program(&mut self, name: &str) -> Result<(), StoreError> {
        self.programs.remove(name);
        Ok(())
    }
}

// ── File-based JSON store ───────────────────────────────────────────

/// File-based graph store: each graph is a JSON file in a directory.
pub struct FileStore {
    root: std::path::PathBuf,
}

impl FileStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(root.join("functions"))?;
        std::fs::create_dir_all(root.join("programs"))?;
        Ok(Self { root })
    }

    fn func_path(&self, name: &str) -> std::path::PathBuf {
        self.root.join("functions").join(format!("{}.json", name))
    }

    fn prog_path(&self, name: &str) -> std::path::PathBuf {
        self.root.join("programs").join(format!("{}.json", name))
    }
}

impl GraphStore for FileStore {
    fn put_function(&mut self, graph: &FunctionGraph) -> Result<(), StoreError> {
        let json = serde_json::to_string_pretty(graph)?;
        std::fs::write(self.func_path(&graph.name), json)?;
        Ok(())
    }

    fn get_function(&self, name: &str) -> Result<FunctionGraph, StoreError> {
        let path = self.func_path(name);
        if !path.exists() {
            return Err(StoreError::NotFound(name.to_string()));
        }
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    fn put_program(&mut self, name: &str, program: &ProgramGraph) -> Result<(), StoreError> {
        let json = serde_json::to_string_pretty(program)?;
        std::fs::write(self.prog_path(name), json)?;
        Ok(())
    }

    fn get_program(&self, name: &str) -> Result<ProgramGraph, StoreError> {
        let path = self.prog_path(name);
        if !path.exists() {
            return Err(StoreError::NotFound(name.to_string()));
        }
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    fn list_functions(&self) -> Result<Vec<String>, StoreError> {
        list_json_names(&self.root.join("functions"))
    }

    fn list_programs(&self) -> Result<Vec<String>, StoreError> {
        list_json_names(&self.root.join("programs"))
    }

    fn delete_function(&mut self, name: &str) -> Result<(), StoreError> {
        let path = self.func_path(name);
        if path.exists() { std::fs::remove_file(path)?; }
        Ok(())
    }

    fn delete_program(&mut self, name: &str) -> Result<(), StoreError> {
        let path = self.prog_path(name);
        if path.exists() { std::fs::remove_file(path)?; }
        Ok(())
    }
}

fn list_json_names(dir: &Path) -> Result<Vec<String>, StoreError> {
    let mut names = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if let Some(name) = entry.path().file_stem() {
                names.push(name.to_string_lossy().to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

// ── Convenience functions ───────────────────────────────────────────

/// Serialize a ProgramGraph to JSON string.
pub fn to_json(program: &ProgramGraph) -> Result<String, StoreError> {
    Ok(serde_json::to_string_pretty(program)?)
}

/// Deserialize a ProgramGraph from JSON string.
pub fn from_json(json: &str) -> Result<ProgramGraph, StoreError> {
    Ok(serde_json::from_str(json)?)
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
    fn memory_store_roundtrip() {
        let mut store = MemoryStore::new();
        let func = sample_func();
        store.put_function(&func).unwrap();

        let loaded = store.get_function("add").unwrap();
        assert_eq!(loaded.name, "add");
        assert_eq!(loaded.nodes.len(), 3);

        let names = store.list_functions().unwrap();
        assert_eq!(names, vec!["add"]);

        store.delete_function("add").unwrap();
        assert!(store.get_function("add").is_err());
    }

    #[test]
    fn memory_store_program() {
        let mut store = MemoryStore::new();
        let prog = sample_program();
        store.put_program("myapp", &prog).unwrap();

        let loaded = store.get_program("myapp").unwrap();
        assert_eq!(loaded.entry, "add");
        assert_eq!(loaded.functions.len(), 1);
    }

    #[test]
    fn file_store_roundtrip() {
        let dir = std::env::temp_dir().join("rce_test_store");
        let _ = std::fs::remove_dir_all(&dir);

        let mut store = FileStore::new(&dir).unwrap();
        let func = sample_func();
        store.put_function(&func).unwrap();

        let loaded = store.get_function("add").unwrap();
        assert_eq!(loaded.name, "add");
        assert_eq!(loaded.nodes.len(), 3);

        let names = store.list_functions().unwrap();
        assert_eq!(names, vec!["add"]);

        store.delete_function("add").unwrap();
        assert!(store.get_function("add").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_program() {
        let dir = std::env::temp_dir().join("rce_test_prog_store");
        let _ = std::fs::remove_dir_all(&dir);

        let mut store = FileStore::new(&dir).unwrap();
        let prog = sample_program();
        store.put_program("myapp", &prog).unwrap();

        let loaded = store.get_program("myapp").unwrap();
        assert_eq!(loaded.entry, "add");
        assert_eq!(loaded.functions.len(), 1);

        let progs = store.list_programs().unwrap();
        assert_eq!(progs, vec!["myapp"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_roundtrip() {
        let prog = sample_program();
        let json = to_json(&prog).unwrap();
        let loaded = from_json(&json).unwrap();
        assert_eq!(loaded.entry, prog.entry);
        assert_eq!(loaded.functions.len(), prog.functions.len());
    }
}
