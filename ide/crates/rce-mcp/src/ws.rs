//! Workspace path policy.
//!
//! Every path an agent names is relative to the workspace root. Absolute
//! paths, `..`, and any symlink on the way are refused, and the resolved
//! parent must canonicalize inside the root. Writes go only to `.rce` source
//! files outside dot-directories, so an agent cannot plant a git hook, an
//! editor or agent config, a build script or a package manifest that a
//! trusted tool would later run (the 2026 failure class in the harness report).
use serde_json::{Value, json};
use std::path::{Component, Path, PathBuf};

/// File names refused for writes even if the extension rule changes later.
pub const DENY_NAMES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "rust-toolchain.toml",
    "package.json",
    "Makefile",
    "justfile",
    ".gitattributes",
    ".gitmodules",
    ".envrc",
    "mcp.json",
    "settings.json",
    "sandbox.json",
    "pyproject.toml",
    "setup.py",
];

pub struct Workspace {
    pub root: PathBuf,
}

impl Workspace {
    pub fn new(root: &Path) -> std::io::Result<Self> {
        Ok(Workspace {
            root: root.canonicalize()?,
        })
    }

    /// The policy document; its BLAKE3 goes into every receipt.
    pub fn policy(&self) -> Value {
        json!({
            "schema": "rce-ide.policy.v1",
            "root": self.root.display().to_string(),
            "paths": "relative only; no `..`; no symlink components; must resolve inside root",
            "write_ext": ["rce"],
            "write_deny_dot_components": true,
            "write_deny_names": DENY_NAMES,
            "propose_edit": "applied only when rce check admits every file in the workspace program",
            "network": "none",
            "memory_writes": "rce-memory only: .md notes under notes/, no dot paths, never agent instruction or hook files (CLAUDE.md, AGENTS.md, ...); facts are expired, never deleted",
        })
    }

    /// Resolve `rel` for reading (must exist) or writing (parent must exist).
    pub fn resolve(&self, rel: &str, write: bool) -> Result<PathBuf, String> {
        if rel.is_empty() {
            return Err("empty path".into());
        }
        let p = Path::new(rel);
        let mut cur = self.root.clone();
        let comps: Vec<Component> = p.components().collect();
        for (i, c) in comps.iter().enumerate() {
            let Component::Normal(name) = c else {
                return Err(format!("`{rel}`: only plain relative paths are allowed"));
            };
            let name_s = name.to_string_lossy();
            if write && name_s.starts_with('.') {
                return Err(format!(
                    "`{rel}`: writes into dot paths (hooks, configs) are refused"
                ));
            }
            cur.push(name);
            match std::fs::symlink_metadata(&cur) {
                Ok(m) if m.file_type().is_symlink() => {
                    return Err(format!("`{rel}`: symlink in path refused"));
                }
                Ok(_) => {}
                Err(_) if write && i + 1 == comps.len() => {}
                Err(_) => return Err(format!("`{rel}`: not found")),
            }
        }
        let parent = cur
            .parent()
            .ok_or("no parent")?
            .canonicalize()
            .map_err(|e| format!("`{rel}`: {e}"))?;
        if !parent.starts_with(&self.root) {
            return Err(format!("`{rel}`: escapes the workspace"));
        }
        if cur.exists() {
            let c = cur.canonicalize().map_err(|e| format!("`{rel}`: {e}"))?;
            if !c.starts_with(&self.root) {
                return Err(format!("`{rel}`: escapes the workspace"));
            }
        }
        if write {
            let fname = cur
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if DENY_NAMES.contains(&fname.as_str()) {
                return Err(format!(
                    "`{rel}`: writes to build, hook or config files are refused"
                ));
            }
            if cur.extension().and_then(|e| e.to_str()) != Some("rce") {
                return Err(format!("`{rel}`: only .rce source files may be written"));
            }
        }
        Ok(cur)
    }

    /// Every `.rce` file under the root as (relative path, source), skipping
    /// dot directories and symlinks. Sorted by path.
    pub fn program(&self) -> Vec<(String, String)> {
        let mut out = vec![];
        let mut stack = vec![self.root.clone()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let Ok(ft) = e.file_type() else { continue };
                if name.starts_with('.') || ft.is_symlink() {
                    continue;
                }
                let p = e.path();
                if ft.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|x| x.to_str()) == Some("rce")
                    && let Ok(src) = std::fs::read_to_string(&p)
                {
                    let rel = p
                        .strip_prefix(&self.root)
                        .unwrap_or(&p)
                        .display()
                        .to_string();
                    out.push((rel, src));
                }
                if out.len() >= 10_000 {
                    break;
                }
            }
        }
        out.sort();
        out
    }

    /// Write via a temporary file and rename, so a crash never leaves half a file.
    pub fn write_atomic(&self, path: &Path, text: &str) -> std::io::Result<()> {
        let dir = path.parent().unwrap_or(&self.root);
        let tmp = dir.join(format!(
            ".{}.rce-mcp.tmp",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)
    }
}
