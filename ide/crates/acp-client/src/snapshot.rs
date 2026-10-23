//! Workspace snapshots, to catch and revert writes that bypassed rce-mcp.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Directories never snapshotted (IDE state, VCS, build output).
pub const SKIP: &[&str] = &[".rce-ide", ".git", "target"];
/// Files larger than this are hashed but not kept for revert.
pub const KEEP_CAP: u64 = 8 << 20;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    /// Relative path → (blake3 hex, contents when small enough).
    pub files: BTreeMap<String, (String, Option<Vec<u8>>)>,
}

impl Snapshot {
    pub fn take(root: &Path) -> std::io::Result<Self> {
        let mut s = Snapshot::default();
        walk(root, root, &mut s)?;
        Ok(s)
    }

    /// Paths created, changed or removed between `self` and `after`.
    pub fn changed(&self, after: &Snapshot) -> Vec<String> {
        let mut v: Vec<String> = after
            .files
            .iter()
            .filter(|(k, (h, _))| self.files.get(*k).map(|x| &x.0) != Some(h))
            .map(|(k, _)| k.clone())
            .collect();
        v.extend(
            self.files
                .keys()
                .filter(|k| !after.files.contains_key(*k))
                .cloned(),
        );
        v.sort();
        v.dedup();
        v
    }

    /// Put `rel` back as it was in `self` (delete it if it did not exist).
    /// `Ok(false)` if the old contents were too large to keep.
    pub fn restore(&self, root: &Path, rel: &str) -> std::io::Result<bool> {
        let p = root.join(rel);
        match self.files.get(rel) {
            None => {
                if std::fs::symlink_metadata(&p).is_ok() {
                    std::fs::remove_file(&p)?;
                }
                Ok(true)
            }
            Some((_, Some(b))) => {
                if let Some(d) = p.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(&p, b)?;
                Ok(true)
            }
            Some((_, None)) => Ok(false),
        }
    }
}

fn walk(root: &Path, dir: &Path, s: &mut Snapshot) -> std::io::Result<()> {
    let mut ents: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    ents.sort();
    for p in ents {
        let rel = p
            .strip_prefix(root)
            .unwrap_or(&p)
            .to_string_lossy()
            .replace('\\', "/");
        let md = std::fs::symlink_metadata(&p)?;
        if md.is_dir() {
            if dir == root && SKIP.contains(&rel.as_str()) {
                continue;
            }
            walk(root, &p, s)?;
        } else if md.is_file() {
            let b = std::fs::read(&p)?;
            let h = blake3::hash(&b).to_hex().to_string();
            let keep = (md.len() <= KEEP_CAP).then_some(b);
            s.files.insert(rel, (h, keep));
        } else if md.file_type().is_symlink() {
            let t = std::fs::read_link(&p)?;
            let h = blake3::hash(format!("symlink:{}", t.display()).as_bytes())
                .to_hex()
                .to_string();
            s.files.insert(rel, (h, None));
        }
    }
    Ok(())
}
