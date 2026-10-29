//! Markdown notes with YAML frontmatter, readable by Obsidian, Logseq or git.
//!
//! Files win: [`plan_sync`] makes memory match the notes directory. A new
//! file becomes a fact, an edited file supersedes its fact (old row expired,
//! new row in the same lineage), a deleted file or `status: forgotten`
//! expires it. Memory writes notes only through [`write_note`], which writes
//! nothing but a plain `.md` file inside the notes directory and never writes
//! agent instruction and hook files (CLAUDE.md, AGENTS.md, .cursor, ...).
use crate::engine::{MemoryEngine, Plan, Result as MResult};
use crate::fact::{Fact, Provenance};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

/// Lowercased file names never written, wherever they would land.
pub const DENY_FILES: &[&str] = &[
    "claude.md",
    "claude.local.md",
    "agents.md",
    "agent.md",
    "gemini.md",
    "qwen.md",
    "copilot-instructions.md",
    "conventions.md",
    "readme.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    ".goosehints",
];

#[derive(Debug, thiserror::Error)]
pub enum NoteError {
    #[error("kept: {0}")]
    NotPerformed(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("frontmatter: {0}")]
    Yaml(String),
    #[error(transparent)]
    Mem(#[from] crate::engine::MemError),
}

/// YAML frontmatter. Every field is optional so hand-written notes work.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Front {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rce_fact: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub front: Front,
    pub body: String,
}

/// Split `---\nYAML\n---\nbody`. A file without frontmatter is all body.
pub fn parse_note(text: &str) -> Result<Note, NoteError> {
    let t = text.strip_prefix('\u{feff}').unwrap_or(text);
    let rest = t
        .strip_prefix("---\n")
        .or_else(|| t.strip_prefix("---\r\n"));
    let Some(rest) = rest else {
        return Ok(Note {
            front: Front::default(),
            body: t.to_string(),
        });
    };
    let mut yaml_end = None;
    let mut pos = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            yaml_end = Some((pos, pos + line.len()));
            break;
        }
        pos += line.len();
    }
    let Some((ye, be)) = yaml_end else {
        return Err(NoteError::Yaml("unterminated frontmatter".into()));
    };
    let yaml = &rest[..ye];
    let front: Front = if yaml.trim().is_empty() {
        Front::default()
    } else {
        serde_norway::from_str(yaml).map_err(|e| NoteError::Yaml(e.to_string()))?
    };
    Ok(Note {
        front,
        body: rest[be..].to_string(),
    })
}

pub fn render_note(n: &Note) -> String {
    let y = serde_norway::to_string(&n.front).expect("frontmatter serializes");
    format!("---\n{}---\n{}", y, n.body)
}

// Civil date conversion (Howard Hinnant's algorithm), UTC.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS[.mmm]Z`, or integer milliseconds.
pub fn parse_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Ok(ms) = s.parse::<i64>() {
        return Some(ms);
    }
    let num = |a: usize, b: usize| s.get(a..b).and_then(|x| x.parse::<i64>().ok());
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    if s.as_bytes().get(4) != Some(&b'-') || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let mut ms = days_from_civil(y, mo, d) * 86_400_000;
    if s.len() > 10 {
        let (h, mi, se) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
        ms += (h * 3600 + mi * 60 + se) * 1000;
        if s.as_bytes().get(19) == Some(&b'.') {
            ms += num(20, 23)?;
        }
    }
    Some(ms)
}

pub fn format_time(ms: i64) -> String {
    let (days, rem) = (ms.div_euclid(86_400_000), ms.rem_euclid(86_400_000));
    let (y, m, d) = civil_from_days(days);
    let (h, mi, s, f) = (
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000,
    );
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{f:03}Z")
}

/// An error for any path memory must not write.
pub fn check_write_path(rel: &str) -> Result<(), NoteError> {
    let p = Path::new(rel);
    for c in p.components() {
        let Component::Normal(n) = c else {
            return Err(NoteError::NotPerformed(format!(
                "`{rel}`: only plain relative paths"
            )));
        };
        let n = n.to_string_lossy().to_lowercase();
        if n.starts_with('.') {
            return Err(NoteError::NotPerformed(format!(
                "`{rel}`: dot paths (hooks, agent configs) are never written"
            )));
        }
        if DENY_FILES.contains(&n.as_str()) || n == "hooks" {
            return Err(NoteError::NotPerformed(format!(
                "`{rel}`: agent instruction and hook files are never written"
            )));
        }
    }
    if p.extension().and_then(|e| e.to_str()) != Some("md") {
        return Err(NoteError::NotPerformed(format!(
            "`{rel}`: notes are .md files only"
        )));
    }
    Ok(())
}

/// Kebab-case slug of a title, ASCII only, at most 64 chars.
pub fn slug(title: &str) -> String {
    let mut s = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
    }
    let s = s.trim_end_matches('-').chars().take(64).collect::<String>();
    if s.is_empty() { "note".into() } else { s }
}

/// The notes directory, relative to a workspace root.
pub struct Notes {
    pub root: PathBuf,
    pub dir: String,
}

impl Notes {
    pub fn new(root: &Path, dir: &str) -> Result<Self, NoteError> {
        let d = format!("{dir}/x.md");
        check_write_path(&d)?;
        Ok(Notes {
            root: root.to_path_buf(),
            dir: dir.trim_end_matches('/').to_string(),
        })
    }

    /// Every `.md` under the notes dir as (workspace-relative path, text).
    /// Dot entries and symlinks are skipped.
    pub fn scan(&self) -> Vec<(String, String)> {
        let mut out = vec![];
        let base = self.root.join(&self.dir);
        let mut stack = vec![base];
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
                } else if p.extension().and_then(|x| x.to_str()) == Some("md")
                    && let Ok(t) = std::fs::read_to_string(&p)
                {
                    let rel = p
                        .strip_prefix(&self.root)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push((rel, t));
                }
            }
        }
        out.sort();
        out
    }

    /// Write a note file atomically after the path checks. No symlink is
    /// followed: every existing component must be a real directory or file.
    pub fn write(&self, rel: &str, text: &str) -> Result<(), NoteError> {
        check_write_path(rel)?;
        if !rel.starts_with(&format!("{}/", self.dir)) {
            return Err(NoteError::NotPerformed(format!(
                "`{rel}`: outside the notes directory"
            )));
        }
        let path = self.root.join(rel);
        let parent = path
            .parent()
            .ok_or_else(|| NoteError::NotPerformed("no parent".into()))?;
        std::fs::create_dir_all(parent)?;
        let mut cur = self.root.clone();
        for c in Path::new(rel).components() {
            cur.push(c);
            if let Ok(m) = std::fs::symlink_metadata(&cur)
                && m.file_type().is_symlink()
            {
                return Err(NoteError::NotPerformed(format!("`{rel}`: symlink in path")));
            }
        }
        let root = self.root.canonicalize()?;
        if !parent.canonicalize()?.starts_with(&root) {
            return Err(NoteError::NotPerformed(format!(
                "`{rel}`: escapes the workspace"
            )));
        }
        let tmp = parent.join(format!(
            ".{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

/// The fact a note file says, as field values (ids and system times aside).
#[derive(Debug, Clone, PartialEq)]
pub struct NoteFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub text: String,
    pub valid_from: Option<i64>,
    pub valid_to: Option<i64>,
    pub provenance: Provenance,
    pub forgotten: bool,
}

pub fn note_fact(rel: &str, n: &Note) -> NoteFact {
    let stem = Path::new(rel)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let title = n
        .front
        .title
        .clone()
        .or_else(|| {
            n.body
                .lines()
                .find_map(|l| l.strip_prefix("# ").map(|t| t.trim().to_string()))
        })
        .unwrap_or(stem);
    NoteFact {
        subject: n
            .front
            .subject
            .clone()
            .unwrap_or_else(|| format!("note:{}", rel.trim_end_matches(".md"))),
        predicate: n.front.predicate.clone().unwrap_or_else(|| "note".into()),
        object: title,
        text: n.body.clone(),
        valid_from: n.front.valid_from.as_deref().and_then(parse_time),
        valid_to: n.front.valid_to.as_deref().and_then(parse_time),
        provenance: n.front.provenance.unwrap_or(Provenance::Extracted),
        forgotten: n.front.status.as_deref() == Some("forgotten"),
    }
}

/// Render a fact as a note.
pub fn fact_note(f: &Fact) -> Note {
    Note {
        front: Front {
            subject: Some(f.subject.clone()),
            predicate: Some(f.predicate.clone()),
            title: Some(f.object.clone()),
            valid_from: Some(format_time(f.valid_from)),
            valid_to: f.valid_to.map(format_time),
            provenance: Some(f.provenance),
            form_hash: Some(f.form_hash.clone()),
            rce_fact: Some(f.lineage),
            receipt: f.receipt_id.clone(),
            tags: vec!["rce-memory".into()],
            status: (!f.live()).then(|| "forgotten".into()),
        },
        body: f.text.clone(),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SyncReport {
    pub imported: Vec<String>,
    pub updated: Vec<String>,
    pub expired: Vec<String>,
    pub unchanged: usize,
    pub errors: Vec<String>,
}

fn same(f: &Fact, nf: &NoteFact) -> bool {
    f.subject == nf.subject
        && f.predicate == nf.predicate
        && f.object == nf.object
        && f.text == nf.text
        && nf.valid_from.is_none_or(|v| v == f.valid_from)
        && nf.valid_to == f.valid_to
        && f.provenance == nf.provenance
}

/// Plan making memory match the files. One plan, so a sync lands whole.
pub fn plan_sync<E: MemoryEngine>(
    notes: &Notes,
    engine: &mut E,
    form_hash: &str,
    now: i64,
) -> MResult<(Plan, SyncReport)> {
    let mut by_src: BTreeMap<String, Fact> = BTreeMap::new();
    for f in engine.list_live()? {
        if let Some(s) = &f.source
            && s.starts_with(&format!("{}/", notes.dir))
        {
            by_src.insert(s.clone(), f);
        }
    }
    let mut plan = Plan::default();
    let mut rep = SyncReport::default();
    for (rel, text) in notes.scan() {
        let cur = by_src.remove(&rel);
        let n = match parse_note(&text) {
            Ok(n) => n,
            Err(e) => {
                rep.errors.push(format!("{rel}: {e}"));
                continue;
            }
        };
        let nf = note_fact(&rel, &n);
        match cur {
            Some(f) if nf.forgotten => {
                plan.expires.push((f.id, now));
                rep.expired.push(rel);
            }
            None if nf.forgotten => rep.unchanged += 1,
            Some(f) if same(&f, &nf) => rep.unchanged += 1,
            cur => {
                let id = engine.next_id()?;
                let (lineage, supersedes) = match &cur {
                    Some(f) => {
                        plan.expires.push((f.id, now));
                        (f.lineage, Some(f.id))
                    }
                    None => (id, None),
                };
                plan.inserts.push(Fact {
                    id,
                    lineage,
                    supersedes,
                    subject: nf.subject,
                    predicate: nf.predicate,
                    object: nf.object,
                    text: nf.text,
                    multi: true,
                    valid_from: nf
                        .valid_from
                        .or(cur.as_ref().map(|f| f.valid_from))
                        .unwrap_or(now),
                    valid_to: nf.valid_to,
                    recorded_at: now,
                    expired_at: None,
                    form_hash: form_hash.into(),
                    provenance: nf.provenance,
                    receipt_id: None,
                    source: Some(rel.clone()),
                });
                if cur.is_some() {
                    rep.updated.push(rel)
                } else {
                    rep.imported.push(rel)
                }
            }
        }
    }
    // Facts whose note file is gone: the file was deleted, files win.
    for (rel, f) in by_src {
        plan.expires.push((f.id, now));
        rep.expired.push(rel);
    }
    Ok((plan, rep))
}
