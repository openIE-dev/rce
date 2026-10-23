//! The editor's text buffer: a rope (ropey, pure Rust) with a cursor and
//! selection, grouped undo/redo, an IME preedit, an internal clipboard and
//! a scroll position. Positions are char indices into the rope; columns are
//! chars within a line (tabs are shown as [`TAB`] spaces by the view).
use ropey::Rope;
use serde::{Deserialize, Serialize};

pub const TAB: usize = 4;

/// One editing command, from the keyboard, the IME, the mouse or a test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditOp {
    /// Typed or committed (IME) text; replaces the selection.
    Insert(String),
    /// Enter: newline keeping the current line's indentation.
    Newline,
    Tab,
    Backspace,
    Delete,
    Move {
        to: Motion,
        select: bool,
    },
    SelectAll,
    Undo,
    Redo,
    Copy,
    Cut,
    /// Paste the given text, or the internal clipboard when `None`.
    Paste(Option<String>),
    /// IME composition in progress (empty clears it).
    Preedit(String),
    /// Mouse: place the cursor (extend = shift-click or drag).
    ClickAt {
        line: usize,
        col: usize,
        extend: bool,
    },
    /// Scroll by this many lines (wheel).
    Scroll(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Motion {
    Left,
    Right,
    Up,
    Down,
    WordLeft,
    WordRight,
    LineStart,
    LineEnd,
    PageUp,
    PageDown,
    DocStart,
    DocEnd,
}

#[derive(Debug, Clone, PartialEq)]
struct Change {
    at: usize,
    removed: String,
    inserted: String,
    before: (usize, Option<usize>),
    after: (usize, Option<usize>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Type,
    Erase,
    Other,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    rope: Rope,
    pub cursor: usize,
    /// The other end of the selection, if any.
    pub anchor: Option<usize>,
    goal: Option<usize>,
    undo: Vec<Change>,
    redo: Vec<Change>,
    last: Kind,
    pub preedit: Option<String>,
    /// First visible line.
    pub scroll: usize,
    /// Visible lines (set by the host from the panel height).
    pub rows: usize,
    pub clipboard: String,
    /// Bumped on every text change.
    pub version: u64,
}

impl Default for Buffer {
    fn default() -> Self {
        Buffer::new("")
    }
}

/// The visible part, ready for the view.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Visible {
    pub first: usize,
    /// Lines as shown (tabs expanded), without the newline.
    pub lines: Vec<String>,
    /// Cursor (line, display column).
    pub cursor: (usize, usize),
    /// Selection per visible line: (line, display start, display end).
    pub selection: Vec<(usize, usize, usize)>,
    pub preedit: Option<String>,
    pub total_lines: usize,
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn expand(line: &str) -> String {
    let mut s = String::new();
    for c in line.chars() {
        if c == '\t' {
            let n = TAB - s.chars().count() % TAB;
            s.extend(std::iter::repeat_n(' ', n));
        } else {
            s.push(c);
        }
    }
    s
}

fn display_col(line: &str, col: usize) -> usize {
    expand(&line.chars().take(col).collect::<String>())
        .chars()
        .count()
}

fn char_col(line: &str, dcol: usize) -> usize {
    let mut d = 0;
    for (i, c) in line.chars().enumerate() {
        let w = if c == '\t' { TAB - d % TAB } else { 1 };
        if d + w > dcol {
            // Nearer edge.
            return if dcol - d > w / 2 { i + 1 } else { i };
        }
        d += w;
    }
    line.chars().count()
}

impl Buffer {
    pub fn new(text: &str) -> Self {
        Buffer {
            rope: Rope::from_str(text),
            cursor: 0,
            anchor: None,
            goal: None,
            undo: vec![],
            redo: vec![],
            last: Kind::Other,
            preedit: None,
            scroll: 0,
            rows: 40,
            clipboard: String::new(),
            version: 0,
        }
    }

    pub fn text(&self) -> String {
        self.rope.to_string()
    }
    pub fn len(&self) -> usize {
        self.rope.len_chars()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn lines(&self) -> usize {
        self.rope.len_lines()
    }
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Line text without its line break.
    pub fn line(&self, l: usize) -> String {
        let s = self.rope.line(l).to_string();
        s.trim_end_matches(['\n', '\r']).to_string()
    }

    fn line_len(&self, l: usize) -> usize {
        self.line(l).chars().count()
    }

    /// (line, char column) of a char index.
    pub fn line_col(&self, pos: usize) -> (usize, usize) {
        let l = self.rope.char_to_line(pos);
        (l, pos - self.rope.line_to_char(l))
    }

    pub fn pos(&self, line: usize, col: usize) -> usize {
        let line = line.min(self.lines() - 1);
        self.rope.line_to_char(line) + col.min(self.line_len(line))
    }

    pub fn selection(&self) -> Option<(usize, usize)> {
        let a = self.anchor?;
        (a != self.cursor).then(|| (a.min(self.cursor), a.max(self.cursor)))
    }

    pub fn selected_text(&self) -> Option<String> {
        self.selection()
            .map(|(a, b)| self.rope.slice(a..b).to_string())
    }

    /// Replace `a..b` with `text`, record it, put the cursor after it.
    fn replace(&mut self, a: usize, b: usize, text: &str, kind: Kind) {
        let removed = self.rope.slice(a..b).to_string();
        let before = (self.cursor, self.anchor);
        self.rope.remove(a..b);
        self.rope.insert(a, text);
        self.cursor = a + text.chars().count();
        self.anchor = None;
        self.goal = None;
        self.version += 1;
        self.redo.clear();
        let after = (self.cursor, None);
        // Group runs of typing (until a space or newline) and runs of erasing.
        if let Some(p) = self.undo.last_mut()
            && kind == self.last
            && kind != Kind::Other
        {
            if kind == Kind::Type
                && removed.is_empty()
                && p.removed.is_empty()
                && a == p.at + p.inserted.chars().count()
                && !p.inserted.ends_with([' ', '\n'])
            {
                p.inserted.push_str(text);
                p.after = after;
                return;
            }
            if kind == Kind::Erase && text.is_empty() && p.inserted.is_empty() {
                if b == p.at {
                    p.removed.insert_str(0, &removed);
                    p.at = a;
                    p.after = after;
                    return;
                }
                if a == p.at {
                    p.removed.push_str(&removed);
                    p.after = after;
                    return;
                }
            }
        }
        self.last = kind;
        self.undo.push(Change {
            at: a,
            removed,
            inserted: text.into(),
            before,
            after,
        });
    }

    fn insert(&mut self, text: &str, kind: Kind) {
        let (a, b) = self.selection().unwrap_or((self.cursor, self.cursor));
        let k = if a != b { Kind::Other } else { kind };
        self.replace(a, b, text, k);
        self.last = k;
    }

    fn word_left(&self, mut p: usize) -> usize {
        let ch = |i: usize| self.rope.char(i);
        while p > 0 && !is_word(ch(p - 1)) {
            p -= 1;
        }
        while p > 0 && is_word(ch(p - 1)) {
            p -= 1;
        }
        p
    }

    fn word_right(&self, mut p: usize) -> usize {
        let n = self.len();
        let ch = |i: usize| self.rope.char(i);
        while p < n && !is_word(ch(p)) {
            p += 1;
        }
        while p < n && is_word(ch(p)) {
            p += 1;
        }
        p
    }

    fn vertical(&mut self, dl: i64) -> usize {
        let (l, c) = self.line_col(self.cursor);
        let goal = self.goal.unwrap_or_else(|| display_col(&self.line(l), c));
        self.goal = Some(goal);
        let nl = (l as i64 + dl).clamp(0, self.lines() as i64 - 1) as usize;
        if nl == l {
            return if dl < 0 { 0 } else { self.len() };
        }
        self.pos(nl, char_col(&self.line(nl), goal))
    }

    fn motion(&mut self, m: Motion, select: bool) -> usize {
        let p = self.cursor;
        let (l, _) = self.line_col(p);
        let page = self.rows.max(2) as i64 - 1;
        match m {
            // Without shift, an arrow collapses a selection to its edge.
            Motion::Left => match self.selection() {
                Some((a, _)) if !select => a,
                _ => p.saturating_sub(1),
            },
            Motion::Right => match self.selection() {
                Some((_, b)) if !select => b,
                _ => (p + 1).min(self.len()),
            },
            Motion::Up => self.vertical(-1),
            Motion::Down => self.vertical(1),
            Motion::PageUp => self.vertical(-page),
            Motion::PageDown => self.vertical(page),
            Motion::WordLeft => self.word_left(p),
            Motion::WordRight => self.word_right(p),
            Motion::LineStart => {
                // Smart home: first non-blank, then column 0.
                let line = self.line(l);
                let ind = line.chars().take_while(|c| c.is_whitespace()).count();
                let start = self.rope.line_to_char(l);
                if p == start + ind { start } else { start + ind }
            }
            Motion::LineEnd => self.pos(l, usize::MAX),
            Motion::DocStart => 0,
            Motion::DocEnd => self.len(),
        }
    }

    /// Apply one command. Returns true when the text changed.
    pub fn apply(&mut self, op: EditOp) -> bool {
        let v = self.version;
        if !matches!(op, EditOp::Preedit(_)) {
            self.preedit = None;
        }
        match op {
            EditOp::Insert(t) => {
                if !t.is_empty() {
                    self.insert(&t, Kind::Type)
                }
            }
            EditOp::Newline => {
                let (l, _) = self.line_col(self.cursor);
                let ind: String = self
                    .line(l)
                    .chars()
                    .take_while(|c| *c == ' ' || *c == '\t')
                    .collect();
                self.insert(&format!("\n{ind}"), Kind::Other)
            }
            EditOp::Tab => {
                let (_, c) = self.line_col(self.cursor);
                let n = TAB - c % TAB;
                self.insert(&" ".repeat(n), Kind::Type)
            }
            EditOp::Backspace => {
                if let Some((a, b)) = self.selection() {
                    self.replace(a, b, "", Kind::Other);
                    self.last = Kind::Other;
                } else if self.cursor > 0 {
                    let c = self.cursor;
                    self.replace(c - 1, c, "", Kind::Erase);
                }
            }
            EditOp::Delete => {
                if let Some((a, b)) = self.selection() {
                    self.replace(a, b, "", Kind::Other);
                    self.last = Kind::Other;
                } else if self.cursor < self.len() {
                    let c = self.cursor;
                    self.replace(c, c + 1, "", Kind::Erase);
                    self.cursor = c;
                }
            }
            EditOp::Move { to, select } => {
                let goal = self.goal;
                let np = self.motion(to, select);
                if !matches!(
                    to,
                    Motion::Up | Motion::Down | Motion::PageUp | Motion::PageDown
                ) {
                    self.goal = None;
                } else if self.goal.is_none() {
                    self.goal = goal;
                }
                if select {
                    self.anchor.get_or_insert(self.cursor);
                } else {
                    self.anchor = None;
                }
                self.cursor = np;
                self.last = Kind::Other;
            }
            EditOp::SelectAll => {
                self.anchor = Some(0);
                self.cursor = self.len();
            }
            EditOp::Undo => {
                if let Some(c) = self.undo.pop() {
                    let end = c.at + c.inserted.chars().count();
                    self.rope.remove(c.at..end);
                    self.rope.insert(c.at, &c.removed);
                    (self.cursor, self.anchor) = c.before;
                    self.version += 1;
                    self.redo.push(c);
                }
                self.last = Kind::Other;
            }
            EditOp::Redo => {
                if let Some(c) = self.redo.pop() {
                    let end = c.at + c.removed.chars().count();
                    self.rope.remove(c.at..end);
                    self.rope.insert(c.at, &c.inserted);
                    (self.cursor, self.anchor) = c.after;
                    self.version += 1;
                    self.undo.push(c);
                }
                self.last = Kind::Other;
            }
            EditOp::Copy => {
                if let Some(t) = self.selected_text() {
                    self.clipboard = t;
                }
            }
            EditOp::Cut => {
                if let Some((a, b)) = self.selection() {
                    self.clipboard = self.rope.slice(a..b).to_string();
                    self.replace(a, b, "", Kind::Other);
                    self.last = Kind::Other;
                }
            }
            EditOp::Paste(t) => {
                let t = t.unwrap_or_else(|| self.clipboard.clone());
                if !t.is_empty() {
                    self.insert(&t, Kind::Other);
                }
            }
            EditOp::Preedit(t) => self.preedit = (!t.is_empty()).then_some(t),
            EditOp::ClickAt { line, col, extend } => {
                let line = line.min(self.lines() - 1);
                let p = self.pos(line, char_col(&self.line(line), col));
                if extend {
                    self.anchor.get_or_insert(self.cursor);
                } else {
                    self.anchor = None;
                }
                self.cursor = p;
                self.goal = None;
                self.last = Kind::Other;
            }
            EditOp::Scroll(d) => {
                let max = self.lines().saturating_sub(1);
                self.scroll = (self.scroll as i64 + d).clamp(0, max as i64) as usize;
                return false;
            }
        }
        self.reveal();
        self.version != v
    }

    /// Scroll so the cursor is visible.
    pub fn reveal(&mut self) {
        let (l, _) = self.line_col(self.cursor);
        let rows = self.rows.max(1);
        if l < self.scroll {
            self.scroll = l;
        } else if l >= self.scroll + rows {
            self.scroll = l + 1 - rows;
        }
    }

    /// Replace everything (an accepted proposal reloaded from disk); keeps
    /// the cursor's line and column where possible, and the undo history
    /// as one change.
    pub fn set_text(&mut self, text: &str) {
        if self.text() == text {
            return;
        }
        let (l, c) = self.line_col(self.cursor);
        let n = self.len();
        self.replace(0, n, text, Kind::Other);
        self.last = Kind::Other;
        self.cursor = self.pos(l, c);
        self.reveal();
    }

    pub fn visible(&self) -> Visible {
        let total = self.lines();
        let first = self.scroll.min(total.saturating_sub(1));
        let last = (first + self.rows.max(1)).min(total);
        let (cl, cc) = self.line_col(self.cursor);
        let sel = self.selection();
        let mut selection = vec![];
        let mut lines = vec![];
        for l in first..last {
            let raw = self.line(l);
            if let Some((a, b)) = sel {
                let s = self.rope.line_to_char(l);
                let e = s + raw.chars().count();
                let (x0, x1) = (a.max(s), b.min(e));
                // An empty line inside a selection still shows one cell.
                if x0 < x1 || (a <= s && b > e) {
                    let d0 = display_col(&raw, x0.saturating_sub(s).min(e - s));
                    let d1 = display_col(&raw, x1.saturating_sub(s).min(e - s));
                    selection.push((l, d0, d1 + usize::from(b > e)));
                }
            }
            lines.push(expand(&raw));
        }
        Visible {
            first,
            lines,
            cursor: (cl, display_col(&self.line(cl), cc)),
            selection,
            preedit: self.preedit.clone(),
            total_lines: total,
        }
    }
}
