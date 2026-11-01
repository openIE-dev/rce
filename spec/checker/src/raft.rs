//! E.2b replication transport: Raft log replication for D, behind a
//! transport trait, with a loopback switch (partitions, reordering, loss)
//! and a TCP transport.
//!
//! What is implemented, following Ongaro and Ousterhout (USENIX ATC 2014,
//! extended version), Figure 2:
//! - terms and votes are made durable ([`Durable::save`]) before a vote is
//!   granted, before a candidate asks for votes, and before an append is
//!   acknowledged;
//! - a vote is granted only to a candidate whose log is at least as
//!   up-to-date, by (last term, last index);
//! - AppendEntries carries (prev_idx, prev_term); a follower does not append it when
//!   its log has no entry at prev_idx with that term (log matching), deletes
//!   a conflicting suffix, and appends what is missing;
//! - the leader advances the commit index only over entries of its own term
//!   held by a quorum (Raft section 5.4.2);
//! - InstallSnapshot: a leader whose follower needs entries that were
//!   compacted sends its snapshot (opaque bytes; the host uses an HS-5
//!   snapshot), and the follower keeps any suffix that matches.
//!
//! - membership changes by joint consensus (Raft section 6): a leader
//!   appends C_old,new; while it is the latest configuration in a log,
//!   elections and commitment need separate majorities of C_old and C_new;
//!   once C_old,new is committed the leader appends C_new, and a leader not
//!   in C_new steps down once C_new is committed. A node uses the latest
//!   configuration in its log, committed or not;
//! - snapshots are sent in chunks (offset, done) and installed when the
//!   last chunk arrives.
//!
//! Timing enters only through `tick()` (logical election and heartbeat
//! timeouts), so a loopback run is a deterministic function of the seed.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::rc::Rc;
use std::sync::mpsc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Msg {
    Vote { term: u64, cand: usize, last_idx: u64, last_term: u64 },
    VoteResp { term: u64, granted: bool },
    Append { term: u64, leader: usize, prev_idx: u64, prev_term: u64, entries: Vec<(u64, String)>, commit: u64 },
    AppendResp { term: u64, ok: bool, match_idx: u64 },
    Snap { term: u64, leader: usize, last_idx: u64, last_term: u64, offset: usize, data: String, done: bool, cfg: Cfg },
    SnapAck { term: u64, next_off: usize },
}

/// A cluster configuration: `old` alone, or joint `old` + `new`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cfg {
    pub old: Vec<usize>,
    pub new: Option<Vec<usize>>,
}

impl Cfg {
    pub fn members(&self) -> BTreeSet<usize> {
        self.old.iter().chain(self.new.iter().flatten()).copied().collect()
    }
    fn maj(set: &[usize], have: &BTreeSet<usize>) -> bool {
        set.iter().filter(|i| have.contains(i)).count() > set.len() / 2
    }
    /// A quorum of this configuration: a majority of `old`, and of `new` when joint.
    pub fn quorum(&self, have: &BTreeSet<usize>) -> bool {
        Cfg::maj(&self.old, have) && self.new.as_ref().is_none_or(|n| Cfg::maj(n, have))
    }
}

/// Configuration entries share the log with D lines; this prefix cannot
/// begin a D line, which is JSON starting with `{`.
pub const CFG_PREFIX: &str = "#cfg ";

fn cfg_of(line: &str) -> Option<Cfg> {
    line.strip_prefix(CFG_PREFIX).and_then(|j| serde_json::from_str(j).ok())
}

impl Msg {
    pub fn term(&self) -> u64 {
        match self {
            Msg::Vote { term, .. } | Msg::VoteResp { term, .. } | Msg::Append { term, .. } | Msg::AppendResp { term, .. } | Msg::Snap { term, .. } | Msg::SnapAck { term, .. } => *term,
        }
    }
}

/// The state Raft keeps on stable storage.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Persisted {
    pub term: u64,
    pub voted: Option<usize>,
    pub snap_idx: u64,
    pub snap_term: u64,
    pub snap: String,
    /// The configuration in force at `snap_idx`.
    pub snap_cfg: Cfg,
    /// Entries after `snap_idx`: (term, D line or configuration entry).
    pub log: Vec<(u64, String)>,
}

/// Stable storage for one node. `save` returns only when the state is durable.
pub trait Durable {
    fn save(&mut self, p: &Persisted) -> io::Result<()>;
    fn load(&mut self) -> io::Result<Persisted>;
}

/// In-memory stable storage that survives a simulated crash (the node is
/// dropped, the `Rc` is kept and handed to the restarted node).
#[derive(Clone, Default)]
pub struct MemDurable(pub Rc<RefCell<Persisted>>);

impl Durable for MemDurable {
    fn save(&mut self, p: &Persisted) -> io::Result<()> {
        *self.0.borrow_mut() = p.clone();
        Ok(())
    }
    fn load(&mut self) -> io::Result<Persisted> {
        Ok(self.0.borrow().clone())
    }
}

/// A file: written to a temporary file, synced, renamed over the old one,
/// and the directory synced, so a crash leaves either state whole.
pub struct FileDurable(pub std::path::PathBuf);

impl Durable for FileDurable {
    fn save(&mut self, p: &Persisted) -> io::Result<()> {
        let tmp = self.0.with_extension("tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(serde_json::to_string(p).map_err(io::Error::other)?.as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.0)?;
        #[cfg(unix)]
        if let Some(d) = self.0.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::File::open(d)?.sync_all()?;
        }
        Ok(())
    }
    fn load(&mut self) -> io::Result<Persisted> {
        match std::fs::read_to_string(&self.0) {
            Ok(s) => serde_json::from_str(&s).map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Persisted::default()),
            Err(e) => Err(e),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

pub struct Node {
    pub id: usize,
    init: Cfg,
    d: Box<dyn Durable>,
    pub p: Persisted,
    pub commit: u64,
    pub role: Role,
    votes: BTreeSet<usize>,
    next: BTreeMap<usize, u64>,
    matched: BTreeMap<usize, u64>,
    snap_off: BTreeMap<usize, usize>,
    /// Chunks of a snapshot being received: (last_idx, bytes so far).
    snap_buf: Option<(u64, String)>,
    /// Bytes per snapshot chunk.
    pub snap_chunk: usize,
    elapsed: u64,
    pub timeout: u64,
    pub heartbeat: u64,
    /// Messages to send: (to, message).
    pub out: Vec<(usize, Msg)>,
    /// Set when stable storage failed: the node stops (like HS-3).
    pub failed: bool,
}

const MAX_BATCH: usize = 64;

impl Node {
    /// Start or restart node `id` of the cluster `0..n` from its stable storage.
    pub fn new(id: usize, n: usize, d: Box<dyn Durable>) -> io::Result<Node> {
        Node::with_members(id, (0..n).collect(), d)
    }
    /// Start or restart with an initial configuration (empty for a node that
    /// joins later and learns its configuration from the leader's log).
    pub fn with_members(id: usize, members: Vec<usize>, mut d: Box<dyn Durable>) -> io::Result<Node> {
        let p = d.load()?;
        let commit = p.snap_idx;
        let init = Cfg { old: members, new: None };
        Ok(Node { id, init, d, p, commit, role: Role::Follower, votes: BTreeSet::new(), next: BTreeMap::new(), matched: BTreeMap::new(), snap_off: BTreeMap::new(), snap_buf: None, snap_chunk: 1 << 16, elapsed: 0, timeout: 10 + 4 * id as u64, heartbeat: 2, out: vec![], failed: false })
    }
    /// The latest configuration in this node's log (committed or not).
    pub fn cfg(&self) -> Cfg {
        self.p.log.iter().rev().find_map(|(_, l)| cfg_of(l)).unwrap_or_else(|| if self.p.snap_cfg.old.is_empty() { self.init.clone() } else { self.p.snap_cfg.clone() })
    }
    fn cfg_upto(&self, idx: u64) -> Cfg {
        let k = (idx.saturating_sub(self.p.snap_idx) as usize).min(self.p.log.len());
        self.p.log[..k].iter().rev().find_map(|(_, l)| cfg_of(l)).unwrap_or_else(|| if self.p.snap_cfg.old.is_empty() { self.init.clone() } else { self.p.snap_cfg.clone() })
    }
    pub fn term(&self) -> u64 {
        self.p.term
    }
    pub fn last_idx(&self) -> u64 {
        self.p.snap_idx + self.p.log.len() as u64
    }
    fn term_at(&self, i: u64) -> Option<u64> {
        if i == self.p.snap_idx {
            return Some(self.p.snap_term);
        }
        if i < self.p.snap_idx || i > self.last_idx() {
            return None;
        }
        Some(self.p.log[(i - self.p.snap_idx - 1) as usize].0)
    }
    fn last_term(&self) -> u64 {
        self.term_at(self.last_idx()).unwrap_or(0)
    }
    fn persist(&mut self) -> bool {
        if self.failed {
            return false;
        }
        if self.d.save(&self.p).is_err() {
            self.failed = true;
            self.out.clear();
        }
        !self.failed
    }
    fn send(&mut self, to: usize, m: Msg) {
        if !self.failed {
            self.out.push((to, m));
        }
    }
    fn others(&self) -> Vec<usize> {
        self.cfg().members().into_iter().filter(|&i| i != self.id).collect()
    }

    /// One logical tick: heartbeats for a leader, election timeout otherwise.
    pub fn tick(&mut self) {
        if self.failed {
            return;
        }
        self.elapsed += 1;
        match self.role {
            Role::Leader => {
                if self.elapsed >= self.heartbeat {
                    self.elapsed = 0;
                    for o in self.others() {
                        self.send_append(o);
                    }
                }
            }
            _ => {
                if self.elapsed >= self.timeout && self.cfg().members().contains(&self.id) {
                    self.start_election();
                }
            }
        }
    }

    pub fn start_election(&mut self) {
        self.elapsed = 0;
        self.role = Role::Candidate;
        self.p.term += 1;
        self.p.voted = Some(self.id);
        if !self.persist() {
            return; // durable term and vote before asking
        }
        self.votes = [self.id].into();
        if self.cfg().quorum(&self.votes) {
            self.become_leader();
            return;
        }
        let (term, last_idx, last_term) = (self.p.term, self.last_idx(), self.last_term());
        for o in self.others() {
            self.send(o, Msg::Vote { term, cand: self.id, last_idx, last_term });
        }
    }

    fn become_leader(&mut self) {
        self.role = Role::Leader;
        self.elapsed = 0;
        let last = self.last_idx();
        self.next = self.others().into_iter().map(|o| (o, last + 1)).collect();
        self.matched = [(self.id, last)].into();
        self.snap_off.clear();
        for o in self.others() {
            self.send_append(o);
        }
    }

    fn send_append(&mut self, to: usize) {
        let term = self.p.term;
        let li = self.last_idx(); let next = *self.next.entry(to).or_insert(li + 1);
        if next <= self.p.snap_idx {
            let off = *self.snap_off.get(&to).unwrap_or(&0);
            let b = self.p.snap.as_bytes();
            let mut end = (off + self.snap_chunk).min(b.len());
            while !self.p.snap.is_char_boundary(end) {
                end += 1;
            }
            let m = Msg::Snap { term, leader: self.id, last_idx: self.p.snap_idx, last_term: self.p.snap_term, offset: off, data: self.p.snap[off..end].to_string(), done: end == b.len(), cfg: self.p.snap_cfg.clone() };
            self.send(to, m);
            return;
        }
        let prev_idx = next - 1;
        let prev_term = self.term_at(prev_idx).unwrap_or(0);
        let from = (prev_idx - self.p.snap_idx) as usize;
        let entries: Vec<(u64, String)> = self.p.log[from..].iter().take(MAX_BATCH).cloned().collect();
        self.send(to, Msg::Append { term, leader: self.id, prev_idx, prev_term, entries, commit: self.commit });
    }

    /// Leader only: append a D line at the current term; returns its index.
    pub fn propose(&mut self, line: &str) -> Option<u64> {
        if self.role != Role::Leader || self.failed {
            return None;
        }
        self.p.log.push((self.p.term, line.to_string()));
        if !self.persist() {
            return None;
        }
        self.matched.insert(self.id, self.last_idx());
        self.advance_commit();
        for o in self.others() {
            self.send_append(o);
        }
        Some(self.last_idx())
    }

    fn advance_commit(&mut self) {
        let cfg = self.cfg();
        let mut n = self.last_idx();
        while n > self.commit {
            let have: BTreeSet<usize> = self.matched.iter().filter(|(_, m)| **m >= n).map(|(i, _)| *i).collect();
            if self.term_at(n) == Some(self.p.term) && cfg.quorum(&have) {
                self.commit = n;
                break;
            }
            n -= 1;
        }
        self.after_commit();
    }

    /// Joint consensus, leader side: C_old,new committed -> append C_new;
    /// C_new committed and this node not in it -> step down.
    fn after_commit(&mut self) {
        if self.role != Role::Leader {
            return;
        }
        let latest = self.cfg();
        let committed = self.cfg_upto(self.commit);
        if let Some(new) = latest.new.clone()
            && committed == latest
        {
            let c = Cfg { old: new, new: None };
            self.p.log.push((self.p.term, format!("{CFG_PREFIX}{}", serde_json::to_string(&c).unwrap())));
            if self.persist() {
                self.matched.insert(self.id, self.last_idx());
                for o in self.others() {
                    let li = self.last_idx(); self.next.entry(o).or_insert(li);
                    self.send_append(o);
                }
            }
        } else if latest.new.is_none() && committed == latest && !latest.old.contains(&self.id) {
            self.role = Role::Follower;
        }
    }

    /// Leader only: start a membership change to `new` (joint consensus).
    /// `None` while another change is in progress.
    pub fn change_membership(&mut self, new: Vec<usize>) -> Option<u64> {
        let cur = self.cfg();
        if self.role != Role::Leader || cur.new.is_some() || self.cfg_upto(self.commit) != cur {
            return None;
        }
        let c = Cfg { old: cur.old, new: Some(new) };
        let i = self.propose(&format!("{CFG_PREFIX}{}", serde_json::to_string(&c).unwrap()))?;
        for o in self.others() {
            if !self.next.contains_key(&o) {
                self.next.insert(o, 1);
                self.send_append(o);
            }
        }
        Some(i)
    }

    /// Compact the log through `upto` (at most the commit index) into a
    /// snapshot whose bytes the caller made from the state at `upto`.
    pub fn compact(&mut self, upto: u64, data: String) -> bool {
        if upto <= self.p.snap_idx || upto > self.commit {
            return false;
        }
        let t = self.term_at(upto).unwrap();
        self.p.snap_cfg = self.cfg_upto(upto);
        let k = (upto - self.p.snap_idx) as usize;
        self.p.log.drain(..k);
        self.p.snap_idx = upto;
        self.p.snap_term = t;
        self.p.snap = data;
        self.persist()
    }

    /// The committed D lines this node holds after its snapshot.
    pub fn committed(&self) -> Vec<String> {
        let k = (self.commit - self.p.snap_idx) as usize;
        self.p.log[..k].iter().filter(|(_, l)| !l.starts_with(CFG_PREFIX)).map(|(_, l)| l.clone()).collect()
    }

    pub fn handle(&mut self, from: usize, m: Msg) {
        if self.failed {
            return;
        }
        if m.term() > self.p.term {
            self.p.term = m.term();
            self.p.voted = None;
            self.role = Role::Follower;
            if !self.persist() {
                return;
            }
        }
        let term = self.p.term;
        match m {
            Msg::Vote { term: t, cand, last_idx, last_term } => {
                let up_to_date = (last_term, last_idx) >= (self.last_term(), self.last_idx());
                let free = self.p.voted.is_none() || self.p.voted == Some(cand);
                let granted = t == term && free && up_to_date;
                if granted {
                    self.p.voted = Some(cand);
                    if !self.persist() {
                        return; // durable vote before the grant leaves
                    }
                    self.elapsed = 0;
                }
                self.send(from, Msg::VoteResp { term, granted });
            }
            Msg::VoteResp { term: t, granted } => {
                if self.role == Role::Candidate && t == term && granted {
                    self.votes.insert(from);
                    if self.cfg().quorum(&self.votes) {
                        self.become_leader();
                    }
                }
            }
            Msg::Append { term: t, prev_idx, prev_term, entries, commit, .. } => {
                if t < term {
                    self.send(from, Msg::AppendResp { term, ok: false, match_idx: 0 });
                    return;
                }
                self.role = Role::Follower;
                self.elapsed = 0;
                // Entries at or below our snapshot are committed: skip them.
                let (mut prev_idx, mut prev_term, mut entries) = (prev_idx, prev_term, entries);
                if prev_idx < self.p.snap_idx {
                    let skip = ((self.p.snap_idx - prev_idx) as usize).min(entries.len());
                    entries.drain(..skip);
                    prev_idx = self.p.snap_idx;
                    prev_term = self.p.snap_term;
                }
                if self.term_at(prev_idx) != Some(prev_term) {
                    let hint = self.commit.min(prev_idx.saturating_sub(1));
                    self.send(from, Msg::AppendResp { term, ok: false, match_idx: hint });
                    return;
                }
                let mut changed = false;
                for (j, e) in entries.iter().enumerate() {
                    let i = prev_idx + 1 + j as u64;
                    match self.term_at(i) {
                        Some(et) if et == e.0 => {}
                        Some(_) => {
                            let keep = (i - self.p.snap_idx - 1) as usize;
                            self.p.log.truncate(keep);
                            self.p.log.push(e.clone());
                            changed = true;
                        }
                        None => {
                            self.p.log.push(e.clone());
                            changed = true;
                        }
                    }
                }
                if changed && !self.persist() {
                    return; // durable before the ack
                }
                let last_new = prev_idx + entries.len() as u64;
                if commit > self.commit {
                    self.commit = commit.min(last_new).max(self.commit);
                }
                self.send(from, Msg::AppendResp { term, ok: true, match_idx: last_new });
            }
            Msg::AppendResp { term: t, ok, match_idx } => {
                if self.role != Role::Leader || t != term {
                    return;
                }
                if ok {
                    let m = self.matched.get(&from).copied().unwrap_or(0).max(match_idx);
                    self.matched.insert(from, m);
                    self.next.insert(from, m + 1);
                    self.snap_off.remove(&from);
                    self.advance_commit();
                    if m < self.last_idx() {
                        self.send_append(from);
                    }
                } else {
                    let cur = self.next.get(&from).copied().unwrap_or(1);
                    self.next.insert(from, (match_idx + 1).min(cur.saturating_sub(1)).max(1));
                    self.send_append(from);
                }
            }
            Msg::SnapAck { term: t, next_off } => {
                if self.role == Role::Leader && t == term {
                    self.snap_off.insert(from, next_off);
                    self.send_append(from);
                }
            }
            Msg::Snap { term: t, last_idx, last_term, offset, data, done, cfg, .. } => {
                if t < term {
                    self.send(from, Msg::AppendResp { term, ok: false, match_idx: 0 });
                    return;
                }
                self.role = Role::Follower;
                self.elapsed = 0;
                // Chunks: accumulate in order; an out-of-order chunk is
                // answered with the offset we need.
                let buf = match self.snap_buf.take() {
                    Some((i, b)) if i == last_idx && offset > 0 => b,
                    _ if offset == 0 => String::new(),
                    other => {
                        let have = other.as_ref().filter(|(i, _)| *i == last_idx).map_or(0, |(_, b)| b.len());
                        self.snap_buf = other;
                        self.send(from, Msg::SnapAck { term, next_off: have });
                        return;
                    }
                };
                if offset != buf.len() {
                    let n = buf.len();
                    self.snap_buf = Some((last_idx, buf));
                    self.send(from, Msg::SnapAck { term, next_off: n });
                    return;
                }
                let mut buf = buf;
                buf.push_str(&data);
                if !done {
                    let n = buf.len();
                    self.snap_buf = Some((last_idx, buf));
                    self.send(from, Msg::SnapAck { term, next_off: n });
                    return;
                }
                let data = buf;
                if last_idx > self.p.snap_idx {
                    self.p.snap_cfg = cfg;
                    if self.term_at(last_idx) == Some(last_term) {
                        let k = (last_idx - self.p.snap_idx) as usize;
                        self.p.log.drain(..k);
                    } else {
                        self.p.log.clear();
                    }
                    self.p.snap_idx = last_idx;
                    self.p.snap_term = last_term;
                    self.p.snap = data;
                    self.commit = self.commit.max(last_idx);
                    if !self.persist() {
                        return;
                    }
                }
                self.send(from, Msg::AppendResp { term, ok: true, match_idx: self.p.snap_idx });
            }
        }
    }
}

/// How messages move between nodes. Delivery may lose, delay, duplicate
/// or reorder messages; Raft is safe under all of them.
pub trait Transport {
    fn send(&mut self, to: usize, m: &Msg);
    /// Messages that have arrived for this node: (from, message).
    fn poll(&mut self) -> Vec<(usize, Msg)>;
}

/// The loopback switch shared by the nodes of one process.
#[derive(Default)]
pub struct Switch {
    q: Vec<(usize, usize, Msg)>,
    /// Directed links that drop everything: (from, to).
    pub cut: BTreeSet<(usize, usize)>,
    /// Shuffle each node's arrivals.
    pub reorder: bool,
    /// Drop one message in every `k` sent (deterministic).
    pub drop_every: Option<u64>,
    pub seed: u64,
    sent: u64,
}

impl Switch {
    pub fn shared() -> Rc<RefCell<Switch>> {
        Rc::new(RefCell::new(Switch { seed: 0x9e37_79b9, ..Default::default() }))
    }
    /// Cut every link between the two groups, both ways.
    pub fn partition(&mut self, a: &[usize], b: &[usize]) {
        for &x in a {
            for &y in b {
                self.cut.insert((x, y));
                self.cut.insert((y, x));
            }
        }
    }
    pub fn heal(&mut self) {
        self.cut.clear();
    }
    fn rnd(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        self.seed
    }
}

pub struct Loopback {
    pub me: usize,
    pub sw: Rc<RefCell<Switch>>,
}

impl Transport for Loopback {
    fn send(&mut self, to: usize, m: &Msg) {
        let mut s = self.sw.borrow_mut();
        s.sent += 1;
        if s.cut.contains(&(self.me, to)) || s.drop_every.is_some_and(|k| s.sent % k == 0) {
            return;
        }
        s.q.push((self.me, to, m.clone()));
    }
    fn poll(&mut self) -> Vec<(usize, Msg)> {
        let mut s = self.sw.borrow_mut();
        let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut s.q).into_iter().partition(|x| x.1 == self.me);
        s.q = rest;
        let mut v: Vec<(usize, Msg)> = mine.into_iter().map(|(f, _, m)| (f, m)).collect();
        if s.reorder {
            for i in (1..v.len()).rev() {
                let j = (s.rnd() % (i as u64 + 1)) as usize;
                v.swap(i, j);
            }
        }
        v
    }
}

/// TCP: each frame is a 4-byte big-endian length, then the JSON of
/// `(from, Msg)`. Connections are opened lazily and dropped on error; a
/// lost message is retransmitted by Raft's heartbeats.
pub struct Tcp {
    me: usize,
    peers: Vec<SocketAddr>,
    conns: BTreeMap<usize, TcpStream>,
    rx: mpsc::Receiver<(usize, Msg)>,
}

impl Tcp {
    /// Listen on `listener` (already bound) and talk to `peers` (indexed by node id).
    pub fn new(me: usize, listener: TcpListener, peers: Vec<SocketAddr>) -> Tcp {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for s in listener.incoming() {
                let Ok(mut s) = s else { continue };
                let tx = tx.clone();
                std::thread::spawn(move || read_frames(&mut s, &tx, None));
            }
        });
        Tcp { me, peers, conns: BTreeMap::new(), rx }
    }
}

impl Transport for Tcp {
    fn send(&mut self, to: usize, m: &Msg) {
        if !self.conns.contains_key(&to) {
            match TcpStream::connect_timeout(&self.peers[to], std::time::Duration::from_millis(200)) {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    self.conns.insert(to, s);
                }
                Err(_) => return,
            }
        }
        let s = self.conns.get_mut(&to).unwrap();
        if s.write_all(&frame(self.me, m)).is_err() {
            self.conns.remove(&to);
        }
    }
    fn poll(&mut self) -> Vec<(usize, Msg)> {
        self.rx.try_iter().collect()
    }
}

/// Read length-prefixed `(from, Msg)` frames from `s` into `tx`; when
/// `only` is set, a frame whose `from` differs is dropped (the TLS peer key
/// fixes who may speak for which node).
fn read_frames<R: Read>(s: &mut R, tx: &mpsc::Sender<(usize, Msg)>, only: Option<usize>) {
    loop {
        let mut len = [0u8; 4];
        if s.read_exact(&mut len).is_err() {
            return;
        }
        let n = u32::from_be_bytes(len) as usize;
        if n > 64 << 20 {
            return;
        }
        let mut b = vec![0; n];
        if s.read_exact(&mut b).is_err() {
            return;
        }
        let Ok(x) = serde_json::from_slice::<(usize, Msg)>(&b) else { return };
        if only.is_some_and(|o| o != x.0) {
            continue;
        }
        if tx.send(x).is_err() {
            return;
        }
    }
}

fn frame(me: usize, m: &Msg) -> Vec<u8> {
    let b = serde_json::to_vec(&(me, m)).unwrap();
    let mut f = (b.len() as u32).to_be_bytes().to_vec();
    f.extend_from_slice(&b);
    f
}

/// Mutual TLS 1.3 with pinned Ed25519 keys (`crate::tls`, the P4 channel
/// identity): a node accepts a connection only from a key in its peer
/// list, and accepts a frame only when the frame's sender id is the id
/// pinned to that key. A node dials a peer only if the peer presents the
/// key pinned for it. Built where build.rs sets `rce_tls`.
#[cfg(rce_tls)]
pub struct TlsTransport {
    me: usize,
    peers: Vec<(SocketAddr, [u8; 32])>,
    client: Vec<Option<std::sync::Arc<rustls::ClientConfig>>>,
    conns: BTreeMap<usize, rustls::StreamOwned<rustls::ClientConnection, TcpStream>>,
    rx: mpsc::Receiver<(usize, Msg)>,
}

#[cfg(rce_tls)]
impl TlsTransport {
    pub fn new(me: usize, listener: TcpListener, peers: Vec<(SocketAddr, [u8; 32])>, id: &crate::tls::Identity) -> Result<TlsTransport, String> {
        let keys: Vec<[u8; 32]> = peers.iter().map(|p| p.1).collect();
        let server = crate::tls::server_config(id, keys.clone())?;
        let client = peers.iter().map(|p| crate::tls::client_config(id, p.1).ok()).collect();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for s in listener.incoming() {
                let Ok(s) = s else { continue };
                let (tx, server, keys) = (tx.clone(), server.clone(), keys.clone());
                std::thread::spawn(move || {
                    let Ok(conn) = rustls::ServerConnection::new(server) else { return };
                    let mut t = rustls::StreamOwned::new(conn, s);
                    while t.conn.is_handshaking() {
                        if t.conn.complete_io(&mut t.sock).is_err() {
                            return;
                        }
                    }
                    let key = t.conn.peer_certificates().and_then(|c| c.first()).and_then(|c| crate::tls::ed25519_key_of(c.as_ref()));
                    let Some(who) = key.and_then(|k| keys.iter().position(|p| *p == k)) else { return };
                    read_frames(&mut t, &tx, Some(who));
                });
            }
        });
        Ok(TlsTransport { me, peers, client, conns: BTreeMap::new(), rx })
    }
}

#[cfg(rce_tls)]
impl Transport for TlsTransport {
    fn send(&mut self, to: usize, m: &Msg) {
        if !self.conns.contains_key(&to) {
            let Some(Some(cfg)) = self.client.get(to).cloned() else { return };
            let Ok(sock) = TcpStream::connect_timeout(&self.peers[to].0, std::time::Duration::from_millis(200)) else { return };
            let _ = sock.set_nodelay(true);
            let Ok(name) = rustls::pki_types::ServerName::try_from(crate::tls::SERVER_NAME) else { return };
            let Ok(conn) = rustls::ClientConnection::new(cfg, name) else { return };
            self.conns.insert(to, rustls::StreamOwned::new(conn, sock));
        }
        let s = self.conns.get_mut(&to).unwrap();
        if s.write_all(&frame(self.me, m)).and_then(|_| s.flush()).is_err() {
            self.conns.remove(&to);
        }
    }
    fn poll(&mut self) -> Vec<(usize, Msg)> {
        self.rx.try_iter().collect()
    }
}

/// A node with its transport.
pub struct Peer {
    pub node: Node,
    pub t: Box<dyn Transport>,
}

impl Peer {
    /// Deliver what arrived, tick once, send what the node produced.
    pub fn step(&mut self) {
        for (f, m) in self.t.poll() {
            self.node.handle(f, m);
        }
        self.node.tick();
        self.flush();
    }
    pub fn flush(&mut self) {
        for (to, m) in std::mem::take(&mut self.node.out) {
            self.t.send(to, &m);
        }
    }
}

/// A loopback cluster in one process; `None` is a crashed node.
pub struct Cluster {
    pub sw: Rc<RefCell<Switch>>,
    pub peers: Vec<Option<Peer>>,
    pub disks: Vec<MemDurable>,
    initial: usize,
}

impl Cluster {
    pub fn new(n: usize) -> Cluster {
        let sw = Switch::shared();
        let disks: Vec<MemDurable> = (0..n).map(|_| MemDurable::default()).collect();
        let peers = (0..n).map(|i| Some(Peer { node: Node::new(i, n, Box::new(disks[i].clone())).unwrap(), t: Box::new(Loopback { me: i, sw: sw.clone() }) })).collect();
        Cluster { sw, peers, disks, initial: n }
    }
    pub fn step(&mut self) {
        for p in self.peers.iter_mut().flatten() {
            p.step();
        }
    }
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
    pub fn crash(&mut self, i: usize) {
        self.peers[i] = None;
        // messages in flight to a crashed node are lost with it
        self.sw.borrow_mut().q.retain(|x| x.1 != i);
    }
    pub fn restart(&mut self, i: usize) {
        // messages sent while it was down were lost with it
        self.sw.borrow_mut().q.retain(|x| x.1 != i);
        let members = if i < self.initial { (0..self.initial).collect() } else { vec![] };
        self.peers[i] = Some(Peer { node: Node::with_members(i, members, Box::new(self.disks[i].clone())).unwrap(), t: Box::new(Loopback { me: i, sw: self.sw.clone() }) });
    }
    /// A new node that joins with no configuration; it learns one from the
    /// leader's log (or snapshot) and votes only once it is a member.
    pub fn add_node(&mut self) -> usize {
        let i = self.peers.len();
        let d = MemDurable::default();
        self.disks.push(d.clone());
        self.peers.push(Some(Peer { node: Node::with_members(i, vec![], Box::new(d)).unwrap(), t: Box::new(Loopback { me: i, sw: self.sw.clone() }) }));
        i
    }
    pub fn node(&self, i: usize) -> &Node {
        &self.peers[i].as_ref().unwrap().node
    }
    pub fn node_mut(&mut self, i: usize) -> &mut Node {
        &mut self.peers[i].as_mut().unwrap().node
    }
    /// The live leader with the highest term, if any.
    pub fn leader(&self) -> Option<usize> {
        self.peers.iter().flatten().filter(|p| p.node.role == Role::Leader).max_by_key(|p| p.node.term()).map(|p| p.node.id)
    }
    /// Run until some node leads, at most `steps` steps.
    pub fn elect(&mut self, steps: usize) -> Option<usize> {
        for _ in 0..steps {
            self.step();
            if let Some(l) = self.leader() {
                return Some(l);
            }
        }
        None
    }
}

/// The host's store on a Raft leader of a loopback [`Cluster`] (E.3 Store):
/// appends are proposed at the barrier, and the barrier succeeds only when
/// every proposed line is committed. A node that lost leadership fails the
/// barrier, so the host stops (HS-3).
pub struct RaftStore {
    pub c: Rc<RefCell<Cluster>>,
    pub me: usize,
    pending: Vec<String>,
    pub max_steps: usize,
}

impl RaftStore {
    pub fn new(c: Rc<RefCell<Cluster>>, me: usize) -> RaftStore {
        RaftStore { c, me, pending: vec![], max_steps: 200 }
    }
}

impl crate::host::Store for RaftStore {
    fn append(&mut self, line: &str) -> io::Result<()> {
        self.pending.push(line.into());
        Ok(())
    }
    fn barrier(&mut self) -> io::Result<()> {
        let mut c = self.c.borrow_mut();
        let mut last = None;
        for l in std::mem::take(&mut self.pending) {
            let Some(p) = c.peers[self.me].as_mut() else { return Err(io::Error::other("node crashed")) };
            last = Some(p.node.propose(&l).ok_or_else(|| io::Error::other("not the leader"))?);
            p.flush();
        }
        let Some(idx) = last else { return Ok(()) };
        let term = c.node(self.me).term();
        for _ in 0..self.max_steps {
            c.step();
            let Some(p) = c.peers[self.me].as_ref() else { return Err(io::Error::other("node crashed")) };
            if p.node.term() != term || p.node.role != Role::Leader {
                return Err(io::Error::other(format!("deposed at term {term}")));
            }
            if p.node.commit >= idx {
                return Ok(());
            }
        }
        Err(io::Error::other(format!("index {idx} not committed (no quorum)")))
    }
    fn caps(&self) -> &'static str {
        "replicated quorum"
    }
}
