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
//! - AppendEntries carries (prev_idx, prev_term); a follower refuses it when
//!   its log has no entry at prev_idx with that term (log matching), deletes
//!   a conflicting suffix, and appends what is missing;
//! - the leader advances the commit index only over entries of its own term
//!   held by a quorum (Raft section 5.4.2);
//! - InstallSnapshot: a leader whose follower needs entries that were
//!   compacted sends its snapshot (opaque bytes; the host uses an HS-5
//!   snapshot), and the follower keeps any suffix that matches.
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
    Snap { term: u64, leader: usize, last_idx: u64, last_term: u64, data: String },
}

impl Msg {
    pub fn term(&self) -> u64 {
        match self {
            Msg::Vote { term, .. } | Msg::VoteResp { term, .. } | Msg::Append { term, .. } | Msg::AppendResp { term, .. } | Msg::Snap { term, .. } => *term,
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
    /// Entries after `snap_idx`: (term, D line).
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
    n: usize,
    d: Box<dyn Durable>,
    pub p: Persisted,
    pub commit: u64,
    pub role: Role,
    votes: BTreeSet<usize>,
    next: Vec<u64>,
    matched: Vec<u64>,
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
    /// Start or restart node `id` of `n` from its stable storage.
    pub fn new(id: usize, n: usize, mut d: Box<dyn Durable>) -> io::Result<Node> {
        let p = d.load()?;
        let commit = p.snap_idx;
        Ok(Node { id, n, d, p, commit, role: Role::Follower, votes: BTreeSet::new(), next: vec![1; n], matched: vec![0; n], elapsed: 0, timeout: 10 + 4 * id as u64, heartbeat: 2, out: vec![], failed: false })
    }
    pub fn quorum(&self) -> usize {
        self.n / 2 + 1
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
        (0..self.n).filter(|&i| i != self.id).collect()
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
                if self.elapsed >= self.timeout {
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
        if self.votes.len() >= self.quorum() {
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
        self.next = vec![self.last_idx() + 1; self.n];
        self.matched = vec![0; self.n];
        self.matched[self.id] = self.last_idx();
        for o in self.others() {
            self.send_append(o);
        }
    }

    fn send_append(&mut self, to: usize) {
        let term = self.p.term;
        if self.next[to] <= self.p.snap_idx {
            let m = Msg::Snap { term, leader: self.id, last_idx: self.p.snap_idx, last_term: self.p.snap_term, data: self.p.snap.clone() };
            self.send(to, m);
            return;
        }
        let prev_idx = self.next[to] - 1;
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
        self.matched[self.id] = self.last_idx();
        self.advance_commit();
        for o in self.others() {
            self.send_append(o);
        }
        Some(self.last_idx())
    }

    fn advance_commit(&mut self) {
        let mut n = self.last_idx();
        while n > self.commit {
            if self.term_at(n) == Some(self.p.term) && self.matched.iter().filter(|&&m| m >= n).count() >= self.quorum() {
                self.commit = n;
                break;
            }
            n -= 1;
        }
    }

    /// Compact the log through `upto` (at most the commit index) into a
    /// snapshot whose bytes the caller made from the state at `upto`.
    pub fn compact(&mut self, upto: u64, data: String) -> bool {
        if upto <= self.p.snap_idx || upto > self.commit {
            return false;
        }
        let t = self.term_at(upto).unwrap();
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
        self.p.log[..k].iter().map(|(_, l)| l.clone()).collect()
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
                    if self.votes.len() >= self.quorum() {
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
                    self.matched[from] = self.matched[from].max(match_idx);
                    self.next[from] = self.matched[from] + 1;
                    self.advance_commit();
                    if self.next[from] <= self.last_idx() {
                        self.send_append(from);
                    }
                } else {
                    self.next[from] = (match_idx + 1).min(self.next[from].saturating_sub(1)).max(1);
                    self.send_append(from);
                }
            }
            Msg::Snap { term: t, last_idx, last_term, data, .. } => {
                if t < term {
                    self.send(from, Msg::AppendResp { term, ok: false, match_idx: 0 });
                    return;
                }
                self.role = Role::Follower;
                self.elapsed = 0;
                if last_idx > self.p.snap_idx {
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
                std::thread::spawn(move || {
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
                        if tx.send(x).is_err() {
                            return;
                        }
                    }
                });
            }
        });
        Tcp { me, peers, conns: BTreeMap::new(), rx }
    }
}

impl Transport for Tcp {
    fn send(&mut self, to: usize, m: &Msg) {
        let b = serde_json::to_vec(&(self.me, m)).unwrap();
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
        let mut f = (b.len() as u32).to_be_bytes().to_vec();
        f.extend_from_slice(&b);
        if s.write_all(&f).is_err() {
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
}

impl Cluster {
    pub fn new(n: usize) -> Cluster {
        let sw = Switch::shared();
        let disks: Vec<MemDurable> = (0..n).map(|_| MemDurable::default()).collect();
        let peers = (0..n).map(|i| Some(Peer { node: Node::new(i, n, Box::new(disks[i].clone())).unwrap(), t: Box::new(Loopback { me: i, sw: sw.clone() }) })).collect();
        Cluster { sw, peers, disks }
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
        let n = self.peers.len();
        self.peers[i] = Some(Peer { node: Node::new(i, n, Box::new(self.disks[i].clone())).unwrap(), t: Box::new(Loopback { me: i, sw: self.sw.clone() }) });
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
