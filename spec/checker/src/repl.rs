//! E.2b replicated D: a reference model of the replicated log a host
//! appends through, and the admitter as a role that fails over.
//!
//! The model follows the Raft rules E.2b cites, on whole logs:
//! - a candidate for term t needs grants from a quorum of replicas, and a
//!   replica grants (and promises t) only when it has promised a lower term;
//! - the new leader starts from the most up-to-date log among the granting
//!   quorum, by (term of the last write, length), so it holds every entry
//!   that a quorum had accepted;
//! - an append at term t is accepted by a replica only while that replica has
//!   promised no higher term (the fence), and is committed when a quorum
//!   accepted it. The leader's [`Store::barrier`] succeeds only then, so an
//!   admitter acks only committed entries and a deposed admitter's barrier
//!   fails, which stops it (HS-3, T39).
//!
//! Messages are function calls; a replica that is down neither grants nor
//! accepts. Real transports, persistence of `promised` and log shipping by
//! suffix are left to the host (E.2b lists what they must preserve).
use crate::host::{Store, verify_links};
use std::cell::RefCell;
use std::io;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
pub struct Replica {
    /// The highest term this replica has granted or accepted.
    pub promised: u64,
    /// The term of the leader that last wrote `log`.
    pub last_term: u64,
    pub log: Vec<String>,
    pub up: bool,
}

#[derive(Debug)]
pub struct Cluster {
    pub replicas: Vec<Replica>,
    /// Length of the committed prefix (accepted by a quorum).
    pub commit: usize,
}

pub type Shared = Rc<RefCell<Cluster>>;

impl Cluster {
    pub fn new(n: usize) -> Shared {
        Rc::new(RefCell::new(Cluster { replicas: vec![Replica { up: true, ..Default::default() }; n], commit: 0 }))
    }
    pub fn quorum(&self) -> usize {
        self.replicas.len() / 2 + 1
    }
    /// Raft-style election for `term`. Returns the log the new leader starts from.
    pub fn elect(&mut self, term: u64) -> Result<Vec<String>, String> {
        let grants: Vec<usize> = (0..self.replicas.len()).filter(|&i| self.replicas[i].up && self.replicas[i].promised < term).collect();
        if grants.len() < self.quorum() {
            return Err(format!("term {term}: {} grants, quorum is {}", grants.len(), self.quorum()));
        }
        for &i in &grants {
            self.replicas[i].promised = term;
        }
        let best = grants.iter().max_by_key(|&&i| (self.replicas[i].last_term, self.replicas[i].log.len())).unwrap();
        Ok(self.replicas[*best].log.clone())
    }
    /// The committed prefix of D, as file text.
    pub fn committed_text(&self) -> String {
        let best = self.replicas.iter().filter(|r| r.log.len() >= self.commit).max_by_key(|r| (r.last_term, r.log.len()));
        best.map(|r| r.log[..self.commit].iter().map(|l| format!("{l}\n")).collect()).unwrap_or_default()
    }
}

/// The store of the admitter holding `term`: appends go to its own copy of D,
/// and the barrier replicates it under the term fence.
pub struct LeaderStore {
    c: Shared,
    term: u64,
    log: Vec<String>,
}

impl LeaderStore {
    pub fn new(c: Shared, term: u64, log: Vec<String>) -> LeaderStore {
        LeaderStore { c, term, log }
    }
}

impl Store for LeaderStore {
    fn append(&mut self, line: &str) -> io::Result<()> {
        self.log.push(line.into());
        Ok(())
    }
    fn barrier(&mut self) -> io::Result<()> {
        let refs: Vec<&str> = self.log.iter().map(String::as_str).collect();
        if verify_links(&refs) != refs.len() {
            return Err(io::Error::other("leader log does not verify"));
        }
        let mut c = self.c.borrow_mut();
        let q = c.quorum();
        let mut acks = 0;
        for r in c.replicas.iter_mut().filter(|r| r.up) {
            if r.promised > self.term {
                continue; // fenced: a higher term was granted here
            }
            r.promised = self.term;
            r.last_term = self.term;
            r.log = self.log.clone();
            acks += 1;
        }
        if acks < q {
            return Err(io::Error::other(format!("term {}: {acks} replicas accepted, quorum is {q} (fenced or partitioned)", self.term)));
        }
        c.commit = self.log.len();
        Ok(())
    }
    fn caps(&self) -> &'static str {
        "replicated quorum"
    }
}
