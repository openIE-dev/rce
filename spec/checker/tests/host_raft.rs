//! E.2b: Raft transport for D. Partitions, reordering and loss, crash and
//! restart with durable term and vote, snapshot install, TCP, and the host
//! admitting through a Raft leader across a failover.
use rce_spec_check::ast::{Arg, Expr};
use rce_spec_check::host::{Host, Sourced, replay};
use rce_spec_check::parse::parse;
use rce_spec_check::raft::*;
use std::cell::RefCell;
use std::rc::Rc;

fn agree(c: &Cluster) {
    // State machine safety on what is committed: every pair of live nodes
    // agrees on the common committed prefix.
    let logs: Vec<Vec<String>> = c.peers.iter().flatten().map(|p| p.node.committed()).collect();
    for a in &logs {
        for b in &logs {
            let k = a.len().min(b.len());
            assert_eq!(a[..k], b[..k]);
        }
    }
}

fn propose_all(c: &mut Cluster, from: usize, to: usize) {
    let l = c.leader().unwrap();
    for i in from..to {
        let p = c.peers[l].as_mut().unwrap();
        p.node.propose(&format!("e{i}")).unwrap();
        p.flush();
        c.step();
    }
}

#[test]
fn replicates_and_commits_on_every_node() {
    let mut c = Cluster::new(3);
    c.elect(100).expect("a leader");
    propose_all(&mut c, 0, 20);
    c.run(30);
    for i in 0..3 {
        assert_eq!(c.node(i).committed().len(), 20, "node {i}");
    }
    agree(&c);
}

#[test]
fn partitioned_leader_cannot_commit_and_its_suffix_is_replaced() {
    let mut c = Cluster::new(5);
    let old = c.elect(100).unwrap();
    propose_all(&mut c, 0, 5);
    c.run(20);
    let committed = c.node(old).committed();
    assert_eq!(committed.len(), 5);
    let minority: Vec<usize> = vec![old, (old + 1) % 5];
    let majority: Vec<usize> = (0..5).filter(|i| !minority.contains(i)).collect();
    c.sw.borrow_mut().partition(&minority, &majority);
    // The old leader keeps accepting proposals but cannot commit them.
    for i in 0..3 {
        c.node_mut(old).propose(&format!("stale{i}"));
        c.peers[old].as_mut().unwrap().flush();
    }
    c.run(80);
    assert_eq!(c.node(old).commit, 5, "no quorum, no commit");
    let new = c.leader().unwrap();
    assert!(majority.contains(&new) && c.node(new).term() > c.node(old).term());
    propose_all(&mut c, 5, 8);
    c.sw.borrow_mut().heal();
    c.run(80);
    for i in 0..5 {
        let l = c.node(i).committed();
        assert_eq!(l[..5], committed[..], "committed prefix kept on node {i}");
        assert!(!l.iter().any(|x| x.starts_with("stale")), "uncommitted suffix of the deposed leader replaced");
        assert_eq!(l.len(), 8);
    }
    agree(&c);
}

#[test]
fn converges_under_reordering_and_loss() {
    let mut c = Cluster::new(3);
    c.sw.borrow_mut().reorder = true;
    c.sw.borrow_mut().drop_every = Some(5);
    c.elect(300).unwrap();
    for i in 0..15 {
        if let Some(l) = c.leader() {
            c.node_mut(l).propose(&format!("e{i}"));
            c.peers[l].as_mut().unwrap().flush();
        }
        c.run(3);
        agree(&c);
    }
    c.sw.borrow_mut().drop_every = None;
    c.run(100);
    agree(&c);
    let n = c.node(c.leader().unwrap()).committed().len();
    assert!(n >= 10, "only {n} committed");
    for i in 0..3 {
        assert_eq!(c.node(i).committed().len(), n);
    }
}

#[test]
fn crash_restart_keeps_term_vote_and_log() {
    let mut c = Cluster::new(3);
    let l = c.elect(100).unwrap();
    propose_all(&mut c, 0, 6);
    c.run(20);
    let f = (l + 1) % 3;
    let term = c.node(f).term();
    c.crash(f);
    propose_all(&mut c, 6, 10);
    c.restart(f);
    assert_eq!(c.node(f).term(), term, "term is durable");
    assert!(c.node(f).p.log.len() >= 6, "log is durable");
    c.run(40);
    assert_eq!(c.node(f).committed().len(), 10, "restarted follower caught up");
    agree(&c);
}

#[test]
fn durable_vote_is_not_given_twice_in_a_term() {
    let disk = MemDurable::default();
    let mut n = Node::new(0, 3, Box::new(disk.clone())).unwrap();
    n.handle(1, Msg::Vote { term: 5, cand: 1, last_idx: 0, last_term: 0 });
    assert_eq!(n.out.pop().unwrap().1, Msg::VoteResp { term: 5, granted: true });
    drop(n); // crash after the grant
    let mut n = Node::new(0, 3, Box::new(disk)).unwrap();
    n.handle(2, Msg::Vote { term: 5, cand: 2, last_idx: 9, last_term: 4 });
    assert_eq!(n.out.pop().unwrap().1, Msg::VoteResp { term: 5, granted: false });
}

#[test]
fn log_matching_appends_no_gap_and_truncates_a_conflict() {
    let mut n = Node::new(0, 3, Box::new(MemDurable::default())).unwrap();
    let e = |t: u64, s: &str| (t, s.to_string());
    n.handle(1, Msg::Append { term: 1, leader: 1, prev_idx: 0, prev_term: 0, entries: vec![e(1, "a"), e(1, "b"), e(1, "c")], commit: 1 });
    assert!(matches!(n.out.pop().unwrap().1, Msg::AppendResp { ok: true, match_idx: 3, .. }));
    // prev (5, 1) is missing: not appended
    n.handle(1, Msg::Append { term: 1, leader: 1, prev_idx: 5, prev_term: 1, entries: vec![e(1, "x")], commit: 1 });
    assert!(matches!(n.out.pop().unwrap().1, Msg::AppendResp { ok: false, .. }));
    // a term-2 leader overwrites entry 2 onwards
    n.handle(2, Msg::Append { term: 2, leader: 2, prev_idx: 1, prev_term: 1, entries: vec![e(2, "B")], commit: 2 });
    assert_eq!(n.p.log, vec![e(1, "a"), e(2, "B")]);
    assert_eq!(n.commit, 2);
}

#[test]
fn snapshot_installs_on_a_lagging_follower() {
    let mut c = Cluster::new(3);
    let l = c.elect(100).unwrap();
    let f = (l + 1) % 3;
    c.crash(f);
    propose_all(&mut c, 0, 12);
    c.run(10);
    let commit = c.node(l).commit;
    assert!(commit >= 12);
    assert!(c.node_mut(l).compact(10, "state@10".into()));
    c.restart(f);
    c.run(60);
    let n = c.node(f);
    assert_eq!(n.p.snap_idx, 10);
    assert_eq!(n.p.snap, "state@10");
    assert_eq!(n.commit, c.node(l).commit);
    assert_eq!(n.committed(), c.node(l).committed());
}

#[test]
fn file_durable_survives_restart() {
    let dir = std::env::temp_dir().join(format!("rce-raft-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("n0.json");
    let _ = std::fs::remove_file(&path);
    let mut n = Node::new(0, 1, Box::new(FileDurable(path.clone()))).unwrap();
    n.start_election();
    assert_eq!(n.role, Role::Leader);
    n.propose("x").unwrap();
    drop(n);
    let n = Node::new(0, 1, Box::new(FileDurable(path.clone()))).unwrap();
    assert_eq!(n.term(), 1);
    assert_eq!(n.p.voted, Some(0));
    assert_eq!(n.p.log, vec![(1, "x".to_string())]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tcp_cluster_elects_and_commits() {
    use std::net::TcpListener;
    let ls: Vec<TcpListener> = (0..3).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<_> = ls.iter().map(|l| l.local_addr().unwrap()).collect();
    let mut peers: Vec<Peer> = ls
        .into_iter()
        .enumerate()
        .map(|(i, l)| Peer { node: Node::new(i, 3, Box::new(MemDurable::default())).unwrap(), t: Box::new(Tcp::new(i, l, addrs.clone())) })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let mut proposed = 0;
    loop {
        for p in peers.iter_mut() {
            p.step();
        }
        if let Some(p) = peers.iter_mut().find(|p| p.node.role == Role::Leader) {
            if proposed < 5 && p.node.propose(&format!("t{proposed}")).is_some() {
                proposed += 1;
                p.flush();
            }
        }
        if proposed == 5 && peers.iter().all(|p| p.node.commit >= 5) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "tcp cluster did not commit in time");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let a = peers[0].node.committed();
    assert!(peers.iter().all(|p| p.node.committed() == a));
}

/// The host admits through a Raft leader; the leader crashes; a new leader
/// is elected, the host resumes from its committed log, takes the role at
/// the new Raft term, and D is one chain that replays.
#[test]
fn host_fails_over_through_raft() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/host/priority.rce")).unwrap();
    let m = parse(&src).unwrap();
    let key = |s: &str, q: u64, c: &str| Sourced::new(s, q, 0, "Key", vec![Arg { label: Some("c".into()), e: Expr::Str(c.into(), 0) }]);
    let c = Rc::new(RefCell::new(Cluster::new(3)));
    let l = c.borrow_mut().elect(100).unwrap();
    let t1 = c.borrow().node(l).term();
    let mut h = Host::resume(&m, "", Box::new(RaftStore::new(c.clone(), l))).unwrap();
    h.take_role(t1, &format!("n{l}")).unwrap();
    h.offer(key("a", 1, "x")).unwrap();
    h.offer(key("b", 1, "y")).unwrap();
    let w = h.close_window(0).unwrap();
    assert_eq!(w.acks.len(), 2);
    drop(h);
    c.borrow_mut().crash(l);
    let l2 = {
        let mut cl = c.borrow_mut();
        let mut got = None;
        for _ in 0..200 {
            cl.step();
            if let Some(x) = cl.leader().filter(|&x| x != l) {
                got = Some(x);
                break;
            }
        }
        got.expect("a new leader")
    };
    let (t2, text) = {
        let cl = c.borrow();
        let n = cl.node(l2);
        // The new leader holds every committed entry; it may also hold
        // entries that the old leader replicated but did not see committed.
        let lines: Vec<String> = n.p.log.iter().map(|(_, s)| s.clone()).collect();
        (n.term(), lines.iter().map(|s| format!("{s}\n")).collect::<String>())
    };
    assert!(t2 > t1);
    let mut h2 = Host::resume(&m, &text, Box::new(RaftStore::new(c.clone(), l2))).unwrap();
    assert_eq!(h2.admitted_upto("a"), 1);
    h2.take_role(t2, &format!("n{l2}")).unwrap();
    h2.offer(key("a", 1, "x")).unwrap(); // retransmission of an acked input
    h2.offer(key("a", 2, "z")).unwrap();
    let w = h2.close_window(1).unwrap();
    assert_eq!(w.dups, vec![("a".to_string(), 1)]);
    assert_eq!(w.acks, vec![("a".to_string(), 2)]);
    let hash = h2.state_hash().to_string();
    drop(h2);
    c.borrow_mut().restart(l);
    c.borrow_mut().run(40);
    let cl = c.borrow();
    let d: String = cl.node(l).committed().iter().map(|s| format!("{s}\n")).collect();
    let (_, rh, _) = replay(&m, &d).unwrap();
    assert_eq!(rh, hash, "the restarted old leader holds the same D");
}

/// Joint consensus: grow 3 -> 5 nodes, then remove the leader. Commits keep
/// going, every node agrees, and the removed leader steps down.
#[test]
fn joint_consensus_adds_and_removes_members() {
    let mut c = Cluster::new(3);
    let l = c.elect(100).unwrap();
    propose_all(&mut c, 0, 4);
    c.run(20);
    let a = c.add_node();
    let b = c.add_node();
    assert!(c.node_mut(l).change_membership(vec![0, 1, 2, a, b]).is_some());
    assert!(c.node_mut(l).change_membership(vec![0]).is_none(), "one change at a time");
    c.peers[l].as_mut().unwrap().flush();
    c.run(60);
    assert_eq!(c.node(l).cfg(), Cfg { old: vec![0, 1, 2, a, b], new: None });
    assert_eq!(c.node(a).committed().len(), 4, "new member caught up");
    let keep: Vec<usize> = (0..5).filter(|&i| i != l).collect();
    assert!(c.node_mut(l).change_membership(keep.clone()).is_some());
    c.peers[l].as_mut().unwrap().flush();
    c.run(150);
    assert_ne!(c.node(l).role, Role::Leader, "a leader outside C_new steps down");
    let l2 = c.leader().expect("a leader among the remaining members");
    assert!(keep.contains(&l2));
    propose_all(&mut c, 4, 7);
    c.run(30);
    for &i in &keep {
        assert_eq!(c.node(i).committed().len(), 7, "node {i}");
    }
    agree(&c);
}

/// While C_old,new is the latest configuration, commitment needs both
/// majorities: with the new members unreachable nothing commits.
#[test]
fn joint_configuration_needs_both_majorities() {
    let mut c = Cluster::new(3);
    let l = c.elect(100).unwrap();
    let a = c.add_node();
    let b = c.add_node();
    let d = c.add_node();
    c.sw.borrow_mut().partition(&[0, 1, 2], &[a, b, d]);
    let i = c.node_mut(l).change_membership(vec![a, b, d]).unwrap();
    c.peers[l].as_mut().unwrap().flush();
    c.run(40);
    assert!(c.node(l).commit < i, "joint entry committed without a majority of C_new");
    c.sw.borrow_mut().heal();
    c.run(150);
    let l2 = c.leader().unwrap();
    assert!([a, b, d].contains(&l2));
}

#[test]
fn chunked_snapshot_installs_under_reordering() {
    let mut c = Cluster::new(3);
    let l = c.elect(100).unwrap();
    let f = (l + 1) % 3;
    c.crash(f);
    propose_all(&mut c, 0, 12);
    c.run(10);
    let big: String = (0..500).map(|i| format!("{i},")).collect();
    assert!(c.node_mut(l).compact(10, big.clone()));
    c.node_mut(l).snap_chunk = 37;
    c.sw.borrow_mut().reorder = true;
    c.restart(f);
    c.run(400);
    let n = c.node(f);
    assert_eq!(n.p.snap_idx, 10);
    assert_eq!(n.p.snap, big);
    assert_eq!(n.committed(), c.node(l).committed());
}

#[test]
fn tls_cluster_elects_commits_and_ignores_unpinned_senders() {
    use rce_spec_check::tls::Identity;
    use std::net::TcpListener;
    let ids: Vec<Identity> = (0..3).map(|i| Identity::from_seed([i as u8 + 1; 32]).unwrap()).collect();
    let ls: Vec<TcpListener> = (0..3).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<_> = ls.iter().map(|l| l.local_addr().unwrap()).collect();
    let pins: Vec<_> = addrs.iter().zip(&ids).map(|(a, i)| (*a, i.public)).collect();
    let mut peers: Vec<Peer> = ls
        .into_iter()
        .enumerate()
        .map(|(i, l)| Peer { node: Node::new(i, 3, Box::new(MemDurable::default())).unwrap(), t: Box::new(TlsTransport::new(i, l, pins.clone(), &ids[i]).unwrap()) })
        .collect();
    // An intruder with an unpinned key claims to be node 0 and asks for votes.
    let rogue = Identity::from_seed([99; 32]).unwrap();
    let mut rogue_pins = pins.clone();
    rogue_pins[0].1 = rogue.public;
    let mut rt = TlsTransport::new(0, TcpListener::bind("127.0.0.1:0").unwrap(), rogue_pins, &rogue).unwrap();
    for to in 1..3 {
        rt.send(to, &Msg::Vote { term: 1000, cand: 0, last_idx: 99, last_term: 99 });
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut proposed = 0;
    loop {
        for p in peers.iter_mut() {
            p.step();
        }
        if let Some(p) = peers.iter_mut().find(|p| p.node.role == Role::Leader) {
            if proposed < 5 && p.node.propose(&format!("t{proposed}")).is_some() {
                proposed += 1;
                p.flush();
            }
        }
        if proposed == 5 && peers.iter().all(|p| p.node.commit >= 5) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "tls cluster did not commit in time");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(peers.iter().all(|p| p.node.term() < 1000), "a frame from an unpinned key was accepted");
    let a = peers[0].node.committed();
    assert!(peers.iter().all(|p| p.node.committed() == a));
}
