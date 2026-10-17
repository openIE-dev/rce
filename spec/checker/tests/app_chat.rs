//! W4 chat v0: three replicas on three hosts exchange ops over P4
//! channels; after polling, every replica shows the same thread in the same
//! (Lamport clock, author) order, the same reactions, presence and call
//! rooms; each replica's D replays with no network.
use rce_spec_check::adapters::chan::chan_adapters;
use rce_spec_check::ast::{Arg, Expr};
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use std::path::Path;

fn port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// The shared part of a thread view (without this replica's unread marker).
fn show(h: &mut Host, conv: &str, topic: &str) -> Vec<String> {
    show_raw(h, conv, topic).into_iter().filter(|l| l != "-- new --\n").collect()
}

fn show_raw(h: &mut Host, conv: &str, topic: &str) -> Vec<String> {
    let a = |l: &str, v: &str| Arg { label: Some(l.into()), e: Expr::Str(v.into(), 0) };
    h.query("Show", vec![a("conv", conv), a("topic", topic)]).unwrap().into_iter().filter_map(|o| match o {
        V::Rec(_, fs) => fs.into_iter().find_map(|(f, v)| match v { V::Text(t) if f == "data" => Some(t.to_string()), _ => None }),
        _ => None,
    }).collect()
}

fn region(h: &Host, field: &str) -> String {
    format!("{:?}", h.state()[0].1.iter().find(|(n, _, _)| n == field).unwrap().1)
}

#[test]
fn three_replicas_converge_and_replay() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/chat.rce")).unwrap();
    let m = parse(&src).unwrap();
    let addrs: Vec<String> = (0..3).map(|_| format!("127.0.0.1:{}", port())).collect();
    let mut stores: Vec<MemStore> = (0..3).map(|_| MemStore::default()).collect();
    let mut hosts = vec![];
    for (i, st) in stores.iter_mut().enumerate() {
        let peers: Vec<String> = addrs.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, a)| a.clone()).collect();
        let ads: Vec<Box<dyn Adapter>> = chan_adapters(&addrs[i], &peers).unwrap().into_iter().map(|x| Box::new(x) as Box<dyn Adapter>).collect();
        let mut h = Host::start_with(&m, Box::new(st), ads).unwrap();
        let ps = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
        h.admit(&src, &format!(r#"Setup(me: "{}", peers: [{ps}])"#, ["ann", "bob", "cy"][i])).unwrap();
        hosts.push(h);
    }
    let ops = [
        (0, r#"Do(op: "ws", conv: "acme", topic: "", body: "Acme"), Do(op: "ch", conv: "acme/general", topic: "", body: "general"), Do(op: "post", conv: "acme/general", topic: "launch", body: "ship it?")"#),
        (1, r#"Do(op: "post", conv: "acme/general", topic: "launch", body: "yes"), Do(op: "presence", conv: "", topic: "", body: "online")"#),
        (2, r#"Do(op: "conv", conv: "dm:ann,cy", topic: "", body: "ann,cy"), Do(op: "post", conv: "dm:ann,cy", topic: "", body: "e2ee:3f9a..."), Do(op: "file", conv: "acme/general", topic: "launch", body: "sha256:ab12 deck.pdf")"#),
        (0, r#"Do(op: "join", conv: "standup", topic: "", body: ""), Do(op: "track", conv: "standup", topic: "video", body: "on")"#),
    ];
    for (i, o) in ops {
        hosts[i].admit(&src, o).unwrap();
    }
    for _ in 0..50 {
        for h in hosts.iter_mut() {
            h.admit(&src, "Poll").unwrap();
        }
        let views: Vec<Vec<String>> = hosts.iter_mut().map(|h| show(h, "acme/general", "launch")).collect();
        if views.iter().all(|v| v.len() == 3) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let views: Vec<Vec<String>> = hosts.iter_mut().map(|h| show(h, "acme/general", "launch")).collect();
    assert_eq!(views[0].len(), 3, "{views:?}");
    assert!(views.windows(2).all(|w| w[0] == w[1]), "replicas disagree: {views:?}");
    for f in ["messages", "workspaces", "channels", "convs", "presence", "calls", "tracks"] {
        let r: Vec<String> = hosts.iter().map(|h| region(h, f)).collect();
        assert!(r.windows(2).all(|w| w[0] == w[1]), "{f} differs: {r:?}");
    }
    assert!(region(&hosts[1], "messages").contains("e2ee:3f9a"), "ciphertext passes through");

    // edits, deletes, unread markers and the thread view
    let key_of = |h: &Host, author: &str, conv: &str| -> String {
        match &h.state()[0].1.iter().find(|(n, _, _)| n == "messages").unwrap().1 {
            V::Map(kv) => kv.iter().find_map(|(k, _)| match k {
                V::Text(t) if t.ends_with(&format!("\u{1f}{author}")) && t.starts_with(conv) => Some(t.to_string()),
                _ => None,
            }).unwrap(),
            _ => panic!(),
        }
    };
    let esc = |k: &str| k.replace('\u{1f}', "\\u{1f}");
    let bob_msg = key_of(&hosts[1], "bob", "acme/general");
    let ann_msg = key_of(&hosts[0], "ann", "acme/general\u{1f}launch");
    hosts[2].admit(&src, r#"Do(op: "read", conv: "acme/general", topic: "launch", body: "")"#).unwrap();
    // cy may not edit bob's message; bob edits it, ann deletes hers
    hosts[2].admit(&src, &format!(r#"Do(op: "edit", conv: "{}", topic: "", body: "hijacked")"#, esc(&bob_msg))).unwrap();
    hosts[1].admit(&src, &format!(r#"Do(op: "edit", conv: "{}", topic: "", body: "yes, shipping today")"#, esc(&bob_msg))).unwrap();
    hosts[0].admit(&src, &format!(r#"Do(op: "delete", conv: "{}", topic: "", body: "")"#, esc(&ann_msg))).unwrap();
    hosts[0].admit(&src, r#"Do(op: "post", conv: "acme/general", topic: "launch", body: "one more thing")"#).unwrap();
    for _ in 0..50 {
        for h in hosts.iter_mut() {
            h.admit(&src, "Poll").unwrap();
        }
        let views: Vec<Vec<String>> = hosts.iter_mut().map(|h| show(h, "acme/general", "launch").into_iter().filter(|l| !l.starts_with("--")).collect()).collect();
        if views.iter().all(|v| v.len() == 4 && v.iter().any(|l| l.contains("(edited)")) && v.iter().any(|l| l.contains("[deleted]"))) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let views: Vec<Vec<String>> = hosts.iter_mut().map(|h| show(h, "acme/general", "launch").into_iter().filter(|l| !l.starts_with("--")).collect()).collect();
    assert!(views.windows(2).all(|w| w[0] == w[1]), "replicas disagree after edits: {views:?}");
    assert!(views[0].contains(&"bob: yes, shipping today (edited)\n".to_string()), "{views:?}");
    assert!(views[0].contains(&"ann: [deleted]\n".to_string()), "{views:?}");
    assert!(!views[0].iter().any(|l| l.contains("hijacked")));
    for f in ["messages", "edited", "deleted"] {
        let r: Vec<String> = hosts.iter().map(|h| region(h, f)).collect();
        assert!(r.windows(2).all(|w| w[0] == w[1]), "{f} differs: {r:?}");
    }
    // cy read the thread before "one more thing": the marker precedes it, one unread
    let cy = show_raw(&mut hosts[2], "acme/general", "launch");
    let at = cy.iter().position(|l| l == "-- new --\n").expect("unread marker");
    assert_eq!(cy[at + 1], "ann: one more thing\n", "{cy:?}");
    let unread: Vec<String> = hosts[2].query("Unread", vec![]).unwrap().into_iter().filter_map(|o| match o {
        V::Rec(_, fs) => fs.into_iter().find_map(|(f, v)| match v { V::Text(t) if f == "data" => Some(t.to_string()), _ => None }),
        _ => None,
    }).collect();
    assert!(unread.contains(&"acme/general #launch: 1 unread\n".to_string()), "{unread:?}");
    // the front end's View query: keys, author, flags; it reads only
    let before = hosts[2].state_hash().to_string();
    let view: Vec<String> = hosts[2].query("View", vec![Arg { label: Some("conv".into()), e: Expr::Str("acme/general".into(), 0) }, Arg { label: Some("topic".into()), e: Expr::Str("launch".into(), 0) }]).unwrap().into_iter().filter_map(|o| match o {
        V::Rec(_, fs) => fs.into_iter().find_map(|(f, v)| match v { V::Text(t) if f == "data" => Some(t.to_string()), _ => None }),
        _ => None,
    }).collect();
    assert_eq!(hosts[2].state_hash().to_string(), before);
    assert_eq!(view.last().map(String::as_str), Some("view-end\n"));
    let rows: Vec<Vec<&str>> = view.iter().filter(|l| l.starts_with("view\t")).map(|l| l.trim_end().split('\t').collect()).collect();
    assert_eq!(rows.len(), 4, "{view:?}");
    assert!(rows.iter().any(|r| r[2] == "bob" && r[3].contains("edited") && r[4] == "yes, shipping today"), "{view:?}");
    assert!(rows.iter().any(|r| r[2] == "ann" && r[3].contains("deleted")), "{view:?}");
    assert!(rows.iter().any(|r| r[3].contains("new") && r[4] == "one more thing"), "{view:?}");
    assert!(hosts.iter().all(|h| region(h, "unsent") == "Int(0, \"u64\")" || region(h, "unsent").starts_with("Int(0")));
    let lives: Vec<String> = hosts.iter().map(|h| h.state_hash().to_string()).collect();
    drop(hosts);
    for (st, live) in stores.iter().zip(lives) {
        let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
        assert_eq!(replay(&m, &text).unwrap().1, live);
    }
}
