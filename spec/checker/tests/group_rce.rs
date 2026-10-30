//! A passive MLS member across epochs in pure RCE (crates/rce/lib/group.rce
//! on the crypto, HPKE, MLS, TreeKEM, framing, protection and Welcome
//! libraries) against the official RFC 9420 passive-client-handling-commit.json
//! for cipher suite 1 (all 13 entries): join from the Welcome, then for every
//! epoch take the proposals by reference, apply the commit's GroupContext
//! extensions, Update, Remove, Add and PSK proposals, process the
//! committer's update path, update the transcript hashes, run the key
//! schedule (external and resumption PSKs) and check the confirmation tag
//! and the epoch authenticator. Every proposal and commit must carry its
//! sender's signature and this epoch's membership tag; a forged commit is
//! not applied. A few steps per epoch, each one host input;
//! the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;
#[path = "support/flat.rs"]
mod flat;

fn combined() -> String {
    let mut s = String::from("# rce-version: 0.5\nmodule group_steps\n");
    s.push_str(&flat::flat(&["crypto", "hpke", "mls", "treekem", "framing", "protect", "welcome", "grp"], false));
    s + APP
}

const APP: &str = r##"
event WLoad(t: text)
event WOpen(kp: text, init: text)
event WInfo(psks: text)
event TFromGi(z: i64)
event TStep(k: i64)
event JSetup(enc: text)
event Prop(msg: text)
event Commit(msg: text)
event Path(z: i64)
event Recv(z: i64)
event Finish(z: i64)

state S
    out: List[text] = []
    seq: i64 = 0
    wx: text = ""
    kp: text = ""
    gsx: text = ""
    egi: text = ""
    gi: List[i64] = []
    mem: text = ""
    ext: text = ""
    tb: List[i64] = []
    pos: i64 = 0
    tend: i64 = 0
    nodes: List[text] = []
    gid: text = ""
    ep: i64 = 0
    cth: text = ""
    gce: text = ""
    me: i64 = 0
    held: text = ""
    interim: text = ""
    init: text = ""
    res: text = ""
    refs: text = ""
    sender: i64 = 0
    ptree: List[text] = []
    old: List[text] = []
    ngce: text = ""
    added: text = ""
    psks: text = ""
    fc: text = ""
    sig: text = ""
    tag: text = ""
    up: List[text] = []
    fp: List[i64] = []
    pgc: text = ""
    commit: text = ""
    mkey: text = ""
    cur_th: text = ""

step S on WLoad(t: t)
    seq += 1
    wx = wx + t
    out = ["O:" + seq.tostr() + "#ok"]

step S on WOpen(kp: k, init: ip)
    seq += 1
    kp = k
    let parts = welcome_secrets(copy(wx), copy(kp), ip).split(".")
    wx = ""
    gsx = copy(parts[0])
    egi = copy(parts[1])
    out = ["O:" + seq.tostr() + "#" + (gsx != "").tostr()]

step S on WInfo(psks: p)
    seq += 1
    ext = p
    let parts = welcome_group_info(copy(gsx), copy(egi), copy(ext)).split(".")
    gi = unhex(copy(parts[0]))
    mem = copy(parts[1])
    out = ["O:" + seq.tostr() + "#" + (gi.len() > 1).tostr()]

step S on TFromGi(z: z)
    seq += 1
    tb = unhex(gi_ratchet_tree(copy(gi)))
    let r = rd_v(copy(tb), 0)
    pos = r[1]
    tend = r[0] + r[1]
    nodes = []
    out = ["O:" + seq.tostr() + "#" + tb.len().tostr()]

step S on TStep(k: k)
    seq += 1
    var j: i64 = 0
    while j < k and pos < tend bound 1024
        let e = rt_node_end(copy(tb), pos as usize)
        nodes.push(rt_node(copy(tb), pos as usize, e))
        pos = e as i64
        j += 1
    if pos < tend
        out = ["O:" + seq.tostr() + "#more"]
    else
        nodes = rt_pad(copy(nodes))
        tb = []
        out = ["O:" + seq.tostr() + "#done"]

step S on JSetup(enc: enc)
    seq += 1
    let we = welcome_epoch(copy(gi), copy(mem)).split(".")
    let es = unhex(copy(we[0]))
    var p = skip_v(copy(gi), 4)
    gid = tohex(get_v(copy(gi), 4))
    ep = 0
    for j in 0..8
        ep = (ep << 8) | gi[p + j]
    p += 8
    p = skip_v(copy(gi), p)
    cth = tohex(get_v(copy(gi), p))
    p = skip_v(copy(gi), p)
    gce = tohex(get_v(copy(gi), p))
    let o = gi_offsets(copy(gi))
    let gtag = get_v(copy(gi), o[1] as usize)
    let signer = gi_signer(copy(gi))
    let mine = kp_leaf_node(mls_body(copy(kp)))
    var i: usize = 0
    while i < nodes.len() bound 4096
        if nodes[i] == mine
            me = (i / 2) as i64
        i += 2
    held = kv_set("", (2 * me).tostr(), enc)
    let gs = unhex(copy(gsx))
    let q = skip_v(copy(gs), 0)
    if gs[q] == 1
        var ps = tohex(get_v(copy(gs), q + 1))
        var x = lca(2 * me, 2 * signer)
        let root = tree_root(copy(nodes))
        var go = true
        while go bound 64
            held = kv_set(held, x.tostr(), node_sk(copy(ps)))
            if x == root
                go = false
            else
                x = rt_parent(x)
                ps = tohex(derive_secret(unhex(ps), "path"))
    interim = tohex(sha256(cat(unhex(copy(cth)), ser_v(gtag))))
    init = tohex(derive_secret(copy(es), "init"))
    res = kv_set("", ep.tostr(), tohex(derive_secret(copy(es), "resumption")))
    mkey = tohex(derive_secret(copy(es), "membership"))
    cur_th = gi_tree_hash(copy(gi))
    out = ["O:" + seq.tostr() + "#" + tohex(derive_secret(es, "authentication")) + "/" + copy(we[1])]

step S on Prop(msg: msg)
    seq += 1
    if authentic(copy(msg), copy(nodes), group_context_x(copy(gid), ep, copy(cur_th), copy(cth), copy(gce)), copy(mkey))
        let r = proposal_ref(msg)
        refs = kv_set(copy(refs), copy(r[0]), copy(r[1]))
        out = ["O:" + seq.tostr() + "#ok"]
    else
        out = ["O:" + seq.tostr() + "#forged"]

step S on Commit(msg: msg)
    seq += 1
    if not authentic(copy(msg), copy(nodes), group_context_x(copy(gid), ep, copy(cur_th), copy(cth), copy(gce)), copy(mkey))
        out = ["O:" + seq.tostr() + "#forged"]
    else
        let f = mls_decode(msg)
        let sb = unhex(copy(f[3]))
        sender = (sb[1] << 24) | (sb[2] << 16) | (sb[3] << 8) | sb[4]
        let content = unhex(copy(f[6]))
        let props = commit_proposals(copy(content), sender, copy(refs))
        refs = ""
        var t = apply_proposals(copy(nodes), props, copy(gce))
        psks = copy(t[t.len() - 1])
        t.pop()
        added = copy(t[t.len() - 1])
        t.pop()
        ngce = copy(t[t.len() - 1])
        t.pop()
        ptree = t
        fc = tohex(cat(cat(cat(ser_v(unhex(copy(f[1]))), unhex(copy(f[2]))), cat(unhex(copy(f[3])), ser_v(unhex(copy(f[4]))))), cat(unhex(copy(f[5])), unhex(copy(f[6])))))
        sig = copy(f[7])
        tag = copy(f[8])
        let v = rd_v(copy(content), 0)
        let end = (v[0] + v[1]) as usize
        let clen = content.len()
        if content[end] == 1
            up = up_parse(tohex(slice(copy(content), end + 1, clen)))
        else
            up = []
        out = ["O:" + seq.tostr() + "#ok"]

step S on Path(z: z)
    seq += 1
    if up.len() > 0
        old = copy(ptree)
        fp = filtered_path_ex(copy(ptree), sender, int_list(copy(added)))
        var t = up_merge(copy(ptree), sender, copy(up), copy(fp))
        t.pop()
        ptree = t
        let th = copy(tree_hashes(copy(ptree))[tree_root(copy(ptree)) as usize])
        pgc = group_context_x(copy(gid), ep + 1, th, copy(cth), copy(ngce))
    out = ["O:" + seq.tostr() + "#ok"]

step S on Recv(z: z)
    seq += 1
    if up.len() > 0
        let r = path_receive(copy(old), copy(up), copy(fp), int_list(copy(added)), copy(pgc), me, copy(held))
        if r.len() == 2
            held = copy(r[0])
            commit = copy(r[1])
            out = ["O:" + seq.tostr() + "#ok"]
        else
            out = ["O:" + seq.tostr() + "#undecryptable"]
    else
        commit = tohex(zero32())
        out = ["O:" + seq.tostr() + "#ok"]

step S on Finish(z: z)
    seq += 1
    let ncth = sha256(cat(cat(unhex(copy(interim)), [0, 1]), cat(unhex(copy(fc)), ser_v(unhex(copy(sig))))))
    interim = tohex(sha256(cat(copy(ncth), ser_v(unhex(copy(tag))))))
    cth = tohex(ncth)
    ep += 1
    nodes = copy(ptree)
    gce = copy(ngce)
    let th = copy(tree_hashes(copy(nodes))[tree_root(copy(nodes)) as usize])
    let gc = unhex(group_context_x(copy(gid), ep, copy(th), copy(cth), copy(gce)))
    let joiner = expand_label(extract(unhex(copy(init)), unhex(copy(commit))), "joiner", copy(gc), 32)
    let m2 = extract(joiner, psk_secret_ids(copy(psks), copy(ext), copy(res)))
    let es = expand_label(m2, "epoch", gc, 32)
    let ok = tohex(hmac(derive_secret(copy(es), "confirm"), unhex(copy(cth)))) == tag
    init = tohex(derive_secret(copy(es), "init"))
    res = kv_set(copy(res), ep.tostr(), tohex(derive_secret(copy(es), "resumption")))
    mkey = tohex(derive_secret(copy(es), "membership"))
    cur_th = th
    out = ["O:" + seq.tostr() + "#" + tohex(derive_secret(es, "authentication")) + "/" + ok.tostr()]
"##;

#[test]
fn official_passive_client_handling_commit() {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/passive-client-handling-commit-suite1.json")).unwrap()).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut n = 0u64;
    let mut step = |ev: String| -> String {
        n += 1;
        h.admit(&src, &ev).unwrap();
        let s = format!("{:?}", h.state());
        let o = s.split("Text(\"O:").nth(1).unwrap().split('"').next().unwrap().to_string();
        let (k, r) = o.split_once('#').unwrap();
        assert_eq!(k.parse::<u64>().unwrap(), n, "step admitted within its fuel: {ev:.60}");
        r.to_string()
    };
    let mut epochs = 0;
    for (i, t) in v.as_array().unwrap().iter().enumerate() {
        assert!(t["ratchet_tree"].is_null());
        for c in t["welcome"].as_str().unwrap().as_bytes().chunks(2048) {
            step(format!(r#"WLoad(t: "{}")"#, std::str::from_utf8(c).unwrap()));
        }
        assert_eq!(step(format!(r#"WOpen(kp: "{}", init: "{}")"#, t["key_package"].as_str().unwrap(), t["init_priv"].as_str().unwrap())), "true");
        let psks: String = t["external_psks"].as_array().unwrap().iter().map(|e| format!("{}:{},", e["psk_id"].as_str().unwrap(), e["psk"].as_str().unwrap())).collect();
        assert_eq!(step(format!(r#"WInfo(psks: "{psks}")"#)), "true");
        step("TFromGi(z: 0)".into());
        while step("TStep(k: 8)".into()) == "more" {}
        assert_eq!(step(format!(r#"JSetup(enc: "{}")"#, t["encryption_priv"].as_str().unwrap())), format!("{}/true", t["initial_epoch_authenticator"].as_str().unwrap()), "entry {i}: join");
        for (e, ep) in t["epochs"].as_array().unwrap().iter().enumerate() {
            for p in ep["proposals"].as_array().unwrap() {
                assert_eq!(step(format!(r#"Prop(msg: "{}")"#, p.as_str().unwrap())), "ok", "entry {i} epoch {e}: proposal authentic");
            }
            if i == 0 && e == 0 {
                // a forged commit (membership tag changed) is not applied and changes nothing
                let c = ep["commit"].as_str().unwrap();
                let flipped = format!("{}{:02x}", &c[..c.len() - 2], u8::from_str_radix(&c[c.len() - 2..], 16).unwrap() ^ 1);
                assert_eq!(step(format!(r#"Commit(msg: "{flipped}")"#)), "forged");
            }
            assert_eq!(step(format!(r#"Commit(msg: "{}")"#, ep["commit"].as_str().unwrap())), "ok", "entry {i} epoch {e}: commit authentic");
            step("Path(z: 0)".into());
            assert_eq!(step("Recv(z: 0)".into()), "ok", "entry {i} epoch {e}: our path secret decrypts");
            assert_eq!(step("Finish(z: 0)".into()), format!("{}/true", ep["epoch_authenticator"].as_str().unwrap()), "entry {i} epoch {e}: epoch authenticator and confirmation tag");
            epochs += 1;
        }
    }
    assert_eq!(epochs, 26);
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
