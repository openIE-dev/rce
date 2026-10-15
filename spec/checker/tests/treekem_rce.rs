//! TreeKEM ratchet trees in pure RCE (crates/rce/lib/treekem.rce on top of
//! lib/crypto.rce and lib/mls.rce) against the official RFC 9420
//! tree-validation.json vectors for cipher suite 1: every tree decodes, every
//! tree hash and every resolution matches, and every parent node is
//! parent-hash valid (the 11 trees up to 16 leaves; hashes and
//! resolutions run on all 14, split across inputs); a tree with one parent
//! hash changed is not. One computation per host input (HC-7a); the log replays.
use rce_spec_check::adapters::RandomAdapter;
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn combined() -> String {
    let mut s = String::from("module treekem_steps\n");
    for f in ["lib/crypto.rce", "lib/hpke.rce", "lib/mls.rce", "lib/treekem.rce"] {
        for l in std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(f)).unwrap().lines() {
            if l.starts_with("module ") || l.starts_with("check ") {
                continue;
            }
            s.push_str(l);
            s.push('\n');
        }
    }
    s + APP
}

const APP: &str = r#"
event Load(t: text)
event Start(z: i64)
event DecodeStep(k: i64)
event HashBatch(lo: i64, n: i64)
event Dump(z: i64)
event Res1(x: i64)
event Valid(t: text, p: i64)
event UpLoad(t: text)
event Merge(s: i64, gid: text, ep: i64, cth: text)
event Recv(i: i64, leaf_sk: text, pss: text)
event GenStart(s: i64, gid: text, sig: text)
event RandomDone(req: u64, status: text, data: text?, err: text)
event GenKeys(z: i64)
event GenLeaf(ep: i64, cth: text)
event GenEnc(j: i64)
event GenDone(z: i64)
event SigOk(s: i64, gid: text)

struct Random
    req: u64
    n: u64

state S
    out: List[text] = []
    nodes: List[text] = []
    hs: List[text] = []
    ord: List[i64] = []
    tb: List[i64] = []
    pos: i64 = 0
    tend: i64 = 0
    upx: text = ""
    up: List[text] = []
    fp: List[i64] = []
    gc: text = ""
    gsender: i64 = 0
    ggid: text = ""
    gsig: text = ""
    rnd: text = ""
    gs: List[text] = []
    geks: List[text] = []
    gph: text = ""
    gtree: List[text] = []
    gleaf: text = ""
    gnodes: List[text] = []
    greq: u64 = 0

step S on Load(t: t)
    for x in unhex(t)
        tb.push(x)
    out.push("O:loaded")

step S on Start(z: z)
    let r = rd_v(copy(tb), 0)
    pos = r[1]
    tend = r[0] + r[1]
    nodes = []
    out.push("O:started")

step S on DecodeStep(k: k)

    var j: i64 = 0
    while j < k and pos < tend bound 1024
        let e = rt_node_end(copy(tb), pos as usize)
        nodes.push(rt_node(copy(tb), pos as usize, e))
        pos = e as i64
        j += 1
    if pos < tend
        out.push("O:more")
    else
        nodes = rt_pad(copy(nodes))
        tb = []
        hs = []
        for _ in 0..nodes.len()
            hs.push("")
        ord = hash_order(nodes.len())
        out.push("O:" + nodes.len().tostr())

step S on HashBatch(lo: lo, n: n)
    let e: usize = if (lo + n) as usize > ord.len() then ord.len() else (lo + n) as usize
    var i = lo as usize
    while i < e bound 1024
        let x = ord[i]
        if x % 2 == 0
            hs[x as usize] = node_hash(copy(nodes[x as usize]), x, "", "")
        else
            let l = tm_left(x) as usize
            let r = (x ^ (3 << (tm_level(x) - 1))) as usize
            hs[x as usize] = node_hash(copy(nodes[x as usize]), x, copy(hs[l]), copy(hs[r]))
        i += 1
    out.push("O:ok")

step S on Dump(z: z)
    var s = "O:"
    for h in copy(hs)
        s = s + h + ","
    out.push(s)

step S on Res1(x: x)
    var s = "O:"
    for r in resolution(copy(nodes), x)
        s = s + r.tostr() + ","
    out.push(s)

step S on UpLoad(t: t)
    upx = upx + t
    out.push("O:ok")

step S on Merge(s: s, gid: gid, ep: ep, cth: cth)
    up = up_parse(copy(upx))
    upx = ""
    fp = filtered_path(copy(nodes), s)
    var nt = up_merge(copy(nodes), s, copy(up), copy(fp))
    let ph = copy(nt[nt.len() - 1])
    nt.pop()
    let th = copy(tree_hashes(copy(nt))[(nt.len() / 2) as usize])
    gc = group_context(gid, ep, copy(th), cth)
    let leaf_ph = copy(nfields(copy(nt[(2 * s) as usize]))[1])
    out.push("O:" + th + "/" + (ph == leaf_ph).tostr())

step S on Recv(i: i, leaf_sk: leaf_sk, pss: pss)
    var held = tohex([(2 * i) & 255]) + ":" + leaf_sk + ","
    for kv in pss.split(",")
        if kv != ""
            let parts = kv.split(":")
            held = held + copy(parts[0]) + ":" + node_sk(copy(parts[1])) + ","
    out.push("O:" + up_receive(copy(nodes), copy(up), copy(fp), copy(gc), i, held))

step S on GenStart(s: s, gid: gid, sig: sig)
    fp = filtered_path(copy(nodes), s)
    gsender = s
    ggid = gid
    gsig = sig
    var total: u64 = 1
    for j in 0..fp.len() / 2
        total += resolution(copy(nodes), fp[2 * j + 1]).len() as u64
    greq += 1
    emit Random(req: greq, n: 32 * total)

step S on RandomDone(req: r, status: st, data: d, err: e)
    rnd = d.or("")
    out.push("O:" + st + "/" + (rnd.len() / 2).tostr())

step S on GenKeys(z: z)
    let m = fp.len() / 2
    gs = up_secrets(tohex(slice(unhex(copy(rnd)), 0, 32)), m as i64)
    geks = []
    var place: List[text] = [copy(nodes[(2 * gsender) as usize])]
    for j in 0..m
        let ek = pubkey(node_sk(copy(gs[j + 1])))
        place.push(copy(ek) + "|")
        geks.push(ek)
    var t = up_merge(copy(nodes), gsender, place, copy(fp))
    gph = copy(t[t.len() - 1])
    t.pop()
    gtree = t
    out.push("O:keys")

step S on GenLeaf(ep: ep, cth: cth)
    gleaf = up_leaf(copy(nodes[(2 * gsender) as usize]), pubkey(copy(gs[0])), copy(gph), copy(gsig), copy(ggid), gsender)
    gtree[(2 * gsender) as usize] = copy(gleaf)
    let th = copy(tree_hashes(copy(gtree))[(gtree.len() / 2) as usize])
    gc = group_context(copy(ggid), ep, copy(th), cth)
    gnodes = []
    out.push("O:" + th)

step S on GenEnc(j: j)
    var off: usize = 1
    for q in 0..j
        off += resolution(copy(nodes), fp[(2 * q + 1) as usize]).len()
    let n = resolution(copy(nodes), fp[(2 * j + 1) as usize]).len()
    let r = tohex(slice(unhex(copy(rnd)), 32 * off, 32 * (off + n)))
    gnodes.push(up_node(copy(nodes), copy(geks[j as usize]), fp[(2 * j + 1) as usize], copy(gc), copy(gs[(j + 1) as usize]), r))
    out.push("O:enc")

step S on GenDone(z: z)
    upx = up_serialize(copy(gleaf), copy(gnodes))
    var s = "O:"
    for j in 1..gs.len() - 1
        s = s + gs[j] + ","
    out.push(s + "/" + gs[gs.len() - 1])

step S on SigOk(s: s, gid: gid)
    out.push("O:" + leaf_signature_ok(copy(up[0]), gid, s).tostr())

step S on Valid(t: t, p: p)
    out.push("O:" + parent_hash_valid(rt_decode(t), p).tostr())
"#;

fn run(max_leaves: usize) {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/tree-validation-suite1.json")).unwrap()).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut step = |ev: String| -> String {
        let outs = |s: String| -> Vec<String> { s.split("Text(\"O:").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect() };
        let n0 = outs(format!("{:?}", h.state())).len();
        h.admit(&src, &ev).unwrap();
        let o = outs(format!("{:?}", h.state()));
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.60}");
        o[n0].to_string()
    };
    let mut checked = 0;
    for (i, t) in v.as_array().unwrap().iter().enumerate() {
        let tree = t["tree"].as_str().unwrap();
        let want: Vec<&str> = t["tree_hashes"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        // then hashing runs bottom up in batches of 16 nodes
        // so every input stays within its fuel (HC-7a), the 64-leaf trees too
        // the hex arrives in 2 KB pieces, decoding runs 8 nodes per input
        for c in tree.as_bytes().chunks(2048) {
            step(format!(r#"Load(t: "{}")"#, std::str::from_utf8(c).unwrap()));
        }
        step("Start(z: 0)".into());
        let w: usize = loop {
            let r = step("DecodeStep(k: 8)".into());
            if r != "more" {
                break r.parse().unwrap();
            }
        };
        for lo in (0..w).step_by(16) {
            step(format!("HashBatch(lo: {lo}, n: 16)"));
        }
        let got = step("Dump(z: 0)".into());
        let got: Vec<&str> = got.split(',').collect();
        assert_eq!(&got[..want.len()], &want[..], "tree-validation {i}: tree hashes");
        for (x, r) in t["resolutions"].as_array().unwrap().iter().enumerate() {
            let w: Vec<i64> = r.as_array().unwrap().iter().map(|y| y.as_i64().unwrap()).collect();
            let got = step(format!("Res1(x: {x})"));
            let got: Vec<i64> = got.split(',').filter(|y| !y.is_empty()).map(|y| y.parse().unwrap()).collect();
            assert_eq!(got, w, "tree-validation {i}: resolution of node {x}");
        }
        if (want.len() + 1) / 2 <= max_leaves {
            for p in (1..want.len()).step_by(2) {
                assert_eq!(step(format!(r#"Valid(t: "{tree}", p: {p})"#)), "true", "tree-validation {i}: parent {p} parent-hash valid");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 14);
    // change the first parent node's parent_hash: some parent is no longer valid
    let t1 = v[2]["tree"].as_str().unwrap();
    let bad = corrupt_parent_hash(t1);
    let mut any_invalid = false;
    for p in (1..15).step_by(2) {
        if step(format!(r#"Valid(t: "{bad}", p: {p})"#)) == "false" {
            any_invalid = true;
        }
    }
    assert!(any_invalid, "a changed parent hash is detected");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// flip the last byte of the first non-empty parent_hash of a parent node
fn corrupt_parent_hash(tree: &str) -> String {
    let b = hex(tree);
    // walk the encoding as rt_decode does, in Rust, to find node 3
    fn rv(b: &[u8], p: usize) -> (usize, usize) {
        match b[p] >> 6 {
            0 => ((b[p] & 63) as usize, p + 1),
            1 => ((((b[p] & 63) as usize) << 8) | b[p + 1] as usize, p + 2),
            _ => ((((b[p] & 63) as usize) << 24) | ((b[p + 1] as usize) << 16) | ((b[p + 2] as usize) << 8) | b[p + 3] as usize, p + 4),
        }
    }
    let sk = |p: usize| {
        let (n, q) = rv(&b, p);
        q + n
    };
    let (_, mut p) = rv(&b, 0);
    loop {
        if b[p] == 0 {
            p += 1;
        } else {
            let typ = b[p + 1];
            let s = p + 2;
            let (n, q2) = rv(&b, sk(s));
            if typ == 2 && n > 0 {
                let mut c = b.clone();
                c[q2 + n - 1] ^= 1;
                return c.iter().map(|x| format!("{x:02x}")).collect();
            }
            p = if typ == 2 {
                sk(sk(sk(s)))
            } else {
                let mut q = sk(sk(s)) + 2;
                q = sk(q);
                for _ in 0..5 {
                    q = sk(q);
                }
                let src = b[q];
                q += 1;
                if src == 1 {
                    q += 16
                }
                if src == 3 {
                    q = sk(q)
                }
                sk(sk(q))
            };
        }
    }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn official_tree_validation_vectors() {
    run(16);
}



/// RFC 9420 treekem.json (suite 1, all 11 entries): for every update path,
/// the merged tree hash equals tree_hash_after, the new leaf carries the
/// parent hash computed down the filtered direct path, and every receiver
/// decrypts exactly its path secret (UpdatePathNode, HPKE) and derives the
/// commit secret. One tree, one merge or one decryption per host input.
#[test]
fn official_treekem_update_paths() {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/treekem-suite1.json")).unwrap()).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut step = |ev: String| -> String {
        let outs = |s: String| -> Vec<String> { s.split("Text(\"O:").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect() };
        let n0 = outs(format!("{:?}", h.state())).len();
        h.admit(&src, &ev).unwrap();
        let o = outs(format!("{:?}", h.state()));
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.60}");
        o[n0].clone()
    };
    let (mut paths, mut recvs) = (0, 0);
    for (e, t) in v.as_array().unwrap().iter().enumerate() {
        for c in t["ratchet_tree"].as_str().unwrap().as_bytes().chunks(2048) {
            step(format!(r#"Load(t: "{}")"#, std::str::from_utf8(c).unwrap()));
        }
        step("Start(z: 0)".into());
        while step("DecodeStep(k: 8)".into()) == "more" {}
        for u in t["update_paths"].as_array().unwrap() {
            for c in u["update_path"].as_str().unwrap().as_bytes().chunks(2048) {
                step(format!(r#"UpLoad(t: "{}")"#, std::str::from_utf8(c).unwrap()));
            }
            let s = u["sender"].as_i64().unwrap();
            let got = step(format!(r#"Merge(s: {s}, gid: "{}", ep: {}, cth: "{}")"#, t["group_id"].as_str().unwrap(), t["epoch"], t["confirmed_transcript_hash"].as_str().unwrap()));
            assert_eq!(got, format!("{}/true", u["tree_hash_after"].as_str().unwrap()), "treekem {e}: tree_hash_after and leaf parent hash");
            paths += 1;
            for lp in t["leaves_private"].as_array().unwrap() {
                let i = lp["index"].as_i64().unwrap();
                let want = &u["path_secrets"][i as usize];
                if i == s || want.is_null() {
                    continue;
                }
                let pss: String = lp["path_secrets"].as_array().unwrap().iter().map(|x| format!("{:02x}:{},", x["node"].as_i64().unwrap(), x["path_secret"].as_str().unwrap())).collect();
                let got = step(format!(r#"Recv(i: {i}, leaf_sk: "{}", pss: "{pss}")"#, lp["encryption_priv"].as_str().unwrap()));
                assert_eq!(got, format!("{}/{}", want.as_str().unwrap(), u["commit_secret"].as_str().unwrap()), "treekem {e}: leaf {i} path secret and commit secret");
                recvs += 1;
            }
        }
    }
    assert_eq!((paths, recvs), (62, 328));
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// The sender side: for every treekem.json (suite 1) entry, the first
/// private leaf generates a fresh update path with 32-byte secrets from the
/// host's logged Random effect (HC-10): leaf secret, path secrets, new node
/// keys, parent hashes, a signed LeafNode and the HPKE ciphertexts. Round
/// trip: the receiver code above merges it (tree hash and parent hash agree,
/// the leaf signature verifies) and every other private leaf decrypts its
/// path secret and the same commit secret. Replay reads the random bytes
/// from the log and reaches the same state.
#[test]
fn generated_update_paths_round_trip() {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/treekem-suite1.json")).unwrap()).unwrap();
    let mut st = MemStore::default();
    let ads: Vec<Box<dyn Adapter>> = vec![Box::new(RandomAdapter)];
    let mut h = Host::start_with(&m, Box::new(&mut st), ads).unwrap();
    let mut step = |ev: String| -> String {
        let outs = |s: String| -> Vec<String> { s.split("Text(\"O:").skip(1).map(|x| x.split('"').next().unwrap().to_string()).collect() };
        let n0 = outs(format!("{:?}", h.state())).len();
        h.admit(&src, &ev).unwrap();
        let o = outs(format!("{:?}", h.state()));
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.60}");
        o[n0].clone()
    };
    let mut recvs = 0;
    for (e, t) in v.as_array().unwrap().iter().enumerate() {
        for c in t["ratchet_tree"].as_str().unwrap().as_bytes().chunks(2048) {
            step(format!(r#"Load(t: "{}")"#, std::str::from_utf8(c).unwrap()));
        }
        step("Start(z: 0)".into());
        while step("DecodeStep(k: 8)".into()) == "more" {}
        let lps = t["leaves_private"].as_array().unwrap();
        let s = lps[0]["index"].as_i64().unwrap();
        let gid = t["group_id"].as_str().unwrap();
        let (ep, cth) = (&t["epoch"], t["confirmed_transcript_hash"].as_str().unwrap());
        let r = step(format!(r#"GenStart(s: {s}, gid: "{gid}", sig: "{}")"#, lps[0]["signature_priv"].as_str().unwrap()));
        assert!(r.starts_with("delivered/"), "entropy delivered: {r}");
        step("GenKeys(z: 0)".into());
        let th = step(format!(r#"GenLeaf(ep: {ep}, cth: "{cth}")"#));
        // one UpdatePathNode per input, as many as the filtered direct path has
        let fp_len = filtered_len(t["ratchet_tree"].as_str().unwrap(), s);
        for j in 0..fp_len {
            step(format!("GenEnc(j: {j})"));
        }
        let made = step("GenDone(z: 0)".into());
        let (pss, commit) = made.split_once('/').unwrap();
        let pss: Vec<&str> = pss.split(',').filter(|x| !x.is_empty()).collect();
        assert_eq!(pss.len(), fp_len);
        let got = step(format!(r#"Merge(s: {s}, gid: "{gid}", ep: {ep}, cth: "{cth}")"#));
        assert_eq!(got, format!("{th}/true"), "entry {e}: generated path merges to the sender's tree hash with a consistent parent hash");
        assert_eq!(step(format!(r#"SigOk(s: {s}, gid: "{gid}")"#)), "true", "entry {e}: new LeafNode signature verifies");
        for lp in &lps[1..] {
            let i = lp["index"].as_i64().unwrap();
            let held: String = lp["path_secrets"].as_array().unwrap().iter().map(|x| format!("{:02x}:{},", x["node"].as_i64().unwrap(), x["path_secret"].as_str().unwrap())).collect();
            let got = step(format!(r#"Recv(i: {i}, leaf_sk: "{}", pss: "{held}")"#, lp["encryption_priv"].as_str().unwrap()));
            let (ps, cs) = got.split_once('/').unwrap_or_else(|| panic!("entry {e}: leaf {i} decrypts: {got}"));
            assert!(pss.contains(&ps), "entry {e}: leaf {i} got one of the generated path secrets");
            assert_eq!(cs, commit, "entry {e}: leaf {i} derives the sender's commit secret");
            recvs += 1;
        }
    }
    assert!(recvs > 40, "{recvs}");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert!(text.contains("RandomDone"), "the random bytes are in the log");
    assert_eq!(replay(&m, &text).unwrap().1, live);
}

/// length of the filtered direct path of leaf s, by RFC 9420 tree math in Rust
fn filtered_len(tree: &str, s: i64) -> usize {
    let nodes = rust_nodes(tree);
    let w = nodes.len() as i64;
    let root = (w - 1) / 2;
    let level = |x: i64| (!x).trailing_zeros() as i64;
    let mut x = 2 * s;
    let mut n = 0;
    while x != root {
        let k = level(x);
        let q = (x & !(1 << (k + 1))) | (1 << k);
        let sb = if x < q { q ^ (3 << (level(q) - 1)) } else { q ^ (1 << (level(q) - 1)) };
        let lo = sb - ((1 << level(sb)) - 1);
        let hi = sb + ((1 << level(sb)) - 1);
        if (lo..=hi).any(|y| nodes[y as usize]) {
            n += 1;
        }
        x = q;
    }
    n
}

/// which nodes of the padded tree are non-blank
fn rust_nodes(tree: &str) -> Vec<bool> {
    let b = hex(tree);
    fn rv(b: &[u8], p: usize) -> (usize, usize) {
        match b[p] >> 6 {
            0 => ((b[p] & 63) as usize, p + 1),
            1 => ((((b[p] & 63) as usize) << 8) | b[p + 1] as usize, p + 2),
            _ => ((((b[p] & 63) as usize) << 24) | ((b[p + 1] as usize) << 16) | ((b[p + 2] as usize) << 8) | b[p + 3] as usize, p + 4),
        }
    }
    let sk = |p: usize| {
        let (n, q) = rv(&b, p);
        q + n
    };
    let (len, mut p) = rv(&b, 0);
    let end = p + len;
    let mut out = vec![];
    while p < end {
        if b[p] == 0 {
            out.push(false);
            p += 1;
            continue;
        }
        out.push(true);
        let typ = b[p + 1];
        let s = p + 2;
        p = if typ == 2 {
            sk(sk(sk(s)))
        } else {
            let mut q = sk(sk(s)) + 2;
            q = sk(q);
            for _ in 0..5 {
                q = sk(q);
            }
            let src = b[q];
            q += 1;
            if src == 1 {
                q += 16
            }
            if src == 3 {
                q = sk(q)
            }
            sk(sk(q))
        };
    }
    let mut l = 1;
    while 2 * l - 1 < out.len() {
        l *= 2;
    }
    out.resize(2 * l - 1, false);
    out
}
