//! TreeKEM ratchet trees in pure RCE (crates/rce/lib/treekem.rce on top of
//! lib/crypto.rce and lib/mls.rce) against the official RFC 9420
//! tree-validation.json vectors for cipher suite 1: every tree decodes, every
//! tree hash and every resolution matches, and every parent node is
//! parent-hash valid (the 11 trees up to 16 leaves; the three
//! 64-leaf trees need more than one input's fuel and wait for a split); a tree with one parent hash changed is
//! not. One computation per host input (HC-7a); the log replays.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::Path;

fn combined() -> String {
    let mut s = String::from("module treekem_steps\n");
    for f in ["lib/crypto.rce", "lib/mls.rce", "lib/treekem.rce"] {
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
event Hashes(t: text)
event Res(t: text)
event Valid(t: text, p: i64)

state S
    out: List[text] = []

step S on Hashes(t: t)
    var s = ""
    for h in tree_hashes(rt_decode(t))
        s = s + h + ","
    out.push(s)

step S on Res(t: t)
    let nodes = rt_decode(t)
    var s = ""
    for x in 0..nodes.len()
        for r in resolution(copy(nodes), x as i64)
            s = s + r.tostr() + ","
        s = s + ";"
    out.push(s)

step S on Valid(t: t, p: p)
    out.push(parent_hash_valid(rt_decode(t), p).tostr())
"#;

fn run(max_leaves: usize) {
    let src = combined();
    let m = parse(&src).unwrap();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/mls/tree-validation-suite1.json")).unwrap()).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut step = |ev: String| -> String {
        let n0 = format!("{:?}", h.state()).matches("Text(\"").count();
        h.admit(&src, &ev).unwrap();
        let s = format!("{:?}", h.state());
        let o: Vec<&str> = s.split("Text(\"").skip(1).map(|x| x.split('"').next().unwrap()).collect();
        assert_eq!(o.len(), n0 + 1, "step admitted within its fuel: {ev:.60}");
        o[n0].to_string()
    };
    let mut checked = 0;
    for (i, t) in v.as_array().unwrap().iter().enumerate() {
        let tree = t["tree"].as_str().unwrap();
        let want: Vec<&str> = t["tree_hashes"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        if (want.len() + 1) / 2 > max_leaves {
            continue; // 64-leaf trees exceed one input's fuel; split per subtree later
        }
        let got = step(format!(r#"Hashes(t: "{tree}")"#));
        let got: Vec<&str> = got.split(',').collect();
        assert_eq!(&got[..want.len()], &want[..], "tree-validation {i}: tree hashes");
        let res = step(format!(r#"Res(t: "{tree}")"#));
        let res: Vec<Vec<i64>> = res.split(';').map(|x| x.split(',').filter(|y| !y.is_empty()).map(|y| y.parse().unwrap()).collect()).collect();
        for (x, r) in t["resolutions"].as_array().unwrap().iter().enumerate() {
            let w: Vec<i64> = r.as_array().unwrap().iter().map(|y| y.as_i64().unwrap()).collect();
            assert_eq!(res[x], w, "tree-validation {i}: resolution of node {x}");
        }
        if (want.len() + 1) / 2 <= max_leaves {
            for p in (1..want.len()).step_by(2) {
                assert_eq!(step(format!(r#"Valid(t: "{tree}", p: {p})"#)), "true", "tree-validation {i}: parent {p} parent-hash valid");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 11);
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


