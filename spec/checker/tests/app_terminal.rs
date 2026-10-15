//! W4 terminal v2: the screen model and scrollback of apps/terminal.rce,
//! fed PTY output as logged PtyReadDone events (no PTY needed): wrapping,
//! scrolling into the scrollback, CSI cursor moves and erase, resize, the
//! Screen and History queries, and replay of D.
use rce_spec_check::host::{Host, MemStore, replay};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use std::path::Path;

fn displays(h: &mut Host, ev: &str, args: Vec<rce_spec_check::ast::Arg>) -> Vec<String> {
    h.query(ev, args).unwrap().into_iter().filter_map(|o| match o {
        V::Rec(_, fs) => fs.into_iter().find_map(|(f, v)| match v { V::Text(t) if f == "data" => Some(t), _ => None }),
        _ => None,
    }).collect()
}

fn read(i: u64, s: &str) -> String {
    format!(r#"PtyReadDone(req: {i}, status: "delivered", data: some({}), err: "")"#, serde_json::to_string(s).unwrap().replace("\\u001b", "\\u{1b}"))
}

#[test]
fn screen_scrollback_and_replay() {
    use rce_spec_check::ast::{Arg, Expr};
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/terminal.rce")).unwrap();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&src, r#"Tick, PtyOpenDone(req: 1, status: "delivered", data: some(7), err: ""), Resize(rows: 5, cols: 20)"#).unwrap();
    let out: String = (1..=12).map(|n| format!("line {n}\r\n")).collect();
    h.admit(&src, &read(2, &out)).unwrap();
    let scr = displays(&mut h, "Screen", vec![]);
    assert_eq!(scr[..5], ["line 9\n", "line 10\n", "line 11\n", "line 12\n", "\n"].map(String::from), "{scr:?}");
    assert_eq!(scr[5], "cursor 5;1\n");
    let int = |n: i128| Expr::Int(n, 0);
    let hist = displays(&mut h, "History", vec![Arg { label: Some("start".into()), e: int(0) }, Arg { label: Some("count".into()), e: int(3) }]);
    assert_eq!(hist, ["line 1\n", "line 2\n", "line 3\n"].map(String::from));
    // a long line wraps at the right edge
    h.admit(&src, &read(3, "abcdefghijklmnopqrstuvwxyz")).unwrap();
    let scr = displays(&mut h, "Screen", vec![]);
    assert_eq!(scr[3], "abcdefghijklmnopqrst\n", "{scr:?}");
    assert_eq!(scr[4], "uvwxyz\n", "{scr:?}");
    // clear, move, write, erase to end of line
    h.admit(&src, &read(4, "\u{1b}[2J\u{1b}[2;5Hhello\u{1b}[2;7H\u{1b}[K\u{1b}[1m!\u{1b}[0m")).unwrap();
    let scr = displays(&mut h, "Screen", vec![]);
    assert_eq!(scr[..3], ["\n", "    he!\n", "\n"].map(String::from), "{scr:?}");
    assert_eq!(scr[5], "cursor 2;8\n");
    let live = h.state_hash().to_string();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay(&m, &text).unwrap().1, live);
}
