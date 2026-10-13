//! `rce lsp` language features (10.7): hover, go-to-definition, completion
//! and document symbols, through the server's message handler.
use rce_spec_check::cli::Server;
use serde_json::{Value, json};

const DOC: &str = "\
kind Shape
    Circle(r: f64)
    Square(side: f64)

struct Point
    x: i64
    y: i64

const LIMIT: i64 = 10

fn area(s: Shape) -> f64
    match s
        Circle(r: r) -> return 3.0 * r * r
        Square(side: x) -> return x * x

fn sum_to(n: i64) -> i64
    var total: i64 = 0
    for i in 0..n
        total += i
    return total

impl Point
    fn norm1(self) -> i64 = self.x + self.y

fn use_point(p: Point) -> i64 = p.norm1() + LIMIT
";

fn server() -> Server {
    let mut s = Server::default();
    s.handle(&json!({"jsonrpc": "2.0", "method": "textDocument/didOpen",
                     "params": {"textDocument": {"uri": "file:///a.rce", "text": DOC}}}));
    s
}

fn ask(s: &mut Server, method: &str, line: usize, ch: usize) -> Value {
    let (out, _) = s.handle(&json!({"jsonrpc": "2.0", "id": 7, "method": method,
        "params": {"textDocument": {"uri": "file:///a.rce"}, "position": {"line": line, "character": ch}}}));
    assert_eq!(out.len(), 1);
    out[0]["result"].clone()
}

/// 0-based (line, column) of the n-th occurrence of `w` in DOC.
fn at(w: &str, n: usize) -> (usize, usize) {
    let mut k = 0;
    for (l, text) in DOC.lines().enumerate() {
        let mut from = 0;
        while let Some(i) = text[from..].find(w) {
            if k == n {
                return (l, text[..from + i].chars().count());
            }
            k += 1;
            from += i + w.len();
        }
    }
    panic!("{w} #{n} not in DOC")
}

fn hover_text(s: &mut Server, w: &str, n: usize) -> String {
    let (l, c) = at(w, n);
    let h = ask(s, "textDocument/hover", l, c + 1);
    h["contents"]["value"].as_str().unwrap_or("").to_string()
}

#[test]
fn initialize_advertises_the_features() {
    let mut s = Server::default();
    let (out, _) =
        s.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}));
    let c = &out[0]["result"]["capabilities"];
    assert_eq!(c["hoverProvider"], true);
    assert_eq!(c["definitionProvider"], true);
    assert_eq!(c["documentSymbolProvider"], true);
    assert_eq!(c["completionProvider"]["triggerCharacters"], json!(["."]));
}

#[test]
fn hover_gives_types_of_locals_and_signatures_of_items() {
    let mut s = server();
    // a local, typed by the checker
    assert_eq!(hover_text(&mut s, "total", 2), "```rce\ntotal: i64\n```");
    let (l, c) = at("+= i", 0);
    let h = ask(&mut s, "textDocument/hover", l, c + 3);
    assert_eq!(h["contents"]["value"], "```rce\ni: i64\n```");
    // a function call shows the signature
    let (l, c) = at("norm1", 1);
    let h = ask(&mut s, "textDocument/hover", l, c);
    assert_eq!(
        h["contents"]["value"],
        "```rce\nfn norm1(self) -> i64 = self.x + self.y\n```"
    );
    assert!(hover_text(&mut s, "LIMIT", 1).contains("const LIMIT: i64 = 10"));
    assert!(hover_text(&mut s, "Shape", 1).contains("kind Shape\n    Circle(r: f64)"));
    assert!(hover_text(&mut s, "Circle", 1).contains("Circle(r: f64)"));
    // keywords and blanks have no hover
    let (l, c) = at("return", 0);
    assert_eq!(ask(&mut s, "textDocument/hover", l, c), Value::Null);
}

#[test]
fn definition_goes_to_the_binding_or_item() {
    let mut s = server();
    let def = |s: &mut Server, w: &str, n: usize| {
        let (l, c) = at(w, n);
        let d = ask(s, "textDocument/definition", l, c);
        (
            d["range"]["start"]["line"].as_u64().unwrap() as usize,
            d["range"]["start"]["character"].as_u64().unwrap() as usize,
        )
    };
    assert_eq!(def(&mut s, "total", 2), at("total", 0));
    assert_eq!(def(&mut s, "LIMIT", 1), at("LIMIT", 0));
    assert_eq!(def(&mut s, "norm1", 1), at("norm1", 0));
    assert_eq!(def(&mut s, "Point", 2), at("Point", 0));
    let (l, c) = at("match", 0);
    assert_eq!(ask(&mut s, "textDocument/definition", l, c), Value::Null);
}

#[test]
fn completion_offers_names_in_scope_and_methods_after_dot() {
    let mut s = server();
    let labels = |v: Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x["label"].as_str().unwrap().to_string())
            .collect()
    };
    // inside sum_to, after "to" of "total += i": prefix "to"
    let (l, c) = at("total", 1);
    let got = labels(ask(&mut s, "textDocument/completion", l, c + 2));
    assert!(got.contains(&"total".to_string()), "{got:?}");
    assert!(got.iter().all(|x| x.starts_with("to")), "{got:?}");
    // at the start of that line: locals, items, variants and keywords
    let got = labels(ask(&mut s, "textDocument/completion", l, c));
    for w in [
        "total", "n", "i", "area", "LIMIT", "Shape", "Circle", "Point", "while",
    ] {
        assert!(got.contains(&w.to_string()), "{w} not in {got:?}");
    }
    // a binding of another function is not offered
    assert!(!got.contains(&"s".to_string()), "{got:?}");
    // after "p." in use_point: user methods, fields and builtin methods
    let (l, c) = at("p.norm1", 0);
    let got = labels(ask(&mut s, "textDocument/completion", l, c + 2));
    for w in ["norm1", "x", "len", "map"] {
        assert!(got.contains(&w.to_string()), "{w} not in {got:?}");
    }
    assert!(!got.contains(&"while".to_string()));
}

#[test]
fn document_symbols_list_items_with_children() {
    let mut s = server();
    let v = ask(&mut s, "textDocument/documentSymbol", 0, 0);
    let names: Vec<&str> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "Shape",
            "Point",
            "LIMIT",
            "area",
            "sum_to",
            "impl Point",
            "use_point"
        ]
    );
    let shape = &v[0];
    assert_eq!(shape["kind"], 10);
    assert_eq!(shape["range"]["start"]["line"], 0);
    assert_eq!(shape["range"]["end"]["line"], 2);
    let kids: Vec<&str> = shape["children"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["name"].as_str().unwrap())
        .collect();
    assert_eq!(kids, ["Circle", "Square"]);
    assert_eq!(
        shape["children"][1]["selectionRange"]["start"],
        json!({"line": 2, "character": 4})
    );
    assert_eq!(v[1]["children"][0]["name"], "x");
    assert_eq!(v[3]["kind"], 12);
    assert_eq!(v[5]["children"][0]["name"], "norm1");
    assert_eq!(v[5]["children"][0]["kind"], 6);
}

#[test]
fn unparsable_document_gives_empty_answers() {
    let mut s = Server::default();
    s.handle(&json!({"jsonrpc": "2.0", "method": "textDocument/didOpen",
                     "params": {"textDocument": {"uri": "file:///a.rce", "text": "fn (\n"}}}));
    assert_eq!(ask(&mut s, "textDocument/hover", 0, 1), Value::Null);
    assert_eq!(ask(&mut s, "textDocument/documentSymbol", 0, 0), json!([]));
}
