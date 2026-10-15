//! W3 RCEdb v0: embedded, in-process, written in RCE over D. Rules R1-R8
//! (src/rcedb.rs): writes are logged events, queries are RCE steps in
//! canonical order that never append, as-of is a prefix fold, reopen
//! refolds D, and emitted SQL loads into SQLite with the same rows.
use rce_spec_check::rcedb::{Db, Dialect};

fn tmp(n: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rcedb-{n}-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn rows_edges_queries_as_of_and_reopen() {
    let p = tmp("core");
    let mut db = Db::open(&p).unwrap();
    db.insert("users", "bob", r#"{"name":"Bob"}"#).unwrap();
    db.insert("users", "alice", r#"{"name":"Alice"}"#).unwrap();
    let mid = db.seq() as usize;
    db.insert("msgs", "0001", r#"{"from":"alice","text":"it's on"}"#).unwrap();
    db.link("alice", "follows", "bob").unwrap();
    db.link("alice", "blocks", "carol").unwrap();
    assert!(db.delete("users", "nobody").is_err(), "refused write");
    let users = db.scan("users", "").unwrap();
    assert_eq!(users.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["alice", "bob"], "canonical order");
    assert_eq!(db.scan("users", "b").unwrap().len(), 1);
    let e = db.edges("alice").unwrap();
    assert_eq!(e.iter().map(|x| x.1.as_str()).collect::<Vec<_>>(), ["blocks", "follows"]);
    let n = db.seq();
    db.scan("users", "").unwrap();
    assert_eq!(db.seq(), n, "a query appends nothing");
    assert!(db.durable().is_some());
    let mut past = db.as_of(mid).unwrap();
    assert_eq!(past.scan("msgs", "").unwrap().len(), 0);
    assert_eq!(past.scan("users", "").unwrap().len(), 2);
    let h = db.state_hash().to_string();
    drop(db);
    let mut db = Db::open(&p).unwrap();
    assert_eq!(db.scan("users", "").unwrap().len(), 2);
    assert_eq!(db.dump().unwrap(), {
        let mut d2 = Db::open(&p).unwrap();
        d2.dump().unwrap()
    });
    let _ = h;
    let _ = std::fs::remove_file(&p);
}

#[test]
fn emitted_sql_loads_into_sqlite() {
    let p = tmp("emit");
    let mut db = Db::open(&p).unwrap();
    db.insert("users", "o'neil", r#"{"name":"O'Neil"}"#).unwrap();
    db.insert("users", "ann", "{}").unwrap();
    db.link("ann", "follows", "o'neil").unwrap();
    for d in [Dialect::Sqlite, Dialect::Postgres, Dialect::DuckDb, Dialect::Cypher] {
        let s = db.emit(d).unwrap();
        assert!(s.contains(db.state_hash()), "provenance");
        assert!(s.contains("O''Neil") || s.contains("O\\'Neil"), "{s}");
    }
    let sql = db.emit(Dialect::Sqlite).unwrap();
    use std::io::Write;
    let Ok(mut c) = std::process::Command::new("sqlite3")
        .arg(":memory:")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    else {
        eprintln!("sqlite3 not installed: load check skipped");
        return;
    };
    write!(c.stdin.take().unwrap(), "{sql}\nSELECT key FROM users ORDER BY key; SELECT count(*) FROM _edges;\n").unwrap();
    let o = c.wait_with_output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(out.trim(), "ann\no'neil\n1");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn typed_schema_refuses_bad_rows_and_index_matches_scan() {
    let p = tmp("typed");
    let mut db = Db::open(&p).unwrap();
    db.insert("free", "x", "anything").unwrap();
    assert!(db.schema("free", "a:text").is_err(), "no schema once a table has rows");
    assert!(db.schema("u", "a:float").is_err(), "unknown type");
    assert!(db.schema("u", "a:text,a:i64").is_err(), "duplicate column");
    db.schema("u", "name:text,age:i64,admin:bool").unwrap();
    for bad in ["Ann\t41", "Ann\tforty\tfalse", "Ann\t041\tfalse", "Ann\t1\tyes", "Ann\t9223372036854775808\tfalse"] {
        let n = db.seq();
        assert!(db.insert("u", "a", bad).is_err(), "{bad:?} refused");
        assert!(db.scan("u", "").unwrap().is_empty());
        let _ = n;
    }
    db.insert("u", "b", "Bob\t30\tfalse").unwrap();
    db.insert("u", "a", "Ann\t30\ttrue").unwrap();
    db.insert("u", "c", "Cy\t-9223372036854775808\tfalse").unwrap();
    let by_scan = db.lookup("u", "age", "30").unwrap();
    assert!(db.index("u", "nope").is_err());
    db.index("u", "age").unwrap();
    assert!(db.index("u", "age").is_err(), "index exists");
    assert_eq!(db.lookup("u", "age", "30").unwrap(), by_scan, "index and scan agree");
    assert_eq!(by_scan.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    db.insert("u", "b", "Bob\t31\tfalse").unwrap();
    db.delete("u", "a").unwrap();
    assert!(db.lookup("u", "age", "30").unwrap().is_empty(), "index follows updates and deletes");
    assert_eq!(db.lookup("u", "age", "31").unwrap().len(), 1);
    let d = db.describe().unwrap();
    assert_eq!(d[1], ("u".into(), 2, "name:text,age:i64,admin:bool".into(), "age".into()));
    let before = (db.dump().unwrap(), d);
    drop(db);
    let mut db = Db::open(&p).unwrap();
    assert_eq!((db.dump().unwrap(), db.describe().unwrap()), before, "the index is a fold of D: reopen rebuilds it");
    assert_eq!(db.lookup("u", "age", "31").unwrap().len(), 1);
    let _ = std::fs::remove_file(&p);
}

#[test]
fn rce_db_cli() {
    let p = tmp("cli");
    let ps = p.to_str().unwrap().to_string();
    let run = |a: &[&str]| -> (i32, String) {
        let mut args = vec!["db".to_string(), ps.clone()];
        args.extend(a.iter().map(|s| s.to_string()));
        let (mut o, mut e) = (vec![], vec![]);
        let c = rce_spec_check::cli::main(&args, &mut std::io::empty(), &mut o, &mut e);
        (c, String::from_utf8(o).unwrap() + &String::from_utf8(e).unwrap())
    };
    assert_eq!(run(&["schema", "u", "name:text,age:i64"]).0, 0);
    assert_eq!(run(&["insert", "u", "b", "Bob\t7"]).0, 0);
    let (c, o) = run(&["insert", "u", "c", "Cy\tseven"]);
    assert_eq!(c, 1, "{o}");
    assert!(o.starts_with("REFUSED"), "{o}");
    assert_eq!(run(&["index", "u", "age"]).0, 0);
    assert_eq!(run(&["lookup", "u", "age", "7"]).1, "b\tBob\t7\n");
    assert_eq!(run(&["scan", "u"]).1, "b\tBob\t7\n");
    assert!(run(&["describe"]).1.contains("indexes age"));
    assert!(run(&["emit", "sqlite"]).1.contains("INSERT INTO \"u\""));
    assert_ne!(run(&["frobnicate"]).0, 0);
    let _ = std::fs::remove_file(&p);
}
