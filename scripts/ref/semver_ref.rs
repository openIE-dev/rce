// Reference for os.pkg's semver port: the semver crate Cargo uses.
// Build: a cargo project with `semver = "=1.0.28"`; stdin lines are
// TAB-separated requests, one answer line each (escapes: \t \n \\ \0).
use semver::{BuildMetadata, Comparator, Prerelease, Version, VersionReq};
use std::io::{self, BufRead, Write};
fn un(s: &str) -> String {
    let mut o = String::new(); let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' { match it.next() { Some('t') => o.push('\t'), Some('n') => o.push('\n'), Some('0') => o.push('\0'), Some(x) => o.push(x), None => {} } } else { o.push(c) }
    }
    o
}
fn r<T: std::fmt::Display, E: std::fmt::Display>(x: Result<T, E>) -> String {
    match x { Ok(v) => format!("ok {}", v), Err(e) => format!("err {}", e) }
}
fn main() {
    let out = io::stdout(); let mut out = out.lock();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap(); let f: Vec<String> = line.split('\t').map(un).collect();
        let a = match f[0].as_str() {
            "V" => r(Version::parse(&f[1])),
            "R" => r(VersionReq::parse(&f[1])),
            "C" => r(Comparator::parse(&f[1])),
            "P" => r(Prerelease::new(&f[1])),
            "B" => r(BuildMetadata::new(&f[1])),
            "O" => { let a = Version::parse(&f[1]).unwrap(); let b = Version::parse(&f[2]).unwrap(); format!("{}", a.cmp(&b) as i32) }
            "M" => { let q = VersionReq::parse(&f[1]).unwrap(); let v = Version::parse(&f[2]).unwrap(); format!("{}", q.matches(&v)) }
            "K" => { let q = Comparator::parse(&f[1]).unwrap(); let v = Version::parse(&f[2]).unwrap(); format!("{}", q.matches(&v)) }
            _ => panic!(),
        };
        writeln!(out, "{}", a.replace('\\', "\\\\").replace('\n', "\\n")).unwrap();
    }
}
