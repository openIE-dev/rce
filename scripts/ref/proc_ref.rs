// Reference program for scripts/gen/proc_gen.py: reads lines from stdin and
// prints what Rust std (rustc 1.99.0, x86_64-unknown-linux-gnu) gives.
//  "S <raw>"     -> Display of ExitStatus::from_raw, code, signal,
//                   core_dumped, stopped_signal, continued, success
//  "C op:hex ..." -> Command built by the ops, then {:?} and {:#?}
// Strings are hex-encoded UTF-8. Fields of the reply are hex-encoded and
// joined by " ".
use std::io::BufRead;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Command, ExitStatus, Stdio};
fn hx(s: &str) -> String { s.bytes().map(|b| format!("{:02x}", b)).collect() }
fn unhx(s: &str) -> String {
    let b: Vec<u8> = (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
    String::from_utf8(b).unwrap()
}
fn opt(o: Option<i32>) -> String { match o { Some(v) => v.to_string(), None => "none".into() } }
fn stdio(v: &str) -> Stdio { match v { "null" => Stdio::null(), "piped" => Stdio::piped(), _ => Stdio::inherit() } }
fn main() {
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut it = line.split(' ');
        match it.next().unwrap() {
            "S" => {
                let e = ExitStatus::from_raw(it.next().unwrap().parse().unwrap());
                println!("{} {} {} {} {} {} {}", hx(&e.to_string()), opt(e.code()), opt(e.signal()), e.core_dumped(), opt(e.stopped_signal()), e.continued(), e.success());
            }
            "C" => {
                let ops: Vec<(String, String)> = it.map(|o| { let (a, b) = o.split_once(':').unwrap(); (a.to_string(), unhx(b)) }).collect();
                let mut c = Command::new(&ops[0].1);
                for (k, v) in &ops[1..] {
                    match k.as_str() {
                        "arg" => { c.arg(v); }
                        "arg0" => { c.arg0(v); }
                        "env" => { let (a, b) = v.split_once('=').unwrap(); c.env(a, b); }
                        "rm" => { c.env_remove(v); }
                        "clear" => { c.env_clear(); }
                        "cwd" => { c.current_dir(v); }
                        "stdin" => { c.stdin(stdio(v)); }
                        "stdout" => { c.stdout(stdio(v)); }
                        "stderr" => { c.stderr(stdio(v)); }
                        "pgroup" => { c.process_group(v.parse().unwrap()); }
                        "uid" => { c.uid(v.parse().unwrap()); }
                        "gid" => { c.gid(v.parse().unwrap()); }
                        _ => panic!("{}", k),
                    }
                }
                println!("{} {}", hx(&format!("{:?}", c)), hx(&format!("{:#?}", c)));
            }
            _ => panic!(),
        }
    }
}
