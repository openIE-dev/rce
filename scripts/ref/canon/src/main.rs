// Reads one JSON value per stdin line; prints `<sha256> <canonical json>`.
// write_canonical is rce src/canon.rs (openIE-dev/rce), unchanged.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::BufRead;

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            out.push_str(&serde_json::to_string(v).expect("scalar json"));
        }
        Value::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(x, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).expect("key json"));
                out.push(':');
                write_canonical(&map[k.as_str()], out);
            }
            out.push('}');
        }
    }
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let mut s = String::new();
        write_canonical(&v, &mut s);
        let h: String = Sha256::digest(s.as_bytes()).iter().map(|b| format!("{:02x}", b)).collect();
        println!("{} {}", h, s);
    }
}
