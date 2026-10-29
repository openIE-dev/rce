//! A tokio service around eight programs printed by `rce emit rust`.
//!
//! Events arrive as newline-delimited JSON on TCP (tokio's net streams). One
//! admitter task owns the program and applies each event in arrival order;
//! each reply is one JSON line with the outcome, the lines `rce run` prints
//! for the event and the state.
//!
//!   rce-interop drive NAME STREAM.json   serve on 127.0.0.1, send the stream
//!                                        as a tokio client, print the transcript
//!   rce-interop bench NAME STREAM.json N per-call boundary and step cost (ns)

macro_rules! programs {
    ($($name:ident = $file:literal),*) => {
        $(
            #[path = "gen"]
            mod $name {
                #[path = $file]
                pub mod m;
                include!("adapter.rs");
            }
        )*
        fn program(name: &str) -> Option<Box<dyn Host>> {
            match name {
                $(stringify!($name) => Some(Box::new($name::Program::new().ok()?)),)*
                _ => None,
            }
        }
        fn bench(name: &str, stream: &[serde_json::Value], n: usize) -> Option<String> {
            match name {
                $(stringify!($name) => Some(bench::run(stream, n, || $name::Program::new().unwrap(),
                    |p, j| p.event(j).unwrap(), |p, e| { let _ = p.step(e); }, |p| p.state_json().to_string())),)*
                _ => None,
            }
        }
    };
}
programs!(counter = "counter.rs", bank = "bank.rs", multi_region = "multi_region.rs", traffic_light = "traffic_light.rs",
    door = "door.rs", inventory = "inventory.rs", rcedb = "rcedb.rs", float_keys = "float_keys.rs");

mod bench;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

/// What the service needs of a program, whatever its emitted value type.
pub trait Host: Send {
    fn initial_line(&self) -> String;
    fn apply(&mut self, seq: u64, line: &str) -> serde_json::Value;
}

type Msg = (String, oneshot::Sender<serde_json::Value>);

/// The single writer: one event at a time, in arrival order.
async fn admitter(mut p: Box<dyn Host>, mut rx: mpsc::Receiver<Msg>) {
    let mut seq = 0;
    while let Some((line, reply)) = rx.recv().await {
        let r = p.apply(seq + 1, &line);
        if r["outcome"] != "not-admitted" {
            seq += 1;
        }
        let _ = reply.send(r);
    }
}

async fn connection(s: TcpStream, first: String, tx: mpsc::Sender<Msg>) -> std::io::Result<()> {
    let (r, mut w) = s.into_split();
    w.write_all(format!("{}\n", serde_json::json!({"lines": [first]})).as_bytes()).await?;
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        let (rtx, rrx) = oneshot::channel();
        if tx.send((line, rtx)).await.is_err() {
            break;
        }
        let reply = rrx.await.unwrap_or(serde_json::Value::Null);
        w.write_all(format!("{reply}\n").as_bytes()).await?;
    }
    Ok(())
}

async fn drive(name: &str, stream: Vec<serde_json::Value>) -> std::io::Result<Vec<String>> {
    let p = program(name).ok_or_else(|| std::io::Error::other(format!("no program {name}")))?;
    let first = p.initial_line();
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(admitter(p, rx));
    let l = TcpListener::bind("127.0.0.1:0").await?;
    let addr = l.local_addr()?;
    tokio::spawn(async move {
        while let Ok((s, _)) = l.accept().await {
            tokio::spawn(connection(s, first.clone(), tx.clone()));
        }
    });
    let (r, mut w) = TcpStream::connect(addr).await?.into_split();
    let mut replies = BufReader::new(r).lines();
    let mut out = vec![];
    let lines_of = |j: serde_json::Value| -> Vec<String> {
        j["lines"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect()
    };
    out.extend(lines_of(serde_json::from_str(&replies.next_line().await?.unwrap_or_default())?));
    for ev in stream {
        w.write_all(format!("{ev}\n").as_bytes()).await?;
        out.extend(lines_of(serde_json::from_str(&replies.next_line().await?.unwrap_or_default())?));
    }
    Ok(out)
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let stream: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&a[3]).expect("stream file")).expect("stream JSON");
    match a[1].as_str() {
        "drive" => {
            let out = drive(&a[2], stream).await.expect("drive");
            print!("{}", out.iter().map(|l| format!("{l}\n")).collect::<String>());
        }
        "bench" => print!("{}", bench(&a[2], &stream, a[4].parse().unwrap()).expect("program")),
        m => panic!("unknown mode {m}"),
    }
}
