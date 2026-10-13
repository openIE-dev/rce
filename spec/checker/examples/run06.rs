//! `cargo run --example run06 -- FILE [EVENTS]`: run a module as `rce run` does.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let src = std::fs::read_to_string(&a[0]).expect("read");
    match rce_spec_check::cli::run(&src, a.get(1).map(String::as_str)) {
        Ok(r) => r.lines.iter().for_each(|l| println!("{l}")),
        Err(e) => println!("REFUSED {e}"),
    }
}
