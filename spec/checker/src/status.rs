//! HC-28: what the host loop reports to its terminal.
//!
//! `rce host run` and `rce host serve` say whether they are waiting for
//! input, admitting one, stopped or stopped on a failure, as records of
//! OSC 7501, the Program Status Protocol that libghostty-vt parses and
//! Herdr shows. A record is `ESC ] 7501 ; key=value:key=value ESC \`; the
//! keys used here are `state`, `app` (a name: ASCII letters, digits and
//! `_.+-`, at most 32 bytes), and `title` and `msg` (standard base64 of
//! UTF-8 text with no control characters, at most 192 and 2048 bytes once
//! decoded).
//!
//! A report is not an effect: it is written to stderr only, never to D,
//! the program never reads it, and replay and the state hash do not
//! depend on it. It is written only when stderr is a terminal, or when
//! `RCE_STATUS=on`; `RCE_STATUS=off` turns it off.
use std::io::{IsTerminal, Write};

/// The `app` of every record.
pub const APP: &str = "rce";

/// The longest `msg` text, in bytes, before encoding.
pub const MSG_MAX: usize = 160;

/// The longest `title` text, in bytes, before encoding (the protocol's
/// limit).
pub const TITLE_MAX: usize = 192;

/// Whether the host writes reports, from the value of `RCE_STATUS` and
/// whether stderr is a terminal: `on` and `off` decide, and any other value
/// or none leaves it to the terminal.
pub fn enabled(var: Option<&str>, terminal: bool) -> bool {
    match var {
        Some("on") => true,
        Some("off") => false,
        _ => terminal,
    }
}

/// The host loop's reporter. It writes a record only when the state word
/// changes, so a run of inputs gives one `working` and one `idle` each.
pub struct Status {
    on: bool,
    title: String,
    last: Option<&'static str>,
}

impl Status {
    /// A reporter for the host loop when `active`, titled `title`; for any
    /// other command it writes nothing.
    pub fn new(active: bool, title: &str) -> Status {
        let on = active && enabled(std::env::var("RCE_STATUS").ok().as_deref(), std::io::stderr().is_terminal());
        Status { on, title: title.to_string(), last: None }
    }

    /// The title of the records that follow (the module name, once it is
    /// read).
    pub fn retitle(&mut self, title: &str) {
        self.title = title.to_string();
    }

    /// The host waits for input.
    pub fn idle(&mut self, entries: u64, state_hash: &str) {
        self.report("idle", &summary(entries, state_hash));
    }

    /// The host admits an input and steps the program.
    pub fn working(&mut self) {
        self.report("working", "admitting input");
    }

    /// The host stopped cleanly and D is complete.
    pub fn done(&mut self, entries: u64, state_hash: &str) {
        self.report("done", &format!("stopped: {}", summary(entries, state_hash)));
    }

    /// The host stopped on a failure, or a program requires change.
    pub fn error(&mut self, why: &str) {
        self.report("error", why);
    }

    fn report(&mut self, state: &'static str, msg: &str) {
        if !self.on || self.last == Some(state) {
            return;
        }
        self.last = Some(state);
        let mut e = std::io::stderr().lock();
        let _ = e.write_all(record(state, &self.title, msg).as_bytes());
        let _ = e.flush();
    }
}

/// `N entries, state H`, with the first 12 hex digits of the state hash.
fn summary(entries: u64, state_hash: &str) -> String {
    let short: String = state_hash.chars().take(12).collect();
    format!("{entries} entries, state {short}")
}

/// One OSC 7501 record for the root record of the terminal.
pub fn record(state: &str, title: &str, msg: &str) -> String {
    let mut fields = vec![format!("state={state}"), format!("app={APP}")];
    let title = text(title, TITLE_MAX);
    if !title.is_empty() {
        fields.push(format!("title={}", base64(title.as_bytes())));
    }
    let msg = text(msg, MSG_MAX);
    if !msg.is_empty() {
        fields.push(format!("msg={}", base64(msg.as_bytes())));
    }
    format!("\x1b]7501;{}\x1b\\", fields.join(":"))
}

/// `s` as one line of at most `max` bytes: each control character becomes
/// a space, and the text is cut at a character boundary.
fn text(s: &str, max: usize) -> String {
    let line: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let line = line.trim();
    let mut end = line.len().min(max);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line[..end].trim_end().to_string()
}

/// Standard base64 with padding (RFC 4648 section 4).
pub fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { TABLE[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_is_the_standard_alphabet_with_padding() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"ab"), "YWI=");
        assert_eq!(base64(b"Apply?"), "QXBwbHk/");
        assert_eq!(base64(b"counter"), "Y291bnRlcg==");
    }

    #[test]
    fn a_record_has_state_app_title_and_msg() {
        assert_eq!(record("idle", "counter", ""), "\x1b]7501;state=idle:app=rce:title=Y291bnRlcg==\x1b\\");
        assert_eq!(record("done", "", "ab"), "\x1b]7501;state=done:app=rce:msg=YWI=\x1b\\");
    }

    #[test]
    fn text_is_one_line_within_its_limit() {
        assert_eq!(text("a\nb\tc\x1b", 99), "a b c");
        assert_eq!(text("ééé", 5), "éé");
        assert_eq!(text(&"x".repeat(300), MSG_MAX).len(), MSG_MAX);
    }

    #[test]
    fn the_variable_decides_before_the_terminal() {
        assert!(enabled(Some("on"), false));
        assert!(!enabled(Some("off"), true));
        assert!(enabled(None, true));
        assert!(!enabled(None, false));
        assert!(!enabled(Some("yes"), false));
    }
}
