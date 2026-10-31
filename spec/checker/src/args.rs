//! Positional command lines (spec/proposals/positional-cli.md, PC-1 to PC-8).
//!
//! An invocation is `PROGRAM COMMAND-PATH WORD... OPERAND...` (PC-1). Each
//! binary declares its commands once, as a table of [`Command`]s (PC-8):
//! the parser reads the table, and the usage text is printed from it, so the
//! two cannot disagree. No argument starts with `-` (PC-6); a removed flag
//! is a usage error whose one line names its new spelling (decision 2 of
//! section 5), and the command is not run.

/// A word of a command (PC-3, PC-4): a mode word has no values, a setting
/// word one or two. `after` changes the values a word takes once another
/// word was given (PC-4: after `tls`, a `peer` takes the address and its
/// pinned key), so that other word must come first.
#[derive(Debug, Clone, Copy)]
pub struct Word {
    pub name: &'static str,
    pub values: &'static [&'static str],
    pub repeat: bool,
    pub after: Option<(&'static str, &'static [&'static str])>,
}

impl Word {
    /// A mode word: no value.
    pub const fn mode(name: &'static str) -> Word {
        Word { name, values: &[], repeat: false, after: None }
    }
    /// A setting word with the named values.
    pub const fn set(name: &'static str, values: &'static [&'static str]) -> Word {
        Word { name, values, repeat: false, after: None }
    }
    /// A setting word that may be given more than once.
    pub const fn many(name: &'static str, values: &'static [&'static str]) -> Word {
        Word { name, values, repeat: true, after: None }
    }
    /// A word whose values are `then` once the word `first` was given.
    pub const fn then(self, first: &'static str, then: &'static [&'static str]) -> Word {
        Word { after: Some((first, then)), ..self }
    }
    fn synopsis(&self) -> String {
        let mut s = self.name.to_string();
        for v in self.values {
            s.push(' ');
            s.push_str(v);
        }
        let mut s = format!("[{s}]");
        if self.repeat {
            s.push_str("...");
        }
        s
    }
}

/// One command of a binary: its path, its words, its operands as the usage
/// line shows them, how many operands it takes, and what it does.
#[derive(Debug, Clone, Copy)]
pub struct Command {
    pub path: &'static [&'static str],
    pub words: &'static [Word],
    pub operands: &'static str,
    pub min: usize,
    pub max: usize,
    pub help: &'static str,
}

/// No upper bound on the number of operands.
pub const MANY: usize = usize::MAX;

impl Command {
    /// The usage line: program, path, words, operands.
    pub fn synopsis(&self, program: &str) -> String {
        let mut parts = vec![program.to_string()];
        parts.extend(self.path.iter().map(|p| p.to_string()));
        parts.extend(self.words.iter().map(Word::synopsis));
        if !self.operands.is_empty() {
            parts.push(self.operands.to_string());
        }
        parts.join(" ")
    }
    fn word(&self, name: &str) -> Option<&Word> {
        self.words.iter().find(|w| w.name == name)
    }
}

/// The usage text of a binary, printed from its table (PC-8): one line per
/// command, the synopsis then what it does; a help text's further lines are
/// indented under it.
pub fn usage(program: &str, table: &[Command]) -> String {
    let mut s = String::from("usage:");
    for c in table {
        let syn = c.synopsis(program);
        let mut help = c.help.lines();
        let first = help.next().unwrap_or("");
        let pad = if syn.len() < 31 { 31 - syn.len() } else { 2 };
        s.push_str(&format!("\n  {syn}{}{first}", " ".repeat(pad)));
        for l in help {
            s.push_str(&format!("\n{}{}", " ".repeat(33), l.trim_start()));
        }
    }
    s
}

/// The command a command line names: the longest path of the table that
/// the arguments start with, and the arguments after it.
pub fn find<'t, 'a>(table: &'t [Command], args: &'a [String]) -> Option<(&'t Command, &'a [String])> {
    table
        .iter()
        .filter(|c| c.path.len() <= args.len() && c.path.iter().zip(args).all(|(p, a)| p == a))
        .max_by_key(|c| c.path.len())
        .map(|c| (c, &args[c.path.len()..]))
}

/// The words and operands of one command line (PC-1).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Parsed {
    pub words: Vec<(&'static str, Vec<String>)>,
    pub operands: Vec<String>,
}

impl Parsed {
    /// Whether the word was given.
    pub fn has(&self, name: &str) -> bool {
        self.words.iter().any(|(n, _)| *n == name)
    }
    /// The first value of a word given once.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.words.iter().find(|(n, _)| *n == name).and_then(|(_, v)| v.first()).map(String::as_str)
    }
    /// The values of every use of a word, in order.
    pub fn all(&self, name: &str) -> Vec<&[String]> {
        self.words.iter().filter(|(n, _)| *n == name).map(|(_, v)| v.as_slice()).collect()
    }
    /// The value of a numeric setting word, or `default`.
    pub fn count(&self, name: &str, default: usize, what: &str) -> Result<usize, String> {
        match self.value(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("{name} {v}: N is {what}")),
        }
    }
}

/// PC-6: the usage error for an argument that starts with `-` and is not a
/// removed flag.
pub fn dash_error(arg: &str) -> String {
    format!("`{arg}`: no argument starts with `-` (PC-6); a file whose name starts with `-` is written ./{arg}")
}

/// Read the arguments after a command's path (PC-1): its words, in any
/// order among themselves, then its operands. A setting word takes the next
/// argument(s) as its values, whatever they spell.
pub fn parse(c: &Command, args: &[String], program: &str) -> Result<Parsed, String> {
    let name = c.path.join(" ");
    let mut p = Parsed::default();
    let mut i = 0;
    while i < args.len() {
        let Some(w) = c.word(&args[i]) else { break };
        if !w.repeat && p.has(w.name) {
            return Err(format!("{name}: the word `{}` is given twice; write: {}", w.name, c.synopsis(program)));
        }
        // PC-4: a word that changes another word's values comes before it
        for o in c.words {
            if o.after.is_some_and(|(f, _)| f == w.name) && p.has(o.name) {
                return Err(format!("{name}: `{}` comes before the first `{}` (PC-4)", w.name, o.name));
            }
        }
        let vals = match w.after {
            Some((f, then)) if p.has(f) => then,
            _ => w.values,
        };
        if !vals.is_empty() && i + vals.len() >= args.len() {
            return Err(format!("{name}: `{}` takes {}", w.name, vals.join(" ")));
        }
        let v: Vec<String> = args[i + 1..i + 1 + vals.len()].to_vec();
        if let Some(bad) = v.iter().find(|x| x.starts_with('-')) {
            return Err(dash_error(bad));
        }
        p.words.push((w.name, v));
        i += 1 + vals.len();
    }
    p.operands = args[i..].to_vec();
    if let Some(bad) = p.operands.iter().find(|x| x.starts_with('-')) {
        return Err(dash_error(bad));
    }
    if p.operands.len() < c.min || p.operands.len() > c.max {
        return Err(format!("usage: {}", c.synopsis(program)));
    }
    Ok(p)
}

/// What became of a removed flag (PC-5): a word of the same name, a word
/// of another name, an operand at a position, or nothing.
#[derive(Debug, Clone, Copy)]
pub enum Fate {
    /// The word with the flag's name without its dashes (PC-5).
    Word,
    /// Removed with nothing in its place (PC-3: `trace --json`).
    Drop,
    /// Its value is the operand at this position (PC-2: `--log D`).
    Operand(usize),
    /// A mode word, and its value the operand at this position
    /// (`trace --streams PATH` becomes `trace streams FILE PATH`).
    WordOperand(&'static str, usize),
    /// A command (`--help` becomes `help`, `--version` becomes `version`).
    Command(&'static str),
    /// A word of another name (`-o OUT` becomes `out OUT`).
    Named(&'static str),
}

/// A removed flag of one binary: its spelling, the command paths it was
/// given to (empty: any), how many values followed it, and its fate.
#[derive(Debug, Clone, Copy)]
pub struct Removed {
    pub flag: &'static str,
    pub paths: &'static [&'static str],
    pub values: usize,
    pub fate: Fate,
}

/// Decision 2 of section 5: when the command line has an argument that
/// starts with `-`, the one line that names its new spelling (for a
/// removed flag) or the PC-6 error. `None` when no argument starts with `-`.
/// The new spelling rewrites the whole command line: the path, then the
/// words in the order given, then the operands.
pub fn removed_flag(program: &str, table: &[Command], removed: &[Removed], args: &[String]) -> Option<String> {
    let first = args.iter().find(|a| a.starts_with('-'))?;
    let (path, rest): (Vec<&str>, &[String]) = match find(table, args) {
        Some((c, r)) => (c.path.to_vec(), r),
        None => (vec![], args),
    };
    let key = path.join(" ");
    let lookup = |f: &str| removed.iter().find(|r| r.flag == f && (r.paths.is_empty() || r.paths.contains(&key.as_str())));
    let flag_of = |a: &str| a.split_once('=').map_or(a, |(f, _)| f).to_string();
    let Some(r0) = lookup(&flag_of(first)) else {
        return Some(dash_error(first));
    };
    if let Fate::Command(c) = r0.fate {
        return Some(format!("{first} was removed; write: {program} {c}"));
    }
    let (mut words, mut ops, mut placed): (Vec<String>, Vec<String>, Vec<(usize, String)>) = (vec![], vec![], vec![]);
    let mut i = 0;
    while i < rest.len() {
        let a = &rest[i];
        if !a.starts_with('-') {
            ops.push(a.clone());
            i += 1;
            continue;
        }
        let f = flag_of(a);
        let Some(r) = lookup(&f) else {
            return Some(dash_error(a));
        };
        let mut vals: Vec<String> = match a.split_once('=') {
            Some((_, v)) => vec![v.to_string()],
            None => {
                let n = r.values.min(rest.len() - i - 1);
                let v = rest[i + 1..i + 1 + n].to_vec();
                i += n;
                v
            }
        };
        i += 1;
        match r.fate {
            Fate::Word => {
                // decision 4: two things joined with `=` are two arguments
                if let (1, "--key" | "--peer", Some((x, y))) = (vals.len(), f.as_str(), vals.first().and_then(|v| v.split_once('='))) {
                    vals = vec![x.to_string(), y.to_string()];
                }
                words.push(f.trim_start_matches('-').to_string());
                words.extend(vals);
            }
            Fate::Drop => {}
            Fate::Operand(k) => placed.extend(vals.into_iter().enumerate().map(|(j, v)| (k + j, v))),
            Fate::Named(w) => {
                words.push(w.to_string());
                words.extend(vals);
            }
            Fate::WordOperand(w, k) => {
                words.push(w.to_string());
                placed.extend(vals.into_iter().map(|v| (k, v)));
            }
            Fate::Command(c) => return Some(format!("{a} was removed; write: {program} {c}")),
        }
    }
    for (k, v) in placed {
        let k = k.min(ops.len());
        ops.insert(k, v);
    }
    let mut line = vec![program.to_string()];
    line.extend(path.iter().map(|s| s.to_string()));
    line.extend(words);
    line.extend(ops.iter().map(|o| shell_word(o)));
    Some(format!("{} was removed; write: {}", flag_of(first), line.join(" ")))
}

/// Read a whole command line against a binary's table: a removed flag or
/// an argument that starts with `-` gives its one line (decision 2, PC-6),
/// a command line that names no command gives the usage text, and
/// otherwise the command and its words and operands. Every `Err` is a
/// usage error, exit status 2.
pub fn read<'t>(program: &str, table: &'t [Command], removed: &[Removed], args: &[String]) -> Result<(&'t Command, Parsed), String> {
    if let Some(line) = removed_flag(program, table, removed, args) {
        return Err(line);
    }
    let (c, r) = find(table, args).ok_or_else(|| usage(program, table))?;
    Ok((c, parse(c, r, program)?))
}

/// `read`, for a binary's `main`: on a usage error, print it to stderr and
/// exit with status 2.
pub fn read_or_exit<'t>(program: &str, table: &'t [Command], removed: &[Removed], args: &[String]) -> (&'t Command, Parsed) {
    read(program, table, removed, args).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    })
}

/// An argument as a shell reader types it: quoted when it holds a space or
/// a character the shell reads.
pub fn shell_word(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:,=@+%".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const T: &[Command] = &[
        Command { path: &["dim"], words: &[Word::mode("json"), Word::set("fuel", &["N"])], operands: "FILE", min: 1, max: 1, help: "x" },
        Command {
            path: &["host", "serve"],
            words: &[Word::mode("tls"), Word::many("peer", &["HOST:PORT"]).then("tls", &["HOST:PORT", "KEY"])],
            operands: "FILE D",
            min: 2,
            max: 2,
            help: "y",
        },
    ];
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }
    #[test]
    fn words_then_operands() {
        let v = s(&["dim", "fuel", "9", "json", "fuel.rce"]);
        let (c, r) = find(T, &v).unwrap();
        let p = parse(c, r, "rce").unwrap();
        assert_eq!(p.value("fuel"), Some("9"));
        assert!(p.has("json"));
        assert_eq!(p.operands, s(&["fuel.rce"]));
        // PC-1: once operands begin, a word is an operand
        assert!(parse(c, &s(&["x.rce", "json"]), "rce").is_err());
    }
    #[test]
    fn tls_then_peer() {
        let v = s(&["host", "serve", "tls", "peer", "a:1", "ab", "peer", "b:2", "cd", "f.rce", "d"]);
        let (c, r) = find(T, &v).unwrap();
        let p = parse(c, r, "rce").unwrap();
        assert_eq!(p.all("peer").len(), 2);
        assert_eq!(p.operands, s(&["f.rce", "d"]));
        let v = s(&["host", "serve", "peer", "a:1", "tls", "f.rce", "d"]);
        let (c, r) = find(T, &v).unwrap();
        assert!(parse(c, r, "rce").unwrap_err().contains("PC-4"));
    }
    #[test]
    fn removed_names_new_spelling() {
        let rm = &[Removed { flag: "--fuel", paths: &[], values: 1, fate: Fate::Word }];
        let m = removed_flag("rce", T, rm, &s(&["dim", "--fuel", "9", "x.rce"])).unwrap();
        assert_eq!(m, "--fuel was removed; write: rce dim fuel 9 x.rce");
        let m = removed_flag("rce", T, rm, &s(&["dim", "--fuel=9", "x.rce"])).unwrap();
        assert_eq!(m, "--fuel was removed; write: rce dim fuel 9 x.rce");
        assert!(removed_flag("rce", T, rm, &s(&["dim", "-q", "x.rce"])).unwrap().contains("PC-6"));
        assert!(removed_flag("rce", T, rm, &s(&["dim", "x.rce"])).is_none());
    }
}
