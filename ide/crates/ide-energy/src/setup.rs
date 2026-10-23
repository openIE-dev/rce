//! Permission setup for energy sources that need it, applied only with the
//! user's explicit consent for each step. Nothing is changed silently: every
//! step is shown in full (files with their exact content, commands with
//! their arguments) and runs only after the user types `yes` for it. Root
//! commands go through `sudo`, which asks the user for their password.
use crate::discover::{Probe, Status};

/// A file a step writes (as root, mode `mode`).
#[derive(Debug, Clone, PartialEq)]
pub struct FileWrite {
    pub path: String,
    pub content: String,
    pub mode: &'static str,
}

/// One setup step.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub tag: &'static str,
    pub title: String,
    /// What it changes and its security cost, shown before asking.
    pub why: String,
    pub file: Option<FileWrite>,
    /// Commands run after the file is in place, each run via sudo.
    pub commands: Vec<Vec<String>>,
}

fn cmd(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

/// The udev rule giving group `power` read access to RAPL energy counters.
pub const RAPL_RULE: &str = "# rce-ide telemetry setup: let group `power` read RAPL energy counters.\n\
ACTION==\"add\", SUBSYSTEM==\"powercap\", KERNEL==\"intel-rapl:*\", RUN+=\"/bin/chgrp power /sys%p/energy_uj\", RUN+=\"/bin/chmod g+r /sys%p/energy_uj\"\n";

/// The steps the probes call for, for user `user`.
pub fn plan(probes: &[Probe], user: &str) -> Vec<Step> {
    let mut out = vec![];
    let needs = |tag: &str| {
        probes
            .iter()
            .any(|p| p.tag == tag && matches!(p.status, Status::NeedsSetup(_)))
    };
    if needs("rapl") {
        out.push(Step {
            tag: "rapl",
            title: "RAPL: udev rule for group `power`".into(),
            why: format!(
                "Linux made energy_uj root-only (5.10, CVE-2020-8694, \"PLATYPUS\": fine-grained \
                 energy readings can leak secrets through power side channels). This adds group \
                 `power`, puts {user} in it, and makes energy_uj group-readable on every boot. \
                 Anyone in `power` can then read the counters. Log out and in for the group to apply."
            ),
            file: Some(FileWrite {
                path: "/etc/udev/rules.d/60-rce-powercap.rules".into(),
                content: RAPL_RULE.into(),
                mode: "0644",
            }),
            commands: vec![
                cmd(&["groupadd", "-f", "power"]),
                cmd(&["usermod", "-aG", "power", user]),
                cmd(&["udevadm", "control", "--reload"]),
                cmd(&["udevadm", "trigger", "--subsystem-match=powercap", "--action=add"]),
            ],
        });
    }
    if needs("powermetrics") {
        out.push(Step {
            tag: "powermetrics",
            title: "powermetrics: passwordless sudo for /usr/bin/powermetrics only".into(),
            why: format!(
                "powermetrics needs root. This lets {user} run exactly /usr/bin/powermetrics with \
                 sudo and no password (any arguments, including writing its output to a file). \
                 Not needed for IOReport or SMC readings, which work without root."
            ),
            file: Some(FileWrite {
                path: crate::powermetrics::SUDOERS.into(),
                content: format!(
                    "{user} ALL=(root) NOPASSWD: {}\n",
                    crate::powermetrics::PATH
                ),
                mode: "0440",
            }),
            commands: vec![cmd(&["visudo", "-cf", "/etc/sudoers.d/rce-powermetrics"])],
        });
    }
    out
}

/// How steps are shown, consented to and run (a terminal, or a test).
pub trait Io {
    fn show(&mut self, text: &str);
    /// True only for an explicit "yes".
    fn consent(&mut self, question: &str) -> bool;
    /// Run one command (already prefixed with sudo); Err on failure.
    fn run(&mut self, argv: &[String]) -> Result<(), String>;
}

/// Outcome of one step.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Applied,
    Declined,
    Failed(String),
}

/// Show each step, ask, and run it only on consent. A failed command stops
/// that step; later steps are still asked.
pub fn apply(steps: &[Step], io: &mut dyn Io) -> Vec<(String, Outcome)> {
    let mut res = vec![];
    for s in steps {
        let mut t = format!("== {}\n{}\n", s.title, s.why);
        if let Some(f) = &s.file {
            t += &format!(
                "\nwrites {} (mode {}, owner root):\n{}",
                f.path, f.mode, f.content
            );
        }
        for c in &s.commands {
            t += &format!("\nruns: sudo {}", c.join(" "));
        }
        io.show(&t);
        if !io.consent(&format!(
            "Apply \"{}\"? Type yes to apply, anything else to skip: ",
            s.title
        )) {
            res.push((s.tag.to_string(), Outcome::Declined));
            continue;
        }
        let mut argvs: Vec<Vec<String>> = vec![];
        let tmp = std::env::temp_dir().join(format!("rce-telemetry-{}.tmp", s.tag));
        if let Some(f) = &s.file {
            if let Err(e) = std::fs::write(&tmp, &f.content) {
                res.push((s.tag.to_string(), Outcome::Failed(e.to_string())));
                continue;
            }
            argvs.push(cmd(&[
                "sudo",
                "install",
                "-o",
                "root",
                "-m",
                f.mode,
                &tmp.to_string_lossy(),
                &f.path,
            ]));
        }
        for c in &s.commands {
            let mut a = vec!["sudo".to_string()];
            a.extend(c.iter().cloned());
            argvs.push(a);
        }
        let mut out = Outcome::Applied;
        for a in &argvs {
            if let Err(e) = io.run(a) {
                out = Outcome::Failed(format!("{}: {e}", a.join(" ")));
                break;
            }
        }
        let _ = std::fs::remove_file(&tmp);
        res.push((s.tag.to_string(), out));
    }
    res
}

/// A terminal: prints to stdout, reads consent from stdin, runs commands
/// with inherited stdio so sudo can ask for the password. Refuses to ask
/// when stdin is not a terminal.
pub struct Terminal;

impl Io for Terminal {
    fn show(&mut self, text: &str) {
        println!("{text}\n");
    }
    fn consent(&mut self, q: &str) -> bool {
        use std::io::{IsTerminal, Write};
        if !std::io::stdin().is_terminal() {
            println!("{q}\nstdin is not a terminal: skipped (consent must be typed)");
            return false;
        }
        print!("{q}");
        let _ = std::io::stdout().flush();
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).is_ok() && s.trim() == "yes"
    }
    fn run(&mut self, argv: &[String]) -> Result<(), String> {
        let st = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .status()
            .map_err(|e| e.to_string())?;
        if st.success() {
            Ok(())
        } else {
            Err(st.to_string())
        }
    }
}
