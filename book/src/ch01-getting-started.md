# Getting started

To try RCE without installing anything, open the
[RCE playground](https://openie-dev.github.io/rce/): it runs the reference
checker in your browser and shows whether a program is accepted and the
trace of each `example`. To run programs with `rce run`, install `rce`.

## Install `rce`

Every commit of `main` that passes the repository's continuous integration
(CI) is released as `rce-<commit>`, and the release `rce-latest` points at
the newest. A release has no typed version number: it is named by its
commit. The install script downloads the newest release for Linux (x86_64
or aarch64), macOS (Apple silicon) or Windows (Git Bash), checks the archive
against the release's `SHA256SUMS` file, and puts `rce` in `~/.local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/openIE-dev/rce/main/scripts/install.sh | sh
```

`sh install.sh COMMIT`, with the first 12 hexadecimal digits of a commit,
installs that commit's release instead (the 30 newest commit releases are
kept), and the environment variable
`RCE_INSTALL_DIR` chooses another directory. The script refuses to install
an archive whose digest does not match. An archive holds `rce` and its
licence files, nothing else.

To build `rce` from source instead, you need Rust. The repository pins the
Rust version in `rust-toolchain.toml`, and `rustup` installs that version
for you:

```sh
git clone https://github.com/openIE-dev/rce
cd rce/spec/checker
cargo build --release --bin rce
install -m 755 target/release/rce ~/.local/bin/
```

A public container image holds the same binary: `ghcr.io/openie-dev/rce`,
tagged `latest` and with each full commit hash. Chapter 16 uses it to run a program
as a service.

Check that `rce` runs. `rce version` prints the commit it was built
from, the specification draft it implements and the release label, and
with no arguments `rce` prints the list of commands:

```sh
rce version
rce
```

## Hello, world

Make a file called `hello.rce`:

```rce
{{#include ../listings/ch01/hello.rce}}
```

`fn main()` declares a function called `main` with no parameters. The body
is the indented line under it. RCE uses indentation for blocks, as Python
does. Indent with spaces; a tab requires change.

`emit` produces an output value. A program does not print directly. It
emits values, and the tool running it decides how to show them.

Run it:

```text
$ rce run hello.rce
{{#include ../listings/ch01/hello.out}}
```

`rce run` evaluated `main` and showed each emitted value.

## Checking without running

```text
$ rce check hello.rce
{{#include ../listings/ch01/hello_check.out}}
```

`rce check` reads the file, checks types and the
other rules, and evaluates every test the file contains. It never runs
`main`. You will use `rce check` and `rce test` far more often than
`rce run`.

## Comments

A comment starts with `#` and runs to the end of the line.

```text
# This line is a comment.
fn main()
    emit 1   # so is the end of this one
```

## What you learned

- `rce run FILE` runs `fn main()` and shows what it emits.
- `rce check FILE` checks a program without running it.
- `scripts/install.sh` installs a released `rce`; `rce version` names the
  commit it was built from.
- Blocks are marked by indentation with spaces.
