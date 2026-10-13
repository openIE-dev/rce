# Getting started

## Install `rce`

`rce` is built from source with Rust. The repository pins the Rust version
in `rust-toolchain.toml`, and `rustup` installs that version for you.

```sh
git clone https://github.com/openIE-dev/rce
cd rce/spec/checker
cargo build --release
```

The program is now at `target/release/rce`. Put it on your `PATH`, for
example:

```sh
install -m 755 target/release/rce ~/.local/bin/
```

Check that it runs. With no arguments it prints the list of commands:

```sh
rce
```

## Hello, world

Make a file called `hello.rce`:

```rce
{{#include ../listings/ch01/hello.rce}}
```

`fn main()` declares a function called `main` with no parameters. The body
is the indented line under it. RCE uses indentation for blocks, as Python
does. Indent with spaces; a tab is refused.

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
- Blocks are marked by indentation with spaces.
