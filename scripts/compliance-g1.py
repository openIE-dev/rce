#!/usr/bin/env python3
"""Group 1 of the compliance migration (spec/proposals/compliance.md, 11.1 step 2).

Group 1 carries the outcome words: COMP-1 to COMP-3, COMP-6, COMP-19 and
COMP-20, and the vocabulary of 6.1, 6.2 and 6.4 to 6.9 that is wording or
naming only. This script applies its rename map to the repository:
identifiers, messages, syntax, data keys, file and directory names. It
leaves the behaviour changes of 8.1 and 8.2 (groups 2 and 3), the named
forms, lifts and reserved constructs (group 4), the host names of pull
request #31, and the prose of the chapters and READMEs, which is reworded by
hand in a separate commit.

usage:
  scripts/compliance-g1.py                 dry run: what would change, per area, and what remains
  scripts/compliance-g1.py --apply         rewrite the tree, move paths with `git mv`, regenerate
  options:
    --no-regen          with --apply, do not run the generators
    --conformance       with --apply, also rerun the conformance suite on the reference adapter
    --repin OLD_RCE     with --apply, write spec/proposals/compliance-repin.tsv (COMP-20 (b)) from
                        the program hashes OLD_RCE gives the earlier texts and the new rce the new ones;
                        `--repin build` first builds OLD_RCE from the tree before the change
    --residual FILE     write every hit that remains after the rewrite, with its class, to FILE
    --rules             print the matches of every rule
    --verbose           print each changed file

Properties:
  * Token-aware. A rule names the segments it may change (code, string,
    comment, or any) in the languages the script lexes (Rust, Python,
    JavaScript, Scala, C, Lean, RCE, shell); an identifier rule matches whole
    identifiers only, so a word inside an unrelated identifier is never
    touched. No rule edits a line that scripts/vocab-allow.tsv allows
    (third-party text, vectors, dated history), a generated file, or a hash.
  * Re-runnable. A second run changes nothing (the dry run checks this on
    every file it would change), and on a newer main the same map applies:
    a rule that finds nothing is reported, never an error.
  * Paths move with `git mv`, before the contents are rewritten; rules name
    the new paths.
"""
import argparse
import bisect
import difflib
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import vocab  # noqa: E402  (scripts/vocab.py: the pattern, the allow-list and the areas)

WORD_RX = re.compile(vocab.WORDS, re.I)
# Text that a rule rewrites although it holds none of the words; a file that
# holds neither the words nor one of these is not read further.
NEEDLES = ("accept", "rce-trace/2", "rce-dim/1", "rce-corpus/1", "seqRef", "pRef", "externals:", "enum { NL")

# --------------------------------------------------------------------------
# Lexing: which part of a file is code, a string or a comment.

LANG_BY_EXT = {
    ".rs": "rust", ".py": "python", ".js": "js", ".mjs": "js", ".ts": "js", ".scala": "scala",
    ".c": "c", ".h": "c", ".lean": "lean", ".rce": "rce", ".sh": "sh", ".scm": "scheme",
}


def lang_of(path):
    base = os.path.basename(path)
    if base in ("pre-push", "pre-commit"):
        return "sh"
    return LANG_BY_EXT.get(os.path.splitext(path)[1])


def _ident(c):
    return c.isalnum() or c == "_"


def scan(text, lang):
    """Sorted list of (start, kind) runs; kind is 'code', 'str' or 'com'."""
    runs = []
    n, i, kind_start, code_from = len(text), 0, 0, 0

    def emit(a, b, kind):
        nonlocal code_from
        if code_from < a:
            runs.append((code_from, "code"))
        runs.append((a, kind))
        code_from = b

    line_c = {"rust": ["//"], "js": ["//"], "scala": ["//"], "c": ["//"], "lean": ["--"],
              "python": ["#"], "rce": ["#"], "sh": ["#"], "scheme": [";"]}[lang]
    block_c = {"rust": ("/*", "*/", True), "js": ("/*", "*/", False), "scala": ("/*", "*/", True),
               "c": ("/*", "*/", False), "lean": ("/-", "-/", True)}.get(lang)
    while i < n:
        c = text[i]
        # line comment
        lc = next((x for x in line_c if text.startswith(x, i)), None)
        if lc and (lang != "sh" or i == 0 or text[i - 1] in " \t\n;"):
            j = text.find("\n", i)
            j = n if j < 0 else j
            emit(i, j, "com")
            i = j
            continue
        if block_c and text.startswith(block_c[0], i):
            depth, j = 1, i + len(block_c[0])
            while j < n and depth:
                if block_c[2] and text.startswith(block_c[0], j):
                    depth += 1
                    j += len(block_c[0])
                elif text.startswith(block_c[1], j):
                    depth -= 1
                    j += len(block_c[1])
                else:
                    j += 1
            emit(i, j, "com")
            i = j
            continue
        # Rust raw strings r"..", r#".."#, br".."
        if lang == "rust" and c in "rb" and (i == 0 or not _ident(text[i - 1])):
            m = re.match(r'b?r(#*)"', text[i:i + 70])
            if m:
                close = '"' + m.group(1)
                j = text.find(close, i + m.end())
                j = n if j < 0 else j + len(close)
                emit(i, j, "str")
                i = j
                continue
        if lang in ("python", "scala") and text.startswith(('"""', "'''"), i) and (lang == "python" or text.startswith('"""', i)):
            q = text[i:i + 3]
            j = text.find(q, i + 3)
            j = n if j < 0 else j + 3
            emit(i, j, "str")
            i = j
            continue
        quotes = {"rust": '"', "js": "\"'`", "scala": '"', "c": "\"'", "lean": '"', "python": "\"'",
                  "rce": '"', "sh": "\"'", "scheme": '"'}[lang]
        if c in quotes:
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\" and not (lang == "sh" and c == "'"):
                    j += 1
                elif text[j] == "\n" and c != "`" and lang not in ("sh", "python"):
                    break
                j += 1
            j = min(n, j + 1)
            emit(i, j, "str")
            i = j
            continue
        if c == "'" and lang in ("rust", "scala", "lean"):
            # a char literal, else a lifetime or a prime
            m = re.match(r"'(\\.[^']*|[^'\\\n])'", text[i:i + 12])
            if m:
                emit(i, i + m.end(), "str")
                i += m.end()
                continue
        i += 1
    if code_from < n:
        runs.append((code_from, "code"))
    return runs


class Segments:
    def __init__(self, text, lang):
        self.runs = scan(text, lang) if lang else [(0, "text")]
        self.starts = [r[0] for r in self.runs]

    def kind(self, pos):
        k = bisect.bisect_right(self.starts, pos) - 1
        return self.runs[max(k, 0)][1]


# --------------------------------------------------------------------------
# Rules.

def git(*a, check=True, cwd=ROOT):
    p = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"git {' '.join(a)}: {p.stderr}")
    return p.stdout


def tracked():
    return [f for f in git("ls-files", "-z").split("\0") if f]



class Rule:
    """A rename. kind: 'tok' (whole identifier), 'lit' (exact text) or 're'.
    where: 'any', or a '+'-joined set of 'code', 'str', 'com'. In a file the
    script does not lex every position is 'text', which only 'any' matches."""

    def __init__(self, kind, old, new, files, where="any", skip=(), note=""):
        self.kind, self.old, self.new, self.note = kind, old, new, note
        self.files = [files] if isinstance(files, str) else list(files)
        self.skip = [skip] if isinstance(skip, str) else list(skip)
        self.where = where
        if kind == "tok":
            self.rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])")
        elif kind == "lit":
            self.rx = re.compile(re.escape(old))
        else:
            self.rx = re.compile(old, re.M)
        self.fglobs = [vocab.glob_re(g) for g in self.files]
        self.sglobs = [vocab.glob_re(g) for g in self.skip]
        self.hits = 0
        self.files_hit = set()
        self.generic = False

    def replacement(self, m):
        if callable(self.new):
            return self.new(m)
        return m.expand(self.new) if self.kind == "re" else self.new

    def applies_to(self, path):
        return any(g.match(path) for g in self.fglobs) and not any(g.match(path) for g in self.sglobs)

    def label(self):
        new = getattr(self.new, "__name__", None) or repr(self.new)
        files = ",".join(self.files)
        return f"{self.kind} {self.old!r} -> {new} [{files if len(files) < 90 else files[:87] + '...'}]"


def keyword_table(m):
    """2.4's table of keywords, laid out again without `refused` (7 columns of 10)."""
    words = [w for w in m.group(0).split() if w != "refused"]
    rows = [words[i:i + 7] for i in range(0, len(words), 7)]
    return "".join("".join(w.ljust(10) for w in r).rstrip() + "\n" for r in rows)


def cli_report(m):
    """`REFUSED FILE: …` in `cli.rs` reads `fit FILE: …` where it reports a text
    against the rules (6.4). The verdicts on records (`manifest-check`, the
    layout check of `fmt`, a Form IR read by `unform`) are group 3 (COMP-15)."""
    before = m.string[:m.start()]
    arm = max(before.rfind('Some("'), before.rfind("\nfn "), before.rfind("\npub fn "))
    head = before[arm:arm + 60].lstrip("\n")
    if head.startswith(('Some("unform")', 'Some("manifest-check")', 'Some("fmt") if !rest')):
        return m.group(0)
    return m.group(1) + "fit " + m.group(2) + ": "


def phrases(table, files, where="com", **k):
    """One rule per (pattern, replacement) of TABLE, applied in order."""
    out = [X(r"(?<!\{)" + a + r"(?![}\w])", b, files, where, **k) for a, b in table]
    for r in out:
        r.generic = True
    return out


# The words in the sense of a static report (6.4): a text is reported, with its rule.
STATIC = [
    (r"\bRefused:", "Reported:"), (r"\bis still refused\b", "is still reported"), (r"\bmust be refused\b", "must be reported"),
    (r"\bmust refuse\b", "must report"), (r"\bare refused\b", "are reported"), (r"\bis refused\b", "is reported"),
    (r"\bbe refused\b", "be reported"), (r"\brefuses with\b", "reports"), (r"\brefuse with\b", "report"),
    (r"\brefused with\b", "reported with"), (r"\brefuses\b", "reports"), (r"\brefuse\b", "report"),
    (r"\brefused\b", "reported"), (r"\brefusing\b", "reporting"), (r"\brefusals\b", "reports"),
    (r"\brefusal\b", "report"), (r"\bRefusals\b", "Reports"), (r"\bRefusal\b", "Report"),
    (r"\brejection rules\b", "rules of section 7"), (r"\brejects\b", "reports"), (r"\brejected\b", "reported"),
]
# The words in the sense of an input's outcome (6.1): the kept outcome.
DELTA = [
    (r"\bthe refusal's\b", "the kept outcome's"), (r"\bif refused\b", "if kept"),
    (r"\bthe record of a refusal\b", "the record of a kept outcome"), (r"\brefuses its event\b", "has the kept outcome"),
    (r"\bor its refusal\b", "or its kept reason"), (r"\bthe refusal of\b", "the kept outcome of"),
    (r"\ba refusal by\b", "a kept outcome by"), (r"\b([Aa]) refused event\b", r"\1 kept event"),
    (r"\brefused (events?)\b", r"kept \1"), (r"\brefuses the (whole )?event\b", r"gives the \1event the kept outcome"),
    (r"\brefuse the (whole )?event\b", r"give the \1event the kept outcome"),
    (r"\bis refused (as a whole|atomically)\b", r"has the kept outcome \1"), (r"\bthe first refusal\b", "the first kept outcome"),
    (r"\brefusals\b", "kept outcomes"), (r"\brefusal\b", "kept outcome"), (r"\bis refused\b", "has the kept outcome"),
    (r"\bare refused\b", "have the kept outcome"), (r"\bmay refuse\b", "may keep the state"), (r"\brefused\b", "kept"),
]


def T(old, new, files, where="code", **k):
    return Rule("tok", old, new, files, where, **k)


def L(old, new, files, where="any", **k):
    return Rule("lit", old, new, files, where, **k)


def X(pattern, repl, files, where="any", **k):
    return Rule("re", pattern, repl, files, where, **k)


# Generated files: regenerated by the repository's own tools, never edited by a rule.
GENERATED = [
    "spec/RCE-SPEC.md", "spec/book/INDEX.md", "spec/book/receipts.json", "spec/data/*.json",
    "tree-sitter/src/parser.c", "tree-sitter/src/grammar.json", "tree-sitter/src/node-types.json",
    "tree-sitter/src/tree_sitter/**", "tree-sitter/tree-sitter-rce.wasm", "spec/checker-wasm/pkg/**",
]
GENERATED_RX = [vocab.glob_re(g) for g in GENERATED]

# Prose a script cannot word well: reworded by hand, in a separate commit. Rules
# still apply there to syntax, paths, identifiers and titles.
HAND = [
    "spec/book/*.md", "book/src/**", "**/README.md", "**/CONTRIBUTING.md", "conformance/*.md",
    "spec/proposals/*.md", "docs/**/*.md", "ide/docs/**", "SURFACE.md", "spec/fuzz/RESULTS.md",
    "stdlib/vectors/SOURCE.md", "ide/release/**", "programs/**/README.md",
]
HAND_RX = [vocab.glob_re(g) for g in HAND]

# Files the regeneration step rewrites from the tools' own output (the run
# transcripts, the trace fixtures, the tree-sitter cross-check, the figures):
# the rules still apply to them, and what the rules leave is regenerated.
REGENERATED = ["book/listings/**/*.out", "programs/**/*.out", "spec/traces/**", "conformance/reports/tree-sitter.json",
               "docs/figures/*.svg"]
REGENERATED_RX = [vocab.glob_re(g) for g in REGENERATED]



# --------------------------------------------------------------------------
# The group 1 rename map. Section numbers are those of spec/proposals/compliance.md.

# Paths (6.5, 6.8, 11.2 "Paths"). The order matters only within one path.
MOVES = [
    (r"^book/listings/(ch\d\d)/refused/", r"book/listings/\1/fit/"),
    (r"^book/listings/ch05/door_refused\.", r"book/listings/ch05/door_kept."),
    (r"^conformance/tests/(.+)-reject\.rce$", r"conformance/tests/\1-fit.rce"),
    (r"^(conformance/tests/concurrency/targeted/lex-merge-bitwise)-refused\.rce$", r"\1-fit.rce"),
    (r"^(conformance/tests/forms-log/targeted/s1-guard-trap)-refuses\.rce$", r"\1-kept.rce"),
    (r"^(conformance/tests/handler-spec/cur-2-stale-strict)-refused\.rce$", r"\1-fit.rce"),
    (r"^(conformance/tests/tooling/emit-(?:js|python|rust|scala))-refused\.rce$", r"\1-fit.rce"),
    (r"^conformance/tests/tooling/run-refused-one-line\.rce$", r"conformance/tests/tooling/run-fit-one-line.rce"),
    (r"^(conformance/tests/types-checking/targeted/t35-refined-input)-refused\.rce$", r"\1-kept.rce"),
    (r"^programs/money/refused/", r"programs/money/fit/"),
    (r"^spec/checker/tests/utf8_refuses\.rs$", r"spec/checker/tests/utf8_kept.rs"),
    (r"^spec/(conc06b?)/reject/", r"spec/\1/fit/"),
    (r"^spec/examples-0\.9/refused/", r"spec/examples-0.9/fit/"),
    (r"^spec/examples/reject/", r"spec/examples/fit/syntax/"),
    (r"^spec/examples/inadmissible/unexpected_refusal\.rce$", r"spec/examples/inadmissible/unexpected_kept.rce"),
    (r"^spec/examples/inadmissible/wrong_refusal\.rce$", r"spec/examples/inadmissible/wrong_reason.rce"),
    (r"^(conformance/tests/evaluation-fuel/harvested/)unexpected-refusal-", r"\1unexpected-kept-"),
    (r"^(conformance/tests/evaluation-fuel/harvested/)wrong-refusal-", r"\1wrong-reason-"),
]

ALL = "**"
RS = "**/*.rs"
CHK = "spec/checker/**"
CONF = "conformance/**"
RT = "spec/checker/src/emit/rt.*"
LEAN = "spec/lean/**"
IDE = "ide/crates/**"

RULES = [
    # ---- 6.1: the outcome keyword of `example` (COMP-2), in every text that holds RCE source
    X(r"->(\s*)refused(\s*):", r"->\1kept\2:", ALL),
    L("refused: trap(", "kept: trap(", ALL),
    L("`refused: trap`", "`kept: trap`", ALL),
    L('Outcome     = Expr | "refused" ":"', 'Outcome     = Expr | "kept" ":"', "spec/book/03-syntax.md"),

    # ---- 6.1: intents that describe an RCE outcome (10: reworded with the module and program hashes)
    L('intent "a bank account that refuses overdrafts"',
      'intent "a bank account that keeps its balance on an overdraft (Insufficient)"', ALL),
    L("a counter that refuses to go negative", "a counter whose `Dec` at zero keeps n at 0", ALL),
    L("an event whose payload is outside its refined type is refused, state unchanged",
      "an event whose payload is outside its refined type has the kept outcome, state unchanged", ALL),
    L("anything else refuses the event, never substitutes",
      "anything else gives the event the kept outcome, never substitutes", ALL),
    L('intent "guard trap refuses (S1)"', 'intent "a guard trap gives the event the kept outcome (S1)"', ALL),
    L('intent "a proportional controller that refuses impossible readings"',
      'intent "a proportional controller that keeps its state on an impossible reading"', ALL),

    # ---- 6.1: the D outcome, the λ record and the trace format (COMP-1)
    L('format!("{{\\"refused\\":{}}}", jstr(&r)),', 'format!("{{\\"kept\\":{}}}", jstr(&r)),', "spec/checker/src/interp.rs"),
    L('                            "{{\\"Refused\\":{{\\"error\\":{},\\"payload\\":{}}}}}",\n'
      '                            jstr(&r),\n'
      '                            form_value(&ev, m).json()\n',
      '                            "{{\\"Kept\\":{{\\"payload\\":{},\\"reason\\":{}}}}}",\n'
      '                            form_value(&ev, m).json(),\n'
      '                            jstr(&r)\n', "spec/checker/src/interp.rs"),
    L("// T8: a refused event leaves the state unchanged.", "// T8: a kept event leaves the state unchanged.", "spec/checker/src/**"),
    T("refused", "kept", "spec/checker/src/interp.rs"),
    L('pub const TRACE_FORMAT: &str = "rce-trace/2";', 'pub const TRACE_FORMAT: &str = "rce-trace/3";', "spec/checker/src/cli.rs"),
    L("/// the first refused event.", "/// the first event with the kept outcome.", "spec/checker/src/cli.rs"),
    L('"format":"rce-trace/2"', '"format":"rce-trace/3"', ["spec/traces/**", "book/listings/**", "spec/checker/tests/**"]),
    L("rce-trace/2", "rce-trace/3", ["spec/traces/*.py", "spec/traces/*.md", "spec/checker/tests/**", "scripts/**"]),
    L('"outcome":{"refused":', '"outcome":{"kept":', ["spec/traces/**", "spec/checker/tests/**", "book/listings/**"]),
    X(r'\{"Refused":\{"error":("[^"]*"),"payload":(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*"|[0-9-]+)\}\}',
      r'{"Kept":{"payload":\2,"reason":\1}}', ["spec/traces/**", "book/listings/**", "spec/checker/tests/**"]),
    L('s[1]["outcome"]["refused"]', 's[1]["outcome"]["kept"]', "spec/checker/tests/utf8_kept.rs"),
    L('"{{\\"event\\":{},\\"refused\\":{}}}"', '"{{\\"event\\":{},\\"kept\\":{}}}"', "spec/checker-wasm/src/lib.rs"),
    L("s.refused ? ' class=\"refused\"' : \"\"}>${esc(s.refused ? \"refused: \" + s.refused",
      "s.kept ? ' class=\"kept\"' : \"\"}>${esc(s.kept ? \"kept: \" + s.kept", "playground/app.js"),
    L(".refused { color", ".kept { color", "playground/style.css"),
    L("`refused at the ${r.stage} stage`", "`fit reported at the ${r.stage} stage`", "playground/app.js"),

    # ---- 6.8: the example's outcome field, `RunOut.refused`, `Example.refused`
    T("refused", "kept", ["spec/checker/src/ast.rs", "spec/checker/src/check.rs", "spec/checker/src/unparse.rs",
                          "spec/checker/src/emit/probe.rs", "spec/checker/src/emit/*.rs", "spec/checker/src/lint*.rs",
                          "spec/checker/src/nameless.rs", "spec/checker/src/core_elab.rs"], where="code"),
    L('None => format!("refused: {r}"),', 'None => format!("kept: {r}"),', "spec/checker/src/unparse.rs"),

    # ---- COMP-2: the parser. `kept` is a contextual word; `refused` leaves the reserved list.
    L('"pub", "recv", "refused", "return",', '"pub", "recv", "return",', "spec/checker/src/parse.rs"),
    L('        let (mut refused, mut value) = (None, None);\n        if self.eat_kw("refused") {\n',
      '        let (mut kept, mut value) = (None, None);\n'
      '        // `kept` is a contextual word (2.4, COMP-2): the outcome keyword\n'
      '        // only when `:` follows it; anywhere else it is a name\n'
      '        if matches!(self.peek(), Tok::Id(w) if w == "kept") && matches!(self.peek2(), Tok::P(":")) {\n'
      '            self.bump();\n', "spec/checker/src/parse.rs"),
    L("                refused = Some(format!(\"trap({k})\"));", "                kept = Some(format!(\"trap({k})\"));", "spec/checker/src/parse.rs"),
    L("                refused = Some(n);", "                kept = Some(n);", "spec/checker/src/parse.rs"),
    L("            events,\n            refused,\n            value,", "            events,\n            kept,\n            value,", "spec/checker/src/parse.rs"),
    L('pub const CONTEXTUAL: &[&str] = &["where", "by"];', 'pub const CONTEXTUAL: &[&str] = &["where", "by", "kept"];',
      "spec/checker/src/lalr.rs"),
    L("/// `where` follows a type in a refined type declaration (5.1).",
      "/// `where` follows a type in a refined type declaration (5.1); `kept`\n/// begins the kept outcome of an `example` (3.2, COMP-2).", "spec/checker/src/lalr.rs"),
    # the keyword lists of the book, the editors and the tools
    X(r"(?ms)^actor     after .*?\byield\n", keyword_table, ["spec/book/02-lexical.md", "book/src/appendix-b-keywords.md"]),
    L('extern actor supervisor node const refused by as mut then', 'extern actor supervisor node const by as mut then', "scripts/corpus.py"),

    # ---- COMP-2: tree-sitter (the grammar, the scanner, the queries, the corpus)
    L("'recv', 'refused', 'return',", "'recv', 'return',", "tree-sitter/grammar.js"),
    L('"recv","refused","return",', '"recv","return",', "tree-sitter/src/scanner.c"),
    X(r"(?<!: NAME; return true;\n      \}\n)(      // contextual word \(5\.1 refined types\): `where`)",
      "      // contextual word (3.2, COMP-2): `kept` followed by `:` where an `example`\n"
      "      // outcome begins is the outcome keyword; anywhere else it is a NAME\n"
      "      if (valid[KEPT] && !valid[SENTINEL] && strcmp(w, \"kept\") == 0) {\n"
      "        lx->mark_end(lx);\n"
      "        while (lx->lookahead == ' ') lx->advance(lx, true);\n"
      "        s->started = 1; s->seen = 1;\n"
      "        lx->result_symbol = lx->lookahead == ':' ? KEPT : NAME; return true;\n"
      "      }\n\\1", "tree-sitter/src/scanner.c"),
    X(r"(enum \{ NL, INDENT, DEDENT, NAME, (?:(?!KEPT)[^}])*?) \};", r"\1, KEPT };", "tree-sitter/src/scanner.c"),
    X(r"(?s)(externals: \$ => \[(?:(?!\$\._kept).)*?)\](,\s*\n\s*extras:)", r"\1, $._kept]\2", "tree-sitter/grammar.js"),
    L("seq('refused', ':',", "seq(alias($._kept, 'kept'), ':',", "tree-sitter/grammar.js"),
    T("outcome_refused", "outcome_kept", ["tree-sitter/grammar.js", "tree-sitter/test/**", "tree-sitter/queries/**", "tree-sitter/bindings/**"], where="any"),
    L('"example" "refused" "check"', '"example" "kept" "check"', "tree-sitter/queries/highlights.scm"),

    # ---- 6.1: `rce run` and the emitted runtimes print `k EVENT kept: E STATE`
    L('lines.push(format!("{} {} refused: {r}", k + 1, ev.show()));',
      'lines.push(format!("{} {} kept: {r} {}", k + 1, ev.show(), show_state(&s)));', "spec/checker/src/cli.rs"),
    L('out.push(format!("{k} {} refused: {r}", ev.show()));',
      'out.push(format!("{k} {} kept: {r} {}", ev.show(), show_state(&s)));', "spec/checker/src/emit/mod.rs"),
    L('println!("{k} {} refused: {r}", show(&ev)); break; }',
      'println!("{k} {} kept: {r} {}", show(&ev), show_state(&s)); break; }', "spec/checker/src/emit/rt.rs"),
    L('print("%d %s refused: %s" % (k, show(ev), r.r)); return',
      'print("%d %s kept: %s %s" % (k, show(ev), r.r, show_state(s))); return', "spec/checker/src/emit/rt.py"),
    L('say(k + " " + show(ev) + " refused: " + e.r); return; }',
      'say(k + " " + show(ev) + " kept: " + e.r + " " + showState(s)); return; }', "spec/checker/src/emit/rt.js"),
    L('say(s"$k " + show(ev) + " refused: " + e.r); return }',
      'say(s"$k " + show(ev) + " kept: " + e.r + " " + show_state(s)); return }', "spec/checker/src/emit/rt.scala"),
    L("/// `rce run FILE EVENTS`: apply δ to the initial state once per event, in\n/// order, printing the state and outputs after each. Stops at the first\n/// refused event.",
      "/// `rce run FILE EVENTS`: apply δ to the initial state once per event, in\n/// order, printing the state and outputs after each. Stops at the first\n/// event with the kept outcome.", "spec/checker/src/cli.rs"),
    L("/// Outcome of `rce run`: the printed lines and whether the run was refused.",
      "/// Outcome of `rce run`: the printed lines and whether an event had the kept\n/// outcome or the run stopped.", "spec/checker/src/cli.rs"),
    T("refused", "kept", ["spec/checker/src/cli.rs"], where="code"),
    T("refused", "kept", ["spec/checker/src/bin/conform.rs", "spec/checker/src/bin/rce-c1.rs", "ide/crates/rce-guest/src/main.rs",
                          "spec/checker/tests/conc06_check.rs", "spec/checker/tests/cli.rs"], where="code"),
    # the host transcript of `rce host run` (an input's outcome)
    L('writeln!(out, "{} {} refused: {r}", x.seq, x.event)', 'writeln!(out, "{} {} kept: {r}", x.seq, x.event)',
      "spec/checker/src/cli.rs"),
    # readers of the run line: the reason is the first word, or `trap(…)`
    L('    let r = last.split("refused: ").nth(1).unwrap_or(&last).trim().to_string();\n    trap_or_refused(&r)\n',
      '    let r = match last.split_once(" kept: ") {\n'
      '        Some((_, rest)) => kept_reason(rest),\n'
      '        None => last.strip_prefix("stopped: ").unwrap_or(&last).trim().to_string(),\n'
      '    };\n'
      '    trap_or(&r, "kept")\n'
      '}\n\n'
      '/// The reason of a `k EVENT kept: REASON STATE` line (10.3): `trap(…)` or a\n'
      '/// declared error name; the state follows it.\n'
      'fn kept_reason(rest: &str) -> String {\n'
      '    if rest.starts_with("trap(") {\n'
      '        return rest.find(") ").map_or(rest.trim(), |i| &rest[..=i]).to_string();\n'
      '    }\n'
      '    rest.split(\' \').next().unwrap_or(rest).to_string()\n',
      "spec/checker/src/bin/conform.rs"),
    L("/// words it (`trap: K` from fn main, `trap(K)` from δ), else `refused:E`.\nfn trap_or_refused(r: &str) -> String {",
      "/// words it (`trap: K` from fn main, `trap(K)` from δ), else `WORD:E` (`kept:E`\n/// for an input, `stopped:E` for a run of `fn main`).\nfn trap_or(r: &str, word: &str) -> String {",
      "spec/checker/src/bin/conform.rs"),
    L('    format!("refused:{r}")\n}', '    format!("{word}:{r}")\n}', "spec/checker/src/bin/conform.rs"),
    L("conc::Outcome::Refused(e) => trap_or_refused(&e),", 'conc::Outcome::Stopped(e) => trap_or(&e, "stopped"),',
      "spec/checker/src/bin/conform.rs"),
    L('json!({"refused": e})', 'json!({"kept": e})', "spec/checker/src/bin/conform.rs"),
    L('if let Some(rf) = st.get("refused") {', 'if let Some(rf) = st.get("kept") {', "spec/checker/src/bin/conform_lean.rs"),
    L('format!("trap:{t}") } else { format!("refused:{why}") };', 'format!("trap:{t}") } else { format!("kept:{why}") };',
      "spec/checker/src/bin/conform_lean.rs"),
    L('("refused", trapJ t)', '("kept", trapJ t)', "spec/lean/CoreMain.lean"),
    L('Json.mkObj [("refused", Json.mkObj', 'Json.mkObj [("kept", Json.mkObj', "spec/lean/CoreMain.lean"),
    L('    """One spelling for an outcome (see ADAPTERS.md): trap:KIND or refused:E."""',
      '    """One spelling for an outcome (see ADAPTERS.md): trap:KIND, kept:E for an input, stopped:E for a run."""',
      "conformance/adapters/emit_adapter.py"),
    L("def outcome_of(r):", 'def outcome_of(r, word="kept"):', "conformance/adapters/emit_adapter.py"),
    L('    return "refused:" + r', '    return word + ":" + r', "conformance/adapters/emit_adapter.py"),
    L('        elif o.startswith("Refused: "):\n            outcome = outcome_of(o[9:])',
      '        elif o.startswith("Stopped: "):\n            outcome = outcome_of(o[9:], "stopped")', "conformance/adapters/emit_adapter.py"),
    L('m = re.search(r"refused[:(]\\s*([A-Za-z_][\\w]*)", err or "")', 'm = re.search(r"kept[:(]\\s*([A-Za-z_][\\w]*)", err or "")',
      "conformance/adapters/emit_adapter.py"),
    L('trap = re.search(r"refused: trap\\(([^)]*)\\)", err or "")', 'trap = re.search(r"kept: trap\\(([^)]*)\\)", err or "")',
      "conformance/adapters/emit_adapter.py"),
    L('    if lines and " refused: " in lines[-1]:\n        outcome = outcome_of(lines.pop().split(" refused: ", 1)[1])',
      '    if lines and " kept: " in lines[-1]:\n        rest = lines.pop().split(" kept: ", 1)[1]\n'
      '        outcome = outcome_of(rest[:rest.find(") ") + 1] if rest.startswith("trap(") and ") " in rest else rest.split(" ", 1)[0])',
      "conformance/adapters/emit_adapter.py"),
    L("when the last event is refused) so the module has an", "when the last event has the kept outcome) so the module has an",
      "conformance/adapters/emit_adapter.py"),

    # ---- 6.2: run outcomes (COMP-6)
    T("Outcome::Refused", "Outcome::Stopped", [RS], where="code"),
    L("pub enum Outcome {\n    Done,\n    Refused(String),\n    Deadlock,\n}", "pub enum Outcome {\n    Done,\n    Stopped(String),\n    Deadlock,\n}",
      "spec/checker/src/conc.rs"),
    L("pub enum Outcome { Done, Refused(String), Deadlock }", "pub enum Outcome { Done, Stopped(String), Deadlock }", "spec/checker/src/emit/rt.rs"),
    L('lines.push(format!("refused: {e}"));', 'lines.push(format!("stopped: {e}"));', "spec/checker/src/cli.rs"),
    L('crate::conc::Outcome::Stopped(e) => format!("{{\\"refused\\":{}}}",', 'crate::conc::Outcome::Stopped(e) => format!("{{\\"stopped\\":{}}}",',
      "spec/checker/src/cli.rs"),
    L('crate::conc::Outcome::Stopped(r) => format!("refused: {r}"),', 'crate::conc::Outcome::Stopped(r) => format!("stopped: {r}"),',
      "spec/checker/src/host.rs"),
    L("outcome Refused: ", "outcome Stopped: ", [CHK, CONF, "book/**", "programs/**"]),
    # the emitted runtimes' stop tuple and kept-outcome class (6.2, 6.8)
    L('["refused", ', '["stopped", ', "spec/checker/src/emit/rt.js"),
    L('("refused", ', '("stopped", ', "spec/checker/src/emit/rt.py"),
    L('V("refused", ', 'V("stopped", ', "spec/checker/src/emit/rt.scala"),
    L("class Refuse extends Error { constructor(r) { super(\"refuse\"); this.r = r; } }",
      "class Kept extends Error { constructor(r) { super(\"kept\"); this.r = r; } }", "spec/checker/src/emit/rt.js"),
    T("Refuse", "Kept", ["spec/checker/src/emit/rt.js", "spec/checker/src/emit/rt.py", "spec/checker/src/emit/rt.scala"], where="code"),
    T("refusal", "kept_reason", ["spec/checker/src/emit/rt.js", "spec/checker/src/emit/rt.py", "spec/checker/src/emit/rt.scala",
                                 "spec/checker/src/emit/rt.rs"], where="code"),
    T("actor_refusal", "actor_kept_reason", ["spec/checker/src/emit/rt.py", "spec/checker/src/emit/rt.scala", "spec/checker/src/emit/rt.rs"],
      where="code"),
    T("actorRefusal", "actorKeptReason", "spec/checker/src/emit/rt.js", where="code"),
    T("refused", "kept", ["spec/checker/src/emit/rt.js", "spec/checker/src/emit/rt.py", "spec/checker/src/emit/rt.scala",
                          "spec/checker/src/emit/rt.rs"], where="code"),
    L("// machine step refusal (interp.rs)", "// the reason of a step's kept outcome (interp.rs)", RT),
    L('"""machine step refusal (interp.rs)"""', '"""the reason of a step\'s kept outcome (interp.rs)"""', RT),
    L("// actor handler refusal (conc.rs)", "// the reason of an actor handler's kept outcome (conc.rs)", RT),
    L('"""actor handler refusal (conc.rs)"""', '"""the reason of an actor handler\'s kept outcome (conc.rs)"""', RT),
    L("Rust Debug of the interpreter's V (refusal text)", "Rust Debug of the interpreter's V (the text of a kept reason)", RT),
    L("so refusal reasons match its", "so kept reasons match its", RT),
    # the actor log entry of a handler's kept outcome (C.13)
    L('"Refused({', '"Kept({', ["spec/checker/src/conc.rs", "spec/checker/src/emit/rt.rs"]),
    L('"Refused(" + ', '"Kept(" + ', ["spec/checker/src/emit/rt.js", "spec/checker/src/emit/rt.scala"]),
    L('"Refused(%s)" % ', '"Kept(%s)" % ', "spec/checker/src/emit/rt.py"),
    L("/// actor log entries the interpreter also logs: Refused, Stopped, Restart, Input, Unhandled",
      "/// actor log entries the interpreter also logs: Kept, Stopped, Restart, Input, Unhandled", RT),
    L('["Refused(", "Stopped(", "Restart(", "Input(", "Unhandled("]', '["Kept(", "Stopped(", "Restart(", "Input(", "Unhandled("]',
      "spec/checker/tests/emit_run.rs"),
    L('starts_with("actorlog Refused")', 'starts_with("actorlog Kept")', "spec/checker/tests/emit_run.rs"),
    T("refused", "kept", "spec/checker/src/conc.rs", where="code"),
    L(r'''.strip_prefix("outcome Stopped: ").map(|r| format!("refused: {r}"))''',
      r'''.strip_prefix("outcome Stopped: ").map(|r| format!("stopped: {r}"))''', "spec/checker/tests/write_back.rs"),

    # ---- 6.4: the static report word and the exit-status constant (6.8)
    T("REFUSED", "OUTSTANDING", ["spec/checker/src/cli.rs", "spec/checker/src/main.rs", "spec/checker/src/bin/*.rs"], where="code"),
    X(r'(writeln!\(out, ")REFUSED (\{p\}|\{\}|\{file\}): ', cli_report, "spec/checker/src/cli.rs"),
    L('println!("REFUSED {p}: {e}")', 'println!("fit {p}: {e}")', "spec/checker/src/main.rs"),
    L('println!("REFUSED {p}: {}", c.errs.join("; "));', 'println!("fit {p}: {}", c.errs.join("; "));', "spec/checker/src/main.rs"),
    X(r"^REFUSED ([^\s:]+): ", r"fit \1: ", ["book/listings/**/*.out", "programs/**/*.out"]),
    L("REFUSED refused/decimal.rce", "fit fit/decimal.rce", ["programs/money/README.md"]),
    L('out.starts_with(&format!("REFUSED {p}: line "))', 'out.starts_with(&format!("fit {p}: line "))', "spec/checker/tests/cli.rs"),
    L("#   - REFUSED", "#   - fit main.rce", "conformance/tests/**"),
    L("re.match(r'(?:REFUSED|FAIL) (\\S+): (.*)', ln)", "re.match(r'(?:fit|FAIL) (\\S+): (.*)', ln)", "stdlib/scripts/one-language.py"),
    L('echo "pre-push: REFUSED ${lref}', 'echo "pre-push: stale spec artefacts for ${lref}', ".githooks/pre-push"),
    L("# Refuse a push whose RCE spec artefacts are stale", "# Stop a push whose RCE spec artefacts are stale", ".githooks/pre-push"),
    # the summary lines of rce-spec-check, and the claims checked against them
    L('"FAIL (0.6 LALR should refuse): {d}/reject/{}"', '"FAIL (0.6 LALR should report): {d}/fit/{}"', "spec/checker/src/bin/spec_check.rs"),
    L('"grammar: 0.6 LALR(1) accepted {acc}/{nacc} parsing 0.6 cases, refused {refd}/{nref} syntax rejects"',
      '"grammar: 0.6 LALR(1) parsed {acc}/{nacc} parsing 0.6 cases, reported {refd}/{nref} syntax fixtures"',
      "spec/checker/src/bin/spec_check.rs"),
    L("// a reject case refused by the Rust parser must be refused by the tables too",
      "// a syntax fixture the Rust parser reports must be reported by the tables too", "spec/checker/src/bin/spec_check.rs"),
    L('load(&rce_files(&spec.join(d).join("reject")))', 'load(&rce_files(&spec.join(d).join("fit")))', "spec/checker/src/bin/spec_check.rs"),
    L('!["reject", "ill_typed", "inadmissible"].contains(&n.as_str())', '!["fit", "ill_typed", "inadmissible"].contains(&n.as_str())',
      "spec/checker/src/bin/spec_check.rs"),
    L('let rej = load(&rce_files(&ex.join("reject")));', 'let rej = load(&rce_files(&ex.join("fit").join("syntax")));',
      "spec/checker/src/bin/spec_check.rs"),
    L('"FAIL (should be refused): {} tree-sitter={t:?} rust={r} lalr={l}"',
      '"FAIL (should be reported): {} tree-sitter={t:?} rust={r} lalr={l}"', "spec/checker/src/bin/spec_check.rs"),
    L('"grammar: accepted {gok}/{} valid examples, refused {grej}/{} invalid ({} agree)"',
      '"grammar: {gok}/{} examples parsed, {grej}/{} syntax fixtures reported ({} agree)"', "spec/checker/src/bin/spec_check.rs"),
    L('println!("FAIL (checker should refuse): {}", name(&f.0))', 'println!("FAIL (checker should report a rule): {}", name(&f.0))',
      "spec/checker/src/bin/spec_check.rs"),
    L('"checker: accepted {nok}/{} examples, refused {nrej}/{} syntax rejects, refused {nill}/{} ill-typed or inadmissible ({} of them inadmissible)"',
      '"checker: {nok}/{} examples in Form, {nrej}/{} syntax fixtures and {nill}/{} type or admission fixtures reported with their rule ({} of them admission)"',
      "spec/checker/src/bin/spec_check.rs"),
    L("//!    Parses every examples/*.rce (must succeed) and examples/reject/*.rce", "//!    Parses every examples/*.rce (must succeed) and examples/fit/syntax/*.rce",
      "spec/checker/src/bin/spec_check.rs"),
    L("//!    accepted), on examples/reject/ (must be refused), and on", "//!    in Form), on examples/fit/syntax/ (each reported with its rule), and on",
      "spec/checker/src/bin/spec_check.rs"),
    L("//!    parsers, then be refused by the type, well-formedness or admission", "//!    parsers, then be reported by the type, well-formedness or admission",
      "spec/checker/src/bin/spec_check.rs"),
    L('a = re.search(r"grammar: accepted (\\d+)/(\\d+) valid examples, refused (\\d+)/(\\d+) invalid", out)',
      'a = re.search(r"grammar: (\\d+)/(\\d+) examples parsed, (\\d+)/(\\d+) syntax fixtures reported", out)', "scripts/claim-counts.py"),
    L('k = re.search(r"checker: accepted (\\d+)/(\\d+) examples, refused (\\d+)/(\\d+) syntax rejects, refused (\\d+)/(\\d+) ill-typed or inadmissible \\((\\d+) of them inadmissible", out)',
      'k = re.search(r"checker: (\\d+)/(\\d+) examples in Form, (\\d+)/(\\d+) syntax fixtures and (\\d+)/(\\d+) type or admission fixtures reported with their rule \\((\\d+) of them admission", out)',
      "scripts/claim-counts.py"),
    L('need(f"grammar {ga[0]}/{ga[1]} accepted, {ga[2]}/{ga[3]} refused" in o, "00-overview: grammar accepted/refused", errs)',
      'need(f"grammar {ga[0]}/{ga[1]} parsed, {ga[2]}/{ga[3]} reported" in o, "00-overview: grammar parsed/reported", errs)',
      "scripts/claim-counts.py"),
    L('need(f"checker {ck[0]}/{ck[1]} accepted, {ck[2]}/{ck[3]} and {ck[4]}/{ck[5]} refused ({ck[6]} of them inadmissible)" in o, "00-overview: checker line", errs)',
      'need(f"checker {ck[0]}/{ck[1]} in Form, {ck[2]}/{ck[3]} and {ck[4]}/{ck[5]} reported ({ck[6]} of them admission)" in o, "00-overview: checker line", errs)',
      "scripts/claim-counts.py"),
    L('if d.is_dir() and d.name not in ("reject", "ill_typed", "inadmissible"))', 'if d.is_dir() and d.name not in ("fit", "ill_typed", "inadmissible"))',
      "scripts/claim-counts.py"),
    L('    for d in ("reject", "ill_typed", "inadmissible"):\n        c["examples_" + d] = len(list((ex / d).glob("*.rce")))',
      '    for d in ("fit/syntax", "ill_typed", "inadmissible"):\n        c["examples_" + d.replace("/", "_")] = len(list((ex / d).glob("*.rce")))',
      "scripts/claim-counts.py"),
    X(r"grammar (\d+)/(\d+) accepted,(\s+)(\d+)/(\d+) refused; checker (\d+)/(\d+) accepted, (\d+)/(\d+) and (\d+)/(\d+) refused \((\d+) of them(\s+)inadmissible\)",
      r"grammar \1/\2 parsed,\3\4/\5 reported; checker \6/\7 in Form, \8/\9 and \10/\11 reported (\12 of them\13admission)",
      "spec/book/00-overview.md"),
    # the fixture directories the tools read (6.5)
    L("examples/reject", "examples/fit/syntax", ["spec/checker/tests/conc06.rs", "spec/wasm-test/test.mjs", "spec/README.md",
                                                "spec/book/*.md", "README.md", "CONTRIBUTING.md"]),
] + phrases([(r"spec/examples/reject/", "spec/examples/fit/syntax/")], ALL, where="any", skip=["conformance/tests/**"]) + [
    L("spec/conc06/reject/", "spec/conc06/fit/", ALL, skip=["conformance/tests/**"]),
    L('"conc06/reject/{n} accepted"', '"conc06/fit/{n} accepted"', "spec/checker/tests/conc06.rs"),
    L('"conc06b/reject/{n} accepted"', '"conc06b/fit/{n} accepted"', "spec/checker/tests/conc06.rs"),
    L('spec_dir().join("conc06/reject")', 'spec_dir().join("conc06/fit")', "spec/checker/tests/conc06.rs"),
    L('spec_dir().join("conc06b/reject")', 'spec_dir().join("conc06b/fit")', "spec/checker/tests/conc06.rs"),
    L('println!("conc06b: {} reject programs refused", rej.len());', 'println!("conc06b: {} syntax fixtures reported", rej.len());',
      "spec/checker/tests/conc06.rs"),
    L('println!("conc06: {} accepted, {} refused", acc.len(), rej.len());', 'println!("conc06: {} accepted, {} reported", acc.len(), rej.len());',
      "spec/checker/tests/conc06.rs"),
    L('files(&dir("reject"))', 'files(&dir("fit"))', ["spec/checker/tests/conc06_check.rs", "spec/checker/tests/conc06b_check.rs"]),
    L('for (const f of files(`${d}/reject`))', 'for (const f of files(`${d}/fit`))', "spec/wasm-test/test.mjs"),
    L('let fs = files("refused");', 'let fs = files("fit");', "spec/checker/tests/v09_examples.rs"),
    L("`../examples-0.9/refused/*.rce`", "`../examples-0.9/fit/*.rce`", "spec/checker/tests/v09_examples.rs"),
    L("-not -path '*/reject/*' -not -path '*/ill_typed/*' -not -path '*/inadmissible/*' -not -path '*/refused/*'",
      "-not -path '*/fit/*' -not -path '*/ill_typed/*' -not -path '*/inadmissible/*'", "scripts/t7-equiv.sh"),
    L("(no Form, no explorable event, or refused standalone)", "(no Form, no explorable event, or reported standalone)", "scripts/t7-equiv.sh"),
    L('if "/refused/" in p.as_posix()', 'if "/fit/" in p.as_posix()', "scripts/corpus.py"),
    # book and programs: the listing directories (6.5) and their checks
    L("listings/chNN/refused/", "listings/chNN/fit/", ALL),
    X(r"listings/(ch\d\d)/refused/", r"listings/\1/fit/", ALL),
    L("listings/ch05/door_refused.out", "listings/ch05/door_kept.out", ALL),
    L("`refused/decimal.rce`", "`fit/decimal.rce`", "programs/**"),
    L("see refused/decimal.rce", "see fit/decimal.rce", "programs/**"),
    L("check refused/decimal.rce", "check fit/decimal.rce", "programs/**"),
    L("#   * NAME.rce outside a refused/ directory must pass `rce test`.", "#   * NAME.rce outside a fit/ directory must pass `rce test`.",
      ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L("#   * refused/NAME.rce must be refused by `rce check`, and the output must\n#     contain the text in refused/NAME.expect. refused/NAME.with lists extra\n#     files (relative to refused/) that belong to the same program.",
      "#   * fit/NAME.rce must be reported by `rce check` (status 1), and the output\n#     must contain the text in fit/NAME.expect. fit/NAME.with lists extra\n#     files (relative to fit/) that belong to the same program.",
      ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L("-not -path '*/refused/*'", "-not -path '*/fit/*'", ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L("# 2. programs that must be refused", "# 2. programs the checker reports, with their rule", ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L('bad "$f (expected refusal containing: $expect)"', 'bad "$f (expected a report containing: $expect)"',
      ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L("find . -path '*/refused/*.rce'", "find . -path '*/fit/*.rce'", ["book/scripts/check-listings.sh", "programs/scripts/check.sh"]),
    L("- [Errors and refusals](ch06-errors.md)", "- [Errors and kept outcomes](ch06-errors.md)", "book/src/SUMMARY.md"),
    L("# Errors and refusals", "# Errors and kept outcomes", "book/src/ch06-errors.md"),
    L("## Refusing an event", "## Keeping the state", "book/src/ch06-errors.md"),

    # ---- 6.4: the title of section 7 (book chapter 9) and its heading in the clause data
    L("## 7. Well-formedness: what is refused at entry", "## 7. Well-formedness: the rules a text meets to have its Form", ALL),
    # the clause data carries the heading of each clause's section; its clause ids
    # follow the sentences, which the hand rewording changes (COMP-20, 10)
    L(". Well-formedness: what is refused at entry", ". Well-formedness: the rules a text meets to have its Form", "conformance/clauses*.json"),

    # ---- 6.1: `rce dim` (10.11): the JSON key, the labels and the report (rce-dim/2)
    L('"format": "rce-dim/1",', '"format": "rce-dim/2",', "spec/checker/src/dim.rs"),
    L("/// The JSON report (`rce-dim/1`, compact, sorted keys).", "/// The JSON report (`rce-dim/2`, compact, sorted keys).", "spec/checker/src/dim.rs"),
    L("(format `rce-dim/1`, compact,", "(format `rce-dim/2`, compact,", "spec/book/12-tooling.md"),
    L('"refuses": d.refuses,', '"keeps": d.keeps,', "spec/checker/src/dim.rs"),
    T("refuses", "keeps", "spec/checker/src/dim.rs", where="code"),
    L('format!("refused:err#{i}")', 'format!("kept:err#{i}")', "spec/checker/src/dim.rs"),
    L('format!("refused:{x}")', 'format!("kept:{x}")', "spec/checker/src/dim.rs"),
    L('format!("refused:{name}")', 'format!("kept:{name}")', "spec/checker/src/dim.rs"),
    L('l.starts_with("refused:")', 'l.starts_with("kept:")', "spec/checker/src/dim.rs"),
    L('l.strip_prefix("refused:") { format!("refused: {x}") }', 'l.strip_prefix("kept:") { format!("kept: {x}") }', "spec/checker/src/dim.rs"),
    L("(emits: {}, refuses: {})", "(emits: {}, keeps: {})", "spec/checker/src/dim.rs"),
    X(r"\(emits: (yes|no), refuses: (yes|no)\)", r"(emits: \1, keeps: \2)", ["book/listings/**", "spec/book/*.md", "book/src/**"]),
    L("`refuses`", "`keeps`", ["spec/book/12-tooling.md", "book/src/ch18-behaviour-dimension.md"]),
    L('B                 refused: Underflow', 'B                 kept: Underflow', "spec/checker/tests/dim.rs"),
    L("/// Some explored step refuses its event.", "/// Some explored step has the kept outcome.", "spec/checker/src/dim.rs"),
    L("/// The label of one step: the canonical text of its outputs, or its refusal.",
      "/// The label of one step: the canonical text of its outputs, or its kept reason.", "spec/checker/src/dim.rs"),
    L("Err(_) => row.push(Some(i)), // T8: a refused event leaves the state unchanged",
      "Err(_) => row.push(Some(i)), // T8: a kept event leaves the state unchanged", "spec/checker/src/dim.rs"),

    # ---- 6.1: the caretaker verdict (W-CAP-5)
    T("Refused", "Kept", "spec/checker/src/caretaker.rs", where="code"),
    T("refusal_record", "capability_record", CHK, where="code"),
    L("/// W-CAP-5: the use is refused with the fixed record", "/// W-CAP-5: the use has the kept outcome, with the fixed record",
      "spec/checker/src/caretaker.rs"),
    T("use_before_revoke_is_admitted_and_after_is_refused", "use_before_revoke_is_admitted_and_after_has_the_kept_outcome", CHK, where="any"),
    T("a_use_of_an_ungranted_caretaker_is_refused_and_replay_is_exact",
      "a_use_of_an_ungranted_caretaker_has_the_kept_outcome_and_replay_is_exact", CHK),
    L("caretaker log fold: refusal record,", "caretaker log fold: capability record,", "spec/checker/src/v09.rs"),

    # ---- 6.5: the conformance protocol: expectations, verdicts, outcomes, finding kinds
    X(r"^# expect: reject\b", "# expect: fit", "conformance/tests/**"),
    X(r"^# expect: accept\b", "# expect: ok", "conformance/tests/**"),
    L('# outcome: "refused:supervisor', '# outcome: "stopped:supervisor', "conformance/tests/**"),
    X(r"^# outcome: refused:", "# outcome: kept:", "conformance/tests/**"),
    L('"verdict": "accept"', '"verdict": "ok"', ["spec/checker/src/bin/conform*.rs", "conformance/adapters/*"]),
    L('"verdict": "reject"', '"verdict": "fit"', ["spec/checker/src/bin/conform*.rs", "conformance/adapters/*"]),
    L('"verdict":"reject"', '"verdict":"fit"', ["spec/checker/src/bin/conform*.rs"]),
    L('verdict: "accept"', 'verdict: "ok"', "conformance/adapters/*.mjs"),
    L('verdict: "reject"', 'verdict: "fit"', "conformance/adapters/*.mjs"),
    X(r'(?<![\w-])"accept"', '"ok"', ["conformance/harness/*.py", "conformance/tools/*.py"], where="str"),
    X(r'(?<![\w-])"reject"', '"fit"', ["conformance/harness/*.py", "conformance/tools/*.py", "conformance/findings/*.json"], where="any"),
    L('return "fail", f"refused at entry: {', 'return "fail", f"fit reported at entry: {', "conformance/harness/run.py"),
    L('"kind": "refused-at-entry"', '"kind": "fit-at-entry"', CONF),
    L("# expect: reject            accept | reject | run | trace", "# expect: fit               ok | fit | run | trace",
      "conformance/harness/rcetest.py"),
    L("# diagnostic: W-LEX-1       reject: the rule code of the first diagnostic", "# diagnostic: W-LEX-1       fit: the rule code of the first diagnostic",
      "conformance/harness/rcetest.py"),
    L("# outcome: trap:...         run: done | refused:E | trap:KIND | deadlock",
      "# outcome: trap:...         run: done | kept:E | stopped:E | trap:KIND | deadlock", "conformance/harness/rcetest.py"),
    L('errs.append(f"{p}: reject without diagnostic")', 'errs.append(f"{p}: fit without diagnostic")', "conformance/tools/validate.py"),
    L('(r"^\\s*example .*-> (?!refused)", ', '(r"^\\s*example .*-> (?!kept)", ', "conformance/tools/tag.py"),
    L('.startswith("refused"): cl |= {"T8"}', '.startswith("kept"): cl |= {"T8"}', "conformance/tools/tag.py"),
    L('cl |= {"T8"} if "refused" in src or', 'cl |= {"T8"} if "kept:" in src or', "conformance/tools/tag.py"),
    L('refused = [a for a in h.get("admitted", []) if "refused" in a[2]]', 'kept = [a for a in h.get("admitted", []) if "kept" in a[2]]',
      "conformance/tools/targeted.py"),
    L('if refused or r.get("error"): findings.append({"test": m["id"], "law": "every input applied (fresh fuel per input)", "reference": refused or r.get("error")})',
      'if kept or r.get("error"): findings.append({"test": m["id"], "law": "every input applied (fresh fuel per input)", "reference": kept or r.get("error")})',
      "conformance/tools/targeted.py"),
    L('put(area, f"{stem}-reject", ', 'put(area, f"{stem}-fit", ', "conformance/tools/harvest.py"),
    L('sys.exit(f"generator wrote a refused program: {r}', 'sys.exit(f"generator wrote a program the checker reports: {r}', "conformance/tools/gen_tree.py"),
    L("# the generator wrote something the checker refuses:", "# the generator wrote something the checker reports:", "conformance/tools/gen_numbers.py"),
    L('else ["accepted"]', 'else ["ok"]', "conformance/tools/gen_props.py"),
    L("# T8 on every trace: a refused event leaves the state exactly as it was",
      "# T8 on every trace: a kept event leaves the state exactly as it was", "conformance/harness/run.py"),
    L('return "fail", f"T8: refused step {', 'return "fail", f"T8: kept step {', "conformance/harness/run.py"),
    L("on a trace that contains a refused event (the harness checks T8 on every",
      "on a trace that contains a kept event (the harness checks T8 on every", "conformance/tools/tag.py"),
    L("| expect | `accept`, `reject`, `run`, `trace`, `host`, `replicate` |", "| expect | `ok`, `fit`, `run`, `trace`, `host`, `replicate` |",
      "conformance/INTERPRETING.md"),
    L("| diagnostic | for `reject`: the rule code the checker must report |", "| diagnostic | for `fit`: the rule code the checker must report |",
      "conformance/INTERPRETING.md"),
    L("| outcome | `done`, `deadlock`, `trap:<kind>`, `refused:<error>` |",
      "| outcome | `done`, `deadlock`, `trap:<kind>`, `kept:<error>` (an input), `stopped:<reason>` (a run of `fn main`) |",
      "conformance/INTERPRETING.md"),
    L("`expect: reject`", "`expect: fit`", CONF),
    L("A test with expect accept/run/trace must parse under both.", "A test with expect ok/run/trace must parse under both.", "conformance/tools/ts_crosscheck.py"),
    L("`refused:<error>`", "`kept:<error>`", CONF),

    # ---- 6.5: the brick catalog (`lowering: {"pending": r}`, `pending_files`)
    L('json!({"refused": ', 'json!({"pending": ', "spec/checker/src/brick.rs"),
    L('"its file is refused by the checker"', '"its file is not yet in Form"', "spec/checker/src/brick.rs"),
    T("refused_files", "pending_files", [CHK, "spec/book/*.md"], where="any"),
    L('`{"refused": reason}`', '`{"pending": reason}`', "spec/book/12-tooling.md"),

    # ---- 6.5: the corpus generator writes rce-corpus/2 rows (published rce-corpus/1 rows stay as published)
    L('"format": "rce-corpus/1"', '"format": "rce-corpus/2"', "scripts/corpus.py"),
    L('"origin": "conformance-reject"', '"origin": "conformance-fit"', "scripts/corpus.py"),
    L('"verdict": "reject"', '"verdict": "fit"', "scripts/corpus.py"),
    L('row.update(verdict="reject",', 'row.update(verdict="fit",', "scripts/corpus.py"),
    L('outcome="caught_by_examples" if code == "W-INT-1" else "refused")', 'outcome="caught_by_examples" if code == "W-INT-1" else "kept")',
      "scripts/corpus.py"),
    L('mut_stats = {"reject": 0,', 'mut_stats = {"fit": 0,', "scripts/corpus.py"),
    L('mut_stats["reject"] += 1', 'mut_stats["fit"] += 1', "scripts/corpus.py"),
    L('is_reject = exp == "reject" or exp.startswith("W-")', 'is_fit = exp == "fit" or exp.startswith("W-")', "scripts/corpus.py"),
    L("(not expect_accept and is_reject)", "(not expect_accept and is_fit)", "scripts/corpus.py"),
    L('"refused": len(refused),', '"fit": len(fit),', "scripts/corpus.py"),
    L('"refused_rows": refused,', '"fit_rows": fit,', "scripts/corpus.py"),
    L('if k != "refused_rows"}', 'if k != "fit_rows"}', "scripts/corpus.py"),
    L("    rows, refused = [], []", "    rows, fit = [], []", "scripts/corpus.py"),
    L("            refused.append(", "            fit.append(", "scripts/corpus.py"),
    L("# negatives: conformance rejects, and token mutants", "# negatives: the conformance tests that expect `fit`, and token mutants", "scripts/corpus.py"),

    # ---- 6.6: claim verdicts that are wording only (W-INT-1 quotes the outcome syntax)
    L('"expected refused: {}, but {evn} was applied",', '"expected kept: {}, but {evn} was applied",', "spec/checker/src/interp.rs"),
    L('bad = Some(format!("event {} ({evn}) refused: {r}", k + 1));', 'bad = Some(format!("event {} ({evn}) kept: {r}", k + 1));',
      "spec/checker/src/interp.rs"),
    # session types: "refusing configuration" reads "stuck configuration" (6.6)
    T("refuses", "receiver_stuck", "spec/checker/src/mpst_async.rs", where="code"),
    L("/// the number explored, or the first stuck or refusing configuration", "/// the number explored, or the first stuck configuration",
      "spec/checker/src/mpst_async.rs"),
    L("//! whose receiver refuses its queue head;", "//! whose receiver cannot take its queue head (a stuck configuration);",
      "spec/checker/src/mpst_async.rs"),
    T("async_explorer_finds_a_refusal", "async_explorer_finds_a_stuck_configuration", CHK, where="any"),
    L('panic!("vector {n}: stuck or refusing at {k:?}")', 'panic!("vector {n}: stuck at {k:?}")', "spec/checker/tests/lean_session_async.rs"),
    L("// role 1 offers only label 1 from 0, but 0 sends label 0: refused at the head",
      "// role 1 offers only label 1 from 0, but 0 sends label 0: stuck at the head", "spec/checker/tests/lean_session_async.rs"),
    L("//! never get stuck and no receiver refuses its queue head", "//! never get stuck, and every receiver can take its queue head",
      "spec/checker/tests/lean_session_async.rs"),

    # ---- 6.7: tools: the IDE receipt verdict (old tags kept as variants, COMP-20 (c))
    L("    /// `rce check` refused; the change was not applied.\n    Refused { diagnostics: Vec<String> },\n"
      "    /// The action was rejected before the checker ran (bad path, policy).\n    Rejected { reason: String },\n",
      "    /// `rce check` reported paths; nothing was written.\n    Fit { diagnostics: Vec<String> },\n"
      "    /// The action was not performed; the reason names the path or the policy.\n    NotPerformed { reason: String },\n"
      "    /// The earlier spelling of `Fit`, read so that stored chains verify (COMP-20 (c)); never written.\n"
      "    Refused { diagnostics: Vec<String> },\n"
      "    /// The earlier spelling of `NotPerformed`, read so that stored chains verify; never written.\n"
      "    Rejected { reason: String },\n", "ide/crates/ide-receipt/src/lib.rs"),
    T("Verdict::Refused", "Verdict::Fit", ["ide/**"], where="code"),
    T("Verdict::Rejected", "Verdict::NotPerformed", ["ide/**"], where="code"),
    # the proposal statuses and the user's decision on a proposed change
    T("ProposalStatus::Refused", "ProposalStatus::Fit", ["ide/**"], where="code"),
    T("ProposalStatus::Rejected", "ProposalStatus::Declined", ["ide/**"], where="code"),
    L("    /// The checker refused it on accept; nothing was written.\n    Refused {",
      "    /// The checker reported paths on accept; nothing was written.\n    Fit {", "ide/crates/rce-ide/src/model.rs"),
    X(r"(\n\s*)Rejected \{(\s*\n\s*receipt)", r"\1Declined {\2", "ide/crates/rce-ide/src/model.rs"),
    L('Verdict::Fit { .. } => "refused".into(),', 'Verdict::Fit { .. } => "fit".into(),', "ide/crates/rce-ide/src/model.rs"),
    L('Verdict::NotPerformed { .. } => "rejected".into(),', 'Verdict::NotPerformed { .. } => "not performed".into(),', "ide/crates/rce-ide/src/model.rs"),
    T("Action::Reject", "Action::Decline", ["ide/**"], where="code"),
    L("    Reject(u64),", "    Decline(u64),", "ide/crates/rce-ide/src/model.rs"),
    L('button(&format!("reject {}", p.id)', 'button(&format!("decline {}", p.id)', "ide/crates/rce-ide/src/view.rs"),
    L('format!("refused by rce check: {}", diagnostics.join("; "))', 'format!("fit reported by rce check: {}", diagnostics.join("; "))',
      "ide/crates/rce-ide/src/view.rs"),
    L('format!("rejected, receipt {}",', 'format!("declined, receipt {}",', "ide/crates/rce-ide/src/view.rs"),
    L('"proposal.reject"', '"proposal.decline"', ["ide/**"]),
    L('reason: "rejected by the user".into(),', 'reason: "declined by the user".into(),', "ide/crates/rce-ide/src/vm.rs"),
    T("proposals_accept_reject_and_refuse_with_receipts", "proposals_accept_decline_and_fit_with_receipts", ["ide/**"], where="any"),
    L("//!   proposals with accept/reject through rce-mcp,", "//!   proposals with accept/decline through rce-mcp,", "ide/crates/rce-ide/src/lib.rs"),
    # acp-client: methods outside the grant
    T("refused_methods", "methods_outside_grant", ["ide/**"], where="any"),
    L("//!   refused methods, reverted files,", "//!   methods outside the grant, reverted files,", "ide/crates/acp-client/src/lib.rs"),
    # rce-graph3d and rce-mcp: a node whose file the checker reports (`fit`)
    T("refused", "fit", ["ide/crates/rce-graph3d/**"], where="code"),
    T("Refuses", "Reports", ["ide/crates/rce-graph3d/**"], where="code"),
    L('["refused"] => Ok(', '["fit"] => Ok(', "ide/crates/rce-graph3d/src/query.rs"),
    L('run(&g, "refused")', 'run(&g, "fit")', "ide/crates/rce-graph3d/src/query.rs"),
    L("[divergence > D] | refused | uses-hw NAME", "[divergence > D] | fit | uses-hw NAME", ["ide/**"]),
    L("with energy filters, refused, uses-hw", "with energy filters, fit, uses-hw", "ide/docs/PARITY.md"),
    L('label: format!("refused: {}", trunc(first, 48)),', 'label: format!("fit: {}", trunc(first, 48)),', "ide/crates/rce-graph3d/src/graph.rs"),
    L('"refused": n.refused})', '"fit": n.fit})', "ide/crates/rce-mcp/src/lib.rs"),
    L('.expect("refusal node")', '.expect("fit node")', "ide/crates/rce-graph3d/tests/graph3d.rs"),
    T("refused_file_gets_error_node", "fit_file_gets_error_node", ["ide/**"], where="any"),
    L("/// The checker refused the file this item lives in (error node -> item).", "/// The checker reported the file this item lives in (error node -> item).",
      "ide/crates/rce-graph3d/src/graph.rs"),
    L("/// The checker refused this node's file.", "/// The checker reported this node's file (`fit`).", "ide/crates/rce-graph3d/src/graph.rs"),
    L("/// checker verdicts (`Some(refusal)` when refused).", "/// checker verdicts (`Some(report)` for a file reported `fit`).", "ide/crates/rce-graph3d/src/graph.rs"),
    L("// A refused file gets an error node naming the refusal, linked to its items.",
      "// A file reported `fit` gets an error node naming the report, linked to its items.", "ide/crates/rce-graph3d/src/graph.rs"),
    # mei-agent: the checker's report concludes `Fit`
    T("Concl::Reject", "Concl::Fit", ["ide/**"], where="code"),
    L("    Reject(Vec<String>),", "    Fit(Vec<String>),", "ide/crates/mei-agent/src/agent.rs"),
    L('"v_reject"', '"v_fit"', ["ide/**"]),
    L('"set_reject"', '"set_fit"', ["ide/**"]),
    L('format!("checker reject: {}", d.join("; "))', 'format!("checker fit: {}", d.join("; "))', "ide/crates/mei-agent/src/agent.rs"),
    L("The checker decides; refused edits are not applied.", "The checker decides; an edit it reports is not applied.", "ide/crates/mei-agent/src/agent.rs"),
    L("`verdict` (none, done, not_done, reject)", "`verdict` (none, done, not_done, fit)", "ide/crates/mei-agent/src/lib.rs"),
    L("escalate needs `v_reject` or `v_not_done`", "escalate needs `v_fit` or `v_not_done`", "ide/crates/mei-agent/src/lib.rs"),
    L("checker reject or a not-done decision", "checker report (`fit`) or a not-done decision", "ide/crates/mei-agent/src/lib.rs"),
    T("climbs_to_g4_only_on_reject_or_not_done", "climbs_to_g4_only_on_fit_or_not_done", ["ide/**"], where="any"),
    L("// Reject at gear0 with the budget spent: fail, never escalate.", "// `fit` at gear0 with the budget spent: fail, never escalate.",
      "ide/crates/mei-agent/tests/agent.rs"),
    L("// Walk to gear4 on rejects; at gear4 a reject fails.", "// Walk to gear4 on `fit` reports; at gear4 a `fit` report fails.",
      "ide/crates/mei-agent/tests/agent.rs"),
    L("// Exactly one checker refusal (G3) on the way up.", "// Exactly one checker report (G3) on the way up.", "ide/crates/mei-agent/tests/agent.rs"),
    L("// G3 proposes a step on an undeclared event: the checker refuses it.",
      "// G3 proposes a step on an undeclared event: the checker reports it (`fit`).", "ide/crates/mei-agent/tests/agent.rs"),
    # rce-mcp: an action not performed, and the checker's reports
    T("rejected", "not_performed", "ide/crates/rce-mcp/src/lib.rs", where="code"),
    T("refusals", "reports", "ide/crates/rce-mcp/src/lib.rs", where="code"),
    L("    let refused = rs\n", "    let fit = rs\n", "ide/crates/rce-mcp/tests/mcp_stdio.rs"),
    L("    assert_eq!(refused, 1);", "    assert_eq!(fit, 1);", "ide/crates/rce-mcp/tests/mcp_stdio.rs"),
    L("// A change the checker refuses: step on an undeclared event. Nothing is written.",
      "// A change the checker reports (`fit`): step on an undeclared event. Nothing is written.", "ide/crates/rce-mcp/tests/mcp_stdio.rs"),
    L("// Policy: escapes, hooks and config files are rejected before the checker runs.",
      "// Policy: escapes, hooks and config files are not performed; the checker does not run.", "ide/crates/rce-mcp/tests/mcp_stdio.rs"),
    # rce-guest: a run's lines; a file the checker reports
    L('println!("refused: {e}");', 'println!("fit: {e}");', "ide/crates/rce-guest/src/main.rs"),
    L("Exit codes follow spec 10.1: 0 ok, 1 refused, 2 usage.", "Exit codes follow spec 10.1: 0 ok, 1 outstanding, 2 usage.",
      "ide/crates/rce-guest/src/main.rs"),
    # the sandbox runs `rce run`: the kept outcome of the third Dec
    L("    assert_eq!(r.exit_code, Some(1)); // the third Dec is refused: Underflow",
      "    assert_eq!(r.exit_code, Some(1)); // the third Dec has the kept outcome: Underflow", "ide/crates/ide-sandbox/tests/sandbox.rs"),
    L('    assert!(r.stdout.contains("refused"), "{}", r.stdout);', '    assert!(r.stdout.contains("kept: Underflow"), "{}", r.stdout);',
      "ide/crates/ide-sandbox/tests/sandbox.rs"),

    # ---- 6.8: identifiers and test names in spec/checker
    T("SAFE_REFUSED", "SAFE_FORMS", CHK, where="code"),
    T("check_accepts_and_refuses", "check_reports_ok_and_fit", CHK, where="any"),
    T("run_events_prints_each_state_and_stops_at_refusal", "run_events_prints_each_state_and_stops_at_the_kept_outcome", CHK, where="any"),
    L('"0 (initial) Counter { n = 0 }\\n1 Dec refused: Underflow\\n"', '"0 (initial) Counter { n = 0 }\\n1 Dec kept: Underflow Counter { n = 0 }\\n"',
      "spec/checker/tests/cli.rs"),
    T("a_state_write_outside_the_refinement_refuses_the_event", "a_state_write_outside_the_refinement_keeps_the_state", CHK, where="any"),
    T("invalid_bytes_refuse_and_leave_state", "invalid_bytes_keep_the_state", CHK, where="any"),
    L("//! 5.15 `utf8`: invalid bytes refuse the event (T8) on the host and in", "//! 5.15 `utf8`: invalid bytes give the event the kept outcome (T8) on the host and in",
      "spec/checker/tests/utf8_kept.rs"),
    T("actor_refusals_match", "actor_kept_outcomes_match", CHK, where="any"),
    T("REFUSALS", "KEPT_OUTCOMES", "spec/checker/tests/emit_run.rs", where="code"),
    L('format!("refusals_{lang:?}")', 'format!("kept_{lang:?}")', "spec/checker/tests/emit_run.rs"),
    L('format!("refusals_win_{lang:?}")', 'format!("kept_win_{lang:?}")', "spec/checker/tests/emit_run.rs"),
    L('"sc_refusals"', '"sc_kept"', "spec/checker/tests/emit_run.rs"),
    L('"refusals and supervision"', '"kept outcomes and supervision"', "spec/checker/tests/emit_run.rs"),
    L("/// handler refusals of every kind:", "/// handler kept outcomes of every kind:", "spec/checker/tests/emit_run.rs"),
    L("/// handler refusals, arithmetic and index traps,", "/// handler kept outcomes, arithmetic and index traps,", "spec/checker/tests/emit_run.rs"),
    L("/// division by zero, index), a refused atomic transaction, and a supervisor", "/// division by zero, index), a kept atomic transaction, and a supervisor",
      "spec/checker/tests/emit_run.rs"),
    T("typed_schema_refuses_bad_rows_and_index_matches_scan", "typed_schema_keeps_the_state_on_bad_rows_and_index_matches_scan", CHK, where="any"),
    T("admitted_and_refused", "admitted_and_reported_fit", CHK, where="any"),
    T("an_irrefutable_if_let_is_refused", "an_irrefutable_if_let_reports_fit", CHK, where="any"),
    T("an_unknown_module_is_still_refused", "an_unknown_module_still_reports_fit", CHK, where="any"),
    T("a_method_outside_the_library_is_refused_at_check_time", "a_method_outside_the_library_reports_fit_at_check_time", CHK, where="any"),
    T("a_dependency_named_after_a_library_namespace_is_refused_with_its_reason",
      "a_dependency_named_after_a_library_namespace_reports_fit_with_its_reason", CHK),
    T("b008_float_rem_is_refused", "b008_float_rem_reports_fit", CHK, where="any"),
    T("b009_f64_min_max_evaluate_and_pow_is_refused", "b009_f64_min_max_evaluate_and_pow_reports_fit", CHK, where="any"),
    T("box_without_its_value_is_refused_not_a_panic", "box_without_its_value_reports_fit_not_a_panic", CHK, where="any"),
    T("conc06_accept_and_reject", "conc06_accept_and_fit", CHK, where="any"),
    T("ctx2_plain_join_and_nonliteral_lit_refused", "ctx2_plain_join_and_nonliteral_lit_report_fit", CHK, where="any"),
    T("ctx2_plain_text_argument_refused", "ctx2_plain_text_argument_reports_fit", CHK, where="any"),
    T("ctx2_text_and_byte_parameters_refused", "ctx2_text_and_byte_parameters_report_fit", CHK, where="any"),
    T("ctx2_untrusted_scalar_refused", "ctx2_untrusted_scalar_reports_fit", CHK, where="any"),
    T("ctx3_model_call_control_dependent_on_untrusted_data_refused", "ctx3_model_call_control_dependent_on_untrusted_data_reports_fit", CHK, where="any"),
    T("ifc7_capability_field_refused", "ifc7_capability_field_reports_fit", CHK, where="any"),
    T("ifc7_guarded_write_from_model_event_refused", "ifc7_guarded_write_from_model_event_reports_fit", CHK, where="any"),
    T("ifc7_trusted_declaration_refused", "ifc7_trusted_declaration_reports_fit", CHK, where="any"),
    T("tri1_exfiltration_on_model_event_refused_and_no_flow_admitted", "tri1_exfiltration_on_model_event_reports_fit_and_no_flow_admitted", CHK, where="any"),
    T("header_context_refuses_cr_and_lf", "header_context_reports_fit_for_cr_and_lf", CHK, where="any"),
    T("ill_typed_programs_refused_by_rule", "ill_typed_programs_report_fit_by_rule", CHK, where="any"),
    T("injection_secrets_and_missing_authority_are_refused", "injection_secrets_and_missing_authority_report_fit", CHK, where="any"),
    T("overlapping_paths_are_refused", "overlapping_paths_report_fit", CHK, where="any"),
    T("private_unqualified_unknown_and_clashing_names_are_refused", "private_unqualified_unknown_and_clashing_names_report_fit", CHK, where="any"),
    T("refused_examples_are_refused_for_the_named_rule", "fit_examples_report_the_named_rule", CHK, where="any"),
    T("spawn_outside_a_scope_is_refused_in_every_module", "spawn_outside_a_scope_reports_fit_in_every_module", CHK, where="any"),
    T("syntax_rejects_refused_by_parser", "syntax_fixtures_reported_by_parser", CHK, where="any"),
    T("w_ref_1_refuses_outside_the_fragment", "w_ref_1_reports_fit_outside_the_fragment", CHK, where="any"),
    T("while_true_is_refused", "while_true_reports_fit", CHK, where="any"),
    # the helpers that expect a report: `refused(src, rule)` and `refuses(src, rule)`
    T("refused", "reports_fit", ["spec/checker/tests/model_boundary.rs", "spec/checker/tests/lvar_freeze.rs", "ide/crates/rce-ide/tests/version08.rs"],
      where="code"),
    T("refuses", "reports_fit", "spec/checker/tests/indexed.rs", where="code"),
    T("REFUSE", "REPORTED", "spec/checker/tests/par_paths.rs", where="code"),
    # Lean names that tests and the audit cite (13.2)
    T("refine_refused_unchanged", "refine_kept_unchanged", ALL, where="any"),
    T("deltaS_refuse_atomic", "deltaS_kept_atomic", ALL, where="any"),
    T("refused_nostep", "kept_nostep", ALL, where="any"),
    T("seqRef", "seqKept", ALL, where="any"),
    T("write_while_borrowed_refused", "write_while_borrowed_no_typing", ALL, where="any"),
    T("overlapping_mut_refused", "overlapping_mut_no_typing", ALL, where="any"),
    T("dangling_refused", "dangling_no_typing", ALL, where="any"),
    T("read_after_partial_move_refused", "read_after_partial_move_no_typing", ALL, where="any"),
    T("use_after_move_refused", "use_after_move_no_typing", ALL, where="any"),
    T("mut_alias_refused", "mut_alias_no_typing", ALL, where="any"),

    # ---- 13.2: Lean (constructors and types; the comments 13.2 rewords)
    T("Refusal", "Reason", [LEAN], where="code+str"),
    T("pRef", "pReason", [LEAN], where="code"),
    T("refused", "kept", [LEAN], where="code"),
    T("refuse", "kept", ["spec/lean/RCE/Core/**", "spec/lean/CoreMain.lean"], where="code"),
    L("/-- **T8 Atomicity**: a refused event leaves the state unchanged. -/", "/-- **T8 Atomicity**: a kept event leaves the state unchanged. -/", LEAN),
    L("/-- **T8 Atomicity**: a refused event leaves every region unchanged, and an\n    output list containing a refusal only arises from a refused event. -/",
      "/-- **T8 Atomicity**: a kept outcome leaves every region unchanged, and an\n    output list containing a kept record only arises from a kept outcome. -/", LEAN),
    L("/-- The checker's count of consumptions, `none` when refused. -/", "/-- The checker's count of consumptions, `none` when the count has no value (W-LIN-1). -/", LEAN),
    L("/-- An append at term `t` (HC-18): refused when a higher term was promised. -/",
      "/-- An append at term `t` (HC-18): not appended when a higher term was promised. -/", LEAN),
    L("a log of the wrong length is refused. -/", "a log of another length differs. -/", LEAN),
    L("  Outcome : Type            -- a value, a trap or the refusal outcome", "  Outcome : Type            -- a value, a trap or the kept outcome", LEAN),
    L("-- the event is refused with the fixed record", "-- the event has the kept outcome, with the fixed record", LEAN),
    L("let mut refused = 0u64;", "let mut kept = 0u64;", LEAN),
    L('println!(\\"cases {} mismatches {} refused {}\\", n, bad, refused);', 'println!(\\"cases {} mismatches {} kept {}\\", n, bad, kept);', LEAN),
    L("`return Err(…)` for refusals,", "`return Err(…)` for kept outcomes,", LEAN),

    # ---- comments and messages, by the sense of the word (6.1 the kept outcome; 6.4 a report)
    L("        let refuse = |f: Flow| match f {", "        let reason_of = |f: Flow| match f {", ["spec/checker/src/conc.rs"]),
    L("            let refuse = |f: Flow| match f {", "            let reason_of = |f: Flow| match f {", ["spec/checker/src/interp.rs"]),
    T("refuse", "reason_of", ["spec/checker/src/conc.rs", "spec/checker/src/interp.rs"], where="code"),
    L("/// δ (6.4): every region on the pre-state; first refusal in declaration order refuses the event atomically.",
      "/// δ (6.4): every region on the pre-state; the first region, in declaration order, whose step keeps the state gives the event the kept outcome atomically.",
      "spec/checker/src/interp.rs"),
    L("// F.6.2: the fixed refusal record of a violated contract clause", "// F.6.2: the fixed contract record of a violated contract clause", "spec/checker/src/interp.rs"),
    L("//! traps, loops trap at their bound, and an event is refused atomically.", "//! traps, loops trap at their bound, and an event's kept outcome is atomic.",
      "spec/checker/src/interp.rs"),
    L("its control outcome (which regions changed, whether and how it was\n//! refused, the constructors",
      "its control outcome (which regions changed, whether it had the kept\n//! outcome and with which reason, the constructors", "spec/checker/src/dim.rs"),
    L("/// Charge `units` cost units (C.6.4); refuses with `trap(\"budget\")` when", "/// Charge `units` cost units (C.6.4); traps with `trap(\"budget\")` when",
      "spec/checker/src/conc.rs"),
    L("/// migrate to this program's tree (`tree_migrates`) refuses before any", "/// migrate to this program's tree (`tree_migrates`) stops the run before any",
      "spec/checker/src/conc.rs"),
    L("/// removes or moves a recorded position refuses before any step with a", "/// removes or moves a recorded position stops the run before any step with a",
      "spec/checker/tests/emit_run.rs"),
    L("    /// bool; refused once the table has rows. Every later row of the table\n    /// is its values in column order separated by tabs, and a row that does\n    /// not match is refused (BadRow).",
      "    /// bool; kept with its error once the table has rows. Every later row of the\n    /// table is its values in column order separated by tabs, and a row that\n    /// does not match has the kept outcome (BadRow).", "spec/checker/src/rcedb.rs"),
    L("//!   admitted to D (HC-1); a refused write leaves state unchanged (T8).", "//!   admitted to D (HC-1); a write with the kept outcome leaves the state unchanged (T8).",
      "spec/checker/src/rcedb.rs"),
    L("//! traps and refusals, checked i128", "//! traps and kept outcomes, checked i128", "spec/checker/src/emit/mod.rs"),
    L("// and refusals, checked i128 arithmetic", "// and kept outcomes, checked i128 arithmetic", "spec/checker/src/emit/rt.rs"),
    L("/// A step body: `return` refuses the event, as in the reference evaluator.", "/// A step body: `return` gives the event the kept outcome, as in the reference evaluator.",
      "spec/checker/src/emit/base.rs"),
    L("// the error refuses the event; only its name is observable (T8)", "// the error gives the event the kept outcome; only its name is observable (T8)",
      "spec/checker/src/core_elab.rs"),
    L("// `--run` without `fn main()` refuses as `conc::run_main` does (10.3)", "// `--run` without `fn main()` is an error, as `conc::run_main` reports (10.3)",
      "spec/checker/src/emit/prog.rs"),
    L(": the refusal is observable", ": the kept outcome is observable", "spec/checker/src/ifc.rs"),
    L("// F.5.7: `Header.encode` refuses CR and LF at run time (a trap)", "// F.5.7: `Header.encode` traps on CR and LF at run time", "spec/checker/src/ifc.rs"),
    L("// F.5.7: a header value is its text; CR and LF are refused (a trap) by the caller", "// F.5.7: a header value is its text; CR and LF trap in the caller",
      "spec/checker/src/index08.rs"),
    L('"an expression statement whose value is {} is refused; bind it to `_` to drop it on purpose"',
      '"an expression statement whose value is {} is not dropped silently; bind it to `_` to drop it on purpose"', "spec/checker/src/check.rs"),
    L("//! grammar of spec section 3. The first error refuses the text.", "//! grammar of spec section 3. The first error is reported, and the parse ends there.",
      "spec/checker/src/parse.rs"),
    L("(the rest are refused as names)", "(the rest are reported where they stand as names)", "spec/checker/src/parse.rs"),
    L("Returns one verdict per file, in order: Ok or the refusal.", "Returns one verdict per file, in order: Ok or the report.", "spec/checker/src/lib.rs"),
    L('"tree-sitter refused".into()', '"tree-sitter reported it".into()', "spec/checker/src/bin/fuzz.rs"),
    L('"LALR(1) (section 3) refused".into()', '"LALR(1) (section 3) reported it".into()', "spec/checker/src/bin/fuzz.rs"),
    L('format!("Rust parser refused: {e}")', 'format!("Rust parser reported: {e}")', "spec/checker/src/bin/fuzz.rs"),
    L('format!("checker refused valid program: {e}")', 'format!("checker reported a valid program: {e}")', "spec/checker/src/bin/fuzz.rs"),
    L('format!("  refused: {kind}")', 'format!("  reported: {kind}")', "spec/checker/src/bin/fuzz.rs"),
    L('format!("  refused by tree-sitter and rce-check: {name}")', 'format!("  reported by tree-sitter and rce-check: {name}")', "spec/checker/src/bin/fuzz.rs"),
    L('"refused by checker (replaced, not compared)"', '"reported by the checker (replaced, not compared)"', "spec/checker/src/bin/rce-c1.rs"),
    L('format!("  refused: {rule}")', 'format!("  reported: {rule}")', "spec/checker/src/bin/rce-c1.rs"),
    L('"rce run error (events rejected)"', '"rce run error (events not read)"', "spec/checker/src/bin/rce-c1.rs"),
    L('"logs refused at some event"', '"logs with a kept outcome at some event"', "spec/checker/src/bin/rce-c1.rs"),
    L('println!("sample checker refusals / run errors:");', 'println!("sample checker reports / run errors:");', "spec/checker/src/bin/rce-c1.rs"),
    L("/// comparison (refused and trapping logs are compared too)", "/// comparison (logs with kept outcomes and traps are compared too)", "spec/checker/src/bin/rce-c1.rs"),
    L('"vector {n} step {i}: A refused {t}"', '"vector {n} step {i}: A cannot take {t}"', "spec/checker/tests/lean_session.rs"),
    L('"vector {n} step {i}: B refused the dual of {t}"', '"vector {n} step {i}: B cannot take the dual of {t}"', "spec/checker/tests/lean_session.rs"),
    L('"reject {n} accepted by', '"syntax fixture {n} accepted by', "spec/checker/tests/conc06.rs"),
    L("//! * T30: atomic transactions across two actors, some refused by a\n//!   handler failure,", "//! * T30: atomic transactions across two actors, some given the kept\n//!   outcome by a handler failure,",
      "spec/checker/tests/lean_models.rs"),
    L('"{} refused: {:?}", x.event, x.outcome', '"{} kept: {:?}", x.event, x.outcome', ["spec/checker/tests/crypto_rce.rs", "spec/checker/tests/ed25519_rce.rs"]),
    L('"unexpected refusal: {e}"', '"unexpected kept outcome: {e}"', "spec/checker/tests/call_site_cost.rs"),
    L("// one unit less and the single input is refused in both", "// one unit less and the single input has the kept outcome in both",
      "spec/checker/tests/fuel_per_input.rs"),
    L('            // an outcome names the error on refusal; compare up to the renamed error\n'
      '            assert_eq!(x.starts_with("\\"outcome\\":\\"refused"), y.starts_with("\\"outcome\\":\\"refused"), "{file}: {x} vs {y}");',
      '            // COMP-1: the outcome kind of every step agrees; a kept outcome names\n'
      '            // the error, which the renaming renames\n'
      '            let kind = |o: &str| if o.starts_with("\\"outcome\\":\\"applied\\"") { "applied" } else if o.starts_with("\\"outcome\\":{\\"kept\\"") { "kept" } else { "other" };\n'
      '            assert_eq!(kind(x), kind(y), "{file}: {x} vs {y}");', "spec/checker/tests/nameless.rs"),
    L('"refused":', '"fit":', "ide/screens/graph3d-model.json"),
    L("//! refused, W-UNUSED-2). Its cost", "//! reported, W-UNUSED-2). Its cost", "spec/checker/tests/loop_cost.rs"),
] + phrases(STATIC, [
    "spec/checker/src/v09.rs", "spec/checker/src/check.rs", "spec/checker/src/check06b.rs", "spec/checker/src/ts.rs",
    "spec/checker/src/link.rs", "spec/checker/src/brick.rs", "spec/checker/src/tier2.rs", "spec/checker/src/bin/conform_lean.rs",
    "spec/checker/src/bin/fuzz.rs", "spec/checker/src/bin/rce-c1.rs", "spec/checker/src/parse.rs",
    "spec/checker/tests/bound.rs", "spec/checker/tests/conc06.rs", "spec/checker/tests/conc06_check.rs", "spec/checker/tests/conc06b_check.rs",
    "spec/checker/tests/conc06b_run.rs", "spec/checker/tests/fallible_literal.rs", "spec/checker/tests/form_roundtrip.rs",
    "spec/checker/tests/indexed.rs", "spec/checker/tests/lean_lvar.rs", "spec/checker/tests/lean_session_global.rs",
    "spec/checker/tests/lean_session_lin.rs", "spec/checker/tests/lean_session_prio.rs", "spec/checker/tests/lvar_freeze.rs",
    "spec/checker/tests/model_boundary.rs", "spec/checker/tests/modules.rs", "spec/checker/tests/mutual.rs", "spec/checker/tests/nameless.rs",
    "spec/checker/tests/par_paths.rs", "spec/checker/tests/reserved_words.rs", "spec/checker/tests/stdlib_costs.rs",
    "spec/checker/tests/v09_examples.rs", "spec/checker/tests/v09_tier3.rs", "spec/checker/tests/bound/*.rce", "spec/checker/tests/mutual/*.rce",
    "spec/checker/tests/brick.rs", "spec/checker/examples/*.rs", "spec/checker/build_stdlib.rs",
    "spec/lean/RCE/Borrow.lean", "spec/lean/RCE/Ownership.lean", "spec/lean/RCE/Linear.lean", "spec/lean/RCE/Session*.lean",
    "spec/lean/RCE/Core/Elab.lean", "conformance/tools/gen_props.py", "conformance/tools/ts_crosscheck.py", "conformance/tools/harvest.py",
    "conformance/tests/evaluation-fuel/harvested/bad-*.rce", "conformance/tests/types-checking/targeted/w-ty-2-nested-option.rce",
    "conformance/tools/targeted.py", "stdlib/tests/effect/effect_probes.py", "stdlib/scripts/one-language.py",
    "spec/conc06b/accept/*.rce", "spec/examples-0.9/fit/*.rce", "spec/wasm-test/test.mjs",
    "ide/crates/rce-ide/src/graph_edit.rs", "ide/crates/rce-ide/src/graph_panel.rs", "ide/crates/rce-ide/src/main.rs",
    "ide/crates/rce-memory/src/form.rs", "ide/crates/rce-mcp/tests/mcp_stdio.rs", "stdlib/vectors/effect/rce-examples/MANIFEST",
    "scripts/rename-reserved.py", "stdlib/scripts/rename-reserved.py",
], where="com+str") + phrases(DELTA, [
    "spec/checker/src/dim.rs", "spec/checker/src/interp.rs", "spec/checker/tests/host_runtime.rs",
    "spec/checker/tests/traces.rs", "spec/lean/RCE/Core/Events.lean", "spec/lean/RCE/Core/Log.lean", "spec/lean/RCE/Form.lean",
    "spec/lean/RCE/Refine.lean", "spec/lean/RCE/WriteBack.lean", "spec/lean/RCE/Bound.lean", "spec/lean/CoreMain.lean",
    "spec/lean/RCE/Emit.lean", "conformance/harness/run.py", "programs/**/*.rce", "apps/sfu.rce", "spec/lift/go/*.py",
], where="com+str") + [

    L("and `reject/` does not parse.", "and `fit/` does not parse.", "spec/checker/tests/conc06b_check.rs"),
    L('Err(e) => println!("REFUSED {e}"),', 'Err(e) => println!("fit: {e}"),', "spec/checker/examples/run06.rs"),
    L(' refused += 1; } }\\n"', ' kept += 1; } }\\n"', "spec/lean/emit-test/Gen.lean"),
    L("trap (decision 7); refuses the event, state unchanged (T8)", "trap (decision 7); the event has the kept outcome, state unchanged (T8)",
      "docs/figures/src/replace/*"),
    L("(trap, refusal, logged outcome)", "(trap, kept outcome, logged outcome)", "docs/figures/src/replace/matrix.py"),
    L("obligation discharged or refused,", "obligation discharged, or enforced at run time,", "docs/figures/src/replace/*"),
    L('"rce checker\\nparse, types, ownership,\\ntermination, rules named\\non refusal"', '"rce checker\\nparse, types, ownership,\\ntermination, rules named\\nin each report"',
      "docs/figures/src/infographics/make_infographics.py"),
    L('"refused: names the rule"', '"fit: names the rule"', "docs/figures/src/infographics/make_infographics.py"),
    L("a tab is refused (W-LEX-1)", "a tab is reported (W-LEX-1)", "playground/index.html"),
    L("# bob tries to edit ann's message: refused by the author rule", "# bob tries to edit ann's message: the author rule keeps the state",
      "scripts/partner-demo.sh"),
    L("bob's edit of ann's message: refused (author rule)", "bob's edit of ann's message: kept outcome (author rule)", "scripts/partner-demo.sh"),
    T("refused", "kept", "ide/crates/rce-memory/src/form.rs", where="code"),
    T("refused_file_is_reported_not_indexed_wrongly", "fit_file_is_reported_not_indexed_wrongly", ["ide/**"], where="any"),
] + phrases(STATIC, ["scripts/corpus.py", "tree-sitter/src/scanner.c", "ide/crates/rce-ide/tests/*.rs"], where="com+str") + [
    T("refused", "reported", ["spec/checker/tests/lean_session_lin.rs", "spec/checker/tests/lean_session_prio.rs"], where="code"),
    L("ill_typed / inadmissible / reject folders", "ill_typed / inadmissible / fit folders", "conformance/tools/harvest.py"),
    L("k \\u2260 j: refused (shape error)", "k \\u2260 j: reported (shape error)", "docs/figures/src/infographics/make_tensor.py"),
    # the reason Lean's `Refusal` takes in the Rust it emits, named in comments
    L("-- return Err(Refusal::", "-- return Err(Reason::", "spec/lean/**"),
]

# Conformance test ids follow their files (rce-test/1: the id is the path).
def _conformance_ids():
    ids = {}
    for f in tracked():
        if not f.startswith("conformance/tests/") or not f.endswith(".rce"):
            continue
        g = f
        for rx, repl in MOVES:
            g = re.sub(rx, repl, g)
        if g != f:
            a, b = f[len("conformance/tests/"):-4], g[len("conformance/tests/"):-4]
            ids[a] = b
            ids[a.rsplit("/", 1)[-1]] = b.rsplit("/", 1)[-1]
    return ids


class MapRule(Rule):
    """Every key of MAPPING, bounded by characters that cannot continue a test id."""

    def __init__(self, mapping, files, note=""):
        self.mapping = mapping
        keys = sorted(mapping, key=len, reverse=True) or ["(?!x)x"]
        super().__init__("re", r"(?<![\w.-])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w-])", "", files, "any", note=note)

    def replacement(self, m):
        return self.mapping[m.group(1)]

    def label(self):
        return f"map of {len(self.mapping)} conformance test ids [{','.join(self.files)}]"


_ID_RULE = MapRule(_conformance_ids(), [CONF, "spec/checker/tests/**"])
RULES.append(_ID_RULE)


# --------------------------------------------------------------------------
# The engine.

def is_text(path):
    try:
        with open(os.path.join(ROOT, path), "rb") as f:
            b = f.read(8192)
    except (FileNotFoundError, IsADirectoryError):
        return False
    return b"\0" not in b


def plan_moves(files, allow):
    """old path -> new path, for every tracked path a move rule renames."""
    plan = {}
    for f in files:
        if allow.reason(f, f, True):
            continue
        g = f
        for rx, repl in MOVES:
            g = re.sub(rx, repl, g)
        if g != f:
            plan[f] = g
    dests = defaultdict(list)
    for a, b in plan.items():
        dests[b].append(a)
    clash = {b: a for b, a in dests.items() if len(a) > 1 or (b in set(files) and b not in plan)}
    return plan, clash


def rewrite(path, text, allow, rules, stats=None):
    """Apply every rule that names PATH; returns the new text."""
    lang = lang_of(path)
    for r in rules:
        if not r.applies_to(path) or not r.rx.search(text):
            continue
        segs = Segments(text, lang) if r.where != "any" else None
        starts = [0] + [m.end() for m in re.finditer("\n", text)]
        allowed_cache = {}

        def allowed(pos):
            # Third-party text, history and domain data are never rewritten. A line
            # a later group rewrites (`pending`) still takes this group's named
            # renames, but not its general rewording.
            k = bisect.bisect_right(starts, pos) - 1
            if k not in allowed_cache:
                end = text.find("\n", starts[k])
                why = allow.reason(path, text[starts[k]: end if end >= 0 else len(text)], False)
                allowed_cache[k] = why is not None and (r.generic or not why.startswith("pending"))
            return allowed_cache[k]

        count = 0

        def sub(m):
            nonlocal count
            if allowed(m.start()):
                return m.group(0)
            if segs is not None:
                kind = segs.kind(m.start())
                # an identifier renamed in code is renamed where a Rust or
                # Python format string names it (`{name}`, `{name:?}`)
                fmt_arg = (kind == "str" and r.kind == "tok" and "code" in r.where.split("+")
                           and m.start() > 0 and m.string[m.start() - 1] == "{" and m.string[m.end():m.end() + 1] in ("}", ":"))
                if not fmt_arg and (kind == "text" or kind not in r.where.split("+")):
                    return m.group(0)
            new = r.replacement(m)
            if new != m.group(0):
                count += 1
            return new

        text2 = r.rx.sub(sub, text)
        if count and stats is not None:
            r.hits += count
            r.files_hit.add(path)
        text = text2
    return text


def changed_lines(a, b):
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def is_generated(path):
    return any(g.match(path) for g in GENERATED_RX)


def is_hand(path):
    return any(g.match(path) for g in HAND_RX)


def residual_class(path, line, allow):
    why = allow.reason(path, line, False)
    if why is not None:
        return why.split(":")[0]
    if is_generated(path) or any(g.match(path) for g in REGENERATED_RX):
        return "generated"
    if is_hand(path):
        return "hand"
    return "outstanding"


def run(args):
    allow = vocab.Allow(os.path.join(ROOT, "scripts", "vocab-allow.tsv"))
    files = tracked()
    plan, clash = plan_moves(files, allow)
    if clash:
        for b, a in sorted(clash.items()):
            print(f"MOVE COLLISION {b} <- {a}")
        return 2
    inv = {v: k for k, v in plan.items()}
    new_files = sorted(plan.get(f, f) for f in files)

    per_area = defaultdict(lambda: defaultdict(int))
    for a, b in plan.items():
        per_area[vocab.area(a)]["paths"] += 1
    changes, not_idem, residual = {}, [], []
    for f in new_files:
        src = inv.get(f, f)
        if is_generated(f) or not is_text(src):
            continue
        try:
            # newline="": the bytes as they are (a test may hold a bare CR on purpose)
            text = open(os.path.join(ROOT, src), encoding="utf-8", newline="").read()
        except UnicodeDecodeError:
            continue
        if not (WORD_RX.search(text) or any(n in text for n in NEEDLES)):
            continue
        new = rewrite(f, text, allow, RULES, stats=True)
        if new != text:
            again = rewrite(f, new, allow, RULES)
            if again != new:
                not_idem.append(f)
            changes[f] = (src, text, new)
            ar = vocab.area(f)
            per_area[ar]["files"] += 1
            per_area[ar]["lines"] += changed_lines(text, new)
        for n, line in enumerate(new.splitlines(), 1):
            if WORD_RX.search(line):
                residual.append((f, n, line, residual_class(f, line, allow)))
    for f in new_files:
        if WORD_RX.search(f) and not allow.reason(f, f, True):
            residual.append((f, 0, f, "outstanding"))

    # report
    print("compliance-g1: " + ("applying" if args.apply else "dry run (nothing written; --apply to write)"))
    print(f"paths moved: {len(plan)}; files rewritten: {len(changes)}; "
          f"lines changed: {sum(v['lines'] for v in per_area.values())}")
    res_area = defaultdict(lambda: defaultdict(int))
    for f, n, line, cls in residual:
        res_area[vocab.area(f)][cls] += 1
    classes = ["outstanding", "hand", "generated", "pending g2", "pending g3", "pending g4", "pending pr31",
               "third-party", "history", "domain", "migration"]
    hdr = "area".ljust(24) + "paths".rjust(7) + "files".rjust(7) + "lines".rjust(7) + " |" + \
        "".join(c.replace("pending ", "").rjust(12) for c in classes)
    print("\nchanged, and the hits that remain after the rewrite (by class):")
    print(hdr)
    areas = sorted(set(per_area) | set(res_area), key=lambda a: (-(per_area[a]["lines"] + sum(res_area[a].values())), a))
    tot = defaultdict(int)
    for a in areas:
        row = [per_area[a]["paths"], per_area[a]["files"], per_area[a]["lines"]]
        rr = [res_area[a][c] for c in classes]
        for k, v in zip(["paths", "files", "lines"] + classes, row + rr):
            tot[k] += v
        print(a.ljust(24) + "".join(str(v).rjust(7) for v in row) + " |" + "".join(str(v).rjust(12) for v in rr))
    print("total".ljust(24) + "".join(str(tot[k]).rjust(7) for k in ["paths", "files", "lines"]) + " |" +
          "".join(str(tot[c]).rjust(12) for c in classes))
    idle = [r for r in RULES if r.hits == 0 and not r.generic]
    if idle:
        print(f"\n{len(idle)} named rules matched nothing on this tree (already applied, or the text moved):")
        for r in idle:
            print("  " + r.label())
    if not_idem:
        print("\nNOT RE-RUNNABLE: a second pass would change " + ", ".join(not_idem))
    if args.rules:
        print("\nrule matches:")
        for r in RULES:
            print(f"  {r.hits:5d} in {len(r.files_hit):3d} files  {r.label()}")
    if args.verbose:
        for f, (src, a, b) in sorted(changes.items()):
            print(f"  {f}{'' if src == f else ' (from ' + src + ')'}: {changed_lines(a, b)} lines")
    if args.residual:
        with open(args.residual, "w", encoding="utf-8") as w:
            for f, n, line, cls in sorted(residual, key=lambda x: (x[3], x[0], x[1])):
                w.write(f"{cls}\t{f}:{n}\t{line.strip()[:300]}\n")
        print(f"\nresidual hits written to {args.residual}")
    if not_idem:
        return 1
    if not args.apply:
        return 0

    # COMP-20 (b): keep an rce built from the tree before the change, to hash the earlier texts
    old_rce = args.repin
    if args.repin == "build":
        old_rce = build_old_rce()
    # apply: moves first (git mv), then the contents
    for a, b in sorted(plan.items()):
        os.makedirs(os.path.dirname(os.path.join(ROOT, b)), exist_ok=True)
        git("mv", a, b)
    for f, (src, a, b) in changes.items():
        with open(os.path.join(ROOT, f), "w", encoding="utf-8", newline="") as w:
            w.write(b)
    print(f"\nwrote {len(changes)} files, moved {len(plan)} paths")
    old_texts = {f: (src, a) for f, (src, a, b) in changes.items() if f.endswith(".rce")}
    if not args.no_regen:
        regenerate(args)
    if old_rce:
        repin(old_rce, old_texts)
    return 0


# --------------------------------------------------------------------------
# Regeneration with the repository's own tools.

def sh(cmd, cwd=ROOT, env=None, ok=(0,)):
    print(f"  $ {cmd}")
    e = dict(os.environ, **(env or {}))
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=e, capture_output=True, text=True)
    if p.returncode not in ok:
        tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
        print("    failed (status %d):\n      %s" % (p.returncode, "\n      ".join(tail)))
        return False
    return True


def regenerate(args):
    print("\nregenerating:")
    checker = os.path.join(ROOT, "spec", "checker")
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(checker, "target"))
    env = {"CARGO_TARGET_DIR": target}
    rce = os.path.join(target, "release", "rce")
    if not shutil.which("cargo"):
        print("  cargo is not present: RCE-SPEC.md, INDEX.md, spec-data.json, spec-tables.json, spec/traces,"
              " tests/brick/catalog.json and the listing outputs are left for CI")
        built = False
    else:
        built = sh("cargo build -q --release --bin rce --bin rce-check --bin rce-book --bin rce-spec-data --bin rce-conform",
                   cwd=checker, env=env)
    if built:
        # a private copy of the binaries: a shared target directory may be rebuilt
        # from another work tree while this step runs
        import tempfile
        bins = tempfile.mkdtemp(prefix="compliance-g1-bin-")
        for b in ("rce", "rce-check", "rce-book", "rce-spec-data", "rce-conform"):
            shutil.copy2(os.path.join(target, "release", b), os.path.join(bins, b))
        rce = os.path.join(bins, "rce")
        # scripts/regen-spec.sh, with the copied binaries
        sh(f"{bins}/rce-book build")
        sh(f"{bins}/rce-spec-data")
        sh("cargo test -q --release --test traces", cwd=checker, env=dict(env, RCE_BLESS_TRACES="1"))
        sh("cargo test -q --release --test brick", cwd=checker, env=dict(env, RCE_BLESS="1"))
        sh("book/scripts/check-listings.sh --bless", env=dict(env, RCE=rce), ok=(0, 1))
        sh("programs/scripts/check.sh --bless", env=dict(env, RCE=rce), ok=(0, 1))
    if shutil.which("tree-sitter"):
        sh("tree-sitter generate --js-runtime native", cwd=os.path.join(ROOT, "tree-sitter"))
        print("  tree-sitter/tree-sitter-rce.wasm (a WebAssembly build) is left for CI")
        if built:
            sh("python3 tools/ts_crosscheck.py ../tree-sitter", cwd=os.path.join(ROOT, "conformance"),
               env={"RCE_CHECK": os.path.join(bins, "rce-check")}, ok=(0, 1))
    else:
        print("  the tree-sitter CLI is not present: tree-sitter/src is left for CI (tree-sitter generate --js-runtime native)")
    print("  spec/checker-wasm/pkg (wasm-pack) is left for playground/build.sh")
    # conformance/clauses.json is not extracted again here: on main it is the
    # extraction of an earlier draft, and its clause ids move with the sentences
    # the hand rewording changes (clause-repin.tsv, 10); the rules rename its headings
    sh("python3 tools/validate.py", cwd=os.path.join(ROOT, "conformance"), ok=(0, 1))
    if args.conformance and built:
        cenv = dict(env, RCE=rce, RCE_CONFORM=os.path.join(bins, "rce-conform"))
        jobs = os.environ.get("CARGO_BUILD_JOBS", "2")
        sh(f"python3 harness/run.py --adapter adapters/reference.json --jobs {jobs} --quiet",
           cwd=os.path.join(ROOT, "conformance"), env=cenv, ok=(0, 1))
        sh("python3 tools/coverage.py reports/results-reference.json", cwd=os.path.join(ROOT, "conformance"))
        sh("python3 tools/parity.py", cwd=os.path.join(ROOT, "conformance"))
    else:
        print("  conformance results: the test ids in conformance/reports are renamed with the tests;"
              " rerun with --conformance to regenerate results-reference.json, coverage and parity")


def build_old_rce():
    """Build `rce` from the tree as it is before the change and keep a copy."""
    import tempfile
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "spec", "checker", "target"))
    if not sh("cargo build -q --release --bin rce", cwd=os.path.join(ROOT, "spec", "checker"), env={"CARGO_TARGET_DIR": target}):
        sys.exit("compliance-g1: could not build rce from the tree before the change")
    keep = os.path.join(tempfile.mkdtemp(prefix="compliance-g1-"), "rce")
    shutil.copy2(os.path.join(target, "release", "rce"), keep)
    return keep


def program_hash(rce, path):
    p = subprocess.run([rce, "hash", path], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.rstrip().endswith("  program"):
            return line.split()[0]
    return "-"


def repin(old_rce, old_texts):
    """COMP-20 (b): earlier program hash -> new program hash, per rewritten .rce file."""
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "spec", "checker", "target"))
    import tempfile
    sh("cargo build -q --release --bin rce", cwd=os.path.join(ROOT, "spec", "checker"), env={"CARGO_TARGET_DIR": target})
    new_rce = os.path.join(tempfile.mkdtemp(prefix="compliance-g1-"), "rce")
    shutil.copy2(os.path.join(target, "release", "rce"), new_rce)
    rows = []
    for f, (src, old) in sorted(old_texts.items()):
        # the earlier text beside the file, so that the modules it uses resolve
        tmp = os.path.join(ROOT, os.path.dirname(f), ".compliance-g1-earlier.rce")
        with open(tmp, "w", encoding="utf-8", newline="") as w:
            w.write(old)
        ha, hb = program_hash(old_rce, tmp), program_hash(new_rce, os.path.join(ROOT, f))
        os.remove(tmp)
        if ha != hb and "-" not in (ha, hb):
            rows.append(f"{src}\t{f}\t{ha}\t{hb}")
    out = os.path.join(ROOT, "spec", "proposals", "compliance-repin.tsv")
    with open(out, "w", encoding="utf-8") as w:
        w.write("# COMP-20 (b): program hashes before and after group 1 of the compliance migration.\n")
        w.write("# One row per program whose text this group changed: earlier path, path, earlier program hash, program hash.\n")
        w.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"  wrote {out}: {len(rows)} programs whose hash changed")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-regen", action="store_true")
    ap.add_argument("--conformance", action="store_true")
    ap.add_argument("--repin", metavar="OLD_RCE")
    ap.add_argument("--residual", metavar="FILE")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.apply and git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("compliance-g1: the work tree has uncommitted changes; commit or stash them first")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
