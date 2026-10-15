#!/usr/bin/env python3
"""Extract normative clauses from the RCE book (spec/book/NN-*.md).

Three kinds of clause, each with a stable id:
  rule     a named rule defined in a bullet, bold lead or table row
           (W-PAR-1, HC-15, N1, F6, TU-2, ...). The id is the rule name.
  theorem  a row of the theorem table (T1..Tn). The id is the theorem name.
  modal    a sentence with MUST / SHALL / must / shall that is not inside a
           named rule. The id is CH<nn>-M-<8 hex of sha256(normalized text)>,
           so it survives reflowing and moving the sentence, and changes only
           when the sentence's words change.
Usage: extract_clauses.py BOOK_DIR > clauses.json
"""
import hashlib, json, re, sys, pathlib

RULE = r"(?:W-[A-Z]+(?:-[A-Z]+)*-\d+[a-z]?|HC-\d+[a-z]?|TU-\d+|CUR-\d+|S-[A-Z]+-\d+|N-[A-Z]+-\d+|[NFRSPLDE]\d{1,2}[a-z]?)"
DEF_PATTERNS = [
    re.compile(r"^\s*[-*]\s+\*\*(" + RULE + r")\b"),          # - **HC-15 Windows.**
    re.compile(r"^\s*[-*]\s+(" + RULE + r")\s"),              # - W-LEX-1 Not UTF-8
    re.compile(r"^\s*\|\s*\*?\*?(" + RULE + r")\*?\*?\s*\|"),  # | N1 | ...
    re.compile(r"^\s*\*\*(" + RULE + r")\b"),                 # **F6** ...
    re.compile(r";\s+(W-[A-Z]+(?:-[A-Z]+)*-\d+)\s"),                      # W-MOD-1 ...; W-MOD-2 import cycle
    re.compile(r"(W-[A-Z]+(?:-[A-Z]+)*-\d+):\s"),                         # W-PAR-1: the footprints ...
]
RANGE = re.compile(r"\b(W-[A-Z]+-)(\d+) to (?:W-[A-Z]+-)?(\d+)\b")
WMENTION = re.compile(r"\bW-[A-Z]+(?:-[A-Z]+)*-\d+\b")
THEOREM = re.compile(r"^\s*\|\s*(T\d+[a-z]?)\s*\|")
MODAL = re.compile(r"\b(MUST|SHALL|must|shall)\b")

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", s)).strip().lower()

def blocks(lines):
    """Paragraph-ish blocks: a bullet or table row with its continuation lines."""
    cur, start, fence = [], 0, False
    for i, l in enumerate(lines):
        if l.lstrip().startswith("```"):
            fence = not fence
            if cur: yield start, cur; cur = []
            continue
        if fence: continue
        new = (not l.strip()) or re.match(r"^\s*([-*]|\d+\.|\||#)\s", l) or l.startswith("|")
        if new and cur:
            yield start, cur; cur = []
        if l.strip():
            if not cur: start = i + 1
            cur.append(l)
    if cur: yield start, cur

def main(book):
    out, seen = [], {}
    for p in sorted(pathlib.Path(book).glob("[0-9][0-9]-*.md")):
        ch = p.name[:2]
        lines = p.read_text(encoding="utf-8").splitlines()
        section = ""
        for start, blk in blocks(lines):
            text = " ".join(x.strip() for x in blk)
            if blk[0].startswith("#"):
                section = blk[0].lstrip("# ").strip(); continue
            m = THEOREM.match(blk[0])
            ids = []
            if m:
                ids.append((m.group(1), "theorem"))
            else:
                for pat in DEF_PATTERNS:
                    for mm in pat.finditer(blk[0] if pat.pattern.startswith("^") else text):
                        ids.append((mm.group(1), "rule"))
            if ids:
                for cid, kind in ids:
                    if cid in seen:   # first definition wins; later rows are restatements
                        seen[cid]["restated"].append(f"{p.name}:{start}")
                        continue
                    c = {"id": cid, "kind": kind, "normative": True, "chapter": p.name, "line": start,
                         "section": section, "text": text[:600], "restated": []}
                    seen[cid] = c; out.append(c)
                continue
            for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z(`])", text):
                if MODAL.search(sent):
                    h = hashlib.sha256(norm(sent).encode()).hexdigest()[:8]
                    cid = f"CH{ch}-M-{h}"
                    if cid in seen: continue
                    info = (text.lstrip().startswith("|") or re.search(r'"[^"]*\b(must|shall)\b', sent) or re.search(r'\b(must|shall)\b[^"]*"\s*(The law|$)', sent)
                            or re.search(r"arXiv|PLDI|POPL|SOSP|OSDI|et al\.|Raft:", sent)
                            or re.search(r"\b(parses|checks|elaborates|reports)\b.*\b(programs|test|tests)\b", sent))
                    c = {"id": cid, "kind": "modal", "normative": not info, "chapter": p.name, "line": start,
                         "section": section, "text": sent[:600], "restated": []}
                    seen[cid] = c; out.append(c)
    # Second pass: every W- rule named anywhere (or by a range "W-PAR-1 to
    # W-PAR-4") is a clause, located at its first mention, so a rule the
    # book names but never spells out still has to be tested.
    for p in sorted(pathlib.Path(book).glob("[0-9][0-9]-*.md")):
        for i, l in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            names = set(WMENTION.findall(l))
            for a, lo, hi in RANGE.findall(l):
                names |= {f"{a}{k}" for k in range(int(lo), int(hi) + 1)}
            for cid in sorted(names):
                if cid not in seen:
                    c = {"id": cid, "kind": "rule", "normative": True, "chapter": p.name, "line": i,
                         "section": "(named)", "text": l.strip()[:600], "restated": []}
                    seen[cid] = c; out.append(c)
    # Profile of each clause. "language" is what an RCE implementation (checker,
    # evaluator, emitters, host) is certified against. "handler-spec" is the
    # profile of chapter 15, "proposed" marks clauses the book itself labels
    # proposed, "tooling" is about the reference's own commands and test
    # programs, "editorial" is a note addressed to the spec's authors.
    for c in out:
        t = c["text"]
        if c["chapter"].startswith("15-"): c["profile"] = "handler-spec"
        elif "(proposed)" in t: c["profile"] = "proposed"
        elif c["kind"] == "modal" and (c["chapter"].startswith("12-") or re.search(r"reference checker\.|^The draft must", t)):
            c["profile"] = "tooling" if not t.startswith("The draft") else "editorial"
        else: c["profile"] = "language"
        if re.search(r"Datalog and Souffl|GQL selectors", t): c["normative"] = False
    digest = hashlib.sha256(json.dumps([c["id"] for c in out]).encode()).hexdigest()
    json.dump({"format": "rce-clauses/1", "book": ("crates/rce/spec/book" if str(book).rstrip("/").endswith("crates/rce/spec/book") else book.name), "id_set_sha256": digest,
               "clauses": out}, sys.stdout, indent=1, ensure_ascii=False)
    print()

if __name__ == "__main__":
    main(sys.argv[1])
