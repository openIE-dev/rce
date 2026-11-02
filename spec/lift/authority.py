"""The evidence authority of a lift report (Appendix D.8, L5 and L6;
spec/proposals/evidence-authority.md).

The levels, verdicts and each probe's claims are data, in authority.json.
`close` decides one claim from its evidence records; `report` assembles a
lift report (format `rce-lift-report/1`). spec/checker/src/lift_authority.rs
applies the same rule to the committed reports and to the vectors of
authority-vectors.json, without Python.

A record is a JSON object:

    {"claim": ID, "revision": N, "verifier": NAME, "authority": LEVEL,
     "kind": "observed" | "derived" | "inferred",
     "result": "holds" | "differs" | "undecided",
     "bindings": {NAME: SHA256-HEX, ...}, "detail": TEXT}

A digest is the SHA-256 of a file's bytes, as 64 lowercase hex digits.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "authority.json")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def load():
    with open(TABLE, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def malformed(r, t):
    """Why a record cannot be read, or None. Integrity comes first: a record
    that names an unknown level, kind or result, or carries a digest that is
    not 64 lowercase hex digits, fails its claim closed."""
    levels = {l["authority"] for l in t["levels"]}
    if not isinstance(r, dict):
        return "a record is not an object"
    for k in ("claim", "verifier", "authority", "kind", "result"):
        if not isinstance(r.get(k), str) or not r[k]:
            return f"the record has no {k}"
    if not isinstance(r.get("revision"), int) or isinstance(r.get("revision"), bool):
        return "the record has no integer revision"
    if r["authority"] not in levels:
        return f"unknown authority `{r['authority']}`"
    if r["kind"] not in t["kinds"]:
        return f"unknown kind `{r['kind']}`"
    if r["result"] not in t["results"]:
        return f"unknown result `{r['result']}`"
    b = r.get("bindings")
    if not isinstance(b, dict):
        return "the record has no bindings"
    for k, v in sorted(b.items()):
        if not isinstance(v, str) or not HEX.match(v):
            return f"binding `{k}` is not a SHA-256 digest"
    return None


def close(claim, bindings, records, t, deps=None):
    """(verdict, reason) of one claim (L6).

    claim: {"id", "revision", "requires", "binds", "depends"?}
    bindings: the digests the report states now, name -> hex or None
    records: every evidence record offered (records for other claims are ignored)
    deps: verdicts of the claims named in claim["depends"]
    """
    level = {l["authority"]: l for l in t["levels"]}
    mine = [r for r in records if isinstance(r, dict) and r.get("claim") == claim["id"]]
    # 1. integrity, then digest binding: both fail closed
    for r in mine:
        why = malformed(r, t)
        if why:
            return "failed", f"malformed evidence: {why}"
    live = [r for r in mine if r["revision"] == claim["revision"]]
    for r in live:
        for k in claim["binds"]:
            if r["bindings"].get(k) is None:
                continue
            if bindings.get(k) is None or r["bindings"][k] != bindings[k]:
                return "failed", f"digest mismatch: {r['verifier']} is bound to another {k}"
    # 2. an applicable record that differs fails the claim; no other record hides it
    for r in live:
        if r["result"] == "differs":
            return "failed", f"{r['verifier']}: differs" + (f": {r['detail']}" if r.get("detail") else "")
    # 3. what keeps a claim open
    for d in claim.get("depends", []):
        if (deps or {}).get(d) != "closed":
            return "open", f"dependency {d} is not closed"
    if not mine:
        return "open", "no evidence"
    if not live:
        return "open", f"stale revision: no record answers revision {claim['revision']}"
    seen = set()
    for r in live:
        if r["verifier"] in seen:
            return "open", f"duplicate evidence from {r['verifier']}"
        seen.add(r["verifier"])
    # 4. authority: a record closes only at its level, of its level's kind, bound to every digest
    for r in live:
        lv = level[r["authority"]]
        if (r["result"] == "holds" and claim["requires"] in lv["closes"] and r["kind"] == lv["kind"]
                and all(r["bindings"].get(k) is not None for k in claim["binds"])):
            return "closed", f"{r['verifier']}: {r['authority']}"
    for r in live:
        if r["result"] == "undecided":
            return "undecided", f"{r['verifier']}: undecided" + (f": {r['detail']}" if r.get("detail") else "")
    for r in live:
        if r["result"] == "holds" and r["kind"] == "inferred":
            return "undecided", f"{r['verifier']}: inferred evidence never closes a claim"
    for r in live:
        if r["result"] == "holds" and r["kind"] != level[r["authority"]]["kind"]:
            return "undecided", f"{r['verifier']}: {r['kind']} evidence is not promoted to {r['authority']}"
    for r in live:
        if r["result"] == "holds" and claim["requires"] in level[r["authority"]]["closes"]:
            missing = [k for k in claim["binds"] if r["bindings"].get(k) is None]
            return "open", f"{r['verifier']} is not bound to {', '.join(missing)}"
    return "open", f"no evidence of authority {claim['requires']}: " + ", ".join(sorted({r["authority"] for r in live}))


def report(lift, file, bindings, records, t=None):
    """The lift report (format rce-lift-report/1) for the claims of `lift`:
    every claim with its required authority, the digests it is bound to, its
    evidence and its verdict. `bindings` maps a claim id to its digests."""
    t = t or load()
    claims, verdicts = [], {}
    for c in t["claims"][lift]:
        b = {k: bindings[c["id"]].get(k) for k in c["binds"]}
        ev = [r for r in records if r.get("claim") == c["id"]]
        v, why = close(c, b, ev, t, verdicts)
        verdicts[c["id"]] = v
        claims.append({"id": c["id"], "revision": c["revision"], "requires": c["requires"],
                       "statement": c["statement"], "bindings": b, "evidence": ev,
                       "verdict": v, "reason": why})
    status = "failed" if "failed" in verdicts.values() else "reported"
    return {"format": "rce-lift-report/1", "lift": lift, "file": file, "claims": claims, "status": status}


def dumps(rep):
    return json.dumps(rep, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def record(claim, verifier, authority, kind, result, bindings, detail=""):
    t = load()
    rev = next(c["revision"] for l in t["claims"].values() for c in l if c["id"] == claim)
    return {"claim": claim, "revision": rev, "verifier": verifier, "authority": authority, "kind": kind,
            "result": result, "bindings": bindings, "detail": detail}
