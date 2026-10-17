#!/usr/bin/env python3
"""Golden vectors for canonical JSON (spec 6.1.1), with an encoder written
from the text of 6.1.1 alone (not from the Rust source).

  scripts/canonical-json-vectors.py --check      verify spec/data/canonical-json-vectors.json
  scripts/canonical-json-vectors.py --generate   rewrite it from the table below

Every vector is (id, note, value, canonical). `canonical` is typed by hand
here, the encoder must reproduce it, and only then is its SHA-256 written.
A value is a tagged list: ["null"], ["bool", b], ["int", "decimal"],
["float", "0x<16 hex digits of the binary64>"], ["text", s], ["list", [v...]],
["rec", [[key, v]...]] (keys in any order), ["map", [[k, v]...]] (pairs in any
order; int or text keys only).
"""
import hashlib, json, math, pathlib, struct, sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "spec" / "data" / "canonical-json-vectors.json"


def float_text(bits):
    x = struct.unpack(">d", struct.pack(">Q", bits))[0]
    assert math.isfinite(x), "NaN and infinities are not Form values"
    if x == 0.0:
        return "-0.0" if bits >> 63 else "0.0"
    sign = "-" if x < 0 else ""
    a = abs(x)
    # shortest round-trip digits and decimal exponent, from repr (shortest)
    r = repr(a)
    if "e" in r:
        m, e = r.split("e")
        e = int(e)
    else:
        m, e = r, 0
    ip, _, fp = m.partition(".")
    digits = (ip + fp).lstrip("0")
    # exponent of the first digit: position of the point relative to digits
    lead_zeros = len(ip + fp) - len((ip + fp).lstrip("0"))
    ex = e + len(ip) - 1 - lead_zeros
    digits = digits.rstrip("0") or "0"
    if 1e-4 <= a < 1e16:
        p = ex + 1
        if p <= 0:
            body = "0." + "0" * (-p) + digits
        elif p >= len(digits):
            body = digits + "0" * (p - len(digits)) + ".0"
        else:
            body = digits[:p] + "." + digits[p:]
        return sign + body
    mant = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{mant}e{ex}"


def text(s):
    out = ['"']
    for c in s:
        o = ord(c)
        if c == '"': out.append('\\"')
        elif c == "\\": out.append("\\\\")
        elif c == "\n": out.append("\\n")
        elif c == "\r": out.append("\\r")
        elif c == "\t": out.append("\\t")
        elif o < 0x20: out.append("\\u%04x" % o)
        else: out.append(c)
    out.append('"')
    return "".join(out)


def codepoints(s):
    return [ord(c) for c in s]


def order_key(v):
    # canonical order of 6.1: null < bool < number < text < list < record
    t = v[0]
    if t == "null": return (0,)
    if t == "bool": return (1, v[1])
    if t == "int": return (2, int(v[1]))
    if t == "text": return (3, codepoints(v[1]))
    raise ValueError("map keys: int or text only in these vectors")


def enc(v):
    t = v[0]
    if t == "null": return "null"
    if t == "bool": return "true" if v[1] else "false"
    if t == "int":
        assert -(2**63) <= int(v[1]) < 2**64
        return str(int(v[1]))
    if t == "float": return float_text(int(v[1], 16))
    if t == "text": return text(v[1])
    if t == "list": return "[" + ",".join(enc(x) for x in v[1]) + "]"
    if t == "rec":
        keys = [k for k, _ in v[1]]
        assert len(set(keys)) == len(keys)
        fs = sorted(v[1], key=lambda kv: codepoints(kv[0]))  # code point order
        return "{" + ",".join(text(k) + ":" + enc(x) for k, x in fs) + "}"
    if t == "map":
        ps = sorted(v[1], key=lambda kv: order_key(kv[0]))
        return "[" + ",".join("[" + enc(k) + "," + enc(x) + "]" for k, x in ps) + "]"
    raise ValueError(t)


def f(x):
    return ["float", "0x%016x" % struct.unpack(">Q", struct.pack(">d", x))[0]]


def fb(bits):
    return ["float", "0x%016x" % bits]


I = lambda n: ["int", str(n)]
T = lambda s: ["text", s]
N = ["null"]

SH = hashlib.sha256(b'{"Main":{"n":0}}').hexdigest()
V = [
 ("null", "literal", N, "null"),
 ("true", "literal", ["bool", True], "true"),
 ("false", "literal", ["bool", False], "false"),
 ("int-0", "integer", I(0), "0"),
 ("int-1", "integer", I(1), "1"),
 ("int-neg-1", "integer", I(-1), "-1"),
 ("int-i64-min", "least i64", I(-2**63), "-9223372036854775808"),
 ("int-i64-max", "greatest i64", I(2**63 - 1), "9223372036854775807"),
 ("int-u64-max", "greatest u64 (6.3 N2); JCS would need a string, see 6.1.1", I(2**64 - 1), "18446744073709551615"),
 ("int-2p53-plus-1", "exact above 2^53; an ECMAScript Number cannot hold it", I(2**53 + 1), "9007199254740993"),
 ("float-zero", "positive zero keeps a fractional digit", f(0.0), "0.0"),
 ("float-neg-zero", "negative zero keeps its sign (N1, F6); JCS prints 0", fb(0x8000000000000000), "-0.0"),
 ("float-one", "a float with no fractional part stays a float (N1)", f(1.0), "1.0"),
 ("float-ten", "", f(10.0), "10.0"),
 ("float-half", "", f(0.5), "0.5"),
 ("float-neg-2.5", "", f(-2.5), "-2.5"),
 ("float-0.1", "shortest round-trip digits", f(0.1), "0.1"),
 ("float-0.1+0.2", "17 significant digits are needed", f(0.1 + 0.2), "0.30000000000000004"),
 ("float-1e-4", "least magnitude written in decimal", f(1e-4), "0.0001"),
 ("float-below-1e-4", "greatest binary64 below 1e-4 uses exponent form", fb(0x3f1a36e2eb1c432c), "9.999999999999999e-5"),
 ("float-1e-5", "JCS prints 0.00001", f(1e-5), "1e-5"),
 ("float-1e-6", "JCS prints 0.000001 (RFC 8785 Appendix B)", f(1e-6), "1e-6"),
 ("float-1.5e-7", "", f(1.5e-7), "1.5e-7"),
 ("float-1e15", "", f(1e15), "1000000000000000.0"),
 ("float-below-1e16", "greatest binary64 below 1e16 is still decimal", fb(0x4341c37937e07fff), "9999999999999998.0"),
 ("float-1e16", "least magnitude written in exponent form; JCS prints 10000000000000000", f(1e16), "1e16"),
 ("float-1e21", "JCS prints 1e+21 (no plus sign here)", f(1e21), "1e21"),
 ("float-1e30", "JCS prints 1e+30 (RFC 8785 section 3.2.2)", f(1e30), "1e30"),
 ("float-1e300", "", f(1e300), "1e300"),
 ("float-min-subnormal", "5e-324", fb(1), "5e-324"),
 ("float-max", "greatest finite binary64", fb(0x7fefffffffffffff), "1.7976931348623157e308"),
 ("float-neg-max", "", fb(0xffefffffffffffff), "-1.7976931348623157e308"),
 ("float-2p68", "RFC 8785 Appendix B prints 295147905179352830000", fb(0x4430000000000000), "2.9514790517935283e20"),
 ("float-333333333.3333333", "RFC 8785 Appendix B row 41b3de4355555555", fb(0x41b3de4355555555), "333333333.3333333"),
 ("float-1424953923781206.2", "RFC 8785 Appendix B row 43143ff3c1cb0959", fb(0x43143ff3c1cb0959), "1424953923781206.2"),
 ("float-tie-down-1424953923781207.2", "exact tie at .25: two 17-digit candidates equally close, kept digit 2 is even, round down", fb(0x43143ff3c1cb095d), "1424953923781207.2"),
 ("float-tie-up-1424953923781206.8", "exact tie at .75: candidates .7 and .8, the even .8 wins, round up", fb(0x43143ff3c1cb095b), "1424953923781206.8"),
 ("float-tie-up-neg-1424953923781206.8", "negative of the round-up tie (the sign does not change the digits)", fb(0xc3143ff3c1cb095b), "-1424953923781206.8"),
 ("float-tie-down-neg-1424953923781207.2", "negative of the round-down tie", fb(0xc3143ff3c1cb095d), "-1424953923781207.2"),
 ("float-neg-3.33e-6", "RFC 8785 Appendix B row becbf647612f3696; JCS prints -0.0000033333333333333333", fb(0xbecbf647612f3696), "-3.3333333333333333e-6"),
 ("float-123456789.125", "", f(123456789.125), "123456789.125"),
 ("text-empty", "", T(""), '""'),
 ("text-ascii", "", T("abc"), '"abc"'),
 ("text-quote", "", T('a"b'), '"a\\"b"'),
 ("text-backslash", "", T("a\\b"), '"a\\\\b"'),
 ("text-slash", "solidus is not escaped", T("a/b"), '"a/b"'),
 ("text-newline", "", T("\n"), '"\\n"'),
 ("text-tab", "", T("\t"), '"\\t"'),
 ("text-cr", "", T("\r"), '"\\r"'),
 ("text-u0000", "", T("\u0000"), '"\\u0000"'),
 ("text-u0008", "backspace: \\u0008 here, \\b in JCS", T("\u0008"), '"\\u0008"'),
 ("text-u000c", "form feed: \\u000c here, \\f in JCS", T("\u000c"), '"\\u000c"'),
 ("text-u001f", "lowercase hex", T("\u001f"), '"\\u001f"'),
 ("text-u007f", "DEL is not escaped", T("\u007f"), '"\u007f"'),
 ("text-e-acute", "raw UTF-8, no normalisation", T("\u00e9"), '"\u00e9"'),
 ("text-e-combining", "differs from the precomposed form", T("e\u0301"), '"e\u0301"'),
 ("text-euro", "", T("\u20ac"), '"\u20ac"'),
 ("text-u2028", "not escaped", T("\u2028"), '"\u2028"'),
 ("text-emoji", "supplementary plane, raw", T("\U0001F600"), '"\U0001F600"'),
 ("text-rfc8785-sample", "string of RFC 8785 section 3.2.2 (decoded); JCS differs only in U+000F case, none here", T("\u20ac$\u000f\nA'B\"\\\\\"/"), '"\u20ac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"'),
 ("list-empty", "", ["list", []], "[]"),
 ("list-mixed", "order kept, int and float distinct", ["list", [I(1), f(1.0), T("a"), N]], '[1,1.0,"a",null]'),
 ("list-nested", "", ["list", [["list", []], ["list", [I(1)]]]], "[[],[1]]"),
 ("rec-empty", "", ["rec", []], "{}"),
 ("rec-sorted", "keys sorted", ["rec", [["b", I(2)], ["a", I(1)]]], '{"a":1,"b":2}'),
 ("rec-case", "code point order: uppercase before lowercase", ["rec", [["a", I(1)], ["B", I(2)]]], '{"B":2,"a":1}'),
 ("rec-prefix", "shorter prefix first", ["rec", [["ab", I(1)], ["a", I(2)], ["", I(3)]]], '{"":3,"a":2,"ab":1}'),
 ("rec-key-escape", "keys use the text rules", ["rec", [["a\"b", I(1)], ["\n", I(2)]]], '{"\\n":2,"a\\"b":1}'),
 ("rec-variant", "a kind variant (6.1)", ["rec", [["x", I(1)], ["$v", T("V")]]], '{"$v":"V","x":1}'),
 ("rec-int-float", "", ["rec", [["b", I(1)], ["a", f(1.0)]]], '{"a":1.0,"b":1}'),
 ("rec-order-rfc8785-data", "the sorting data of RFC 8785 section 3.2.3: code point order differs from JCS (UTF-16 order) for U+FB33 and U+1F600",
   ["rec", [["\u20ac", T("Euro Sign")], ["\r", T("Carriage Return")], ["\ufb33", T("Hebrew Letter Dalet With Dagesh")],
            ["1", T("One")], ["\U0001F600", T("Emoji: Grinning Face")], ["\u0080", T("Control")], ["\u00f6", T("Latin Small Letter O With Diaeresis")]]],
   '{"\\r":"Carriage Return","1":"One","\u0080":"Control","\u00f6":"Latin Small Letter O With Diaeresis","\u20ac":"Euro Sign","\ufb33":"Hebrew Letter Dalet With Dagesh","\U0001F600":"Emoji: Grinning Face"}'),
 ("rec-nested", "", ["rec", [["z", ["rec", [["b", N], ["a", ["bool", True]]]]], ["y", ["list", [I(-1)]]]]], '{"y":[-1],"z":{"a":true,"b":null}}'),
 ("log-state", "state of one region with one field: the bytes behind a state_hash (6.1)", ["rec", [["Main", ["rec", [["n", I(0)]]]]]], '{"Main":{"n":0}}'),
 ("log-entry-body", "the bytes behind an entry hash (6.1): first entry, prev_hash 64 zeros",
   ["rec", [["state_hash", T(SH)], ["seq", I(0)], ["prev_hash", T("0" * 64)], ["event", ["rec", [["type", T("Inc")], ["payload", ["rec", []]]]]]]],
   '{"event":{"payload":{},"type":"Inc"},"prev_hash":"' + "0" * 64 + '","seq":0,"state_hash":"' + SH + '"}'),
 ("map-int-keys", "a Map is a list of [k, v] pairs in canonical order of k (6.1)", ["map", [[I(3), T("c")], [I(1), T("a")], [I(2), T("b")]]], '[[1,"a"],[2,"b"],[3,"c"]]'),
 ("map-text-keys", "text keys by code point", ["map", [[T("b"), I(2)], [T("B"), I(1)], [T("\u00e9"), I(3)]]], '[["B",1],["b",2],["\u00e9",3]]'),
 ("map-neg-int-keys", "numbers by value", ["map", [[I(1), N], [I(-5), N]]], '[[-5,null],[1,null]]'),
]


def build():
    out = []
    for id_, note, val, canon in V:
        got = enc(val)
        assert got == canon, f"{id_}: encoder gave {got!r}, expected {canon!r}"
        out.append({"id": id_, "note": note, "value": val, "canonical": canon,
                    "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest()})
    return {"format": "rce-canonical-json-vectors/1", "spec": "6.1.1", "vectors": out}


def dump(d):
    # one vector per line, ASCII only so the file has no encoding ambiguity
    head = '{"format":%s,"spec":%s,"vectors":[\n' % (json.dumps(d["format"]), json.dumps(d["spec"]))
    body = ",\n".join(json.dumps(v, ensure_ascii=True, separators=(", ", ": ")) for v in d["vectors"])
    return head + body + "\n]}\n"


def main():
    d = build()
    if "--generate" in sys.argv:
        OUT.write_text(dump(d))
        print("wrote", OUT, len(d["vectors"]), "vectors")
        return
    on_disk = json.loads(OUT.read_text())
    assert on_disk == d, "vectors file differs from the table: run --generate and review the diff"
    for v in on_disk["vectors"]:
        assert enc(v["value"]) == v["canonical"], v["id"]
        assert hashlib.sha256(v["canonical"].encode("utf-8")).hexdigest() == v["sha256"], v["id"]
    print("canonical-json vectors OK:", len(on_disk["vectors"]))


main()
