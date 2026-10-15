# Usage: python3 scripts/gen/enc_gen.py > appendix-a.rce, then replace the
# lines after "# RFC 8949 Appendix A" in tests/enc/enc_checks.rce.
# Generates the RFC 8949 Appendix A section of tests/enc/enc_checks.rce.
# Each entry: (diagnostic notation from Appendix A, RCE Cbor value, hex).
# Only items in the enc model (no floats, no ints beyond i64, definite
# lengths) are taken; text items that are non-ASCII are encoded only.
A = [
("0","CInt(v: 0)","00"),("1","CInt(v: 1)","01"),("10","CInt(v: 10)","0a"),("23","CInt(v: 23)","17"),
("24","CInt(v: 24)","1818"),("25","CInt(v: 25)","1819"),("100","CInt(v: 100)","1864"),("1000","CInt(v: 1000)","1903e8"),
("1000000","CInt(v: 1000000)","1a000f4240"),("1000000000000","CInt(v: 1000000000000)","1b000000e8d4a51000"),
("-1","CInt(v: -1)","20"),("-10","CInt(v: -10)","29"),("-100","CInt(v: -100)","3863"),("-1000","CInt(v: -1000)","3903e7"),
("false","CSimple(v: 20)","f4"),("true","CSimple(v: 21)","f5"),("null","CSimple(v: 22)","f6"),("undefined","CSimple(v: 23)","f7"),
("simple(16)","CSimple(v: 16)","f0"),("simple(255)","CSimple(v: 255)","f8ff"),
('0("2013-03-21T20:04:00Z")','CTag(tag: 0, item: Box(CText(t: "2013-03-21T20:04:00Z")))',"c074323031332d30332d32315432303a30343a30305a"),
("1(1363896240)","CTag(tag: 1, item: Box(CInt(v: 1363896240)))","c11a514b67b0"),
("23(h'01020304')","CTag(tag: 23, item: Box(CBytes(b: [1, 2, 3, 4])))","d74401020304"),
("24(h'6449455446')","CTag(tag: 24, item: Box(CBytes(b: [100, 73, 69, 84, 70])))","d818456449455446"),
('32("http://www.example.com")','CTag(tag: 32, item: Box(CText(t: "http://www.example.com")))',"d82076687474703a2f2f7777772e6578616d706c652e636f6d"),
("h''","CBytes(b: [])","40"),("h'01020304'","CBytes(b: [1, 2, 3, 4])","4401020304"),
('""','CText(t: "")',"60"),('"a"','CText(t: "a")',"6161"),('"IETF"','CText(t: "IETF")',"6449455446"),
('"\\"\\\\"','CText(t: "\\"\\\\")',"62225c"),
('"\\u00fc"','CText(t: "\\u{fc}")',"62c3bc"),('"\\u6c34"','CText(t: "\\u{6c34}")',"63e6b0b4"),
('"\\ud800\\udd51"','CText(t: "\\u{10151}")',"64f0908591"),
("[]","CArr(items: [])","80"),("[1, 2, 3]","CArr(items: [CInt(v: 1), CInt(v: 2), CInt(v: 3)])","83010203"),
("[1, [2, 3], [4, 5]]","CArr(items: [CInt(v: 1), CArr(items: [CInt(v: 2), CInt(v: 3)]), CArr(items: [CInt(v: 4), CInt(v: 5)])])","8301820203820405"),
("[1, 2, ..., 25]","CArr(items: [%s])"%", ".join("CInt(v: %d)"%i for i in range(1,26)),"98190102030405060708090a0b0c0d0e0f101112131415161718181819"),
("{}","CMap(keys: [], vals: [])","a0"),
("{1: 2, 3: 4}","CMap(keys: [CInt(v: 1), CInt(v: 3)], vals: [CInt(v: 2), CInt(v: 4)])","a201020304"),
('{"a": 1, "b": [2, 3]}','CMap(keys: [CText(t: "a"), CText(t: "b")], vals: [CInt(v: 1), CArr(items: [CInt(v: 2), CInt(v: 3)])])',"a26161016162820203"),
('["a", {"b": "c"}]','CArr(items: [CText(t: "a"), CMap(keys: [CText(t: "b")], vals: [CText(t: "c")])])',"826161a161626163"),
('{"a": "A", ..., "e": "E"}','CMap(keys: [%s], vals: [%s])'%(", ".join('CText(t: "%s")'%c for c in "abcde"),", ".join('CText(t: "%s")'%c.upper() for c in "abcde")),"a56161614161626142616361436164614461656145"),
]
for d,v,h in A:
    rt = '"\\u' not in d
    fn = "cbor_rt" if rt else "cbor_enc_only"
    print("# %s" % d)
    print('check %s(%s, "%s")' % (fn, v, h))
