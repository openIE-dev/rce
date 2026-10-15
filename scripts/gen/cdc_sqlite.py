"""SQLite session changesets and patchsets recorded with APSW (SQLite's
own sqlite3session/sqlite3changeset_* API), with APSW's iteration of
each as the reference decoding. Output: vectors/data/sqlite/changesets.json."""
import apsw, json, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "vectors/data/sqlite/changesets.json"


def val(v):
    if v is apsw.no_change:
        return {"nochange": True}
    if v is None:
        return None
    if isinstance(v, bool):
        raise TypeError
    if isinstance(v, int):
        return {"i": str(v)}
    if isinstance(v, float):
        return {"f": struct.pack(">d", v).hex()}
    if isinstance(v, str):
        return {"t": v}
    if isinstance(v, (bytes, bytearray)):
        return {"b": bytes(v).hex()}
    raise TypeError(type(v))


def ref(blob):
    out = []
    for ch in apsw.Changeset.iter(blob):
        out.append({"op": ch.op, "table": ch.name, "ncol": ch.column_count, "pk": [i in set(ch.pk_columns) for i in range(ch.column_count)],
                    "indirect": bool(ch.indirect),
                    "old": None if ch.old is None else [val(x) for x in ch.old],
                    "new": None if ch.new is None else [val(x) for x in ch.new]})
    return out


CASES = [
    ("basic", ["create table t(id integer primary key, name text, score real, pic blob)"],
     ["insert into t values (1, 'alice', 1.5, x'00ff')",
      "insert into t values (2, 'bob', null, null)",
      "update t set name = 'al''ice é😀', score = -0.25 where id = 1",
      "delete from t where id = 2"]),
    ("existing", ["create table k(a text, b integer, c, primary key (b, a))",
                  "insert into k values ('x', 1, 'old'), ('y', 2, 9223372036854775807), ('z', 3, 1e300)"],
     ["update k set c = 'new' where b = 1",
      "update k set c = -9223372036854775808 where b = 2",
      "delete from k where b = 3",
      "insert into k values ('w', -1, 3.0e-320)",
      "update k set b = 10 where b = 1"]),
    ("multi", ["create table a(id integer primary key, v)", "create table b(id text primary key, v)",
               "insert into a values (5, 'five')"],
     ["insert into b values ('k', 1)", "update a set v = zeroblob(3) where id = 5", "insert into a values (6, 127)",
      "insert into a values (7, 128)", "insert into a values (8, 32768)", "insert into a values (9, 2147483648)",
      "insert into a values (10, 140737488355328)", "insert into a values (11, -1)", "delete from b"]),
    ("noop", ["create table n(id integer primary key, v)", "insert into n values (1, 'same')"],
     ["update n set v = 'same' where id = 1", "insert into n values (2, 'tmp')", "delete from n where id = 2"]),
]


def main():
    out = {"sqlite": apsw.sqlite_lib_version(), "apsw": apsw.apsw_version(), "cases": []}
    for name, setup, work in CASES:
        con = apsw.Connection(":memory:")
        for s in setup:
            con.execute(s)
        ses = apsw.Session(con, "main")
        ses.attach()
        for s in work:
            con.execute(s)
        cs = ses.changeset()
        ps = ses.patchset()
        out["cases"].append({"name": name, "setup": setup, "work": work, "changeset": cs.hex(), "patchset": ps.hex(),
                             "changeset_ref": ref(cs), "patchset_ref": ref(ps),
                             "inverted": apsw.Changeset.invert(cs).hex(), "inverted_ref": ref(apsw.Changeset.invert(cs))})
    # A hand-made changeset whose column count is a 9-byte varint (256:
    # 0x80 x7, 0x81, 0x00), checked against SQLite's own iterator.
    blob = bytes([0x54]) + bytes([0x80] * 7 + [0x81, 0x00]) + bytes([1] + [0] * 255) + b"w\x00" + bytes([18, 0, 1]) + (7).to_bytes(8, "big") + bytes([5] * 255)
    out["crafted"] = [{"name": "varint9", "changeset": blob.hex(), "changeset_ref": ref(blob)}]
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(OUT, [(c["name"], len(c["changeset_ref"])) for c in out["cases"]])


main()
