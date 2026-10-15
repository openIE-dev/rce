# Subscriptions over a live PostgreSQL table: random transactions are
# decoded with wal2json (format 2, REPLICA IDENTITY FULL), and for each
# subscription query the rows to push are computed by the server itself:
# (SELECT .. WHERE q on the new table) EXCEPT (.. on the old table) as
# inserts, the reverse as deletes. Writes vectors/data/sync/pg_subscriptions.json.
# Run with /workspace/.venv-pg/bin/python (psycopg2).
import json, os, random, sys
import psycopg2
DSN = os.environ.get("PGDSN", "host=/tmp port=55432 user=postgres dbname=postgres")
out = sys.argv[1]
c = psycopg2.connect(DSN); c.autocommit = True
q = c.cursor()
q.execute("select version()"); ver = q.fetchone()[0]
q.execute("select extversion from pg_extension where extname='wal2json'")
q.execute("drop table if exists sub_t, sub_old")
q.execute("create table sub_t (id int primary key, a int, b text)")
q.execute("alter table sub_t replica identity full")
q.execute("select pg_drop_replication_slot(slot_name) from pg_replication_slots where slot_name='sub_slot'")
q.execute("select pg_create_logical_replication_slot('sub_slot', 'wal2json')")
QUERIES = [
    [], [["a", ">", 3]], [["a", "<=", 0]], [["b", "=", "x"]], [["b", "<", "m"], ["a", "!=", 2]],
    [["id", ">=", 5], ["a", "<", 6]], [["b", ">", "B"]],
]
def sql(qs):
    if not qs: return "true"
    return " and ".join('%s %s %s' % (k, op, ("'%s' collate \"C\"" % v) if isinstance(v, str) else v) for k, op, v in qs)
def rows():
    q.execute("select id, a, b from sub_t order by id")
    return [{"id": r[0], "a": r[1], "b": r[2]} for r in q.fetchall()]
rng = random.Random(6262)
cases = []
for n in range(30):
    before = rows()
    q.execute("drop table if exists sub_old"); q.execute("create table sub_old as select * from sub_t")
    q.execute("begin")
    for _ in range(rng.randrange(1, 6)):
        r = rng.random(); i = rng.randrange(1, 10)
        a = rng.choice([None, -1, 0, 2, 3, 4, 7]); b = rng.choice([None, "x", "y", "B", "a", "zz", "é", ""])
        if r < 0.5: q.execute("insert into sub_t values (%s,%s,%s) on conflict (id) do update set a=excluded.a, b=excluded.b", (i, a, b))
        elif r < 0.75: q.execute("update sub_t set a=%s where id=%s", (a, i))
        else: q.execute("delete from sub_t where id=%s", (i,))
    q.execute("commit")
    q.execute("select data from pg_logical_slot_get_changes('sub_slot', null, null, 'format-version', '2', 'include-types', 'false')")
    ch = []
    for (d,) in q.fetchall():
        j = json.loads(d)
        if j.get("table") != "sub_t": continue
        cv = lambda cols: {x["name"]: x["value"] for x in cols} if cols else None
        ch.append({"action": j["action"], "old": cv(j.get("identity")), "new": cv(j.get("columns"))})
    subs = []
    for qs in QUERIES:
        w = sql(qs)
        q.execute("select id, a, b from (select * from sub_t where %s except select * from sub_old where %s) s order by id" % (w, w))
        ins = [{"id": r[0], "a": r[1], "b": r[2]} for r in q.fetchall()]
        q.execute("select id, a, b from (select * from sub_old where %s except select * from sub_t where %s) s order by id" % (w, w))
        dels = [{"id": r[0], "a": r[1], "b": r[2]} for r in q.fetchall()]
        subs.append({"query": qs, "where": w, "deletes": dels, "inserts": ins})
    cases.append({"before": before, "changes": ch, "after": rows(), "subscriptions": subs})
q.execute("select pg_drop_replication_slot('sub_slot')")
json.dump({"generator": "scripts/gen/sync_pg.py", "server": ver, "plugin": "wal2json 2.6 format-version 2", "seed": 6262, "cases": cases}, open(out, "w"), indent=1, ensure_ascii=False)
