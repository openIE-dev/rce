"""Record logical decoding output from a live PostgreSQL server.

For each case: create tables, a publication and one slot per plugin
(pgoutput proto 1 with messages, wal2json format 1 and 2), run the
workload, then read every slot. Output: vectors/data/postgres/pg_live.json.
Run against a server with wal_level=logical (see vectors/SOURCE.md).
"""
import json, os, sys, psycopg2

DSN = os.environ.get("PGDSN", "host=/tmp port=55432 user=postgres dbname=postgres")
OUT = sys.argv[1] if len(sys.argv) > 1 else "vectors/data/postgres/pg_live.json"

BIG = "x" * 2100 + "y" * 2100  # TOASTed (compression off below)

CASES = [
    ("basic", """
create table t1(id int primary key, name text, ok bool, amt numeric(10,2), note varchar(20));
""", """
insert into t1 values (1,'alice',true,12.50,null);
begin; update t1 set name='al''ice "x"', amt=-3.1 where id=1; insert into t1 values (2,'bob',false,0,'n'); commit;
delete from t1 where id=1;
"""),
    ("types", """
create table ty(id bigint primary key, s smallint, f4 real, f8 double precision, n numeric, b bytea,
  j json, jb jsonb, ts timestamptz, d date, u uuid, arr int[], t text, o oid);
""", """
set timezone='UTC';
insert into ty values (1, -32768, 1.5, 1e300, 123456789012345678901234567890.0001, '\\x00ff10',
  '{"a": [1, 2]}', '{"b": {"c": null}}', '2024-02-29 12:34:56.789+00', '2024-01-01',
  'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{1,2,3}', E'tab\\tnl\\nq"\\\\é😀', 42);
insert into ty values (2, 0, 'NaN', '-Infinity', 'NaN', '', 'null', '[]', null, null, null, '{}', '', 0);
insert into ty values (3, 1, 'Infinity', -0.0, -1e-5, null, '"s"', '1', 'infinity', '-infinity', null, null, null, null);
"""),
    ("identity", """
create table full_t(a int, b text); alter table full_t replica identity full;
create table idx_t(a int not null, b int not null, c text); create unique index idx_t_ab on idx_t(a, b);
alter table idx_t replica identity using index idx_t_ab;
create table pk_t(k int primary key, v text);
""", """
insert into full_t values (1,'one'),(2,null);
update full_t set b='uno' where a=1;
delete from full_t where a=2;
insert into idx_t values (1,2,'x');
update idx_t set c='y';
update idx_t set b=3;
delete from idx_t;
insert into pk_t values (10,'ten');
update pk_t set k=11 where k=10;
update pk_t set v='eleven';
delete from pk_t;
"""),
    ("toast", """
create table big(id int primary key, payload text, n int); alter table big alter column payload set storage external;
""", """
insert into big values (1, '%s', 0);
update big set n = 1 where id = 1;
update big set payload = 'small' where id = 1;
delete from big;
""" % BIG),
    ("truncate", """
create table p(id int primary key); create table c(id int primary key, pid int references p(id));
create table s(id serial primary key, v int);
""", """
insert into p values (1); insert into c values (1,1); insert into s(v) values (5);
truncate p, c;
truncate s restart identity;
truncate p cascade;
"""),
    ("multi", """
create table a(id int primary key, v text); create table b(id int primary key, v text);
""", """
begin;
insert into a values (1,'a1');
insert into b values (1,'b1');
select pg_logical_emit_message(true, 'pfx', 'transactional msg');
update a set v='a2';
delete from b;
commit;
select pg_logical_emit_message(false, 'nontx', E'\\\\x00\\\\x01'::text);
begin; insert into a values (2, null); rollback;
begin; insert into a values (3,'three'); savepoint s1; insert into a values (4,'four'); rollback to s1; insert into a values (5,'five'); commit;
"""),
]


def run(cur, sql):
    for st in [s.strip() for s in sql.split(";\n") if s.strip()]:
        cur.execute(st)


def main():
    out = {"server": None, "cases": []}
    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("select version()")
    out["server"] = cur.fetchone()[0]
    for name, ddl, work in CASES:
        db = "cdc_" + name
        cur.execute(f"drop database if exists {db}")
        cur.execute(f"create database {db}")
        c2 = psycopg2.connect(DSN.replace("dbname=postgres", "dbname=" + db))
        c2.autocommit = True
        k = c2.cursor()
        k.execute("set default_toast_compression = 'pglz'")
        run(k, ddl)
        k.execute("create publication pub for all tables")
        k.execute("select pg_create_logical_replication_slot(%s, 'pgoutput')", (db + "_pg",))
        k.execute("select pg_create_logical_replication_slot(%s, 'wal2json')", (db + "_w2",))
        k.execute("select pg_create_logical_replication_slot(%s, 'wal2json')", (db + "_w1",))
        # psql-style statements, autocommit off for explicit begin blocks
        c2.autocommit = True
        for st in [s.strip() for s in work.split(";\n") if s.strip()]:
            k.execute(st)
        k.execute("select lsn::text, xid::text, encode(data,'hex') from pg_logical_slot_get_binary_changes(%s,null,null,'proto_version','1','publication_names','pub','messages','true')", (db + "_pg",))
        pg = [list(r) for r in k.fetchall()]
        k.execute("select lsn::text, xid::text, data from pg_logical_slot_get_changes(%s,null,null,'format-version','2','include-xids','1','include-lsn','1','include-transaction','1')", (db + "_w2",))
        w2 = [list(r) for r in k.fetchall()]
        k.execute("select lsn::text, xid::text, data from pg_logical_slot_get_changes(%s,null,null,'format-version','1','include-xids','1','include-lsn','1')", (db + "_w1",))
        w1 = [list(r) for r in k.fetchall()]
        for s in (db + "_pg", db + "_w2", db + "_w1"):
            k.execute("select pg_drop_replication_slot(%s)", (s,))
        c2.close()
        out["cases"].append({"name": name, "ddl": ddl, "workload": work, "pgoutput": pg, "wal2json_v2": w2, "wal2json_v1": w1})
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(OUT, sum(len(c["pgoutput"]) for c in out["cases"]), "pgoutput msgs")


main()
