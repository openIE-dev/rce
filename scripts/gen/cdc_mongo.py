"""Record change stream events from a live mongod (single-node replica
set). Events are saved as canonical Extended JSON v2 (bson.json_util).
Output: vectors/data/mongodb/live_events.json."""
import json, os, sys
from pymongo import MongoClient
from bson import json_util, ObjectId, Decimal128, Int64, Binary, Timestamp
from bson.json_util import JSONOptions, JSONMode
import datetime

URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27099/?directConnection=true")
OUT = sys.argv[1] if len(sys.argv) > 1 else "vectors/data/mongodb/live_events.json"
CANON = JSONOptions(json_mode=JSONMode.CANONICAL)

c = MongoClient(URI)
db = c["cdc"]
c.drop_database("cdc")
db.create_collection("users", changeStreamPreAndPostImages={"enabled": True})
db.create_collection("plain")
cs = db.watch(full_document="whenAvailable", full_document_before_change="whenAvailable", show_expanded_events=True)
u, p = db["users"], db["plain"]
u.insert_one({"_id": ObjectId("58a4eb4a30c75625e00d2820"), "name": "alice", "n": Int64(1), "d": Decimal128("1.50"),
              "when": datetime.datetime(2024, 2, 29, 12, 0, tzinfo=datetime.timezone.utc), "tags": ["a", "b"], "bin": Binary(b"\x00\xff", 0)})
u.update_one({"name": "alice"}, {"$set": {"email": "alice@10gen.com", "home.town": "Oxford"}, "$unset": {"bin": ""}, "$inc": {"n": 2}})
u.update_one({"name": "alice"}, {"$push": {"tags": "c"}})
u.update_one({"name": "alice"}, [{"$set": {"tags": {"$slice": ["$tags", 1]}}}])
u.update_one({"name": "alice"}, {"$set": {"arr": [{"k": i, "pad": "x" * 40} for i in range(20)], "a.b": 1}})
u.update_one({"name": "alice"}, [{"$set": {"arr": {"$slice": ["$arr", 15]}}}])
u.update_one({"name": "alice"}, [{"$set": {"arr": {"$concatArrays": [{"$slice": ["$arr", 12]}, [{"k": 99, "pad": "y"}]]}}}])
u.update_one({"name": "alice"}, {"$set": {"arr.3.k": -3}, "$unset": {"arr.4.pad": ""}})
u.replace_one({"name": "alice"}, {"name": "alice", "v": 2.5, "nested": {"x": None, "y": True}})
with c.start_session() as s:
    with s.start_transaction():
        p.insert_one({"_id": 1, "s": "tx one"}, session=s)
        p.insert_one({"_id": 2, "s": "tx two \u00e9\U0001F600"}, session=s)
p.delete_one({"_id": 1})
u.delete_one({"name": "alice"})
p.rename("plain2")
db["plain2"].drop()
c.drop_database("cdc")
evs = []
while True:
    e = cs.try_next()
    if e is None:
        if evs and evs[-1]["operationType"] == "invalidate":
            break
        if len(evs) > 40:
            break
        continue
    evs.append(e)
    if e["operationType"] == "dropDatabase":
        break
out = {"server": c.server_info()["version"], "events": [json_util.dumps(e, json_options=CANON) for e in evs]}
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print(OUT, len(evs), [e["operationType"] for e in evs])
