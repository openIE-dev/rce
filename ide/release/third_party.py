# List every third-party crate linked into rce-ide and rce-mcp with its declared license.
import json, subprocess
m = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1"], text=True))
res = {n["id"]: n for n in m["resolve"]["nodes"]}
ws = set(m["workspace_members"])
roots = [i for i in ws if "/crates/rce-ide#" in i or "/crates/rce-mcp#" in i]
seen, st = set(), list(roots)
while st:
    x = st.pop()
    if x in seen: continue
    seen.add(x)
    for d in res[x]["deps"]:
        if any(k["kind"] is None for k in d["dep_kinds"]): st.append(d["pkg"])
print("Third-party crates in this release (name version: declared license)\n")
for p in sorted(m["packages"], key=lambda p: (p["name"], p["version"])):
    if p["id"] in seen and p["id"] not in ws:
        print(f'{p["name"]} {p["version"]}: {p["license"] or "NO LICENSE DECLARED"}')
