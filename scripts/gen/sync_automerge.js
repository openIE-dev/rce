// Concurrent map edits on three Automerge 2 replicas, merged in random
// order. Records every op (id counter@actor, key, set/del, value, pred)
// from the decoded change history, shuffled, with Automerge's merged
// document and getConflicts per key. Writes
// vectors/data/sync/automerge_map.json. NODE_PATH must reach @automerge/automerge.
const A = require('@automerge/automerge');
const fs = require('fs');
const path = require('path');
let s = 4242;
const rnd = (n) => { s = (s * 48271) % 2147483647; return s % n; };
const KEYS = ['k1', 'k2', 'k3', 'k4'];
const VALS = [1, 2, -7, 'a', 'b', '', true, false, null];
const sessions = [];
for (let n = 0; n < 60; n++) {
  let rs = ['aa', 'bb', 'cc'].map((actor) => A.init({ actor }));
  for (let step = rnd(12) + 2; step > 0; step--) {
    const i = rnd(3);
    if (rnd(4) === 0) { const j = rnd(3); if (j !== i) rs[i] = A.merge(rs[i], rs[j]); continue; }
    rs[i] = A.change(rs[i], (d) => {
      for (let k = rnd(2) + 1; k > 0; k--) {
        const key = KEYS[rnd(KEYS.length)];
        if (rnd(4) === 0 && key in d) delete d[key]; else d[key] = VALS[rnd(VALS.length)];
      }
    });
  }
  let m = A.clone(rs[0], { actor: 'dd' });
  m = A.merge(m, rs[2]); m = A.merge(m, rs[1]);
  const ops = [];
  for (const c of A.getAllChanges(m)) {
    const ch = A.decodeChange(c);
    ch.ops.forEach((o, k) => ops.push({ id: (ch.startOp + k) + '@' + ch.actor, key: o.key, action: o.action, value: o.action === 'set' ? o.value : null, pred: o.pred }));
  }
  for (let i = ops.length - 1; i > 0; i--) { const j = rnd(i + 1); [ops[i], ops[j]] = [ops[j], ops[i]]; }
  const conflicts = {};
  for (const k of KEYS) { const c = A.getConflicts(m, k); if (c) conflicts[k] = Object.keys(c).sort(); }
  sessions.push({ ops, doc: A.toJS(m), conflicts });
}
const pj = JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve('@automerge/automerge')), '../../package.json')));
fs.writeFileSync(process.argv[2], JSON.stringify({ generator: 'scripts/gen/sync_automerge.js', reference: '@automerge/automerge ' + pj.version, seed: 4242, sessions }, null, 1));
