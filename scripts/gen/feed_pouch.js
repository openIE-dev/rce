// Runs PouchDB 9.0.0's checkpointer (pouchdb-checkpointer) on seeded
// random replication histories: compareReplicationLogs (ported from
// CouchDB couch_replicator) and updateCheckpoint (history update).
const fs = require('fs');
const src = fs.readFileSync(require.resolve('pouchdb-checkpointer/lib/index.js'), 'utf8')
  + '\nmodule.exports.__cmp = compareReplicationLogs; module.exports.__upd = updateCheckpoint;';
const m = new module.constructor(); m.paths = module.paths; m._compile(src, 'cp.js');
const cmp = m.exports.__cmp, upd = m.exports.__upd;
let seed = 5959;
function rnd(n) { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; }
function hist(pool) {
  const h = []; const n = rnd(6);
  const used = new Set();
  for (let i = 0; i < n; i++) { const s = pool[rnd(pool.length)]; if (used.has(s)) continue; used.add(s); h.push({ session_id: s, last_seq: String(rnd(500)) }); }
  return h;
}
const out = { compare: [], update: [] };
for (let i = 0; i < 200; i++) {
  const pool = ['s' + rnd(4), 's' + rnd(9), 's' + rnd(9), 'a', 'b'];
  const sh = hist(pool), th = hist(pool);
  const src = { session_id: sh.length && rnd(3) ? sh[0].session_id : 'x' + rnd(3), last_seq: sh.length ? sh[0].last_seq : '0', history: sh };
  const tgt = { session_id: th.length && rnd(3) ? th[0].session_id : 'x' + rnd(3), last_seq: th.length ? th[0].last_seq : '0', history: th };
  const r = cmp(src, tgt);
  out.compare.push({ src, tgt, last_seq: String(r.last_seq) });
}
(async () => {
  for (let i = 0; i < 120; i++) {
    const pool = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    const h = hist(pool);
    const doc = rnd(5) ? { _id: 'ck', session_id: h.length ? h[0].session_id : 'z', last_seq: h.length ? h[0].last_seq : '0', history: h } : null;
    const store = {};
    if (doc) store.ck = JSON.parse(JSON.stringify(doc));
    const db = { adapter: 'memory', get: async (id) => { if (store[id]) return JSON.parse(JSON.stringify(store[id])); const e = new Error('nf'); e.status = 404; throw e; }, put: async (d) => { store[d._id] = d; return { ok: true }; } };
    const session = pool[rnd(pool.length)];
    const seq = rnd(4) ? String(rnd(600)) : (doc ? doc.last_seq : '0');
    await upd(db, 'ck', seq, session, { cancelled: false });
    const after = store.ck || null;
    out.update.push({ before: doc, session, seq, after: after ? { session_id: after.session_id, last_seq: after.last_seq, history: after.history } : null });
  }
  fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 0));
  console.log(out.compare.length, out.update.length);
})();
