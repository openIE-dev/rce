// Records PouchDB _changes responses (normal feed; style main_only and
// all_docs; include_docs; conflicts; since) on a scripted workload.
// Run with NODE_PATH pointing at a tree with pouchdb-memory.
// Output: vectors/data/couchdb/pouchdb_changes.json
const PouchDB = require('pouchdb-memory');
const fs = require('fs');
(async () => {
  const db = new PouchDB('cdc');
  const a = await db.put({ _id: 'fresh', n: 1 });
  let u = await db.put({ _id: 'updated', v: 'one' });
  u = await db.put({ _id: 'updated', _rev: u.rev, v: 'two', nested: { x: [1, null, true] } });
  const d = await db.put({ _id: 'deleted', v: 0 });
  await db.remove('deleted', d.rev);
  // a conflict: two branches of the same doc
  await db.bulkDocs([{ _id: 'conf', _rev: '1-aaa', v: 'left' }, { _id: 'conf', _rev: '1-bbb', v: 'right' }], { new_edits: false });
  await db.put({ _id: 'utf', s: 'é😀 "q"\n' });
  const out = { pouchdb: require('pouchdb-memory/package.json').version, cases: [] };
  const opts = [
    {},
    { style: 'all_docs' },
    { include_docs: true },
    { include_docs: true, conflicts: true, style: 'all_docs' },
    { since: 3 },
    { since: 3, limit: 2 },
    { descending: true },
    { doc_ids: ['updated', 'deleted'] },
  ];
  for (const o of opts) {
    const r = await db.changes(o);
    out.cases.push({ opts: o, response: JSON.stringify(r) });
  }
  fs.writeFileSync(process.argv[2] || 'vectors/data/couchdb/pouchdb_changes.json', JSON.stringify(out, null, 1));
  console.log(out.cases.length, 'cases');
})();
