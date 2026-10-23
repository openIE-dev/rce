// Random JSON documents and patches run through fast-json-patch 3.1.1
// (Starcounter-Jack/JSON-Patch): compare(a, b) and applyPatch(doc, ops,
// validate=true). Writes vectors/data/sync/fast_json_patch.json.
// NODE_PATH must reach fast-json-patch.
const jp = require('fast-json-patch');
const fs = require('fs');
let s = 20261015;
const rnd = (n) => { s = (s * 48271) % 2147483647; return s % n; };
const KEYS = ['a', 'b', 'c', 'x/y', 'm~n', '', '0'];
function val(d) {
  const r = rnd(d > 2 ? 4 : 7);
  if (r === 0) return rnd(100) - 50;
  if (r === 1) return ['s', 't', 'é', ''][rnd(4)];
  if (r === 2) return [true, false, null][rnd(3)];
  if (r === 3) return rnd(3) === 0 ? 1.5 : 7;
  if (r === 4) { const o = {}; for (let i = rnd(4); i > 0; i--) o[KEYS[rnd(KEYS.length)]] = val(d + 1); return o; }
  const a = []; for (let i = rnd(4); i > 0; i--) a.push(val(d + 1)); return a;
}
function mutate(v, d) {
  if (d > 3 || v === null || typeof v !== 'object') return rnd(3) === 0 ? val(d) : v;
  if (Array.isArray(v)) {
    const a = v.map((x) => mutate(x, d + 1));
    if (rnd(3) === 0) a.splice(rnd(a.length + 1), 0, val(d + 1));
    if (a.length && rnd(3) === 0) a.splice(rnd(a.length), 1);
    return a;
  }
  const o = {};
  for (const k of Object.keys(v)) if (rnd(5)) o[k] = mutate(v[k], d + 1);
  if (rnd(3) === 0) o[KEYS[rnd(KEYS.length)]] = val(d + 1);
  return o;
}
function paths(v, at, out) {
  out.push(at);
  if (v && typeof v === 'object') for (const k of Object.keys(v)) paths(v[k], at + '/' + String(k).replace(/~/g, '~0').replace(/\//g, '~1'), out);
  return out;
}
function randop(doc) {
  const ps = paths(doc, '', []);
  const p = () => (rnd(6) === 0 ? ['/nope', '/a/b/c', '/-', '/01', '/~2'][rnd(5)] : ps[rnd(ps.length)]);
  const op = ['add', 'remove', 'replace', 'move', 'copy', 'test'][rnd(6)];
  const o = { op, path: p() };
  if (op === 'add' && rnd(3) === 0 && Array.isArray(doc)) o.path = '/' + (rnd(2) ? '-' : String(rnd(doc.length + 2)));
  if (['add', 'replace', 'test'].includes(op)) o.value = op === 'test' && rnd(2) ? jp.deepClone(jp.getValueByPointer(doc, ps[rnd(ps.length)])) : val(2);
  if (['move', 'copy'].includes(op)) o.from = p();
  return o;
}
const diffs = [], applies = [];
let cyclic = 0;
for (let i = 0; i < 150; i++) {
  const a = { r: val(0), s: val(0) };
  const b = mutate(a, 0);
  const ops = jp.compare(a, b);
  const got = jp.applyPatch(jp.deepClone(a), ops, true, false).newDocument;
  if (JSON.stringify(got) !== JSON.stringify(b)) throw new Error('fjp self-check');
  diffs.push({ a, b, ops });
}
for (let i = 0; i < 200; i++) {
  const doc = { r: val(0), s: val(0) };
  const ops = [];
  let cur = jp.deepClone(doc), result = null, error = null;
  for (let k = rnd(3) + 1; k > 0; k--) ops.push(randop(cur));
  try { result = jp.applyPatch(jp.deepClone(doc), jp.deepClone(ops), true, false).newDocument; } catch (e) { error = e.name + ': ' + (e.message || '').split('\n')[0]; }
  // copy from "" (or an ancestor) leaves fast-json-patch's result
  // sharing structure with itself (a cycle); such cases are dropped.
  try { JSON.stringify(result); } catch (e) { cyclic++; continue; }
  applies.push({ doc, ops, result, error });
}
fs.writeFileSync(process.argv[2], JSON.stringify({ generator: 'scripts/gen/sync_fjp.js', reference: 'fast-json-patch 3.1.1', seed: 20261015, dropped_cyclic: cyclic, diffs, applies }, null, 1));
