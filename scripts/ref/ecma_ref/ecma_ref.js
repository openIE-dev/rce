// Reads JSON lines {setup, expr}; prints JSON lines with the result
// "typeof:String(v)", "err" for a thrown error, or "skip" for values
// outside the integer subset (non-integer or unsafe numbers).
const vm = require('vm');
const rl = require('readline').createInterface({ input: process.stdin });
function bad(v) { return typeof v === 'number' && (!Number.isSafeInteger(v) && !Number.isNaN(v) || Object.is(v, -0)); }
function walk(v, seen) {
  if (bad(v)) return true;
  if (v && typeof v === 'object' && !seen.has(v)) { seen.add(v); for (const k of Object.keys(v)) if (walk(v[k], seen)) return true; }
  return false;
}
rl.on('line', (l) => {
  const { setup, expr } = JSON.parse(l);
  const ctx = vm.createContext({});
  let out;
  try {
    vm.runInContext(setup, ctx);
    const v = vm.runInContext('(' + expr + '\n)', ctx);
    if (walk(v, new Set()) || typeof v === 'function') out = 'skip';
    else out = typeof v + ':' + (v === null ? 'null' : String(v));
  } catch (e) { out = 'err'; }
  console.log(JSON.stringify(out));
});
