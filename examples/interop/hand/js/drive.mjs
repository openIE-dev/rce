// Run a hand-written program on a stream and print what `rce run` prints.
// The display rules are the part of RCE's display these programs use.
//   node drive.mjs NAME STREAM.json
import { readFileSync } from "node:fs";
import { PROGRAMS, Kept } from "./programs.mjs";
import { decode } from "../../js/rce_host.mjs";

const showFloat = (x) => (Object.is(x, -0) ? "-0" : String(x));
// float keys order as IEEE 754 totalOrder: -0 before 0
const keyCmp = (a, b) => (typeof a === "number" ? a - b || (Object.is(a, -0) ? -1 : Object.is(b, -0) ? 1 : 0) : a < b ? -1 : a > b ? 1 : 0);
function show(v) {
  if (typeof v === "bigint" || typeof v === "boolean") return String(v);
  if (typeof v === "number") return showFloat(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(show).join(", ") + "]";
  if (v instanceof Map) {
    const ps = [...v.entries()].map(([k, x]) => (x && typeof x === "object" && "x" in x ? [x.x, x.n] : [k, x]));
    return "{" + ps.sort((p, q) => keyCmp(p[0], q[0])).map(([k, x]) => show(k) + ": " + show(x)).join(", ") + "}";
  }
  if (v && v.kind) return v.kind;
  throw new TypeError(String(v));
}
const showRegion = (r) => r.$ + " { " + Object.entries(r).filter(([k]) => k !== "$").map(([k, v]) => k + " = " + show(v)).join(", ") + " }";
const showRec = (n, fs) => { const e = Object.entries(fs); return n + (e.length ? "(" + e.map(([k, v]) => k + ": " + show(v)).join(", ") + ")" : ""); };

const p = PROGRAMS[process.argv[2]];
let s = p.init();
const out = ["0 (initial) " + s.map(showRegion).join("; ")];
decode(readFileSync(process.argv[3], "utf8")).forEach((e, i) => {
  const k = i + 1;
  const { type, ...fs } = e;
  const es = showRec(type, fs);
  let r;
  try { r = p.step(s, e); } catch (x) { if (!(x instanceof Kept)) throw x; out.push(k + " " + es + " kept: " + x.r); return; }
  s = r[0];
  out.push(k + " " + es + " " + s.map(showRegion).join("; "), ...r[1].map(([n, f]) => "  emit " + showRec(n, f)));
});
process.stdout.write(out.map((x) => x + "\n").join(""));
