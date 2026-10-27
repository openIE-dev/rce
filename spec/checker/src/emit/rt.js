// RCE runtime for emitted JavaScript (spec 10.5). It mirrors the reference
// evaluator value by value: integers are BigInt with range traps, f32 is
// rounded once from f64 (Math.fround), values are never mutated in place.
"use strict";
class Trap extends Error { constructor(m) { super(m); this.m = m; } }
class Fail extends Error { constructor(v) { super("fail"); this.v = v; } }
class Stop extends Error { constructor() { super("stop"); } }
class Ret extends Error { constructor(v) { super("ret"); this.v = v; } }
class Refuse extends Error { constructor(r) { super("refuse"); this.r = r; } }
const trap = (m) => { throw new Trap(m); };
// Fuel (6.7, 10.5): one unit at each point where the reference evaluator
// ticks; each observation starts with full fuel and empty constant caches.
const FUEL_ARGS = (typeof process !== "undefined" && process.argv) ? process.argv : [];
let FUEL_LIMIT = BigInt((FUEL_ARGS.find((a) => a.startsWith("--fuel=")) || "--fuel=50000000").slice(7));
const FUEL_TRACE = FUEL_ARGS.includes("--fuel-trace");
// per worker: every task and par branch is its own worker (own pool)
let FUEL = FUEL_LIMIT;
function fuel() { if (FUEL === 0n) trap("comptime fuel exhausted"); FUEL -= 1n; }
// `bound n` (6.7): active calls of each bounded function, per worker
const DEPTH = new Map();
function depthEnter(k, n) { const d = DEPTH.get(k) || 0; if (BigInt(d) >= n) trap("recursion bound"); DEPTH.set(k, d + 1); }
function depthExit(k) { const d = DEPTH.get(k) || 0; if (d > 0) DEPTH.set(k, d - 1); }
function fuelReset() { DEPTH.clear(); FUEL = FUEL_LIMIT; for (const k of Object.keys(KC)) delete KC[k]; }
function fuelReport() { if (FUEL_TRACE) say("fuel " + (FUEL_LIMIT - FUEL)); }
// `--fuel-batch=n1,n2,..`: every observation once per limit in this process,
// each pass headed by `#fuel-batch n`
function observeAll(f) {
  const b = FUEL_ARGS.find((a) => a.startsWith("--fuel-batch="));
  if (!b) { f(); return; }
  for (const n of b.slice(13).split(",")) { FUEL_LIMIT = BigInt(n); say("#fuel-batch " + n); f(); }
}
class Cancel extends Error { constructor(f) { super("cancel"); this.f = f; } }
const isFlow = (e) => e instanceof Trap || e instanceof Fail || e instanceof Stop || e instanceof Ret || e instanceof Cancel;

const P2 = (n) => 1n << BigInt(n);
const RANGES = { i8: [-128n, 127n], i16: [-32768n, 32767n], i32: [-P2(31), P2(31) - 1n], i64: [-P2(63), P2(63) - 1n],
  u8: [0n, 255n], u16: [0n, 65535n], u32: [0n, P2(32) - 1n], u64: [0n, P2(64) - 1n], usize: [0n, P2(64) - 1n] };
const irange = (t) => RANGES[t] || RANGES.i64;
const I128 = [-P2(127), P2(127) - 1n];

class Int { constructor(v, t) { this.v = v; this.t = t; } }
class F { constructor(v) { this.v = v; } }
class F32 { constructor(v) { this.v = v; } }
class UnitT {}
const UNIT = new UnitT();
class Tup { constructor(xs) { this.xs = xs; } }
class Lst { constructor(xs) { this.xs = xs; } }
// a Set[τ]: elements in canonical order without duplicates (7.1)
class St { constructor(xs) { this.xs = xs; } }
class Mp { constructor(ps) { this.ps = ps; } }
class Opt { constructor(x) { this.x = x; } }
const NONE = new Opt(null);
class Rec { constructor(n, fs) { this.n = n; this.fs = fs; } }
class Err { constructor(e) { this.e = e; } }
class FnV { constructor(n) { this.n = n; } }
class Clo { constructor(f) { this.f = f; } }
class Rng { constructor(a, b, t) { this.a = a; this.b = b; this.t = t; } }
class Gen { constructor(run) { this.run = run; } }
const some = (x) => new Opt(x);
const lit = (v) => new Int(v, "lit");
const isBool = (v) => v === true || v === false;
const isStr = (v) => typeof v === "string";

// ---- formatting (Rust Display / Debug) ----
function digitsOf(s) {
  let m = s, e = 0;
  const k = s.indexOf("e");
  if (k >= 0) { m = s.slice(0, k); e = parseInt(s.slice(k + 1), 10); }
  let d, p;
  const i = m.indexOf(".");
  if (i >= 0) { d = m.slice(0, i) + m.slice(i + 1); p = i + e; } else { d = m; p = m.length + e; }
  let z = 0;
  while (z < d.length - 1 && d[z] === "0") z++;
  d = d.slice(z); p -= z;
  d = d.replace(/0+$/, "") || "0";
  return [d, p];
}
function plain(neg, d, p) {
  let r;
  if (d === "0") r = "0";
  else if (p <= 0) r = "0." + "0".repeat(-p) + d;
  else if (p >= d.length) r = d + "0".repeat(p - d.length);
  else r = d.slice(0, p) + "." + d.slice(p);
  return (neg ? "-" : "") + r;
}
const negz = (x) => x < 0 || Object.is(x, -0);
function fdisplay(x) {
  if (x === 0) return negz(x) ? "-0" : "0";
  const a = Math.abs(x), [d, p] = digitsOf(String(a));
  if (a >= 1e-5 && a < 1e16) return plain(x < 0, d, p);
  const e = p - 1;
  return (x < 0 ? "-" : "") + d[0] + (d.length > 1 ? "." + d.slice(1) : "") + "e" + (e >= 0 ? "+" : "") + e;
}
function fdebug(x) {
  if (x === 0) return negz(x) ? "-0.0" : "0.0";
  const a = Math.abs(x);
  const [d, p] = digitsOf(String(a));
  if (a >= 1e-4 && a < 1e16) { const s = plain(x < 0, d, p); return s.includes(".") ? s : s + ".0"; }
  return (x < 0 ? "-" : "") + d[0] + (d.length > 1 ? "." + d.slice(1) : "") + "e" + (p - 1);
}
function f32digits(a) {
  for (let q = 1; q <= 9; q++) {
    const s = a.toExponential(q - 1);
    if (Math.fround(Number(s)) === a) return digitsOf(s);
  }
  return digitsOf(String(a));
}
function f32display(x) {
  if (x === 0) return negz(x) ? "-0" : "0";
  const [d, p] = f32digits(Math.abs(x));
  return plain(x < 0, d, p);
}
function bitlen(a) { return a === 0n ? 0 : a.toString(2).length; }
function intToF32(n) {
  const a = n < 0n ? -n : n;
  if (a < P2(24)) return Number(n);
  const sh = bitlen(a) - 24;
  let m = a >> BigInt(sh);
  const rem = a - (m << BigInt(sh)), half = 1n << BigInt(sh - 1);
  if (rem > half || (rem === half && (m & 1n) === 1n)) m += 1n;
  let r = Math.fround(Number(m) * Math.pow(2, sh));
  return n < 0n ? -r : r;
}
function rdebug(s) {
  let o = '"';
  for (const c of s) {
    const k = c.codePointAt(0);
    if (c === "\t") o += "\\t"; else if (c === "\r") o += "\\r"; else if (c === "\n") o += "\\n";
    else if (c === "\\") o += "\\\\"; else if (c === '"') o += '\\"'; else if (c === "\0") o += "\\0";
    else if (k < 0x20 || (k >= 0x7f && k <= 0x9f) || (k >= 0x300 && k <= 0x36f)) o += "\\u{" + k.toString(16) + "}";
    else o += c;
  }
  return o + '"';
}
function show(v) {
  if (v === true) return "true";
  if (v === false) return "false";
  if (isStr(v)) return rdebug(v);
  if (v instanceof Int) return v.v.toString();
  if (v instanceof LVar) return "lvar(" + show(v.v) + ")";
  if (v instanceof F) return fdisplay(v.v);
  if (v instanceof UnitT) return "()";
  if (v instanceof Tup) return "(" + v.xs.map(show).join(", ") + ")";
  if (v instanceof Lst || v instanceof St) return "[" + v.xs.map(show).join(", ") + "]";
  if (v instanceof Mp) return "{" + v.ps.map(([a, b]) => show(a) + ": " + show(b)).join(", ") + "}";
  if (v instanceof Opt) return v.x === null ? "none" : "some(" + show(v.x) + ")";
  if (v instanceof Rec) return v.fs.length === 0 ? v.n : v.n + "(" + v.fs.map(([a, b]) => a + ": " + show(b)).join(", ") + ")";
  if (v instanceof Err) return "error " + show(v.e);
  if (v instanceof FnV) return v.n;
  if (v instanceof Clo) return "<fn>";
  if (v instanceof Rng) return v.a + ".." + v.b;
  if (v instanceof F32) return f32display(v.v);
  if (v instanceof Gen) return "<iter>";
  if (v instanceof Chan) return "<chan>";
  if (v instanceof Task) return "<task>";
  if (v instanceof Actor) return "<actor " + v.ty + ">";
  if (v instanceof Session) return "<session>";
  return "?";
}
const same = (a, b) => Object.getPrototypeOf(a) === Object.getPrototypeOf(b);
function eq(a, b) {
  if (isBool(a) || isBool(b)) return isBool(a) && isBool(b) && a === b;
  if (isStr(a) || isStr(b)) return isStr(a) && isStr(b) && a === b;
  if (!same(a, b)) return false;
  if (a instanceof Int || a instanceof F || a instanceof F32) return a.v === b.v;
  if (a instanceof UnitT) return true;
  if (a instanceof Tup || a instanceof Lst) return a.xs.length === b.xs.length && a.xs.every((x, i) => eq(x, b.xs[i]));
  if (a instanceof St) return a.xs.length === b.xs.length && a.xs.every((x, i) => keq(x, b.xs[i]));
  if (a instanceof Mp) return a.ps.length === b.ps.length && a.ps.every(([k, v]) => b.ps.some(([k2, v2]) => keq(k, k2) && eq(v, v2)));
  if (a instanceof Opt) return a.x === null || b.x === null ? a.x === null && b.x === null : eq(a.x, b.x);
  if (a instanceof Rec) return a.n === b.n && a.fs.length === b.fs.length && a.fs.every(([p, x], i) => p === b.fs[i][0] && eq(x, b.fs[i][1]));
  if (a instanceof Err) return eq(a.e, b.e);
  if (a instanceof Chan || a instanceof Task || a instanceof Actor) return a.id === b.id;
  if (a instanceof Session) return a.out === b.out && a.inn === b.inn;
  if (a instanceof LVar) return a === b;
  return false;
}
const c3 = (x, y) => (x > y ? 1 : x < y ? -1 : 0);
function cpcmp(a, b) {
  const x = Array.from(a), y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) { const r = c3(x[i].codePointAt(0), y[i].codePointAt(0)); if (r) return r; }
  return c3(x.length, y.length);
}
function cmp(a, b) {
  if (a instanceof Int && b instanceof Int) return c3(a.v, b.v);
  if ((a instanceof F && b instanceof F) || (a instanceof F32 && b instanceof F32)) return Number.isNaN(a.v) || Number.isNaN(b.v) ? 0 : c3(a.v, b.v);
  if (isStr(a) && isStr(b)) return cpcmp(a, b);
  if (isBool(a) && isBool(b)) return c3(a ? 1 : 0, b ? 1 : 0);
  if ((a instanceof Tup && b instanceof Tup) || (a instanceof Lst && b instanceof Lst) || (a instanceof St && b instanceof St)) {
    for (let i = 0; i < Math.min(a.xs.length, b.xs.length); i++) { const r = cmp(a.xs[i], b.xs[i]); if (r) return r; }
    return c3(a.xs.length, b.xs.length);
  }
  return trap("values are not ordered");
}
function truthy(v) { if (isBool(v)) return v; return trap("expected bool, found " + show(v)); }
function chk(x, t) {
  const tt = t === "lit" ? "i64" : t;
  const [lo, hi] = irange(tt);
  if (x < lo || x > hi) trap("overflow: " + x + " is outside " + tt);
  return new Int(x, t);
}
const fin = (r, K) => (Number.isFinite(r) ? new K(r) : trap("non-finite float"));
function arith(op, a, b) { return sizeOk(arithRaw(op, a, b)); }
function arithRaw(op, a, b) {
  if (a instanceof Int && b instanceof Int) {
    const x = a.v, y = b.v, ty = a.t === "lit" ? b.t : a.t;
    const bits = ty === "i8" || ty === "u8" ? 8 : ty === "i16" || ty === "u16" ? 16 : ty === "i32" || ty === "u32" ? 32 : 64;
    switch (op) {
      case "+": return chk(x + y, ty);
      case "-": return chk(x - y, ty);
      case "*": return chk(x * y, ty);
      case "/": if (y === 0n) trap("division by zero"); return chk(x / y, ty);
      case "%": if (y === 0n) trap("division by zero"); chk(x / y, ty); return chk(x % y, ty);
      case "&": return chk(x & y, ty);
      case "|": return chk(x | y, ty);
      case "^": return chk(x ^ y, ty);
      case "<<": {
        if (y < 0n || y >= BigInt(bits)) trap("shift amount out of range");
        const r = (x << y) & (P2(bits) - 1n);
        return chk(ty.startsWith("i") && r >= P2(bits - 1) ? r - P2(bits) : r, ty);
      }
      case ">>": if (y < 0n || y >= BigInt(bits)) trap("shift amount out of range"); return chk(x >> y, ty);
    }
    return trap("bad operator " + op);
  }
  const ff = a instanceof F && b instanceof F, f3 = a instanceof F32 && b instanceof F32;
  if (ff || f3) {
    const x = a.v, y = b.v;
    let r;
    if (op === "+") r = x + y; else if (op === "-") r = x - y; else if (op === "*") r = x * y; else if (op === "/") r = x / y;
    else return trap("bad float operator");
    return f3 ? fin(Math.fround(r), F32) : fin(r, F);
  }
  if (a instanceof Lst && b instanceof Lst && ["+", "-", "*", "/", "%"].includes(op)) {
    if (a.xs.length !== b.xs.length) trap("lane count mismatch");
    return new Lst(a.xs.map((x, i) => arith(op, x, b.xs[i])));
  }
  if (a instanceof F && b instanceof Int) return arith(op, a, new F(Number(b.v)));
  if (a instanceof Int && b instanceof F) return arith(op, new F(Number(a.v)), b);
  if (isStr(a) && isStr(b) && op === "+") return a + b;
  return trap("cannot apply " + op + " to " + show(a) + " and " + show(b));
}
function contains(c, x) {
  if (c instanceof Lst || c instanceof Tup) return c.xs.some((y) => eq(y, x));
  if (c instanceof St) return c.xs.some((y) => keq(y, x));
  if (c instanceof Mp) return c.ps.some(([k]) => keq(k, x));
  if (isStr(c)) return isStr(x) && c.includes(x);
  if (c instanceof Rng) return x instanceof Int && c.a <= x.v && x.v < c.b;
  return trap("`in` needs a collection");
}
function binop(op, x, y) {
  switch (op) {
    case "==": return eq(x, y);
    case "!=": return !eq(x, y);
    case "<": return cmp(x, y) < 0;
    case "<=": return cmp(x, y) <= 0;
    case ">": return cmp(x, y) > 0;
    case ">=": return cmp(x, y) >= 0;
    case "in": return contains(y, x);
    case "not in": return !contains(y, x);
    case "..": case "..=":
      if (x instanceof Int && y instanceof Int) return new Rng(x.v, op === ".." ? y.v : y.v + 1n, x.t === "lit" ? y.t : x.t);
      return trap("bad range");
  }
  return arith(op, x, y);
}
function mapSet(ps, k, v) {
  const out = ps.slice();
  for (let i = 0; i < out.length; i++) if (keq(out[i][0], k)) { out[i] = [out[i][0], v]; return out; }
  let i = 0;
  while (i < out.length && canonCmp(out[i][0], k) < 0) i++;
  out.splice(i, 0, [k, v]);
  return out;
}
function mkMap(pairs) { let ps = []; for (const [k, v] of pairs) ps = mapSet(ps, k, v); return new Mp(ps); }
function listRepeat(x, n) {
  if (n instanceof Int && n.v >= 0n && n.v <= 10000000n) { charge(4n + n.v); return new Lst(Array(Number(n.v)).fill(x)); }
  return trap("bad repeat count");
}
const enc = new TextEncoder(), dec = new TextDecoder("utf-8", { fatal: true });
// SHA-256 (FIPS 180-4) of a byte array as 64 lowercase hex digits (5.15 `sha256`).
const SHA_K = Uint32Array.from([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
function sha256hex(m) {
  const n = m.length, L = (((n + 8) >> 6) + 1) << 6, d = new Uint8Array(L);
  d.set(m); d[n] = 0x80;
  const bits = BigInt(n) * 8n;
  for (let i = 0; i < 8; i++) d[L - 1 - i] = Number((bits >> BigInt(8 * i)) & 0xffn);
  const h = Uint32Array.from([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
  const w = new Uint32Array(64), rr = (x, k) => (x >>> k) | (x << (32 - k));
  for (let o = 0; o < L; o += 64) {
    for (let i = 0; i < 16; i++) w[i] = (d[o + 4 * i] << 24) | (d[o + 4 * i + 1] << 16) | (d[o + 4 * i + 2] << 8) | d[o + 4 * i + 3];
    for (let i = 16; i < 64; i++) {
      const s0 = rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >>> 3), s1 = rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, e4, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const t1 = (hh + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i]) >>> 0;
      const t2 = ((rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g; g = f; f = e; e = (e4 + t1) >>> 0; e4 = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    [a, b, c, e4, e, f, g, hh].forEach((x, i) => { h[i] = (h[i] + x) >>> 0; });
  }
  return Array.from(h, (x) => x.toString(16).padStart(8, "0")).join("");
}
const u8s = (s) => enc.encode(s);
function index(c, i) {
  if (c instanceof Lst && i instanceof Int) {
    if (i.v >= 0n && i.v < BigInt(c.xs.length)) return c.xs[Number(i.v)];
    return trap("index " + i.v + " out of range");
  }
  if (isStr(c) && i instanceof Int) {
    const b = u8s(c);
    if (i.v >= 0n && i.v < BigInt(b.length)) return new Int(BigInt(b[Number(i.v)]), "u8");
    return trap("index " + i.v + " out of range");
  }
  if (c instanceof Mp) { for (const [a, v] of c.ps) if (keq(a, i)) return v; return trap("missing key " + show(i)); }
  return trap("cannot index " + show(c));
}
function slice(c, ta, tb) {
  charge(1n);
  let xs, text = null;
  if (c instanceof Lst) xs = c.xs; else if (isStr(c)) { text = u8s(c); xs = text; } else trap("cannot slice " + show(c));
  const n = BigInt(xs.length);
  const av = ta ? ta() : null, bv = tb ? tb() : null;
  const norm = (v, d) => {
    if (v === null) return d;
    if (v instanceof Int) { let k = v.v; if (k < 0n) k = n + k; if (k < 0n || k > n) trap("slice index out of range"); return k; }
    return trap("bad slice bound");
  };
  const lo = Number(norm(av, 0n)), hi = Number(norm(bv, n));
  if (lo > hi) trap("slice bounds reversed");
  if (text !== null) {
    const N = text.length, bd = (k) => k === 0 || k === N || (text[k] & 0xc0) !== 0x80;
    if (!bd(lo) || !bd(hi)) trap("not a char boundary");
    return dec.decode(text.slice(lo, hi));
  }
  return new Lst(xs.slice(lo, hi));
}
function field(v, m) {
  if (v instanceof Rec) { for (const [n, x] of v.fs) if (n === m) return x; return trap("no field " + m); }
  return trap(show(v) + " has no field " + m);
}
function tmember(v, i) { if (v instanceof Tup && i < v.xs.length) return v.xs[i]; return trap("bad tuple index"); }
function neg(v) {
  if (v instanceof Int) return chk(-v.v, v.t);
  if (v instanceof F) return new F(-v.v);
  if (v instanceof F32) return new F32(-v.v);
  return trap("bad negation");
}
function bitnot(v) {
  if (v instanceof Int) { const [lo, hi] = irange(v.t === "lit" ? "i64" : v.t); return new Int(lo === 0n ? hi - v.v : ~v.v, v.t); }
  return trap("bad ~");
}
const not_ = (v) => !truthy(v);
const sat128 = (x) => (x < I128[0] ? I128[0] : x > I128[1] ? I128[1] : x);
const truncBig = (f) => BigInt(Math.trunc(f)); // exact (N4)
function cast(v, T) {
  const toF32 = T[0] === "f32", n = T[0] === "int" ? T[1] : null;
  if (v instanceof Int && n === null && toF32) return new F32(intToF32(v.v));
  if (v instanceof F && n === null && toF32) { const r = Math.fround(v.v); if (!Number.isFinite(r)) trap("non-finite float"); return new F32(r); }
  if (v instanceof F32 && n === null && !toF32) return new F(v.v);
  if (v instanceof F32 && n !== null) return chk(truncBig(v.v), n);
  if (v instanceof Int && n !== null) return chk(v.v, n);
  if (v instanceof F && n !== null) { if (!Number.isFinite(v.v)) trap("non-finite float"); return chk(truncBig(v.v), n); }
  if (isBool(v) && n !== null) return chk(v ? 1n : 0n, n);
  if (v instanceof Int && n === null) return new F(Number(v.v));
  return v;
}
// F.5.2: SHA-256 derived delimiter, rehashed until the text does not contain it
function promptEncode(s) {
  let h = sha256hex(enc.encode(s));
  for (;;) {
    const d = "data-" + h;
    if (!s.includes(d)) return "<<" + d + "\n" + s + "\n" + d + ">>";
    const raw = new Uint8Array(32);
    for (let i = 0; i < 32; i++) raw[i] = parseInt(h.slice(2 * i, 2 * i + 2), 16);
    h = sha256hex(raw);
  }
}
function ctxLit(s) { charge(4n); return s; }
function ctxEncode(c, s) {
  charge(4n + sizeOf(s));
  if (!isStr(s)) trap("context text");
  if (c === "Html") return Array.from(s).map((ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch] ?? ch).join("");
  if (c === "Url") return Array.from(enc.encode(s)).map((b) => ((b >= 48 && b <= 57) || (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || b === 45 || b === 46 || b === 95 || b === 126) ? String.fromCharCode(b) : "%" + b.toString(16).toUpperCase().padStart(2, "0")).join("");
  if (c === "Sql") return "'" + s.split("'").join("''") + "'";
  if (c === "ShellArg") return s;
  if (c === "Prompt") return promptEncode(s);
  if (c === "Header") { if (s.includes("\r") || s.includes("\n")) trap("Header.encode: a header value contains CR or LF (F.5.7)"); return s; }
  let o = '"';
  for (const ch of s) {
    const u = ch.codePointAt(0);
    if (ch === '"') o += '\\"'; else if (ch === "\\") o += "\\\\"; else if (ch === "\n") o += "\\n"; else if (ch === "\r") o += "\\r"; else if (ch === "\t") o += "\\t";
    else if (u < 0x20) o += "\\u" + u.toString(16).padStart(4, "0"); else o += ch;
  }
  return o + '"';
}
function ptm(t, v) {
  if (t[0] === "lit") return t[1];
  if (t[0] === "self") return v instanceof Int ? v.v : null;
  if (isStr(v)) return BigInt(enc.encode(v).length);
  if (v instanceof Lst || v instanceof St) return BigInt(v.xs.length);
  if (v instanceof Mp) return BigInt(v.ps.length);
  return null;
}
function pholds(p, v) {
  const k = p[0];
  if (k === "cmp") {
    const x = ptm(p[2], v), y = ptm(p[3], v);
    if (x === null || y === null) return false;
    switch (p[1]) { case "<": return x < y; case "<=": return x <= y; case ">": return x > y; case ">=": return x >= y; case "==": return x === y; default: return x !== y; }
  }
  if (k === "cls") {
    if (!isStr(v)) return false;
    if (p[1] === "isdigit") return /^[0-9]+$/.test(v);
    if (p[1] === "isalpha") return /^[A-Za-z]+$/.test(v);
    if (p[1] === "isspace") return /^[ \t\n\r\f]+$/.test(v);
    return v === "";
  }
  if (k === "and") return pholds(p[1], v) && pholds(p[2], v);
  if (k === "or") return pholds(p[1], v) || pholds(p[2], v);
  return !pholds(p[1], v);
}
function fit(v, T) {
  if (T[0] === "refn") { v = fit(v, T[1]); if (!pholds(T[3], v)) trap("refinement: value is outside " + T[2]); return v; }
  const k = T[0];
  if (v instanceof Int) {
    if (k === "int") { const [lo, hi] = irange(T[1]); if (v.v < lo || v.v > hi) trap("overflow: " + v.v + " is outside " + T[1]); return new Int(v.v, T[1]); }
    if (k === "f64") return new F(Number(v.v));
    if (k === "f32") return new F32(intToF32(v.v));
  }
  if (v instanceof Opt && v.x !== null && k === "opt") return new Opt(fit(v.x, T[1]));
  if (v instanceof Lst && (k === "list" || k === "arr")) return new Lst(v.xs.map((x) => fit(x, T[1])));
  if (v instanceof Mp && k === "map") return new Mp(v.ps.map(([a, b]) => [fit(a, T[1]), fit(b, T[2])]));
  if (v instanceof Tup && k === "tup" && v.xs.length === T[1].length) return new Tup(v.xs.map((x, i) => fit(x, T[1][i])));
  if (v instanceof Err) return v;
  if (k === "union" || k === "ref") return fit(v, T[1]);
  return v;
}
const fitall = (a, Ts) => a.slice(0, Ts.length).map((x, i) => fit(x, Ts[i]));
function wrapsat(m, x, t, y) {
  if (!(y instanceof Int)) trap(m);
  const r = m.endsWith("add") ? x + y.v : m.endsWith("sub") ? x - y.v : x * y.v;
  const [lo, hi] = irange(t);
  if (m.startsWith("wrapping")) { const w = hi - lo + 1n; let q = (r - lo) % w; if (q < 0n) q += w; return new Int(q + lo, t); }
  return new Int(r < lo ? lo : r > hi ? hi : r, t);
}
const WS = new Set(Array.from("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"));
function rtrim(s) { const c = Array.from(s); let i = 0, j = c.length; while (i < j && WS.has(c[i])) i++; while (j > i && WS.has(c[j - 1])) j--; return c.slice(i, j).join(""); }
function rwords(s) { const out = []; let cur = ""; for (const c of s) { if (WS.has(c)) { if (cur) { out.push(cur); cur = ""; } } else cur += c; } if (cur) out.push(cur); return out; }
function rlines(s) { if (s === "") return []; const p = s.split("\n"); if (p[p.length - 1] === "") p.pop(); return p.map((x) => (x.endsWith("\r") ? x.slice(0, -1) : x)); }
const aupper = (s) => s.replace(/[a-z]/g, (c) => c.toUpperCase());
const alower = (s) => s.replace(/[A-Z]/g, (c) => c.toLowerCase());
function toint(s) {
  const t = rtrim(s), u = t[0] === "+" || t[0] === "-" ? t.slice(1) : t;
  if (u === "" || !/^[0-9]+$/.test(u)) trap("not an integer: " + rdebug(s));
  const n = BigInt(t[0] === "+" ? u : t);
  if (n < I128[0] || n > I128[1]) trap("not an integer: " + rdebug(s));
  return chk(n, "i64");
}
function fround(x) { let r = Math.trunc(x); if (Math.abs(x - r) >= 0.5) r += Math.sign(x); return r; }
function sortList(xs) {
  let err = null;
  const r = xs.slice().sort((p, q) => { try { return cmp(p, q); } catch (e) { if (e instanceof Trap) { if (!err) err = e; return 0; } throw e; } });
  if (err) throw err;
  return r;
}
function apply(f, vals) {
  if (f instanceof Clo) return f.f(vals);
  if (f instanceof FnV) return RAW[f.n](vals);
  return trap(show(f) + " is not callable");
}
function callValue(f, vals) {
  if (f instanceof FnV) { fuel(); return FNS[f.n](null, vals)[0]; }
  if (f instanceof Clo) return f.f(vals);
  return trap(show(f) + " is not callable");
}
function builtin(r, m, a) { const [v, w] = builtinRaw(r, m, a); return [sizeOk(v), w === null || w === undefined ? w : sizeOk(w)]; }
function builtinRaw(r, m, a) {
  if (r instanceof Chan && m === "close") { comm(["close", r]); return [UNIT, null]; }
  if (r instanceof Task && m === "join") { const rp = comm(["join", r.id]); return [rp[0] === "val" ? rp[1] : UNIT, null]; }
  if (r instanceof Session && m === "offer") {
    charge(8n);
    const rp = comm(["recv", new Chan(r.inn, r.cap)]);
    if (rp[0] === "val" && rp[1] instanceof Opt && rp[1].x !== null) {
      const v = rp[1].x;
      return [v instanceof Rec && v.fs.length === 0 ? new Rec("$label:" + v.n, []) : v, null];
    }
    return trap("session closed");
  }
  if (r instanceof Session && m === "close") { comm(["close", new Chan(r.out, r.cap)]); return [UNIT, null]; }
  if (r instanceof LVarP) {
    if (m === "put") { lvarPut(r, a.length > 0 ? a[0] : UNIT); return [UNIT, null]; }
    if (m === "wait") return [lvarWait(r, a.length > 0 ? a[0] : UNIT), null];
    if (m === "freeze") return [parRpc(["freeze", r.id])[1], null];
  }
  if (r instanceof LVar) {
    if (m === "put") { lvarPut(r, a.length > 0 ? a[0] : UNIT); return [UNIT, null]; }
    if (m === "wait") return [lvarWait(r, a.length > 0 ? a[0] : UNIT), null];
    if (m === "freeze") { r.frz = true; return [r.v, null]; }
  }
  if (r instanceof Int) {
    const x = r.v, ty = r.t;
    if (["checked_add", "checked_sub", "checked_mul", "checked_div"].includes(m)) {
      const op = { checked_add: "+", checked_sub: "-", checked_mul: "*" }[m] || "/";
      try { return [new Opt(arith(op, r, a[0])), null]; } catch (e) { if (e instanceof Trap) return [NONE, null]; throw e; }
    }
    if (["wrapping_add", "wrapping_sub", "wrapping_mul", "saturating_add", "saturating_sub", "saturating_mul"].includes(m)) return [wrapsat(m, x, ty, a[0]), null];
    if (m === "tostr") return [x.toString(), null];
    if (m === "abs") return [chk(x < 0n ? -x : x, ty), null];
    if (m === "min" || m === "max") {
      if (!(a[0] instanceof Int)) trap(m);
      const y = a[0].v;
      return [new Int(m === "min" ? (x < y ? x : y) : x > y ? x : y, ty), null];
    }
  }
  if (r instanceof F) {
    const x = r.v;
    if (m === "sqrt") { if (x < 0) trap("non-finite float"); return [new F(Math.sqrt(x)), null]; }
    if (m === "abs") return [new F(Math.abs(x)), null];
    if (m === "min" || m === "max") {
      if (!(a[0] instanceof F)) trap(m);
      const y = a[0].v;
      return [new F(m === "min" ? (x <= y ? x : y) : x >= y ? x : y), null];
    }
    if (m === "floor") return [new F(Math.floor(x)), null];
    if (m === "ceil") return [new F(Math.ceil(x)), null];
    if (m === "round") return [new F(fround(x)), null];
    if (m === "tostr") return [fdisplay(x), null];
  }
  if (isBool(r) && m === "tostr") return [r ? "true" : "false", null];
  if (isStr(r) && (m === "len" || m === "len_bytes")) return [new Int(BigInt(u8s(r).length), "usize"), null];
  if (isStr(r) && m === "len_chars") return [new Int(BigInt([...r].length), "usize"), null];
  if (r instanceof F32) {
    if (m === "abs") return [new F32(Math.abs(r.v)), null];
    if (m === "tostr") return [f32display(r.v), null];
  }
  if (r instanceof Gen && m === "collect") {
    const [items, , err] = r.run(Number.MAX_SAFE_INTEGER);
    if (err) throw err;
    return [new Lst(items), null];
  }
  if (isStr(r)) {
    switch (m) {
      case "strip": return [rtrim(r), null];
      case "upper": return [aupper(r), null];
      case "sha256": return [sha256hex(enc.encode(r)), null];
      case "lower": return [alower(r), null];
      case "reverse": return [Array.from(r).reverse().join(""), null];
      case "tostr": return [r, null];
      case "raw": return [r, null];
      case "chars": return [new Lst(Array.from(r)), null];
      case "words": return [new Lst(rwords(r)), null];
      case "lines": return [new Lst(rlines(r)), null];
      case "split":
        if (!isStr(a[0])) trap("split");
        if (a[0] === "") return [new Lst(["", ...Array.from(r), ""]), null];
        return [new Lst(r.split(a[0])), null];
      case "isdigit": return [r !== "" && /^[0-9]+$/.test(r), null];
      case "isalpha": return [r !== "" && /^[A-Za-z]+$/.test(r), null];
      case "isspace": return [r !== "" && /^[ \t\n\x0c\r]+$/.test(r), null];
      case "isempty": return [r === "", null];
      case "contains": case "startswith": case "endswith":
        if (!isStr(a[0])) trap(m);
        return [m === "contains" ? r.includes(a[0]) : m === "startswith" ? r.startsWith(a[0]) : r.endsWith(a[0]), null];
      case "toint": return [toint(r), null];
      case "join":
        if (!(a[0] instanceof Lst)) trap("join");
        return [a[0].xs.map((x) => (isStr(x) ? x : show(x))).join(r), null];
    }
  }
  if (r instanceof Lst) {
    const xs = r.xs;
    switch (m) {
      case "len": return [new Int(BigInt(xs.length), "usize"), null];
      case "isempty": return [xs.length === 0, null];
      case "append": case "push": return [UNIT, new Lst([...xs, a[0]])];
      case "pop": return xs.length === 0 ? [NONE, new Lst([])] : [new Opt(xs[xs.length - 1]), new Lst(xs.slice(0, -1))];
      case "clear": return [UNIT, new Lst([])];
      case "sort": return [UNIT, new Lst(sortList(xs))];
      case "with": return [new Lst([...xs, a[0]]), null];
      case "reverse": return [new Lst(xs.slice().reverse()), null];
      case "contains": return [xs.some((x) => eq(x, a[0])), null];
      case "get": {
        if (!(a[0] instanceof Int)) trap("get");
        const k = a[0].v;
        return [k >= 0n && k < BigInt(xs.length) ? new Opt(xs[Number(k)]) : NONE, null];
      }
      case "utf8": { if (!xs.every((x) => x instanceof Int && x.v >= 0n && x.v <= 255n)) trap("utf8: byte outside 0..255"); try { return [new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(Uint8Array.from(xs, (x) => Number(x.v))), null]; } catch (e) { if (e instanceof Trap) throw e; trap("utf8: invalid UTF-8"); } }
      case "sum": { let acc = new Int(0n, "lit"); for (const x of xs) acc = arith("+", acc, x); return [acc, null]; }
      case "map": return [new Lst(xs.map((x) => (fuel(), apply(a[0], [x])))), null];
      case "filter": return [new Lst(xs.filter((x) => (fuel(), apply(a[0], [x])) === true)), null];
      case "pscan": return [pscan(xs, a[0], a[1], false), null];
      case "pscan_incl": return [pscan(xs, a[0], a[1], true), null];
      case "fold": { let acc = a[0]; for (const x of xs) { fuel(); acc = apply(a[1], [acc, x]); } return [acc, null]; }
    }
  }
  if (r instanceof St) {
    switch (m) {
      case "union": return [mergeBuiltin(r, a[0]), null];
      case "inter": return [new St(r.xs.filter((x) => a[0].xs.some((y) => eq(x, y)))), null];
      case "insert": return [UNIT, mergeBuiltin(r, new St([a[0]]))];
      case "tolist": return [new Lst([...r.xs]), null];
      case "len": return [new Int(BigInt(r.xs.length), "usize"), null];
      case "contains": return [r.xs.some((x) => eq(x, a[0])), null];
    }
  }
  if (r instanceof Mp) {
    const ps = r.ps;
    switch (m) {
      case "merge": return [mapMerge(ps, a[0].ps), null];
      case "len": return [new Int(BigInt(ps.length), "usize"), null];
      case "get": for (const [k, v] of ps) if (keq(k, a[0])) return [v, null]; return [new Err(new Rec("Missing", [])), null];
      case "contains": return [ps.some(([k]) => keq(k, a[0])), null];
      case "insert": return [UNIT, new Mp(mapSet(ps, a[0], a[1]))];
      case "remove": {
        const i = ps.findIndex(([k]) => keq(k, a[0]));
        if (i < 0) return [NONE, new Mp(ps.slice())];
        return [new Opt(ps[i][1]), new Mp([...ps.slice(0, i), ...ps.slice(i + 1)])];
      }
      case "keys": return [new Lst(ps.map((p) => p[0])), null];
      case "values": return [new Lst(ps.map((p) => p[1])), null];
    }
  }
  if (r instanceof Opt) {
    if (m === "isnone") return [r.x === null, null];
    if (m === "issome") return [r.x !== null, null];
    if (m === "or") return [r.x === null ? a[0] : r.x, null];
  }
  return trap(show(r) + " has no method " + m + " at comptime");
}
// C.6.4 method work, as conc::method_work in the reference evaluator.
const LIN_TEXT = new Set(["strip", "upper", "lower", "reverse", "tostr", "chars", "words", "lines", "split", "isdigit", "isalpha", "isspace", "contains", "startswith", "endswith", "toint", "join", "sha256"]), LIN_LIST = new Set(["reverse", "contains", "sum", "map", "filter", "fold", "utf8"]), LIN_SET = new Set(["union", "inter", "insert", "contains", "tolist"]), LIN_MAP = new Set(["get", "contains", "insert", "remove", "keys", "values", "merge"]);
function sizeOf(v) {
  if (isStr(v)) return BigInt(u8s(v).length);
  if (v instanceof Lst || v instanceof St) return BigInt(v.xs.length);
  if (v instanceof Mp) return BigInt(v.ps.length);
  return 0n;
}
function ceilLog2(n) { return n <= 1n ? 0n : BigInt((n - 1n).toString(2).length); }
function methodWork(recv, m, args) {
  const n = sizeOf(recv), k = args.length ? sizeOf(args[0]) : 0n;
  if ((isStr(recv) && LIN_TEXT.has(m)) || (recv instanceof Lst && LIN_LIST.has(m)) || (recv instanceof St && LIN_SET.has(m)) || (recv instanceof Mp && LIN_MAP.has(m))) return n + k;
  if (recv instanceof Lst && m === "sort") return n * ceilLog2(n);
  return 0n;
}
function callMethod(recv, m, args, sized) {
  const head = recv instanceof Rec ? (KIND_OF[recv.n] ?? recv.n) : recv instanceof Lst ? "List" : recv instanceof St ? "Set" : null;
  if (head !== null && METHODS[head + "." + m]) { fuel(); const [v, outs, s] = METHODS[head + "." + m](recv, args); return [v, outs, s]; }
  charge(4n + (sized ? methodWork(recv, m, args) : 0n));
  const [v, s] = builtin(recv, m, args);
  return [v, [], s];
}
function construct(name, fields, args) {
  const n = fields.length, vals = Array(n).fill(null);
  args.forEach(([lb, v], i) => {
    let k = i;
    if (lb !== null) { const j = fields.findIndex((f) => f[0] === lb); if (j >= 0) k = j; }
    if (k < n) vals[k] = fit(v, fields[k][1]);
  });
  return new Rec(name, fields.map(([f, T, init], k) => {
    let v = vals[k];
    if (v === null) { if (init === null) trap("missing field " + f); v = fit(init(), T); }
    return [f, v];
  }));
}
function pmatch(P, v, b) {
  switch (P[0]) {
    case "wild": return true;
    case "bind":
      if (v instanceof Rec && v.fs.length === 0 && v.n.startsWith("$label:")) return v.n.slice(7) === P[1].split("::").pop();
      b[P[1]] = v; return true;
    case "tag":
      if (v instanceof Rec && v.fs.length === 0 && v.n.startsWith("$label:")) return v.n.slice(7) === P[1].split("::").pop();
      return v instanceof Rec && v.n === P[1];
    case "some": return v instanceof Opt && v.x !== null ? pmatch(P[1], v.x, b) : false;
    case "ctor": {
      if (!(v instanceof Rec) || v.n !== P[1]) return false;
      for (let i = 0; i < P[2].length; i++) {
        const [lb, q] = P[2][i];
        let fv = null;
        if (lb !== null) { const f = v.fs.find(([n]) => n === lb); if (f) fv = f[1]; } else if (i < v.fs.length) fv = v.fs[i][1];
        if (fv === null || !pmatch(q, fv, b)) return false;
      }
      return true;
    }
    case "lit": return eq(P[1](), v);
    case "range": { const x = P[1](), y = P[3](); return cmp(v, x) >= 0 && (P[2] ? cmp(v, y) <= 0 : cmp(v, y) < 0); }
    case "tup":
      if (!(v instanceof Tup) || v.xs.length !== P[1].length) return false;
      for (let i = 0; i < P[1].length; i++) if (!pmatch(P[1][i], v.xs[i], b)) return false;
      return true;
    case "alt": for (const q of P[1]) if (pmatch(q, v, b)) return true; return false;
  }
  return false;
}
function pm(P, v) { const b = {}; return pmatch(P, v, b) ? b : null; }
const YIELDS = [];
function yield_(v) {
  if (YIELDS.length === 0) trap("yield outside an iterator");
  const [buf, cap] = YIELDS[YIELDS.length - 1];
  if (buf.length >= LOOP_LIMIT) trap("loop bound");
  buf.push(v);
  if (buf.length >= cap) throw new Stop();
}
function genRun(body, cap) {
  YIELDS.push([[], cap]);
  let fin = true, err = null;
  try { body(); } catch (e) { if (e instanceof Stop) fin = false; else if (e instanceof Trap || e instanceof Fail) err = e; else { YIELDS.pop(); throw e; } }
  const items = YIELDS.pop()[0];
  return [items, fin, err];
}
function iterItems(v) {
  if (v instanceof Lst || v instanceof St) return v.xs;
  if (v instanceof Rng) {
    if (v.b - v.a > BigInt(LOOP_LIMIT) * 10n) trap("loop bound");
    const out = [];
    for (let i = v.a; i < v.b; i++) out.push(new Int(i, v.t));
    return out;
  }
  if (isStr(v)) return Array.from(v);
  if (v instanceof Mp) return v.ps.map(([k, x]) => new Tup([k, x]));
  if (v instanceof Gen) { const [items, , err] = v.run(Number.MAX_SAFE_INTEGER); if (err) throw err; return items; }
  return trap("cannot iterate over " + show(v));
}
function* iterate(v) {
  if (!(v instanceof Gen)) { yield* iterItems(v); return; }
  let done = 0, cap = 64;
  for (;;) {
    const [items, fin, err] = v.run(cap);
    for (const x of items.slice(done)) { done++; yield x; }
    if (err) throw err;
    if (fin) return;
    cap *= 2;
  }
}
function listcomp(itv, P, names, body, cond) {
  const out = [];
  for (const x of iterItems(itv)) {
    fuel();
    const b = {};
    if (!pmatch(P, x, b)) continue;
    const args = names.map((n) => (n in b ? b[n] : UNIT));
    if (cond !== null && !truthy(cond(...args))) continue;
    out.push(body(...args));
  }
  charge(4n + BigInt(out.length));
  return new Lst(out);
}
const catch_ = (x, h) => (x instanceof Err ? h(x.e) : x);
function try_(x, els) {
  if (x instanceof Err) throw new Fail(els ? els() : x.e);
  if (x instanceof Opt && !els) return x; // ok value of `T? ! E` [E-Try-V]
  if (x instanceof Opt) { if (x.x === null) { if (els) throw new Fail(els()); trap("try on none"); } return x.x; }
  return x;
}
function tryEval(th) { try { return th(); } catch (e) { if (isFlow(e)) return null; throw e; } }
function compound(op, cur, r) { charge(opCost(op, cur)); return compoundUncharged(op, cur, r); }
function compoundUncharged(op, cur, r) { const t = cur instanceof Int ? cur.t : "lit"; const x = arith(op, cur, r); return x instanceof Int ? chk(x.v, t) : x; }
function fixtag(cur, v) { return cur instanceof Int && cur.t !== "lit" && v instanceof Int ? chk(v.v, cur.t) : v; }
function setField(rec, m, v) {
  if (!(rec instanceof Rec)) trap("member assignment on a non-record");
  const i = rec.fs.findIndex(([n]) => n === m);
  if (i < 0) trap("no field");
  const fs = rec.fs.slice(); fs[i] = [m, v];
  return new Rec(rec.n, fs);
}
function setTuple(t, i, v) { if (t instanceof Tup && i < t.xs.length) { const xs = t.xs.slice(); xs[i] = v; return new Tup(xs); } return trap("bad tuple assignment"); }
function setIndex(c, k, v) {
  if (c instanceof Lst && k instanceof Int) {
    if (k.v < 0n || k.v >= BigInt(c.xs.length)) trap("index " + k.v + " out of range");
    const xs = c.xs.slice(); xs[Number(k.v)] = v; return new Lst(xs);
  }
  if (c instanceof Mp) return new Mp(mapSet(c.ps, k, v));
  return trap("bad indexed assignment");
}
const loopBound = (v) => (v instanceof Int ? v.v : BigInt(LOOP_LIMIT));
const arenaBound = (v) => (v instanceof Int ? v.v : trap("bad arena bound"));
function check_(v) { if (!truthy(v)) trap("check failed"); }
function jesc(s) { let n = 2; for (const c of s) { if ('"\\\n\r\t'.includes(c)) n += 2; else if (c.codePointAt(0) < 0x20) n += 6; else n += u8s(c).length; } return n; }
function jsonSize(v) {
  if (v === true) return 4;
  if (v === false) return 5;
  if (v instanceof Int) return v.v.toString().length;
  if (v instanceof F || v instanceof F32) return fdebug(v.v).length;
  if (isStr(v)) return jesc(v);
  if (v instanceof UnitT) return 4;
  if (v instanceof Opt) return v.x === null ? 4 : jsonSize(v.x);
  if (v instanceof Err) return jsonSize(v.e);
  const seq = (xs, n) => 2 + xs.reduce((a, b) => a + b, 0) + Math.max(n - 1, 0);
  if (v instanceof Tup || v instanceof Lst || v instanceof St) return seq(v.xs.map(jsonSize), v.xs.length);
  if (v instanceof Mp) return seq(v.ps.map(([a, b]) => 3 + jsonSize(a) + jsonSize(b)), v.ps.length);
  if (v instanceof Rec) return 2 + jesc(v.n) + 1 + seq(v.fs.map(([a, b]) => jesc(a) + 1 + jsonSize(b)), v.fs.length);
  if (v instanceof Rng) return 3 + v.a.toString().length + v.b.toString().length;
  if (v instanceof FnV) return jesc(v.n);
  return 2;
}
function arenaCheck(bound, vals) { if (BigInt(vals.reduce((a, v) => a + jsonSize(v), 0)) > bound) trap("size limit"); }
function why(e) {
  if (e instanceof Trap) return "trap: " + e.m;
  if (e instanceof Fail) return "failed with " + show(e.v);
  if (e instanceof Stop) return "Stop";
  return "Ret";
}
// Rust Debug of the interpreter's V (refusal text); closures, iterators
// and lvars print a placeholder
function dbgV(v) {
  if (v === true || v === false) return "Bool(" + v + ")";
  if (isStr(v)) return "Text(" + rdebug(v) + ")";
  if (v instanceof Int) return "Int(" + v.v.toString() + ", " + rdebug(v.t) + ")";
  if (v instanceof F) return "F(" + fdebugFull(v.v) + ")";
  if (v instanceof F32) return "F32(" + fdebugFull(v.v) + ")";
  if (v instanceof UnitT) return "Unit";
  if (v instanceof Tup) return "Tuple([" + v.xs.map(dbgV).join(", ") + "])";
  if (v instanceof Lst) return "List([" + v.xs.map(dbgV).join(", ") + "])";
  if (v instanceof St) return "Set([" + v.xs.map(dbgV).join(", ") + "])";
  if (v instanceof Mp) return "Map([" + v.ps.map(([a, b]) => "(" + dbgV(a) + ", " + dbgV(b) + ")").join(", ") + "])";
  if (v instanceof Opt) return v.x === null ? "Opt(None)" : "Opt(Some(" + dbgV(v.x) + "))";
  if (v instanceof Rec) return "Rec(" + rdebug(v.n) + ", [" + v.fs.map(([a, b]) => "(" + rdebug(a) + ", " + dbgV(b) + ")").join(", ") + "])";
  if (v instanceof Err) return "Err(" + dbgV(v.e) + ")";
  if (v instanceof FnV) return "Fn(" + rdebug(v.n) + ")";
  if (v instanceof Rng) return "Range(" + v.a + ", " + v.b + ", " + rdebug(v.t) + ")";
  if (v instanceof Chan) return "Chan(" + rdebug(v.id) + ", " + v.cap + ")";
  if (v instanceof Task) return "Task(" + rdebug(v.id) + ")";
  if (v instanceof Actor) return "Actor(" + rdebug(v.id) + ", " + rdebug(v.ty) + ")";
  if (v instanceof Session) return "Session(" + rdebug(v.out) + ", " + rdebug(v.inn) + ", " + v.cap + ")";
  if (v instanceof Clo) return "Closure(..)";
  if (v instanceof Gen) return "Gen(..)";
  return "LVar(..)";
}
function fdebugFull(x) { return Number.isNaN(x) ? "NaN" : x === Infinity ? "inf" : x === -Infinity ? "-inf" : fdebug(x); }
function dbgFlow(e) {
  if (e instanceof Ret) return "Ret(" + dbgV(e.v) + ")";
  if (e instanceof Fail) return "Fail(" + dbgV(e.v) + ")";
  if (e instanceof Trap) return "Trap(" + rdebug(e.m) + ")";
  if (e instanceof Stop) return "Stop";
  if (e instanceof Cancel) return "Cancel(" + e.f + ")";
  return "?";
}
// machine step refusal (interp.rs)
function refusal(e) {
  if (e instanceof Fail) return e.v instanceof Rec ? e.v.n : "?";
  if (e instanceof Trap) return "trap(" + e.m + ")";
  return "trap(" + dbgFlow(e) + ")";
}
// actor handler refusal (conc.rs)
function actorRefusal(e) {
  if (e instanceof Fail && e.v instanceof Rec) return e.v.n;
  if (e instanceof Trap) return "trap(" + e.m + ")";
  return "trap(" + dbgFlow(e) + ")";
}
const OUT = [];
function emit(v) { if (TASK !== null) { comm(["emit", v]); return; } OUT.push(v); }
const LINES = [];
const say = (s) => { LINES.push(s); };
function obs(label, th) {
  fuelReset();
  try { say(label + " = " + show(th())); } catch (e) { if (!isFlow(e)) throw e; say(label + " " + why(e)); }
  fuelReport();
}
const showState = (s) => s.map(([r, fs]) => r + " { " + fs.map(([n, v]) => n + " = " + show(v)).join(", ") + " }").join("; ");
function delta(s, ev) {
  const nxt = [], outs = [];
  for (const [name, fs] of s) {
    let got = null;
    for (const st of STEPS[name] || []) {
      OUT.length = 0;
      let r;
      try { r = st(fs, ev); } catch (e) { if (isFlow(e)) throw new Refuse(refusal(e)); throw e; }
      if (r !== null) { got = [r, OUT.slice()]; break; }
    }
    if (got === null) nxt.push([name, fs]); else { nxt.push([name, got[0]]); outs.push(...got[1]); }
  }
  return [nxt, outs];
}
function runExample(line, events) {
  fuelReset();
  try { runExample1(line, events); } finally { fuelReport(); }
}
function runExample1(line, events) {
  say("example " + line);
  let s;
  try { s = initialState(); } catch (e) { if (!isFlow(e)) throw e; say("initial " + why(e)); return; }
  say("0 (initial) " + showState(s));
  const d = new DLog();
  try { runEvents(s, events, d); } finally { if (TRACE) say("trace " + d.prev); }
}
function runEvents(s, events, d) {
  for (let k = 1; k <= events.length; k++) {
    const [name, th] = events[k - 1];
    let ev;
    try { ev = th(); } catch (e) { if (!isFlow(e)) throw e; say(k + " " + name + " " + why(e)); return; }
    if (ev === null) { say(k + " " + name + " not an event"); return; }
    let r;
    try { r = delta(s, ev); } catch (e) { if (e instanceof Refuse) { if (TRACE) d.add(name, ev, s, true); say(k + " " + show(ev) + " refused: " + e.r); return; } throw e; }
    s = r[0];
    if (TRACE) d.add(name, ev, s, false);
    say(k + " " + show(ev) + " " + showState(s));
    for (const o of r[1]) say("  emit " + show(o));
  }
}
// ---- 6.1 the log D: canonical JSON (N3) and the SHA-256 entry chain, as interp.rs writes it
const TRACE = FUEL_ARGS.includes("--trace");
const D_GENESIS = "0".repeat(64);
const cpCmp = (a, b) => { const x = Array.from(a, (c) => c.codePointAt(0)), y = Array.from(b, (c) => c.codePointAt(0)); for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1; return x.length === y.length ? 0 : x.length < y.length ? -1 : 1; };
class FRec { constructor(fs) { this.fs = fs.slice().sort((a, b) => cpCmp(a[0], b[0])); } }
class FFl { constructor(x) { this.x = x; } }
function formValue(v) {
  if (v === true || v === false) return v;
  if (isStr(v)) return v;
  if (v instanceof Int) return BigInt(v.v);
  if (v instanceof F || v instanceof F32) return new FFl(v.v);
  if (v instanceof UnitT) return null;
  if (v instanceof Opt) return v.x === null ? null : formValue(v.x);
  if (v instanceof Tup || v instanceof Lst || v instanceof St) return v.xs.map(formValue);
  if (v instanceof Mp) { const l = v.ps.map(([a, b]) => [formValue(a), formValue(b)]); l.sort((p, q) => fvCmp(p[0], q[0])); return l; }
  if (v instanceof Rec) { const f = v.fs.map(([k, x]) => [k, formValue(x)]); if (VARIANTS.has(v.n)) f.push(["$v", v.n]); return new FRec(f); }
  if (v instanceof Err) return new FRec([["$err", formValue(v.e)]]);
  if (v instanceof Rng) { const o = []; for (let i = BigInt(v.a); i < BigInt(v.b); i++) o.push(i); return o; }
  if (v instanceof FnV) return new FRec([["$fn", v.n]]);
  return null;
}
function formValuePos(v) {
  // 6.1, T7: the positional embedding hashed by D (names replaced by positions)
  if (v instanceof Opt) return v.x === null ? null : formValuePos(v.x);
  if (v instanceof Tup || v instanceof Lst || v instanceof St) return v.xs.map(formValuePos);
  if (v instanceof Mp) { const l = v.ps.map(([a, b]) => [formValue(a), formValuePos(a), formValuePos(b)]); l.sort((p, q) => fvCmp(p[0], q[0])); return l.map(([, a, b]) => [a, b]); }
  if (v instanceof Rec) {
    const at = (f) => { const p = v.fs.find(([k]) => k === f); return p ? formValuePos(p[1]) : null; };
    if (Object.hasOwn(DECL_FIELDS, v.n)) return DECL_FIELDS[v.n].map(at);
    if (Object.hasOwn(VARIANT_INDEX, v.n)) { const [k, fs] = VARIANT_INDEX[v.n]; return [BigInt(k), fs.map(at)]; }
    return formValue(v);
  }
  if (v instanceof Err) return new FRec([["$err", formValuePos(v.e)]]);
  return formValue(v);
}
function fvRank(x) {
  if (x === null) return 0;
  if (x === true || x === false) return 1;
  if (typeof x === "bigint" || x instanceof FFl) return 2;
  if (typeof x === "string") return 3;
  if (Array.isArray(x)) return 4;
  return 5;
}
function fvCmp(a, b) {
  const ra = fvRank(a), rb = fvRank(b);
  if (ra !== rb) return ra < rb ? -1 : 1;
  if (ra === 1) return a === b ? 0 : a ? 1 : -1;
  if (ra === 2) {
    if (typeof a === "bigint" && typeof b === "bigint") return a < b ? -1 : a > b ? 1 : 0;
    const x = typeof a === "bigint" ? Number(a) : a.x, y = typeof b === "bigint" ? Number(b) : b.x;
    return x < y ? -1 : x > y ? 1 : 0;
  }
  if (ra === 3) return cpCmp(a, b);
  if (ra === 4) { for (let i = 0; i < Math.min(a.length, b.length); i++) { const c = fvCmp(a[i], b[i]); if (c) return c; } return a.length === b.length ? 0 : a.length < b.length ? -1 : 1; }
  if (ra === 5) { for (let i = 0; i < Math.min(a.fs.length, b.fs.length); i++) { const c = cpCmp(a.fs[i][0], b.fs[i][0]) || fvCmp(a.fs[i][1], b.fs[i][1]); if (c) return c; } return a.fs.length === b.fs.length ? 0 : a.fs.length < b.fs.length ? -1 : 1; }
  return 0;
}
function floatJson(x) {
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [m0, e0] = Math.abs(x).toExponential().split("e");
  const ex = Number(e0), digits = m0.replace(".", ""), sign = x < 0 ? "-" : "", a = Math.abs(x);
  if (a >= 1e-4 && a < 1e16) {
    const p = ex + 1;
    let body;
    if (p <= 0) body = "0." + "0".repeat(-p) + digits;
    else if (p >= digits.length) body = digits + "0".repeat(p - digits.length) + ".0";
    else body = digits.slice(0, p) + "." + digits.slice(p);
    return sign + body;
  }
  return sign + m0 + "e" + ex;
}
function jstr(s) {
  let o = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') o += '\\"'; else if (ch === "\\") o += "\\\\"; else if (ch === "\n") o += "\\n";
    else if (ch === "\r") o += "\\r"; else if (ch === "\t") o += "\\t";
    else if (c < 0x20) o += "\\u" + c.toString(16).padStart(4, "0"); else o += ch;
  }
  return o + '"';
}
function fvJson(x) {
  if (x === null) return "null";
  if (x === true) return "true";
  if (x === false) return "false";
  if (typeof x === "bigint") return x.toString();
  if (x instanceof FFl) return floatJson(x.x);
  if (typeof x === "string") return jstr(x);
  if (Array.isArray(x)) return "[" + x.map(fvJson).join(",") + "]";
  return "{" + x.fs.map(([k, v]) => jstr(k) + ":" + fvJson(v)).join(",") + "}";
}
const dSha = (s) => sha256hex(enc.encode(s));
class DLog {
  constructor() { this.prev = D_GENESIS; this.seq = 0; }
  // one entry, printed with --trace: d applied|kept HASH BODY (HASH = SHA-256 of BODY)
  add(name, ev, st, kept) {
    const ty = Object.hasOwn(EVENT_INDEX, name) ? String(EVENT_INDEX[name]) : jstr(name);
    const evj = '{"payload":' + fvJson(formValuePos(ev)) + ',"type":' + ty + "}";
    const sj = fvJson(st.map(([r, fs]) => fs.map(([n, v]) => formValuePos(v))));
    const body = '{"event":' + evj + ',"prev_hash":"' + this.prev + '","seq":' + this.seq + ',"state_hash":"' + dSha(sj) + '"}';
    const h = dSha(body);
    say("d " + (kept ? "kept" : "applied") + " " + h + " " + body);
    this.prev = h; this.seq++;
  }
}
function runMain(f) {
  fuelReset();
  try { runMain1(f); } finally { fuelReport(); }
}
function runMain1(f) {
  say("main");
  OUT.length = 0;
  try {
    fuel();
    const v = f(null, [])[0];
    for (const o of OUT) say("emit " + show(o));
    say("main = " + show(v));
  } catch (e) {
    if (!isFlow(e)) throw e;
    for (const o of OUT) say("emit " + show(o));
    say("main " + why(e));
  }
}

// ---- draft 0.6 (the draft 0.6 concurrency chapter, not published here C.5.2, C.11.2)
function parIdent(op, b) {
  if (b instanceof Int) {
    if (op === "+" || op === "|" || op === "^") return new Int(0n, b.t);
    if (op === "*") return new Int(1n, b.t);
    if (op === "&") { const [lo, hi] = RANGES[b.t === "lit" ? "i64" : b.t] || RANGES.i64; return new Int(lo === 0n ? hi : -1n, b.t); }
  }
  if (b instanceof F) { if (op === "+") return new F(-0.0); if (op === "*") return new F(1.0); }
  if (b instanceof F32) { if (op === "+") return new F32(-0.0); if (op === "*") return new F32(1.0); }
  if (b === true || b === false) { if (op === "&") return true; if (op === "|" || op === "^") return false; }
  return trap("no identity for " + op + "= on " + show(b));
}
function parCombine(op, a, b) {
  charge(opCost(op, a));
  if ((a === true || a === false) && (b === true || b === false)) {
    if (op === "&") return a && b;
    if (op === "|") return a || b;
    if (op === "^") return a !== b;
  }
  return compoundUncharged(op, a, b);
}
function parTree(op, c, i) {
  if (c.length === 0) return i;
  if (c.length === 1) return c[0];
  const h = 1 << (31 - Math.clz32(c.length - 1));
  const l = parTree(op, c.slice(0, h), i);
  const r = parTree(op, c.slice(h), i);
  return parCombine(op, l, r);
}
function pscan(xs, f, e, inclusive) {
  const n = xs.length;
  if (n === 0) return new Lst([]);
  const sums = new Map();
  const up = (lo, hi) => {
    let v;
    if (hi - lo === 1) v = xs[lo];
    else { const h = lo + (1 << (31 - Math.clz32(hi - lo - 1))); const l = up(lo, h); const r = up(h, hi); v = apply(f, [l, r]); }
    sums.set(lo + ":" + hi, v);
    return v;
  };
  const out = new Array(n).fill(null);
  const down = (lo, hi, off) => {
    if (hi - lo === 1) { out[lo] = off; return; }
    const h = lo + (1 << (31 - Math.clz32(hi - lo - 1)));
    const ls = sums.get(lo + ":" + h);
    const roff = off === null ? ls : apply(f, [off, ls]);
    down(lo, h, off);
    down(h, hi, roff);
  };
  up(0, n);
  down(0, n, null);
  return new Lst(out.map((o, i) => inclusive ? (o === null ? xs[i] : apply(f, [o, xs[i]])) : (o === null ? e : o)));
}


// ---- cost accounting for `with budget` (C.6.1, C.6.4), as interp `charge`
const BUD = [];
function charge(u) {
  if (BUD.length === 0) return;
  u = BigInt(u);
  for (const b of BUD) if (b[0] + u > b[1]) trap("budget");
  for (const b of BUD) b[0] += u;
}
function budgetEnter(n) {
  BUD.push([0n, markerInt(n)]);
}
let BUD_LAST = 0n;
function budgetExit() { const b = BUD.pop(); if (b) BUD_LAST = b[0]; }
function budgetLast() { return new Int(BUD_LAST, "u64"); }
function opCost(op, x) {
  if (x instanceof F || x instanceof F32) return op === "+" || op === "-" || op === "*" ? 2n : op === "/" || op === "%" ? 8n : 1n;
  return 1n;
}
function cindex(c, i) { charge(1n); return index(c, i); }
function cfield(v, m) { charge(1n); return field(v, m); }
function ctmember(v, i) { charge(1n); return tmember(v, i); }
function cneg(v) { charge(1n); return neg(v); }
function cbitnot(v) { charge(1n); return bitnot(v); }
function cnot(v) { charge(1n); return not_(v); }
function cbinop(op, x, y) { charge(opCost(op, x)); return binop(op, x, y); }
function preMethod(recv, m) {
  const head = recv instanceof Rec ? (KIND_OF[recv.n] ?? recv.n) : recv instanceof Lst ? "List" : recv instanceof St ? "Set" : null;
  if (head !== null && METHODS[head + "." + m]) charge(4n);
  return recv;
}
function preValue(f) { if (f instanceof FnV) charge(4n); return f; }

function markerInt(v) { if (v instanceof Int) return v.v; return trap("expected an integer, got " + show(v)); }

// ---- LVars (C.11.8) and the built-in Merge instances (C.11.5)
class LVar { constructor(v, top) { this.v = v; this.top = top; this.frz = false; } }
const lvarNew = (a) => new LVar(a.length > 0 ? a[0] : UNIT, a.length > 1 ? a[1] : null);
function crank(v) {
  if (v === UNIT || (v instanceof Opt && v.x === null)) return 0;
  if (isBool(v)) return 1;
  if (v instanceof Int || v instanceof F || v instanceof F32) return 2;
  if (isStr(v)) return 3;
  if (v instanceof Lst || v instanceof St || v instanceof Tup) return 4;
  return 5;
}
const ccmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
function bcmp(a, b) { const x = u8s(a), y = u8s(b); for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return ccmp(x[i], y[i]); return ccmp(x.length, y.length); }
const keq = (a, b) => canonCmp(a, b) === 0; // map/set key identity (6.1)
function canonCmp(a, b) {
  const ra = crank(a), rb = crank(b);
  if (ra !== rb) return ccmp(ra, rb);
  if (ra === 1) return ccmp(a, b);
  if (a instanceof Int && b instanceof Int) return ccmp(a.v, b.v);
  if (ra === 2) {
    // int before float (never equal keys); floats by IEEE 754 totalOrder (6.1)
    if (a instanceof Int) return -1;
    if (b instanceof Int) return 1;
    const x = Number(a.v), y = Number(b.v);
    if (x === 0 && y === 0) return ccmp(Object.is(x, -0) ? 0 : 1, Object.is(y, -0) ? 0 : 1);
    return ccmp(x, y);
  }
  if (ra === 3) return bcmp(a, b);
  if (ra === 4) { for (let i = 0; i < Math.min(a.xs.length, b.xs.length); i++) { const c = canonCmp(a.xs[i], b.xs[i]); if (c !== 0) return c; } return ccmp(a.xs.length, b.xs.length); }
  if (a instanceof Opt && b instanceof Opt && a.x !== null && b.x !== null) return canonCmp(a.x, b.x);
  if (a instanceof Rec && b instanceof Rec) {
    if (a.n !== b.n) return bcmp(a.n, b.n);
    for (let i = 0; i < Math.min(a.fs.length, b.fs.length); i++) { const c = bcmp(a.fs[i][0], b.fs[i][0]) || canonCmp(a.fs[i][1], b.fs[i][1]); if (c !== 0) return c; }
    return ccmp(a.fs.length, b.fs.length);
  }
  return bcmp(show(a), show(b));
}
function mergeBuiltin(a, b) {
  if (a instanceof Int && b instanceof Int) return new Int(a.v > b.v ? a.v : b.v, a.t === "lit" ? b.t : a.t);
  if (isBool(a) && isBool(b)) return a || b;
  if (a instanceof Lst && b instanceof Lst) {
    const v = [...a.xs, ...b.xs].sort(canonCmp), out = [];
    for (const x of v) if (out.length === 0 || canonCmp(out[out.length - 1], x) !== 0) out.push(x);
    return new Lst(out);
  }
  if (a instanceof St && b instanceof St) {
    const v = [...a.xs, ...b.xs].sort(canonCmp), out = [];
    for (const x of v) if (out.length === 0 || canonCmp(out[out.length - 1], x) !== 0) out.push(x);
    return new St(out);
  }
  if (a instanceof Mp && b instanceof Mp) {
    const out = [...a.ps];
    for (const [k, v] of b.ps) { const i = out.findIndex(([k2]) => keq(k2, k)); if (i >= 0) out[i] = [out[i][0], mergeBuiltin(out[i][1], v)]; else out.push([k, v]); }
    out.sort((p, q) => canonCmp(p[0], q[0]));
    return new Mp(out);
  }
  if (a instanceof Tup && b instanceof Tup && a.xs.length === b.xs.length) return new Tup(a.xs.map((p, i) => mergeBuiltin(p, b.xs[i])));
  if (a instanceof Rec && b instanceof Rec && a.n === b.n && a.fs.length === b.fs.length) return new Rec(a.n, a.fs.map(([k, p], i) => [k, mergeBuiltin(p, b.fs[i][1])]));
  if (a instanceof Opt && a.x === null) return b;
  if (b instanceof Opt && b.x === null) return a;
  if (a instanceof Opt && b instanceof Opt) return new Opt(mergeBuiltin(a.x, b.x));
  return trap("no Merge instance for " + show(a) + " and " + show(b));
}
function mapMerge(xs, ys) {
  const out = xs.slice();
  for (const [k, v] of ys) {
    const i = out.findIndex(([k2]) => keq(k2, k));
    if (i >= 0) out[i] = [out[i][0], mergeValues(out[i][1], v)]; else out.push([k, v]);
  }
  out.sort((p, q) => canonCmp(p[0], q[0]));
  return new Mp(out);
}
function setOf(v) { return new St(mergeBuiltin(v, new Lst([])).xs); }
function mergeValues(a, b) {
  if (a instanceof Rec) { const h = KIND_OF[a.n] ?? a.n; const f = METHODS[h + ".merge"]; if (f) { charge(4n); return f(a, [b])[0]; } }
  return mergeBuiltin(a, b);
}
let PUTS = 0;
const puts = () => PUTS;
function lvarPut(c, v) {
  if (c instanceof LVarP) {
    // a main-thread lvar seen from a par worker: merge here (charging this
    // branch, as the evaluator does), then compare-and-set on the owner
    let cur = parRpc(["get", c.id])[1];
    for (;;) {
      const nv = mergeValues(cur, v);
      if (c.top !== undefined && c.top !== null && eq(nv, c.top)) trap("lvar conflict");
      const r = parRpc(["cas", c.id, cur, nv]);
      if (r[0] === "ok") { if (r[1]) PUTS++; return; }
      if (r[0] === "trap") trap(r[1]);
      cur = r[1];
    }
  }
  const nv = mergeValues(c.v, v); if (c.top !== null && eq(nv, c.top)) trap("lvar conflict");
  if (!eq(nv, c.v)) { if (c.frz) trap("lvar put after freeze"); PUTS++; }
  c.v = nv; }
// merge for a threshold test: the program's Merge, charged nothing (interp `merge_free`)
function mergeFree(a, b) {
  const bud = BUD.splice(0, BUD.length), f = FUEL;
  FUEL = FUEL_LIMIT;
  try { return mergeValues(a, b); } finally { FUEL = f; BUD.push(...bud); }
}
function lvarAbove(ths, cur) { for (const t of ths) if (eq(mergeFree(t, cur), cur)) return t; return null; }
function lvarIncomp(top, ths) {
  for (let i = 0; i < ths.length; i++) for (let j = i + 1; j < ths.length; j++) {
    if (eq(ths[i], ths[j])) continue;
    if (top === undefined || top === null || !eq(mergeFree(ths[i], ths[j]), top)) trap("lvar thresholds are not pairwise incompatible");
  }
}
let SCHED = 0;
function lvarWait(c, q) {
  if (!(q instanceof Lst)) trap("wait needs a list of thresholds");
  lvarIncomp(c.top, q.xs);
  if (c instanceof LVarP) {
    const r = parRpc([SCHED > 0 ? "try" : "wait", c.id, q]);
    if (r[0] === "val") return r[1];
    if (r[0] === "no") trap("$block");
    return trap(r[1]);
  }
  { const t = lvarAbove(q.xs, c.v); if (t !== null) return t; }
  if (SCHED > 0) trap("$block");
  return trap("lvar wait cannot be satisfied");
}
/// the restart scheduler made no progress. As in the evaluator, a par
/// context decides on its own branches only: puts made outside it do not
/// count as its progress (a nested par inside a worker branch runs on that
/// worker's restart scheduler)
function parStuck(pc) { return PUTS === pc; }
function parEnter() { SCHED++; }
function parExit() { SCHED--; }
const budSnap = () => BUD.map((b) => [...b]);
function budRestore(s) { BUD.length = 0; for (const b of s) BUD.push(b); }
// a restarted par attempt (C.14) restores the budgets and the fuel
const parSnap = () => [budSnap(), FUEL];
function parRestore(s) { budRestore(s[0]); FUEL = s[1]; }

// ---- workgroups (C.11.2)
function lanesOf(v) { const n = markerInt(v); return [n > 0n ? Number(n) : 0, n]; }
const groupRec = (lane, n) => new Rec("$group", [["lane", new Int(BigInt(lane), "usize")], ["lanes", new Int(n, "usize")]]);


// ---- the tick runtime for `--run` (6.8, chapter C), as the Rust and Python
// runtimes. Every task runs on its own worker thread (worker_threads), so
// tasks run in parallel; the scheduler on the main thread admits moves tick
// by tick. Workers share no memory: values cross as copies, and a closure,
// iterator or lvar cannot be passed to another task (it traps). A request
// and its reply are exchanged synchronously with receiveMessageOnPort and
// Atomics.wait on a shared flag. `par` and `for ... par` run their branches
// on workers too (see "par and for ... par on workers" below).
const WT = process.getBuiltinModule("worker_threads");
const CRYPTO = process.getBuiltinModule("crypto");
class Chan { constructor(id, cap) { this.id = id; this.cap = cap; } }
class Task { constructor(id) { this.id = id; } }
class Actor { constructor(id, ty) { this.id = id; this.ty = ty; } }
class Session { constructor(out, inn, cap) { this.out = out; this.inn = inn; this.cap = cap; } }
const CANCEL_TASK = -1;
const hid = (key) => CRYPTO.createHash("sha256").update(key, "utf8").digest("hex").slice(0, 16);
function encV(v) {
  if (v === null || v === undefined || v === true || v === false || typeof v === "string" || typeof v === "number" || typeof v === "bigint") return v;
  if (Array.isArray(v)) return { k: "arr", xs: v.map(encV) };
  if (v instanceof Int) return { k: "i", v: v.v, t: v.t };
  if (v instanceof F) return { k: "f", v: v.v };
  if (v instanceof F32) return { k: "f32", v: v.v };
  if (v instanceof UnitT) return { k: "u" };
  if (v instanceof Tup) return { k: "tup", xs: v.xs.map(encV) };
  if (v instanceof Lst) return { k: "lst", xs: v.xs.map(encV) };
  if (v instanceof St) return { k: "st", xs: v.xs.map(encV) };
  if (v instanceof Mp) return { k: "mp", ps: v.ps.map(([a, b]) => [encV(a), encV(b)]) };
  if (v instanceof Opt) return { k: "o", x: v.x === null ? null : encV(v.x) };
  if (v instanceof Rec) return { k: "r", n: v.n, fs: v.fs.map(([a, b]) => [a, encV(b)]) };
  if (v instanceof Err) return { k: "e", e: encV(v.e) };
  if (v instanceof FnV) return { k: "fn", n: v.n };
  if (v instanceof Rng) return { k: "rg", a: v.a, b: v.b, t: v.t };
  if (v instanceof Chan) return { k: "ch", id: v.id, cap: v.cap };
  if (v instanceof Task) return { k: "tk", id: v.id };
  if (v instanceof Actor) return { k: "ac", id: v.id, ty: v.ty };
  if (v instanceof Session) return { k: "se", out: v.out, inn: v.inn, cap: v.cap };
  if (v instanceof LVar && LVENC !== null) return { k: "lv", id: LVENC(v), top: v.top === null ? null : encV(v.top) };
  if (v instanceof LVarP) return { k: "lv", id: v.id, top: v.top === null ? null : encV(v.top) };
  return trap("a closure, iterator or lvar cannot be passed to another task (JavaScript workers share no memory)");
}
function decV(v) {
  if (v === null || typeof v !== "object") return v;
  switch (v.k) {
    case "arr": return v.xs.map(decV);
    case "i": return new Int(v.v, v.t);
    case "f": return new F(v.v);
    case "f32": return new F32(v.v);
    case "u": return UNIT;
    case "tup": return new Tup(v.xs.map(decV));
    case "lst": return new Lst(v.xs.map(decV));
    case "st": return new St(v.xs.map(decV));
    case "mp": return new Mp(v.ps.map(([a, b]) => [decV(a), decV(b)]));
    case "o": return v.x === null ? NONE : new Opt(decV(v.x));
    case "r": return new Rec(v.n, v.fs.map(([a, b]) => [a, decV(b)]));
    case "e": return new Err(decV(v.e));
    case "fn": return new FnV(v.n);
    case "rg": return new Rng(v.a, v.b, v.t);
    case "ch": return new Chan(v.id, v.cap);
    case "tk": return new Task(v.id);
    case "ac": return new Actor(v.id, v.ty);
    case "se": return new Session(v.out, v.inn, v.cap);
    case "lv": return LVDEC(v.id, v.top === null ? null : decV(v.top));
  }
  return v;
}
// ---- par and for ... par on workers. Each branch (or chunk of iterations)
// runs on its own worker on a copy of the branch environment (locals are
// serialized; a closure or iterator in scope keeps the par on the
// single-thread restart scheduler). Lvars stay on the thread that made
// them: a worker sees a proxy and puts by get / merge / compare-and-set,
// and waits through the owner, which runs a small scheduler while the
// branches run: it answers lvar requests, wakes waiters when a put
// changes a value, and when every live branch is blocked it makes the
// waits trap as the evaluator's ParCtx does. Results resolve in source
// order as in the Rust and Python runtimes. A par inside a worker branch
// runs on that worker's single-thread restart scheduler.
class LVarP { constructor(id, top) { this.id = id; this.top = top; } }
let LVENC = null;
let LVDEC = (id, top) => new LVarP(id, top);
let PARW = null;
function parRpc(msg) {
  const { port, ctl, i } = PARW;
  port.postMessage(encV(msg));
  Atomics.add(ctl, 0, 1); Atomics.notify(ctl, 0);
  for (;;) {
    while (Atomics.load(ctl, i) === 0) Atomics.wait(ctl, i, 0);
    Atomics.store(ctl, i, 0);
    const m = WT.receiveMessageOnPort(port);
    if (m !== undefined) return decV(m.message);
  }
}
function parChunks(n) {
  const os = process.getBuiltinModule("os");
  const e = Number(process.env.RCE_PAR_THREADS || 0);
  const w = e > 0 ? e : (os.availableParallelism ? os.availableParallelism() : 4);
  return Math.max(1, Math.min(n, w));
}
function parSplit(xs, k) { const n = xs.length; return Array.from({ length: k }, (_, i) => xs.slice(Math.floor(i * n / k), Math.floor((i + 1) * n / k))); }
function parCanWorker(env) {
  if (PARW !== null || process.env.RCE_PAR_SEQ === "1") return false;
  const reg = [];
  LVENC = (c) => { let i = reg.indexOf(c); if (i < 0) { i = reg.length; reg.push(c); } return i; };
  try { encV(env); return true; } catch (e) { if (e instanceof Trap) return false; throw e; } finally { LVENC = null; }
}
function encFlow(e) {
  if (e === null) return null;
  if (e instanceof Trap) return ["trap", e.m];
  if (e instanceof Fail) return ["fail", e.v];
  if (e instanceof Stop) return ["stop"];
  if (e instanceof Cancel) return ["cancel", e.f];
  return ["trap", "par branch escaped"];
}
function decFlow(e) {
  if (e === null) return null;
  if (e[0] === "trap") return new Trap(e[1]);
  if (e[0] === "fail") return new Fail(e[1]);
  if (e[0] === "stop") return new Stop();
  return new Cancel(e[1]);
}
/// run PAR_FNS[names[i]](env, extras[i]) on workers; returns [[flow|null, cost, result]]
function parWorkers(names, env, extras) {
  const k = names.length;
  const reg = [];
  LVENC = (c) => { let i = reg.indexOf(c); if (i < 0) { i = reg.length; reg.push(c); } return i; };
  let envE, extE;
  try { envE = encV(env); extE = encV(extras); } finally { LVENC = null; }
  if (process.env.RCE_PAR_TRACE) process.getBuiltinModule("fs").appendFileSync(process.env.RCE_PAR_TRACE, "par: " + k + " workers\n");
  const ctl = new Int32Array(new SharedArrayBuffer(4 * (k + 1)));
  const bud = budSnap();
  const ws = names.map((n, i) => {
    const mc = new WT.MessageChannel();
    const w = new WT.Worker(new URL(import.meta.url), { argv: process.argv.slice(2),
      workerData: { par: true, fn: n, env: envE, extra: extE.xs[i], bud, fuel: FUEL, sab: ctl.buffer, i: i + 1, port: mc.port2 },
      transferList: [mc.port2], resourceLimits: { stackSizeMb: 64 } });
    w.unref();
    return { w, port: mc.port1, i: i + 1, st: "run", q: null, res: null };
  });
  const reply = (x, msg) => { x.st = "run"; x.port.postMessage(encV([...msg, PUTS])); Atomics.store(ctl, x.i, 1); Atomics.notify(ctl, x.i); };
  const sat = (x) => lvarAbove(x.q[1].xs, reg[x.q[0]].v);
  for (;;) {
    const seen = Atomics.load(ctl, 0);
    let any = false;
    for (const x of ws) {
      let m;
      while (x.st !== "done" && (m = WT.receiveMessageOnPort(x.port)) !== undefined) {
        any = true;
        const r = decV(m.message);
        const c = reg[r[1]];
        switch (r[0]) {
          case "done": x.st = "done"; x.res = [decFlow(r[1]), r[2], r[3], r[4]]; break;
          case "get": reply(x, ["val", c.v]); break;
          case "freeze": c.frz = true; reply(x, ["val", c.v]); break;
          case "cas":
            if (eq(c.v, r[2])) {
              const ch = !eq(r[3], c.v);
              if (ch && c.frz) { reply(x, ["trap", "lvar put after freeze"]); break; }
              if (ch) PUTS++; c.v = r[3]; reply(x, ["ok", ch]);
            }
            else reply(x, ["retry", c.v]);
            break;
          case "try": { x.q = [r[1], r[2]]; const t = sat(x); x.q = null; reply(x, t === null ? ["no"] : ["val", t]); break; }
          case "wait": { x.q = [r[1], r[2]]; const t = sat(x); if (t === null) { x.st = "wait"; x.ver = PUTS; } else { x.q = null; reply(x, ["val", t]); } break; }
        }
      }
    }
    for (const x of ws) {
      if (x.st === "wait") { const t = sat(x); if (t !== null) { x.q = null; reply(x, ["val", t]); } }
    }
    if (ws.every((x) => x.st === "done")) break;
    if (!any && ws.every((x) => x.st !== "run")) {
      for (const x of ws) {
        if (x.st === "wait") reply(x, ["trap", "lvar wait cannot be satisfied"]);
      }
      continue;
    }
    if (!any) Atomics.wait(ctl, 0, seen, 50);
  }
  LVDEC = (id) => reg[id];
  const f0 = FUEL;
  try { return ws.map((x) => [x.res[0] === null ? null : x.res[0], x.res[1], x.res[2] === null ? null : decV(x.res[2]), x.res[3], f0]); }
  finally { LVDEC = (id, top) => new LVarP(id, top); for (const x of ws) x.port.close(); }
}
function parResolve(outs) {
  let rem = null;
  for (const [s, l] of BUD) { const r = l - s; if (rem === null || r < rem) rem = r; }
  let cum = 0n, fcum = 0n;
  for (const [e, c, , u, f0] of outs) {
    if (u !== undefined) {
      fcum += u;
      FUEL = f0 > fcum ? f0 - fcum : 0n;
      if (fcum > f0) trap("comptime fuel exhausted");
    }
    if (rem !== null && ((e instanceof Trap && e.m === "budget") || cum + c > rem)) trap("budget");
    if (e !== null) throw e;
    cum += c;
  }
  for (const b of BUD) b[0] += cum;
  return outs.map((o) => o[2]);
}
function merge3(base, cur, nw) {
  if (eq(nw, base)) return cur;
  if (eq(cur, base)) return nw;
  if (base instanceof Rec && cur instanceof Rec && nw instanceof Rec && base.n === cur.n && cur.n === nw.n && base.fs.length === cur.fs.length && cur.fs.length === nw.fs.length)
    return new Rec(nw.n, base.fs.map(([k, b], j) => [k, merge3(b, cur.fs[j][1], nw.fs[j][1])]));
  for (const T of [Tup, Lst])
    if (base instanceof T && cur instanceof T && nw instanceof T && base.xs.length === cur.xs.length && cur.xs.length === nw.xs.length)
      return new T(base.xs.map((b, j) => merge3(b, cur.xs[j], nw.xs[j])));
  return nw;
}
// the worker side of a par branch
function parMain() {
  const d = WT.workerData;
  PARW = { port: d.port, ctl: new Int32Array(d.sab), i: d.i };
  budRestore(d.bud);
  FUEL = d.fuel;
  const s0 = BUD.length ? BUD[0][0] : 0n;
  let err = null, w = null;
  try { w = PAR_FNS[d.fn](decV(d.env), decV(d.extra)); } catch (e) { if (!isFlow(e)) throw e; err = e; }
  const s1 = BUD.length ? BUD[0][0] : 0n;
  let msg;
  const used = d.fuel - FUEL;
  try { msg = encV(["done", encFlow(err), s1 - s0, w, used]); }
  catch (e) { if (!isFlow(e)) throw e; msg = encV(["done", encFlow(err !== null ? err : e), s1 - s0, null, used]); }
  d.port.postMessage(msg);
  Atomics.add(PARW.ctl, 0, 1); Atomics.notify(PARW.ctl, 0);
  d.port.close();
}
// a one-way synchronous link: post, raise the flag, the other side waits on it
function linkSend(port, flag, i, msg) { port.postMessage(msg); Atomics.store(flag, i, 1); Atomics.notify(flag, i); }
function linkRecv(port, flag, i) {
  for (;;) {
    while (Atomics.load(flag, i) === 0) Atomics.wait(flag, i, 0);
    Atomics.store(flag, i, 0);
    const m = WT.receiveMessageOnPort(port);
    if (m !== undefined) return m.message;
  }
}
let TASK = null;
let ATOMIC = null;
let HSENDS = null;
function comm(op) {
  if (TASK === null) return trap("concurrency is not available at comptime");
  let e;
  try { e = encV([op, TASK.dl.map(([f, x]) => [f, x])]); } catch (x) { throw x; }
  linkSend(TASK.req, TASK.flag, 0, e);
  const [tick, rep] = decV(linkRecv(TASK.rep, TASK.flag, 1));
  TASK.now = tick;
  if (rep[0] === "cancel") throw new Cancel(rep[1]);
  if (rep[0] === "failed") throw new Fail(rep[1]);
  return rep;
}
function fresh_(kind, line) { const idx = TASK.created++; return hid(kind + "|" + line + "|" + TASK.id + "|" + idx); }
function chanNew(cap, line) {
  if (TASK === null) return trap("channels are not available at comptime");
  return new Chan(fresh_("chan", line), markerInt(cap()));
}
function sessionNew(cap, line) {
  if (TASK === null) return trap("channels are not available at comptime");
  const n = markerInt(cap());
  const a = fresh_("chan", line), b = fresh_("chan", line);
  return new Tup([new Session(a, b, n), new Session(b, a, n)]);
}
function ord_(key) { const k = TASK.ords[key] || 0; TASK.ords[key] = k + 1; return k; }
// stable path of C.11.11: supervisor position (or type), creating task, ordinal
function actorNew(ty, pos) {
  if (TASK === null) return trap("cannot call " + ty + " at comptime");
  TASK.created++;
  return new Actor(hid("actor|" + pos + "|" + TASK.id + "|" + ord_("actor|" + pos)), ty);
}
function send(t, v) {
  charge(8n);
  if (t instanceof Session) t = new Chan(t.out, t.cap);
  if (ATOMIC !== null) { ATOMIC.push([t, v]); return; }
  if (HSENDS !== null) { HSENDS.push([t, v]); return; }
  comm(["send", t, v]);
}
function recv(c) {
  charge(8n);
  const session = c instanceof Session;
  if (session) c = new Chan(c.inn, c.cap);
  const rp = comm(["recv", c]);
  if (rp[0] !== "val") return trap("bad receive");
  const v = rp[1];
  if (session) { if (v instanceof Opt && v.x !== null) return v.x; return trap("session closed"); }
  return v;
}
function sessionChoose(r, label) {
  if (!(r instanceof Session)) return trap(show(r) + " has no method choose at comptime");
  charge(8n);
  comm(["send", new Chan(r.out, r.cap), new Rec(label, [])]);
  return UNIT;
}
function spawn_(f, args, prio, site) {
  charge(32n);
  if (TASK === null) return trap("concurrency is not available at comptime");
  TASK.created++; const idx = ord_("spawn");
  let p = BigInt(prio); p = p < 0n ? 0n : p > 255n ? 255n : p;
  const rp = comm(["spawn", f, args, Number(p), site, idx, TASK.scopes.length ? TASK.scopes[TASK.scopes.length - 1] : 0]);
  return new Task(rp[1]);
}
function select_(arms, line, nondet) {
  let k = 0;
  if (TASK !== null) { k = TASK.selK[line] || 0; TASK.selK[line] = k + 1; }
  const a2 = arms.map((a) => a[0] === "recv" && a[1] instanceof Session ? ["recv", new Chan(a[1].inn, a[1].cap), a[2]]
    : a[0] === "send" && a[1] instanceof Session ? ["send", new Chan(a[1].out, a[1].cap), a[2]] : a);
  const rp = comm(["select", a2, [line, k], nondet]);
  return [rp[1], rp[2] === null ? NONE : rp[2]];
}
function sarmAfter(n) { const x = markerInt(n); return ["after", Number(x < 0n ? 0n : x)]; }
function dlEnter(n) {
  if (TASK === null) return -2;
  const f = TASK.nextFrame++;
  TASK.dl.push([f, TASK.now + Number(n < 0n ? 0n : n)]);
  return f;
}
function dlExit() { if (TASK !== null) TASK.dl.pop(); }
function scopeEnter() { if (TASK === null) return 0; const f = TASK.nextFrame++; TASK.scopes.push(f); return f; }
function scopeExit() { if (TASK !== null) TASK.scopes.pop(); }
function awaitScope(sid) { if (TASK !== null) comm(["await", sid]); }
function atomicEnter() { const o = ATOMIC; ATOMIC = []; return o; }
function atomicRestore(o) { const b = ATOMIC || []; ATOMIC = o; return b; }
function atomicCommit(buf) {
  if (ATOMIC !== null) { ATOMIC.push(...buf); return; }
  if (TASK !== null) { comm(["atomic", buf]); return; }
  trap("concurrency is not available at comptime");
}
// the worker side: run one task
function taskMain() {
  const d = WT.workerData;
  const flag = new Int32Array(d.sab);
  TASK = { id: d.id, req: d.req, rep: d.rep, flag, created: 0, ords: {}, now: 0, dl: [], nextFrame: 1, scopes: [0], selK: {} };
  const [tick, first] = decV(linkRecv(d.rep, flag, 1));
  let r;
  if (first[0] === "go") {
    TASK.now = tick;
    try { const a = decV(d.args); for (let i = 0; i <= a.length; i++) fuel(); const v = FNS[d.f](null, a)[0]; awaitScope(0); r = encV(["ok", v]); }
    catch (e) { if (!isFlow(e)) throw e; r = encV(["err", e instanceof Cancel, why(e)]); }
  } else r = encV(["err", true, "cancelled"]);
  linkSend(d.req, flag, 0, { k: "arr", xs: [{ k: "arr", xs: ["done", r] }, { k: "arr", xs: [] }] });
}
class SlotJ {
  constructor(id, prio, parent, scope) {
    Object.assign(this, { id, prio, parent, scope, st: ["ready", ["go"]], result: null, joined: false, cancel: false, holding: null, released: false, since: 0 });
  }
}
class SchedJ {
  constructor(links) {
    Object.assign(this, { slots: [], chans: new Map(), staged: [], outputs: [], rng: BigInt(0x5eed | 1), tick: 0, stop: null, maxTicks: 100000,
      actors: new Map(), mailStaged: [], txns: [], txnStaged: [], heldLinks: [], parted: [], links, restarts: new Map(), alog: [], rlog: [], rrep: restartInputs(), events: eventInputs() });
  }
  spawnTask(f, args, id, prio, parent, scope) {
    const sl = new SlotJ(id, prio, parent, scope);
    const sab = new SharedArrayBuffer(8);
    const a = new WT.MessageChannel(), b = new WT.MessageChannel();
    sl.flag = new Int32Array(sab); sl.req = a.port1; sl.rep = b.port2;
    sl.worker = new WT.Worker(new URL(import.meta.url), { argv: process.argv.slice(2), workerData: { sab, req: a.port2, rep: b.port1, f, args: encV(args), id }, transferList: [a.port2, b.port1], resourceLimits: { stackSizeMb: 64 } });
    this.slots.push(sl);
  }
  post(sl, tick, rep) { linkSend(sl.rep, sl.flag, 1, encV([tick, rep])); }
  take(sl) { return decV(linkRecv(sl.req, sl.flag, 0)); }
  order() {
    const v = [];
    this.slots.forEach((s, i) => { if (s.st[0] !== "done") v.push(i); });
    v.sort((i, j) => this.slots[j].prio - this.slots[i].prio || (this.slots[i].id < this.slots[j].id ? -1 : this.slots[i].id > this.slots[j].id ? 1 : 0));
    return v;
  }
  chan(v) {
    if (!(v instanceof Chan)) return null;
    if (!this.chans.has(v.id)) this.chans.set(v.id, { cap: v.cap, buf: [], held: [], closed: false });
    return v.id;
  }
  space(c) { const st = this.chans.get(c); return BigInt(st.buf.length + this.staged.filter(([x]) => x === c).length) < st.cap; }
  takeMsg(c) {
    const st = this.chans.get(c);
    if (!st) return null;
    if (st.buf.length) {
      const m = st.buf.shift();
      if (st.held.length) { const [h, o] = st.held.shift(); st.buf.push(h); if (o !== null) this.slots[o].released = true; }
      return [m];
    }
    if (st.held.length) { const [h, o] = st.held.shift(); if (o !== null) this.slots[o].released = true; return [h]; }
    return null;
  }
  readable(c) { const st = this.chans.get(c); const has = st.buf.length > 0 || st.held.length > 0; return [has, !has && st.closed]; }
  route(from, target, m) {
    const to = target instanceof Actor ? (NODE_OF[target.ty] ?? 0) : 0;
    const key = Math.min(from, to) + "," + Math.max(from, to);
    if (from !== to && this.parted.includes(key)) { this.heldLinks.push([from, to, target, m]); return; }
    if (target instanceof Actor) this.mailStaged.push([target.id, target.ty, m]);
    else if (target instanceof Chan) {
      const c = this.chan(target);
      if (this.space(c)) this.staged.push([c, m]); else this.chans.get(c).held.push([m, null]);
    }
  }
  ensureActor(id, ty) { if (!this.actors.has(id)) this.actors.set(id, { ty, fields: actorInit(ty), mailbox: [], node: NODE_OF[ty] ?? 0 }); }
  handler(id, msg) {
    const st = this.actors.get(id);
    for (const h of ACTOR_HANDLERS[st.ty] || []) {
      OUT.length = 0; HSENDS = [];
      let r;
      try { r = h(st.fields, msg); } catch (e) { if (!isFlow(e)) throw e; HSENDS = null; OUT.length = 0; return ["err", actorRefusal(e)]; }
      const sends = HSENDS; HSENDS = null;
      if (r === null) continue;
      const outs = OUT.slice(); OUT.length = 0;
      return ["ok", [r, outs, sends]];
    }
    return ["ok", null];
  }
  onCrash(id, reason) {
    // restart as input (C.11.11)
    this.alog.push("Refused(" + reason + ")");
    const ty = this.actors.get(id).ty;
    const sv = SUPERVISORS[ty];
    if (!sv) return;
    const [sup, strat, tys, child, bound] = sv;
    const tick = this.tick;
    let dec;
    if (this.rrep !== null) {
      if (this.rrep.length && !SUP_TREE.includes(this.rrep[0][1] + "/" + this.rrep[0][2])) { this.stop = ["refused", "trap: upgrade removed supervisor position " + this.rrep[0][1] + "/" + this.rrep[0][2]]; return; }
      if (this.rrep.length === 0 || this.rrep[0][3] !== id || this.rrep[0][1] !== sup) { this.stop = ["refused", "replay diverged"]; return; }
      dec = this.rrep.shift();
    } else {
      const key = sup + "\u0000" + child;
      const h = (this.restarts.get(key) || []).filter((t) => RESTART_WINDOW === 0 || tick - t < RESTART_WINDOW);
      h.push(tick); this.restarts.set(key, h);
      const stop = h.length > bound;
      const pos = Math.max(tys.indexOf(ty), 0);
      const reset = [];
      if (!stop) for (const aid of [...this.actors.keys()].sort()) {
        const a = this.actors.get(aid);
        const hit = strat === "one_for_all" ? tys.includes(a.ty) : strat === "rest_for_one" ? tys.includes(a.ty) && tys.indexOf(a.ty) >= pos : aid === id;
        if (hit) reset.push(aid);
      }
      dec = [tick, sup, child, id, stop, reset];
    }
    this.rlog.push(dec);
    const c = this.rlog.filter((r) => r[1] === sup && r[2] === child && (RESTART_WINDOW === 0 || tick - r[0] < RESTART_WINDOW)).length;
    if (dec[4]) { this.alog.push("Stopped(supervisor " + sup + ")"); this.stop = ["refused", "supervisor " + sup + " stopped: child " + child + " restarted more than " + bound + " times"]; return; }
    this.alog.push("Restart(" + child + ", " + id + ", " + reason + ", tick " + tick + ", " + c + ")");
    for (const aid of dec[5]) { const a = this.actors.get(aid); if (a) a.fields = actorInit(a.ty); }
  }
  commitHandler(id, r) {
    if (r === null) return;
    const [fs, outs, sends] = r;
    const a = this.actors.get(id);
    a.fields = fs;
    this.outputs.push(...outs);
    for (const [t, v] of sends) this.route(a.node, t, v);
  }
  actorsPhase() {
    let progress = false;
    const busy = new Set();
    const txns = this.txns; this.txns = [];
    for (const txn of txns) {
      progress = true;
      const results = []; let refused = null;
      for (const [id, ty, m] of txn) {
        this.ensureActor(id, ty); busy.add(id);
        const r = this.handler(id, m);
        if (r[0] === "err") { refused = r[1]; break; }
        results.push([id, r[1]]);
      }
      if (refused !== null) {
        const failed = results.length;
        txn.forEach(([id], k) => { if (k === failed) this.onCrash(id, refused); else this.alog.push("Refused(" + refused + ")"); });
        if (this.stop !== null) break;
        continue;
      }
      for (const [id, r] of results) this.commitHandler(id, r);
    }
    for (const id of [...this.actors.keys()].sort()) {
      if (busy.has(id)) continue;
      const a = this.actors.get(id);
      if (!a.mailbox.length) continue;
      const m = a.mailbox.shift(); progress = true;
      const r = this.handler(id, m);
      if (r[0] === "ok") this.commitHandler(id, r[1]); else this.onCrash(id, r[1]);
      if (this.stop !== null) break;
    }
    return progress;
  }
  // HC-6: an event at this tick is logged as Input and staged to every live
  // actor that handles it, in actor id order; else Unhandled
  eventsPhase() {
    const evs = this.events.filter((e) => e[0] === this.tick).map((e) => e[1]);
    for (const v of evs) {
      this.alog.push("Input(" + show(v) + ")");
      const to = [...this.actors.keys()].sort().filter((id) => (ACTOR_EVENTS[this.actors.get(id).ty] || []).includes(v.n));
      if (!to.length) this.alog.push("Unhandled(" + show(v) + ")");
      for (const id of to) this.mailStaged.push([id, this.actors.get(id).ty, v]);
    }
    return evs.length > 0;
  }
  linksPhase() {
    let any = false;
    for (const [, part, a, b] of this.links.filter((e) => e[0] === this.tick)) {
      const x = NODE_NAMES.indexOf(a), y = NODE_NAMES.indexOf(b);
      if (x < 0 || y < 0) continue;
      any = true;
      const key = Math.min(x, y) + "," + Math.max(x, y);
      if (part) { if (!this.parted.includes(key)) this.parted.push(key); }
      else {
        this.parted = this.parted.filter((k) => k !== key);
        const held = this.heldLinks; this.heldLinks = [];
        for (const [f, t, target, m] of held) {
          if (Math.min(f, t) + "," + Math.max(f, t) === key) this.route(f, target, m); else this.heldLinks.push([f, t, target, m]);
        }
      }
    }
    return any;
  }
  failedValue(i) {
    const r = this.slots[i].result;
    return new Rec("TaskFailed", [["task", this.slots[i].id], ["reason", r !== null && r[0] === "err" ? r[1] : ""]]);
  }
  pickSelect(arms, key, nondet) {
    const ready = [];
    arms.forEach((a, k) => {
      let ok = false;
      if (a[0] === "recv") { const c = this.chan(a[1]); if (c !== null) { const [has, closed] = this.readable(c); ok = (has && a[2] !== 2) || (closed && a[2] !== 1); } }
      else if (a[0] === "send") { const c = this.chan(a[1]); if (c !== null) ok = this.space(c) && !this.chans.get(c).closed; }
      if (ok) ready.push(k);
    });
    if (!ready.length) return null;
    if (!nondet) {
      const n = arms.length, st = key[1] % n;
      for (let j = 0; j < n; j++) if (ready.includes((st + j) % n)) return (st + j) % n;
    }
    const M = (1n << 64n) - 1n;
    this.rng ^= (this.rng << 13n) & M; this.rng ^= this.rng >> 7n; this.rng ^= (this.rng << 17n) & M;
    return ready[Number(this.rng % BigInt(ready.length))];
  }
  handle(i, req) {
    const [op, dl] = req;
    const sl = this.slots[i];
    for (const [fid, e] of dl) {
      if (this.tick > e) {
        if (sl.holding !== null) { const c = sl.holding; sl.holding = null; const st = this.chans.get(c); if (st) st.held = st.held.filter((x) => x[1] !== i); }
        sl.released = false; sl.since = 0;
        return ["cancel", fid];
      }
    }
    if (sl.cancel) return ["cancel", CANCEL_TASK];
    switch (op[0]) {
      case "emit": this.outputs.push(op[1]); return ["val", UNIT];
      case "send": {
        const [, target, v] = op;
        if (target instanceof Actor) { this.route(0, target, v); return ["val", UNIT]; }
        const c = this.chan(target);
        if (c === null) return null;
        if (sl.holding !== null) { if (sl.released) { sl.released = false; sl.holding = null; return ["val", UNIT]; } return null; }
        if (this.chans.get(c).closed) { this.stop = ["refused", "trap: send on a closed channel"]; return ["cancel", CANCEL_TASK]; }
        if (this.space(c)) { this.staged.push([c, v]); return ["val", UNIT]; }
        this.chans.get(c).held.push([v, i]); sl.holding = c;
        return null;
      }
      case "recv": {
        const c = this.chan(op[1]);
        if (c === null) return null;
        const [has, closed] = this.readable(c);
        if (has) return ["val", new Opt(this.takeMsg(c)[0])];
        if (closed) return ["val", NONE];
        return null;
      }
      case "close": { const c = this.chan(op[1]); if (c === null) return null; this.chans.get(c).closed = true; return ["val", UNIT]; }
      case "select": {
        const [, arms, key, nondet] = op;
        const a = this.pickSelect(arms, key, nondet);
        if (a !== null) {
          sl.since = 0;
          const arm = arms[a];
          if (arm[0] === "recv") { const m = this.takeMsg(this.chan(arm[1])); return ["arm", a, m === null ? NONE : new Opt(m[0])]; }
          if (arm[0] === "send") { this.staged.push([this.chan(arm[1]), arm[2]]); return ["arm", a, null]; }
          return ["arm", a, null];
        }
        const d = arms.findIndex((x) => x[0] === "default");
        if (d >= 0) return ["arm", d, null];
        const j = arms.findIndex((x) => x[0] === "after");
        if (j >= 0) {
          if (sl.since === 0) sl.since = this.tick;
          if (this.tick - sl.since >= arms[j][1]) { sl.since = 0; return ["arm", j, null]; }
        }
        return null;
      }
      case "spawn": return ["spawned", hid("spawn|" + sl.id + "|" + op[5])];
      case "join": {
        const j = this.slots.findIndex((t) => t.id === op[1]);
        if (j < 0 || this.slots[j].st[0] !== "done") return null;
        this.slots[j].joined = true;
        const r = this.slots[j].result;
        return ["val", r[0] === "ok" ? r[1] : new Err(this.failedValue(j))];
      }
      case "await": {
        const kids = [];
        this.slots.forEach((t, x) => { if (t.parent === i && t.scope === op[1]) kids.push(x); });
        if (kids.some((x) => this.slots[x].st[0] !== "done")) return null;
        for (const x of kids) if (this.slots[x].result[0] === "err" && !this.slots[x].joined) { this.slots[x].joined = true; return ["failed", this.failedValue(x)]; }
        return ["val", UNIT];
      }
      case "atomic": {
        const sends = op[1];
        const chans = sends.map(([t]) => this.chan(t)).filter((c) => c !== null);
        for (const c of chans) {
          const need = chans.filter((x) => x === c).length, st = this.chans.get(c);
          const staged = this.staged.filter(([x]) => x === c).length;
          if (st.closed || BigInt(st.buf.length + staged + need) > st.cap) return null;
        }
        const txn = [];
        for (const [t, v] of sends) { if (t instanceof Actor) txn.push([t.id, t.ty, v]); else this.staged.push([this.chan(t), v]); }
        if (txn.length) this.txnStaged.push(txn);
        return ["val", UNIT];
      }
    }
    return null;
  }
  finishTask(i, r) {
    const res = r[0] === "ok" ? r : r[1] ? ["err", "cancelled"] : ["err", r[2]];
    const failed = r[0] === "err" && !r[1];
    const sl = this.slots[i]; sl.result = res;
    if (i === 0) { if (res[0] === "err") return ["refused", res[1]]; }
    else if (failed) this.slots.forEach((t, k) => { if (k !== i && t.parent === sl.parent && t.scope === sl.scope) t.cancel = true; });
    for (const t of this.slots) if (t.parent === i) t.cancel = true;
    return null;
  }
  run() {
    if (this.stop !== null) return this.stop;
    for (;;) {
      this.tick += 1;
      if (this.tick > this.maxTicks) return ["refused", "trap: tick limit"];
      let progress = this.linksPhase();
      progress = this.eventsPhase() || progress;
      const order = this.order();
      const moved = [];
      for (const i of order) {
        const sl = this.slots[i];
        if (sl.st[0] === "ready") { const rep = sl.st[1]; sl.st = ["done"]; this.post(sl, this.tick, rep); moved.push(i); }
      }
      for (const i of moved) { progress = true; this.slots[i].st = ["pending", this.take(this.slots[i])]; }
      progress = this.actorsPhase() || progress;
      if (this.stop !== null) return this.stop;
      for (const i of order) {
        const sl = this.slots[i];
        if (sl.st[0] !== "pending") continue;
        const req = sl.st[1]; sl.st = ["done"];
        if (req[0][0] === "done") { progress = true; const o = this.finishTask(i, req[0][1]); if (o !== null) return o; continue; }
        const rep = this.handle(i, req);
        if (this.stop !== null) return this.stop;
        if (rep === null) { sl.st = ["pending", req]; continue; }
        progress = true;
        if (req[0][0] === "spawn" && rep[0] === "spawned") {
          const [, f, args, prio, , , scope] = req[0];
          if (!(f in FNS)) return ["refused", "trap: no task function " + f];
          this.spawnTask(f, args, rep[1], prio, i, scope);
        }
        sl.st = ["ready", rep];
      }
      for (const [c, m] of this.staged) this.chans.get(c).buf.push(m);
      this.staged = [];
      for (const [id, ty, m] of this.mailStaged) { this.ensureActor(id, ty); this.actors.get(id).mailbox.push(m); }
      this.mailStaged = [];
      this.txns.push(...this.txnStaged); this.txnStaged = [];
      const allDone = this.slots.every((s) => s.st[0] === "done");
      const idle = [...this.actors.values()].every((a) => !a.mailbox.length) && !this.txns.length && !this.mailStaged.length;
      const fev = this.events.some((e) => e[0] > this.tick);
      const future = this.links.some((e) => e[0] > this.tick) || fev;
      if (allDone && idle && !fev && (!this.heldLinks.length || !future)) return ["done"];
      if (!progress && idle && !future) {
        const timers = this.slots.some((s) => s.st[0] === "pending" && (s.st[1][1].length > 0 || (s.st[1][0][0] === "select" && s.st[1][0][1].some((a) => a[0] === "after"))));
        if (!timers) return ["deadlock"];
      }
    }
  }
}
function actorInit(ty) { const f = ACTOR_INIT[ty]; return f ? f() : []; }
function runTasks(main) {
  const links = [];
  for (const a of process.argv) if (a.startsWith("--link=")) { const p = a.slice(7).split(","); links.push([Number(p[0]), p[1] === "partition", p[2], p[3]]); }
  const s = new SchedJ(links);
  const rec = restartTreeArg();
  if (s.rrep !== null && rec !== null) { const e = treeMigrates(rec, SUP_TREE); if (e !== null) s.stop = ["refused", e]; }
  s.spawnTask(main, [], hid("main"), 0, null, 0);
  const o = s.run();
  const lines = s.outputs.map((v) => "emit " + show(v));
  if (process.argv.includes("--actor-log")) for (const e of s.alog) lines.push("actorlog " + e);
  if (process.argv.includes("--restart-log")) lines.push("restartlog v1 tree=" + SUP_TREE.join(","));
  if (process.argv.includes("--restart-log")) for (const r of s.rlog) lines.push("restartin " + r[0] + "|" + r[1] + "|" + r[2] + "|" + r[3] + "|" + (r[4] ? 1 : 0) + "|" + r[5].join(","));
  lines.push(o[0] === "done" ? "outcome Done" : o[0] === "deadlock" ? "outcome Deadlock" : "outcome Refused: " + o[1]);
  process.stdout.write(lines.map((x) => x + "\n").join(""), () => process.exit(0));
}

// --restart-tree=Sup/child,...: the schema of the replayed restart log
function restartTreeArg() {
  for (const a of process.argv) if (a.startsWith("--restart-tree=")) return a.slice(15).split(",").filter((x) => x);
  return null;
}
function group_(v) {
  const g = [];
  for (const p of v) {
    const i = p.indexOf("/"), sup = i < 0 ? p : p.slice(0, i), c = i < 0 ? "" : p.slice(i + 1);
    if (g.length && g[g.length - 1][0] === sup) g[g.length - 1][1].push(c); else g.push([sup, [c]]);
  }
  return g;
}
// migration rule (C.11.11): every recorded position keeps its supervisor
// and child; new ones only after the recorded ones (conc::tree_migrates)
function treeMigrates(rec, cur) {
  const r = group_(rec), c = group_(cur);
  for (let k = 0; k < r.length; k++) for (let j = 0; j < r[k][1].length; j++) {
    if (!(k < c.length && c[k][0] === r[k][0] && c[k][1][j] === r[k][1][j])) {
      const p = r[k][0] + "/" + r[k][1][j];
      return (cur.includes(p) ? "trap: upgrade reordered supervisor position " : "trap: upgrade removed supervisor position ") + p;
    }
  }
  return null;
}
const INT_TYS = ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "usize"];
// program events (HC-6): --event=TICK|Name|field=value,...
function eventInputs() {
  const v = [];
  for (const a of process.argv) {
    if (!a.startsWith("--event=")) continue;
    const r = a.slice(8);
    const bad = (m) => { process.stderr.write("bad --event " + r + ": " + m + "\n"); process.exit(2); };
    const i1 = r.indexOf("|"), i2 = i1 < 0 ? -1 : r.indexOf("|", i1 + 1);
    const tick = Number(i1 < 0 ? r : r.slice(0, i1));
    if (!Number.isInteger(tick) || i1 < 0) bad("tick");
    const name = i2 < 0 ? r.slice(i1 + 1) : r.slice(i1 + 1, i2);
    if (!(name in EVENT_FIELDS)) bad("no event " + name);
    const given = {};
    for (const y of (i2 < 0 ? "" : r.slice(i2 + 1)).split(",")) { const e = y.indexOf("="); if (e > 0) given[y.slice(0, e)] = y.slice(e + 1); }
    const fs = [];
    for (const [f, t] of EVENT_FIELDS[name]) {
      if (!(f in given)) bad("missing field " + f);
      const x = given[f];
      let val;
      if (t === "bool") val = x === "true";
      else if (t === "text") val = x;
      else if (INT_TYS.includes(t)) { if (!/^-?[0-9]+$/.test(x)) bad("field " + f); val = new Int(BigInt(x), t); }
      else bad("field " + f + " has type " + t + ", not an integer, bool or text");
      fs.push([f, val]);
    }
    v.push([tick, new Rec(name, fs)]);
  }
  return v;
}
// logged restart inputs (C.11.11): --restart=TICK|SUP|CHILD|ID|STOP|RESET,...
function restartInputs() {
  const v = [];
  for (const a of process.argv) if (a.startsWith("--restart=")) {
    const p = a.slice(10).split("|");
    v.push([Number(p[0]), p[1], p[2], p[3], p[4] === "1", (p[5] || "").split(",").filter((x) => x)]);
  }
  if (v.length === 0 && !process.argv.includes("--restart-replay")) return null;
  return v;
}

// N9: a list or text built by any operation has at most 1,000,000 elements (code points)
function sizeOk(v) {
  if (v instanceof Lst && v.xs.length > 1000000) trap("size limit");
  if (typeof v === "string" && v.length > 1000000 && [...v].length > 1000000) trap("size limit");
  return v;
}
