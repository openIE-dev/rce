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
const isFlow = (e) => e instanceof Trap || e instanceof Fail || e instanceof Stop || e instanceof Ret;

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
  const [d, p] = digitsOf(String(Math.abs(x)));
  return plain(x < 0, d, p);
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
  if (v instanceof F) return fdisplay(v.v);
  if (v instanceof UnitT) return "()";
  if (v instanceof Tup) return "(" + v.xs.map(show).join(", ") + ")";
  if (v instanceof Lst) return "[" + v.xs.map(show).join(", ") + "]";
  if (v instanceof Mp) return "{" + v.ps.map(([a, b]) => show(a) + ": " + show(b)).join(", ") + "}";
  if (v instanceof Opt) return v.x === null ? "none" : "some(" + show(v.x) + ")";
  if (v instanceof Rec) return v.fs.length === 0 ? v.n : v.n + "(" + v.fs.map(([a, b]) => a + ": " + show(b)).join(", ") + ")";
  if (v instanceof Err) return "error " + show(v.e);
  if (v instanceof FnV) return v.n;
  if (v instanceof Clo) return "<fn>";
  if (v instanceof Rng) return v.a + ".." + v.b;
  if (v instanceof F32) return f32display(v.v);
  if (v instanceof Gen) return "<iter>";
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
  if (a instanceof Mp) return a.ps.length === b.ps.length && a.ps.every(([k, v]) => b.ps.some(([k2, v2]) => eq(k, k2) && eq(v, v2)));
  if (a instanceof Opt) return a.x === null || b.x === null ? a.x === null && b.x === null : eq(a.x, b.x);
  if (a instanceof Rec) return a.n === b.n && a.fs.length === b.fs.length && a.fs.every(([p, x], i) => p === b.fs[i][0] && eq(x, b.fs[i][1]));
  if (a instanceof Err) return eq(a.e, b.e);
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
  if ((a instanceof Tup && b instanceof Tup) || (a instanceof Lst && b instanceof Lst)) {
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
function arith(op, a, b) {
  if (a instanceof Int && b instanceof Int) {
    const x = a.v, y = b.v, ty = a.t === "lit" ? b.t : a.t;
    const bits = ty === "i8" || ty === "u8" ? 8 : ty === "i16" || ty === "u16" ? 16 : ty === "i32" || ty === "u32" ? 32 : 64;
    switch (op) {
      case "+": return chk(x + y, ty);
      case "-": return chk(x - y, ty);
      case "*": return chk(x * y, ty);
      case "/": if (y === 0n) trap("division by zero"); return chk(x / y, ty);
      case "%": if (y === 0n) trap("division by zero"); return chk(x % y, ty);
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
  if (c instanceof Mp) return c.ps.some(([k]) => eq(k, x));
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
  for (let i = 0; i < out.length; i++) if (eq(out[i][0], k)) { out[i] = [out[i][0], v]; return out; }
  out.push([k, v]);
  return out;
}
function mkMap(pairs) { let ps = []; for (const [k, v] of pairs) ps = mapSet(ps, k, v); return new Mp(ps); }
function listRepeat(x, n) {
  if (n instanceof Int && n.v >= 0n && n.v <= 10000000n) return new Lst(Array(Number(n.v)).fill(x));
  return trap("bad repeat count");
}
const enc = new TextEncoder(), dec = new TextDecoder("utf-8", { fatal: true });
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
  if (c instanceof Mp) { for (const [a, v] of c.ps) if (eq(a, i)) return v; return trap("missing key " + show(i)); }
  return trap("cannot index " + show(c));
}
function slice(c, ta, tb) {
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
const truncBig = (f) => sat128(BigInt(Math.trunc(f)));
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
function fit(v, T) {
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
  return new Int(n, "i64");
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
  if (f instanceof FnV) return FNS[f.n](null, vals)[0];
  if (f instanceof Clo) return f.f(vals);
  return trap(show(f) + " is not callable");
}
function builtin(r, m, a) {
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
    if (m === "floor") return [new F(Math.floor(x)), null];
    if (m === "ceil") return [new F(Math.ceil(x)), null];
    if (m === "round") return [new F(fround(x)), null];
    if (m === "tostr") return [fdisplay(x), null];
  }
  if (isBool(r) && m === "tostr") return [r ? "true" : "false", null];
  if (isStr(r) && m === "len") return [new Int(BigInt(u8s(r).length), "usize"), null];
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
      case "lower": return [alower(r), null];
      case "reverse": return [Array.from(r).reverse().join(""), null];
      case "tostr": return [r, null];
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
      case "sum": { let acc = new Int(0n, "lit"); for (const x of xs) acc = arith("+", acc, x); return [acc, null]; }
      case "map": return [new Lst(xs.map((x) => apply(a[0], [x]))), null];
      case "filter": return [new Lst(xs.filter((x) => apply(a[0], [x]) === true)), null];
      case "fold": { let acc = a[0]; for (const x of xs) acc = apply(a[1], [acc, x]); return [acc, null]; }
    }
  }
  if (r instanceof Mp) {
    const ps = r.ps;
    switch (m) {
      case "len": return [new Int(BigInt(ps.length), "usize"), null];
      case "get": for (const [k, v] of ps) if (eq(k, a[0])) return [v, null]; return [new Err(new Rec("Missing", [])), null];
      case "contains": return [ps.some(([k]) => eq(k, a[0])), null];
      case "insert": return [UNIT, new Mp(mapSet(ps, a[0], a[1]))];
      case "remove": {
        const i = ps.findIndex(([k]) => eq(k, a[0]));
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
function callMethod(recv, m, args) {
  const head = recv instanceof Rec ? (KIND_OF[recv.n] ?? recv.n) : recv instanceof Lst ? "List" : null;
  if (head !== null && METHODS[head + "." + m]) { const [v, outs, s] = METHODS[head + "." + m](recv, args); return [v, outs, s]; }
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
    case "bind": b[P[1]] = v; return true;
    case "tag": return v instanceof Rec && v.n === P[1];
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
  if (v instanceof Lst) return v.xs;
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
    const b = {};
    if (!pmatch(P, x, b)) continue;
    const args = names.map((n) => (n in b ? b[n] : UNIT));
    if (cond !== null && !truthy(cond(...args))) continue;
    out.push(body(...args));
  }
  return new Lst(out);
}
const catch_ = (x, h) => (x instanceof Err ? h(x.e) : x);
function try_(x, els) {
  if (x instanceof Err) throw new Fail(els ? els() : x.e);
  if (x instanceof Opt) { if (x.x === null) { if (els) throw new Fail(els()); trap("try on none"); } return x.x; }
  return x;
}
function tryEval(th) { try { return th(); } catch (e) { if (isFlow(e)) return null; throw e; } }
function compound(op, cur, r) { const t = cur instanceof Int ? cur.t : "lit"; const x = arith(op, cur, r); return x instanceof Int ? chk(x.v, t) : x; }
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
  if (v instanceof Tup || v instanceof Lst) return seq(v.xs.map(jsonSize), v.xs.length);
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
function refusal(e) {
  if (e instanceof Fail) return e.v instanceof Rec ? e.v.n : "?";
  if (e instanceof Trap) return "trap(" + e.m + ")";
  if (e instanceof Stop) return "trap(Stop)";
  return "trap(Ret)";
}
const OUT = [];
const emit = (v) => { OUT.push(v); };
const LINES = [];
const say = (s) => { LINES.push(s); };
function obs(label, th) {
  try { say(label + " = " + show(th())); } catch (e) { if (!isFlow(e)) throw e; say(label + " " + why(e)); }
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
  say("example " + line);
  let s;
  try { s = initialState(); } catch (e) { if (!isFlow(e)) throw e; say("initial " + why(e)); return; }
  say("0 (initial) " + showState(s));
  for (let k = 1; k <= events.length; k++) {
    const [name, th] = events[k - 1];
    let ev;
    try { ev = th(); } catch (e) { if (!isFlow(e)) throw e; say(k + " " + name + " " + why(e)); return; }
    if (ev === null) { say(k + " " + name + " not an event"); return; }
    let r;
    try { r = delta(s, ev); } catch (e) { if (e instanceof Refuse) { say(k + " " + show(ev) + " refused: " + e.r); return; } throw e; }
    s = r[0];
    say(k + " " + show(ev) + " " + showState(s));
    for (const o of r[1]) say("  emit " + show(o));
  }
}
function runMain(f) {
  say("main");
  OUT.length = 0;
  try {
    const v = f(null, [])[0];
    for (const o of OUT) say("emit " + show(o));
    say("main = " + show(v));
  } catch (e) {
    if (!isFlow(e)) throw e;
    for (const o of OUT) say("emit " + show(o));
    say("main " + why(e));
  }
}
