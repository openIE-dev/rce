// Reference ECMAScript number text and RFC 8785 canonical forms for
// data.json, from Node 22 and the canonicalize 2.0.0 package (npm; an
// RFC 8785 implementation by an author of the RFC).
//
//   cd stdlib && NODE_PATH=<dir with node_modules/canonicalize> node scripts/gen/json_node.js
//
// Writes vectors/data/json/node22_es.json:
//   es: [IEEE 754 bits as 16 hex digits, String(x)] for seeded random and
//       edge doubles (ECMA-262 Number::toString, what JSON.stringify writes)
//   canon_num: [lexeme, canonicalize(JSON.parse(lexeme)) or null when the
//       lexeme is outside the double range]
//   canon: [JSON text, canonicalize(JSON.parse(text))] for seeded random
//       documents without duplicate member names
'use strict';
const fs = require('fs');
const path = require('path');
const canonicalize = require('canonicalize');

let s = 20261025 >>> 0;
function rnd() {
  // mulberry32
  s = (s + 0x6d2b79f5) >>> 0;
  let t = s;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function ri(n) { return Math.floor(rnd() * n); }
function pick(xs) { return xs[ri(xs.length)]; }

const buf = new DataView(new ArrayBuffer(8));
function bitsOf(x) { buf.setFloat64(0, x); return buf.getBigUint64(0).toString(16).padStart(16, '0'); }
function fromBits(hi, lo) { buf.setUint32(0, hi); buf.setUint32(4, lo); return buf.getFloat64(0); }

const es = [];
const edge = [0, -0, 1, -1, 0.1, 1e21, 1e-7, 1e-6, 9.999999999999999e20, 999999999999999900000, 1e20, 123e-20, 5e-324, -5e-324,
  1.7976931348623157e308, 2.2250738585072014e-308, 2.225073858507201e-308, 9007199254740992, 9007199254740993, 2 ** 53 - 1,
  4.35, 0.000001, 0.0000009999999999999999, 1.5, 100, 1e15, 1e16, 1e17, 12345678901234567890, 0.3, 2 / 3, Math.PI];
for (const x of edge) es.push([bitsOf(x), String(x)]);
while (es.length < 700) {
  const k = ri(4);
  let x;
  if (k === 0) {
    x = fromBits((ri(0x7ff00000) | (ri(2) << 31)) >>> 0, ri(4294967296) >>> 0);
  } else if (k === 1) {
    x = (rnd() - 0.5) * Math.pow(10, ri(50) - 25);
  } else if (k === 2) {
    x = ri(1e9) * Math.pow(10, ri(30) - 10);
  } else {
    x = fromBits(ri(0x100000) >>> 0, ri(4294967296) >>> 0);
  }
  if (!Number.isFinite(x)) continue;
  es.push([bitsOf(x), String(x)]);
}

function lexeme() {
  const k = ri(6);
  const sign = ri(3) === 0 ? '-' : '';
  if (k === 0) return sign + String(ri(1e6));
  if (k === 1) return sign + ri(10) + '.' + String(ri(1e9)) + pick(['e', 'E']) + pick(['', '+', '-']) + ri(330);
  if (k === 2) return sign + '0.' + '0'.repeat(ri(30)) + String(1 + ri(1e9));
  if (k === 3) return sign + String(1 + ri(9)) + Array.from({ length: 15 + ri(30) }, () => ri(10)).join('');
  if (k === 4) return pick(['1e400', '-1e400', '1e-400', '0', '-0', '0.0', '1e21', '1e-7', '1.7976931348623157e308', '1.7976931348623159e308', '5e-324', '2.4703282292062328e-324']);
  return sign + String(ri(1000)) + 'e' + (ri(40) - 20);
}
const canon_num = [];
for (let i = 0; i < 300; i++) {
  const t = lexeme();
  const v = JSON.parse(t);
  canon_num.push([t, Number.isFinite(v) ? canonicalize(v) : null]);
}

const CH = 'abcXYZ019 ~!@#$%^&*()_+-=[]{}|;:,.<>?/"\\'.split('').concat(
  Array.from({ length: 32 }, (_, i) => String.fromCharCode(i)),
  ['\x7f', '\x80', '\xe9', 'Ā', 'ࠀ', ' ', '﻿', '￿', '\u{10000}', '\u{1f600}', '\u{10ffff}', '', 'דּ', '中']);
function str(n) { let o = ''; for (let i = 0; i < n; i++) o += pick(CH); return o; }
function val(d) {
  const k = ri(d < 4 ? 8 : 5);
  if (k === 0) return 'null';
  if (k === 1) return pick(['true', 'false']);
  if (k === 2) return lexeme().replace(/^-?/, (m) => m).replace(/e4\d\d|e-4\d\d/i, 'e2');
  if (k === 3 || k === 4) return JSON.stringify(str(pick([0, 1, 2, 5, 9])));
  if (k === 5 || k === 6) return '[' + Array.from({ length: pick([0, 1, 2, 4]) }, () => val(d + 1)).join(',') + ']';
  const keys = [...new Set(Array.from({ length: pick([0, 1, 2, 3, 5]) }, () => str(pick([0, 1, 2, 3]))))];
  return '{' + keys.map((x) => JSON.stringify(x) + ':' + val(d + 1)).join(',') + '}';
}
const canon = [];
while (canon.length < 200) {
  const t = val(0);
  const v = JSON.parse(t);
  let ok = true;
  JSON.parse(t, (k, x) => { if (typeof x === 'number' && !Number.isFinite(x)) ok = false; return x; });
  if (!ok) continue;
  canon.push([t, canonicalize(v)]);
}

const out = { node: process.version, canonicalize: require('canonicalize/package.json').version, es, canon_num, canon };
fs.writeFileSync(path.join('vectors/data/json/node22_es.json'), JSON.stringify(out, null, 1) + '\n');
console.log('es', es.length, 'canon_num', canon_num.length, 'canon', canon.length);
