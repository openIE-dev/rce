// Host-side adapter for a module printed by `rce emit js FILE`.
//
// The emitted module is the step function of the program (spec 10.5): it
// exports initialState() and delta(state, event), and the host calls them
// (Appendix E). This adapter is the boundary: JSON or plain values in,
// emitted values to the step, display text and canonical JSON out.
// Integers are BigInt inside the emitted code, so JSON is read with the
// source text of each number (JSON.parse source access, Node 21 and later)
// and a u64 or i64 above 2^53 keeps every digit.

const INT_TYPES = new Set(["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "usize"]);

export class BadEvent extends Error {}

/** Parse JSON keeping integers exact: every integer literal becomes a BigInt. */
export function decode(text) {
  return JSON.parse(text, (k, v, c) => (typeof v === "number" && c && /^-?\d+$/.test(c.source) ? BigInt(c.source) : v));
}

/** JSON text from plain values, keeping what JSON.stringify loses: a BigInt
 * keeps every digit, and -0 stays -0.0 (JSON.stringify(-0) is "0", and the
 * program can tell -0.0 from 0.0, for example as map keys). */
export function encode(v) {
  return JSON.stringify(v, (k, x) => (typeof x === "bigint" ? JSON.rawJSON(x.toString()) : Object.is(x, -0) ? JSON.rawJSON("-0.0") : x));
}

export class Program {
  constructor(m) {
    this.m = m;
    m.fuelReset();
    this.state = m.initialState();
  }

  /** An event of the program from {type: NAME, field: value, ...}. */
  event(obj) {
    const m = this.m;
    const name = obj && obj.type;
    if (!Object.hasOwn(m.EVENT_FIELDS, name)) throw new BadEvent("no event " + name);
    const fs = [];
    for (const [f, t] of m.EVENT_FIELDS[name]) {
      if (!Object.hasOwn(obj, f)) throw new BadEvent("missing field " + f);
      fs.push([f, this.value(t, obj[f], f)]);
    }
    return new m.Rec(name, fs);
  }

  value(t, x, f) {
    const m = this.m;
    if (t === "text") { if (typeof x !== "string") throw new BadEvent("field " + f + " is text"); return x; }
    if (t === "bool") { if (typeof x !== "boolean") throw new BadEvent("field " + f + " is bool"); return x; }
    if (INT_TYPES.has(t)) {
      if (typeof x === "number" && !Number.isSafeInteger(x)) throw new BadEvent("field " + f + " is an integer (exact)");
      if (typeof x !== "bigint" && typeof x !== "number") throw new BadEvent("field " + f + " is an integer");
      try { return m.chk(BigInt(x), t); } catch (e) { throw new BadEvent("field " + f + ": " + e.m); }
    }
    if (t === "f64") {
      if (typeof x !== "number" && typeof x !== "bigint") throw new BadEvent("field " + f + " is f64");
      return new m.F(Number(x));
    }
    throw new BadEvent("field " + f + " has type " + t + ", which this adapter does not read");
  }

  /** Apply one event: {kind: "applied", emits} or {kind: "kept", reason}; a kept step keeps the state (HC-21). */
  step(ev) {
    const m = this.m;
    m.fuelReset(); // HC-7a: full fuel for each input
    try {
      const [s, outs] = m.delta(this.state, ev);
      this.state = s;
      return { kind: "applied", emits: outs };
    } catch (e) {
      if (e instanceof m.Kept) return { kind: "kept", reason: e.r };
      throw e;
    }
  }

  show(v) { return this.m.show(v); }
  showState() { return this.m.showState(this.state); }
  initialLine() { return "0 (initial) " + this.showState(); }

  /** The state as canonical JSON text: {region: {field: value}}, keys sorted. */
  stateJson() {
    const m = this.m;
    return "{" + this.state.map(([r, fs]) => [r, fs]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .map(([r, fs]) => JSON.stringify(r) + ":" + m.fvJson(m.formValue(new m.Rec(r, fs)))).join(",") + "}";
  }

  /** The lines `rce run` prints for event number k. */
  lines(k, ev, o) {
    if (o.kind === "kept") return [k + " " + this.show(ev) + " kept: " + o.reason];
    return [k + " " + this.show(ev) + " " + this.showState(), ...o.emits.map((x) => "  emit " + this.show(x))];
  }
}

export async function load(path) {
  const url = path.startsWith("file:") ? path : (await import("node:url")).pathToFileURL(path).href;
  return new Program(await import(url));
}
