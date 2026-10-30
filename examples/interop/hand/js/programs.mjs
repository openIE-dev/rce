// The eight programs written by hand as idiomatic JavaScript, for
// comparison with what `rce emit js` prints: plain objects for state, a
// switch on the event type, and `throw new Kept(reason)` when an event does
// not apply. Integers that can exceed 2^53 (the bank's i64) are BigInt; map
// keys that RCE keeps apart and a JS Map does not (-0 and 0) are keyed by
// text. Checked on the streams of streams.json only (no overflow traps, no fuel).
export class Kept extends Error { constructor(r) { super(r); this.r = r; } }

// ---- counter (01_counter.rce)
const counter = {
  init: () => [{ $: "Counter", n: 0n }],
  step([s], e) {
    switch (e.type) {
      case "Inc": return [[{ ...s, n: s.n + 1n }], []];
      case "Dec": if (s.n === 0n) throw new Kept("Underflow"); return [[{ ...s, n: s.n - 1n }], []];
    }
  },
};

// ---- bank (04_bank_account.rce)
const bank = {
  init: () => [{ $: "Account", balance: 0n }],
  step([s], e) {
    const a = BigInt(e.amount);
    switch (e.type) {
      case "Deposit": return [[a > 0n ? { ...s, balance: s.balance + a } : s], []];
      case "Withdraw":
        if (a > s.balance) throw new Kept("Insufficient");
        return [[{ ...s, balance: s.balance - a }], [["Withdrawn", { amount: a }]]];
    }
  },
};

// ---- multi_region (16_multi_region.rce)
const multi_region = {
  init: () => [{ $: "Cart", items: [] }, { $: "Stats", adds: 0n, removes: 0n }],
  step([cart, stats], e) {
    switch (e.type) {
      case "AddItem": return [[{ ...cart, items: [...cart.items, e.name] }, { ...stats, adds: stats.adds + 1n }], []];
      case "RemoveItem": return [[{ ...cart, items: cart.items.filter((i) => i !== e.name) }, { ...stats, removes: stats.removes + 1n }], []];
      case "Reset": return [[{ ...cart, items: [] }, stats], []];
    }
  },
};

// ---- traffic_light (17_traffic_light.rce)
const LIGHTS = ["Red", "Green", "Yellow"];
const traffic_light = {
  init: () => [{ $: "Signal", light: { kind: "Red" } }],
  step: ([s]) => [[{ ...s, light: { kind: LIGHTS[(LIGHTS.indexOf(s.light.kind) + 1) % 3] } }], []],
};

// ---- door (30_state_machine_guarded.rce)
const door = {
  init: () => [{ $: "Door", open: false, locked: false }],
  step([s], e) {
    switch (e.type) {
      case "Open": if (s.locked) throw new Kept("Locked"); return [[{ ...s, open: true }], []];
      case "Close": return [[{ ...s, open: false }], []];
      case "Lock": if (s.open) throw new Kept("DoorOpen"); return [[{ ...s, locked: true }], []];
      case "Unlock": return [[{ ...s, locked: false }], []];
    }
  },
};

// ---- inventory (31_inventory.rce)
const inventory = {
  init: () => [{ $: "Inventory", stock: new Map() }],
  step([s], e) {
    const q = BigInt(e.qty), have = s.stock.get(e.sku) ?? 0n;
    switch (e.type) {
      case "Receive": return [[{ ...s, stock: new Map(s.stock).set(e.sku, have + q) }], []];
      case "Ship":
        if (have < q) throw new Kept("OutOfStock");
        return [[{ ...s, stock: new Map(s.stock).set(e.sku, have - q) }], [["Shipped", { sku: e.sku, qty: q }]]];
    }
  },
};

// ---- rcedb (55_rcedb.rce)
const rcedb = {
  init: () => [{ $: "Db", rows: new Map(), writes: 0n, durable: 0n }],
  step([s], e) {
    switch (e.type) {
      case "Put": return [[{ ...s, rows: new Map(s.rows).set(e.key, e.val), writes: s.writes + 1n }], []];
      case "Delete": {
        if (!s.rows.has(e.key)) throw new Kept("Missing");
        const rows = new Map(s.rows);
        rows.delete(e.key);
        return [[{ ...s, rows, writes: s.writes + 1n }], []];
      }
      case "Synced": { const n = BigInt(e.upto); return [[n > s.durable ? { ...s, durable: n } : s], []]; }
      case "Scan": return [[s], [...s.rows.keys()].sort().filter((k) => k >= e.start).map((k) => ["Row", { key: k, val: s.rows.get(k) }])];
    }
  },
};

// ---- float_keys (58_float_keys.rce)
// a float key as text, so -0 and 0 stay two keys (a Map uses SameValueZero)
export const fkey = (x) => (Object.is(x, -0) ? "-0" : String(x));
const float_keys = {
  init: () => [{ $: "Tally", m: new Map() }],
  step([s], e) {
    const x = e.type === "Add" ? Number(e.x) : Number(e.x) * -1;
    const k = fkey(x);
    const m = new Map(s.m);
    m.set(k, { x, n: (m.get(k)?.n ?? 0n) + 1n });
    return [[{ ...s, m }], []];
  },
};

// ---- end
export const PROGRAMS = { counter, bank, multi_region, traffic_light, door, inventory, rcedb, float_keys };
