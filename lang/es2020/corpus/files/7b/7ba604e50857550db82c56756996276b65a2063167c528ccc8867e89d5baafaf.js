// SPDX-License-Identifier: MIT OR Apache-2.0
// Optional chaining short-circuits on null and undefined only; nullish
// coalescing replaces only null and undefined, unlike ||.
const o = { a: { b: 0, c: "", f() { return "called"; } }, arr: [10], n: null };
console.log(o?.a?.b, o.x?.y, o.x?.y.z.w, o.n?.foo, o.a?.f(), o.a?.g?.(), o.arr?.[0], o.missing?.[0]);
console.log(o.a.b ?? "default", o.a.c ?? "default", o.n ?? "default", o.x ?? "default", false ?? true, NaN ?? 1);
console.log(o.a.b || "default", o.a.c || "default", (null || undefined) ?? "both", 0 ?? (1 || 2));
let calls = 0;
const side = () => (++calls, "side");
console.log(null?.[side()], undefined?.x(side()), "x"?.length, calls);
console.log(delete o?.a?.b, o.a, delete o.nothing?.b);
