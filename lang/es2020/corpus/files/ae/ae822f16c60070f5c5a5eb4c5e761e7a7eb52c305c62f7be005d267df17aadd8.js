// SPDX-License-Identifier: MIT OR Apache-2.0
// Generators run lazily: yield, next with a value, return, throw, yield*,
// and iterables made with Symbol.iterator.
function* counter(n) {
  let i = 0;
  while (i < n) {
    const got = yield i;
    i += got === undefined ? 1 : got;
  }
  return "done";
}
const g = counter(10);
console.log(JSON.stringify([g.next(), g.next(), g.next(3), g.next()]));
console.log(JSON.stringify([...counter(3)]), JSON.stringify(counter(1).return("early")));
function* outer() {
  const r = yield* counter(2);
  yield r;
}
console.log([...outer()].join(","));
const log = [];
function* lazy() {
  for (let i = 0; i < 3; i++) {
    log.push("make " + i);
    yield i;
  }
}
const it = lazy();
log.push("created");
log.push("got " + it.next().value);
console.log(log.join("; "), [...it].join(","), log[log.length - 1]);
function* guarded() {
  try {
    yield 1;
    yield 2;
  } catch (e) {
    yield "caught " + e;
  } finally {
    log.push("cleanup");
  }
}
const gg = guarded();
console.log(gg.next().value, gg.throw("boom").value, gg.next().done, log[log.length - 1]);
const range = { from: 1, to: 4, *[Symbol.iterator]() { for (let v = this.from; v <= this.to; v++) yield v; } };
console.log([...range], Math.max(...range), Array.from(range, (v) => v * v));
