// SPDX-License-Identifier: MIT OR Apache-2.0
// The order of promise jobs: synchronous code first, then reactions in the
// order they were queued, with each await taking its own turns.
const log = [];
log.push("sync 1");
Promise.resolve().then(() => log.push("then A1")).then(() => log.push("then A2"));
Promise.resolve().then(() => log.push("then B1")).then(() => log.push("then B2"));
(async () => {
  log.push("async start");
  await null;
  log.push("after await 1");
  await Promise.resolve();
  log.push("after await 2");
})();
new Promise((resolve) => {
  log.push("executor runs at once");
  resolve("value");
}).then((v) => log.push("resolved " + v));
Promise.reject(new Error("no")).catch((e) => log.push("caught " + e.message));
Promise.allSettled([Promise.resolve(1), Promise.reject(2)]).then((r) => log.push(JSON.stringify(r)));
Promise.all([1, Promise.resolve(2)]).then((v) => log.push("all " + v));
Promise.race([new Promise(() => {}), Promise.resolve("race")]).then((v) => log.push(v));
log.push("sync 2");
Promise.resolve().then(() => Promise.resolve().then(() => Promise.resolve())).then(() => {
  log.push("last");
  for (const line of log) console.log(line);
});
