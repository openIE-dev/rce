// Load an emitted JS module where a browser would: in a fresh realm (node:vm)
// whose globals are only the language's own and the web's TextEncoder,
// TextDecoder and console; no `process`, no Node modules. Then apply a
// stream and print the transcript. Needs node --experimental-vm-modules.
//   node --experimental-vm-modules browser_drive.mjs MODULE.mjs STREAM.json
import vm from "node:vm";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { Program, decode } from "./rce_host.mjs";

const ctx = vm.createContext({ TextEncoder, TextDecoder, console });
const mod = new vm.SourceTextModule(readFileSync(process.argv[2], "utf8"), {
  context: ctx,
  identifier: pathToFileURL(process.argv[2]).href,
  initializeImportMeta: (meta) => { meta.url = pathToFileURL(process.argv[2]).href; },
});
await mod.link(() => { throw new Error("the emitted module imports nothing"); });
await mod.evaluate();
if (vm.runInContext("typeof process", ctx) !== "undefined") throw new Error("the realm has process");
const p = new Program(mod.namespace);
const lines = [p.initialLine()];
let k = 0;
for (const obj of decode(readFileSync(process.argv[3], "utf8"))) {
  const ev = p.event(obj);
  k += 1;
  lines.push(...p.lines(k, ev, p.step(ev)));
}
process.stdout.write(lines.map((x) => x + "\n").join(""));
