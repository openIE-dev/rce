// rce-adapter/1 for the wasm playground (spec/checker-wasm, the
// checker compiled to wasm32 and loaded the way the browser playground loads
// it). It serves `check` only: the playground parses, checks and admits.
// RCE_WASM_PKG names the wasm-pack output directory (--target web).
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
const pkg = process.env.RCE_WASM_PKG;
const mod = await import(path.resolve(pkg, "rce_spec_check_wasm.js"));
mod.initSync({ module: fs.readFileSync(path.resolve(pkg, "rce_spec_check_wasm_bg.wasm")) });
const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  const req = JSON.parse(line);
  let r;
  if (req.op === "hello") r = { protocol: "rce-adapter/1", implementation: "rce wasm playground", ops: ["check"] };
  else if (req.op === "check") {
    const files = req.files || { "main.rce": req.source };
    const main = files["main.rce"];
    // other modules of the program: their paths, for `use` (check_with)
    const known = Object.entries(files).filter(([k]) => k !== "main.rce")
      .map(([, s]) => (s.match(/^module\s+([\w.]+)/m) || [])[1]).filter(Boolean).join(",");
    const v = JSON.parse(mod.check_with(main, known));
    r = v.ok ? { verdict: "ok", diagnostics: [] }
             : { verdict: "fit", diagnostics: v.errors.map(e => ({ code: e.rule || null, message: e.msg })) };
  } else r = { error: `op ${req.op} is not served by the playground` };
  r.id = req.id;
  process.stdout.write(JSON.stringify(r) + "\n");
}
