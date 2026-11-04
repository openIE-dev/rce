// Node-side smoke test for the @rce/runtime bundle. Verifies the wasm
// loads and the CPU path produces the expected MLP output without needing
// a browser. WebGPU path is skipped (no `navigator.gpu` in vanilla Node).
//
// Run from repo root:
//   node crates/rce-vm-web/demo/node_smoke.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(here, "..", "pkg");

// wasm-bindgen --target web emits ESM that uses `import.meta.url` for the
// default wasm fetch path. In Node we hand it the bytes explicitly.
const wasmBytes = await readFile(resolve(pkgDir, "rce_vm_web_bg.wasm"));

const mod = await import(resolve(pkgDir, "rce_vm_web.js"));
await mod.default({ module_or_path: wasmBytes });

const { Vm, Module } = mod;

const fgBytes = await readFile(resolve(here, "mlp.fg"));
const stBytes = await readFile(resolve(here, "mlp.safetensors"));

const vm = new Vm();
const module = Module.load(new Uint8Array(fgBytes), new Uint8Array(stBytes));

console.log(`module=${module.name}  nodes=${module.node_count}  params=(${module.params})`);

const inputs = JSON.stringify({ x: [[1, 2, 3, 4]] });
const t0 = performance.now();
const out = vm.run(module, inputs);
const dt = (performance.now() - t0).toFixed(2);
console.log(`[CPU] ${dt} ms => ${out}`);

const expected = [
  0.032058604, 0.087144323, 0.236882836, 0.643914236,
];
const got = JSON.parse(out)[0];
for (let i = 0; i < expected.length; i++) {
  if (Math.abs(got[i] - expected[i]) > 1e-3) {
    console.error(`mismatch at index ${i}: got ${got[i]}, expected ${expected[i]}`);
    process.exit(1);
  }
}
console.log("OK — output matches CLI reference within 1e-3");
