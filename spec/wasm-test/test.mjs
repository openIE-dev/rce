// Runs the committed wasm builds over the spec corpora: the tree-sitter
// grammar (tree-sitter-rce.wasm) and the reference checker (checker-wasm/pkg).
// A stale build fails here. Run: npm ci && npm test (from spec/wasm-test).
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Parser, Language } from "web-tree-sitter";
import init, { check } from "../checker-wasm/pkg/rce_spec_check_wasm.js";

const spec = join(dirname(fileURLToPath(import.meta.url)), "..");
const V06 = "# rce-version: 0.6\n", V07 = "# rce-version: 0.7\n";
const files = (d) =>
  readdirSync(join(spec, d)).filter((f) => f.endsWith(".rce")).sort().map((f) => join(spec, d, f));
// `pre` replaces a declared version line, or is prepended when there is none.
const read = (f, pre = "") => { const s = readFileSync(f, "utf8"); return pre ? pre + s.replace(/^# rce-version:[^\n]*\n/, "") : s; };

await Parser.init();
const parser = new Parser();
parser.setLanguage(await Language.load(join(spec, "tree-sitter/tree-sitter-rce.wasm")));
await init({ module_or_path: readFileSync(join(spec, "checker-wasm/pkg/rce_spec_check_wasm_bg.wasm")) });

let fails = 0, runs = 0;
const expect = (what, f, ok, want) => {
  runs++;
  if (ok !== want) { fails++; console.log(`FAIL ${what}: ${f.startsWith(spec) ? f.slice(spec.length + 1) : f} (got ${ok ? "accept" : "refuse"})`); }
};
const parses = (src) => !parser.parse(src).rootNode.hasError;
const checks = (src) => JSON.parse(check(src)).ok;
const stage = (src) => JSON.parse(check(src)).stage;

// 0.5 corpora.
const valid = files("examples");
// the multi-module package of example 26 parses file by file; it checks as a program (rce check DIR)
for (const f of files("examples/26_use_module/shop")) expect("ts accept", f, parses(read(f)), true);
for (const f of valid) { expect("ts accept", f, parses(read(f)), true); expect("check accept", f, checks(read(f)), true); }
for (const f of files("examples/reject")) { expect("ts refuse", f, parses(read(f)), false); expect("check refuse", f, checks(read(f)), false); }
for (const d of ["examples/ill_typed", "examples/inadmissible"])
  for (const f of files(d)) { expect("ts accept", f, parses(read(f)), true); expect("check refuse", f, checks(read(f)), false); }

// 0.6 corpora, declared per module (spec 2.7).
for (const d of ["conc06", "conc06b"]) {
  for (const f of files(`${d}/accept`)) { expect("ts accept 0.6", f, parses(read(f, V06)), true); expect("check accept 0.6", f, checks(read(f, V06)), true); }
  for (const f of files(`${d}/reject`)) expect("check refuse 0.6", f, stage(read(f, V06)) !== "ok", true);
  for (const f of files(`${d}/ill_typed`)) expect("check refuse 0.6", f, checks(read(f, V06)), false);
}

// 0.7 has the 0.6 syntax and rules (C.13): the same corpora under 0.7.
for (const d of ["conc06", "conc06b"]) {
  for (const f of files(`${d}/accept`)) { expect("ts accept 0.7", f, parses(read(f, V07)), true); expect("check accept 0.7", f, checks(read(f, V07)), true); }
  for (const f of files(`${d}/ill_typed`)) expect("check refuse 0.7", f, checks(read(f, V07)), false);
}

// The version gate: 0.6 words are names in an undeclared module.
const par05 = "fn f() -> i64\n    var par: i64 = 1\n    return par\n";
expect("ts gate", "par in 0.5", parses(par05), true);
expect("gate", "par in 0.5", checks("fn f() -> i64\n    var par: i64 = 1\n    return par\n"), true);
expect("ts gate", "par in 0.7 is reserved", parses(V07 + par05), false);
expect("gate", "unknown version", checks("# rce-version: 9.9\nfn f() -> i64\n    return 1\n"), false);

console.log(`${runs - fails}/${runs} wasm checks passed`);
process.exit(fails ? 1 : 0);
