// Runs the committed wasm builds over the spec corpora: the tree-sitter
// grammar (tree-sitter-rce.wasm) and the reference checker (checker-wasm/pkg).
// A stale build fails here. Run: npm ci && npm test (from spec/wasm-test).
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Parser, Language } from "web-tree-sitter";
import init, { check } from "../checker-wasm/pkg/rce_spec_check_wasm.js";

const spec = join(dirname(fileURLToPath(import.meta.url)), "..");
const files = (d) =>
  readdirSync(join(spec, d)).filter((f) => f.endsWith(".rce")).sort().map((f) => join(spec, d, f));
const read = (f) => readFileSync(f, "utf8");

await Parser.init();
const parser = new Parser();
parser.setLanguage(await Language.load(join(spec, "../tree-sitter/tree-sitter-rce.wasm")));
await init({ module_or_path: readFileSync(join(spec, "checker-wasm/pkg/rce_spec_check_wasm_bg.wasm")) });

let fails = 0, runs = 0;
const expect = (what, f, ok, want) => {
  runs++;
  if (ok !== want) { fails++; console.log(`FAIL ${what}: ${f.startsWith(spec) ? f.slice(spec.length + 1) : f} (got ${ok ? "accept" : "requires change"})`); }
};
const parses = (src) => !parser.parse(src).rootNode.hasError;
const checks = (src) => JSON.parse(check(src)).ok;
const stage = (src) => JSON.parse(check(src)).stage;

// 0.5 corpora.
const valid = files("examples");
// the multi-module package of example 26 parses file by file; it checks as a program (rce check DIR)
for (const f of files("examples/26_use_module/shop")) expect("ts accept", f, parses(read(f)), true);
for (const f of valid) { expect("ts accept", f, parses(read(f)), true); expect("check accept", f, checks(read(f)), true); }
for (const f of files("examples/change/syntax")) { expect("ts requires change", f, parses(read(f)), false); expect("check requires change", f, checks(read(f)), false); }
for (const d of ["examples/change/types", "examples/change/admission"])
  for (const f of files(d)) { expect("ts accept", f, parses(read(f)), true); expect("check requires change", f, checks(read(f)), false); }

// The concurrency corpora (Appendix C): one language, so they are checked as written (spec 2.7).
for (const d of ["conc06", "conc06b"]) {
  for (const f of files(`${d}/accept`)) { expect("ts accept", f, parses(read(f)), true); expect("check accept", f, checks(read(f)), true); }
  for (const f of files(`${d}/change`)) expect("check requires change", f, stage(read(f)) !== "ok", true);
  for (const f of files(`${d}/ill_typed`)) expect("check requires change", f, checks(read(f)), false);
}

// One language (2.7): the Appendix C words are reserved in every module, and a
// release label selects nothing, so any label gives the verdict of no label.
const parAsName = "fn f() -> i64\n    var par: i64 = 1\n    return par\n";
expect("ts one language", "par is reserved in every module", parses(parAsName), false);
expect("one language", "par is reserved in every module", checks(parAsName), false);
const plain = "fn f() -> i64\n    return 1\n";
for (const label of ["0.5", "0.6", "0.7", "9.9"]) expect("one language", `label ${label} selects nothing`, checks(`# rce-version: ${label}\n` + plain), checks(plain));

// Batteries included (4.1): the standard library resolves by `use` in the browser checker too.
expect("stdlib", "use enc", checks("module demo\nuse enc\n\nfn main()\n    let b = [104, 105]\n    emit enc.b64_encode(&b, false, true)\n"), true);
expect("stdlib", "use of an unknown module requires change", checks("module demo\nuse nothere\n\nfn main()\n    emit 1\n"), false);
// `use` is the one way to bring a module in (4.1): crypto's items are reached by its name,
// and a `#!lib crypto` line is a comment that places nothing in front of the module.
expect("stdlib", "use crypto", checks("module demo2\nuse crypto\n\nfn main()\n    emit crypto.sha256([97]).len()\n"), true);
expect("stdlib", "a #!lib line is a comment", checks("module demo4\n#!lib crypto\n\nfn main()\n    emit sha256([97]).len()\n"), false);

console.log(`${runs - fails}/${runs} wasm checks passed`);
process.exit(fails ? 1 : 0);
