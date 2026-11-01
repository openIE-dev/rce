// The DOM renderer's test (ui/web/render.js), in headless Chromium.
//
// Fixture: the frames of test/shop.rce, written by `rce host run frames`
// (shop.frames.jsonl), and `rce host replay frames` of the same D
// (shop.replay.txt), which holds each frame's ui.rules findings. Both are
// output of the reference checker; spec/checker/tests/host_frames.rs
// (`ui_web_fixture_is_current`) regenerates them and requires them equal.
//
// For each frame, in order, the renderer applies the frame's ui.view.diff
// operations, and the test requires:
//   1. the DOM is the view tree: the same shape, tags, texts and
//      attributes, every key a DOM id, the presentation attributes as
//      inline style, the root's lang and title the document's;
//   2. axe-core, run with the WCAG 2.0, 2.1 and 2.2 level A and AA rules,
//      reports no violation on a frame with no ui.rules finding, and on a
//      frame with findings exactly the violations EXPECT names for them.
//
// Run: npm test (in ui/web). CHROMIUM names a browser binary to use
// instead of Playwright's own.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const WEB = path.join(__dirname, "..");
const AXE = require.resolve("axe-core/axe.min.js");
const AXE_VERSION = require("axe-core/package.json").version;
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

// The axe-core rules a ui.rules finding corresponds to. A finding is
// matched by its first words and its reason; [] records a finding no
// axe-core rule reports (a difference of coverage, not of verdict).
const EXPECT = [
  { finding: /^1\.4\.3 fails .*text contrast/, axe: ["color-contrast"], what: "contrast" },
  { finding: /^4\.1\.2 fails .*has no accessible name/, axe: ["button-name"], what: "name and role" },
  { finding: /^4\.1\.2 fails .*focusable but hidden from assistive technology/, axe: ["aria-hidden-focus"], what: "keyboard focus on a hidden control" },
  { finding: /^2\.1\.1 fails .*interactive but not focusable/, axe: [], what: "keyboard: a widget role with no tabindex" },
];

function read(name) {
  return fs.readFileSync(path.join(__dirname, name), "utf8");
}

/** The findings under each frame of a `rce host replay frames` report. */
function findings(report) {
  const out = [];
  for (const l of report.split("\n")) {
    if (l.startsWith("frame ")) out.push([]);
    else if (l.startsWith("ui.rules:")) break;
    else if (l.startsWith("  finding ") && out.length) out[out.length - 1].push(l.slice(10));
  }
  return out;
}

function rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}

/** The DOM a view tree should give (renderer contract, ui/web/README.md). */
function expected(v, i) {
  if (v.tag[i] === "") return { text: v.text[i] };
  const attrs = {};
  v.names[i].forEach((n, k) => { if (n !== "style") attrs[n] = v.values[i][k]; });
  if (v.key[i] !== "" && !("id" in attrs)) attrs.id = v.key[i];
  const a = (n) => (n in attrs ? attrs[n] : undefined);
  const style = {
    color: a("fg") ? rgb(a("fg")) : "",
    backgroundColor: a("bg") ? rgb(a("bg")) : "",
    fontSize: a("font-px") ? a("font-px") + "px" : "",
    fontWeight: a("font-weight") || "",
    minWidth: a("w") ? a("w") + "px" : "",
    minHeight: a("h") ? a("h") + "px" : "",
    border: a("border") ? "1px solid " + rgb(a("border")) : "",
  };
  return { tag: v.tag[i], attrs, style, kids: v.kids[i].map((c) => expected(v, c)) };
}

/** The DOM under document.body, in the same shape (runs in the page). */
function dom() {
  const walk = (n) => {
    if (n.nodeType === 3) return { text: n.data };
    const attrs = {};
    for (const a of n.attributes) if (a.name !== "style") attrs[a.name] = a.value;
    const s = n.style;
    const style = { color: s.color, backgroundColor: s.backgroundColor, fontSize: s.fontSize, fontWeight: s.fontWeight, minWidth: s.minWidth, minHeight: s.minHeight, border: s.border };
    return { tag: n.localName, attrs, style, kids: [...n.childNodes].map(walk) };
  };
  return { roots: document.body.childNodes.length, tree: walk(document.body.firstChild), lang: document.documentElement.lang, title: document.title };
}

async function main() {
  const sink = read("shop.frames.jsonl");
  const report = read("shop.replay.txt");
  const lines = sink.split("\n").filter((l) => l !== "");
  // the sink and the replay hold the same frames, byte for byte (HC-29)
  assert.deepEqual(report.split("\n").filter((l) => l.startsWith("frame ")).map((l) => l.slice(6)), lines);
  const fs_ = findings(report);
  assert.equal(fs_.length, lines.length);
  const frames = lines.map((l) => JSON.parse(l));
  const kinds = new Set(frames.flatMap((f) => (f.ops || []).map((o) => o.op)));
  assert.deepEqual([...kinds].sort(), ["create", "move", "remove", "set", "text", "unset"], "the fixture exercises every kind of operation");

  const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
  const page = await browser.newPage();
  await page.route("http://rce.test/**", (route) => {
    const p = path.join(WEB, new URL(route.request().url()).pathname);
    if (!p.startsWith(WEB) || !fs.existsSync(p)) return route.fulfill({ status: 404, body: "" });
    route.fulfill({ status: 200, contentType: p.endsWith(".js") ? "text/javascript" : "text/html", body: fs.readFileSync(p) });
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("http://rce.test/test/page.html");
  await page.waitForFunction(() => window.renderer !== undefined);
  await page.addScriptTag({ path: AXE });

  let clean = 0;
  const met = new Map(EXPECT.map((e) => [e.what, 0]));
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    await page.evaluate((l) => window.renderer.apply(l), lines[i]);
    const got = await page.evaluate(dom);
    const want = expected(f.view, 0);
    assert.equal(got.roots, 1, `frame ${i}: one root`);
    assert.deepEqual(got.tree, want, `frame ${i} (seq ${f.seq}): the DOM is not the view tree`);
    const root = Object.fromEntries(f.view.names[0].map((n, k) => [n, f.view.values[0][k]]));
    assert.equal(got.lang, root.lang || "");
    assert.equal(got.title, root.title || "");

    const r = await page.evaluate((tags) => window.axe.run(document, { runOnly: { type: "tag", values: tags }, resultTypes: ["violations"] }), TAGS);
    const ids = r.violations.map((v) => v.id).sort();
    const fi = fs_[i];
    if (fi.length === 0) {
      assert.deepEqual(ids, [], `frame ${i} (seq ${f.seq}): ui.rules finds nothing, axe-core reports ${ids.join(", ")}`);
      clean += 1;
      continue;
    }
    const want_ids = new Set();
    for (const t of fi) {
      const e = EXPECT.find((x) => x.finding.test(t));
      assert.ok(e, `frame ${i}: no expectation for the ui.rules finding "${t}"`);
      e.axe.forEach((x) => want_ids.add(x));
      met.set(e.what, met.get(e.what) + 1);
    }
    assert.deepEqual(ids, [...want_ids].sort(), `frame ${i} (seq ${f.seq}): ui.rules finds ${fi.join("; ")}`);
    console.log(`frame ${i} seq ${f.seq}: ui.rules ${fi.map((t) => t.split(" (")[0]).join("; ")} -> axe-core ${ids.join(", ") || "no violation"}`);
  }
  assert.deepEqual(errors, []);
  for (const [what, n] of met) assert.ok(n > 0, `the fixture has no frame for ${what}`);
  await browser.close();
  console.log(`ok: ${frames.length} frames rendered from ${lines.length} sink lines, each DOM equal to its view tree; axe-core ${AXE_VERSION} (${TAGS.join(", ")}): no violation on the ${clean} frames ui.rules passes, the expected violations on the ${frames.length - clean} it fails`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
