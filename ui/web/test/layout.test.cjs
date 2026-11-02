// The ui.layout model (stdlib/lib/ui/layout.rce) against Chromium.
//
// Fixture: the views of test/layout.rce as frame sink lines
// (layout.frames.jsonl) and the boxes ui.layout gives each of them in the
// four modes ui.rules uses (layout.boxes.txt): "normal" at the root's
// viewport width vw, "reflow" at 320 CSS px (WCAG 1.4.10), "resize" with
// every font size doubled (1.4.4) and "spacing" with the text spacing of
// 1.4.12. Both are output of the reference checker;
// spec/checker/tests/host_frames.rs (`ui_web_layout_fixture_is_current`)
// regenerates them and requires them equal.
//
// For each view and mode the renderer (render.js) draws the view into a
// body as wide as the mode's viewport, with the cell font rce-cell.ttf; the
// test then requires, exactly:
//   1. every element's getBoundingClientRect() is the border box ui.layout
//      gives it (x, y, width, height in CSS px, the model's centi-px / 100);
//   2. every text run has as many lines as ui.layout gives it, and each line
//      (the run's Range client rects, merged per line) starts at the run's
//      x, is as wide as the model's line, and starts lh * j below the
//      first; the first line's top less the half-leading rounded down to a
//      whole px (where Chromium places the glyphs) is the run's y.
//   3. in the normal mode, axe-core's rule target-size (WCAG 2.5.8, with its
//      spacing exception) reports exactly the targets on which ui.rules
//      finds 2.5.8 failing, and passes every other target.
// It prints how many boxes, lines and targets it compared; nothing may
// differ.
//
// Run: node test/layout.test.cjs (in ui/web; npm test runs it after the
// renderer's test). CHROMIUM names a browser binary to use instead of
// Playwright's own.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const WEB = path.join(__dirname, "..");
const AXE = require.resolve("axe-core/axe.min.js");
const SPACING_CSS = [
  "#rce-ui-host, #rce-ui-host * { line-height: 1.5 !important; letter-spacing: 0.12em !important; word-spacing: 0.16em !important; }",
  '#rce-ui-host p, #rce-ui-host [role="paragraph"] { margin-bottom: 2em !important; }',
].join("\n");

function read(name) {
  return fs.readFileSync(path.join(__dirname, name), "utf8");
}

/** view index -> mode -> node -> {box: [x, y, w, h], lines?: [w...], lh?} (centi-px) */
function parseBoxes(text) {
  const out = [];
  for (const l of text.split("\n")) {
    if (!l.startsWith("view ")) continue;
    const parts = l.split(";");
    const vi = Number(parts[0].slice(5));
    const modes = {};
    let cur = null;
    for (const part of parts.slice(1)) {
      if (part.startsWith("target-size")) {
        modes.targetSize = part.slice(12).split(",").filter((x) => x !== "").sort();
      } else if (part.startsWith("mode ")) {
        cur = {};
        modes[part.slice(5)] = cur;
      } else {
        const w = part.split(" ");
        const e = { box: w.slice(1, 5).map(Number) };
        if (w[5] === "lines") {
          e.lines = w[6].split(",").map(Number);
          e.lh = Number(w[8]);
        }
        cur[Number(w[0])] = e;
      }
    }
    out[vi] = modes;
  }
  return out;
}

const MODES = {
  normal: (vw) => ({ width: vw, scale: 1, spacing: false }),
  reflow: () => ({ width: 320, scale: 1, spacing: false }),
  resize: (vw) => ({ width: vw, scale: 2, spacing: false }),
  spacing: (vw) => ({ width: vw, scale: 1, spacing: true }),
};

/** Draw one sink line in a fresh page and measure it (runs in the page). */
async function measure(arg) {
  const { line, width, scale, spacing, css } = arg;
  document.body.style.width = width + "px";
  const r = new window.Renderer(document.body, { document: true, textScale: scale });
  if (spacing) {
    const st = document.createElement("style");
    st.textContent = css;
    document.head.appendChild(st);
  }
  await document.fonts.load('16px "RCE Cell"');
  r.apply(line);
  await document.fonts.ready;
  const v = r.view;
  const els = {};
  const runs = {};
  for (let i = 0; i < v.tag.length; i++) {
    const node = r.live[i];
    if (v.tag[i] !== "") {
      const q = node.getBoundingClientRect();
      els[i] = [q.x, q.y, q.width, q.height];
      continue;
    }
    // the first text node of a run of adjacent text siblings
    if (node.previousSibling && node.previousSibling.nodeType === 3) continue;
    let last = node;
    while (last.nextSibling && last.nextSibling.nodeType === 3) last = last.nextSibling;
    const rg = document.createRange();
    rg.setStart(node, 0);
    rg.setEnd(last, last.data.length);
    const lines = [];
    for (const q of rg.getClientRects()) {
      if (q.width === 0 && q.height === 0) continue;
      const l = lines.find((x) => x.y === q.y);
      if (l) {
        const right = Math.max(l.x + l.w, q.x + q.width);
        l.x = Math.min(l.x, q.x);
        l.w = right - l.x;
      } else lines.push({ x: q.x, y: q.y, w: q.width, h: q.height });
    }
    lines.sort((a, b) => a.y - b.y);
    runs[i] = lines;
  }
  return { els, runs, font: document.fonts.check('16px "RCE Cell"') };
}

async function main() {
  const lines = read("layout.frames.jsonl").split("\n").filter((l) => l !== "");
  const model = parseBoxes(read("layout.boxes.txt"));
  assert.equal(model.length, lines.length, "one boxes line per view");
  const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
  const page = await browser.newPage();
  await page.route("http://rce.test/**", (route) => {
    const p = path.join(WEB, new URL(route.request().url()).pathname);
    if (!p.startsWith(WEB) || !fs.existsSync(p)) return route.fulfill({ status: 404, body: "" });
    const type = p.endsWith(".js") ? "text/javascript" : p.endsWith(".ttf") ? "font/ttf" : "text/html";
    route.fulfill({ status: 200, contentType: type, body: fs.readFileSync(p) });
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  let nbox = 0;
  let ntarget = 0;
  let nsmall = 0;
  let nline = 0;
  let nruns = 0;
  const px = (c) => c / 100;
  for (let vi = 0; vi < lines.length; vi++) {
    const f = JSON.parse(lines[vi]);
    const root = Object.fromEntries(f.view.names[0].map((n, k) => [n, f.view.values[0][k]]));
    const vw = Number(root.vw || 1280);
    for (const [mode, spec] of Object.entries(MODES)) {
      const m = spec(vw);
      const want = model[vi][mode];
      assert.ok(want, `view ${vi}: no ${mode} boxes`);
      await page.goto("http://rce.test/test/layout.html");
      await page.waitForFunction(() => window.Renderer !== undefined);
      const got = await page.evaluate(measure, { line: lines[vi], width: m.width, scale: m.scale, spacing: m.spacing, css: SPACING_CSS });
      assert.ok(got.font, "the cell font is loaded");
      const where = `view ${vi} (${mode})`;
      if (mode === "normal") {
        await page.addScriptTag({ path: AXE });
        const r = await page.evaluate(() => window.axe.run(document, { runOnly: { type: "rule", values: ["target-size"] } }));
        const keys = (xs) => xs.flatMap((x) => x.nodes.map((n) => n.target.join(" ").replace(/^#/, ""))).sort();
        const viol = keys(r.violations);
        assert.deepEqual(keys(r.incomplete), [], `${where}: axe-core target-size is incomplete`);
        assert.deepEqual(viol, model[vi].targetSize, `${where}: axe-core target-size fails ${viol.join(",") || "nothing"}, ui.rules 2.5.8 fails ${model[vi].targetSize.join(",") || "nothing"}`);
        ntarget += viol.length + keys(r.passes).length;
        nsmall += viol.length;
      }
      for (let i = 0; i < f.view.tag.length; i++) {
        const w = want[i];
        if (f.view.tag[i] !== "") {
          assert.ok(w, `${where} node ${i}: no model box`);
          assert.deepEqual(got.els[i], w.box.map(px), `${where} element ${i} <${f.view.tag[i]}>: browser ${JSON.stringify(got.els[i])}, ui.layout ${JSON.stringify(w.box.map(px))}`);
          nbox += 1;
          continue;
        }
        const g = got.runs[i];
        if (!w) {
          assert.ok(g === undefined || g.length === 0, `${where} text ${i}: the browser shows a run ui.layout does not`);
          continue;
        }
        assert.ok(g, `${where} text ${i}: no run in the browser`);
        const [x, y] = w.box.map(px);
        const lh = px(w.lh);
        assert.equal(g.length, w.lines.length, `${where} run ${i} "${f.view.text[i]}": ${g.length} lines in the browser, ${w.lines.length} in ui.layout (${JSON.stringify(g)})`);
        const fc = g[0].h;
        const lead = Math.floor((lh - fc) / 2);
        for (let j = 0; j < g.length; j++) {
          assert.equal(g[j].x, x, `${where} run ${i} line ${j}: x`);
          assert.equal(g[j].w, px(w.lines[j]), `${where} run ${i} line ${j}: width`);
          assert.equal(g[j].y - lead, y + j * lh, `${where} run ${i} line ${j}: top`);
          nline += 1;
        }
        nruns += 1;
      }
    }
  }
  assert.deepEqual(errors, []);
  await browser.close();
  console.log(`ok: ${lines.length} views in ${Object.keys(MODES).length} modes; ui.layout and Chromium agree on all ${nbox} element boxes and all ${nline} lines of ${nruns} text runs; ui.rules 2.5.8 and axe-core target-size agree on all ${ntarget} targets (${nsmall} fail)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
