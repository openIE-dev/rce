// Drive the served 3D graph page in headless Chrome over the DevTools
// protocol: load, check the backend, click a node (linked-view jump),
// orbit, scrub, switch to hardware and telemetry, read back each frame.
// Usage: node --experimental-websocket graph3d-check.mjs URL OUTDIR
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const [url, out] = process.argv.slice(2);
const port = 9334;
const chrome = spawn(process.env.CHROME || "google-chrome", [
  "--headless=new", "--no-sandbox", `--remote-debugging-port=${port}`,
  "--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=vulkan",
  "--window-size=1400,920", "--user-data-dir=/tmp/rce-cdp-graph3d", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); } }
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map(); const logs = [];
ws.addEventListener("message", (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
  if (d.method === "Runtime.consoleAPICalled") logs.push(d.params.type + ": " + d.params.args.map((a) => a.value ?? a.description).join(" "));
  if (d.method === "Runtime.exceptionThrown") logs.push("exception: " + JSON.stringify(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text));
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evalJs = async (e) => (await send("Runtime.evaluate", { expression: e, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send("Runtime.enable"); await send("Page.enable");
await send("Page.navigate", { url });
for (let i = 0; i < 60; i++) { await sleep(500); if (await evalJs("!!window.g3")) break; }
const res = { gpu: await evalJs("navigator.gpu ? navigator.gpu.requestAdapter().then(a => a ? 'webgpu adapter' : 'no webgpu adapter') : 'no navigator.gpu'") };
res.status0 = await evalJs("window.g3 && g3.status()");
const snap = async (name) => {
  const data = await evalJs(`(async () => {
    const c = document.getElementById('c');
    g3.snapshot_begin();
    while (!g3.snapshot_ready()) await new Promise((r) => setTimeout(r, 20));
    const px = g3.snapshot_take();
    const o = document.createElement('canvas'); o.width = c.width; o.height = c.height;
    o.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, px.length), c.width, c.height), 0, 0);
    return o.toDataURL('image/png');
  })()`);
  if (data) writeFileSync(`${out}/${name}`, Buffer.from(data.split(",")[1], "base64"));
};
if (res.status0) {
  // Linked view: click the node labelled Inc through a real mouse event on the canvas.
  const p = await evalJs("(() => { const p = g3.node_screen('Inc'); const r = document.getElementById('c').getBoundingClientRect(); return p ? [p[0] + r.left, p[1] + r.top] : null; })()");
  if (p) for (const type of ["mousePressed", "mouseReleased"]) await send("Input.dispatchMouseEvent", { type, x: p[0], y: p[1], button: "left", clickCount: 1 });
  await sleep(200);
  res.afterClick = await evalJs("g3.status()");
  res.log = await evalJs("document.getElementById('log').textContent");
  await evalJs("g3.drag(80, 20)");
  await snap("web-graph3d.png");
  await evalJs("g3.scrub(0.3)"); res.scrubbed = await evalJs("g3.status()");
  await evalJs("g3.set_mode('hardware')"); await sleep(200); await snap("web-hardware.png");
  res.hardware = await evalJs("g3.status()");
  await evalJs("g3.set_mode('telemetry')"); await sleep(200); await snap("web-telemetry.png");
  if (process.env.PARITY) {
    await evalJs("g3.set_mode('graph')"); await sleep(200);
    await evalJs(`g3.set_style('{"divergence":true}')`); await evalJs("g3.drag(80, 20)"); await sleep(200); await snap("web-divergence.png");
    await evalJs(`g3.set_style('{"divergence":false,"mapping":{"size":"degree","color":"joules","glow":"actions"}}')`); await sleep(200); await snap("web-mapping.png");
    await evalJs(`g3.set_style('{"mapping":{}}')`);
    for (let i = 0; i < 15; i++) await evalJs("g3.wheel(120)");
    await sleep(200); res.zoomed = await evalJs("g3.status()"); await snap("web-zoom-files.png");
    for (let i = 0; i < 15; i++) await evalJs("g3.wheel(-120)");
  }
}
res.logs = logs;
console.log(JSON.stringify(res, null, 1));
ws.close(); chrome.kill();
process.exit(0);
