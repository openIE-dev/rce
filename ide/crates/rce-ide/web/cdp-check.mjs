// Drive the served web IDE in headless Chrome over the DevTools protocol:
// load it, type into the editor, take a screenshot, print the status line
// and console. Usage: node --experimental-websocket cdp-check.mjs URL OUT.png
// (Node 20; Chrome on PATH or $CHROME).
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const [url, out] = process.argv.slice(2);
const port = 9333;
const chrome = spawn(process.env.CHROME || "google-chrome", [
  "--headless=new", "--no-sandbox", `--remote-debugging-port=${port}`,
  "--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=vulkan",
  "--window-size=1400,940", "--user-data-dir=/tmp/rce-cdp-profile", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
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
for (let i = 0; i < 60; i++) { await sleep(500); if (await evalJs("!!window.ide")) break; }
const status0 = await evalJs("document.getElementById('log').textContent");
const gpu = await evalJs("navigator.gpu ? (navigator.gpu.requestAdapter().then(a => a ? 'webgpu adapter' : 'no webgpu adapter')) : 'no navigator.gpu'");
// Click into the editor on line 3, type, select a word, undo.
const key = async (k, mods = 0, text) => {
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, modifiers: mods, text });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: k, modifiers: mods });
};
const click = async (x, y) => {
  for (const type of ["mousePressed", "mouseReleased"]) await send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
};
let status1 = null, clip = null;
if (await evalJs("!!window.ide")) {
  await click(60, 120);
  await key("End"); await key("Enter");
  for (const ch of "-- typed in the browser") await send("Input.insertText", { text: ch });
  await key("ArrowLeft", 2); await key("ArrowRight", 8 | 2); // ctrl+left, ctrl+shift+right
  await sleep(500);
  status1 = await evalJs("document.getElementById('log').textContent");
  // Clipboard: the page's copy/paste handlers (synthetic events carry a
  // DataTransfer; a real Ctrl+C/V fires the same events).
  clip = await evalJs(`(() => {
    const ime = document.querySelector('textarea');
    const ev = (type, d) => ime.dispatchEvent(new ClipboardEvent(type, { clipboardData: d, bubbles: true, cancelable: true }));
    const d = new DataTransfer(); ev('copy', d);
    const copied = d.getData('text/plain');
    const p = new DataTransfer(); p.setData('text/plain', 'PASTED-FROM-OS'); ev('paste', p);
    return { copied, pasted: ide.editor_json().includes('PASTED-FROM-OS') };
  })()`);
}
// The canvas itself, read back in the same task as a frame (headless
// screenshots do not always composite WebGPU canvases).
const data = await evalJs(`(async () => {
  const ide = window.ide, c = document.getElementById('c');
  ide.snapshot_begin();
  while (!ide.snapshot_ready()) await new Promise((r) => setTimeout(r, 20));
  const px = ide.snapshot_take();
  const o = document.createElement('canvas'); o.width = c.width; o.height = c.height;
  o.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, px.length), c.width, c.height), 0, 0);
  return o.toDataURL('image/png');
})()`);
if (data) writeFileSync(out, Buffer.from(data.split(",")[1], "base64"));
console.log(JSON.stringify({ gpu, status0, status1, clip, logs }, null, 1));
ws.close(); chrome.kill();
process.exit(0);
