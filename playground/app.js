import init, { check } from "./pkg/rce_spec_check_wasm.js";

const $ = (id) => document.getElementById(id);
const src = $("src"), pick = $("pick");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const show = (v) => (v !== null && typeof v === "object") ? JSON.stringify(v) : String(v);

function render(r) {
  const v = $("verdict");
  if (r.ok) { v.className = "verdict ok"; v.textContent = `accepted (${r.examples.length} example${r.examples.length === 1 ? "" : "s"} hold)`; }
  else { v.className = "verdict bad"; v.textContent = `fit reported at the ${r.stage} stage`; }
  $("errors").innerHTML = r.errors.map((e) => `<div class="err"><b>${esc(e.rule || "")}</b> ${esc(e.msg)}</div>`).join("");
  $("examples").innerHTML = r.examples.map((ex) => {
    const rows = ex.steps.map((s) => `<tr><td>${esc(s.event)}</td><td${s.kept ? ' class="kept"' : ""}>${esc(s.kept ? "kept: " + s.kept : show(s.state))}</td></tr>`).join("");
    return `<h3>example on line ${ex.line}</h3><table><tr><th>event</th><th>state after</th></tr>${rows}</table>`;
  }).join("");
  $("form").textContent = r.form ? JSON.stringify(r.form, null, 2) : "";
}

let timer = 0;
function run() {
  try { render(JSON.parse(check(src.value))); }
  catch (e) { $("verdict").className = "verdict bad"; $("verdict").textContent = "checker error: " + e; }
}
const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 200); };

src.addEventListener("input", schedule);
src.addEventListener("keydown", (e) => {
  if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "    "); }
});

function encode(s) { return btoa(String.fromCharCode(...new TextEncoder().encode(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); }
function decode(s) { const b = atob(s.replace(/-/g, "+").replace(/_/g, "/")); return new TextDecoder().decode(Uint8Array.from(b, (c) => c.charCodeAt(0))); }

$("share").addEventListener("click", async () => {
  const url = location.href.split("#")[0] + "#code=" + encode(src.value);
  history.replaceState(null, "", url);
  try { await navigator.clipboard.writeText(url); $("share").textContent = "Link copied"; } catch { $("share").textContent = "Link in address bar"; }
  setTimeout(() => ($("share").textContent = "Copy link"), 1500);
});

await init();
const examples = await (await fetch("examples.json")).json();
pick.innerHTML = examples.map((e, i) => `<option value="${i}">${esc(e.name)}</option>`).join("");
pick.addEventListener("change", () => { src.value = examples[pick.value].src; run(); });
try { $("ver").textContent = (await (await fetch("version.txt")).text()).trim(); } catch {}
const m = location.hash.match(/^#code=(.+)$/);
src.value = m ? decode(m[1]) : examples[0].src;
run();
