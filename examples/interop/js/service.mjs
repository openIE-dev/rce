// A Node service around an emitted JS module: an ordinary node:http
// "request" handler takes POST /events (one JSON event per request) and
// answers with the outcome, the lines `rce run` prints for it and the
// state. Node runs handlers one at a time on its event loop, so the step is
// the single writer without a lock.
//
//   node service.mjs MODULE.mjs STREAM.json   serve on 127.0.0.1, post the
//                                             stream with fetch, print the transcript
import http from "node:http";
import { readFileSync } from "node:fs";
import { load, decode, encode, BadEvent } from "./rce_host.mjs";

export function serve(p) {
  let seq = 0;
  return http.createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const send = (code, text) => { res.writeHead(code, { "content-type": "application/json" }); res.end(text); };
      if (req.method === "GET" && req.url === "/initial") return send(200, JSON.stringify({ lines: [p.initialLine()] }));
      if (req.method !== "POST" || req.url !== "/events") return send(404, "{}");
      let ev;
      try { ev = p.event(decode(body)); } catch (e) {
        if (e instanceof BadEvent || e instanceof SyntaxError) return send(422, JSON.stringify({ outcome: "not-admitted", why: e.message }));
        throw e;
      }
      seq += 1;
      const o = p.step(ev);
      const outcome = o.kind === "applied" ? "applied" : "kept: " + o.reason;
      send(200, '{"seq":' + seq + ',"outcome":' + JSON.stringify(outcome) + ',"lines":' + JSON.stringify(p.lines(seq, ev, o)) + ',"state":' + p.stateJson() + "}");
    });
  });
}

if (import.meta.url === (await import("node:url")).pathToFileURL(process.argv[1]).href) {
  const p = await load(process.argv[2]);
  // the stream file is read with exact integers too, as a client would send them
  const stream = decode(readFileSync(process.argv[3], "utf8"));
  const server = serve(p);
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + server.address().port;
  const lines = (await (await fetch(base + "/initial")).json()).lines;
  for (const ev of stream) {
    const body = encode(ev);
    const r = await fetch(base + "/events", { method: "POST", body });
    lines.push(...(await r.json()).lines);
  }
  server.close();
  process.stdout.write(lines.map((x) => x + "\n").join(""));
}
