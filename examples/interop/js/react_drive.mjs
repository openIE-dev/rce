// Render ProgramView with react-test-renderer, dispatch each event of a
// stream as a DOM-style CustomEvent inside act(), and print the transcript
// the component renders.
//   node react_drive.mjs MODULE.mjs STREAM.json
import React from "react";
import TestRenderer from "react-test-renderer";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { Program, decode, BadEvent } from "./rce_host.mjs";
import { ProgramView } from "./Program.mjs";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const m = await import(pathToFileURL(process.argv[2]).href);
const adapter = new Program(m); // used only for its JSON-to-event boundary
const host = { event: (o) => adapter.event(o), BadEvent };
const stream = decode(readFileSync(process.argv[3], "utf8"));
const source = new EventTarget();
let root;
TestRenderer.act(() => { root = TestRenderer.create(React.createElement(React.StrictMode, null, React.createElement(ProgramView, { m, host, source }))); });
for (const ev of stream) TestRenderer.act(() => { source.dispatchEvent(new CustomEvent("rce-event", { detail: ev })); });
process.stdout.write(root.root.findByProps({ "data-role": "transcript" }).children.join(""));
