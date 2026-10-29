// A React component around an emitted JS module. The emitted delta is the
// reducer of useReducer: (state, event) -> state, pure, so React may call
// it twice (StrictMode) and get the same state. Events come from an
// ordinary DOM-style event handler: a listener on an EventTarget that the
// page (or a WebSocket, or a test) dispatches `rce-event` events on, and a
// button that sends the program's first event type.
import React from "react";

const h = React.createElement;

export function reducerOf(m) {
  return (s, ev) => {
    const k = s.lines.length > 0 ? s.k + 1 : 1;
    m.fuelReset(); // HC-7a: full fuel for each input
    try {
      const [next, outs] = m.delta(s.state, ev);
      const lines = [k + " " + m.show(ev) + " " + m.showState(next), ...outs.map((o) => "  emit " + m.show(o))];
      return { state: next, k, lines: [...s.lines, ...lines] };
    } catch (e) {
      if (!(e instanceof m.Kept)) throw e;
      // a kept step keeps the state (HC-21); the transcript records why
      return { state: s.state, k, lines: [...s.lines, k + " " + m.show(ev) + " kept: " + e.r] };
    }
  };
}

export function initOf(m) {
  m.fuelReset();
  const state = m.initialState();
  return { state, k: 0, lines: ["0 (initial) " + m.showState(state)] };
}

export function ProgramView({ m, host, source }) {
  const [s, dispatch] = React.useReducer(React.useMemo(() => reducerOf(m), [m]), m, initOf);
  React.useEffect(() => {
    if (!source) return undefined;
    const onEvent = (e) => {
      try { dispatch(host.event(e.detail)); } catch (err) { if (!(err instanceof host.BadEvent)) throw err; }
    };
    source.addEventListener("rce-event", onEvent);
    return () => source.removeEventListener("rce-event", onEvent);
  }, [source, host]);
  const first = Object.keys(m.EVENT_FIELDS)[0];
  return h("section", null,
    h("button", { type: "button", onClick: () => dispatch(host.event({ type: first })), disabled: m.EVENT_FIELDS[first].length > 0 }, first),
    h("pre", { "data-role": "transcript" }, s.lines.join("\n") + "\n"));
}
