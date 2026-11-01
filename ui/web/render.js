// A DOM renderer for ui.view frames (spec/proposals/human-interface.md, H4;
// Appendix E, HC-29). No framework and no dependency.
//
// Input: the sink lines of `rce host run frames FILE` (or of
// `rce host replay frames`), one frame each: {seq, view, ops, view_digest,
// frame_digest, ...}. `view` is the ui.view arena (tag, key, text, names,
// values, parent, kids; node 0 the root). `ops` are the operations
// ui.view.diff gives from the previous frame's view to this one, in its
// order, or null for the first frame.
//
// What the renderer does (ui/web/README.md states what it guarantees):
// - a node with tag "" is a DOM Text node holding its text; any other node
//   is an element of that tag, and an element's own text is not shown;
// - every attribute is set by name, as the view holds it (WAI-ARIA roles,
//   states and properties, tabindex, lang, href, ...), except `style`,
//   which the renderer owns;
// - a node with a key and no `id` attribute has the DOM id of its key, so
//   references by key (aria-labelledby, aria-describedby, aria-controls,
//   href="#K") resolve in the DOM;
// - the presentation attributes of rce.ui.attrs/1 become inline style:
//   fg color, bg background-color, font-px font-size, font-weight
//   font-weight, w min-width, h min-height, border a 1px solid border;
// - with {document: true}, the root's lang and title are also the
//   document's language and title;
// - the node with focused="true" receives focus after the frame.
//
// Live ids follow ui.view.diff: before a frame, node i of the previous view
// has id i; a node created from the new view's node j has id |a| + j. After
// the operations the DOM has the new view's shape, which the renderer
// checks, and node i of the new view has id i for the next frame.

const STYLE = {
  fg: (s, v) => { s.color = v; },
  bg: (s, v) => { s.backgroundColor = v; },
  "font-px": (s, v) => { s.fontSize = v + "px"; },
  "font-weight": (s, v) => { s.fontWeight = v; },
  w: (s, v) => { s.minWidth = v + "px"; },
  h: (s, v) => { s.minHeight = v + "px"; },
  border: (s, v) => { s.border = "1px solid " + v; },
};

/** Presentation attributes the renderer maps to inline style. */
export const PRESENTATION = Object.keys(STYLE);

/** The ui.view operation of each op name the sink writes. */
export const OPS = ["remove", "create", "move", "set", "unset", "text"];

function attrsOf(view, i) {
  const m = new Map();
  view.names[i].forEach((n, k) => m.set(n, view.values[i][k]));
  return m;
}

/** The style text one node's attributes give (empty when none apply). */
export function styleOf(attrs) {
  const probe = { style: {} };
  for (const [n, v] of attrs) if (STYLE[n]) STYLE[n](probe.style, v);
  return probe.style;
}

export class Renderer {
  /**
   * @param {Element} host the element the view's root is placed in
   * @param {{document?: boolean}} opts
   */
  constructor(host, opts = {}) {
    this.host = host;
    this.doc = host.ownerDocument;
    this.asDocument = !!opts.document;
    this.view = null; // the view the DOM shows
    this.live = []; // live id -> DOM node
    this.frames = 0;
  }

  /** Apply one sink line (an object or its JSON text). */
  apply(line) {
    const f = typeof line === "string" ? JSON.parse(line) : line;
    const b = f.view;
    if (this.view === null || f.ops === null) {
      this.host.replaceChildren(this.build(b, 0, null));
    } else {
      this.applyOps(this.view, b, f.ops);
    }
    this.live = this.number(b);
    this.view = b;
    this.frames += 1;
    this.finish(b);
    return f;
  }

  /** The DOM subtree of node j of view b; `ids` (when given) maps j -> live id. */
  build(b, j, ids) {
    const d = this.doc;
    if (b.tag[j] === "") {
      const t = d.createTextNode(b.text[j]);
      if (ids) ids(j, t);
      return t;
    }
    const e = d.createElement(b.tag[j]);
    const attrs = attrsOf(b, j);
    for (const [n, v] of attrs) if (n !== "style") e.setAttribute(n, v);
    if (b.key[j] !== "" && !attrs.has("id")) e.id = b.key[j];
    this.restyle(e, attrs);
    if (ids) ids(j, e);
    for (const c of b.kids[j]) e.appendChild(this.build(b, c, ids));
    return e;
  }

  restyle(e, attrs) {
    e.removeAttribute("style");
    for (const [n, v] of attrs) if (STYLE[n]) STYLE[n](e.style, v);
  }

  /** The operations of ui.view.diff(a, b), in order. */
  applyOps(a, b, ops) {
    const off = a.tag.length;
    const live = this.live.slice();
    const place = (node, parent, before) => {
      if (parent < 0) {
        this.host.replaceChildren(node);
        return;
      }
      const p = live[parent];
      const ref = before >= 0 ? live[before] : null;
      p.insertBefore(node, ref && ref.parentNode === p ? ref : null);
    };
    for (const o of ops) {
      const n = o.nd;
      switch (o.op) {
        case "remove":
          live[n].remove();
          break;
        case "create": {
          const node = this.build(b, n - off, (j, x) => { live[off + j] = x; });
          place(node, o.parent, o.before);
          break;
        }
        case "move":
          place(live[n], o.parent, o.before);
          break;
        case "set":
        case "unset": {
          const e = live[n];
          if (o.op === "set" && o.name !== "style") e.setAttribute(o.name, o.value);
          if (o.op === "unset") e.removeAttribute(o.name);
          break;
        }
        case "text":
          live[n].data = o.value;
          break;
        default:
          throw new Error(`unknown operation ${o.op}`);
      }
    }
  }

  /** Check that the DOM has b's shape, and number its nodes as b does. */
  number(b) {
    const live = new Array(b.tag.length);
    const root = this.host.firstChild;
    if (!root || this.host.childNodes.length !== 1) throw new Error("the host holds no single root");
    const walk = (i, node) => {
      const text = b.tag[i] === "";
      const ok = text ? node.nodeType === 3 && node.data === b.text[i] : node.nodeType === 1 && node.localName === b.tag[i];
      if (!ok) throw new Error(`node ${i}: the DOM does not have the view's shape`);
      live[i] = node;
      if (text) return;
      const kids = b.kids[i];
      if (node.childNodes.length !== kids.length) throw new Error(`node ${i}: ${node.childNodes.length} children, the view has ${kids.length}`);
      kids.forEach((c, k) => walk(c, node.childNodes[k]));
    };
    walk(0, root);
    return live;
  }

  /** Ids, style and focus after a frame; the document's language and title. */
  finish(b) {
    let focus = null;
    for (let i = 0; i < b.tag.length; i++) {
      const e = this.live[i];
      if (!e || b.tag[i] === "") continue;
      const attrs = attrsOf(b, i);
      if (!attrs.has("id")) {
        if (b.key[i] !== "") e.id = b.key[i];
        else e.removeAttribute("id");
      }
      this.restyle(e, attrs);
      if (attrs.get("focused") === "true") focus = e;
    }
    if (this.asDocument) {
      const r = attrsOf(b, 0);
      if (r.has("lang")) this.doc.documentElement.lang = r.get("lang");
      if (r.has("title")) this.doc.title = r.get("title");
    }
    if (focus && typeof focus.focus === "function") focus.focus();
  }
}
