// Reference for lib/ui/view.rce: Vue 3.5.44 (@vue/runtime-core) driven
// through a recording renderer. Each case renders a <div> with the old
// children, clears the log, renders the new children, and prints the
// container-level operations in order:
//   c L A  create the node labelled L and insert it before A ("-" = end)
//   m L A  move L before A
//   r L    remove L
// A node keeps the label it was created with. Child tokens:
//   N    keyed <span> (key N)        t#N  keyed <t> (key N)
//   t:x  unkeyed <t> labelled x      ~x   text node x
//   !    null (Vue renders a comment)
// Usage: node vue_ref.js <old>/<new>   (tokens "," separated), or lines
// of old/new on stdin with no argument.
const { createRenderer, h } = require('@vue/runtime-core')
let log = [], seen = new Set(), ROOT = null
const node = (o) => Object.assign({ kids: [], parent: null }, o)
const ops = {
  createElement: (tag) => node({ tag }),
  createText: (t) => node({ tag: '~', label: t }),
  createComment: () => node({ tag: '!', label: '!' }),
  setText: (n, t) => { n.text = t },
  setElementText: (n, t) => { n.text = t },
  parentNode: (n) => n.parent,
  nextSibling: (n) => { const p = n.parent; if (!p) return null; return p.kids[p.kids.indexOf(n) + 1] || null },
  querySelector: () => null,
  insert: (child, parent, anchor) => {
    if (parent === ROOT) {
      log.push((seen.has(child) ? 'm ' : 'c ') + child.label + ' ' + (anchor ? anchor.label : '-'))
      seen.add(child)
    }
    if (child.parent) { const p = child.parent; p.kids.splice(p.kids.indexOf(child), 1) }
    const i = anchor ? parent.kids.indexOf(anchor) : -1
    if (i < 0) parent.kids.push(child); else parent.kids.splice(i, 0, child)
    child.parent = parent
  },
  remove: (child) => {
    const p = child.parent
    if (p === ROOT) log.push('r ' + child.label)
    if (p) { p.kids.splice(p.kids.indexOf(child), 1); child.parent = null }
  },
  patchProp: (el, k, prev, next) => {
    if (k === 'data-k' && el.label === undefined) el.label = next
    if (k === 'data-key') el.key = next
  },
}
const { render } = createRenderer(ops)
function item(t) {
  let m
  if (t === '!') return null
  if ((m = /^~(.*)$/.exec(t))) return m[1]
  if ((m = /^([a-z]+):(.+)$/.exec(t))) return h(m[1], { 'data-k': m[2] }, m[2])
  if ((m = /^([a-z]+)#(\d+)$/.exec(t))) return h(m[1], { key: +m[2], 'data-k': m[2], 'data-key': m[2] }, m[2])
  if (/^\d+$/.test(t)) return h('span', { key: +t, 'data-k': t, 'data-key': t }, t)
  throw new Error('bad token ' + t)
}
const want = (t) => t === '!' ? '!' : t[0] === '~' ? '~' : /^\d+$/.test(t) ? 'span#' + t : t.includes('#') ? t : t.split(':')[0]
function run(a, b) {
  const host = ops.createElement('host')
  const mk = (xs) => h('div', xs.map(item))
  render(mk(a), host)
  ROOT = host.kids[0]
  seen = new Set(ROOT.kids)
  log = []
  render(mk(b), host)
  const got = ROOT.kids.map((n) => n.tag + (n.key != null ? '#' + n.key : '')).join(',')
  if (got !== b.map(want).join(',')) throw new Error('vue result ' + got + ' for ' + b)
  return log.join(';')
}
module.exports = { run }
const parse = (s) => (s === '' ? [] : s.split(','))
if (require.main === module) {
  const arg = process.argv[2]
  if (arg !== undefined) { const [a, b] = arg.split('/'); console.log(run(parse(a), parse(b))) }
  else for (const l of require('fs').readFileSync(0, 'utf8').split('\n')) { if (!l) continue; const [a, b] = l.split('/'); console.log(run(parse(a), parse(b))) }
}
