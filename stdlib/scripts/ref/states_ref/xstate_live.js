// Live reference for lib/ui/states.rce: XState 5.33.2. Each stdin line is
// JSON {src, evs}: a chart in the states.rce text form (machine 0 only;
// no send/invoke) and external events. Prints per line
// "<final|live> <atomic states in document order> | <entry/exit trace> | <vars>".
const { createMachine, createActor, assign, raise, enqueueActions, stateIn } = require('xstate')
function build(src, trace) {
  const L = src.split('\n')
  const st = {}, order = [], vars = []
  let cur = null
  const blk = () => []
  for (const l of L) {
    const w = l.split(' ')
    if (w[0] === 'state') {
      const n = { id: w[1], kind: w[2], parent: w[3], init: w[4], entry: [], exit: [], trans: [], kids: [] }
      st[w[1]] = n; order.push(w[1]); if (w[3] !== '-') st[w[3]].kids.push(w[1])
    } else if (w[0] === 'data') vars.push([w[1], +w[2].split(':')[1]])
    else if (w[0] === 'onentry') { cur = []; st[w[1]].entry.push(cur) }
    else if (w[0] === 'onexit') { cur = []; st[w[1]].exit.push(cur) }
    else if (w[0] === 'transition') { cur = []; st[w[1]].trans.push({ ev: w[2], cond: w[3], int: w[4] === 'int', tg: w[5], acts: cur }) }
    else if (w[0] === 'end') cur = null
    else if (w[0] === 'machine' || w[0] === '') {}
    else cur.push(w)
  }
  const atom = (ctx, a, b) => (a === 'n' ? +b : ctx[b])
  const evalc = (c, ctx) => {
    const p = c.split(':')
    const x = atom(ctx, p[2], p[3]), y = atom(ctx, p[4], p[5])
    return { eq: x === y, ne: x !== y, lt: x < y, le: x <= y, gt: x > y, ge: x >= y }[p[1]]
  }
  const assignOf = (w) => { const p = w[2].split(':'); return assign({ [w[1]]: ({ context }) => context[p[2]] + atom(context, p[3], p[4]) }) }
  function acts(lines) {
    const out = []
    for (let i = 0; i < lines.length; i++) {
      const w = lines[i]
      if (w[0] === 'raise') out.push(raise({ type: w[1] }))
      else if (w[0] === 'assign') out.push(assignOf(w))
      else if (w[0] === 'log') {}
      else if (w[0] === 'if') {
        // if c / [else] / fi with flat bodies
        let j = i + 1; const th = [], el = []; let tgt = th
        for (; lines[j][0] !== 'fi'; j++) { if (lines[j][0] === 'else') tgt = el; else tgt.push(lines[j]) }
        const c = w[1]
        out.push(enqueueActions(({ context, enqueue }) => {
          for (const x of (evalc(c, context) ? th : el)) {
            if (x[0] === 'raise') enqueue.raise({ type: x[1] })
            else if (x[0] === 'assign') enqueue(assignOf(x))
          }
        }))
        i = j
      } else throw new Error('action ' + w[0])
    }
    return out
  }
  const xev = (e) => e.startsWith('done.state.') ? 'xstate.' + e : e
  function guard(c) {
    if (c === '-') return undefined
    if (c.startsWith('in:')) return stateIn('#' + c.slice(3))
    return ({ context }) => evalc(c, context)
  }
  function conf(id) {
    const n = st[id]
    const o = { id }
    if (n.kind === 'shallow' || n.kind === 'deep') {
      o.type = 'history'; o.history = n.kind
      if (n.trans.length) o.target = n.trans[0].tg.split(',').map((t) => '#' + t)
      return o
    }
    const real = n.kids.filter((k) => !['shallow', 'deep'].includes(st[k].kind))
    if (n.kind === 'parallel') o.type = 'parallel'
    else if (n.kind === 'final') o.type = 'final'
    else if (real.length) o.initial = n.init !== '-' ? n.init : real[0]
    o.entry = [() => { if (id !== '_scxml') trace.push('+' + id) }, ...n.entry.flatMap(acts)]
    o.exit = [...n.exit.flatMap(acts), () => { if (id !== '_scxml' && !done.v) trace.push('-' + id) }]
    o.on = {}; o.always = []
    for (const t of n.trans) {
      const tr = { actions: acts(t.acts), reenter: !t.int }
      if (t.tg !== '-') tr.target = t.tg.split(',').map((x) => '#' + x)
      const g = guard(t.cond); if (g) tr.guard = g
      if (t.ev === '-') o.always.push(tr)
      else for (const e of t.ev.split(',')) (o.on[xev(e)] ||= []).push(tr)
    }
    if (n.kids.length) { o.states = {}; for (const k of n.kids) o.states[k] = conf(k) }
    return o
  }
  const done = { v: false }
  const root = conf('_scxml')
  root.context = Object.fromEntries(vars)
  return { m: createMachine(root), order, st, done }
}
const out = []
for (const line of require('fs').readFileSync(0, 'utf8').split('\n')) {
  if (!line) continue
  const { src, evs } = JSON.parse(line)
  const trace = []
  try {
    const b = build(src, trace)
    const a = createActor(b.m)
    a.subscribe({ complete: () => { b.done.v = true }, error: () => {} })
    a.start()
    for (const e of evs) a.send({ type: e })
    const s = a.getSnapshot()
    const leaves = b.order.filter((id) => ['atomic', 'final'].includes(b.st[id].kind) && s.matches !== undefined && s._nodes.some((n) => n.id === id && n.type !== 'compound' && n.type !== 'parallel'))
    out.push((s.status === 'done' ? 'final ' : 'live ') + leaves.join(',') + ' | ' + trace.join(' ') + ' | ' + Object.keys(s.context).map((k) => s.context[k]).join(','))
  } catch (e) { out.push('ERR ' + e.message.split('\n')[0]) }
}
console.log(out.join('\n'))
