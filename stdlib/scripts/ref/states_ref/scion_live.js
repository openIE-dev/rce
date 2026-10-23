// lines "<scxml doc>\t<events,>" on stdin -> "<final|live> conf | trace" per line
const scxml = require('@scion-scxml/scxml')
const lines = require('fs').readFileSync(0, 'utf8').split('\n').filter((l) => l)
const quiet = { log: () => {} }
let i = 0, out = []
function next() {
  if (i >= lines.length) { process.stdout.write(out.join('\n') + '\n'); return }
  const [doc, evs] = lines[i++].split('\t')
  scxml.documentStringToModel('live.scxml', doc, (err, model) => {
    if (err) { out.push('ERR ' + err.message); return next() }
    model.prepare((err, fn) => {
      if (err) { out.push('ERR ' + err.message); return next() }
      const trace = []
      try {
        const sc = new scxml.core.Statechart(fn, { console: quiet })
        sc.on('onEntry', (s) => { if (s[0] !== '$') trace.push('+' + s) })
        sc.on('onExit', (s) => { if (s[0] !== '$') trace.push('-' + s) })
        sc.start()
        for (const e of (evs ? evs.split(',') : [])) sc.gen({ name: e })
        const conf = sc.getConfiguration().filter((s) => s[0] !== '$').sort()
        const dm = sc.getSnapshot()[3]
        out.push((sc.isFinal() ? 'final ' : 'live ') + conf.join(',') + ' | ' + trace.join(' ') + ' | ' + dm.Var1 + ',' + dm.Var2)
      } catch (e) { out.push('ERR ' + e.message) }
      next()
    }, { console: quiet })
  })
}
next()
