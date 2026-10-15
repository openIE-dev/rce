// Run an SCXML document in SCION (@scion-scxml/scxml 4.3.28) with a
// virtual clock; print the outcome and the entry/exit trace.
const scxml = require('@scion-scxml/scxml')
const fs = require('fs')
function runFile(path, events, cb) {
  scxml.documentStringToModel(path, fs.readFileSync(path, 'utf8'), (err, model) => {
    if (err) return cb('ERR ' + err.message)
    model.prepare((err, fn) => {
      if (err) return cb('ERR ' + err.message)
      const trace = []
      const sc = new scxml.core.Statechart(fn, { console: { log: () => {} } })
      sc.on('onEntry', (s) => trace.push('+' + s))
      sc.on('onExit', (s) => trace.push('-' + s))
      sc.start()
      setTimeout(() => {
        const conf = sc.getConfiguration().slice().sort()
        cb((sc.isFinal() ? 'final ' : 'live ') + conf.join(',') + ' | ' + trace.join(' '))
      }, 2500)
    }, { console: { log: () => {} } })
  })
}
module.exports = { runFile }
if (require.main === module) runFile(process.argv[2], [], (r) => console.log(r))
