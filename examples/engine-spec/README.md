# `examples/engine-spec/`

These files are written in the intent-specification format of the RCE
engine (objectives, acceptances, deliverables and their reasons), not in the
RCE language. `rce check` does not read them, and no check in this
repository runs them. They were kept under `examples/` with the extension
`.rce`, where the checker reported that they require change at line 1
(`spec` is not a declaration); they now carry the extension `.spec`.

`form-arith.engine.json` and `form-demo.engine.json` are outputs of the
engine for two of them. The engine is not public code (see the RCE paper,
section 10), so nothing here can be reproduced from this repository.
