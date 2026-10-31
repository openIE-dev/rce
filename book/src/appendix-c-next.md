# Appendix C: where to read next

- **The specification**, in `spec/book/` of the
  [rce repository](https://github.com/openIE-dev/rce), is the definition of
  the language. Start with `00-overview.md`. Section 7 lists every rule
  name you saw in a "requires change" report, such as `W-OWN-1` or `W-INT-1`.
- **Appendix C of the specification** (`spec/book/14-draft06-concurrency.md`)
  defines the concurrency of chapter 14 in full, with the workgroups,
  sessions, supervisors, LVars and replicated state this book leaves out.
  It applies to every module: a `# rce-version:` line is a release label
  that selects nothing (specification 2.7).
- **Appendix F of the specification** (`spec/book/17-draft09.md`) holds the
  draft 0.9 rules: the hygiene rules of chapters 2 and 3, contracts,
  measures, `if let`, templates, labels and capabilities. Its section F.12,
  and `rce v09-status`, say which rules the checker implements.
- **Appendix E of the specification** (`spec/book/16-host-contract.md`) is
  the host contract behind chapter 16, and `docs/DEPLOY.md` runs a program
  as a service.
- **The examples** in the specification (`spec/examples/`,
  `spec/examples-0.9/`, `spec/conc06/` and `spec/conc06b/`) are small
  programs for each feature, including programs that must require change.
- **The proposal ledger** (`spec/proposals/LEDGER.md`) records every
  proposal to change the language and its decision.
- **[programs/](https://github.com/openIE-dev/rce/tree/main/programs)** has
  longer programs grouped by area, among them data, hardware, media, money,
  networks, science, systems and text.
- **`ide/`** in the repository holds the RCE IDE, whose checker and
  language server are the ones this book uses.
