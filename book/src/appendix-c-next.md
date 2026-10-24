# Appendix C: where to read next

- **The specification**, in `spec/book/` of the
  [rce repository](https://github.com/openIE-dev/rce), is the definition of
  the language. Start with `00-overview.md`. Section 7 lists every rule
  name you saw in a refusal, such as `W-OWN-1` or `W-INT-1`.
- **Concurrency** is Appendix C of the specification
  (`spec/book/14-draft06-concurrency.md`): `par`, budgets, workgroups,
  LVars, sessions and supervisors. A `# rce-version:` line does not turn
  those on for one module. The label names a release, and there is one
  language, the specification at one commit (draft 0.9).
- **The examples** in the specification (`spec/examples/`) are small
  programs for each feature, including programs that must be refused.
- **[programs/](../../programs/)** has
  longer programs grouped by area: simulation, control, media, networks,
  text, data, money, science, systems and hardware.
- **The IDE** sources are in [`ide/`](https://github.com/openIE-dev/rce/tree/main/ide)
  of the same repository and use the same checker.
