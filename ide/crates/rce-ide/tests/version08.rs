//! The IDE's diagnostics cover the current language (the checker at
//! ../spec/checker of the same commit): refined types `where` (5.1), the
//! indexed values of 5.16 (Text[Html|Url|Sql|Json], Secret, Cap, Guarded),
//! mutual recursion (W-REC-1) and declared bounds `fn … bound n` (6.7).
//! There is one language (decision 64): a `# rce-version:` line is a
//! tolerated, ignored label, never a gate.
use rce_ide::lsp::Lsp;

fn diags(src: &str) -> Vec<String> {
    let mut l = Lsp::in_process();
    l.open("file:///t.rce", src).into_iter().map(|d| format!("{}: {}", d.line, d.message)).collect()
}
fn clean(src: &str) {
    let d = diags(src);
    assert!(d.is_empty(), "{d:?}");
}
fn reports_fit(src: &str, code: &str, line: usize) {
    let d = diags(src);
    assert!(d.iter().any(|x| x.starts_with(&format!("{line}: ")) && x.contains(code)), "want {code} on line {line}: {d:?}");
}

#[test]
fn refined_types_check() {
    clean("type Pct = u32 where self <= 100\nfn f(p: Pct) -> u32 = p\n");
    clean("type Sku = text where self.len() > 0 and self.len() <= 12\n");
}

#[test]
fn indexed_values_of_version_08() {
    clean("# rce-version: 0.8\nfn f(s: text) -> text = (Html.lit(\"<b>\") + Html.encode(s) + Html.lit(\"</b>\")).raw()\n");
    clean("# rce-version: 0.8\nstruct Root\nstate P\n    limit: Guarded[u32, Root] = 100\nevent Set(admin: Cap[Root], limit: u32)\nstep P on Set(admin: _, limit: n) -> n\n");
    reports_fit("# rce-version: 0.8\nfn f(s: text) -> Text[Html] = Html.lit(\"<b>\") + Url.encode(s)\n", "W-CTX-1", 2);
    reports_fit("# rce-version: 0.8\nstate V\n    s: Secret[text]? = none\nevent E(p: Secret[text])\nstep V on E(p: p)\n    emit p\n", "W-SEC-1", 6);
}

#[test]
fn version_label_is_not_a_gate() {
    // One language: the label is ignored and `Secret` is always a known name.
    clean("# rce-version: 0.7\nstate A\n    n: u64 = 0\nevent Login(pw: Secret[text])\nstep A on Login(pw: _) -> n + 1\n");
}

#[test]
fn mutual_recursion_and_bounds() {
    reports_fit("fn a(n: u32) -> u32\n    return b(n)\nfn b(n: u32) -> u32\n    return a(n)\n", "W-REC-1", 2);
    clean("fn a(n: u32) -> u32 bound 3\n    return b(n)\nfn b(n: u32) -> u32\n    return a(n)\n");
}
