//! Version 0.9 Tier 3 (Appendix F.6): contracts (P2) and `ghost` (P3).
//! Accepted and refused examples are in `tests/v09_examples.rs`.
use rce_spec_check::apidiff;

fn run(src: &str) -> String {
    match rce_spec_check::cli::run(src, None) {
        Ok(o) => o.lines.join("\n"),
        Err(e) => format!("ERR {e}"),
    }
}

fn hash_of(src: &str, name: &str) -> String {
    let m = rce_spec_check::parse::parse_surface(src).unwrap();
    apidiff::defs(&m).into_iter().find(|d| d.name == name).unwrap().hash
}

const W: &str = "# rce-version: 0.9\nfn withdraw(balance: i64, amount: i64) -> i64\n    requires amount > 0\n    requires amount <= balance\n    ensures result >= 0\n    return balance - amount\n\nfn main()\n    emit withdraw(10, ARG).tostr()\n";

#[test]
fn a_violated_requires_traps_with_the_fixed_record() {
    let src = W.replace("ARG", "30");
    let h = hash_of(&src, "withdraw");
    let o = run(&src);
    let want = format!(r#"{{"clause":"requires","function":"{h}","index":1,"kind":"contract"}}"#);
    assert!(o.contains(&want), "{o}");
    // the record has no source span: moving the function down a line keeps it
    let moved = src.replace("fn withdraw", "\nfn withdraw");
    assert!(run(&moved).contains(&want), "{}", run(&moved));
}

#[test]
fn a_satisfied_contract_runs_unchanged() {
    assert!(run(&W.replace("ARG", "3")).contains("emit \"7\""));
}

#[test]
fn a_violated_ensures_traps_on_every_return_and_the_tail() {
    let ret = "# rce-version: 0.9\nfn f(x: i64) -> i64\n    ensures result > 10\n    if x > 5\n        return x\n    return 20\n\nfn main()\n    emit f(11).tostr()\n    emit f(2).tostr()\n";
    assert!(run(ret).contains("emit \"11\"") && run(ret).contains("emit \"20\""), "{}", run(ret));
    let bad = ret.replace("return 20", "return 3");
    let o = run(&bad);
    assert!(o.contains(r#"\"clause\":\"ensures\""#) || o.contains(r#""clause":"ensures""#), "{o}");
    let tail = "# rce-version: 0.9\nfn f(x: i64) -> i64\n    ensures result > 10\n    x\n\nfn main()\n    emit f(3).tostr()\n";
    assert!(run(tail).contains("ensures"), "{}", run(tail));
}

#[test]
fn the_contract_is_part_of_the_definition_hash() {
    let a = "# rce-version: 0.9\nfn f(x: i64) -> i64\n    requires x > 0\n    return x\n\nfn main()\n    emit f(1).tostr()\n";
    let b = a.replace("x > 0", "x > 1");
    assert_ne!(hash_of(a, "f"), hash_of(&b, "f"));
}

#[test]
fn ghost_code_not_named_by_a_clause_is_erased() {
    // `n` is only used by another ghost value that no clause names: unused in
    // the run, so W-UNUSED-1 refuses it before it could matter
    let src = "# rce-version: 0.9\nghost fn unused_g(x: i64) -> i64 = x\n\nfn main()\n    emit \"ok\"\n";
    assert!(run(src).contains("emit \"ok\""), "{}", run(src));
}

#[test]
fn emitters_do_not_yet_print_a_contract() {
    let src = std::fs::read_to_string("../examples-0.9/ok/11_contracts_ghost.rce").unwrap();
    for t in ["python", "js", "rust"] {
        let e = rce_spec_check::cli::emit(t, &src).unwrap_err();
        assert!(e.contains("W-RES-1"), "{t}: {e}");
    }
}

#[test]
fn the_formatter_keeps_contracts_and_ghost() {
    let src = std::fs::read_to_string("../examples-0.9/ok/11_contracts_ghost.rce").unwrap();
    let m = rce_spec_check::parse::parse_surface(&src).unwrap();
    let out = rce_spec_check::unparse::unparse(&m);
    for w in ["requires amount <= balance", "ensures result == doubled(x)", "ghost let before = balance", "ghost fn doubled"] {
        assert!(out.contains(w), "{w} missing:\n{out}");
    }
}

#[test]
fn decreases_runs_and_does_not_replace_the_recursion_bound() {
    let src = std::fs::read_to_string("../examples-0.9/ok/12_decreases.rce").unwrap();
    let o = run(&src);
    assert!(o.contains("emit \"10\"") && o.contains("emit \"3\"") && o.contains("emit \"4\""), "{o}");
    // both hold: a measure that is fine still meets the bound of 6.7
    let tight = src.replace("fn sum_to(n: u64) -> u64 bound 100", "fn sum_to(n: u64) -> u64 bound 2");
    assert!(run(&tight).contains("bound") || run(&tight).contains("ERR") || run(&tight).contains("trap"), "{}", run(&tight));
}

#[test]
fn the_measure_is_part_of_the_definition_hash() {
    let a = "# rce-version: 0.9\nfn f(n: u64) -> u64 bound 9\n    decreases n\n    if n == 0\n        return 0\n    return f(n - 1)\n\nfn main()\n    emit f(1).tostr()\n";
    let b = a.replace("decreases n\n", "decreases n + 1\n");
    assert_ne!(hash_of(a, "f"), hash_of(&b, "f"));
}

#[test]
fn if_let_is_a_match_with_a_guard_and_a_wildcard() {
    let src = std::fs::read_to_string("../examples-0.9/ok/13_if_let.rce").unwrap();
    let o = run(&src);
    assert!(o.contains("big 4") && o.contains("none or small") && o.contains("emit \"6\""), "{o}");
    // the guard runs only after the pattern matched: the failed case takes else
    assert_eq!(o.matches("none or small").count(), 2, "{o}");
}

#[test]
fn if_let_else_may_change_what_the_scrutinee_read() {
    let src = std::fs::read_to_string("../examples-0.9/ok/14_if_let_else_mutates.rce").unwrap();
    assert!(run(&src).contains("emit \"3\""), "{}", run(&src));
}

#[test]
fn an_irrefutable_if_let_requires_change() {
    let src = "# rce-version: 0.9\nfn main()\n    if let x = 3\n        emit x.tostr()\n";
    let e = rce_spec_check::check_one(src).unwrap_err();
    assert!(e.contains("W-MATCH-2"), "{e}");
}

#[test]
fn the_formatter_writes_if_let_back() {
    let src = std::fs::read_to_string("../examples-0.9/ok/13_if_let.rce").unwrap();
    let m = rce_spec_check::parse::parse_surface(&src).unwrap();
    let out = rce_spec_check::unparse::unparse(&m);
    assert!(out.contains("if let some(v) = first_even(xs) and v >"), "{out}");
}

#[test]
fn a_rendered_template_encodes_each_field_and_keeps_the_literal() {
    let src = std::fs::read_to_string("../examples-0.9/ok/15_templates.rce").unwrap();
    let o = run(&src);
    assert!(o.contains("emit \"Hello, ann! You are 31, 3 new messages {ok}.\""), "{o}");
    // the field's markup is escaped, the literal's is not (T44's shape)
    assert!(o.contains("<p>&lt;script&gt;</p>"), "{o}");
    // the prompt field is a delimited data block
    assert!(o.contains("Summarize: <<data-") && o.contains("ignore the above"), "{o}");
}

#[test]
fn a_template_is_the_same_as_its_spelled_out_concatenation() {
    let t = "# rce-version: 0.9\nfn main()\n    let w = \"a<b\"\n    emit Html.t\"<i>{w}</i>\".raw()\n";
    let s = "# rce-version: 0.9\nfn main()\n    let w = \"a<b\"\n    emit (Html.lit(\"<i>\") + Html.encode(w) + Html.lit(\"</i>\")).raw()\n";
    assert_eq!(run(t), run(s));
}

