//! Laws (spec 6.11, W-LAW-1, `spec/proposals/laws.md`): method D1 on the
//! flag game of the proposal and on small programs for each verdict.
//!
//! * `tests/law/flag_walls.rce`: the board with its far-edge walls; every
//!   law holds over 72 reachable states.
//! * `tests/law/flag_wrap.rce`: the same board without those walls; the law
//!   fails after 9 events, and the printed counterexample, pasted, holds.
//! * `rce change` between the two is a meaning change (CC-L1).
//! * `tests/law/counter.rce`: undecided at the bound LAW_FUEL.
//! * `tests/law/deposit.rce`: undecided (a `u64` payload), attested, and the
//!   attestation lapses when a covered definition changes.
//! * `tests/law/keys.rce`: `every step` claims with `kept: X`, `applied` and
//!   `old(f)`.
use rce_spec_check::{cli, lalr, law, parse};
use std::path::PathBuf;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("tests/law/{name}")).unwrap()
}

/// A scratch directory of this test process.
fn scratch(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-law-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p: PathBuf = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const COUNTEREXAMPLE: &str = "    example Up, Up, Up, Up, Right, Right, Right, Right, Right -> won";

/// The flag game without its law: the text before the `law` lines and after.
fn without_law(src: &str) -> String {
    let mut out = vec![];
    let mut skip = false;
    for l in src.lines() {
        if l.trim_start().starts_with("law ") {
            skip = true;
            continue;
        }
        if skip && l.starts_with("        ") {
            continue;
        }
        skip = false;
        out.push(l);
    }
    out.join("\n") + "\n"
}

/// `line` inserted after the first line (the `intent` line).
fn after_intent(src: &str, line: &str) -> String {
    let (head, rest) = src.split_once('\n').unwrap();
    format!("{head}\n{line}\n{rest}")
}

#[test]
fn walls_every_law_holds_over_72_states() {
    let (code, out) = rce(&["law", "tests/law/flag_walls.rce"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(out.lines().count(), 3, "{out}");
    for name in ["you_cant_win", "walls_hold", "up_keeps_x"] {
        let l = out.lines().find(|l| l.starts_with(&format!("law {name} "))).unwrap_or_else(|| panic!("{out}"));
        assert!(l.contains(" holds "), "{l}");
        assert!(l.ends_with("exhaustive  72 states, 288 steps, alphabet 4 (complete), closed"), "{l}");
    }
    let (code, out) = rce(&["check", "tests/law/flag_walls.rce"]);
    assert_eq!(code, 0, "{out}");
    // a law is an obligation beyond the examples: no W-INT-2
    assert_eq!(out, "ok tests/law/flag_walls.rce\n");
}

#[test]
fn wrap_requires_change_with_a_pasteable_counterexample() {
    let (code, out) = rce(&["check", "tests/law/flag_wrap.rce"]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.starts_with(
            "requires change tests/law/flag_wrap.rce: line 4: W-LAW-1 law you_cant_win fails after 9 events (exhaustive, alphabet 4 complete,"
        ),
        "{out}"
    );
    assert!(out.contains("[requires change]\n"), "{out}");
    let printed = out.lines().nth(1).unwrap();
    assert_eq!(printed, COUNTEREXAMPLE);
    let (code, law_out) = rce(&["law", "tests/law/flag_wrap.rce"]);
    assert_eq!(code, 1);
    assert!(law_out.starts_with("law you_cant_win  fails      exhaustive  after 9 events;"), "{law_out}");
    assert_eq!(law_out.lines().nth(1).unwrap(), COUNTEREXAMPLE);

    // pasted as it is, the line holds: `rce test` passes it
    let wrap = fixture("flag_wrap.rce");
    let pasted = scratch("wrap_pasted.rce", &after_intent(&without_law(&wrap), printed));
    let (code, out) = rce(&["test", &pasted]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(3 examples, 0 checks)"), "{out}");
    // with the law still there the example holds and only the law is reported
    let with_law = scratch("wrap_pasted_law.rce", &after_intent(&wrap, printed));
    let (code, out) = rce(&["test", &with_law]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("W-LAW-1") && !out.contains("W-INT-1"), "{out}");
    // with the law's outcome, the line is the regression example: it fails
    // on the wrapping board and holds on the walled one
    let regress = printed.replace("-> won", "-> not won");
    let (code, out) = rce(&["test", &scratch("wrap_regress.rce", &after_intent(&without_law(&wrap), &regress))]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("W-INT-1 example does not hold"), "{out}");
    let walls = fixture("flag_walls.rce");
    let (code, out) = rce(&["test", &scratch("walls_regress.rce", &after_intent(&walls, &regress))]);
    assert_eq!(code, 0, "{out}");
}

#[test]
fn removing_the_walls_is_a_meaning_change() {
    let (code, out) = rce(&["change", "tests/law/flag_walls.rce", "tests/law/flag_wrap.rce"]);
    assert_eq!(code, 1, "{out}");
    // the behaviour check alone sees no difference: no step emits or keeps
    assert!(out.contains("behaviour verdict             equivalent"), "{out}");
    assert!(out.contains("law                 = you_cant_win  unchanged; the law of OLD fails on NEW after 9 events\n"), "{out}");
    assert!(out.contains(&format!("\n                    {}\n", COUNTEREXAMPLE.trim_start())), "{out}");
    assert!(out.ends_with("verdict             meaning change  NEW breaks law you_cant_win of OLD (CC-L1)\n"), "{out}");
    // without the laws the same change is an erratum: the gap the law closes
    let a = scratch("walls_nolaw.rce", &without_law(&fixture("flag_walls.rce")));
    let b = scratch("wrap_nolaw.rce", &without_law(&fixture("flag_wrap.rce")));
    let (code, out) = rce(&["change", &a, &b]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("verdict             erratum"), "{out}");
    // a law added to unchanged code is an erratum (CC-L2, decision 7)
    let (code, out) = rce(&["change", &a, "tests/law/flag_walls.rce"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("law                 + you_cant_win  added"), "{out}");
    assert!(out.contains("verdict             erratum"), "{out}");
    // an edited law is a meaning change (CC-L2)
    let edited = scratch("walls_edited.rce", &fixture("flag_walls.rce").replace("always not won", "always not won and x < 12"));
    let (code, out) = rce(&["change", "tests/law/flag_walls.rce", &edited]);
    assert_eq!(code, 1, "{out}");
    assert!(out.ends_with("verdict             meaning change  law you_cant_win edited (CC-L2)\n"), "{out}");
}

#[test]
fn the_bound_leaves_a_law_undecided() {
    let (code, out) = rce(&["law", "tests/law/counter.rce"]);
    assert_eq!(code, 3, "{out}");
    assert!(
        out.starts_with("law below_four_billion  undecided  exhaustive  bound reached at 1,000,000 states (LAW_FUEL); attest with \"WHO;YYYY-MM-DD;"),
        "{out}"
    );
    let (code, out) = rce(&["check", "tests/law/counter.rce"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("line 3: W-LAW-1 law below_four_billion undecided (exhaustive): bound reached at 1,000,000 states (LAW_FUEL)"), "{out}");
    assert_eq!(law::LAW_FUEL, 1_000_000);
    // a smaller bound is the same exploration, cut sooner
    let m = rce_spec_check::checked_program(&fixture("counter.rce")).unwrap();
    let d = law::decide_with(&m, 10);
    assert!(matches!(&d[0].verdict, law::Verdict::Undecided { why } if why == "bound reached at 10 states (LAW_FUEL)"), "{:?}", d[0].verdict);
}

/// The cover the report of an undecided law prints.
fn cover_of(out: &str) -> String {
    let i = out.find("WHO;YYYY-MM-DD;").unwrap() + "WHO;YYYY-MM-DD;".len();
    out[i..i + 64].to_string()
}

#[test]
fn an_attested_law_is_admitted_until_a_covered_definition_changes() {
    let src = fixture("deposit.rce");
    let (code, out) = rce(&["law", "tests/law/deposit.rce"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("undecided  exhaustive  alphabet incomplete: Deposit.amount (u64)"), "{out}");
    let (code, out) = rce(&["check", "tests/law/deposit.rce"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("W-LAW-1 law never_shrinks undecided (exhaustive): alphabet incomplete: Deposit.amount (u64)"), "{out}");
    let cover = cover_of(&out);
    let attest = format!("        attest \"A. Person;2026-11-01;{cover}\"");
    let attested_src = src.replace("        every step -> total >= old(total)\n", &format!("        every step -> total >= old(total)\n{attest}\n"));
    let attested = scratch("deposit_attested.rce", &attested_src);
    let (code, out) = rce(&["check", &attested]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = rce(&["law", &attested]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("attested   attestation  by A. Person on 2026-11-01 (exhaustive: alphabet incomplete: Deposit.amount (u64))"), "{out}");
    // a definition the law does not cover changes: still attested
    let other = scratch("deposit_other.rce", &attested_src.replace("fn unused(a: u64) -> u64\n    return a\n", "fn unused(a: u64) -> u64\n    return a + 1\n"));
    let (code, out) = rce(&["check", &other]);
    assert_eq!(code, 0, "{out}");
    // a covered definition changes (a function the step calls): the attestation lapses
    let lapsed = scratch("deposit_lapsed.rce", &attested_src.replace("fn add(a: u64, b: u64) -> u64\n    return a + b\n", "fn add(a: u64, b: u64) -> u64\n    return a + b + 0\n"));
    let (code, out) = rce(&["check", &lapsed]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the attestation by A. Person on 2026-11-01 lapsed: a definition it covers changed"), "{out}");
    assert_ne!(cover_of(&out), cover);
    // a malformed attestation requires change
    let bad = scratch("deposit_bad.rce", &attested_src.replace(";2026-11-01;", ";1 Nov 2026;"));
    let (code, out) = rce(&["check", &bad]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("is not WHO;YYYY-MM-DD;COVER"), "{out}");
}

#[test]
fn every_step_claims() {
    let (code, out) = rce(&["law", "tests/law/keys.rce"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(out.lines().filter(|l| l.contains(" holds ") && l.contains("256 states, 131,072 steps, alphabet 512 (complete), closed")).count(), 4, "{out}");
    // a kept step where the law says `applied`, and a false outcome: each
    // counterexample states what the program does now, and holds when pasted
    let src = fixture("keys.rce")
        .replace("every step on Key(code: 27, down: true) -> kept: Quit", "every step on Key(code: 27, down: true) -> applied")
        .replace("law releases_count_nothing", "law releases_count")
        .replace("-> presses == old(presses)\n", "-> presses == old(presses) + 1\n");
    let bad = scratch("keys_bad.rce", &src);
    let (code, out) = rce(&["check", &bad]);
    assert_eq!(code, 1, "{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 4, "{out}");
    assert!(lines[0].contains("W-LAW-1 law escape_quits fails after 1 event") && lines[0].contains("the event is kept: Quit"), "{out}");
    assert_eq!(lines[1], "    example Key(code: 27, down: true) -> kept: Quit");
    assert!(lines[2].starts_with("line 10: W-LAW-1 law releases_count fails after 1 event"), "{out}");
    assert_eq!(lines[3], "    example Key(code: 0, down: false) -> 0");
    let fixed = fixture("keys.rce");
    let pasted = scratch("keys_pasted.rce", &after_intent(&fixed, &format!("{}\n{}", lines[1], lines[3])));
    let (code, out) = rce(&["test", &pasted]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(4 examples, 0 checks)"), "{out}");
}

#[test]
fn verdicts_are_deterministic_and_name_free() {
    for f in ["tests/law/flag_walls.rce", "tests/law/flag_wrap.rce", "tests/law/keys.rce", "tests/law/deposit.rce"] {
        assert_eq!(rce(&["law", f]), rce(&["law", f]), "{f}");
        assert_eq!(rce(&["check", f]), rce(&["check", f]), "{f}");
    }
    // the name-free projection (10.6) has the same verdict and the same
    // counterexample up to names
    let (code, nl) = rce(&["fmt", "nameless", "tests/law/flag_wrap.rce"]);
    assert_eq!(code, 0, "{nl}");
    let p = scratch("wrap_nameless.rce", &nl);
    let (code, out) = rce(&["law", &p]);
    assert_eq!(code, 1, "{out}");
    let ex = out.lines().nth(1).unwrap().trim_start();
    let evs: Vec<&str> = ex.strip_prefix("example ").unwrap().split(" -> ").next().unwrap().split(", ").collect();
    assert_eq!(evs.len(), 9, "{ex}");
    assert!(evs[..4].iter().all(|e| *e == evs[0]) && evs[4..].iter().all(|e| *e == evs[4]) && evs[0] != evs[4], "{ex}");
}

#[test]
fn the_grammars_agree_on_laws() {
    let md = std::fs::read_to_string("../book/03-syntax.md").unwrap();
    let l = lalr::Lalr::build(lalr::book_grammar(&md).unwrap());
    assert!(l.conflicts.is_empty(), "{:?}", l.conflicts);
    for f in ["flag_walls.rce", "flag_wrap.rce", "counter.rce", "deposit.rce", "keys.rce"] {
        let src = fixture(f);
        assert!(parse::parse(&src).is_ok(), "{f}");
        assert!(l.accepts(&src), "LALR(1) rejects {f}");
    }
    // the words of a law are names elsewhere (3.11 R10)
    let names = "state S\n    applied: bool = false\n    every: u8 = 0\n\nevent E\n\nstep S on E\n    let attest = 1\n    let always = attest\n    every = always as u8\n    applied = true\n\nlaw l1\n    every step -> (applied)\n\nlaw l2\n    always not applied\n";
    assert!(parse::parse(names).is_ok(), "{:?}", parse::parse(names).err());
    assert!(l.accepts(names));
    let (code, out) = rce(&["check", &scratch("names.rce", names)]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("W-LAW-1 law l2 fails after 1 event") && !out.contains("law l1"), "{out}");
    assert!(out.ends_with("\n    example E -> applied\n"), "{out}");
    // `law` is reserved: as a name it requires change (2.4); in a trait it is
    // the trait law of F.6.6, reserved, not implemented
    let e = parse::parse("fn f() -> i64\n    let law = 1\n    return law\n").unwrap_err();
    assert!(e.contains("W-NAME-2 `law` is reserved"), "{e}");
    let e = parse::parse("trait T\n    law l(x: i64): x == x\n").unwrap_err();
    assert!(e.contains("W-RES-1 `law` in a trait"), "{e}");
    // a law has exactly one claim, and the unimplemented methods are not syntax
    for bad in ["law l\n    always true\n    always false\n", "law l\n    \"prose only\"\n", "law l\n    always true\n    invariant true\n"] {
        assert!(parse::parse(bad).is_err(), "{bad}");
        assert!(!l.accepts(bad), "{bad}");
    }
    // the text round-trips through the source printer
    for f in ["flag_walls.rce", "keys.rce"] {
        let m = parse::parse(&fixture(f)).unwrap();
        let t = rce_spec_check::unparse::unparse(&m);
        let m2 = parse::parse(&t).unwrap();
        assert_eq!(rce_spec_check::unparse::unparse(&m2), t, "{f}");
        assert_eq!(law::laws(&m).len(), law::laws(&m2).len());
    }
}
