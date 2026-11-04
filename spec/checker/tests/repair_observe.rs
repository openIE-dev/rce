//! Rung f of `rce repair` (spec 10.13, RF-1 to RF-7): proposals written
//! outside RCE and judged by the gate of every rung; `observe`, which
//! records each proposal the checker would admit and never closes; `auto`,
//! which closes with the stochastic label; the registered promotion
//! criterion and its evaluator; and the check that the mode row of rung f
//! follows the recorded evaluation.
use rce_spec_check::admission::Mode;
use rce_spec_check::manifest::sha256_hex;
use rce_spec_check::{cli, proposals, repair};
use serde_json::{Value, json};

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn json_of(out: &str) -> Value {
    serde_json::from_str(out).unwrap_or_else(|e| panic!("{e}: {out}"))
}

/// A directory of its own for one test, removed at the end.
struct Dir(std::path::PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("rce-repair-observe-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn put(&self, name: &str, text: &str) -> String {
        let p = self.0.join(name);
        std::fs::write(&p, text).unwrap();
        p.to_str().unwrap().to_string()
    }
    fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.0).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read(p: &str) -> String {
    std::fs::read_to_string(p).unwrap()
}

fn canonical(text: &str) -> String {
    rce_spec_check::unparse::unparse(&rce_spec_check::parse::parse(text).unwrap())
}

/// `tests/repair/outside.rce` with the greeting fixed: no exact rung edits
/// a string, so only a proposal closes it.
const FIXED: &str = "intent \"a greeting and the size of a set\"\n    check greet() == \"hello\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hello\"\n\nfn size(n: u8) -> u8\n    return n\n";
/// Fixed another way: a second candidate the checker admits.
const FIXED2: &str = "intent \"a greeting and the size of a set\"\n    check greet() == \"hello\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hel\" + \"lo\"\n\nfn size(n: u8) -> u8\n    return n\n";
/// Fixed, but `size` now has edits and `size(2) >= 0` holds on every one of
/// them: the check is vacuous (W-MUT-1).
const VACUOUS: &str = "intent \"a greeting and the size of a set\"\n    check greet() == \"hello\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hello\"\n\nfn size(n: u8) -> u8\n    return n + 1\n";
/// Still wrong: the check fails.
const WRONG: &str = "intent \"a greeting and the size of a set\"\n    check greet() == \"hello\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hellp\"\n\nfn size(n: u8) -> u8\n    return n\n";
/// The check changed to what the program does.
const EDITS: &str = "intent \"a greeting and the size of a set\"\n    check greet() == \"hellx\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hellx\"\n\nfn size(n: u8) -> u8\n    return n\n";
/// The failing check removed.
const DROPS: &str = "intent \"a greeting and the size of a set\"\n    check size(2) >= 0\n\nfn greet() -> text\n    return \"hellx\"\n\nfn size(n: u8) -> u8\n    return n\n";
const NOPARSE: &str = "fn greet( -> text\n";

const OUTSIDE: &str = "tests/repair/outside.rce";
const COUNTER: &str = "tests/repair/counter.rce";
const COUNTER_GOOD: &str = "tests/repair/counter_good.rce";

/// The original of `tests/repair/counter.rce` (its GOOD), with `step on Inc`
/// written as the proposer writes it.
fn counter(inc: &str, show: &str) -> String {
    read(COUNTER_GOOD).replace("-> n + 2", &format!("-> {inc}")).replace("value: n)", &format!("value: {show})"))
}

fn proposals_file(d: &Dir, name: &str, list: &[(&str, &str)]) -> String {
    d.put(name, &proposals::write(list))
}

#[test]
fn a_proposals_file_is_canonical_json_and_each_proposal_has_its_digest() {
    let bytes = proposals::write(&[("person", FIXED), ("local model", WRONG)]);
    let v: Value = serde_json::from_str(&bytes).unwrap();
    assert_eq!(bytes, rce_spec_check::interp::canon_json(&v) + "\n");
    let ps = proposals::read("p.json", &bytes).unwrap();
    assert_eq!(ps.sha256, sha256_hex(bytes.as_bytes()));
    assert_eq!(ps.list.len(), 2);
    assert_eq!(ps.list[0].origin, "person");
    assert_eq!(ps.list[1].text, WRONG);
    // the digest: SHA-256 of `rce-proposal/1`, a line end and the text
    assert_eq!(ps.list[0].sha256, sha256_hex(format!("rce-proposal/1\n{FIXED}").as_bytes()));
    assert_eq!(ps.list[0].sha256, proposals::digest(FIXED));
    // anything else is a usage error that names RF-1
    let pretty = serde_json::to_string_pretty(&v).unwrap() + "\n";
    for bad in [
        pretty.as_str(),
        bytes.trim_end(),
        "{\"format\":\"rce-proposals/2\",\"proposals\":[]}\n",
        "{\"format\":\"rce-proposals/1\",\"proposals\":[{\"origin\":\"\",\"text\":\"x\"}]}\n",
        "{\"format\":\"rce-proposals/1\",\"proposals\":[{\"origin\":\"a\\nb\",\"text\":\"x\"}]}\n",
        "{\"format\":\"rce-proposals/1\",\"proposals\":[{\"model\":\"m\",\"origin\":\"a\",\"text\":\"x\"}]}\n",
        "{\"format\":\"rce-proposals/1\",\"model_calls\":1,\"proposals\":[]}\n",
        "[]\n",
    ] {
        let e = proposals::read("p.json", bad).unwrap_err();
        assert!(e.starts_with("p.json: RF-1 "), "{e}");
    }
    let d = Dir::new("rf1");
    let p = d.put("p.json", &pretty);
    let (code, out, err) = rce(&["repair", "proposals", &p, OUTSIDE]);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("RF-1") && out.is_empty(), "{err}");
}

#[test]
fn observe_records_and_never_closes() {
    let d = Dir::new("observe");
    let file = d.put("outside.rce", &read(OUTSIDE));
    let p = proposals_file(&d, "p.json", &[("person", WRONG), ("person", FIXED), ("local model", FIXED2)]);
    let before = d.names();
    let (code, out, err) = rce(&["repair", "json", "proposals", &p, &file]);
    assert_eq!(code, 1, "{out}{err}");
    let r = json_of(&out);
    // rung f is observe by its row in 10.13: nothing closes, nothing is written
    assert_eq!(r["closed"], false);
    assert!(r["rung"].is_null() && r["repaired"].is_null(), "{r}");
    assert_eq!(r["stochastic"], false);
    assert_eq!(r["model_calls"], 0);
    assert_eq!(read(&file), read(OUTSIDE));
    assert_eq!(d.names(), before);
    let f = &r["proposals"];
    assert_eq!(f["mode"], "observe");
    assert_eq!(f["ran"], true);
    assert_eq!((f["given"].as_u64(), f["judged"].as_u64(), f["admitted"].as_u64(), f["observed"].as_u64()), (Some(3), Some(3), Some(2), Some(2)), "{f}");
    assert_eq!(r["tried"]["f"], 3);
    let recs = f["records"].as_array().unwrap();
    assert!(recs[0]["verdict"].as_str().unwrap().starts_with("requires change"), "{f}");
    assert_eq!(recs[0]["observed"], false);
    for (k, text, origin) in [(1, FIXED, "person"), (2, FIXED2, "local model")] {
        let x = &recs[k];
        assert_eq!(x["index"], k + 1);
        assert_eq!(x["verdict"], "admitted");
        assert_eq!(x["observed"], true);
        assert_eq!(x["origin"], origin);
        assert_eq!(x["sha256"], proposals::digest(text).as_str());
        assert_eq!(x["mutation"]["vacuous"], 0);
        let diff: Vec<&str> = x["diff"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
        assert!(diff.iter().any(|l| l.starts_with('+') && l.contains("return \"hel")), "{diff:?}");
    }
    // every exact rung ran first and closed nothing
    assert!(r["attempts"].as_array().unwrap().iter().filter(|a| a["rung"] != "f").all(|a| a["verdict"].as_str().unwrap().starts_with("requires change")));
    // the attention line counts the open program and the two observed proposals
    assert_eq!(err, "attention: 1 requires change, 0 recommended, 0 altered, 2 observed\n");
    let (code, out, err) = rce(&["repair", "proposals", &p, &file]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.starts_with("rung      not closed\n"), "{out}");
    assert!(out.contains(", f 3 (candidates judged by the checker; 0 model calls)\n"), "{out}");
    assert!(out.contains("proposals 3 given, 3 judged, 2 admitted, 2 observed (rung f observe; sha256 "), "{out}");
    assert!(out.contains(&format!("observed  proposal 2 (origin `person`, sha256 {}): admitted;", proposals::digest(FIXED))), "{out}");
    assert!(out.contains("rung f is observe, so nothing closes"), "{out}");
    assert!(out.ends_with(&format!("not closed {file}\n")), "{out}");
    assert_eq!(err, "attention: 1 requires change, 0 recommended, 0 altered, 2 observed\n");
    assert_eq!(d.names(), before);
}

fn library(file: &str, list: &[(&str, &str)], mode: Mode, known: Option<&str>) -> (repair::Outcome, Value) {
    let ps = proposals::read("p.json", &proposals::write(list)).unwrap();
    let k = known.map(|g| repair::known(g, &read(g)).unwrap());
    let o = repair::repair_with(file, &read(file), &[], k.as_ref(), Some(repair::Outside { proposals: &ps, mode }), repair::DEFAULT_BOUND);
    let v = json_of(&repair::report_json(file, &o));
    (o, v)
}

#[test]
fn auto_closes_with_the_stochastic_label() {
    // auto is reached only through the library: no command word sets a mode
    let (o, r) = library(OUTSIDE, &[("person", WRONG), ("local model", FIXED), ("person", FIXED2)], Mode::Auto, None);
    assert!(o.closed);
    assert_eq!(r["rung"], "f");
    assert_eq!(r["stochastic"], true);
    assert_eq!(r["model_calls"], 0);
    let edit = r["edit"].as_str().unwrap();
    assert_eq!(edit, format!("proposal 2 (origin `local model`, sha256 {})", proposals::digest(FIXED)));
    let fixed = r["repaired"].as_str().unwrap();
    assert_eq!(fixed, canonical(FIXED));
    assert_eq!(rce_spec_check::check_one(fixed), Ok(()));
    let f = &r["proposals"];
    assert_eq!(f["mode"], "auto");
    assert_eq!((f["judged"].as_u64(), f["admitted"].as_u64(), f["observed"].as_u64()), (Some(2), Some(1), Some(0)), "{f}");
    assert_eq!(f["records"][1]["admitted"], true);
    assert_eq!(f["records"][1]["observed"], false);
    assert_eq!(f["records"][2]["verdict"], "not judged: proposal 2 closed the program");
    let text = repair::report(OUTSIDE, &o);
    assert!(text.starts_with("rung      f outside proposal (stochastic)\n"), "{text}");
    assert!(text.contains("edit      proposal 2 (origin `local model`"), "{text}");
    assert!(text.ends_with(&format!("closed {OUTSIDE}\n")), "{text}");
    assert!(r["verdict"].as_str().unwrap().starts_with(&format!("ok {OUTSIDE}: admitted; 0 examples and 2 checks hold")), "{r}");
}

#[test]
fn off_judges_nothing() {
    let (o, r) = library(OUTSIDE, &[("person", FIXED)], Mode::Off, None);
    assert!(!o.closed);
    assert_eq!(r["proposals"]["ran"], false);
    assert_eq!(r["proposals"]["judged"], 0);
    assert_eq!(r["tried"]["f"], 0);
    assert_eq!(r["proposals"]["records"][0]["verdict"], "not judged: rung f is off");
    assert!(r["skipped"].as_array().unwrap().iter().any(|s| s.as_str().unwrap().starts_with("f: rung f is off")), "{r}");
}

#[test]
fn the_modes_table_gives_rung_f_observe() {
    // RF-3: rung f has a row, `observe`; the exact rungs have none
    assert_eq!(rce_spec_check::specread::rung_mode("f"), Some("observe"));
    assert_eq!(proposals::mode(), Mode::Observe);
    for r in ["a", "b", "c", "d", "e"] {
        assert_eq!(rce_spec_check::specread::rung_mode(r), None, "{r}");
    }
    let d: Value = serde_json::from_str(&read("../data/spec-data.json")).unwrap();
    let rows = d["repair_modes"]["table"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["rung"], "f");
    assert_eq!(rows[0]["mode"], "observe");
    assert_eq!(d["repair_modes"]["from"], "12-tooling.md");
}

#[test]
fn rung_f_never_runs_when_an_exact_rung_closes() {
    let d = Dir::new("exact");
    let p = proposals_file(&d, "p.json", &[("person", FIXED)]);
    let (code, out, err) = rce(&["repair", "json", "proposals", &p, "tests/repair/enumerate.rce"]);
    assert_eq!(code, 0, "{out}{err}");
    let r = json_of(&out);
    assert_eq!(r["rung"], "d");
    assert_eq!(r["stochastic"], false);
    assert_eq!(r["tried"]["f"], 0);
    assert_eq!(r["proposals"]["ran"], false);
    assert_eq!(r["proposals"]["judged"], 0);
    assert_eq!(r["proposals"]["records"][0]["verdict"], "not judged: rung d closed the program");
    assert!(r["attempts"].as_array().unwrap().iter().all(|a| a["rung"] != "f"), "{r}");
    assert_eq!(err, "attention: 0 requires change, 0 recommended, 0 altered, 0 observed\n");
    // nor when the program already complies, nor when it does not parse
    let (code, out, _) = rce(&["repair", "json", "proposals", &p, "tests/repair/complies.rce"]);
    assert_eq!(code, 0, "{out}");
    let r = json_of(&out);
    assert_eq!(r["rung"], "none");
    assert_eq!(r["proposals"]["records"][0]["verdict"], "not judged: FILE complies with the Form");
    let bad = d.put("bad.rce", NOPARSE);
    let (code, out, _) = rce(&["repair", "json", "proposals", &p, &bad]);
    assert_eq!(code, 1, "{out}");
    let r = json_of(&out);
    assert_eq!(r["proposals"]["ran"], false);
    assert!(r["skipped"].as_array().unwrap().iter().any(|s| s.as_str().unwrap().starts_with("f: the text does not parse")), "{r}");
}

#[test]
fn a_proposal_the_checker_or_w_mut_1_refuses_is_not_admitted() {
    let d = Dir::new("refused");
    let p = proposals_file(&d, "p.json", &[("person", VACUOUS), ("person", WRONG), ("person", NOPARSE)]);
    let (code, out, err) = rce(&["repair", "json", "proposals", &p, OUTSIDE]);
    assert_eq!(code, 1, "{out}{err}");
    let r = json_of(&out);
    let f = &r["proposals"];
    assert_eq!((f["judged"].as_u64(), f["admitted"].as_u64(), f["observed"].as_u64()), (Some(3), Some(0), Some(0)), "{f}");
    let v = |k: usize| f["records"][k]["verdict"].as_str().unwrap().to_string();
    // its examples and checks hold, and `size(2) >= 0` holds on every edit of `n + 1`
    assert!(v(0).starts_with("requires change") && v(0).contains("W-MUT-1"), "{}", v(0));
    assert!(v(1).starts_with("requires change") && v(1).contains("W-INT-1"), "{}", v(1));
    assert!(v(2).starts_with("requires change"), "{}", v(2));
    assert!(f["records"].as_array().unwrap().iter().all(|x| x["admitted"] == false && x.get("diff").is_none()), "{f}");
    assert_eq!(err, "attention: 1 requires change, 0 recommended, 0 altered, 0 observed\n");
    // in auto too: a refused proposal never closes
    let (o, _) = library(OUTSIDE, &[("person", VACUOUS), ("person", WRONG)], Mode::Auto, None);
    assert!(!o.closed && !o.stochastic);
}

#[test]
fn a_proposal_that_edits_what_judges_it_requires_change() {
    let other = format!("module other\n\n{FIXED}");
    let relaid = FIXED.replace("    return \"hello\"", "    return (\"hello\")");
    let (o, r) = library(
        OUTSIDE,
        &[("person", EDITS), ("person", DROPS), ("person", &other), ("person", &read(OUTSIDE)), ("person", FIXED), ("person", &relaid)],
        Mode::Auto,
        None,
    );
    let v = |k: usize| r["proposals"]["records"][k]["verdict"].as_str().unwrap().to_string();
    // a changed check, a removed check and another module: RF-2, never judged by admission
    assert!(v(0).starts_with("requires change: RF-2 ") && v(0).contains("removed `check greet() == \"hello\"`") && v(0).contains("added `check greet() == \"hellx\"`"), "{}", v(0));
    assert!(v(1).starts_with("requires change: RF-2 ") && v(1).contains("removed `check greet() == \"hello\"`") && !v(1).contains("added"), "{}", v(1));
    assert!(v(2).starts_with("requires change: RF-2 the proposal names another module"), "{}", v(2));
    assert_eq!(v(3), "not judged: the proposal is FILE itself, in its canonical text");
    assert_eq!(v(4), "admitted");
    // auto stops at the first admitted proposal
    assert_eq!(v(5), "not judged: proposal 5 closed the program");
    assert!(o.closed && o.stochastic);
    // in observe, a proposal with the canonical text of an earlier one is judged once
    let (_, r) = library(OUTSIDE, &[("person", FIXED), ("local model", &relaid)], Mode::Observe, None);
    if canonical(&relaid) == canonical(FIXED) {
        assert_eq!(r["proposals"]["records"][1]["verdict"], "not judged: the candidate of proposal 1, judged once");
        assert_eq!(r["proposals"]["observed"], 1);
    } else {
        assert_eq!(r["proposals"]["observed"], 2);
    }
    // what an exact rung did not admit is not judged again
    let attempts: Vec<&Value> = r["attempts"].as_array().unwrap().iter().filter(|a| a["rung"] == "f").collect();
    assert_eq!(attempts.len(), r["proposals"]["judged"].as_u64().unwrap() as usize);
}

#[test]
fn the_report_is_byte_stable() {
    let d = Dir::new("bytes");
    let p = proposals_file(&d, "p.json", &[("person", FIXED), ("local model", WRONG)]);
    let args = ["repair", "json", "known", COUNTER_GOOD, "proposals", &p, OUTSIDE];
    let (c1, a, e1) = rce(&args);
    let (c2, b, e2) = rce(&args);
    assert_eq!((c1, &a, &e1), (c2, &b, &e2));
    let v = json_of(&a);
    // canonical JSON (6.1.1) and one line end
    assert_eq!(a, rce_spec_check::interp::canon_json(&v) + "\n");
    assert_eq!(v["format"], "rce-repair/1");
    assert_eq!(v["bound"], 2000);
    assert_eq!(v["inputs"], json!({
        "file": sha256_hex(read(OUTSIDE).as_bytes()),
        "known": sha256_hex(read(COUNTER_GOOD).as_bytes()),
        "proposals": sha256_hex(read(&p).as_bytes()),
        "modules": [],
    }));
    assert_eq!(v["proposals"]["sha256"], sha256_hex(read(&p).as_bytes()).as_str());
    let (_, t1, _) = rce(&["repair", "proposals", &p, OUTSIDE]);
    let (_, t2, _) = rce(&["repair", "proposals", &p, OUTSIDE]);
    assert_eq!(t1, t2);
    // without PROPOSALS the members are there, and rung f is not
    let (_, out, _) = rce(&["repair", "json", "tests/repair/enumerate.rce"]);
    let v = json_of(&out);
    assert!(v["proposals"].is_null() && v["inputs"]["proposals"].is_null() && v["tried"].get("f").is_none(), "{v}");
}

#[test]
fn an_observed_proposal_is_compared_with_good() {
    let same = counter("n + 2", "n");
    let equal = counter("2 + n", "n");
    let contra = counter("n + 2", "n + 1");
    let d = Dir::new("good");
    let p = proposals_file(&d, "p.json", &[("person", &same), ("person", &equal), ("local model", &contra)]);
    let (code, out, err) = rce(&["repair", "json", "known", COUNTER_GOOD, "proposals", &p, COUNTER]);
    assert_eq!(code, 1, "{out}{err}");
    let r = json_of(&out);
    let f = &r["proposals"];
    assert_eq!(f["observed"], 3, "{f}");
    let k = |i: usize| f["records"][i]["known"].clone();
    assert_eq!(k(0)["result"], "restored");
    assert_eq!(k(1)["result"], "behaviour-equal");
    assert_eq!(k(2)["result"], "contradicts");
    assert_eq!(k(2)["after"], json!(["Show()"]));
    assert!(k(2)["why"].as_str().unwrap().contains("GOOD emits Shown(value: 0), the proposal emits Shown(value: 1)"), "{}", k(2));
    assert_eq!(err, "attention: 1 requires change, 0 recommended, 0 altered, 3 observed\n");
    let (_, text, _) = rce(&["repair", "known", COUNTER_GOOD, "proposals", &p, COUNTER]);
    assert!(text.contains("          against GOOD: restored\n"), "{text}");
    assert!(text.contains("          against GOOD: contradicts: after Show(): GOOD emits Shown(value: 0), the proposal emits Shown(value: 1)\n"), "{text}");
    // a program with no step is not compared
    let (_, r) = library(OUTSIDE, &[("person", FIXED)], Mode::Observe, Some("tests/repair/outside_good.rce"));
    let x = &r["proposals"]["records"][0]["known"];
    assert_eq!(x["result"], "restored", "{x}");
    let (_, r) = library(OUTSIDE, &[("person", FIXED2)], Mode::Observe, Some("tests/repair/outside_good.rce"));
    let x = &r["proposals"]["records"][0]["known"];
    assert_eq!(x["result"], "not compared", "{x}");
}

// ---------- the registered criterion (RF-5 to RF-7) ----------

const REG: &str = "../proposals/repair-observe/criterion.json";
const PIN: &str = "../proposals/repair-observe/criterion.json.sha256";

#[test]
fn the_registration_is_dated_and_pinned() {
    let bytes = read(REG);
    let reg = proposals::registration(&bytes).unwrap();
    assert_eq!(reg.date, "2026-11-03");
    assert_eq!(reg.corpus_sha256, "6ae255a32211f620364ef713b1a36f7daac6ab46f22ac318b45acb2a6da9af50");
    assert_eq!(reg.tasks, 151);
    assert!(reg.known);
    assert_eq!(reg.bound, 2000);
    assert_eq!(reg.share, (2, 100));
    assert_eq!(read(PIN), format!("{}  criterion.json\n", sha256_hex(bytes.as_bytes())));
    let v: Value = serde_json::from_str(&bytes).unwrap();
    for c in ["C1", "C2", "C3"] {
        assert!(v["criterion"][c].as_str().is_some_and(|t| !t.is_empty()), "{c}");
    }
    assert_eq!(v["promotes"]["from"], "observe");
    assert_eq!(v["promotes"]["to"], "auto");
}

/// A registration of the same form as the real one, for a corpus of the
/// fixtures, with the largest share `share`.
fn registration(corpus: &str, tasks: usize, share: [u64; 2]) -> String {
    let mut v: Value = serde_json::from_str(&read(REG)).unwrap();
    v["id"] = json!("test");
    v["corpus"]["sha256"] = json!(sha256_hex(corpus.as_bytes()));
    v["corpus"]["tasks"] = json!(tasks);
    v["criterion"]["max_stochastic_share"] = json!(share);
    serde_json::to_string_pretty(&v).unwrap() + "\n"
}

/// The observation of a two-task corpus: `enumerate.rce`, which rung d
/// closes, and `counter.rce`, which only a proposal closes, each run as the
/// registration says (`known ORIGINAL`, `observe`, the default bound).
/// Returns the corpus and the records.
fn observation(d: &Dir, counter_proposals: &[(&str, &str)]) -> (String, String) {
    let (_, out, _) = rce(&["repair", "json", "tests/repair/enumerate.rce"]);
    let good = d.put("enumerate_good.rce", json_of(&out)["repaired"].as_str().unwrap());
    let pe = proposals_file(d, "pe.json", &[("person", FIXED)]);
    let pc = proposals_file(d, "pc.json", counter_proposals);
    let mut records = String::new();
    for (g, p, f) in [(good.as_str(), pe.as_str(), "tests/repair/enumerate.rce"), (COUNTER_GOOD, pc.as_str(), COUNTER)] {
        let (_, out, _) = rce(&["repair", "json", "known", g, "proposals", p, f]);
        records.push_str(&out);
    }
    let task = |id: &str, f: &str, g: &str| json!({"id": id, "mutant_sha256": sha256_hex(read(f).as_bytes()), "original_sha256": sha256_hex(read(g).as_bytes())}).to_string() + "\n";
    let corpus = task("enumerate", "tests/repair/enumerate.rce", &good) + &task("counter", COUNTER, COUNTER_GOOD);
    (corpus, records)
}

#[test]
fn the_criterion_holds_when_every_condition_holds() {
    let d = Dir::new("holds");
    let (corpus, records) = observation(&d, &[("person", &counter("n + 2", "n"))]);
    // one close of two from rung f is within a share of 1 of 2
    let reg = registration(&corpus, 2, [1, 2]);
    let v = proposals::evaluate(&reg, &corpus, &records).unwrap();
    assert_eq!(v["holds"], true, "{v}");
    assert_eq!(v["reasons"], json!([]));
    assert_eq!(v["closes"], json!({"exact": 1, "stochastic": 1, "max_share": [1, 2]}));
    assert_eq!(v["observed"]["proposals"], 1);
    assert_eq!(v["observed"]["w_mut_1"], 1);
    assert_eq!(v["observed"]["known"]["restored"], 1);
    assert_eq!(v["proposals"]["origins"], json!({"person": 1}));
    assert_eq!(v["registration"]["sha256"], sha256_hex(reg.as_bytes()).as_str());
    // the command: text, json, byte-stable, status 0
    let (r, c, x) = (d.put("reg.json", &reg), d.put("corpus.jsonl", &corpus), d.put("records.jsonl", &records));
    let (code, out, err) = rce(&["repair", "criterion", &r, &c, &x]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("closes        1 by rungs a to e, 1 that rung f would close: 1 of 2 (50.0%), at most 1 of 2\n"), "{out}");
    assert!(out.ends_with("criterion     holds: the mode row of rung f may be changed to auto (RF-7)\n"), "{out}");
    let (code, j1, _) = rce(&["repair", "criterion", "json", &r, &c, &x]);
    let (_, j2, _) = rce(&["repair", "criterion", "json", &r, &c, &x]);
    assert_eq!(code, 0);
    assert_eq!(j1, j2);
    assert_eq!(j1, proposals::evaluation_json(&v));
}

#[test]
fn the_criterion_fails_on_each_condition() {
    let d = Dir::new("fails");
    let (corpus, records) = observation(&d, &[("person", &counter("n + 2", "n"))]);
    let why = |reg: &str, recs: &str| {
        let v = proposals::evaluate(reg, &corpus, recs).unwrap();
        assert_eq!(v["holds"], false, "{v}");
        v["reasons"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>().join("; ")
    };
    // the share: 1 of 2 closes is above 2 of 100
    let reg = registration(&corpus, 2, [2, 100]);
    assert!(why(&reg, &records).contains("1 of 2 closes would come from rung f (50.0%), above 2 of 100"));
    let (r, c, x) = (d.put("reg.json", &reg), d.put("corpus.jsonl", &corpus), d.put("records.jsonl", &records));
    let (code, out, _) = rce(&["repair", "criterion", &r, &c, &x]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("criterion     does not hold: 1 of 2 closes"), "{out}");
    let reg = registration(&corpus, 2, [1, 2]);
    // W-MUT-1: an observed proposal whose record shows a vacuous obligation
    let tampered: String = records
        .lines()
        .map(|l| {
            let mut v: Value = serde_json::from_str(l).unwrap();
            if v["proposals"]["observed"] == 1 {
                v["proposals"]["records"][0]["mutation"]["vacuous"] = json!(1);
            }
            v.to_string() + "\n"
        })
        .collect();
    assert!(why(&reg, &tampered).contains("1 of 1 observed proposals do not pass W-MUT-1"));
    // GOOD: an observed proposal that contradicts the original
    let d2 = Dir::new("fails-good");
    let (corpus2, records2) = observation(&d2, &[("local model", &counter("n + 2", "n + 1"))]);
    assert_eq!(corpus2, corpus);
    assert!(why(&reg, &records2).contains("1 observed proposals contradict GOOD"));
    // coverage: a task without a record, and an open task with no proposal
    let first = records.lines().next().unwrap().to_string() + "\n";
    assert!(why(&reg, &first).contains("1 tasks have no record"));
    let d3 = Dir::new("fails-none");
    let (_, records3) = observation(&d3, &[]);
    assert!(why(&reg, &records3).contains("1 tasks that rungs a to e do not close had no proposal judged"));
    // a record made without GOOD is not the registered observation
    let (_, plain, _) = rce(&["repair", "json", "proposals", &d.put("p.json", &proposals::write(&[("person", FIXED)])), COUNTER]);
    assert!(why(&reg, &(first.clone() + &plain)).contains("1 tasks were not run with the registered GOOD"));
    // a corpus that is not the registered one is a usage error
    let e = proposals::evaluate(&reg, &(corpus.clone() + "\n"), &records).unwrap_err();
    assert!(e.contains("RF-6"), "{e}");
    let c2 = d.put("other.jsonl", &(corpus.clone() + "\n"));
    let (code, _, err) = rce(&["repair", "criterion", &d.put("reg2.json", &reg), &c2, &x]);
    assert_eq!(code, 2, "{err}");
}

#[test]
fn the_mode_row_of_rung_f_is_auto_only_with_a_met_criterion() {
    // the repository: rung f is observe, the registration is pinned, and no
    // evaluation is recorded
    let (reg, pin) = (read(REG), read(PIN));
    assert!(!std::path::Path::new("../proposals/repair-observe/evaluation.json").exists());
    let line = proposals::promotion_check(proposals::mode(), &reg, &pin, None).unwrap();
    assert!(line.starts_with("rung f of rce repair is observe; promotion criterion rce-repair-rung-f-promotion-2026-11-03 (registered 2026-11-03"), "{line}");
    assert!(line.ends_with("no evaluation recorded"), "{line}");
    // auto with no evaluation requires change
    let e = proposals::promotion_check(Mode::Auto, &reg, &pin, None).unwrap_err();
    assert!(e.starts_with("RF-7 the mode row of rung f is auto"), "{e}");
    // a registration edited after it was pinned
    let e = proposals::promotion_check(Mode::Observe, &reg.replace("2026-11-03, before", "2026-11-03,  before"), &pin, None).unwrap_err();
    assert!(e.contains("not the pinned"), "{e}");
    // an evaluation that meets the criterion allows auto; one that does not, or
    // whose `holds` disagrees with its counts, or of another registration, does not
    let d = Dir::new("row");
    let (corpus, records) = observation(&d, &[("person", &counter("n + 2", "n"))]);
    let reg = registration(&corpus, 2, [1, 2]);
    let pin = format!("{}  criterion.json\n", sha256_hex(reg.as_bytes()));
    let met = proposals::evaluation_json(&proposals::evaluate(&reg, &corpus, &records).unwrap());
    assert!(proposals::promotion_check(Mode::Auto, &reg, &pin, Some(&met)).unwrap().ends_with("the recorded evaluation meets it"));
    let strict = registration(&corpus, 2, [2, 100]);
    let strict_pin = format!("{}  criterion.json\n", sha256_hex(strict.as_bytes()));
    let unmet = proposals::evaluation_json(&proposals::evaluate(&strict, &corpus, &records).unwrap());
    let e = proposals::promotion_check(Mode::Auto, &strict, &strict_pin, Some(&unmet)).unwrap_err();
    assert!(e.contains("does not meet it") && e.contains("above 2 of 100"), "{e}");
    // observe with an unmet evaluation: nothing is promoted, nothing fails
    assert!(proposals::promotion_check(Mode::Observe, &strict, &strict_pin, Some(&unmet)).is_ok());
    let mut forged: Value = serde_json::from_str(&unmet).unwrap();
    forged["holds"] = json!(true);
    forged["reasons"] = json!([]);
    let e = proposals::promotion_check(Mode::Auto, &strict, &strict_pin, Some(&forged.to_string())).unwrap_err();
    assert!(e.contains("says `holds: true`"), "{e}");
    let e = proposals::promotion_check(Mode::Auto, &strict, &strict_pin, Some(&met)).unwrap_err();
    assert!(e.contains("not of this registration"), "{e}");
}
