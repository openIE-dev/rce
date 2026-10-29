use rce_memory::engine::{MemEngine, MemoryEngine};
use rce_memory::fact::Provenance;
use rce_memory::notes::*;

#[test]
fn parse_render_round_trip() {
    let text = "---\nsubject: Account\npredicate: decision\ntitle: Refuse overdrafts\nvalid_from: 2026-10-12\ntags:\n- design\n---\n# Refuse overdrafts\n\nBody with [[wikilink]].\n";
    let n = parse_note(text).unwrap();
    assert_eq!(n.front.subject.as_deref(), Some("Account"));
    assert_eq!(n.front.tags, ["design"]);
    assert!(n.body.starts_with("# Refuse overdrafts"));
    let again = parse_note(&render_note(&n)).unwrap();
    assert_eq!(again, n);
}

#[test]
fn no_frontmatter_and_bad_frontmatter() {
    let n = parse_note("# Just a note\ntext").unwrap();
    assert_eq!(n.front, Front::default());
    assert!(parse_note("---\nsubject: x\nno end").is_err());
    let f = note_fact("notes/idea.md", &n);
    assert_eq!(
        (f.subject.as_str(), f.predicate.as_str(), f.object.as_str()),
        ("note:notes/idea", "note", "Just a note")
    );
}

#[test]
fn time_round_trip() {
    for ms in [0i64, 1_760_288_400_123, -86_400_000, 951_782_400_000] {
        assert_eq!(parse_time(&format_time(ms)), Some(ms), "{ms}");
    }
    assert_eq!(parse_time("2026-10-12"), Some(1_791_763_200_000));
    assert_eq!(format_time(1_791_763_200_000), "2026-10-12T00:00:00.000Z");
    assert_eq!(parse_time("12345"), Some(12345));
    assert_eq!(parse_time("2026-13-01"), None);
}

#[test]
fn hook_and_instruction_files_are_not_written() {
    for bad in [
        "notes/CLAUDE.md",
        "notes/AGENTS.md",
        "AGENTS.md",
        ".cursor/rules.md",
        "notes/.git/hooks/x.md",
        "notes/hooks/pre.md",
        "notes/x.sh",
        "../x.md",
        "/etc/x.md",
        "notes/.cursorrules",
        ".github/copilot-instructions.md",
        "notes/x/GEMINI.md",
    ] {
        assert!(check_write_path(bad).is_err(), "{bad} must be refused");
    }
    check_write_path("notes/design/refuse-overdrafts.md").unwrap();
    let d = tempfile::tempdir().unwrap();
    let ns = Notes::new(d.path(), "notes").unwrap();
    assert!(ns.write("notes/CLAUDE.md", "x").is_err());
    assert!(ns.write("other/x.md", "x").is_err(), "outside notes dir");
    assert!(!d.path().join("notes/CLAUDE.md").exists());
    assert!(Notes::new(d.path(), ".cursor").is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_notes_dir_is_not_used() {
    let d = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(out.path(), d.path().join("notes")).unwrap();
    let ns = Notes::new(d.path(), "notes").unwrap();
    assert!(ns.write("notes/a.md", "x").is_err());
    assert!(!out.path().join("a.md").exists());
}

fn sync(ns: &Notes, e: &mut MemEngine, now: i64) -> SyncReport {
    let (p, r) = plan_sync(ns, e, "h1", now).unwrap();
    if !p.is_empty() {
        e.commit(&p).unwrap();
    }
    r
}

#[test]
fn sync_both_ways_files_win() {
    let d = tempfile::tempdir().unwrap();
    let ns = Notes::new(d.path(), "notes").unwrap();
    let mut e = MemEngine::default();
    std::fs::create_dir_all(d.path().join("notes/sub")).unwrap();
    std::fs::write(
        d.path().join("notes/a.md"),
        "---\nsubject: Account\npredicate: decision\n---\n# Refuse overdrafts\nbecause.\n",
    )
    .unwrap();
    std::fs::write(d.path().join("notes/sub/b.md"), "# B\n").unwrap();
    std::fs::write(d.path().join("notes/.hidden.md"), "# hidden\n").unwrap();
    let r = sync(&ns, &mut e, 100);
    assert_eq!(r.imported, ["notes/a.md", "notes/sub/b.md"]);
    let a = e
        .list_live()
        .unwrap()
        .into_iter()
        .find(|f| f.subject == "Account")
        .unwrap();
    assert_eq!(
        (a.object.as_str(), a.provenance, a.valid_from),
        ("Refuse overdrafts", Provenance::Extracted, 100)
    );
    // idempotent
    let r = sync(&ns, &mut e, 110);
    assert_eq!((r.unchanged, r.imported.len()), (2, 0));
    // a hand edit wins over memory: superseded in the same lineage
    std::fs::write(d.path().join("notes/a.md"), "---\nsubject: Account\npredicate: decision\n---\n# Refuse overdrafts\nbecause regulators.\n").unwrap();
    let r = sync(&ns, &mut e, 120);
    assert_eq!(r.updated, ["notes/a.md"]);
    let h = e.history(a.lineage).unwrap();
    assert_eq!(h.len(), 2);
    assert_eq!(h[0].expired_at, Some(120));
    assert!(h[1].text.contains("regulators"));
    assert_eq!(h[1].valid_from, 100, "world time kept across an edit");
    // memory changed behind the file's back: the file still wins
    let mut stale = h[1].clone();
    stale.id = e.next_id().unwrap();
    stale.text = "memory-only change".into();
    e.commit(&rce_memory::engine::Plan {
        inserts: vec![stale],
        expires: vec![(h[1].id, 125)],
        ..Default::default()
    })
    .unwrap();
    let r = sync(&ns, &mut e, 130);
    assert_eq!(r.updated, ["notes/a.md"]);
    let live = e
        .list_live()
        .unwrap()
        .into_iter()
        .find(|f| f.subject == "Account")
        .unwrap();
    assert!(live.text.contains("regulators"));
    // deleting a file expires its fact; nothing is deleted from memory
    std::fs::remove_file(d.path().join("notes/sub/b.md")).unwrap();
    let r = sync(&ns, &mut e, 140);
    assert_eq!(r.expired, ["notes/sub/b.md"]);
    assert!(
        e.scan()
            .unwrap()
            .iter()
            .any(|f| f.object == "B" && f.expired_at == Some(140))
    );
    // status: forgotten expires without deleting the file
    std::fs::write(
        d.path().join("notes/a.md"),
        "---\nsubject: Account\npredicate: decision\nstatus: forgotten\n---\n# Refuse overdrafts\n",
    )
    .unwrap();
    let r = sync(&ns, &mut e, 150);
    assert_eq!(r.expired, ["notes/a.md"]);
    assert!(e.list_live().unwrap().is_empty());
    // a bad note is reported, not fatal
    std::fs::write(
        d.path().join("notes/bad.md"),
        "---\nsubject: [unclosed\n---\n",
    )
    .unwrap();
    let r = sync(&ns, &mut e, 160);
    assert_eq!(r.errors.len(), 1);
}

#[test]
fn fact_to_note_to_fact_round_trip() {
    let d = tempfile::tempdir().unwrap();
    let ns = Notes::new(d.path(), "notes").unwrap();
    let mut e = MemEngine::default();
    let mut n = rce_memory::conformance::nf(
        "Account",
        "decision",
        "Refuse overdrafts",
        1_791_763_200_000,
    );
    n.text = "Body.\n".into();
    n.multi = true;
    n.provenance = Provenance::Inferred;
    n.source = Some("notes/account-decision.md".into());
    let id = e.store(&n, 5).unwrap().new_id.unwrap();
    let f = e.fetch(id).unwrap();
    ns.write("notes/account-decision.md", &render_note(&fact_note(&f)))
        .unwrap();
    let text = std::fs::read_to_string(d.path().join("notes/account-decision.md")).unwrap();
    assert!(text.starts_with("---\nsubject: Account\n"), "{text}");
    assert!(text.contains("valid_from: 2026-10-12T00:00:00.000Z"));
    let r = sync(&ns, &mut e, 10);
    assert_eq!(r.unchanged, 1, "{r:?}");
    assert_eq!(e.scan().unwrap().len(), 1);
}
