use rce_mcp::items::{Edit, apply, items};
use rce_mcp::ws::Workspace;
use std::fs;

const COUNTER: &str =
    include_str!("../../../vendor/rce-spec/examples/01_counter.rce");

#[test]
fn splits_top_level_items() {
    let names: Vec<String> = items(COUNTER).into_iter().map(|i| i.name).collect();
    assert_eq!(
        names,
        [
            "intent \"a counter that never goes negative\"",
            "state Counter",
            "event Inc",
            "event Dec",
            "error Underflow",
            "step Counter on Inc",
            "step Counter on Dec",
        ]
    );
}

#[test]
fn edits_by_name() {
    let s = apply(
        COUNTER,
        &[Edit::InsertItem {
            after: Some("event Dec".into()),
            text: "event Reset".into(),
        }],
    )
    .unwrap();
    assert!(s.contains("event Dec\n\nevent Reset\n"));
    let s = apply(
        &s,
        &[Edit::DeleteItem {
            name: "event Reset".into(),
        }],
    )
    .unwrap();
    assert_eq!(s, COUNTER);
    let s = apply(
        COUNTER,
        &[Edit::ReplaceItem {
            name: "step Counter on Inc".into(),
            text: "step Counter on Inc -> n + 2".into(),
        }],
    )
    .unwrap();
    assert!(s.contains("on Inc -> n + 2"));
    assert!(
        apply(
            COUNTER,
            &[Edit::DeleteItem {
                name: "event Nope".into()
            }]
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn path_policy() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("ws");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join(".git/hooks")).unwrap();
    fs::write(root.join("src/a.rce"), COUNTER).unwrap();
    fs::write(t.path().join("outside.rce"), COUNTER).unwrap();
    std::os::unix::fs::symlink(t.path(), root.join("link")).unwrap();
    std::os::unix::fs::symlink(t.path().join("outside.rce"), root.join("out.rce")).unwrap();
    let ws = Workspace::new(&root).unwrap();
    assert!(ws.resolve("src/a.rce", false).is_ok());
    assert!(ws.resolve("src/a.rce", true).is_ok());
    assert!(ws.resolve("src/new.rce", true).is_ok());
    for bad in [
        "../outside.rce",
        "/etc/passwd",
        "link/outside.rce",
        "out.rce",
        "src/../../outside.rce",
    ] {
        assert!(ws.resolve(bad, false).is_err(), "{bad}");
        assert!(ws.resolve(bad, true).is_err(), "{bad}");
    }
    for bad in [
        ".git/hooks/pre-commit",
        ".vscode/settings.rce",
        "Cargo.toml",
        "build.rs",
        "src/notes.txt",
        "mcp.json",
    ] {
        assert!(ws.resolve(bad, true).is_err(), "{bad}");
    }
    // The program walk skips symlinks and dot dirs.
    let prog: Vec<String> = ws.program().into_iter().map(|p| p.0).collect();
    assert_eq!(prog, ["src/a.rce"]);
}
