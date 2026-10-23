//! The editor's rope buffer: cursor, selection, undo/redo, IME, scrolling.
use rce_ide::buffer::{Buffer, EditOp as E, Motion as M};

fn mv(to: M) -> E {
    E::Move { to, select: false }
}
fn sel(to: M) -> E {
    E::Move { to, select: true }
}
fn typed(b: &mut Buffer, s: &str) {
    for c in s.chars() {
        b.apply(if c == '\n' {
            E::Newline
        } else {
            E::Insert(c.into())
        });
    }
}

#[test]
fn typing_groups_into_words_for_undo_and_redo() {
    let mut b = Buffer::new("");
    typed(&mut b, "event Inc");
    assert_eq!(b.text(), "event Inc");
    b.apply(E::Undo);
    assert_eq!(b.text(), "event ");
    b.apply(E::Undo);
    assert_eq!(b.text(), "");
    assert!(!b.can_undo());
    b.apply(E::Redo);
    b.apply(E::Redo);
    assert_eq!(b.text(), "event Inc");
    assert_eq!(b.cursor, 9);
    // A new edit clears redo.
    b.apply(E::Undo);
    typed(&mut b, "Dec");
    assert!(!b.can_redo());
    assert_eq!(b.text(), "event Dec");
    // Backspaces group too.
    for _ in 0..3 {
        b.apply(E::Backspace);
    }
    assert_eq!(b.text(), "event ");
    b.apply(E::Undo);
    assert_eq!(b.text(), "event Dec");
}

#[test]
fn shift_motion_selects_and_typing_replaces() {
    let mut b = Buffer::new("step Counter on Inc -> n + 1");
    b.apply(mv(M::WordRight));
    b.apply(mv(M::Right));
    b.apply(sel(M::WordRight));
    assert_eq!(b.selected_text().as_deref(), Some("Counter"));
    b.apply(E::Insert("Clock".into()));
    assert_eq!(b.text(), "step Clock on Inc -> n + 1");
    b.apply(E::Undo);
    assert_eq!(b.text(), "step Counter on Inc -> n + 1");
    assert_eq!(b.selected_text().as_deref(), Some("Counter"));
    // Arrow without shift collapses to the selection's edge.
    b.apply(mv(M::Left));
    assert_eq!(b.cursor, 5);
    assert_eq!(b.selection(), None);
    b.apply(E::SelectAll);
    b.apply(E::Delete);
    assert!(b.is_empty());
}

#[test]
fn vertical_motion_keeps_the_goal_column() {
    let mut b = Buffer::new("long line here\nab\nanother long line");
    b.apply(mv(M::LineEnd)); // col 14
    b.apply(mv(M::Down));
    assert_eq!(b.line_col(b.cursor), (1, 2)); // clamped
    b.apply(mv(M::Down));
    assert_eq!(b.line_col(b.cursor), (2, 14)); // goal restored
    b.apply(mv(M::DocStart));
    b.apply(sel(M::Down));
    b.apply(sel(M::Down));
    let v = b.visible();
    // Two whole lines (each shows its line break as one cell); the caret
    // sits at the start of the third.
    assert_eq!(v.selection, vec![(0, 0, 15), (1, 0, 3)]);
}

#[test]
fn newline_keeps_indentation_and_smart_home() {
    let mut b = Buffer::new("state Counter\n    n: u64 = 0");
    b.apply(mv(M::DocEnd));
    b.apply(E::Newline);
    b.apply(E::Insert("m: u64 = 1".into()));
    assert_eq!(b.line(2), "    m: u64 = 1");
    b.apply(mv(M::LineStart));
    assert_eq!(b.line_col(b.cursor), (2, 4));
    b.apply(mv(M::LineStart));
    assert_eq!(b.line_col(b.cursor), (2, 0));
    b.apply(E::Tab);
    assert_eq!(b.line(2), "        m: u64 = 1");
}

#[test]
fn cut_copy_paste() {
    let mut b = Buffer::new("event Inc\nevent Dec\n");
    b.apply(sel(M::Down));
    b.apply(E::Cut);
    assert_eq!(b.text(), "event Dec\n");
    b.apply(mv(M::DocEnd));
    b.apply(E::Paste(None));
    assert_eq!(b.text(), "event Dec\nevent Inc\n");
    b.apply(E::Paste(Some("error Underflow".into())));
    assert_eq!(b.line(2), "error Underflow");
    b.apply(mv(M::LineStart));
    b.apply(sel(M::LineEnd));
    b.apply(E::Copy);
    assert_eq!(b.clipboard, "error Underflow");
}

#[test]
fn ime_preedit_then_commit() {
    let mut b = Buffer::new("intent \"\"");
    b.apply(mv(M::LineEnd));
    b.apply(mv(M::Left));
    assert!(!b.apply(E::Preedit("かうん".into())));
    assert_eq!(b.visible().preedit.as_deref(), Some("かうん"));
    assert_eq!(b.text(), "intent \"\""); // composition is not text yet
    assert!(b.apply(E::Insert("カウンター".into())));
    assert_eq!(b.text(), "intent \"カウンター\"");
    assert_eq!(b.visible().preedit, None);
    assert_eq!(b.line_col(b.cursor), (0, 13));
    b.apply(E::Backspace);
    assert_eq!(b.text(), "intent \"カウンタ\"");
}

#[test]
fn scrolling_reveals_the_cursor_and_clicks_map_columns() {
    let text: String = (1..=100).map(|i| format!("event E{i}\n")).collect();
    let mut b = Buffer::new(&text);
    b.rows = 10;
    b.apply(mv(M::PageDown));
    assert_eq!(b.line_col(b.cursor).0, 9);
    b.apply(mv(M::PageDown));
    assert_eq!(b.line_col(b.cursor).0, 18);
    assert_eq!(b.scroll, 9);
    let v = b.visible();
    assert_eq!((v.first, v.lines.len()), (9, 10));
    assert_eq!(v.lines[0], "event E10");
    b.apply(E::Scroll(50));
    assert_eq!(b.scroll, 59);
    b.apply(E::ClickAt {
        line: 60,
        col: 6,
        extend: false,
    });
    assert_eq!(b.line_col(b.cursor), (60, 6));
    b.apply(E::ClickAt {
        line: 61,
        col: 99,
        extend: true,
    });
    assert_eq!(b.selected_text().as_deref(), Some("E61\nevent E62"));
    b.apply(mv(M::DocEnd));
    assert_eq!(b.scroll, 91);
}

#[test]
fn tabs_display_as_spaces_and_click_maps_back() {
    let mut b = Buffer::new("\tn: u64");
    let v = b.visible();
    assert_eq!(v.lines[0], "    n: u64");
    b.apply(E::ClickAt {
        line: 0,
        col: 5,
        extend: false,
    });
    assert_eq!(b.line_col(b.cursor), (0, 2));
    assert_eq!(b.visible().cursor, (0, 5));
}

#[test]
fn large_file_edits_stay_fast() {
    let text: String = (0..200_000).map(|i| format!("event E{i}\n")).collect();
    let mut b = Buffer::new(&text);
    let t = std::time::Instant::now();
    b.apply(mv(M::DocEnd));
    for _ in 0..1000 {
        b.apply(E::Insert("x".into()));
        b.apply(mv(M::PageUp));
    }
    for _ in 0..1000 {
        b.apply(E::Undo);
    }
    assert_eq!(b.len(), text.chars().count());
    assert!(t.elapsed().as_secs() < 5, "{:?}", t.elapsed());
}
