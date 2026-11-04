//! Integration tests for `frontend_attrs::extract_allows_from_source`.

use rce_graph::frontend_attrs::extract_allows_from_source;

#[test]
fn extracts_single_allows_marker_above_fn() {
    let src = "\
# @allows: regulatory:financial
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(map.len(), 1);
    assert_eq!(map["pay"], vec!["regulatory:financial"]);
}

#[test]
fn extracts_comma_separated_list() {
    let src = "\
# @allows: regulatory:financial, pii:read, crypto:secret-material
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(
        map["pay"],
        vec!["regulatory:financial", "pii:read", "crypto:secret-material"]
    );
}

#[test]
fn supports_double_slash_comments() {
    let src = "\
// @allows: regulatory:financial
fn pay() -> int { return 0; }
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["regulatory:financial"]);
}

#[test]
fn supports_pub_fn() {
    let src = "\
# @allows: pii:read
pub fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["pii:read"]);
}

#[test]
fn stacks_multiple_allows_lines() {
    let src = "\
# @allows: regulatory:financial
# @allows: pii:read
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["regulatory:financial", "pii:read"]);
}

#[test]
fn allow_marker_can_sit_above_a_doc_comment() {
    let src = "\
# @allows: regulatory:financial
# Pay an outbound transfer.
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["regulatory:financial"]);
}

#[test]
fn marker_attaches_only_to_immediately_following_fn() {
    let src = "\
# @allows: regulatory:financial

fn pay() -> int:
    return 0

fn other() -> int:
    return 1
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["regulatory:financial"]);
    assert!(!map.contains_key("other"));
}

#[test]
fn no_marker_yields_empty_map() {
    let src = "\
fn pay() -> int:
    return 0

fn other() -> int:
    return 1
";
    let map = extract_allows_from_source(src);
    assert!(map.is_empty());
}

#[test]
fn dedups_within_function() {
    let src = "\
# @allows: regulatory:financial, regulatory:financial
# @allows: regulatory:financial
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert_eq!(map["pay"], vec!["regulatory:financial"]);
}

#[test]
fn ignores_marker_followed_by_code_not_fn() {
    // A `let` line breaks the chain: marker is no longer immediately
    // above a fn, so it must NOT bind to anything.
    let src = "\
# @allows: regulatory:financial
let x = 1

fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    assert!(!map.contains_key("pay"));
}

#[test]
fn empty_allows_payload_yields_empty_list_not_panic() {
    let src = "\
# @allows:
fn pay() -> int:
    return 0
";
    let map = extract_allows_from_source(src);
    // Empty payload — no entries inserted (we filter empty strings).
    assert!(!map.contains_key("pay"));
}
