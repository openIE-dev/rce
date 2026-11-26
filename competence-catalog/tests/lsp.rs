use competence_catalog::lsp_server::{handle, LspState};
use serde_json::json;

#[test]
fn lsp_edit_time_gate_reports_errors_before_runtime() {
    let mut st = LspState::new();
    let init = handle(&mut st, &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}));
    assert!(init[0]["result"]["capabilities"]["textDocumentSync"].is_number());
    // register probes: empty env -> 2
    handle(&mut st, &json!({"jsonrpc":"2.0","id":2,"method":"rce/setProbes","params":{"uri":"file:///a.rce","probes":[[{},2]]}}));
    // bad edit: returns 3, probes expect 2 -> diagnostic AT EDIT TIME
    let bad = handle(&mut st, &json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///a.rce"},"contentChanges":[{"text":"return 3;"}]}}));
    assert_eq!(bad[0]["method"], "textDocument/publishDiagnostics");
    assert!(bad[0]["params"]["diagnostics"][0]["message"].as_str().unwrap().contains("replay divergence"));
    // good edit -> clear diagnostics
    let good = handle(&mut st, &json!({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///a.rce"},"contentChanges":[{"text":"let x = 1; return x + 1;"}]}}));
    assert_eq!(good[0]["params"]["diagnostics"].as_array().unwrap().len(), 0);
    // admit -> served, no escaped errors
    let adm = handle(&mut st, &json!({"jsonrpc":"2.0","id":3,"method":"rce/admit","params":{"uri":"file:///a.rce"}}));
    assert_eq!(adm[0]["result"]["served"], true);
    assert_eq!(st.loop_.stats.escaped_errors, 0);
    assert_eq!(st.loop_.stats.debug_sessions, 0);
    // completion serves admitted competence surfaces
    let comp = handle(&mut st, &json!({"jsonrpc":"2.0","id":4,"method":"textDocument/completion","params":{"textDocument":{"uri":"file:///a.rce"}}}));
    assert!(comp[0]["result"]["items"].as_array().unwrap().len() > 20);
}

#[test]
fn lsp_parse_error_is_edit_time_diagnostic() {
    let mut st = LspState::new();
    let r = handle(&mut st, &json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///b.rce","text":"let = ;"}}}));
    assert_eq!(r[0]["method"], "textDocument/publishDiagnostics");
    assert!(r[0]["params"]["diagnostics"][0]["message"].as_str().unwrap().starts_with("parse:"));
}
