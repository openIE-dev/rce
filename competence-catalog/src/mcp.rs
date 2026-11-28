//! MCP tool surface (Stately/JetBrains-Junie parity): expose the admitted-IR understanding as
//! deterministic MCP tools an agent can call. Every tool is a pure function of the Form / LUT —
//! no model call, no guess. tools/list mirrors adapter_gen::mcp_tools plus understanding tools.

use serde_json::{json, Value};

use crate::program::{lift_program, replay_admit, generative_probes, Env};

/// tools/list descriptor.
pub fn tools_list() -> Value {
    json!({ "tools": [
        { "name": "admit", "description": "Replay-gate a candidate program; returns admitted + content key." },
        { "name": "xref", "description": "Cross-references (calls/defs/uses) from the admitted Form." },
        { "name": "call_graph", "description": "Call graph {nodes,edges} from the Form." },
        { "name": "statechart", "description": "Deterministic behavior statechart {states,transitions} from the Form." },
        { "name": "ui_spec", "description": "Constrained declarative UI (allow-listed catalog) from the Form." },
        { "name": "property_probes", "description": "Generative property battery over the program's free inputs." },
        { "name": "sql_schema", "description": "DB-as-competence: lift CREATE TABLE into {tables,columns,types}." },
        { "name": "sql_query", "description": "Lift SELECT into {projection,source,filter}." },
        { "name": "resolve", "description": "Ask-the-map: name-free behavior key of a program over its generative probe battery; equal behaviour -> equal key." }
    ] })
}

/// tools/call dispatcher. args: {src, probes?}. Deterministic; errors are returned, never guessed.
pub fn call(tool: &str, args: &Value) -> Result<Value, String> {
    let src = args["src"].as_str().ok_or("args.src required")?;
    match tool {
        "sql_schema" => return Ok(crate::sql::schema(src)),
        "sql_query" => return Ok(crate::sql::query(src)),
        _ => {}
    }
    let stmt = lift_program(src).or_else(|_| crate::lift_rust::lift_rust(src))?;
    match tool {
        "admit" => {
            let probes: Vec<(Env, i128)> = parse_probes(&args["probes"]);
            let ok = if probes.is_empty() { true } else { replay_admit(&stmt, &probes) };
            Ok(json!({ "admitted": ok, "key": crate::program::program_key(&stmt).ok() }))
        }
        "xref" => Ok(serde_json::to_value(crate::xref::xref(&stmt)).map_err(|e| e.to_string())?),
        "call_graph" => Ok(crate::xref::call_graph(&stmt)),
        "statechart" => Ok(crate::behavior::statechart(&stmt)),
        "ui_spec" => Ok(serde_json::from_str(&crate::genui::ui_spec(src)?).map_err(|e| e.to_string())?),
        "property_probes" => {
            let p = generative_probes(&stmt);
            Ok(json!({ "count": p.len() }))
        }
        "resolve" => {
            // name-free behavior key: outputs over the generative battery, order-independent.
            let p = generative_probes(&stmt);
            let mut outs: Vec<i128> = p.iter().map(|(_, r)| *r).collect();
            outs.sort_unstable();
            let key = rcevc_core::sha256(format!("{outs:?}").as_bytes()).hex();
            Ok(json!({ "key": key, "probes": p.len() }))
        }
        _ => Err(format!("unknown tool {tool}")),
    }
}

fn parse_probes(v: &Value) -> Vec<(Env, i128)> {
    let mut out = Vec::new();
    if let Some(arr) = v.as_array() {
        for p in arr {
            let mut env = Env::new();
            if let Some(o) = p["env"].as_object() {
                for (k, val) in o { if let Some(i) = val.as_i64() { env.insert(k.clone(), i as i128); } }
            }
            if let Some(r) = p["result"].as_i64() { out.push((env, r as i128)); }
        }
    }
    out
}
