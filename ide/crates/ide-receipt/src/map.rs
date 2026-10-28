//! Map the two existing receipt formats onto [`Draft`].
//!
//! The mappers read the JSON those crates write (`mol.receipt.v1` from
//! mixture-of-limits `mol-receipt`, and mei-studio `mei-receipt`), so this
//! crate does not link either crate or their dependency trees. They copy
//! only what the source asserts: a measurement is carried over only when the
//! source names a real meter that has a tag here; anything else stays `None`
//! and the reason is kept in `detail.unmapped`.
use crate::{ActionKind, Draft, Measured, MeterSource, Verdict};
use serde_json::{Value, json};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MapError {
    #[error("not a {0} receipt: {1}")]
    Shape(&'static str, String),
}

fn f64_at(v: &Value, k: &str) -> Option<f64> {
    match v.get(k)? {
        Value::Number(n) => n.as_f64(),
        // mol `Joules` is a newtype that serializes as a bare number; accept [x] too.
        Value::Array(a) => a.first()?.as_f64(),
        _ => None,
    }
}

/// mol-core `MeasureSource` (snake_case) onto the IDE tag set.
pub fn mol_source(s: &str) -> Option<MeterSource> {
    match s {
        "rapl" => Some(MeterSource::Rapl),
        "nvml" => Some(MeterSource::Nvml),
        "powermetrics" => Some(MeterSource::Powermetrics),
        _ => None,
    }
}

/// A `mol.receipt.v1` JSON value.
pub fn from_mol(v: &Value, policy_hash: &str) -> Result<Draft, MapError> {
    let schema = v.get("schema").and_then(Value::as_str).unwrap_or("");
    if schema != "mol.receipt.v1" {
        return Err(MapError::Shape("mol", format!("schema `{schema}`")));
    }
    let tier = v.get("cascade_answered").and_then(Value::as_str);
    let mut d = Draft::new(
        if tier == Some("model") {
            ActionKind::ModelCall
        } else {
            ActionKind::Other("mol.ask".into())
        },
        tier.map(|t| format!("mol.{t}"))
            .unwrap_or_else(|| "mol.none".into()),
        policy_hash,
    );
    d.e_sim_j = f64_at(v, "estimated_j");
    let src = v
        .get("measure_source")
        .and_then(Value::as_str)
        .unwrap_or("unavailable");
    let mut unmapped = vec![];
    match (f64_at(v, "measured_j"), mol_source(src)) {
        (Some(j), Some(s)) => {
            d.e_meas = Some(Measured {
                joules: j,
                source: s,
            })
        }
        (Some(_), None) => unmapped.push(format!(
            "measured_j from `{src}` has no IDE meter tag; dropped"
        )),
        (None, _) => {}
    }
    d.verdict = match v.get("limit_fired") {
        Some(Value::Null) | None => Verdict::NotRun,
        Some(f) => Verdict::NotPerformed {
            reason: format!("limit fired: {f}"),
        },
    };
    d.detail = json!({
        "origin": "mol.receipt.v1",
        "origin_id": v.get("id"),
        "origin_timestamp": v.get("timestamp"),
        "estimate_kind": v.get("estimate_kind"),
        "energy_honesty": v.get("energy_honesty"),
        "unmapped": unmapped,
    });
    Ok(d)
}

/// A mei-studio `mei-receipt` JSON value.
pub fn from_mei(v: &Value, policy_hash: &str) -> Result<Draft, MapError> {
    let op = v
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| MapError::Shape("mei", "no `op`".into()))?;
    let j = f64_at(v, "joules").ok_or_else(|| MapError::Shape("mei", "no `joules`".into()))?;
    let kind = v
        .get("measured_vs_modelled")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let substrate = v
        .get("substrate")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let tier = v
        .get("tier")
        .map(|t| t.to_string().trim_matches('"').to_string())
        .unwrap_or_default();
    let mut d = Draft::new(
        ActionKind::Other(format!("mei.{op}")),
        format!("mei.{tier}"),
        policy_hash,
    );
    let mut unmapped = vec![];
    if kind == "measured" {
        // mei-receipt says "measured" but not by which meter; take the tag
        // from the substrate name only when it names one.
        let src = ["rapl", "nvml", "powermetrics", "jetson"]
            .iter()
            .find(|s| substrate.contains(*s));
        match src {
            Some(&"rapl") => {
                d.e_meas = Some(Measured {
                    joules: j,
                    source: MeterSource::Rapl,
                })
            }
            Some(&"nvml") => {
                d.e_meas = Some(Measured {
                    joules: j,
                    source: MeterSource::Nvml,
                })
            }
            Some(&"powermetrics") => {
                d.e_meas = Some(Measured {
                    joules: j,
                    source: MeterSource::Powermetrics,
                })
            }
            Some(_) => {
                d.e_meas = Some(Measured {
                    joules: j,
                    source: MeterSource::Jetson,
                })
            }
            None => unmapped.push(format!(
                "measured joules on substrate `{substrate}` name no meter; dropped"
            )),
        }
    } else {
        d.e_sim_j = Some(j);
    }
    d.detail = json!({
        "origin": "mei-receipt",
        "origin_id": v.get("receipt_id"),
        "origin_timestamp_ns": v.get("timestamp_ns"),
        "input_hash": v.get("input_hash"),
        "output_hash": v.get("output_hash"),
        "grant_id": v.get("grant_id"),
        "unmapped": unmapped,
    });
    Ok(d)
}
