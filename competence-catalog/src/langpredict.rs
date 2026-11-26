//! Language prediction — capability-gap analysis for the AI / hybrid-fabric era.
//!
//! The constellation measures what existing languages realize. This module asks the inverse
//! question: **what capability is still missing**, and what language would realize it natively?
//! For each compute-fabric class it names the native constructs the current universal taxonomy
//! (113) does NOT yet express, and emits a *predicted language* spec whose primitives ARE the
//! fabric's primitives — eliminating the impedance tax between OS, hardware and language.
//!
//! Honesty contract: this is ANALYSIS (a gap + a spec), not a synthesized compiler. The native
//! constructs are deliberately kept OUT of `structure::CONSTRUCTS` so the measured 113×N matrix
//! stays truthful; extending the taxonomy and synthesizing these languages is the registered
//! advanced step, sequenced AFTER full language ingestion and RCE-JEPA headway.

use serde::Serialize;

use crate::structure::CONSTRUCTS;

/// A compute-fabric class and the native constructs a language must express to target it
/// without an impedance tax. These are capability requirements, not yet taxonomy members.
pub struct FabricClass {
    pub id: &'static str,
    pub label: &'static str,
    pub native_constructs: &'static [&'static str],
    pub predicted_name: &'static str,
    pub paradigm: &'static str,
}

pub const FABRICS: &[FabricClass] = &[
    FabricClass {
        id: "quantum", label: "Quantum (gate-model / superconducting / ion)",
        native_constructs: &["qubit", "superpose", "entangle", "measure", "phase", "gate", "teleport", "no_clone"],
        predicted_name: "Superpose", paradigm: "quantum-native, reversible-linear",
    },
    FabricClass {
        id: "thermodynamic", label: "Thermodynamic / reversible / Landauer-bounded",
        native_constructs: &["reverse", "dissipate", "landauer", "entropy_budget", "heat", "reversible_gate", "energy_type"],
        predicted_name: "Landauer", paradigm: "thermodynamic-native, energy-typed",
    },
    FabricClass {
        id: "spatial", label: "Spatial / dataflow (CGRA / FPGA / wafer-scale)",
        native_constructs: &["place", "route", "pipeline", "spatial_loop", "reconfig", "bitstream"],
        predicted_name: "Tessera", paradigm: "spatial-native, place-and-route",
    },
    FabricClass {
        id: "neuromorphic", label: "Neuromorphic / spiking (Loihi / SpiNNaker / Akida)",
        native_constructs: &["spike", "synapse", "plasticity", "event_time", "membrane"],
        predicted_name: "Spike", paradigm: "event-native, spiking",
    },
    FabricClass {
        id: "probabilistic", label: "Probabilistic / stochastic (p-bit / sampling)",
        native_constructs: &["sample", "distribution", "bernoulli", "infer", "pbit"],
        predicted_name: "Alea", paradigm: "probabilistic-native, sampling",
    },
    FabricClass {
        id: "analog", label: "Analog / in-memory (crossbar / MAC / conductance)",
        native_constructs: &["conductance", "mac", "crossbar", "drift", "analog_read"],
        predicted_name: "Ohm", paradigm: "analog-native, in-memory",
    },
];

/// A predicted language: the missing capability made into a native surface.
#[derive(Debug, Clone, Serialize)]
pub struct PredictedLanguage {
    pub fabric: String,
    pub name: String,
    pub paradigm: String,
    /// Native constructs absent from the current taxonomy (the measured gap).
    pub missing_constructs: Vec<String>,
    /// Why this eliminates the impedance tax.
    pub rationale: String,
}

/// The capability gap for a fabric: native constructs NOT in the current universal taxonomy.
pub fn gap(f: &FabricClass) -> Vec<&'static str> {
    f.native_constructs.iter().copied().filter(|c| !CONSTRUCTS.contains(c)).collect()
}

/// Predict the language that natively realizes a fabric's missing capability.
pub fn predict(f: &FabricClass) -> PredictedLanguage {
    let missing: Vec<String> = gap(f).iter().map(|s| s.to_string()).collect();
    PredictedLanguage {
        fabric: f.id.to_string(),
        name: f.predicted_name.to_string(),
        paradigm: f.paradigm.to_string(),
        missing_constructs: missing,
        rationale: format!(
            "{}'s primitives are the {} fabric's own operations, so a program maps 1:1 to the \
             hardware with no OS/ABI translation layer — the impedance tax between OS, hardware \
             and language goes to zero. The Law of Coding applies: once the native competence is \
             admitted, each act costs the lookup floor, not a re-proposal across an abstraction gap.",
            f.predicted_name, f.label
        ),
    }
}

/// All predicted languages (the capability-gap frontier for the AI / hybrid-fabric era).
pub fn predict_all() -> Vec<PredictedLanguage> {
    FABRICS.iter().map(predict).collect()
}
