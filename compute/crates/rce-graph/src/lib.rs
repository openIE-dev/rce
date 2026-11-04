//! rce-graph: RCE's graph term, the FunctionGraph.
//!
//! A `FunctionGraph` is the numeric body of a step: nodes, wires between
//! ports, an output node and a typed result. The checker of `spec/checker`
//! builds graph terms from RCE source (`use std.tensor`, spec 6.14) and the
//! CPU backend evaluates them as the reference. This crate holds the graph
//! types, the builder, a graph interpreter and the graph stores.
//!
//! The lowering from RCE's earlier grammar is retired: RCE source reaches
//! the graph term only through `std.tensor` (compute/INTEGRATION.md).

// Mathematical iteration code (Lentz continued fractions, eigenvalue
// solvers, PCA decomposition) routinely initializes `let mut f = 1.0;`
// before the first loop iteration overwrites it. Those placeholder
// initializers are idiomatic, not bugs.
#![allow(dead_code, unused_assignments, unused_variables, unused_mut)]
pub mod builder;
pub mod capability_rewrite;
pub mod effect_check;
pub mod frontend_attrs;
pub mod graph;
pub mod interpret;
pub mod jouledb_store;
pub mod math_runtime;
pub mod dataframe;
pub mod distributions;
pub mod interpolation;
pub mod ode;
pub mod optimize;
pub mod string_ops;
pub mod ml_models;
pub mod advanced_stats;
pub mod distributions_extended;
pub mod special_functions;
pub mod ml_extended;
pub mod dataframe_extended;
pub mod signal_extended;
pub mod linalg_extended;
pub mod polynomials;
pub mod coordinates;
pub mod graph_algorithms;
pub mod combinatorics;
pub mod hypothesis;
pub mod image_processing;
pub mod sparse;
pub mod timeseries;
pub mod financial;
pub mod geometry;
pub mod random_sampling;
pub mod sets;
pub mod clustering;
pub mod spatial;
pub mod text_analysis;
pub mod numerical;
pub mod neural;
pub mod metaheuristic;
pub mod audio;
pub mod tensor_ops;
pub mod attractors;
pub mod axis_bridge;
pub mod bioinformatics;
pub mod feature_eng;
pub mod filter_design;
pub mod physics;
pub mod control_systems;
pub mod info_theory;
pub mod probability;
pub mod crypto_primitives;
pub mod units;
pub mod linalg_decomp;
pub mod compression;
pub mod calendar;
pub mod error_correction;
pub mod autodiff;
pub mod cross_validation;
pub mod string_algorithms;
pub mod geospatial;
pub mod econometrics;
pub mod network_metrics;
pub mod regression_extended;
pub mod special_extended;
pub mod pde;
pub mod matrix_io;
pub mod ml_kernels;
pub mod image_advanced;
pub mod type_conversions;
pub mod portfolio;
pub mod symbolic;
pub mod noise;
pub mod color;
pub mod queue_stack;
pub mod wavelets;
pub mod survival;
pub mod sampling_design;
pub mod bayesian;
pub mod dimensionality;
pub mod ensemble;
pub mod counting;
pub mod robust_stats;
pub mod smoothing;
pub mod matrix_creation;
pub mod search_sort;
pub mod array_ops;
pub mod validation;
pub mod distance_metrics;
pub mod constants;
pub mod math_utils;
pub mod store;
pub mod term;


pub use graph::{FunctionGraph, GraphNode, LiteralValue, LoopSubGraph, LoopVarDef, OpKind,
    Param, ProgramGraph, TypeDefineKind, Wire, WireKind};

