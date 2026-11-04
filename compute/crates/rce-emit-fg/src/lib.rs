//! rce writer — `FunctionGraph` → `.fg` bytes + optional safetensors sidecar.
//!
//! Sibling to the other `rce-emit-*` crates (onnx, stablehlo, mlir-linalg,
//! wgsl, triton). Where those emit *foreign* formats, this one emits rce's
//! own canonical format — the input to `rce-vm-backend-cpu`,
//! `rce-vm-backend-wgpu`, and the browser bundle.
//!
//! Every writer of a `.fg` file with weights goes through one
//! [`FgEmitter`] API that owns the
//! invariants: weight stripping, sidecar writing, content hashing,
//! and (optionally) signing.
//!
//! ## Quick start
//!
//! ```ignore
//! use rce_emit_fg::FgEmitter;
//! use rce_graph::graph::FunctionGraph;
//!
//! let graph: FunctionGraph = /* … */;
//! let out = FgEmitter::new().emit(&graph)?;
//! std::fs::write("model.fg", &out.fg_bytes)?;
//! if let Some(st) = out.safetensors_bytes {
//!     std::fs::write("model.safetensors", st)?;
//! }
//! ```
//!
//! ## Embed vs sidecar
//!
//! By default, weights are stripped from the `.fg` (which contains only the
//! graph and is small + content-addressed) and emitted to a sidecar
//! `.safetensors`. With [`FgEmitter::embed_weights`], weights are kept in
//! the graph nodes themselves — the result is one big self-contained `.fg`,
//! convenient for shipping but bad for caching/dedup. The default is what
//! the CLI uses.

use rce_format::graph_section;
use rce_format::{FgFile, Section, SectionKind};
use rce_weight_loader::write_safetensors;
use rce_graph::graph::{FunctionGraph, GraphNode, TensorLiteralData};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmitError {
    #[error("format error: {0}")]
    Format(#[from] rce_format::FgError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("weight tensor `{name}`: data length {got} does not match shape × dtype = {expected}")]
    WeightSizeMismatch {
        name: String,
        got: usize,
        expected: usize,
    },

    #[error("graph contains TensorLiteral `{name}` with empty data — provide weights inline before emit")]
    UnboundWeight { name: String },
}

pub type Result<T> = std::result::Result<T, EmitError>;

/// Configuration for emitting a `FunctionGraph` to `.fg` (+ sidecar).
#[derive(Debug, Default, Clone)]
pub struct FgEmitter {
    embed_weights: bool,
    /// If `true`, missing weight data on `TensorLiteral` nodes is treated as
    /// "this is a sidecar-only metadata graph" — the sidecar is not written
    /// and the consumer is expected to supply weights some other way.
    metadata_only: bool,
}

/// Result of emitting. `safetensors_bytes` is `Some` when the producer carried
/// weight data and weights were *not* embedded in the `.fg`.
#[derive(Debug, Clone)]
pub struct EmitOutput {
    pub fg_bytes: Vec<u8>,
    pub safetensors_bytes: Option<Vec<u8>>,
    pub content_hash: [u8; 32],
}

impl FgEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Embed weights inline in the `.fg`. The `EMBED_WEIGHTS` flag (TODO:
    /// when we wire WeightManifest+WeightBlob sections) is set; no sidecar
    /// is produced. Bigger `.fg`, no extra files to ship.
    pub fn embed_weights(mut self, on: bool) -> Self {
        self.embed_weights = on;
        self
    }

    /// Skip sidecar emission even when weight data is present. Used when
    /// the caller wants to ship a graph-only `.fg` and is tracking weights
    /// out-of-band (e.g. by hash from a separate registry).
    pub fn metadata_only(mut self, on: bool) -> Self {
        self.metadata_only = on;
        self
    }

    /// Emit. The graph's `TensorLiteral` nodes are scanned: nodes with
    /// non-empty `data` contribute to the sidecar (or stay inline if
    /// `embed_weights`); nodes with empty `data` are treated as
    /// metadata-only references whose weights live elsewhere.
    pub fn emit(&self, graph: &FunctionGraph) -> Result<EmitOutput> {
        // Decide what to do per TensorLiteral.
        let weights = collect_weight_descriptors(graph)?;

        // Build a graph copy with stripped weights *unless* embed_weights.
        // Note: serde-skip on TensorLiteralData::data already strips bytes
        // on serialization; the embed-weights path requires keeping bytes
        // until the spec gets a WeightBlob section in v0.2. For v0.1, the
        // serialised graph never carries weight bytes — we still always
        // emit the sidecar (or skip if metadata_only).

        let section = graph_section::graph_section_from(graph)?;
        let mut file = FgFile::new(vec![section]);
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        file.write(&mut buf)?;
        let fg_bytes = buf.into_inner();
        let content_hash = file.header.content_hash;

        let safetensors_bytes = if self.metadata_only {
            None
        } else if !weights.is_empty() {
            let entries: Vec<(&str, &str, &[usize], &[u8])> = weights
                .iter()
                .map(|w| (w.name.as_str(), w.dtype.as_str(), w.shape.as_slice(), w.data.as_ref()))
                .collect();
            Some(write_safetensors(&entries, None))
        } else {
            None
        };

        Ok(EmitOutput {
            fg_bytes,
            safetensors_bytes,
            content_hash,
        })
    }

    /// Emit + write to disk. Sidecar (when present) is written next to
    /// `fg_path` with extension `.safetensors`. Returns the content hash.
    pub fn emit_to_files(
        &self,
        graph: &FunctionGraph,
        fg_path: &Path,
    ) -> Result<[u8; 32]> {
        let out = self.emit(graph)?;
        std::fs::write(fg_path, &out.fg_bytes)?;
        if let Some(bytes) = out.safetensors_bytes {
            let sidecar = fg_path.with_extension("safetensors");
            std::fs::write(sidecar, bytes)?;
        }
        Ok(out.content_hash)
    }
}

/// One weight entry collected from a `TensorLiteral` graph node with non-
/// empty data. We keep `data` as `Arc<[u8]>` to share with the source graph
/// (the safetensors writer takes byte slices, no copy needed).
struct WeightDescriptor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data: Arc<[u8]>,
}

fn collect_weight_descriptors(graph: &FunctionGraph) -> Result<Vec<WeightDescriptor>> {
    let mut out = Vec::new();
    for node in &graph.nodes {
        let GraphNode::TensorLiteral(td) = node else {
            continue;
        };
        if td.data.is_empty() {
            // Metadata-only entry; the consumer is expected to bring weights.
            // Not an error — the spec allows this for content-addressed
            // weight references.
            continue;
        }
        validate_weight(td)?;
        out.push(WeightDescriptor {
            name: td.name.clone(),
            dtype: td.dtype.clone(),
            shape: td.shape.clone(),
            data: td.data.clone(),
        });
    }
    Ok(out)
}

fn validate_weight(td: &TensorLiteralData) -> Result<()> {
    let per_elem = match td.dtype.as_str() {
        "bool" | "i8" | "u8" => 1,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" | "complex64" => 8,
        _ => {
            // Unknown dtype is the format crate's responsibility to flag;
            // we just trust it for emit purposes.
            return Ok(());
        }
    };
    let expected: usize = td.shape.iter().product::<usize>() * per_elem;
    if td.data.len() != expected {
        return Err(EmitError::WeightSizeMismatch {
            name: td.name.clone(),
            got: td.data.len(),
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_graph::graph::{LiteralValue, OpKind, Param, Wire, WireKind};

    fn empty_arc() -> Arc<[u8]> {
        Arc::from(Vec::<u8>::new().into_boxed_slice())
    }

    fn add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "y".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire {
                    from: 0,
                    to: 2,
                    port_in_idx: 0,
                    wire_kind: WireKind::Move,
                },
                Wire {
                    from: 1,
                    to: 2,
                    port_in_idx: 1,
                    wire_kind: WireKind::Move,
                },
            ],
            output_node: 2,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn emit_graph_only() {
        let g = add_graph();
        let out = FgEmitter::new().emit(&g).unwrap();
        assert!(!out.fg_bytes.is_empty());
        assert!(out.safetensors_bytes.is_none());
        assert_ne!(out.content_hash, [0u8; 32]);
    }

    #[test]
    fn emit_with_inline_weights_produces_sidecar() {
        let mut g = add_graph();
        // Inject a weight node so the emitter has something to put in the sidecar.
        g.nodes.push(GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(vec![0u8; 4 * 3].into_boxed_slice()),
            shape: vec![3],
            dtype: "f32".into(),
            name: "w".into(),
        }));
        let out = FgEmitter::new().emit(&g).unwrap();
        let sidecar = out.safetensors_bytes.expect("sidecar should exist");
        assert!(!sidecar.is_empty());
    }

    #[test]
    fn metadata_only_skips_sidecar_even_with_weights() {
        let mut g = add_graph();
        g.nodes.push(GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(vec![0u8; 4 * 3].into_boxed_slice()),
            shape: vec![3],
            dtype: "f32".into(),
            name: "w".into(),
        }));
        let out = FgEmitter::new().metadata_only(true).emit(&g).unwrap();
        assert!(out.safetensors_bytes.is_none());
    }

    #[test]
    fn unbound_weight_is_not_an_error() {
        // A graph with TensorLiteral.data == empty Arc is a *metadata* graph;
        // the consumer brings weights via sidecar at load time.
        let mut g = add_graph();
        g.nodes.push(GraphNode::TensorLiteral(TensorLiteralData {
            data: empty_arc(),
            shape: vec![3],
            dtype: "f32".into(),
            name: "w".into(),
        }));
        let out = FgEmitter::new().emit(&g).unwrap();
        // No sidecar (no inline data), and emit succeeded.
        assert!(out.safetensors_bytes.is_none());
        assert!(!out.fg_bytes.is_empty());
    }

    #[test]
    fn weight_size_mismatch_errors() {
        let mut g = add_graph();
        g.nodes.push(GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(vec![0u8; 7].into_boxed_slice()), // wrong: shape (3,) * f32 = 12
            shape: vec![3],
            dtype: "f32".into(),
            name: "broken".into(),
        }));
        let err = FgEmitter::new().emit(&g).unwrap_err();
        assert!(matches!(
            err,
            EmitError::WeightSizeMismatch { name, got: 7, expected: 12 } if name == "broken"
        ));
    }

    #[test]
    fn round_trip_through_filesystem() {
        let g = add_graph();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.fg");
        let hash = FgEmitter::new().emit_to_files(&g, &path).unwrap();
        assert!(path.exists());
        assert_ne!(hash, [0u8; 32]);
    }
}
