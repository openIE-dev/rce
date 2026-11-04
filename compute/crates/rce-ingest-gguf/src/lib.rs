//! rce-ingest-gguf — parse GGUF model files into RCE graphs.
//!
//! GGUF is the binary format used by llama.cpp, whisper.cpp, and the broader
//! GGML ecosystem for quantized LLM weights.  This crate parses GGUF v3
//! headers, metadata key-value pairs, and tensor layout information, then
//! converts the result into `rce_graph::ProgramGraph` representations that
//! capture the model architecture as RCE compute graphs.
//!
//! Only metadata and tensor descriptors are read — actual weight blobs are
//! never loaded into memory.
//!
//! # Usage
//!
//! ```ignore
//! use rce_ingest_gguf::{GgufFile, to_program_graph};
//!
//! let data = std::fs::read("model.gguf")?;
//! let gguf = GgufFile::parse(&data)?;
//! let program = to_program_graph(&gguf);
//! ```

pub mod dequant;
pub mod mmap;
pub mod parser;
pub mod quant;
pub mod sampling;
pub mod to_rce;
pub mod tokenizer;

pub use mmap::{advise_post_load, mmap_load_sequential, LayerAdvice, PostLoadAdvice};

pub use dequant::{
    dequantize_to_f32, read_tensor_f32, tensor_byte_size, ReadError, STQ1_0_CODEBOOK,
};
pub use parser::{GgufDtype, GgufFile, GgufMetadata, GgufTensorInfo, GgufValue, ParseError};
pub use quant::{QuantInfo, quant_info};
pub use sampling::{sample, sample_with_history, SampleConfig};
pub use to_rce::to_program_graph;
pub use tokenizer::{TokenType, Tokenizer};
