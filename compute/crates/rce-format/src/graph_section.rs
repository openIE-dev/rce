//! A graph term in a `.fg` file: read/write a `rce_graph::graph::FunctionGraph` (or
//! `ProgramGraph`) directly via canonical CBOR encoding into a Graph section.

use crate::error::Result;
use crate::section::{decode_cbor, encode_cbor, Section, SectionKind};

pub fn graph_section_from<T: serde::Serialize>(graph: &T) -> Result<Section> {
    Ok(Section {
        kind: SectionKind::Graph,
        payload: encode_cbor(graph)?,
    })
}

pub fn graph_from_section<T: serde::de::DeserializeOwned>(section: &Section) -> Result<T> {
    decode_cbor(&section.payload)
}

/// The `.fg` file of a graph term (spec 6.13, GT-2): its `Graph` section
/// holds the graph and its `TypeTable` section the term's typing (the
/// shapes, element types and levels of `rce_graph::term::TermTypes`).
pub fn term_file<G: serde::Serialize, T: serde::Serialize>(graph: &G, types: &T) -> Result<crate::FgFile> {
    Ok(crate::FgFile::new(vec![
        graph_section_from(graph)?,
        Section {
            kind: SectionKind::TypeTable,
            payload: encode_cbor(types)?,
        },
    ]))
}

/// The content hash of a graph term's `.fg` file: the BLAKE3-256 of each
/// section's kind, length and bytes, exactly as [`crate::FgFile::write`]
/// takes it. It is the term's key in RCE's core (GT-2): the checker writes
/// it, and a term whose key is not this hash of its body and typing
/// requires change.
pub fn term_content_hash<G: serde::Serialize, T: serde::Serialize>(graph: &G, types: &T) -> Result<[u8; 32]> {
    let mut file = term_file(graph, types)?;
    let mut buf = std::io::Cursor::new(Vec::new());
    file.write(&mut buf)?;
    Ok(file.header.content_hash)
}
