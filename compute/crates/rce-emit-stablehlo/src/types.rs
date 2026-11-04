//! graph type strings → MLIR/StableHLO type strings.
//!
//! StableHLO uses MLIR tensor types: `tensor<f32>`, `tensor<4x3xf32>`,
//! `tensor<i64>`, etc. Scalars are rank-0 tensors: `tensor<f32>`.

/// Convert a graph type string to an MLIR tensor element type.
pub fn mlir_element_type(rce: &str) -> Option<&'static str> {
    Some(match rce {
        "f32" | "float" => "f32",
        "f64" | "double" => "f64",
        "f16" => "f16",
        "bf16" => "bf16",
        "i8" => "i8",
        "i16" => "i16",
        "i32" => "i32",
        "i64" | "int" => "i64",
        "u8" => "ui8",
        "u16" => "ui16",
        "u32" => "ui32",
        "u64" => "ui64",
        "bool" => "i1",
        _ => return None,
    })
}

/// Wrap an element type as a rank-0 (scalar) tensor type.
pub fn mlir_scalar_tensor(rce: &str) -> Option<String> {
    mlir_element_type(rce).map(|et| format!("tensor<{et}>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_maps() {
        assert_eq!(mlir_element_type("f32"), Some("f32"));
        assert_eq!(mlir_scalar_tensor("f32"), Some("tensor<f32>".into()));
    }

    #[test]
    fn bool_maps_to_i1() {
        assert_eq!(mlir_element_type("bool"), Some("i1"));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(mlir_element_type("tensor<?>"), None);
    }
}
