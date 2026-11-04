//! graph type strings -> MLIR type strings.
//!
//! Same element-type mapping as StableHLO (both use MLIR's type system),
//! plus helpers to distinguish integer vs float types -- needed because
//! the linalg/arith dialect requires choosing `arith.addi` vs `arith.addf`.

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

/// Returns true if the graph type string denotes a floating-point type.
pub fn is_float_type(rce: &str) -> bool {
    matches!(rce, "f16" | "bf16" | "f32" | "float" | "f64" | "double")
}

/// Returns true if the graph type string denotes an integer type.
pub fn is_int_type(rce: &str) -> bool {
    matches!(
        rce,
        "i8" | "i16" | "i32" | "i64" | "int" | "u8" | "u16" | "u32" | "u64"
    )
}

/// Returns true if the graph type string denotes a boolean type.
pub fn is_bool_type(rce: &str) -> bool {
    rce == "bool"
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

    #[test]
    fn float_detection() {
        assert!(is_float_type("f32"));
        assert!(is_float_type("f64"));
        assert!(is_float_type("f16"));
        assert!(is_float_type("bf16"));
        assert!(is_float_type("float"));
        assert!(is_float_type("double"));
        assert!(!is_float_type("i32"));
        assert!(!is_float_type("bool"));
    }

    #[test]
    fn int_detection() {
        assert!(is_int_type("i32"));
        assert!(is_int_type("i64"));
        assert!(is_int_type("int"));
        assert!(is_int_type("u8"));
        assert!(!is_int_type("f32"));
        assert!(!is_int_type("bool"));
    }

    #[test]
    fn bool_detection() {
        assert!(is_bool_type("bool"));
        assert!(!is_bool_type("i1"));
        assert!(!is_bool_type("i32"));
    }
}
