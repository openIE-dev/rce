//! graph type strings -> Triton `tl.*` type strings.
//!
//! Triton's type system maps directly to GPU register types:
//! `tl.float32`, `tl.float64`, `tl.int32`, `tl.int64`, `tl.int1` (bool).

/// Convert a graph type string to a Triton `tl.*` type name.
pub fn triton_type(rce: &str) -> Option<&'static str> {
    Some(match rce {
        "f16" => "tl.float16",
        "bf16" => "tl.bfloat16",
        "f32" | "float" => "tl.float32",
        "f64" | "double" => "tl.float64",
        "i8" => "tl.int8",
        "i16" => "tl.int16",
        "i32" | "int" => "tl.int32",
        "i64" => "tl.int64",
        "u8" => "tl.uint8",
        "u16" => "tl.uint16",
        "u32" => "tl.uint32",
        "u64" => "tl.uint64",
        "bool" => "tl.int1",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_maps() {
        assert_eq!(triton_type("f32"), Some("tl.float32"));
    }

    #[test]
    fn float_alias_maps() {
        assert_eq!(triton_type("float"), Some("tl.float32"));
    }

    #[test]
    fn f64_maps() {
        assert_eq!(triton_type("f64"), Some("tl.float64"));
    }

    #[test]
    fn i32_maps() {
        assert_eq!(triton_type("i32"), Some("tl.int32"));
    }

    #[test]
    fn i64_maps() {
        assert_eq!(triton_type("i64"), Some("tl.int64"));
    }

    #[test]
    fn bool_maps_to_int1() {
        assert_eq!(triton_type("bool"), Some("tl.int1"));
    }

    #[test]
    fn bf16_maps() {
        assert_eq!(triton_type("bf16"), Some("tl.bfloat16"));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(triton_type("tensor<?>"), None);
    }
}
