//! graph type strings -> WGSL type strings.
//!
//! WGSL compute shaders have a limited type set:
//! - `f32` (no f64 in compute shaders)
//! - `i32`, `u32`
//! - `bool`
//!
//! Types that exceed WGSL precision (f64, i64) are demoted with a comment.

/// Convert a graph type string to a WGSL type string.
///
/// Returns `None` for types with no WGSL representation (strings, etc.).
pub fn wgsl_type(rce: &str) -> Option<&'static str> {
    Some(match rce {
        "f32" | "float" => "f32",
        // WGSL has no f64 in compute; demote to f32.
        "f64" | "double" => "f32",
        "f16" => "f16",
        "i32" | "int" => "i32",
        // WGSL has no i64; demote to i32.
        "i64" => "i32",
        "u8" | "u16" | "u32" => "u32",
        "u64" => "u32",
        "bool" => "bool",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_maps() {
        assert_eq!(wgsl_type("f32"), Some("f32"));
    }

    #[test]
    fn f64_demotes_to_f32() {
        assert_eq!(wgsl_type("f64"), Some("f32"));
    }

    #[test]
    fn i32_maps() {
        assert_eq!(wgsl_type("i32"), Some("i32"));
    }

    #[test]
    fn i64_demotes_to_i32() {
        assert_eq!(wgsl_type("i64"), Some("i32"));
    }

    #[test]
    fn u8_promotes_to_u32() {
        assert_eq!(wgsl_type("u8"), Some("u32"));
    }

    #[test]
    fn bool_maps() {
        assert_eq!(wgsl_type("bool"), Some("bool"));
    }

    #[test]
    fn string_returns_none() {
        assert_eq!(wgsl_type("String"), None);
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(wgsl_type("tensor<?>"), None);
    }
}
