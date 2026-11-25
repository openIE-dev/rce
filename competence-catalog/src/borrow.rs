//! Type/borrow (mutability) layer — the structural gotcha in cross-language ports.
//!
//! Ownership/borrowing (Rust) constrains which ports are *legal*, not the computed value. Porting
//! Rust→Haskell/Python drops the borrow structure (value-preserving); porting into Rust must
//! *introduce* a valid ownership/borrow structure. This module models the affine/borrow rules
//! (one owner; many immutable borrows XOR one mutable borrow; a move invalidates the source) so a
//! port can be checked for structural legality, complementing the value-level `semantics::preserve`.

use std::collections::HashMap;

#[derive(Default)]
pub struct Ownership {
    moved: HashMap<String, bool>,
    imm: HashMap<String, u32>,
    mutb: HashMap<String, bool>,
}

impl Ownership {
    pub fn new() -> Self { Self::default() }
    pub fn declare(&mut self, v: &str) { self.moved.entry(v.into()).or_insert(false); }

    /// Many immutable borrows allowed unless a mutable borrow is active.
    pub fn imm_borrow(&mut self, v: &str) -> Result<(), String> {
        if *self.mutb.get(v).unwrap_or(&false) { return Err(format!("{v}: immutable borrow while mutably borrowed")); }
        if *self.moved.get(v).unwrap_or(&false) { return Err(format!("{v}: use after move")); }
        *self.imm.entry(v.into()).or_insert(0) += 1;
        Ok(())
    }

    /// Exactly one mutable borrow, exclusive of all others.
    pub fn mut_borrow(&mut self, v: &str) -> Result<(), String> {
        if *self.mutb.get(v).unwrap_or(&false) { return Err(format!("{v}: second mutable borrow")); }
        if *self.imm.get(v).unwrap_or(&0) > 0 { return Err(format!("{v}: mutable borrow while immutably borrowed")); }
        if *self.moved.get(v).unwrap_or(&false) { return Err(format!("{v}: use after move")); }
        self.mutb.insert(v.into(), true);
        Ok(())
    }

    pub fn release_mut(&mut self, v: &str) { self.mutb.insert(v.into(), false); }

    /// A move transfers ownership; the source is invalidated.
    pub fn move_val(&mut self, v: &str) -> Result<(), String> {
        if *self.mutb.get(v).unwrap_or(&false) || *self.imm.get(v).unwrap_or(&0) > 0 {
            return Err(format!("{v}: cannot move while borrowed"));
        }
        self.moved.insert(v.into(), true);
        Ok(())
    }

    pub fn use_after_move(&self, v: &str) -> bool { *self.moved.get(v).unwrap_or(&false) }
}

/// The structural direction of a port between an ownership language and a GC/immutable one.
pub fn port_structural(from_ownership: bool, to_ownership: bool) -> &'static str {
    match (from_ownership, to_ownership) {
        (true, false) => "drop borrow structure; value-preserving (GC/immutable target manages lifetime)",
        (false, true) => "introduce a valid ownership/borrow structure; check affine + borrow rules",
        (true, true) => "re-derive ownership/borrow structure for the target's rules",
        (false, false) => "no ownership structure on either side; value-level preserve suffices",
    }
}