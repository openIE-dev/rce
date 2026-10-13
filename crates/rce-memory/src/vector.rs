//! Optional vector gear. No embedding model is bundled: an implementation
//! answers from embeddings it already holds (computed by some earlier,
//! receipted action), or returns `None` to pass to the next gear.
pub trait VectorIndex {
    fn name(&self) -> &str;
    /// Keys are `fact:<row id>` or `def:<name>`; scores higher is better.
    fn search(&self, query: &str, k: usize) -> Option<Vec<(String, f32)>>;
}
