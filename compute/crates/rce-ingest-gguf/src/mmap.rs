//! Shared mmap helpers for rce-* runtimes.
//!
//! Every rce-* runtime opens its GGUF/safetensors checkpoint with
//! `Mmap::map(&file)`. None of them issue any `madvise` hint, so the
//! kernel defaults to the equivalent of `MADV_NORMAL` — random-fault
//! every page on first touch, which is the worst possible pattern for
//! a cold-cache multi-GB checkpoint that we walk in file order once
//! at load time and then randomly thereafter.
//!
//! `mmap_load_sequential` issues `MADV_SEQUENTIAL` immediately after
//! mapping, which on macOS + Linux tells the kernel to prefetch
//! pages aggressively and drop them once consumed — cutting cold-load
//! latency on multi-GB models without holding the pages resident.
//! Callers can then opt back into `Normal` (`Mmap::advise`) once the
//! sequential load pass is done, before random-access inference.
//!
//! For runtimes operating in **model > RAM** regime (large MoE,
//! 70B-class dense), the runtime can also call `advise_layer` to
//! issue per-layer `MADV_WILLNEED` just before layer N's forward,
//! and `MADV_DONTNEED` after to bound resident set. See
//! [`LayerAdvice`].

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
// `madvise` exists only on Unix; elsewhere every hint below is a no-op,
// which is allowed: the kernel is free to ignore a hint anyway.
#[cfg(unix)]
use memmap2::{Advice, UncheckedAdvice};

/// Open `path` and `mmap` it with `MADV_SEQUENTIAL` set immediately,
/// so the in-load weight scan doesn't pessimize the page cache.
/// Wrapped in `Arc<Mmap>` so weight tensors can hold cheap references
/// to slices of the mapping without lifetime gymnastics.
///
/// SAFETY: the mmap aliases the file for its lifetime. Callers must
/// ensure no concurrent writer truncates or modifies the file while
/// any tensor view derived from this mapping is live.
pub fn mmap_load_sequential(path: impl AsRef<Path>) -> std::io::Result<Arc<Mmap>> {
    let file = File::open(path.as_ref())?;
    // SAFETY: see fn-level doc.
    let mmap = unsafe { Mmap::map(&file)? };
    // Errors are non-fatal — the kernel is free to ignore the hint.
    #[cfg(unix)]
    let _ = mmap.advise(Advice::Sequential);
    Ok(Arc::new(mmap))
}

/// After the sequential load scan completes, switch the hint to
/// `Normal` (or `Random` if the access pattern is genuinely random).
/// This is the second half of the `mmap_load_sequential` pair —
/// without it, the kernel may keep dropping pages we still want.
pub fn advise_post_load(mmap: &Mmap, advice: PostLoadAdvice) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let a = match advice {
            PostLoadAdvice::Normal => Advice::Normal,
            PostLoadAdvice::Random => Advice::Random,
        };
        mmap.advise(a)
    }
    #[cfg(not(unix))]
    {
        let _ = (mmap, advice);
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PostLoadAdvice {
    Normal,
    Random,
}

/// Per-layer advise wrapper for **model > RAM** regimes: just before
/// layer N's forward, call `will_need(layer_byte_range)` to ask the
/// kernel to prefetch; after the layer is consumed (or you're moving
/// to layer N+1's working set), call `dont_need(byte_range)` to drop
/// the resident pages.
///
/// Both are non-fatal hints. On macOS, `MADV_DONTNEED` is treated as
/// a "may discard" — the kernel will satisfy a later page fault by
/// re-reading from the file mapping, which is exactly what we want
/// for layer streaming. On Linux, the behavior is similar.
pub struct LayerAdvice<'a> {
    mmap: &'a Mmap,
    /// Tensor data starts at this byte offset within the mmap.
    /// Layer ranges are caller-computed: `[layer_data_start +
    /// layer_byte_offsets[N], layer_data_start +
    /// layer_byte_offsets[N+1])`.
    data_start: usize,
}

impl<'a> LayerAdvice<'a> {
    pub fn new(mmap: &'a Mmap, data_start: usize) -> Self {
        Self { mmap, data_start }
    }

    /// Hint that we will need bytes in `[start, start + len)` soon
    /// (relative to `data_start`).
    pub fn will_need(&self, start: usize, len: usize) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            self.mmap.advise_range(Advice::WillNeed, self.data_start + start, len)
        }
        #[cfg(not(unix))]
        {
            let _ = (start, len);
            Ok(())
        }
    }

    /// Hint that bytes in `[start, start + len)` are no longer in the
    /// working set and may be dropped from the page cache (relative
    /// to `data_start`). Marked unsafe because `MADV_DONTNEED` can
    /// race with concurrent dirty writes; rce's read-only checkpoint
    /// mappings make this benign.
    ///
    /// SAFETY: caller must guarantee no concurrent writer to the
    /// underlying file, which holds for rce's read-only checkpoint
    /// mappings.
    pub unsafe fn dont_need(&self, start: usize, len: usize) -> std::io::Result<()> {
        // SAFETY: caller-asserted no concurrent writer; see fn doc.
        #[cfg(unix)]
        unsafe {
            self.mmap.unchecked_advise_range(
                UncheckedAdvice::DontNeed,
                self.data_start + start,
                len,
            )
        }
        #[cfg(not(unix))]
        {
            let _ = (start, len);
            Ok(())
        }
    }
}
