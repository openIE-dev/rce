//! The system clipboard, behind a trait so the view model is testable.
//!
//! Desktop: [`System`] uses `arboard` without default features: Win32,
//! AppKit (objc2) and X11 (x11rb, pure Rust; on Wayland through XWayland).
//! Its Wayland data-control backend is left out because it pulls `cc`.
//! If the clipboard can't be opened (no display, headless), [`System`]
//! falls back to an in-process one, so copy/paste still work inside the IDE.
//! The web build uses the browser's copy/cut/paste events instead (web.rs).

pub trait Clipboard {
    fn get(&mut self) -> Option<String>;
    fn set(&mut self, text: &str);
    fn name(&self) -> &'static str;
}

/// In-process only.
#[derive(Debug, Default)]
pub struct Memory(pub Option<String>);

impl Clipboard for Memory {
    fn get(&mut self) -> Option<String> {
        self.0.clone()
    }
    fn set(&mut self, text: &str) {
        self.0 = Some(text.to_string());
    }
    fn name(&self) -> &'static str {
        "memory"
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use system::System;

#[cfg(not(target_arch = "wasm32"))]
mod system {
    use super::{Clipboard, Memory};

    pub struct System {
        cb: Option<arboard::Clipboard>,
        fallback: Memory,
    }

    impl System {
        pub fn new() -> Self {
            System {
                cb: arboard::Clipboard::new().ok(),
                fallback: Memory::default(),
            }
        }
        pub fn is_system(&self) -> bool {
            self.cb.is_some()
        }
    }

    impl Default for System {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Clipboard for System {
        fn get(&mut self) -> Option<String> {
            match self.cb.as_mut() {
                Some(c) => c.get_text().ok(),
                None => self.fallback.get(),
            }
        }
        fn set(&mut self, text: &str) {
            self.fallback.set(text);
            if let Some(c) = self.cb.as_mut() {
                let _ = c.set_text(text.to_string());
            }
        }
        fn name(&self) -> &'static str {
            if self.cb.is_some() {
                "system"
            } else {
                "memory (no system clipboard)"
            }
        }
    }
}
