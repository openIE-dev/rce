/// Format version constant. Stored as a u32 in the header.
///
/// Layout: `MAJOR << 16 | MINOR << 8 | PATCH`.
/// - MAJOR: breaking changes; readers MUST refuse unknown major.
/// - MINOR: additive (new section kinds, new flags); readers MUST tolerate.
/// - PATCH: clarifications, no on-disk impact.
pub const FORMAT_VERSION: u32 = (0u32 << 16) | (1u32 << 8) | 0u32; // 0.1.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatVersion {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

impl FormatVersion {
    pub const CURRENT: Self = Self {
        major: 0,
        minor: 1,
        patch: 0,
    };

    pub fn from_u32(raw: u32) -> Self {
        Self {
            major: ((raw >> 16) & 0xFF) as u8,
            minor: ((raw >> 8) & 0xFF) as u8,
            patch: (raw & 0xFF) as u8,
        }
    }

    pub fn to_u32(&self) -> u32 {
        ((self.major as u32) << 16) | ((self.minor as u32) << 8) | (self.patch as u32)
    }

    /// Reader compatibility check: a reader at `Self::CURRENT` accepts a
    /// file written at `file_version` iff majors match.
    pub fn is_compatible_with(&self, file_version: Self) -> bool {
        self.major == file_version.major
    }
}

impl std::fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
