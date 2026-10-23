//! PNG output for offscreen renders (the `png` crate: pure Rust).
pub fn encode(w: u32, h: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = vec![];
    {
        let mut e = png::Encoder::new(&mut out, w, h);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        let mut wr = e.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(rgba).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

pub fn write(path: &str, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    std::fs::write(path, encode(w, h, rgba)?).map_err(|e| format!("{path}: {e}"))
}
