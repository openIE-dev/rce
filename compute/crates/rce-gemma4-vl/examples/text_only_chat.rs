//! Sanity check: drive rce-gemma4 via the SAME chat-frame
//! tokenization the multimodal path uses, but with NO image. Confirms
//! the chat template tokenization is right; isolates whether the
//! image splice or the template is at fault.
use rce_gemma4::Gemma4Runtime;
use rce_ingest_gguf::{sample_with_history, GgufValue, SampleConfig};
fn main() {
    let txt = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let rt = Gemma4Runtime::load(txt).expect("load");
    let tk = rt.tokenizer();
    let bos = tk.bos_token_id.unwrap_or(2);
    let eos = tk.eos_token_id.unwrap_or(1);
    let is_gpt2 = rt.gguf.metadata.iter()
        .find(|kv| kv.key == "tokenizer.ggml.model")
        .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
        .unwrap_or(false);
    let encode = |s: &str| -> Vec<i32> {
        if is_gpt2 { tk.encode_gpt2(s, false) } else { tk.encode(s, false) }
    };
    let mut ids: Vec<i32> = Vec::new();
    ids.push(bos);
    ids.extend(encode("<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"));
    eprintln!("prompt ids ({} tok) = {:?}", ids.len(), &ids[..ids.len().min(20)]);
    let mut state = rt.new_state();
    let mut next = -1i32;
    for (i, &id) in ids.iter().enumerate() {
        let lg = rt.forward_logits(&mut state, id);
        if i == ids.len() - 1 {
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            eprintln!("post-prompt top5={:?}", idx[..5].iter().map(|&j| (j, lg[j])).collect::<Vec<_>>());
            next = idx[0] as i32;
        }
    }
    let mut out = Vec::new();
    for _ in 0..30 {
        if next == eos { eprintln!("(eos)"); break; }
        out.push(next);
        let lg = rt.forward_logits(&mut state, next);
        let (am, _) = lg.iter().copied().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(i,v),(j,w)| if w>v {(j,w)} else {(i,v)});
        next = am as i32;
    }
    eprintln!("answer = {:?}", tk.decode_sequence(&out, true));
}
