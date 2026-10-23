//! The provider catalog: one OpenRouter-style table for every model MEI may call.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GearTier {
    /// Local OpenAI-compatible server (vLLM, llama.cpp, Ollama).
    G3,
    /// Frontier model through a hosted provider.
    G4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provider {
    pub provider: String,
    /// OpenAI-compatible base, e.g. `http://127.0.0.1:8000/v1` or `https://openrouter.ai/api/v1`.
    pub base_url: String,
    pub model_id: String,
    /// Name of the environment variable holding the key. `None` for a local
    /// server that needs none. The key value never enters a receipt.
    pub key_env: Option<String>,
    pub gear: GearTier,
    /// USD per million tokens, as published by the provider (input, output).
    pub price_in_per_mtok: Option<f64>,
    pub price_out_per_mtok: Option<f64>,
    /// Prior energy per token, joules. A configured estimate, never a
    /// measurement: a remote call charged with it is tagged `estimated-remote`.
    pub energy_prior_j_per_token: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub providers: Vec<Provider>,
}

impl Catalog {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// The entry for `gear` with the lowest energy prior; entries with no
    /// prior come after those with one, then by input price, then by order.
    pub fn pick(&self, gear: GearTier) -> Option<&Provider> {
        let mut c: Vec<(usize, &Provider)> = self
            .providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.gear == gear)
            .collect();
        c.sort_by(|(i, a), (j, b)| {
            let k = |p: &Provider| {
                (
                    p.energy_prior_j_per_token.unwrap_or(f64::INFINITY),
                    p.price_in_per_mtok.unwrap_or(f64::INFINITY),
                )
            };
            k(a).partial_cmp(&k(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(j))
        });
        c.first().map(|(_, p)| *p)
    }
}
