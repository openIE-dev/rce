//! HTTP clients: System-One (`/v1/systemone`) and OpenAI-compatible chat.
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    #[error("http: {0}")]
    Transport(String),
    #[error("http {0}: {1}")]
    Status(u16, String),
    #[error("response: {0}")]
    Shape(String),
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

fn post(url: &str, key: Option<&str>, body: &Value, timeout: Duration) -> Result<Value, HttpError> {
    let mut req = agent(timeout)
        .post(url)
        .header("Content-Type", "application/json");
    if let Some(k) = key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    let mut resp = req
        .send_json(body)
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    let status = resp.status().as_u16();
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    if !(200..300).contains(&status) {
        return Err(HttpError::Status(status, text.chars().take(500).collect()));
    }
    serde_json::from_str(&text).map_err(|e| HttpError::Shape(format!("not JSON: {e}")))
}

/// A System-One endpoint: Jev (TypeSafe), Laya (laya-serve, Laya Studio) or Kev.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOne {
    /// Base URL without the path, e.g. `http://127.0.0.1:8080` or `https://api.laya.studio`.
    pub base_url: String,
    pub model: Option<String>,
    pub key_env: Option<String>,
    pub timeout: Duration,
}

/// One `noul` answer: probability that the statement is true.
#[derive(Debug, Clone, PartialEq)]
pub struct Noul {
    pub p_true: f64,
    pub confidence: Option<f64>,
    pub input_tokens: u64,
    pub model: Option<String>,
}

impl SystemOne {
    /// Ask one `noul` question about `state`.
    pub fn noul(&self, state: &Value, id: &str, instructions: &str) -> Result<Noul, HttpError> {
        let key = self.key_env.as_deref().and_then(|k| std::env::var(k).ok());
        let mut body = json!({"state": state, "questions": {id: {"type": "noul", "instructions": instructions}}});
        if let Some(m) = &self.model {
            body["model"] = json!(m);
        }
        let url = format!("{}/v1/systemone", self.base_url.trim_end_matches('/'));
        let v = post(&url, key.as_deref(), &body, self.timeout)?;
        let a = &v["answers"][id];
        let p = a["noul"]
            .as_f64()
            .ok_or_else(|| HttpError::Shape(format!("answers.{id}.noul missing")))?;
        if !(0.0..=1.0).contains(&p) {
            return Err(HttpError::Shape(format!("noul {p} outside [0, 1]")));
        }
        Ok(Noul {
            p_true: p,
            confidence: a["confidence"].as_f64(),
            input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0),
            model: v["model"].as_str().map(str::to_string),
        })
    }
}

/// An OpenAI-compatible chat completion.
#[derive(Debug, Clone, PartialEq)]
pub struct Chat {
    pub content: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// True when the server reported usage; otherwise token counts are 0.
    pub usage_reported: bool,
}

pub fn chat(
    base_url: &str,
    key: Option<&str>,
    model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> Result<Chat, HttpError> {
    let body = json!({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    });
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let v = post(&url, key, &body, timeout)?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| HttpError::Shape("choices[0].message.content missing".into()))?
        .to_string();
    let u = &v["usage"];
    Ok(Chat {
        content,
        tokens_in: u["prompt_tokens"].as_u64().unwrap_or(0),
        tokens_out: u["completion_tokens"].as_u64().unwrap_or(0),
        usage_reported: u.is_object(),
    })
}
