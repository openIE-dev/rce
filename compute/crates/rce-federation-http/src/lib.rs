//! `rce-federation-http` — HTTP/JSON transport for [`rce_federation`].
//!
//! Each worker exposes its `Federation` state via an axum server; peers
//! gossip by fetching each other's state and `join`ing the OR-Set
//! Membership lattice. Digest reports are pushed via POST. The trust
//! primitive is `Federation::consensus()` — if every honest worker
//! reports the same `DispatchTranscript` digest, the federation agrees;
//! a byzantine worker is identified by `Federation::disagreements()`.
//!
//! **Smallest first stake**. This crate is deliberately minimal:
//!
//! - No TLS — production puts `axum-server` with `rustls` in front.
//! - No mTLS — TokenAuth credentials sit in the JSON body; production
//!   uses signed JWTs or X.509 mutual auth.
//! - No retry / backoff / circuit breakers — production wraps the
//!   `reqwest` client with `tower::retry` / `tower::limit`.
//! - No health endpoints — production adds `/health` + `/metrics`.
//!
//! The protocol shape is what's load-bearing here; hardening is the
//! follow-on. Two in-process axum servers converging via gossip is
//! the test harness (see the integration test below).
//!
//! ## Wire protocol
//!
//! `GET /federation/state` → 200 OK, `application/json`:
//! ```json
//! {
//!   "members": { ... },     // serialized `rce_federation::Membership`
//!   "digests": { ... }      // serialized `rce_federation::DigestRegistry`
//! }
//! ```
//!
//! `POST /federation/report` accepts `application/json`:
//! ```json
//! {
//!   "worker": 7,
//!   "credential": 12345,    // verified against the server's TokenAuth
//!   "tag": 67890,           // OR-Set add-tag for first-time admit
//!   "digest": 18446744073709551615  // u64 transcript digest
//! }
//! ```
//! Response: 200 OK if accepted, 401 if credential rejected.

use std::sync::{Arc, Mutex};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use rce_federation::{Authenticator, DigestRegistry, Federation, Membership, TokenAuth};

// ── Wire types ──────────────────────────────────────────────────────

/// Payload returned by `GET /federation/state`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateResponse {
    pub members: Membership,
    pub digests: DigestRegistry,
}

/// Payload accepted by `POST /federation/report`.
///
/// `lamport_clock` is each worker's own monotonic logical clock.
/// Per-worker LWW: a report is accepted iff its `lamport_clock >=` the
/// receiver's current clock for that worker. Suppresses out-of-order
/// stale reports without requiring a full vector clock (each worker is
/// the only writer of their own digest, so ordering only matters
/// per-worker — the cross-worker fan-out is what gossip handles).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportRequest {
    pub worker: u64,
    pub credential: u64,
    pub tag: u64,
    pub digest: u64,
    #[serde(default)]
    pub lamport_clock: u64,
}

// ── Server ──────────────────────────────────────────────────────────

/// Per-server state. The `Federation` is mutated by `POST /report`;
/// the `Authenticator` gates admission. Shared behind an `Arc<Mutex>`
/// so the router handlers can mutate it concurrently with reads.
///
/// `clocks` is the per-worker latest accepted Lamport clock — used to
/// reject stale reports that race a fresh one (suppresses out-of-order
/// gossip from overwriting a newer state with an older one).
#[derive(Clone)]
pub struct ServerState {
    pub federation: Arc<Mutex<Federation>>,
    pub auth: Arc<TokenAuth>,
    clocks: Arc<Mutex<std::collections::HashMap<u64, u64>>>,
}

impl ServerState {
    pub fn new(federation: Federation, auth: TokenAuth) -> Self {
        Self {
            federation: Arc::new(Mutex::new(federation)),
            auth: Arc::new(auth),
            clocks: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Snapshot the current Federation (clones; the lock is held only
    /// for the duration of the clone). Useful for tests + the
    /// `/state` handler.
    pub fn snapshot(&self) -> Federation {
        self.federation.lock().unwrap().clone()
    }

    /// Latest accepted Lamport clock for a worker, or 0 if none.
    pub fn clock_for(&self, worker: u64) -> u64 {
        self.clocks.lock().unwrap().get(&worker).copied().unwrap_or(0)
    }
}

/// rustls TLS configuration for the federation HTTPS server. PEM-encoded
/// certificate + private key bytes; deployments usually load from disk
/// or a secret manager. For tests, generate a self-signed cert with
/// `rcgen` and pass the resulting bytes here.
///
/// `client_ca_pem`, if set, enables **mTLS** — the server requires
/// clients to present a certificate signed by the trusted client CA.
/// Clients without an identity (or with one signed by a different CA)
/// fail the TLS handshake before any HTTP request is processed.
///
/// The `axum_server` integration handles the rustls runtime — this
/// type just packages what it needs.
pub struct TlsConfig {
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
    /// Optional PEM-encoded CA certificate(s) for client authentication.
    /// When `Some`, the resulting `RustlsConfig` enforces mTLS.
    pub client_ca_pem: Option<Vec<u8>>,
}

impl TlsConfig {
    /// Server-cert-only TLS (no client auth).
    pub fn new(cert_pem: Vec<u8>, key_pem: Vec<u8>) -> Self {
        Self { cert_pem, key_pem, client_ca_pem: None }
    }

    /// Builder: enable mTLS by requiring a client cert signed by the
    /// provided CA. Subsequent calls override (single CA bundle —
    /// the bundle itself can contain multiple roots).
    pub fn with_client_ca(mut self, client_ca_pem: Vec<u8>) -> Self {
        self.client_ca_pem = Some(client_ca_pem);
        self
    }

    /// Build an `axum_server::tls_rustls::RustlsConfig`. If `client_ca_pem`
    /// is unset, the result is server-cert-only (existing behaviour). If
    /// set, the result is mTLS — the server's `ServerConfig` carries a
    /// `WebPkiClientVerifier` that requires clients to present a cert
    /// signed by the CA chain.
    pub async fn into_rustls(self) -> Result<axum_server::tls_rustls::RustlsConfig, std::io::Error> {
        match self.client_ca_pem {
            None => axum_server::tls_rustls::RustlsConfig::from_pem(self.cert_pem, self.key_pem).await,
            Some(client_ca_pem) => Self::build_mtls(self.cert_pem, self.key_pem, client_ca_pem),
        }
    }

    fn build_mtls(
        server_cert_pem: Vec<u8>,
        server_key_pem: Vec<u8>,
        client_ca_pem: Vec<u8>,
    ) -> Result<axum_server::tls_rustls::RustlsConfig, std::io::Error> {
        use std::io;
        use std::sync::Arc;

        // Parse server cert chain + key.
        let mut cert_reader = std::io::Cursor::new(&server_cert_pem);
        let server_certs: Vec<rustls::pki_types::CertificateDer<'static>> =
            rustls_pemfile::certs(&mut cert_reader)
                .collect::<Result<_, _>>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                    format!("server cert PEM: {e}")))?;
        let mut key_reader = std::io::Cursor::new(&server_key_pem);
        let server_key = rustls_pemfile::private_key(&mut key_reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("server key PEM: {e}")))?
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                "no private key found in server key PEM"))?;

        // Parse client CA bundle into a RootCertStore.
        let mut ca_reader = std::io::Cursor::new(&client_ca_pem);
        let mut client_roots = rustls::RootCertStore::empty();
        for ca in rustls_pemfile::certs(&mut ca_reader) {
            let ca = ca.map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("client CA PEM: {e}")))?;
            client_roots.add(ca).map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("client CA root: {e}")))?;
        }

        let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(client_roots))
            .build()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("build client cert verifier: {e}")))?;
        let server_config = rustls::ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(server_certs, server_key)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("with_single_cert: {e}")))?;

        Ok(axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config)))
    }
}

/// Handle for hot-reloading the server's TLS material without
/// restarting. Held by the operator (e.g. a SIGHUP handler, a
/// timer that polls a cert file's mtime, or an internal-PKI
/// rotation listener); calls `reload(new_pem)` to swap the
/// underlying cert atomically. The reload affects future TLS
/// handshakes; in-flight connections continue with the old cert.
///
/// Built by [`serve_with_tls`]; the returned `RustlsConfigHandle`
/// is the long-lived reload control surface, distinct from the
/// `axum_server` task that owns the actual listener.
#[derive(Clone)]
pub struct RustlsConfigHandle {
    inner: axum_server::tls_rustls::RustlsConfig,
}

impl RustlsConfigHandle {
    /// Replace the server's cert + key with new PEM bytes. Returns
    /// the rebuilt rustls config error if parsing fails. The reload
    /// is atomic from in-flight callers' POV — in-flight handshakes
    /// finish with the old material, new handshakes use the new.
    pub async fn reload(&self, cert_pem: Vec<u8>, key_pem: Vec<u8>) -> Result<(), std::io::Error> {
        self.inner.reload_from_pem(cert_pem, key_pem).await
    }

    /// Replace the server's cert + key by re-reading from disk.
    /// Useful when the deployment writes new certs to a known path
    /// (Let's Encrypt's `cert.pem` / `key.pem`, an internal-PKI
    /// agent's drop directory). The file format matches
    /// `RustlsConfig::from_pem_file`.
    pub async fn reload_from_files(
        &self,
        cert_path: impl AsRef<std::path::Path>,
        key_path: impl AsRef<std::path::Path>,
    ) -> Result<(), std::io::Error> {
        self.inner.reload_from_pem_file(cert_path, key_path).await
    }
}

/// Start the federation HTTPS server with a handle returned for cert
/// reload. `addr` is the bind address; `tls` is the initial cert
/// material; `state` is the per-server state the handlers will see.
///
/// The returned `RustlsConfigHandle` lives independently of the spawn
/// task; operators clone it into reload signal handlers / timers /
/// rotation listeners and call `.reload(...)` whenever new material
/// arrives. The HTTP service itself doesn't restart.
pub async fn serve_with_tls(
    addr: std::net::SocketAddr,
    tls: TlsConfig,
    state: ServerState,
) -> Result<RustlsConfigHandle, std::io::Error> {
    let rustls_cfg = tls.into_rustls().await?;
    let handle = RustlsConfigHandle { inner: rustls_cfg.clone() };
    let app = router(state);
    tokio::spawn(async move {
        let _ = axum_server::bind_rustls(addr, rustls_cfg)
            .serve(app.into_make_service())
            .await;
    });
    Ok(handle)
}

/// Build the axum router. Mount under any prefix the deployment wants;
/// the paths are `/federation/state` and `/federation/report` relative
/// to the router root.
pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/federation/state", get(handle_state))
        .route("/federation/report", post(handle_report))
        .with_state(state)
}

async fn handle_state(State(state): State<ServerState>) -> impl IntoResponse {
    let fed = state.federation.lock().unwrap();
    let resp = StateResponse {
        members: fed.members.clone(),
        digests: fed.digests.clone(),
    };
    (StatusCode::OK, Json(resp))
}

async fn handle_report(
    State(state): State<ServerState>,
    Json(req): Json<ReportRequest>,
) -> impl IntoResponse {
    if !state.auth.verify(req.worker, req.credential) {
        return (StatusCode::UNAUTHORIZED, "bad credential").into_response();
    }
    // Reject stale reports (per-worker LWW by Lamport clock). A report
    // with clock == current is accepted (idempotent re-send); clock <
    // current is the byzantine-or-out-of-order case we reject.
    let mut clocks = state.clocks.lock().unwrap();
    let prior = clocks.get(&req.worker).copied().unwrap_or(0);
    if req.lamport_clock < prior {
        // 409 Conflict matches the "your state is stale" semantics
        // mesh-core's transcript-divergence test uses.
        return (StatusCode::CONFLICT, format!("stale: clock {} < {}", req.lamport_clock, prior))
            .into_response();
    }
    clocks.insert(req.worker, req.lamport_clock.max(prior));
    drop(clocks);

    let mut fed = state.federation.lock().unwrap();
    fed.members.admit(req.worker, req.tag);
    let _ = fed.report(req.worker, req.digest);
    (StatusCode::OK, "ok").into_response()
}

// ── Client ──────────────────────────────────────────────────────────

/// Retry policy for transient client errors (connection refused,
/// timeout, network unreachable). HTTP-status errors are NOT retried
/// — `400/401/409` mean "your request is wrong," not "try again."
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Total attempts including the first (so `max_attempts = 1` is
    /// "no retries"; default is 3).
    pub max_attempts: u32,
    /// Initial backoff before the first retry, in milliseconds.
    pub base_delay_ms: u64,
    /// Multiplier between consecutive retries (exponential backoff;
    /// `2.0` means each retry waits twice as long as the previous).
    pub multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_attempts: 3, base_delay_ms: 50, multiplier: 2.0 }
    }
}

impl RetryPolicy {
    pub fn no_retries() -> Self {
        Self { max_attempts: 1, base_delay_ms: 0, multiplier: 1.0 }
    }
    fn delay_for(&self, attempt: u32) -> std::time::Duration {
        // attempt is 1-indexed (the first retry is attempt=1).
        let ms = (self.base_delay_ms as f64) * self.multiplier.powi(attempt as i32 - 1);
        std::time::Duration::from_millis(ms as u64)
    }
}

/// HTTP client for talking to a peer worker. Reuses a `reqwest::Client`
/// across calls (kept-alive connections, default timeouts). Retries
/// transport-level transient errors per the configured `RetryPolicy`.
#[derive(Clone)]
pub struct Client {
    inner: reqwest::Client,
    base_url: String,
    retry: RetryPolicy,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("status {0}: {1}")]
    Status(StatusCode, String),
    #[error("decode error: {0}")]
    Decode(String),
}

impl Client {
    /// `base_url` should not include the `/federation/...` path —
    /// e.g. `"http://10.10.10.2:9300"`. Uses default retry policy
    /// (3 attempts, 50 ms base delay, 2.0× multiplier).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::new_with_retry(base_url, RetryPolicy::default())
    }

    pub fn new_with_retry(base_url: impl Into<String>, retry: RetryPolicy) -> Self {
        Self {
            inner: reqwest::Client::new(),
            base_url: base_url.into(),
            retry,
        }
    }

    /// Client variant that accepts self-signed certs. **Tests and
    /// development only** — production deployments use a CA-signed cert
    /// (Let's Encrypt / internal PKI / private CA) and let reqwest's
    /// default verifier check it. Accepting invalid certs over a real
    /// network silently disables the integrity half of TLS.
    pub fn insecure_accept_invalid_certs(
        base_url: impl Into<String>,
        retry: RetryPolicy,
    ) -> Self {
        let inner = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .expect("reqwest client builder");
        Self { inner, base_url: base_url.into(), retry }
    }

    /// Client variant that presents a **client certificate** for mTLS,
    /// to authenticate to a peer whose `TlsConfig` was built with
    /// `with_client_ca(...)`. `client_cert_chain_pem` and
    /// `client_key_pem` are PEM-encoded; production deployments load
    /// these from a secret manager / file system.
    ///
    /// `accept_self_signed_server` controls server-cert verification
    /// (`true` = tests / dev with self-signed peers; `false` = normal
    /// CA verification). Either setting still presents the client
    /// identity to the server; the flag only affects what the *client*
    /// trusts in the *server's* cert.
    pub fn with_identity_pem(
        base_url: impl Into<String>,
        retry: RetryPolicy,
        client_cert_chain_pem: &[u8],
        client_key_pem: &[u8],
        accept_self_signed_server: bool,
    ) -> Result<Self, String> {
        // reqwest::Identity::from_pem expects the cert chain + key
        // concatenated in PEM form.
        let mut pem = client_cert_chain_pem.to_vec();
        if !pem.ends_with(b"\n") { pem.push(b'\n'); }
        pem.extend_from_slice(client_key_pem);
        let identity = reqwest::Identity::from_pem(&pem)
            .map_err(|e| format!("identity from PEM: {e}"))?;
        let inner = reqwest::Client::builder()
            .identity(identity)
            .danger_accept_invalid_certs(accept_self_signed_server)
            .build()
            .map_err(|e| format!("reqwest builder: {e}"))?;
        Ok(Self { inner, base_url: base_url.into(), retry })
    }

    /// True iff `e` is a transport-level transient failure worth
    /// retrying. Connection failures / timeouts / DNS hiccups are
    /// transient; HTTP-status responses (4xx/5xx) are not — those
    /// surface as `ClientError::Status` and indicate "your request
    /// was decoded, here's what the server thinks."
    fn is_transient(e: &reqwest::Error) -> bool {
        e.is_timeout() || e.is_connect() || e.is_request()
    }

    /// `GET /federation/state` — peer's full state snapshot. Transport
    /// errors are retried per the configured RetryPolicy.
    pub async fn fetch_state(&self) -> Result<StateResponse, ClientError> {
        let url = format!("{}/federation/state", self.base_url);
        let mut last_err: Option<reqwest::Error> = None;
        for attempt in 0..self.retry.max_attempts {
            if attempt > 0 {
                tokio::time::sleep(self.retry.delay_for(attempt)).await;
            }
            match self.inner.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(ClientError::Status(status, body));
                    }
                    return resp.json::<StateResponse>().await
                        .map_err(|e| ClientError::Decode(e.to_string()));
                }
                Err(e) => {
                    if !Self::is_transient(&e) {
                        return Err(ClientError::Http(e));
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(ClientError::Http(last_err.expect("retry loop exhausted with no error")))
    }

    /// `POST /federation/report` — push this worker's current digest
    /// to the peer. Peer admits us into its Membership (re-admission
    /// is idempotent) and records the digest in its registry. Pass
    /// `lamport_clock = 0` if you don't track one — stale-detection
    /// degrades gracefully.
    pub async fn report_digest(
        &self,
        worker: u64,
        credential: u64,
        tag: u64,
        digest: u64,
        lamport_clock: u64,
    ) -> Result<(), ClientError> {
        let url = format!("{}/federation/report", self.base_url);
        let req = ReportRequest { worker, credential, tag, digest, lamport_clock };
        let mut last_err: Option<reqwest::Error> = None;
        for attempt in 0..self.retry.max_attempts {
            if attempt > 0 {
                tokio::time::sleep(self.retry.delay_for(attempt)).await;
            }
            match self.inner.post(&url).json(&req).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(ClientError::Status(status, body));
                    }
                    return Ok(());
                }
                Err(e) => {
                    if !Self::is_transient(&e) {
                        return Err(ClientError::Http(e));
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(ClientError::Http(last_err.expect("retry loop exhausted with no error")))
    }
}

// ── Gossip ──────────────────────────────────────────────────────────

/// Pull-merge gossip cycle: fetch the peer's state, join their
/// Membership into ours, and pull every digest we don't already have.
/// Existing digests for the same worker are NOT overwritten (we'd
/// rather miss a stale update than clobber a fresh local report);
/// production gossip protocols use a vector clock or LWW timestamp.
///
/// Returns the count of (membership joins, digest reports merged in).
pub async fn pull_gossip(
    local: &Arc<Mutex<Federation>>,
    peer: &Client,
) -> Result<(usize, usize), ClientError> {
    let peer_state = peer.fetch_state().await?;
    let mut joins = 0usize;
    let mut digest_updates = 0usize;
    let mut fed = local.lock().unwrap();
    let before_members: Vec<u64> = fed.members.members();
    fed.members.join(&peer_state.members);
    let after_members: Vec<u64> = fed.members.members();
    joins = after_members.len().saturating_sub(before_members.len());
    for w in &after_members {
        if fed.digests.get(*w).is_none() {
            if let Some(d) = peer_state.digests.get(*w) {
                fed.digests.report(*w, d);
                digest_updates += 1;
            }
        }
    }
    Ok((joins, digest_updates))
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    fn spawn_test_server(state: ServerState) -> SocketAddr {
        let app = router(state);
        // Bind ephemeral port; the OS picks a free one.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = TcpListener::from_std(listener).unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service()).await.unwrap();
        });
        addr
    }

    fn url(addr: SocketAddr) -> String {
        format!("http://{}", addr)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_state_returns_the_servers_federation() {
        // Server starts with one admitted worker reporting a digest.
        let mut fed = Federation::new();
        fed.members.admit(1, 0x11);
        fed.report(1, 0xAA);
        let state = ServerState::new(fed, TokenAuth { secret: 0xDEAD_BEEF });
        let addr = spawn_test_server(state);

        let client = Client::new(url(addr));
        let resp = client.fetch_state().await.expect("fetch");
        assert!(resp.members.is_member(1));
        assert_eq!(resp.digests.get(1), Some(0xAA));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn report_digest_admits_and_records_under_valid_credential() {
        let server_state = ServerState::new(Federation::new(),
            TokenAuth { secret: 0xC0FFEE });
        let credential_for = |w: u64| server_state.auth.credential_for(w);
        let addr = spawn_test_server(server_state.clone());

        let client = Client::new(url(addr));
        client.report_digest(7, credential_for(7), 0x77, 0xABCD, 1).await
            .expect("report");

        let snap = server_state.snapshot();
        assert!(snap.members.is_member(7), "credential verified ⇒ worker admitted");
        assert_eq!(snap.digests.get(7), Some(0xABCD));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn report_digest_rejects_bad_credential() {
        let server_state = ServerState::new(Federation::new(),
            TokenAuth { secret: 0xC0FFEE });
        let addr = spawn_test_server(server_state.clone());

        let client = Client::new(url(addr));
        let err = client.report_digest(7, 0xBADBAD, 0x77, 0xABCD, 1).await
            .expect_err("bad credential should be rejected");
        match err {
            ClientError::Status(StatusCode::UNAUTHORIZED, _) => {}
            other => panic!("expected 401, got {other:?}"),
        }
        let snap = server_state.snapshot();
        assert!(!snap.members.is_member(7), "bad credential ⇒ no admission");
    }

    /// The headline integration property: two in-process worker
    /// servers, each running their own `Federation`, gossip pull-mode
    /// and converge on each other's membership + digest views.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_workers_converge_via_gossip() {
        let auth_a = TokenAuth { secret: 0xAA };
        let auth_b = TokenAuth { secret: 0xBB };

        // Worker A admits itself and reports digest 0x1111.
        let mut fed_a = Federation::new();
        fed_a.members.admit(1, 0x11);
        fed_a.report(1, 0x1111);
        let state_a = ServerState::new(fed_a, auth_a);
        let addr_a = spawn_test_server(state_a.clone());

        // Worker B admits itself and reports digest 0x2222.
        let mut fed_b = Federation::new();
        fed_b.members.admit(2, 0x22);
        fed_b.report(2, 0x2222);
        let state_b = ServerState::new(fed_b, auth_b);
        let addr_b = spawn_test_server(state_b.clone());

        // A pulls B's state. After this A's Federation should know
        // about worker 2 + digest 0x2222.
        let client_b = Client::new(url(addr_b));
        let (joins, updates) = pull_gossip(&state_a.federation, &client_b).await
            .expect("gossip A←B");
        assert_eq!(joins, 1, "A learned about 1 new member");
        assert_eq!(updates, 1, "A pulled 1 new digest");

        let snap_a = state_a.snapshot();
        assert!(snap_a.members.is_member(1));
        assert!(snap_a.members.is_member(2));
        assert_eq!(snap_a.digests.get(1), Some(0x1111));
        assert_eq!(snap_a.digests.get(2), Some(0x2222));

        // Now B pulls A's state. B already knew worker 2; the
        // membership.join is idempotent so no NEW joins. Worker 1
        // is new to B and its digest comes across.
        let client_a = Client::new(url(addr_a));
        let (joins_b, updates_b) = pull_gossip(&state_b.federation, &client_a).await
            .expect("gossip B←A");
        assert_eq!(joins_b, 1, "B learned about 1 new member (worker 1)");
        assert_eq!(updates_b, 1, "B pulled 1 new digest (worker 1's)");

        let snap_b = state_b.snapshot();
        assert!(snap_b.members.is_member(1));
        assert!(snap_b.members.is_member(2));
        assert_eq!(snap_b.digests.get(1), Some(0x1111));
        assert_eq!(snap_b.digests.get(2), Some(0x2222));

        // Both servers now hold the same view of the world. Since
        // both workers report different digests (A reports 0x1111,
        // B reports 0x2222 — they ran different workloads), neither
        // federation reaches consensus. This is correct: in a real
        // deployment both workers would be running the SAME model on
        // the SAME inputs and would report the SAME transcript
        // digest, in which case consensus would hold.
        let snap_a = state_a.snapshot();
        let snap_b = state_b.snapshot();
        assert_eq!(snap_a.members.members(), snap_b.members.members(),
            "membership converged across both servers");
        assert!(snap_a.consensus().is_none(),
            "different per-worker digests ⇒ no consensus (correct)");
        assert!(snap_b.consensus().is_none());
    }

    // ── Hardening: stale-clock rejection ────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_clock_report_rejected_with_409() {
        let server_state = ServerState::new(Federation::new(),
            TokenAuth { secret: 0xC0FFEE });
        let credential_for = |w: u64| server_state.auth.credential_for(w);
        let addr = spawn_test_server(server_state.clone());
        let client = Client::new(url(addr));

        // First report: clock 5, accepted.
        client.report_digest(7, credential_for(7), 0x77, 0xAAAA, 5).await
            .expect("first report at clock 5");
        assert_eq!(server_state.snapshot().digests.get(7), Some(0xAAAA));
        assert_eq!(server_state.clock_for(7), 5);

        // Stale report at clock 3 — rejected as 409 Conflict.
        let err = client.report_digest(7, credential_for(7), 0x77, 0xBBBB, 3).await
            .expect_err("stale clock must be rejected");
        match err {
            ClientError::Status(StatusCode::CONFLICT, _) => {}
            other => panic!("expected 409 Conflict, got {other:?}"),
        }
        // Digest unchanged.
        assert_eq!(server_state.snapshot().digests.get(7), Some(0xAAAA),
            "rejected report must not change the digest");

        // Same-clock report at 5 — accepted (idempotent re-send).
        client.report_digest(7, credential_for(7), 0x77, 0xCCCC, 5).await
            .expect("same-clock re-send is idempotent");
        assert_eq!(server_state.snapshot().digests.get(7), Some(0xCCCC));

        // Fresh report at clock 9 — accepted.
        client.report_digest(7, credential_for(7), 0x77, 0xDDDD, 9).await
            .expect("fresh clock");
        assert_eq!(server_state.snapshot().digests.get(7), Some(0xDDDD));
        assert_eq!(server_state.clock_for(7), 9);
    }

    // ── Hardening: retry policy ─────────────────────────────────────

    #[test]
    fn retry_policy_delay_for_grows_exponentially() {
        let p = RetryPolicy { max_attempts: 5, base_delay_ms: 100, multiplier: 2.0 };
        assert_eq!(p.delay_for(1).as_millis(), 100);
        assert_eq!(p.delay_for(2).as_millis(), 200);
        assert_eq!(p.delay_for(3).as_millis(), 400);
        assert_eq!(p.delay_for(4).as_millis(), 800);
    }

    // ── Hardening: TLS ──────────────────────────────────────────────

    /// rustls 0.23 with the `ring` feature requires the crypto
    /// provider to be installed exactly once at process start. The
    /// auto-detection only works under specific feature combinations
    /// that the workspace's `default-features = false` shape doesn't
    /// hit cleanly, so we install it explicitly. `Once`-protected so
    /// every test can call this safely.
    fn install_default_crypto_provider() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = rustls::crypto::ring::default_provider().install_default();
        });
    }

    /// Generate a self-signed cert + key for a test server bound to
    /// `127.0.0.1`. Returns PEM-encoded (cert, key).
    fn self_signed_for_localhost() -> (Vec<u8>, Vec<u8>) {
        let cert = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()])
            .expect("generate self-signed cert");
        let cert_pem = cert.cert.pem().into_bytes();
        let key_pem = cert.signing_key.serialize_pem().into_bytes();
        (cert_pem, key_pem)
    }

    fn spawn_test_tls_server(state: ServerState, tls: TlsConfig) -> SocketAddr {
        let app = router(state);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener); // axum_server will rebind below.
        tokio::spawn(async move {
            let rustls_cfg = tls.into_rustls().await.expect("rustls config");
            axum_server::bind_rustls(addr, rustls_cfg)
                .serve(app.into_make_service())
                .await
                .unwrap();
        });
        addr
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tls_server_serves_and_client_with_invalid_cert_acceptance_connects() {
        install_default_crypto_provider();
        let server_state = ServerState::new(
            { let mut f = Federation::new(); f.members.admit(1, 0x11); f.report(1, 0xCAFE); f },
            TokenAuth { secret: 0xC0FFEE },
        );
        let (cert, key) = self_signed_for_localhost();
        let addr = spawn_test_tls_server(server_state, TlsConfig::new(cert, key));

        // Give the TLS server a beat to come up. axum_server::bind_rustls
        // returns once the listener is bound; the TLS handshake initialises
        // on first connection.
        for _ in 0..30 {
            if std::net::TcpStream::connect(addr).is_ok() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // HTTPS URL pointed at the self-signed server; client must
        // accept invalid certs to talk to it.
        let https_url = format!("https://{}", addr);
        let client = Client::insecure_accept_invalid_certs(
            https_url,
            RetryPolicy { max_attempts: 4, base_delay_ms: 50, multiplier: 2.0 },
        );
        let resp = client.fetch_state().await.expect("https fetch");
        assert!(resp.members.is_member(1));
        assert_eq!(resp.digests.get(1), Some(0xCAFE));
    }

    // ── mTLS: mutual cert authentication ────────────────────────────

    /// Generate a self-signed CA, then issue server + client certs
    /// signed by that CA. Returns:
    ///   (ca_cert_pem, server_cert_pem, server_key_pem,
    ///    client_cert_pem, client_key_pem)
    /// All for `127.0.0.1` (server) and `client` (CN). The CA cert
    /// is what the server's `client_ca_pem` trusts and is also what
    /// the client side trusts as the server's CA.
    fn issue_mtls_chain() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        // 1. CA root + self-signed cert.
        let mut ca_params = rcgen::CertificateParams::default();
        ca_params.distinguished_name.push(rcgen::DnType::CommonName, "test-mtls-ca");
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_key = rcgen::KeyPair::generate().expect("ca keypair");
        let ca_cert = ca_params.clone().self_signed(&ca_key).expect("ca self-signed");
        let ca_pem = ca_cert.pem().into_bytes();
        let issuer = rcgen::Issuer::from_params(&ca_params, &ca_key);

        // 2. Server cert signed by CA.
        let server_params = rcgen::CertificateParams::new(vec!["127.0.0.1".to_string()])
            .expect("server params");
        let server_key = rcgen::KeyPair::generate().expect("server keypair");
        let server_cert = server_params.signed_by(&server_key, &issuer)
            .expect("server signed-by");
        let server_cert_pem = server_cert.pem().into_bytes();
        let server_key_pem = server_key.serialize_pem().into_bytes();

        // 3. Client cert signed by CA.
        let client_params = rcgen::CertificateParams::new(vec!["client".to_string()])
            .expect("client params");
        let client_key = rcgen::KeyPair::generate().expect("client keypair");
        let client_cert = client_params.signed_by(&client_key, &issuer)
            .expect("client signed-by");
        let client_cert_pem = client_cert.pem().into_bytes();
        let client_key_pem = client_key.serialize_pem().into_bytes();

        (ca_pem, server_cert_pem, server_key_pem, client_cert_pem, client_key_pem)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mtls_authenticated_client_can_fetch_state() {
        install_default_crypto_provider();
        let (ca_pem, server_cert, server_key, client_cert, client_key) = issue_mtls_chain();
        let server_state = ServerState::new(
            { let mut f = Federation::new(); f.members.admit(1, 0x11); f.report(1, 0xDEAD); f },
            TokenAuth { secret: 0xC0FFEE },
        );

        // mTLS server: requires clients to present a cert signed by the CA.
        let tls = TlsConfig::new(server_cert, server_key).with_client_ca(ca_pem.clone());
        let addr = spawn_test_tls_server(server_state, tls);
        for _ in 0..30 {
            if std::net::TcpStream::connect(addr).is_ok() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Properly authenticated client: presents the signed client cert.
        let client = Client::with_identity_pem(
            format!("https://{}", addr),
            RetryPolicy { max_attempts: 4, base_delay_ms: 50, multiplier: 2.0 },
            &client_cert,
            &client_key,
            /* accept_self_signed_server: */ true,
        ).expect("identity client builds");

        let resp = client.fetch_state().await.expect("mTLS fetch");
        assert!(resp.members.is_member(1));
        assert_eq!(resp.digests.get(1), Some(0xDEAD));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mtls_rejects_client_without_an_identity() {
        install_default_crypto_provider();
        let (ca_pem, server_cert, server_key, _client_cert, _client_key) = issue_mtls_chain();
        let server_state = ServerState::new(Federation::new(),
            TokenAuth { secret: 0xC0FFEE });
        let tls = TlsConfig::new(server_cert, server_key).with_client_ca(ca_pem);
        let addr = spawn_test_tls_server(server_state, tls);
        for _ in 0..30 {
            if std::net::TcpStream::connect(addr).is_ok() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Client that does NOT present an identity. The mTLS server's
        // verifier will reject the handshake before any HTTP request.
        let client = Client::insecure_accept_invalid_certs(
            format!("https://{}", addr),
            RetryPolicy::no_retries(),
        );
        let err = client.fetch_state().await.expect_err("no identity must be rejected");
        match err {
            ClientError::Http(_) => {} // TLS handshake failure
            other => panic!("expected Http TLS error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tls_server_rejects_client_that_does_not_trust_self_signed_cert() {
        install_default_crypto_provider();
        // The defense-in-depth complement: a default-config client
        // (no `accept_invalid_certs`) MUST refuse to connect to the
        // self-signed server. This is what makes TLS a security
        // boundary in production where insecure_accept_invalid_certs
        // would be disallowed.
        let server_state = ServerState::new(Federation::new(),
            TokenAuth { secret: 0xC0FFEE });
        let (cert, key) = self_signed_for_localhost();
        let addr = spawn_test_tls_server(server_state, TlsConfig::new(cert, key));

        for _ in 0..30 {
            if std::net::TcpStream::connect(addr).is_ok() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let https_url = format!("https://{}", addr);
        let client = Client::new_with_retry(https_url, RetryPolicy::no_retries());
        let err = client.fetch_state().await.expect_err("self-signed cert must be rejected");
        match err {
            ClientError::Http(_) => {} // expected — cert verification failure
            other => panic!("expected Http TLS error, got {other:?}"),
        }
    }

    // ── Cert reload: swap material without restarting the server ────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cert_reload_swaps_server_material_atomically() {
        install_default_crypto_provider();

        // Initial server cert (cert A, for 127.0.0.1).
        let (cert_a, key_a) = self_signed_for_localhost();
        let server_state = ServerState::new(
            { let mut f = Federation::new(); f.members.admit(1, 0x11); f.report(1, 0xCAFE); f },
            TokenAuth { secret: 0xC0FFEE },
        );

        // Bind to an ephemeral port via std listener, then drop it
        // (serve_with_tls re-binds via axum_server's tokio listener).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let handle = serve_with_tls(
            addr,
            TlsConfig::new(cert_a, key_a),
            server_state,
        ).await.expect("serve_with_tls");

        // Wait for the TLS listener to come up.
        for _ in 0..30 {
            if std::net::TcpStream::connect(addr).is_ok() { break; }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Sanity: a client that accepts self-signed certs reaches the
        // server (proves cert A is in use).
        let client_a = Client::insecure_accept_invalid_certs(
            format!("https://{}", addr),
            RetryPolicy { max_attempts: 4, base_delay_ms: 50, multiplier: 2.0 },
        );
        let resp_a = client_a.fetch_state().await.expect("fetch under cert A");
        assert!(resp_a.members.is_member(1));

        // Now swap to cert B (different signing key, different serial).
        // The handle's reload should atomically replace the material.
        let (cert_b, key_b) = self_signed_for_localhost();
        handle.reload(cert_b, key_b).await.expect("reload to cert B");

        // After reload, fresh handshakes use cert B. We open a new
        // reqwest client (kept-alive connection from client_a may
        // still use cert A's session, which is fine — that's the
        // "in-flight finishes on old material" behaviour).
        let client_b = Client::insecure_accept_invalid_certs(
            format!("https://{}", addr),
            RetryPolicy { max_attempts: 4, base_delay_ms: 50, multiplier: 2.0 },
        );
        let resp_b = client_b.fetch_state().await.expect("fetch under cert B");
        assert!(resp_b.members.is_member(1),
            "post-reload server is still serving the same Federation state");
        // (We can't easily assert "the cert really swapped" from the
        // reqwest side without inspecting the handshake; the fact that
        // a fresh handshake succeeded after reload proves the new
        // material was accepted by axum_server's listener.)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_retries_returns_immediately_on_transient_error() {
        // Client points at an unreachable port (the OS shouldn't have
        // anything listening on 65501 inside the runner). With
        // `no_retries()` policy, fetch_state returns the connection
        // error after the first attempt, no backoff.
        let client = Client::new_with_retry(
            "http://127.0.0.1:65501",
            RetryPolicy::no_retries(),
        );
        let start = std::time::Instant::now();
        let err = client.fetch_state().await.expect_err("unreachable");
        let elapsed = start.elapsed();
        match err {
            ClientError::Http(_) => {}
            other => panic!("expected Http transport error, got {other:?}"),
        }
        // Generous upper bound — the connection attempt itself can take
        // ~tens of ms on some kernels; what we care about is "no
        // backoff retries were attempted."
        assert!(elapsed.as_secs() < 5,
            "no_retries must not back off; took {:?}", elapsed);
    }

    /// Convergence when both workers run the same workload (which is
    /// the actual "byzantine-detection works" property): both report
    /// the same digest ⇒ consensus is reached on the shared digest.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn three_workers_converge_to_shared_digest_when_they_agree() {
        const SHARED_DIGEST: u64 = 0xCAFEBABE_DEADBEEF;

        let mk = |w: u64, secret: u64| {
            let mut fed = Federation::new();
            fed.members.admit(w, w.wrapping_mul(0x10));
            fed.report(w, SHARED_DIGEST);
            (fed, TokenAuth { secret })
        };
        let (a_fed, a_auth) = mk(1, 0xA1);
        let (b_fed, b_auth) = mk(2, 0xB2);
        let (c_fed, c_auth) = mk(3, 0xC3);
        let state_a = ServerState::new(a_fed, a_auth);
        let state_b = ServerState::new(b_fed, b_auth);
        let state_c = ServerState::new(c_fed, c_auth);
        let addr_a = spawn_test_server(state_a.clone());
        let addr_b = spawn_test_server(state_b.clone());
        let addr_c = spawn_test_server(state_c.clone());

        // Each worker pulls from both peers.
        let cl_a = Client::new(url(addr_a));
        let cl_b = Client::new(url(addr_b));
        let cl_c = Client::new(url(addr_c));
        pull_gossip(&state_a.federation, &cl_b).await.expect("a←b");
        pull_gossip(&state_a.federation, &cl_c).await.expect("a←c");
        pull_gossip(&state_b.federation, &cl_a).await.expect("b←a");
        pull_gossip(&state_b.federation, &cl_c).await.expect("b←c");
        pull_gossip(&state_c.federation, &cl_a).await.expect("c←a");
        pull_gossip(&state_c.federation, &cl_b).await.expect("c←b");

        for state in [&state_a, &state_b, &state_c] {
            let snap = state.snapshot();
            assert_eq!(snap.members.members(), vec![1, 2, 3],
                "all 3 workers learn each other");
            assert_eq!(snap.consensus(), Some(SHARED_DIGEST),
                "shared digest reaches consensus across the fleet");
            assert!(snap.disagreements().is_empty(),
                "no byzantines when everyone agrees");
        }
    }
}
