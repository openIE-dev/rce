spec "ask-server" v1.0.0

reason "JWP backend for askdavidc.ai — zero HTTP surface, encrypted bidirectional protocol" by "david"
currency max_age=168h

# ── Objectives ──────────────────────────────────────────────────

objective O1 "Accept authenticated JWP connections on port 9090"
  metric concurrent_connections target 1000
  reason "Primary entry point for all askdavidc clients" by "david"

objective O2 "Route commands to 14+ everything-app services"
  metric service_coverage target 14
  reason "Single backend for settings, shortcuts, a11y, i18n, collab, marketplace, permissions, isolation, autosave, undo, datalink, PDF, code, lint" by "david"

objective O3 "Bridge JWP to WebSocket on port 9091"
  metric ws_latency_ms target 50
  reason "Browser clients connect via WebSocket — bridge must be transparent" by "david"

objective O4 "Execute AI cascade queries with energy metering"
  metric queries_per_second target 100
  reason "Core AI functionality — every query gets a picojoule energy receipt" by "david"

# ── Constraints ──────────────────────────────────────────────────

constraint C1 substrate "Rust, tokio async runtime, ARM64 Apple Silicon"
  reason "Production runs on a single on-premises host" by "david"

constraint C2 performance "Max 256MB RSS baseline"
  reason "Shared server with Caddy + 50 sites" by "david"

constraint C3 regulatory "All traffic encrypted via JWP — no plaintext"
  reason "Zero-trust protocol — HMAC-SHA256, device-code, or WebAuthn passkey auth" by "david"

constraint C4 compatibility "JWP wire protocol v1 backward compatible"
  reason "Deployed clients depend on stable wire format" by "david"

# ── Acceptance Criteria ──────────────────────────────────────────

acceptance A1 validates O1
  property "for all valid JWP handshake frames, connection established within 100ms"
  reason "Auth must be fast — users notice >100ms" by "david"

acceptance A2 validates O2
  property "for all service commands, handler dispatches without panic"
  reason "Service routing must never crash regardless of input" by "david"

acceptance A3 validates O3
  property "for all WebSocket messages, JWP frame round-trips within 50ms"
  reason "Bridge latency SLA" by "david"

acceptance A4 validates O4
  property "for all cascade queries, energy receipt is non-zero and accurate"
  reason "Energy metering is the key differentiator — must be trustworthy" by "david"

# ── Deliverables ──────────────────────────────────────────────────

deliverable D1 fulfills O1
  schema "JwpConnection"
  reason "Authenticated connection with session state" by "david"

deliverable D2 fulfills O2
  schema "ServiceResponse"
  reason "Unified response envelope from any service" by "david"

deliverable D3 fulfills O3
  schema "WsBridgeFrame"
  reason "WebSocket ↔ JWP bridge frame" by "david"

deliverable D4 fulfills O4
  schema "CascadeResult"
  reason "AI query result with energy receipt" by "david"

# ── Roles ──────────────────────────────────────────────────────────

role R1 "jwp-listener" execution_node
  blast_radius auth
  reason "TCP listener on 9090 — accepts and authenticates JWP connections (handles auth state)" by "david"

role R2 "ws-bridge" execution_node
  blast_radius public
  reason "WebSocket bridge on 9091 — visible to external browser clients" by "david"

role R3 "service-router" execution_node
  blast_radius subgraph
  reason "Dispatches commands to the 14 everything-app services" by "david"

role R4 "cascade-executor" execution_node
  blast_radius service
  reason "Runs AI queries through the L0-L10 cascade pipeline (calls external LLMs)" by "david"

role R5 "connection-coordinator" coordinator
  blast_radius subgraph
  reason "Manages connection lifecycle, auth state, session cleanup" by "david"

role R6 "foreman" ai_foreman
  blast_radius local
  reason "Resolves novel query patterns that escape the cascade" by "david"

# ── Dependencies ──────────────────────────────────────────────────

dependency DEP1 crate "jwp"
  reason "Joule Wire Protocol implementation" by "david"

dependency DEP2 crate "jwp-auth"
  reason "HMAC-SHA256, device-code, WebAuthn authentication" by "david"

dependency DEP3 crate "ask-core"
  reason "CascadePipeline, JouleDB, tantivy integration" by "david"

dependency DEP4 crate "tokio"
  reason "Async runtime for concurrent connections" by "david"

dependency DEP5 service "jouledb"
  reason "Persistent storage for sessions, usage, billing" by "david"

# ── Escalation Conditions ──────────────────────────────────────────

escalation E1 from "jwp-listener" to R5 timeout 5000ms
  reason "Connection handshake timeout — coordinator decides retry or reject" by "david"

escalation E2 from "service-router" to R6 timeout 30000ms
  reason "Service handler failed — AI foreman may need to generate a new handler" by "david"

escalation E3 from "cascade-executor" to R6 timeout 60000ms
  reason "Cascade exhausted all levels — novel query needs LLM intervention" by "david"

# Data flows: real edges so a task takes more than one step.
flow F1 from DEP1 to R1
  reason "JWP frames arrive at the listener" by "david"

flow F2 from DEP2 to R1
  reason "the listener authenticates each connection" by "david"

flow F3 from R1 to R3
  reason "authenticated requests are routed to a service" by "david"

flow F4 from R3 to R4
  reason "queries go to the cascade executor" by "david"

flow F5 from DEP3 to R4
  reason "ask-core supplies the cascade pipeline" by "david"

flow F6 from R4 to D4
  reason "the cascade returns a result" by "david"

flow F7 from R3 to D2
  reason "routed calls produce a service response" by "david"

flow F8 from R1 to R2
  reason "browser clients are bridged over WebSocket" by "david"

flow F9 from R2 to D3
  reason "the bridge emits WebSocket frames" by "david"

flow F10 from R1 to D1
  reason "accepted connections are reported" by "david"

change_control approvers ["david"] review_period 24h
