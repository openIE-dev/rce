spec "web-api" v0.1.0

reason "Production web API for OpenIE platform" by "david"
currency max_age=168h

objective O1 "Serve HTTP requests at 1000 rps"
  metric requests_per_second target 1000
  reason "Core business requirement — API must handle production traffic" by "david"

objective O2 "Sub-100ms p99 latency"
  metric p99_latency_ms target 100
  reason "User experience SLA — slow APIs kill products" by "david"

objective O3 "Zero unhandled errors in production"
  metric unhandled_error_rate target 0
  reason "Every error must be caught, logged, and escalated" by "david"

constraint C1 substrate "Must run on ARM64 Apple Silicon (Mac Studio)"
  reason "Deployment target is a single on-premises host" by "david"

constraint C2 performance "Max memory 512MB RSS"
  reason "Shared hosting environment with other services" by "david"

constraint C3 compatibility "OpenAPI 3.1 compliant"
  reason "External consumers depend on stable API contract" by "david"

acceptance A1 validates O1
  property "for all valid HTTP requests, response completes within 1s"
  reason "Availability — requests must not hang" by "david"

acceptance A2 validates O2
  property "for 99th percentile of requests, latency < 100ms"
  reason "Performance SLA" by "david"

acceptance A3 validates O3
  property "for all inputs including malformed, no panic occurs"
  reason "Robustness — panics are unacceptable in production" by "david"

deliverable D1 fulfills O1
  schema "HttpResponse"
  reason "Primary API output — JSON response envelope" by "david"

deliverable D2 fulfills O3
  schema "ErrorReport"
  reason "Structured error reporting for observability" by "david"

role R1 "request-handler" execution_node
  reason "Receives and processes incoming HTTP requests" by "david"

role R2 "auth-validator" execution_node
  reason "Validates JWT tokens and extracts claims" by "david"

role R3 "graph-coordinator" coordinator
  reason "Manages graph topology and applies mutations" by "david"

role R4 "ai-foreman" ai_foreman
  reason "Resolves ambiguities and generates handler implementations" by "david"

dependency DEP1 service "auth-service"
  reason "JWT validation requires public key from auth service" by "david"

dependency DEP2 data "jouledb"
  reason "Persistent storage for request logs and state" by "david"

escalation E1 from "request-handler" to R3 timeout 5000ms
  reason "Handler timeout — coordinator must intervene" by "david"

escalation E2 from "auth-validator" to R4 timeout 10000ms
  reason "Auth failure pattern requires AI analysis" by "david"

# Data flows: real edges so a task takes more than one step.
flow F1 from DEP1 to R2
  reason "auth-service keys feed the JWT validator" by "david"

flow F2 from R2 to R1
  reason "validated claims reach the request handler" by "david"

flow F3 from DEP2 to R1
  reason "stored state is read by the request handler" by "david"

flow F4 from R1 to D1
  reason "handled requests become HTTP responses" by "david"

flow F5 from R2 to D2
  reason "auth failures become structured error reports" by "david"

change_control approvers ["david"] review_period 24h
