spec "follow-bible-ingest-schemas" v1.0.0

reason "Multi-format Bible text ingestion pipeline — USFM, eBible ZIP, TEI, specialized corpora" by "david"
currency max_age=336h

# ── Objectives ──────────────────────────────────────────────────

objective O1 "Ingest 50+ Bible translations from TOML manifests"
  metric translations_ingested target 50
  reason "Core-50 manifest defines the initial translation set" by "david"

objective O2 "Support 7+ source formats via pluggable adapters"
  metric adapter_count target 7
  reason "USFM, eBible ZIP, normalized JSON, Nag Hammadi TEI, Ge'ez, Charles 1913, Lightfoot, M.R. James" by "david"

objective O3 "Idempotent re-runs via SHA-256 content-addressed cache"
  metric cache_hit_rate target 95
  reason "Re-ingestion should skip already-processed translations" by "david"

objective O4 "Export to static JSON tree for browser serving"
  metric export_time_seconds target 30
  reason "Follow app loads Bible text from static JSON — must be fast to regenerate" by "david"

# ── Constraints ──────────────────────────────────────────────────

constraint C1 substrate "Rust, edition 2024"
  reason "Consistent with joulesperbit workspace" by "david"

constraint C2 performance "Complete full ingest in under 5 minutes"
  reason "Developer workflow — re-ingest should not block" by "david"

constraint C3 resource "Max 2GB memory during ingest"
  reason "Runs alongside other services on dev machine" by "david"

# ── Acceptance Criteria ──────────────────────────────────────────

acceptance A1 validates O1
  property "for all translations in core-50.toml, ingest produces non-empty book set"
  reason "Every translation must yield at least one book" by "david"

acceptance A2 validates O2
  property "for all registered adapters, parse of well-formed input produces valid Source"
  reason "Adapters must not crash on valid input" by "david"

acceptance A3 validates O3
  property "for all previously ingested translations, re-ingest produces identical output"
  reason "Idempotency — same input always produces same output" by "david"

acceptance A4 validates O4
  property "for all ingested translations, exported JSON is valid and loadable"
  reason "Browser must be able to parse the exported JSON tree" by "david"

# ── Deliverables ──────────────────────────────────────────────────

deliverable D1 fulfills O1
  schema "IngestReport"
  reason "Per-translation success/failure/warning summary" by "david"

deliverable D2 fulfills O2
  schema "Source"
  reason "Parsed Bible text with book/chapter/verse structure" by "david"

deliverable D3 fulfills O3
  schema "CacheEntry"
  reason "SHA-256 keyed cached translation data" by "david"

deliverable D4 fulfills O4
  schema "JsonExportTree"
  reason "Static JSON directory tree for browser serving" by "david"

# ── Roles ──────────────────────────────────────────────────────────

role R1 "manifest-loader" execution_node
  reason "Reads TOML manifests and builds the translation work queue" by "david"

role R2 "adapter-registry" execution_node
  reason "Maps source format kind to the correct parser adapter" by "david"

role R3 "cache-manager" execution_node
  reason "SHA-256 content-addressed cache for idempotent re-runs" by "david"

role R4 "ingest-runner" execution_node
  reason "Orchestrates per-translation ingestion end-to-end" by "david"

role R5 "store-writer" execution_node
  reason "Persists parsed translations to follow-bible-store" by "david"

role R6 "json-exporter" execution_node
  reason "Generates static JSON tree from ingested data" by "david"

role R7 "pipeline-coordinator" coordinator
  reason "Manages pipeline ordering and error aggregation" by "david"

# ── Dependencies ──────────────────────────────────────────────────

dependency DEP1 crate "follow-bible"
  reason "Core Bible data model (Book, Chapter, Verse)" by "david"
  input_schema {"type":"object","required":["book","chapter","verse"],"properties":{"book":{"type":"string"},"chapter":{"type":"integer"},"verse":{"type":"integer"},"text":{"type":"string"}}}

dependency DEP2 crate "follow-bible-usfm"
  reason "USFM parser" by "david"
  input_schema {
    "type": "object",
    "required": ["text", "translation"],
    "properties": {
      "text": {"type": "string"},
      "translation": {"type": "string"}
    }
  }

dependency DEP3 crate "follow-bible-store"
  reason "Persistent store for ingested translations" by "david"
  input_schema {"type":"object","required":["translation_id"],"properties":{"translation_id":{"type":"string"},"books":{"type":"array","items":{"type":"string"}}}}

dependency DEP4 data "core-50.toml"
  reason "Manifest defining the 50 core translations to ingest" by "david"
  input_schema {"type":"object","required":["manifest_path"],"properties":{"manifest_path":{"type":"string"}}}

# ── Escalation Conditions ──────────────────────────────────────────

escalation E1 from "adapter-registry" to R7 timeout 10000ms
  reason "Unknown source format — no adapter registered" by "david"

escalation E2 from "ingest-runner" to R7 timeout 60000ms
  reason "Translation ingest failed — malformed source data" by "david"

escalation E3 from "json-exporter" to R7 timeout 30000ms
  reason "Export failed — disk full or permission error" by "david"

# Data flows: real edges so a task takes more than one step.
flow F1 from DEP4 to R1
  reason "the manifest lists what to ingest" by "david"

flow F2 from R1 to R2
  reason "each manifest entry picks an adapter" by "david"

flow F3 from R2 to R4
  reason "adapters hand sources to the ingest runner" by "david"

flow F4 from DEP2 to R4
  reason "USFM parsing happens during ingest" by "david"

flow F5 from R4 to R5
  reason "parsed translations are written to the store" by "david"

flow F6 from R5 to D1
  reason "store writes roll up into the ingest report" by "david"

flow F7 from R5 to R6
  reason "stored translations are exported as JSON" by "david"

flow F8 from R6 to D4
  reason "exports form the JSON tree" by "david"

flow F9 from DEP3 to R3
  reason "the store backs the cache manager" by "david"

flow F10 from R3 to D3
  reason "cache manager emits cache entries" by "david"

change_control approvers ["david"] review_period 24h
