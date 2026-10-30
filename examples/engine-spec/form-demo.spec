spec "orders-form" v0.1.0
reason "Demo for the Form compiler: two feeds, one validator, two sinks" by "david"
currency max_age=24h

objective O1 "Record valid orders in the ledger"
  metric recorded target 1
  reason "the ledger is the product" by "david"

objective O2 "Keep an audit trail of order ids"
  metric audited target 1
  reason "every accepted order is auditable" by "david"

acceptance A1 validates O1
  property "for all orders missing id or amount, nothing reaches the ledger and the validator escalates"
  reason "bad orders must not be recorded" by "david"

deliverable D1 fulfills O1
  schema "LedgerEntry"
  reason "recorded orders" by "david"

deliverable D2 fulfills O2
  schema "AuditRecord"
  reason "audit ids" by "david"

role R1 "validator" execution_node
  reason "checks orders" by "david"

role R2 "ledger-writer" execution_node
  reason "wraps orders as ledger entries" by "david"

role R3 "lead" coordinator
  reason "takes escalations" by "david"

role R4 "auditor" execution_node
  reason "extracts the audit id" by "david"

dependency DEP1 crate "orders-feed"
  reason "orders from the shop" by "david"

dependency DEP2 crate "manual-entry"
  reason "orders typed in by staff" by "david"

flow F1 from DEP1 to R1
  reason "shop orders are validated" by "david"

flow F2 from DEP2 to R1
  reason "manual orders are validated" by "david"

flow F3 from R1 to R2
  reason "valid orders go to the ledger writer" by "david"

flow F4 from R2 to D1
  reason "ledger entries are recorded" by "david"

flow F5 from R1 to R4
  reason "valid orders are audited" by "david"

flow F6 from R4 to D2
  reason "audit ids are kept" by "david"

escalation E1 from "validator" to R3 timeout 5000ms
  reason "invalid orders go to the lead" by "david"

change_control approvers ["david"] review_period 24h
