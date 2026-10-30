spec "soak-g2-schemas" v0.1.0

reason "Soak spec whose roles declare JSON output schemas, so the gear stack has something to synthesize against (G2)" by "david"
currency max_age=168h

objective O1 "Every order is priced and booked"
  metric orders_booked target 1000
  reason "Throughput target for the soak" by "david"

acceptance A1 validates O1
  property "for all orders, the booked record carries order_id and amount"
  reason "The ledger must be able to reconcile" by "david"

deliverable D1 fulfills O1
  schema "PricedOrder"
  reason "Priced order for the ledger" by "david"
  output_schema {"type":"object","required":["data"],"properties":{"data":{"type":"object"}}}

deliverable D2 fulfills O1
  schema "LedgerRow"
  reason "Ledger row" by "david"
  output_schema {"type":"object","required":["order_id","amount"],"properties":{"order_id":{"type":"string"},"amount":{"type":"integer"}}}

role R1 "order-envelope" execution_node
  reason "Wraps the raw order in an envelope" by "david"
  output_schema {"type":"object","required":["data"],"properties":{"data":{"type":"object"}}}

role R2 "amount-picker" execution_node
  reason "Extracts the amount from the envelope" by "david"
  output_schema {
    "type": "object",
    "required": ["amount"],
    "properties": {"amount": {"type": "integer"}}
  }

role R3 "ledger-writer" execution_node
  reason "Writes the ledger row" by "david"
  output_schema {"type":"object","required":["order_id","amount"],"properties":{"order_id":{"type":"string"},"amount":{"type":"integer"}}}

role R4 "order-tag" execution_node
  reason "Tags an order with its id only" by "david"
  output_schema {"type":"string"}

role R5 "coordinator" coordinator
  reason "Takes escalations" by "david"

dependency DEP1 service "orders-api"
  reason "Order events" by "david"
  input_schema {"type":"object","required":["order_id","amount"],"properties":{"order_id":{"type":"string"},"amount":{"type":"integer"}}}

escalation E1 from "amount-picker" to R5 timeout 5000ms
  reason "Soak wiring for this example" by "david"

flow F1 from DEP1 to R1
  reason "Soak wiring for this example" by "david"
flow F2 from R1 to R2
  reason "Soak wiring for this example" by "david"
flow F3 from R2 to D1
  reason "Soak wiring for this example" by "david"
flow F4 from DEP1 to R3
  reason "Soak wiring for this example" by "david"
flow F5 from R3 to D2
  reason "Soak wiring for this example" by "david"
flow F6 from DEP1 to R4
  reason "Soak wiring for this example" by "david"

change_control approvers ["david"] review_period 24h
