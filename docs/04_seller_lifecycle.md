# Seller Lifecycle Mapping

## Lifecycle Stages

### Stage 0 — Registration (Day 0)
- Available data: identity data, device/IP at signup, stated business category, payout details
- Missing: any behavioral history (cold-start problem)
- Fraud relevance: synthetic identity fraud, collusion rings

### Stage 1 — Onboarding / Listing (Day 0–7)
- Available data: catalog data (items, pricing, listing velocity), profile completeness
- Missing: real transaction/buyer interaction
- Fraud relevance: stolen-goods fraud, triangulation fraud (catalog red flags)

### Stage 2 — First Transactions (Day 0–30, "trust-building window")
- Available data: transaction velocity, buyer identities, payment methods, early complaints
- Missing: enough volume for statistical reliability
- Fraud relevance: bust-out fraud, collusion (buyer/seller device overlap)

### Stage 3 — Growth / Scaling (Day 30–90)
- Available data: enough history for behavioral baselines, revenue trend
- Fraud relevance: abnormal revenue acceleration (growth-monitoring model, Day 11)

### Stage 4 — Established (Day 90+)
- Out of scope for this project — existing production controls are strongest here;
  project deliberately focuses on the early-lifecycle gap before this point.

## Key Design Rule
Every signal designed in Day 7 must be tagged with the earliest lifecycle stage
at which it is actually available. A feature requiring 30 days of history cannot
be used in a Day 2 decisioning policy — this mirrors how real feature stores tag
features by "minimum data maturity."
