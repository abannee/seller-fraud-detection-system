# Fraud Investigator Thinking

## Method
Investigation questions are grounded in the six fraud typologies (docs/03) and
mapped to the seller lifecycle stage at which each becomes observable
(docs/04), following: Investigation Question -> Business Question ->
Observable Signal -> ML Feature.

## Investigation Questions

### 1. Identity & setup anomalies (synthetic identity, collusion; Stage 0)
- Question: does registration info look manufactured rather than lived-in?
- Signal: fresh email domain, identity-detail mismatches, device/IP inconsistency
- Feature: identity_consistency_score, email_domain_age_days

### 2. Velocity mismatch with organic growth (bust-out; Stage 1-2)
- Question: would a legitimate new seller really list/sell this fast?
- Signal: listings-per-hour in first 24h, transaction count in first 72h
- Feature: listing_velocity_24h, txn_count_first_72h

### 3. Catalog/product plausibility (stolen goods, triangulation; Stage 1)
- Question: does what they're selling make sense together?
- Signal: high-ticket category concentration, price clustering near thresholds
- Feature: category_entropy, high_ticket_item_ratio

### 4. Buyer-side collusion patterns (collusion; Stage 2)
- Question: are the same buyers, cards, or devices showing up suspiciously often?
- Signal: repeat-buyer concentration, device/IP overlap across accounts
- Feature: top5_buyer_revenue_share, buyer_seller_device_overlap

### 5. Payout and withdrawal behavior (bust-out, account takeover; Stage 2-3)
- Question: are they trying to extract money before scrutiny catches up?
- Signal: time from last transaction to payout request
- Feature: days_since_first_transaction (at payout)

### 6. Behavioral inconsistency over time (account takeover; Stage 3)
- Question: did something change abruptly that a normal business wouldn't do?
- Signal: long quiet history followed by a late abrupt spike
- Feature: behavior_change_index (generalized in the growth-monitoring model, topic 11)
