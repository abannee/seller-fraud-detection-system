# EDA & Root-Cause Analysis

## Purpose
Validates the signal design from docs/07 against the engineered feature
table (docs/08) before any model is trained — checking whether each feature
actually separates fraud from legitimate sellers as designed, and where it
does not.

## Key Findings

### Masking effect in aggregate fraud/not-fraud views
payout_lag_days averages 18.7 days for bust-out fraud (strong signal) but
86.7 days for collusion (no signal) — nearly identical to legitimate
sellers' 89.0. Aggregating all fraud types into one bucket masks this;
per-typology evaluation is required (carried into modeling, docs/12).

### Multicollinearity: category_entropy and high_ticket_item_ratio
These two features correlate at 0.95 — both are downstream measurements
of the same underlying behavior (catalog concentration in high-value
categories, typical of stolen-goods/triangulation fraud). Treated as a
redundant pair going into modeling and interpretation (docs/12, docs/14).

### Outlier lift analysis (IQR-flagged points, fraud rate vs. base rate)
| Feature | Lift |
|---|---|
| category_entropy | 14.08x |
| identity_consistency_score | 10.77x |
| high_ticket_item_ratio | 7.99x |
| behavior_change_index | 3.75x |
| payout_lag_days | 3.63x |
| listing_velocity_24h | 3.17x |
| txn_count_first_72h(_zscore) | 0.77-0.79x (below 1 — outliers here are NOT fraud-indicative) |

Raw/normalized early transaction count outliers are, if anything, slightly
more associated with legitimate high-volume sellers than fraud — refines
the topic-6 hypothesis that early velocity alone signals bust-out fraud;
it only holds combined with the spike-after-quiet-period shape, not as a
standalone outlier flag.

## Root-Cause Finding: Typology Coverage Gaps
- Well covered: stolen goods/triangulation, synthetic identity, collusion
  (each has 1-2 strong, typology-specific features)
- Thinly covered: bust-out (payout_lag_days only), account takeover
  (behavior_change_index only) — single-feature dependence with no
  redundant backup signal. Motivates a dedicated growth-monitoring model
  (docs/11) rather than relying on the general feature set alone.
