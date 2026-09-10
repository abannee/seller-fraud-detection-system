# Explainability (SHAP)

## Method
SHAP TreeExplainer (exact Shapley values for tree ensembles) on the
XGBoost model from docs/12, providing per-seller additive attribution --
closing the traceability requirement from docs/06 (every flagged seller
needs a specific, defensible reason, not just a score).

## Global Feature Importance (mean |SHAP value|)
payout_lag_days (2.22) leads, followed by category_entropy (1.08),
top5_buyer_revenue_share (0.96), behavior_change_index (0.95),
identity_consistency_score (0.90). listing_velocity_24h is essentially
unused (0.0001) -- despite being designed in docs/07 to target bust-out
fraud, payout_lag_days captures that same behavior more directly and the
model leaned on it exclusively. payout_lag_days' magnitude is partly
inflated by the 999-sentinel encoding for "no payout yet" (docs/12),
worth noting alongside the genuine underlying signal.

## Per-Typology Validation Against docs/07 Signal Catalog
5 of 6 typologies show the model's dominant SHAP driver exactly matching
the signal the catalog was designed to target (bust-out -> payout_lag_days,
account takeover -> behavior_change_index, stolen goods/triangulation ->
category_entropy, synthetic identity -> identity_consistency_score).
Collusion is a partial match: the model also leans on
identity_consistency_score in addition to the catalog's intended
top5_buyer_revenue_share, tracing back to a secondary design choice in
the docs/05 generator (collusion sellers were also given reduced identity
consistency) that the docs/07 catalog didn't originally anticipate.

## Redundant-Pair Resolution (docs/09)
Of the 0.95-correlated category_entropy / high_ticket_item_ratio pair,
category_entropy carries ~98% of their combined SHAP importance. Tree
ensembles concentrate importance on one correlated feature rather than
splitting credit evenly the way a linear model's coefficients might.

## Individual Case Explanations
Per-seller SHAP breakdowns produce investigator-readable output, e.g.
"flagged primarily because catalog category concentration (category_entropy
= 0.92) pushed risk up by 7.17" -- directly answering the traceability
requirement from docs/06.
