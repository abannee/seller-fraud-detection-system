# Modeling (XGBoost / LightGBM)

## Method
- Combined feature table: docs/08 engineered features + docs/11 growth-
  monitoring signal (max_percentile_jump_7d)
- Chronological train/test split (80/20 by registration_date, cutoff
  2025-10-19) rather than random -- simulates only ever training on past
  data, and can surface temporal drift a random split would hide
- Class imbalance handled via scale_pos_weight (loss reweighting, not
  resampling) -- ~12.7x weight on fraud examples during training
- Naive rule-based baseline built to stand in for an existing production
  control, for incremental-lift comparison

## Results (chronological holdout test set)
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Naive rule baseline | precision=0.144, recall=0.594 | n/a (not a scored ranking) |
| XGBoost | 0.9959 | 0.9825 |
| LightGBM | 0.9967 | 0.9852 |

## Incremental Lift (precision at recall matched to the naive rule)
XGBoost precision at recall=0.594: 1.000 vs. rule's 0.144 -- a 6.95x
precision lift at identical fraud-catch rate.

## Per-Typology Recall at threshold=0.5 (XGBoost)
Account-takeover, bust-out, stolen-goods, triangulation: 100%.
Collusion: 80%. Synthetic identity: 77.8% -- trailing typologies are
consistent with docs/09's coverage-gap finding (narrower signal support).

## Important Caveat
These metrics are near-ceiling because the synthetic dataset (docs/05)
uses deterministic, non-overlapping typology signatures with no noise
layer. Real production fraud data typically shows substantially more
overlap between fraud and legitimate behavior; real-world PR-AUC would be
expected to be materially lower. These results validate the modeling
pipeline and methodology, not a claim about real-world performance.
