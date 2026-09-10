# Seller Fraud Detection System
### Early-lifecycle fraud strategy & ML for a marketplace-style seller risk program

## Overview
This project simulates an industry-grade seller fraud detection system for a payment
facilitator / online marketplace. It is built as a production simulation, the goal is to
reproduce the end-to-end thinking, tooling, and trade-offs of a real Seller Risk /
Fraud Strategy team, from business problem to a deployable decisioning policy.

No proprietary data, models, or code is used here — all data is synthetic or public, and all logic is rebuilt from first principles.

## What This Project Demonstrates
The objective was not to maximize a model's AUC on a static dataset. It was to
simulate the full loop of an early-lifecycle seller risk program: understanding why
sellers commit fraud, designing signals that are actually available at the right
point in the seller lifecycle, engineering features from those signals, validating
them with exploratory and root-cause analysis, building and evaluating models
against precision-recall and operational cost trade-offs, explaining model decisions
in investigator-readable terms, and translating results into a tiered production
policy — including the data pipeline design that would serve it in production.

## Approach

| # | Stage | What it produced |
|---|---|---|
| 1 | Project Vision | Scope and success definition |
| 2 | Business Understanding | Stakeholder map (Risk/Product/Ops/Engineering), business economics |
| 3 | Fraud Motivation & Typologies | Six named fraud typologies with distinct behavioral signatures |
| 4 | Seller Lifecycle Mapping | Data-availability-by-stage map, Day 0 to Day 90+ |
| 5 | Synthetic Data Design | 5,000-seller dataset with typology-specific behavioral injection |
| 6 | Fraud Investigator Thinking | Investigation questions mapped to typologies and lifecycle stage |
| 7 | Risk Signal Design | Ten-signal catalog with source, availability, and risk direction |
| 8 | Feature Engineering | Point-in-time-safe feature pipeline (pandas) |
| 9 | EDA & Root-Cause Analysis | Outlier-lift analysis, typology coverage gaps identified |
| 10 | Unsupervised Clustering | Validated signal design without using the fraud label |
| 11 | Growth-Monitoring Model | Peer-baseline trajectory anomaly signal |
| 12 | Modeling (XGBoost/LightGBM) | Chronologically-split models, incremental lift vs. a naive rule |
| 13 | Evaluation | Precision-recall curve, cost-based threshold, walk-forward back-test |
| 14 | Explainability (SHAP) | Per-typology and per-case model interpretability |
| 15 | Policy Design & Deployment Simulation | Multi-tier action policy, deployment stability check |

## Key Results (verified against this synthetic dataset)
- **Incremental lift over a naive rule-based control:** at matched recall, the
  XGBoost model achieved 1.000 precision vs. the rule's 0.144 — a ~7x precision
  lift at the same fraud-catch rate.
- **Signal design validated three independent ways:** SHAP attribution matched
  the original signal catalog's intended target for 5 of 6 fraud typologies;
  unsupervised clustering (with no fraud label) rediscovered the strongest
  typology pair as a natural cluster; outlier-lift analysis showed up to 14x
  fraud enrichment in the top signal.
- **Cost-based threshold selection** reduced total expected operational cost by
  ~85% compared to a naive 0.5 default threshold, using a stated cost model
  (review cost vs. actual dollar loss).
- **Multi-tier policy** further reduced cost by ~36% vs. a single-threshold
  policy, by removing manual review cost from high-confidence cases.
- **Stability confirmed via walk-forward back-testing** across four sequential
  time windows — no single lucky split.

## Important Caveat on These Numbers
Model performance metrics in this project (ROC-AUC ~0.99, near-perfect precision
at moderate recall) are **not representative of real-world fraud model
performance**. The synthetic dataset (docs/05) uses deterministic, non-overlapping
typology signatures with no noise layer, which makes fraud and legitimate behavior
far more separable than in real production data. These results validate the
**methodology and pipeline** — signal design, feature engineering, modeling,
evaluation, explainability, and policy design — not a claim about expected
real-world model performance. See docs/12 for the full discussion.

## Tech Stack
- **Data / Analysis:** Python, pandas, NumPy
- **ML:** XGBoost, LightGBM, scikit-learn, SHAP
- **SQL / Pipeline design:** BigQuery (partitioning, clustering, incremental merge)
- **Evaluation:** precision-recall analysis, cost-based threshold optimization, walk-forward back-testing

## Repository Structure
\`\`\`
├── docs/              # Stage-by-stage design docs (business → deployment), 01-16
├── data/
│   ├── raw/            # Synthetic seller/listing/transaction/payout data
│   └── processed/       # Engineered features, model outputs, policy simulation
├── notebooks/
├── src/
│   ├── data_generation/
│   ├── feature_engineering/
│   ├── eda/
│   ├── clustering/
│   ├── growth_monitoring/
│   ├── modeling/
│   ├── evaluation/
│   ├── explainability/
│   └── policy/
├── sql/                # BigQuery pipeline design (docs/16)
├── tests/
└── README.md
\`\`\`

