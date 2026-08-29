# Seller Fraud Detection System
### Early-lifecycle fraud strategy & ML for a PayPal/marketplace-style seller risk program

## Overview
This project simulates an industry-grade seller fraud detection system for a payment
facilitator / online marketplace (PayPal, Amazon Payments, Stripe-style seller risk).
It is built as a **production simulation**, not a Kaggle competition — the goal is to
reproduce the end-to-end thinking, tooling, and trade-offs of a real Seller Risk /
Fraud Strategy team, from business problem to deployed decisioning logic.

The project is inspired by ~10 years of hands-on experience in seller risk strategy,
fraud modeling, and AML/transaction analytics across marketplace and payments
businesses. No proprietary data, models, or code from any employer is used here —
all data is synthetic or public, and all logic is rebuilt from first principles.

## Business Problem
Marketplaces and payment facilitators onboard sellers continuously, and a meaningful
share of losses originate in the **early lifecycle** — the window before a seller has enough transaction history for standard risk models to work well.             

The objective:

> Identify fraudulent sellers as early as possible, while minimizing false positives
> and friction for genuine low-risk merchants.

This is a precision–recall and operational-cost balancing problem as much as a
modeling problem — every extra fraud dollar caught has to be weighed against
reviewer capacity, merchant experience, and revenue impact.

## Approach
The project follows a deliberate sequence rather than jumping straight to modeling:

**Business → Fraud Behaviour → Observable Signal → Business Feature → ML Feature → Model → Deployment**

1. **Business & fraud understanding** — mapped the seller lifecycle, fraud
   motivations, and where existing controls have gaps.
2. **Risk signal design** — translated fraud behaviours into observable, computable
   signals available at or near seller Day 0.
3. **Feature engineering** — built seller behavioral and transactional features
   (rolling time-window aggregations, event-level features) from early-lifecycle
   activity to surface incremental fraud signal beyond baseline controls.
4. **Root-cause analysis + unsupervised clustering** — used transaction-level RCA
   combined with clustering to find recurring seller behaviour patterns and isolate
   systemic control gaps.
5. **Growth-monitoring model** — a dedicated model to flag abnormal seller revenue
   acceleration as an early/emerging risk pattern, enabling intervention before losses
   compound.
6. **Modeling** — XGBoost and LightGBM models trained on early-lifecycle features,
   evaluated with precision–recall analysis, threshold optimization, and controlled
   back-testing against existing production-style controls.
7. **Policy design** — validated insights translated into production-style policies
   and automated real-time seller risk actions, not just model scores in a vacuum.
8. **Pipeline engineering** — a ~3M+ record daily analytics pipeline in BigQuery,
   optimized via query restructuring, table partitioning, clustering, and incremental
   daily loads (~51% runtime reduction), so features are available on time for
   downstream decisioning and ML.

## Tech Stack
- **Data / SQL:** BigQuery (partitioning, clustering, incremental loads)
- **ML:** Python, XGBoost, LightGBM, scikit-learn, SHAP
- **Analysis:** pandas, feature engineering (rolling windows, event-level aggregation),
  unsupervised clustering
- **Evaluation:** precision–recall analysis, threshold optimization, back-testing

## Project Status
This repo is being built and documented in stages (business understanding → fraud
motivation → seller lifecycle → data collection → risk signals → feature engineering →
investigator-style hypothesis generation → modeling → deployment simulation). See
`/docs` for the stage-by-stage write-up. Results and model performance numbers will
be added here as those stages are completed.

## Repository Structure
