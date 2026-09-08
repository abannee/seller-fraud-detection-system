# Project Vision

## Core Vision
This project simulates an industry-grade seller fraud detection system for a
payment facilitator / online marketplace (PayPal, Amazon Payments, Stripe-style
seller risk). It is built as a production simulation, not a Kaggle competition —
the goal is to reproduce the end-to-end thinking, tooling, and trade-offs of a
real Seller Risk / Fraud Strategy team, from business problem to deployed
decisioning logic.

## What This Project Demonstrates
The objective was not to maximize a model's AUC on a static dataset. It was to
simulate the full loop of an early-lifecycle seller risk program:
understanding why sellers commit fraud, designing signals that are actually
available at the right point in the seller lifecycle, engineering features from
those signals, building and evaluating models against precision-recall and
operational cost trade-offs, and translating results into policies a real risk
team could act on — including the data pipeline that would serve it in
production.

## Scope
**In scope:**
- Early-lifecycle seller fraud (Day 0 to ~90 days)
- Full loop: business understanding through deployment simulation

**Out of scope:**
- Real production infrastructure (BigQuery/SQL simulation, not a live system)
- Real seller/PII data (synthetic dataset only, built in Day 5)
- Established-seller fraud and buyer-side fraud
