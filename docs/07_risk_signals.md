# Risk Signal Design

## Signal vs. Feature
A signal is a conceptual, measurable quantity tied to a specific business
question. A feature is its concrete, model-ready encoding. Keeping these
separate means every feature (topic 8) traces back to a named business
justification here, rather than being invented ad hoc from the data.

## Signal Catalog

| Signal | Measures | Source | Available from | Targets | Direction |
|---|---|---|---|---|---|
| identity_consistency_score | Internal consistency of registration identity | sellers | Stage 0 | Synthetic identity | Lower = riskier |
| email_domain_age_days | Age of seller's email domain at registration | sellers | Stage 0 | Synthetic identity | Lower = riskier |
| listing_velocity_24h | Listings/hour in first 24h | listings | Stage 1 | Bust-out | Higher = riskier |
| txn_count_first_72h | Transaction count in first 72h vs. category baseline | transactions | Stage 2 | Bust-out | Higher (relative) = riskier |
| category_entropy | Diversity/coherence of listed categories | listings | Stage 1 | Stolen goods, triangulation | Lower + high-ticket concentration = riskier |
| high_ticket_item_ratio | Share of listings in high-ticket price bands | listings | Stage 1 | Stolen goods, triangulation | Higher = riskier |
| top5_buyer_revenue_share | Revenue concentration among top 5 buyers | transactions | Stage 2 | Collusion | Higher = riskier |
| buyer_seller_device_overlap | Device/IP overlap between seller and buyers | sellers, transactions | Stage 2 | Collusion | Any overlap = riskier |
| payout_lag_days | Days from last transaction to payout request | transactions, payouts | Stage 2-3 | Bust-out, account takeover | Lower = riskier |
| behavior_change_index | Deviation from seller's own historical baseline | transactions (time series) | Stage 3 | Account takeover | Higher = riskier |

## Design Rules
1. Every signal is stage-tagged with the earliest point it is computable.
2. Risk direction is stated explicitly (not all signals are "higher = riskier").
3. Every signal traces to a named investigator question (docs/06) and typology (docs/03).
4. Baseline-relative signals are flagged as needing a comparison population,
   to be handled explicitly during feature engineering (docs/08).
