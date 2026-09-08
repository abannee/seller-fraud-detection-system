# Data Collection & Synthetic Data Design

## Schema
| Table | Populated from | Lifecycle stage |
|---|---|---|
| sellers.csv | Registration/identity data, device/IP, category | Stage 0 |
| listings.csv | Catalog data: items, pricing, timing | Stage 1 |
| transactions.csv | Buyer interactions, velocity, value | Stage 2 |
| payouts.csv | Cash-out requests and timing | Stage 2-3 |

## Typology-to-Behavior Encoding
- Bust-out: quiet early activity, sharp value spike, immediate payout
- Stolen goods / triangulation: larger catalogs skewed toward high-ticket
  categories at prices clustered under round-number thresholds
- Account takeover: long normal history, abrupt late-window spike
- Collusion: transaction volume concentrated among a small repeating buyer pool
- Synthetic identity: low identity-consistency score, fresh email domain,
  typically low real transaction volume

## Result
5,000 synthetic sellers generated, ~7.1% fraud rate, across all six typologies
from docs/03_fraud_motivation.md. Fixed random seed (42) for reproducibility.
No real seller, buyer, or PII data used anywhere in this dataset.
