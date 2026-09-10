# Policy Design & Deployment Simulation

## Multi-Tier Policy
Proportional response, replacing docs/13's single 0.39 threshold with four
tiers matching intervention severity to model confidence:
- AUTO_BLOCK (score >= 0.90): automated hard decline, no review cost
- MANUAL_REVIEW (0.39-0.90): sent to Ops review queue ($50/case)
- SOFT_FRICTION (0.10-0.39): extra verification, seller can still proceed
  ($10/case for false positives; assumed 50% fraud deterrence -- a stated
  placeholder pending real A/B validation, not a measured fact)
- MONITOR (<0.10): approved, passively logged

## Results
| Tier | n | fraud_rate | cost |
|---|---|---|---|
| AUTO_BLOCK | 57 | 100.0% | $0.00 |
| MANUAL_REVIEW | 9 | 44.4% | $250.00 |
| SOFT_FRICTION | 21 | 9.5% | $1,223.44 |
| MONITOR | 922 | 0.1% | $0.00 |

Total policy cost: $1,473.43 vs. $2,316.87 for the single-threshold policy
(36% reduction) -- driven by removing review cost from the 57 high-
confidence cases the model can safely auto-block. Fraud fully intercepted
(AUTO_BLOCK + MANUAL_REVIEW): 95.3%. Approval rate (no more than soft
friction): 93.5%.

**Caveat on AUTO_BLOCK:** 0% observed false-positive rate on only 57
samples is consistent with a true rate up to ~5-6% -- needs validation on
much larger volume, and likely an audit/appeal path, before trusting full
automation on an irreversible action in real production.

## Deployment Simulation (walk-forward, 4 windows)
| Window | Fraud caught | Approval rate |
|---|---|---|
| 1 | 90.7% | 93.6% |
| 2 | 94.7% | 93.8% |
| 3 | 94.2% | 91.6% |
| 4 | 95.2% | 93.0% |

Both metrics stay in a tight range across time -- the specific stability
check a deployment sign-off review would look for before greenlighting a
policy, beyond a single holdout AUC.

## Next Step Not Implemented Here: Champion-Challenger
Real production deployment would run this policy in shadow mode alongside
the existing policy for a burn-in period before cutover, comparing
would-have decisions against actual outcomes -- standard practice for any
new fraud policy, not implemented here (an engineering system, not a data
science deliverable).
