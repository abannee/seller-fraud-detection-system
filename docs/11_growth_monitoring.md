# Growth-Monitoring Model

## Why a Separate Model
Account-takeover fraud is rare (32/5,000 sellers, 0.64%) and relies on a
single point-in-time feature (behavior_change_index, docs/08) despite
forming a genuinely separable cluster (docs/10). A shared classifier would
let this rare pattern get diluted by the majority of easier-to-separate
fraud types. Growth monitoring is reframed as a time-series anomaly
detection problem (trajectory shape) rather than a point-in-time
classification problem, justifying a dedicated model.

## Method
- Peer-baseline percentile rank: for each day-since-registration, compute
  each seller's percentile rank of cumulative revenue against legitimate
  peers at the same lifecycle day (robust to the right-skew typical of
  revenue distributions, unlike a raw z-score).
- Anomaly signal: rolling 7-day change in percentile rank. Large jumps
  (e.g. 40th -> 95th percentile within a week) are the structural-break
  signature shared by both bust-out and account-takeover fraud.

## Result
A single trajectory-based signal generalizes across two typologies that
previously needed separate narrow features (payout_lag_days for bust-out,
behavior_change_index for account-takeover) — evaluated against the
existing labels for validation.
