# 8 — Feature Engineering

## Point-in-Time Discipline
Every feature is computed using only events within 90 days of a seller's own
registration date (per-seller relative cutoff, not a global calendar cutoff).
This prevents temporal leakage — using information that would not have
existed yet at the point a real decision needed to be made.

## Features Computed
| Feature | Method | Source |
|---|---|---|
| listing_velocity_24h | Listings/hour in first 24h | listings |
| category_entropy | Shannon entropy of category distribution | listings |
| high_ticket_item_ratio | Share of listings above a price threshold | listings |
| txn_count_first_72h | Raw transaction count in first 72h | transactions |
| txn_count_first_72h_zscore | Category-baseline-normalized version of above | transactions |
| top5_buyer_revenue_share | Revenue concentration among top 5 buyers | transactions |
| payout_lag_days | Days from last transaction to payout | transactions, payouts |
| behavior_change_index | % change in avg transaction amount, first half vs second half | transactions |
| buyer_seller_device_overlap | Not computed — requires buyer-side device data not yet in the generator (documented gap, deferred) | n/a |

## Result
5,000-row feature table with clear separation by fraud typology, e.g.
top5_buyer_revenue_share averaging 1.0 for collusion vs ~0.6-0.77 for
other types; category_entropy near 0 for legitimate/coherent sellers vs
~0.8 for stolen-goods/triangulation sellers; payout_lag_days averaging
~19 days for bust-out vs ~89 days for legitimate sellers.
