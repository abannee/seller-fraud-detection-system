"""
Feature engineering — Seller Fraud Detection System (Project 1)

Computes the signal catalog from docs/07_risk_signals.md as concrete,
model-ready features. Point-in-time discipline: every feature is computed
using only data that would have existed by a fixed decision horizon
(90 days post-registration), never using data from after that point.

Input:  data/raw/sellers.csv, listings.csv, transactions.csv, payouts.csv
Output: data/processed/seller_features.csv
"""

import numpy as np
import pandas as pd

DECISION_HORIZON_DAYS = 90  # matches LIFECYCLE_HORIZON_DAYS in the generator


def shannon_entropy(counts: pd.Series) -> float:
    """Shannon entropy of a category distribution, base-2. 0 = single category
    (fully concentrated), higher = more spread across categories."""
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def load_data():
    sellers = pd.read_csv("data/raw/sellers.csv", parse_dates=["registration_date"])
    listings = pd.read_csv("data/raw/listings.csv", parse_dates=["listed_date"])
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["transaction_date"])
    payouts = pd.read_csv("data/raw/payouts.csv", parse_dates=["payout_date"])
    return sellers, listings, transactions, payouts


def clip_to_horizon(df, seller_date_col, event_date_col, sellers):
    """Keep only events within DECISION_HORIZON_DAYS of each seller's
    registration date — this is the point-in-time cut that prevents leakage."""
    merged = df.merge(sellers[["seller_id", "registration_date"]], on="seller_id", how="left")
    days_since_reg = (merged[event_date_col] - merged["registration_date"]).dt.days
    return merged[(days_since_reg >= 0) & (days_since_reg <= DECISION_HORIZON_DAYS)].copy()


def build_features(sellers, listings, transactions, payouts):
    listings_c = clip_to_horizon(listings, "registration_date", "listed_date", sellers)
    txns_c = clip_to_horizon(transactions, "registration_date", "transaction_date", sellers)

    feats = sellers[["seller_id", "registration_date", "business_category",
                      "identity_consistency_score", "email_domain_age_days",
                      "is_fraud", "fraud_type"]].copy()

    # --- listing_velocity_24h: listings per hour in first 24h ---
    listings_c["hours_since_reg"] = (
        (listings_c["listed_date"] - listings_c["registration_date"]).dt.total_seconds() / 3600
    )
    first_24h = listings_c[listings_c["hours_since_reg"] <= 24]
    velocity = first_24h.groupby("seller_id").size() / 24
    feats["listing_velocity_24h"] = feats["seller_id"].map(velocity).fillna(0.0)

    # --- category_entropy + high_ticket_item_ratio (from listings) ---
    entropy = listings_c.groupby("seller_id")["category"].apply(
        lambda s: shannon_entropy(s.value_counts())
    )
    feats["category_entropy"] = feats["seller_id"].map(entropy).fillna(0.0)

    HIGH_TICKET_THRESHOLD = 199.99
    high_ticket_ratio = listings_c.groupby("seller_id").apply(
        lambda g: (g["price"] >= HIGH_TICKET_THRESHOLD).mean(), include_groups=False
    )
    feats["high_ticket_item_ratio"] = feats["seller_id"].map(high_ticket_ratio).fillna(0.0)

    # --- txn_count_first_72h + baseline-relative z-score ---
    txns_c["hours_since_reg"] = (
        (txns_c["transaction_date"] - txns_c["registration_date"]).dt.total_seconds() / 3600
    )
    first_72h = txns_c[txns_c["hours_since_reg"] <= 72]
    txn_count_72h = first_72h.groupby("seller_id").size()
    feats["txn_count_first_72h"] = feats["seller_id"].map(txn_count_72h).fillna(0).astype(int)

    # Baseline: category-level mean/std of txn_count_first_72h, computed only
    # from the feature table itself (still point-in-time safe — it's a
    # population statistic, not a future individual observation).
    cat_baseline = feats.groupby("business_category")["txn_count_first_72h"].agg(["mean", "std"])
    cat_baseline["std"] = cat_baseline["std"].replace(0, np.nan)
    feats = feats.merge(cat_baseline, left_on="business_category", right_index=True, how="left")
    feats["txn_count_first_72h_zscore"] = (
        (feats["txn_count_first_72h"] - feats["mean"]) / feats["std"]
    ).fillna(0.0)
    feats = feats.drop(columns=["mean", "std"])

    # --- top5_buyer_revenue_share ---
    def top5_share(g):
        rev_by_buyer = g.groupby("buyer_id")["amount"].sum().sort_values(ascending=False)
        total = rev_by_buyer.sum()
        if total == 0:
            return 0.0
        return rev_by_buyer.head(5).sum() / total

    top5 = txns_c.groupby("seller_id").apply(top5_share, include_groups=False)
    feats["top5_buyer_revenue_share"] = feats["seller_id"].map(top5).fillna(0.0)

    # --- payout_lag_days (already computed at generation time) ---
    payout_lag = payouts.set_index("seller_id")["days_since_first_transaction"]
    feats["payout_lag_days"] = feats["seller_id"].map(payout_lag)  # NaN if no payout yet

    # --- behavior_change_index: % change in avg txn amount, first half vs second half ---
    def behavior_change(g):
        g = g.sort_values("transaction_date")
        n = len(g)
        if n < 4:
            return 0.0
        mid = n // 2
        first_half_mean = g["amount"].iloc[:mid].mean()
        second_half_mean = g["amount"].iloc[mid:].mean()
        if first_half_mean == 0:
            return 0.0
        return (second_half_mean - first_half_mean) / first_half_mean

    change_idx = txns_c.groupby("seller_id").apply(behavior_change, include_groups=False)
    feats["behavior_change_index"] = feats["seller_id"].map(change_idx).fillna(0.0)

    # --- buyer_seller_device_overlap: NOT COMPUTED — data gap ---
    # The synthetic dataset (topic 5) does not yet generate buyer-side device/IP
    # fingerprints, only seller-side. This signal from the topic-7 catalog is
    # deferred until the generator is extended with buyer device data — logged
    # here explicitly rather than faked, and column added as NaN placeholder.
    feats["buyer_seller_device_overlap"] = np.nan

    return feats


def main():
    sellers, listings, transactions, payouts = load_data()
    feats = build_features(sellers, listings, transactions, payouts)
    feats.to_csv("data/processed/seller_features.csv", index=False)

    print(f"Feature table: {feats.shape[0]} rows, {feats.shape[1]} columns")
    print(feats.groupby("fraud_type")[[
        "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
        "txn_count_first_72h_zscore", "top5_buyer_revenue_share",
        "payout_lag_days", "behavior_change_index"
    ]].mean().round(3))


if __name__ == "__main__":
    main()