"""
Growth-Monitoring Model — Seller Fraud Detection System (Project 1)

Tracks each seller's cumulative revenue trajectory against a peer baseline
(legitimate sellers only) and flags large percentile-rank jumps as a
structural-break anomaly signal -- generalizing across bust-out and
account-takeover typologies, which topics 8-10 covered with separate,
narrower point-in-time features.

Input:  data/raw/sellers.csv, data/raw/transactions.csv
Output: data/processed/growth_monitoring_features.csv
"""

import numpy as np
import pandas as pd

DECISION_HORIZON_DAYS = 90
ROLLING_WINDOW_DAYS = 7


def load_data():
    sellers = pd.read_csv("data/raw/sellers.csv", parse_dates=["registration_date"])
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["transaction_date"])
    return sellers, transactions


def build_daily_cumulative_revenue(sellers, transactions):
    """One row per (seller_id, day_since_reg) with cumulative revenue to that day."""
    txns = transactions.merge(sellers[["seller_id", "registration_date"]], on="seller_id")
    txns["day_since_reg"] = (txns["transaction_date"] - txns["registration_date"]).dt.days
    txns = txns[(txns["day_since_reg"] >= 0) & (txns["day_since_reg"] <= DECISION_HORIZON_DAYS)]

    # Full grid: every seller x every day 0..90, forward-filled cumulative revenue.
    daily = txns.groupby(["seller_id", "day_since_reg"])["amount"].sum().reset_index()
    all_sellers = sellers["seller_id"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_sellers, range(DECISION_HORIZON_DAYS + 1)], names=["seller_id", "day_since_reg"]
    )
    daily_full = daily.set_index(["seller_id", "day_since_reg"]).reindex(full_index, fill_value=0.0)
    daily_full = daily_full.reset_index()
    daily_full["cumulative_revenue"] = daily_full.groupby("seller_id")["amount"].cumsum()
    return daily_full


def compute_peer_percentiles(daily_full, sellers):
    """For each day_since_reg, rank every seller's cumulative revenue against
    the LEGITIMATE peer population only at that same day -- fraud sellers
    should not inform the baseline they're being compared against."""
    legit_ids = set(sellers.loc[~sellers["is_fraud"], "seller_id"])
    daily_full["is_legit_peer"] = daily_full["seller_id"].isin(legit_ids)

    def percentile_rank(group):
        peer_values = group.loc[group["is_legit_peer"], "cumulative_revenue"]
        if len(peer_values) < 5:
            return pd.Series(0.5, index=group.index)
        return group["cumulative_revenue"].apply(
            lambda x: (peer_values < x).mean()
        )

    daily_full["percentile_rank"] = daily_full.groupby("day_since_reg", group_keys=False).apply(
        percentile_rank
    )
    return daily_full


def compute_max_percentile_jump(daily_full):
    """Rolling 7-day change in percentile rank, per seller; take the max
    jump observed anywhere in the seller's first 90 days -- the anomaly signal."""
    daily_full = daily_full.sort_values(["seller_id", "day_since_reg"])
    daily_full["percentile_jump_7d"] = (
        daily_full.groupby("seller_id")["percentile_rank"].diff(ROLLING_WINDOW_DAYS)
    )
    max_jump = daily_full.groupby("seller_id")["percentile_jump_7d"].max()
    return max_jump.rename("max_percentile_jump_7d")


def main():
    sellers, transactions = load_data()
    daily_full = build_daily_cumulative_revenue(sellers, transactions)
    daily_full = compute_peer_percentiles(daily_full, sellers)
    max_jump = compute_max_percentile_jump(daily_full)

    result = sellers[["seller_id", "is_fraud", "fraud_type"]].merge(
        max_jump, on="seller_id", how="left"
    )
    result["max_percentile_jump_7d"] = result["max_percentile_jump_7d"].fillna(0.0)
    result.to_csv("data/processed/growth_monitoring_features.csv", index=False)

    print("=== Max 7-day percentile jump, by fraud type ===")
    print(result.groupby("fraud_type")["max_percentile_jump_7d"].mean().round(3))


if __name__ == "__main__":
    main()