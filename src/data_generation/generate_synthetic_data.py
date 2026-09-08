"""
Synthetic seller/marketplace dataset generator — Seller Fraud Detection System (Project 1)

Generates a realistic-shaped synthetic dataset covering the seller lifecycle
(Stage 0-3 per docs/04_seller_lifecycle.md) with fraud typologies injected
per docs/03_fraud_motivation.md. No real data of any kind is used.

Output: data/raw/sellers.csv, listings.csv, transactions.csv, payouts.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_SELLERS = 5000
OBS_END = datetime(2026, 1, 1)          # fixed "today" for reproducibility
OBS_WINDOW_DAYS = 365                    # sellers registered within the last year
LIFECYCLE_HORIZON_DAYS = 90              # per Section-1 scope: early lifecycle only

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Beauty", "Toys",
    "Sports", "Books", "Grocery", "Jewelry", "Automotive Parts",
]

FRAUD_TYPES = [
    "none", "bust_out", "stolen_goods", "triangulation",
    "account_takeover", "collusion", "synthetic_identity",
]
FRAUD_TYPE_WEIGHTS = [0.92, 0.025, 0.02, 0.015, 0.008, 0.007, 0.005]


def random_dates(start: datetime, n: int, window_days: int) -> np.ndarray:
    offsets = RNG.integers(0, window_days, size=n)
    return np.array([start + timedelta(days=int(o)) for o in offsets])


def build_sellers(n: int) -> pd.DataFrame:
    obs_start = OBS_END - timedelta(days=OBS_WINDOW_DAYS)
    registration_date = random_dates(obs_start, n, OBS_WINDOW_DAYS)

    fraud_type = RNG.choice(FRAUD_TYPES, size=n, p=FRAUD_TYPE_WEIGHTS)
    is_fraud = fraud_type != "none"

    category = RNG.choice(CATEGORIES, size=n)

    identity_consistency = np.clip(RNG.normal(0.85, 0.10, size=n), 0, 1)
    identity_consistency[fraud_type == "synthetic_identity"] = np.clip(
        RNG.normal(0.30, 0.12, size=(fraud_type == "synthetic_identity").sum()), 0, 1
    )
    identity_consistency[fraud_type == "collusion"] = np.clip(
        RNG.normal(0.55, 0.15, size=(fraud_type == "collusion").sum()), 0, 1
    )

    email_domain_age_days = RNG.integers(1, 3000, size=n)
    email_domain_age_days[fraud_type == "synthetic_identity"] = RNG.integers(
        0, 14, size=(fraud_type == "synthetic_identity").sum()
    )

    df = pd.DataFrame({
        "seller_id": [f"S{i:06d}" for i in range(n)],
        "registration_date": registration_date,
        "business_category": category,
        "identity_consistency_score": identity_consistency.round(3),
        "email_domain_age_days": email_domain_age_days,
        "device_id": [f"D{RNG.integers(0, n * 2):06d}" for _ in range(n)],
        "ip_hash": [f"IP{RNG.integers(0, n * 2):06d}" for _ in range(n)],
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
    })
    return df


def build_listings(sellers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    listing_id = 0
    for _, s in sellers.iterrows():
        if s["fraud_type"] == "bust_out":
            n_listings = RNG.integers(3, 8)
        elif s["fraud_type"] in ("stolen_goods", "triangulation"):
            n_listings = RNG.integers(10, 30)
        else:
            n_listings = RNG.integers(1, 15)

        for _ in range(n_listings):
            day_offset = RNG.integers(0, min(LIFECYCLE_HORIZON_DAYS, 30))
            listed_date = s["registration_date"] + timedelta(days=int(day_offset))

            if s["fraud_type"] in ("stolen_goods", "triangulation"):
                price = RNG.choice([49.99, 99.99, 199.99, 499.99, 999.99],
                                    p=[0.1, 0.2, 0.3, 0.25, 0.15])
                category = RNG.choice(["Electronics", "Jewelry"], p=[0.7, 0.3])
            else:
                price = round(RNG.gamma(shape=2.0, scale=25.0) + 5, 2)
                category = s["business_category"]

            rows.append({
                "listing_id": f"L{listing_id:07d}",
                "seller_id": s["seller_id"],
                "listed_date": listed_date,
                "category": category,
                "price": price,
            })
            listing_id += 1
    return pd.DataFrame(rows)


def build_transactions(sellers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    txn_id = 0
    n_sellers = len(sellers)
    for _, s in sellers.iterrows():
        reg = s["registration_date"]
        ftype = s["fraud_type"]

        if ftype == "bust_out":
            quiet_days = RNG.integers(10, 25)
            n_quiet_txns = RNG.integers(2, 6)
            for _ in range(n_quiet_txns):
                offset = RNG.integers(0, quiet_days)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=1.0, n_sellers=n_sellers))
                txn_id += 1
            spike_day = quiet_days + RNG.integers(1, 5)
            n_spike_txns = RNG.integers(8, 20)
            for _ in range(n_spike_txns):
                offset = spike_day + RNG.integers(0, 3)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=6.0, n_sellers=n_sellers))
                txn_id += 1

        elif ftype == "collusion":
            n_txns = RNG.integers(15, 40)
            buyer_pool = [f"B_COL_{s['seller_id']}_{k}" for k in range(RNG.integers(2, 5))]
            for _ in range(n_txns):
                offset = RNG.integers(0, LIFECYCLE_HORIZON_DAYS)
                row = _txn_row(txn_id, s, reg, offset, amount_scale=1.2, n_sellers=n_sellers)
                row["buyer_id"] = RNG.choice(buyer_pool)
                rows.append(row)
                txn_id += 1

        elif ftype in ("triangulation", "stolen_goods"):
            n_txns = RNG.integers(5, 15)
            for _ in range(n_txns):
                offset = RNG.integers(0, LIFECYCLE_HORIZON_DAYS)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=2.5, n_sellers=n_sellers))
                txn_id += 1

        elif ftype == "account_takeover":
            n_normal = RNG.integers(10, 25)
            for _ in range(n_normal):
                offset = RNG.integers(0, LIFECYCLE_HORIZON_DAYS - 15)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=1.0, n_sellers=n_sellers))
                txn_id += 1
            n_ato_spike = RNG.integers(5, 12)
            for _ in range(n_ato_spike):
                offset = LIFECYCLE_HORIZON_DAYS - RNG.integers(1, 10)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=5.0, n_sellers=n_sellers))
                txn_id += 1

        else:
            n_txns = RNG.integers(0, 30) if ftype == "none" else RNG.integers(0, 5)
            for _ in range(n_txns):
                offset = RNG.integers(0, LIFECYCLE_HORIZON_DAYS)
                rows.append(_txn_row(txn_id, s, reg, offset, amount_scale=1.0, n_sellers=n_sellers))
                txn_id += 1

    return pd.DataFrame(rows)


def _txn_row(txn_id, seller_row, reg_date, day_offset, amount_scale, n_sellers):
    amount = round(max(5.0, RNG.gamma(shape=2.0, scale=20.0) * amount_scale), 2)
    return {
        "transaction_id": f"T{txn_id:08d}",
        "seller_id": seller_row["seller_id"],
        "buyer_id": f"B{RNG.integers(0, n_sellers * 3):07d}",
        "transaction_date": reg_date + timedelta(days=int(day_offset)),
        "amount": amount,
    }


def build_payouts(sellers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    payout_id = 0
    txn_by_seller = transactions.groupby("seller_id")
    for _, s in sellers.iterrows():
        if s["seller_id"] not in txn_by_seller.groups:
            continue
        seller_txns = txn_by_seller.get_group(s["seller_id"]).sort_values("transaction_date")
        first_txn_date = seller_txns["transaction_date"].iloc[0]
        total_revenue = seller_txns["amount"].sum()

        if s["fraud_type"] in ("bust_out", "account_takeover"):
            lag_days = RNG.integers(0, 3)
        elif s["fraud_type"] == "none":
            lag_days = RNG.integers(7, 30)
        else:
            lag_days = RNG.integers(1, 10)

        payout_date = seller_txns["transaction_date"].iloc[-1] + timedelta(days=int(lag_days))
        rows.append({
            "payout_id": f"P{payout_id:06d}",
            "seller_id": s["seller_id"],
            "payout_date": payout_date,
            "amount": round(total_revenue * RNG.uniform(0.7, 0.95), 2),
            "days_since_first_transaction": (payout_date - first_txn_date).days,
        })
        payout_id += 1
    return pd.DataFrame(rows)


def main():
    sellers = build_sellers(N_SELLERS)
    listings = build_listings(sellers)
    transactions = build_transactions(sellers)
    payouts = build_payouts(sellers, transactions)

    sellers.to_csv("data/raw/sellers.csv", index=False)
    listings.to_csv("data/raw/listings.csv", index=False)
    transactions.to_csv("data/raw/transactions.csv", index=False)
    payouts.to_csv("data/raw/payouts.csv", index=False)

    print(f"sellers: {len(sellers)} rows, fraud rate: {sellers['is_fraud'].mean():.2%}")
    print(sellers["fraud_type"].value_counts())
    print(f"listings: {len(listings)} rows")
    print(f"transactions: {len(transactions)} rows")
    print(f"payouts: {len(payouts)} rows")


if __name__ == "__main__":
    main()