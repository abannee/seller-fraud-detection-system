"""
EDA & Root-Cause Analysis — Seller Fraud Detection System (Project 1)

Explores the engineered feature table (docs/08) to:
1. Validate that fraud typologies actually show up as separable patterns
   (justifying the signal design in docs/07 before any model is trained)
2. Identify outliers and distributional quirks that could break a naive model
3. Perform root-cause analysis connecting statistical patterns back to
   specific, named control gaps

Input:  data/processed/seller_features.csv
Output: printed report + data/processed/eda_outlier_flags.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

NUMERIC_FEATURES = [
    "identity_consistency_score", "email_domain_age_days",
    "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
    "txn_count_first_72h", "txn_count_first_72h_zscore",
    "top5_buyer_revenue_share", "payout_lag_days", "behavior_change_index",
]


def univariate_by_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median per feature, split fraud vs. non-fraud — the first, coarsest
    check: does this feature even move in the expected direction?"""
    return df.groupby("is_fraud")[NUMERIC_FEATURES].agg(["mean", "median"]).round(3)


def univariate_by_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Same, but split by typology — checks whether a feature is only useful
    for ONE fraud type (masked when lumped into a single fraud/not-fraud split)."""
    return df.groupby("fraud_type")[NUMERIC_FEATURES].mean().round(3)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation among engineered features — flags redundant
    features (near-duplicate signal) before they're fed into a model."""
    return df[NUMERIC_FEATURES].corr().round(2)


def iqr_outliers(df: pd.DataFrame, col: str) -> pd.Series:
    """Classic IQR outlier rule: flag points beyond 1.5x the interquartile
    range from Q1/Q3. Returns a boolean mask."""
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, what fraction of outliers (by IQR) are actually
    fraud vs. legitimate — this is the root-cause-relevant question: are
    outliers concentrated in fraud, or are they just noisy legit sellers?"""
    rows = []
    for col in NUMERIC_FEATURES:
        mask = iqr_outliers(df.dropna(subset=[col]), col)
        outlier_idx = df.dropna(subset=[col])[mask].index
        n_outliers = len(outlier_idx)
        if n_outliers == 0:
            continue
        fraud_rate_in_outliers = df.loc[outlier_idx, "is_fraud"].mean()
        overall_fraud_rate = df["is_fraud"].mean()
        rows.append({
            "feature": col,
            "n_outliers": n_outliers,
            "fraud_rate_in_outliers": round(fraud_rate_in_outliers, 3),
            "overall_fraud_rate": round(overall_fraud_rate, 3),
            "lift": round(fraud_rate_in_outliers / overall_fraud_rate, 2) if overall_fraud_rate > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def main():
    df = pd.read_csv("data/processed/seller_features.csv")

    print("=== Univariate: fraud vs. non-fraud (mean/median) ===")
    print(univariate_by_fraud(df))

    print("\n=== Univariate: by fraud typology (mean) ===")
    print(univariate_by_typology(df))

    print("\n=== Correlation matrix (engineered features) ===")
    print(correlation_matrix(df))

    print("\n=== Outlier report: IQR-flagged points, fraud lift ===")
    report = outlier_report(df)
    print(report.to_string(index=False))

    report.to_csv("data/processed/eda_outlier_flags.csv", index=False)


if __name__ == "__main__":
    main()