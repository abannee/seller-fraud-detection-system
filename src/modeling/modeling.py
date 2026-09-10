"""
Modeling (XGBoost / LightGBM) — Seller Fraud Detection System (Project 1)

Trains gradient-boosted models on the engineered features (docs/08) plus
the growth-monitoring signal (docs/11), using a CHRONOLOGICAL train/test
split (not random) to simulate real deployment: train on earlier-registered
sellers, evaluate on later-registered sellers the model has never seen.

Also computes a naive rule-based baseline standing in for "existing
production controls," to measure incremental lift from ML -- directly
reflecting the original project framing (uncovering incremental fraud
signal beyond existing controls).

Input:  data/processed/seller_features.csv, growth_monitoring_features.csv
Output: printed metrics report + data/processed/model_predictions.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

FEATURE_COLUMNS = [
    "identity_consistency_score", "email_domain_age_days",
    "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
    "txn_count_first_72h", "txn_count_first_72h_zscore",
    "top5_buyer_revenue_share", "payout_lag_days", "behavior_change_index",
    "max_percentile_jump_7d",
]


def load_and_merge():
    feats = pd.read_csv("data/processed/seller_features.csv", parse_dates=["registration_date"])
    growth = pd.read_csv("data/processed/growth_monitoring_features.csv")
    df = feats.merge(growth[["seller_id", "max_percentile_jump_7d"]], on="seller_id", how="left")
    # payout_lag_days is NaN for sellers with no payout -- for modeling, encode
    # "no payout yet" as a large sentinel value (low risk on this dimension)
    # rather than imputing with the mean, which would fabricate a mid-risk signal.
    df["payout_lag_days"] = df["payout_lag_days"].fillna(999)
    return df


def naive_production_rule(df: pd.DataFrame) -> pd.Series:
    """Stand-in for a simple existing rule-based control: flags sellers who
    trip any one of a few obvious, individually-interpretable thresholds.
    This is intentionally crude -- it's the baseline the ML model needs to
    beat, not a strawman."""
    return (
        (df["payout_lag_days"] <= 2)
        | (df["high_ticket_item_ratio"] >= 0.5)
        | (df["top5_buyer_revenue_share"] >= 0.9)
    )


def chronological_split(df: pd.DataFrame, train_frac=0.8):
    cutoff = df["registration_date"].quantile(train_frac)
    train = df[df["registration_date"] < cutoff].copy()
    test = df[df["registration_date"] >= cutoff].copy()
    return train, test, cutoff


def evaluate(name, y_true, y_score):
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    print(f"  {name}: ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
          f"(base rate={y_true.mean():.4f})")
    return roc_auc, pr_auc


def main():
    df = load_and_merge()
    train, test, cutoff = chronological_split(df)
    print(f"Chronological split at {cutoff.date()}: "
          f"train={len(train)} sellers, test={len(test)} sellers\n")
    print(f"Train fraud rate: {train['is_fraud'].mean():.4f}  "
          f"Test fraud rate: {test['is_fraud'].mean():.4f}\n")

    X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"].astype(int)

    # --- Baseline: naive production rule ---
    rule_flag_test = naive_production_rule(test)
    rule_precision = precision_score(y_test, rule_flag_test)
    rule_recall = recall_score(y_test, rule_flag_test)
    print("=== Baseline: naive rule-based production control ===")
    print(f"  precision={rule_precision:.4f}  recall={rule_recall:.4f}  "
          f"flags={rule_flag_test.sum()} of {len(test)} sellers\n")

    # --- XGBoost ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_scores = lgb_model.predict_proba(X_test)[:, 1]

    print("=== Model performance (chronological holdout test set) ===")
    evaluate("XGBoost ", y_test, xgb_scores)
    evaluate("LightGBM", y_test, lgb_scores)

    # --- Incremental lift over the naive rule, at matched recall ---
    target_recall = rule_recall
    thresholds = np.sort(xgb_scores)[::-1]
    for t in thresholds:
        pred = xgb_scores >= t
        if recall_score(y_test, pred) >= target_recall:
            matched_precision = precision_score(y_test, pred)
            print(f"\n=== Incremental lift check (XGBoost, recall matched to rule's {target_recall:.4f}) ===")
            print(f"  XGBoost precision at matched recall: {matched_precision:.4f}")
            print(f"  Naive rule precision:                {rule_precision:.4f}")
            print(f"  Lift: {matched_precision / rule_precision:.2f}x" if rule_precision > 0 else "  Lift: n/a")
            break

    # --- Per-typology recall, at a fixed score threshold (0.5) ---
    print("\n=== Per-typology recall at threshold=0.5 (XGBoost) — masking check from docs/09 ===")
    test_result = test.copy()
    test_result["xgb_score"] = xgb_scores
    test_result["xgb_flag"] = xgb_scores >= 0.5
    per_type = test_result[test_result["is_fraud"]].groupby("fraud_type")["xgb_flag"].mean()
    print(per_type.round(3))

    test_result[["seller_id", "is_fraud", "fraud_type", "xgb_score"]].to_csv(
        "data/processed/model_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()