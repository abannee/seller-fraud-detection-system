"""
Explainability (SHAP) — Seller Fraud Detection System (Project 1)

Explains the XGBoost model's predictions using SHAP (SHapley Additive
exPlanations), closing the loop opened in docs/06: every flagged seller
should have a traceable, investigator-readable reason, not just a score.

Three things this script produces:
1. Global feature importance (mean |SHAP value|) -- which features matter
   most across the whole test set
2. Per-typology SHAP profile -- which features actually drive each fraud
   typology's score, checked against the docs/07 signal catalog's stated
   targets
3. Individual case explanations for a few real flagged sellers, in
   investigator-readable language

Input:  data/processed/seller_features.csv, growth_monitoring_features.csv
Output: printed report + data/processed/shap_values.csv
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

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
    df["payout_lag_days"] = df["payout_lag_days"].fillna(999)
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.8):
    cutoff = df["registration_date"].quantile(train_frac)
    train = df[df["registration_date"] < cutoff].copy()
    test = df[df["registration_date"] >= cutoff].copy()
    return train, test


def main():
    df = load_and_merge()
    train, test = chronological_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].astype(int)
    X_test = test[FEATURE_COLUMNS]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("=== Global feature importance (mean |SHAP value|) ===")
    print(mean_abs_shap.round(4))

    combined = mean_abs_shap["category_entropy"] + mean_abs_shap["high_ticket_item_ratio"]
    print(f"\n  category_entropy + high_ticket_item_ratio combined importance: {combined:.4f}")
    print(f"  (docs/09 flagged these as a 0.95-correlated redundant pair -- "
          f"SHAP splits credit between them rather than crediting one)")

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS, index=test.index)
    shap_df["fraud_type"] = test["fraud_type"].values
    print("\n=== Per-typology mean SHAP value (signed -- positive pushes toward fraud) ===")
    profile = shap_df.groupby("fraud_type")[FEATURE_COLUMNS].mean().round(3)
    print(profile.loc[[t for t in profile.index if t != "none"]])

    print("\n=== Individual case explanations (investigator-readable) ===")
    test_reset = test.reset_index(drop=True)
    shap_reset = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS)
    base_value = explainer.expected_value

    fraud_examples = test_reset[test_reset["is_fraud"]].head(3)
    for idx in fraud_examples.index:
        seller_id = test_reset.loc[idx, "seller_id"]
        fraud_type = test_reset.loc[idx, "fraud_type"]
        row_shap = shap_reset.loc[idx].sort_values(key=np.abs, ascending=False)
        print(f"\n  Seller {seller_id} (actual type: {fraud_type})")
        print(f"    base rate (log-odds): {base_value:.3f}")
        for feat, val in row_shap.head(3).items():
            direction = "pushed UP (riskier)" if val > 0 else "pushed DOWN (safer)"
            print(f"    - {feat} = {test_reset.loc[idx, feat]:.3f}: {direction} by {val:+.3f}")

    shap_out = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLUMNS])
    shap_out["seller_id"] = test["seller_id"].values
    shap_out["fraud_type"] = test["fraud_type"].values
    shap_out.to_csv("data/processed/shap_values.csv", index=False)


if __name__ == "__main__":
    main()