"""
Policy Design & Deployment Simulation — Seller Fraud Detection System

Converts the model score (docs/12) and cost-optimal threshold (docs/13)
into a multi-tier production POLICY -- the actual artifact a risk team
ships, not just a score. Then simulates applying that policy across the
walk-forward windows from docs/13 to check stability over time.

Tiers (informed by docs/02's stakeholder map -- each tier trades off
Risk's loss-prevention goal against Product's low-friction goal
differently):
  - AUTO_BLOCK   (score >= 0.90): high-confidence hard decline
  - MANUAL_REVIEW(0.39 <= score < 0.90): sent to Ops review queue
  - SOFT_FRICTION(0.10 <= score < 0.39): extra verification step, seller can still proceed
  - MONITOR      (score < 0.10): approved, logged for passive monitoring

Input:  data/processed/model_predictions.csv, seller_features.csv,
        growth_monitoring_features.csv, data/raw/transactions.csv
Output: printed report + data/processed/policy_simulation.csv
"""

import numpy as np
import pandas as pd
import xgboost as xgb

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

TIER_THRESHOLDS = [
    (0.90, "AUTO_BLOCK"),
    (0.39, "MANUAL_REVIEW"),
    (0.10, "SOFT_FRICTION"),
    (0.00, "MONITOR"),
]

FP_COST_BY_TIER = {"AUTO_BLOCK": 200.0, "MANUAL_REVIEW": 50.0, "SOFT_FRICTION": 10.0, "MONITOR": 0.0}
FRAUD_DETERRENCE_BY_TIER = {"AUTO_BLOCK": 1.0, "MANUAL_REVIEW": 1.0, "SOFT_FRICTION": 0.5, "MONITOR": 0.0}

FEATURE_COLUMNS = [
    "identity_consistency_score", "email_domain_age_days",
    "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
    "txn_count_first_72h", "txn_count_first_72h_zscore",
    "top5_buyer_revenue_share", "payout_lag_days", "behavior_change_index",
    "max_percentile_jump_7d",
]


def assign_tier(score: float) -> str:
    for threshold, tier in TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "MONITOR"


def simulate_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["xgb_score"].apply(assign_tier)

    def tier_cost(row):
        if row["is_fraud"]:
            deterrence = FRAUD_DETERRENCE_BY_TIER[row["tier"]]
            return row["total_revenue"] * (1 - deterrence)
        else:
            return FP_COST_BY_TIER[row["tier"]]

    df["cost"] = df.apply(tier_cost, axis=1)
    return df


def policy_summary(df: pd.DataFrame):
    print("=== Multi-tier policy summary ===")
    summary = df.groupby("tier").agg(
        n_sellers=("seller_id", "count"),
        n_fraud=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
        total_cost=("cost", "sum"),
    ).round(3)
    tier_order = ["AUTO_BLOCK", "MANUAL_REVIEW", "SOFT_FRICTION", "MONITOR"]
    summary = summary.reindex(tier_order)
    print(summary)

    total_cost = df["cost"].sum()
    approval_rate = (df["tier"].isin(["SOFT_FRICTION", "MONITOR"])).mean()
    fraud_caught_rate = df[df["is_fraud"]]["tier"].isin(["AUTO_BLOCK", "MANUAL_REVIEW"]).mean()
    print(f"\n  Total policy cost: ${total_cost:,.2f}")
    print(f"  Sellers proceeding with no more than soft friction: {approval_rate:.1%}")
    print(f"  Fraud fully intercepted (AUTO_BLOCK or MANUAL_REVIEW): {fraud_caught_rate:.1%}")
    return total_cost


def compare_to_single_threshold(df: pd.DataFrame, total_cost_multitier: float):
    df = df.copy()
    flagged = df["xgb_score"] >= 0.39
    fp_cost = ((flagged) & (~df["is_fraud"])).sum() * FP_COST_BY_TIER["MANUAL_REVIEW"]
    fn_cost = df.loc[(~flagged) & (df["is_fraud"]), "total_revenue"].sum()
    single_threshold_cost = fp_cost + fn_cost

    print(f"\n=== Comparison: multi-tier policy vs. single-threshold policy (docs/13) ===")
    print(f"  Single-threshold (0.39, binary flag) total cost: ${single_threshold_cost:,.2f}")
    print(f"  Multi-tier policy total cost:                    ${total_cost_multitier:,.2f}")
    diff = single_threshold_cost - total_cost_multitier
    print(f"  Difference: ${diff:,.2f} "
          f"({'multi-tier cheaper' if diff > 0 else 'single-threshold cheaper'})")


def deployment_simulation(feats: pd.DataFrame, n_folds: int = 4):
    feats = feats.sort_values("registration_date").reset_index(drop=True)
    cutoffs = feats["registration_date"].quantile(np.linspace(0.4, 0.9, n_folds + 1)).values

    print("\n=== Deployment simulation across walk-forward windows ===")
    for i in range(n_folds):
        train_cutoff, test_cutoff = cutoffs[i], cutoffs[i + 1]
        train = feats[feats["registration_date"] < train_cutoff]
        test = feats[
            (feats["registration_date"] >= train_cutoff) & (feats["registration_date"] < test_cutoff)
        ].copy()
        if test["is_fraud"].sum() < 3 or train["is_fraud"].sum() < 3:
            print(f"  window {i+1}: skipped (too few fraud cases)")
            continue

        X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].astype(int)
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
        )
        model.fit(X_train, y_train)
        test["xgb_score"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        test["tier"] = test["xgb_score"].apply(assign_tier)

        fraud_caught = test[test["is_fraud"]]["tier"].isin(["AUTO_BLOCK", "MANUAL_REVIEW"]).mean()
        approval_rate = test["tier"].isin(["SOFT_FRICTION", "MONITOR"]).mean()
        tier_counts = test["tier"].value_counts().reindex(
            ["AUTO_BLOCK", "MANUAL_REVIEW", "SOFT_FRICTION", "MONITOR"], fill_value=0
        )
        print(f"  window {i+1}: n={len(test)}, fraud_caught={fraud_caught:.1%}, "
              f"approval_rate={approval_rate:.1%}, tiers={dict(tier_counts)}")


def main():
    preds = pd.read_csv("data/processed/model_predictions.csv")
    txns = pd.read_csv("data/raw/transactions.csv")
    revenue = txns.groupby("seller_id")["amount"].sum().rename("total_revenue")
    df = preds.merge(revenue, on="seller_id", how="left").fillna({"total_revenue": 0.0})

    sim = simulate_policy(df)
    total_cost = policy_summary(sim)
    compare_to_single_threshold(df, total_cost)

    sim.to_csv("data/processed/policy_simulation.csv", index=False)

    feats = pd.read_csv("data/processed/seller_features.csv", parse_dates=["registration_date"])
    growth = pd.read_csv("data/processed/growth_monitoring_features.csv")
    feats = feats.merge(growth[["seller_id", "max_percentile_jump_7d"]], on="seller_id", how="left")
    feats["payout_lag_days"] = feats["payout_lag_days"].fillna(999)
    deployment_simulation(feats)


if __name__ == "__main__":
    main()