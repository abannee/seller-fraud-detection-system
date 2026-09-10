"""
Evaluation: Precision-Recall, Thresholds, Back-testing — Project 1

Three things this script does, each answering a distinct question left
open by docs/12:
1. Precision-recall curve -- how does the model trade off precision and
   recall across all possible thresholds (not just 0.5)?
2. Cost-based threshold selection -- given a stated, explicit cost model
   (false negative = lost revenue, false positive = review cost), which
   single threshold minimizes total expected operational cost?
3. Walk-forward back-testing -- does model performance hold up across
   multiple sequential time windows, or was the single chronological split
   in docs/12 a lucky/unlucky one-off?

Input:  data/processed/model_predictions.csv, data/processed/seller_features.csv,
        data/raw/transactions.csv, data/raw/sellers.csv
Output: printed report + data/processed/threshold_cost_curve.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import xgboost as xgb

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

REVIEW_COST = 50.0  # assumed flat operational cost per manual review (false positive)

FEATURE_COLUMNS = [
    "identity_consistency_score", "email_domain_age_days",
    "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
    "txn_count_first_72h", "txn_count_first_72h_zscore",
    "top5_buyer_revenue_share", "payout_lag_days", "behavior_change_index",
    "max_percentile_jump_7d",
]


def load_predictions_with_cost():
    preds = pd.read_csv("data/processed/model_predictions.csv")
    txns = pd.read_csv("data/raw/transactions.csv")
    revenue = txns.groupby("seller_id")["amount"].sum().rename("total_revenue")
    df = preds.merge(revenue, on="seller_id", how="left").fillna({"total_revenue": 0.0})
    return df


def precision_recall_report(df: pd.DataFrame):
    precisions, recalls, thresholds = precision_recall_curve(df["is_fraud"], df["xgb_score"])
    print("=== Precision-Recall curve (sampled) ===")
    for target_recall in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        idx = np.argmin(np.abs(recalls - target_recall))
        print(f"  recall~={recalls[idx]:.3f}  precision={precisions[idx]:.3f}  "
              f"threshold={thresholds[min(idx, len(thresholds)-1)]:.4f}")
    return precisions, recalls, thresholds


def cost_based_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Sweeps candidate thresholds, computes total expected operational cost
    at each: FP cost = fixed review cost per false flag; FN cost = the
    actual revenue extracted by that specific fraud seller (assumes catching
    fraud in time prevents that loss -- a simplifying assumption, stated
    explicitly rather than hidden)."""
    rows = []
    for t in np.arange(0.01, 1.00, 0.01):
        flagged = df["xgb_score"] >= t
        fp_mask = flagged & (~df["is_fraud"])
        fn_mask = (~flagged) & (df["is_fraud"])
        fp_cost = fp_mask.sum() * REVIEW_COST
        fn_cost = df.loc[fn_mask, "total_revenue"].sum()
        total_cost = fp_cost + fn_cost
        rows.append({
            "threshold": round(t, 2), "n_flagged": int(flagged.sum()),
            "fp_count": int(fp_mask.sum()), "fn_count": int(fn_mask.sum()),
            "fp_cost": fp_cost, "fn_cost": fn_cost, "total_cost": total_cost,
        })
    return pd.DataFrame(rows)


def walk_forward_backtest(feats: pd.DataFrame, n_folds: int = 4):
    """Expanding-window back-test: fold k trains on everything registered
    before a growing cutoff and tests on the next chronological slice --
    checks whether performance is stable across time, or whether the single
    split in docs/12 happened to land on an easy or lucky period."""
    feats = feats.sort_values("registration_date").reset_index(drop=True)
    cutoffs = feats["registration_date"].quantile(
        np.linspace(0.4, 0.9, n_folds + 1)
    ).values

    print("\n=== Walk-forward back-test (expanding window) ===")
    for i in range(n_folds):
        train_cutoff = cutoffs[i]
        test_cutoff = cutoffs[i + 1]
        train = feats[feats["registration_date"] < train_cutoff]
        test = feats[
            (feats["registration_date"] >= train_cutoff)
            & (feats["registration_date"] < test_cutoff)
        ]
        if test["is_fraud"].sum() < 3 or train["is_fraud"].sum() < 3:
            print(f"  fold {i+1}: skipped (too few fraud cases in this window)")
            continue

        X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].astype(int)
        X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"].astype(int)

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
        )
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        print(f"  fold {i+1}: train={len(train)} (n_fraud={y_train.sum()}), "
              f"test={len(test)} (n_fraud={y_test.sum()}), "
              f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")


def main():
    df = load_predictions_with_cost()

    precision_recall_report(df)

    cost_df = cost_based_threshold(df)
    best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    print("\n=== Cost-based optimal threshold ===")
    print(f"  Assumed cost model: false positive = ${REVIEW_COST:.0f} review cost, "
          f"false negative = actual seller revenue extracted")
    print(f"  Optimal threshold: {best_row['threshold']:.2f}")
    print(f"  At this threshold: {int(best_row['n_flagged'])} sellers flagged, "
          f"{int(best_row['fp_count'])} false positives, {int(best_row['fn_count'])} false negatives")
    print(f"  Total expected cost: ${best_row['total_cost']:,.2f} "
          f"(FP cost=${best_row['fp_cost']:,.2f}, FN cost=${best_row['fn_cost']:,.2f})")

    default_row = cost_df.iloc[(cost_df["threshold"] - 0.50).abs().argmin()]
    print(f"\n  For comparison, threshold=0.50 total cost: ${default_row['total_cost']:,.2f}")
    savings = default_row["total_cost"] - best_row["total_cost"]
    print(f"  Savings from cost-optimal threshold: ${savings:,.2f}")

    cost_df.to_csv("data/processed/threshold_cost_curve.csv", index=False)

    feats = pd.read_csv("data/processed/seller_features.csv", parse_dates=["registration_date"])
    growth = pd.read_csv("data/processed/growth_monitoring_features.csv")
    feats = feats.merge(growth[["seller_id", "max_percentile_jump_7d"]], on="seller_id", how="left")
    feats["payout_lag_days"] = feats["payout_lag_days"].fillna(999)
    walk_forward_backtest(feats)


if __name__ == "__main__":
    main()