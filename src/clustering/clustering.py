"""
Unsupervised Clustering — Seller Fraud Detection System (Project 1)

Clusters sellers on engineered features (docs/08) WITHOUT using the fraud
label, to find recurring behavioral patterns and check whether they align
with known typologies (docs/03) or reveal something the typology list
missed — the root-cause-analysis use of clustering described in docs/09.

Input:  data/processed/seller_features.csv
Output: printed cluster profile report + data/processed/seller_clusters.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

CLUSTER_FEATURES = [
    "identity_consistency_score", "email_domain_age_days",
    "listing_velocity_24h", "category_entropy", "high_ticket_item_ratio",
    "txn_count_first_72h", "top5_buyer_revenue_share",
    "behavior_change_index",
]


def prepare_matrix(df: pd.DataFrame):
    X = df[CLUSTER_FEATURES].copy()
    # Standardize: without this, email_domain_age_days (scale ~0-3000) would
    # dominate Euclidean distance over e.g. category_entropy (scale 0-1).
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def choose_k(X_scaled, k_range=range(2, 11)):
    """Silhouette score for each candidate k -- picks the k whose clusters
    are most internally cohesive and externally separated, without using
    the fraud label at all (this must stay fully unsupervised)."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster")[CLUSTER_FEATURES].mean().round(3)
    profile["n_sellers"] = df.groupby("cluster").size()
    profile["fraud_rate"] = df.groupby("cluster")["is_fraud"].mean().round(3)
    dominant = (
        df[df["is_fraud"]]
        .groupby("cluster")["fraud_type"]
        .agg(lambda s: s.value_counts().idxmax() if len(s) else "n/a")
    )
    profile["dominant_fraud_type"] = dominant
    return profile


def main():
    df = pd.read_csv("data/processed/seller_features.csv")
    X_scaled = prepare_matrix(df)

    scores = choose_k(X_scaled)
    print("=== Silhouette scores by k (unsupervised selection) ===")
    for k, s in scores.items():
        print(f"  k={k}: silhouette={s:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"\nSelected k={best_k} (highest silhouette score)\n")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    profile = profile_clusters(df, labels)
    print("=== Cluster profile (silhouette-optimal k) ===")
    print(profile)

    # Forced higher-k exploration, matching the number of named typologies,
    # to check whether thinner-signal typologies (bust-out, ATO, collusion,
    # synthetic identity) form separable clusters even when not silhouette-optimal.
    FORCED_K = 7
    km_forced = KMeans(n_clusters=FORCED_K, random_state=42, n_init=10)
    labels_forced = km_forced.fit_predict(X_scaled)
    profile_forced = profile_clusters(df, labels_forced)
    print(f"\n=== Cluster profile (forced k={FORCED_K}, exploratory) ===")
    print(profile_forced)

    df["cluster"] = labels
    df[f"cluster_k{FORCED_K}"] = labels_forced
    df.to_csv("data/processed/seller_clusters.csv", index=False)


if __name__ == "__main__":
    main()