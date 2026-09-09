# Unsupervised Clustering

## Method
K-means on standardized engineered features (docs/08), with k selected via
silhouette score, run without using the fraud label. Clustering is used as
a root-cause-analysis tool (docs/09): do naturally-forming behavioral
groups align with known typologies (docs/03), or reveal gaps in them?

## Findings

### Silhouette-optimal k=2
Score 0.633 (next-best k scored ~0.17-0.19) — a single dominant fault line
in the feature space. Resulting clusters: a 156-seller cluster with 99.4%
fraud rate (stolen-goods/triangulation catalog-mismatch signature) vs. the
remaining 4,844 sellers at 4.1% fraud rate. Strong unsupervised validation
of the catalog-concentration signal design.

### Forced k=7 (matching typology count)
Surfaces one additional genuine cluster: 162 sellers with a 28.4% fraud
rate (~4x base-rate enrichment), dominated by account-takeover, separated
by behavior_change_index. Four other clusters are NOT meaningfully
separated by fraud-relevant features — they split mainly along
email_domain_age_days, which is largely uninformative noise outside the
synthetic-identity typology. Flagged explicitly as a clustering artifact
rather than reported as a false discovery.

## Root-Cause Conclusion
Collusion and synthetic-identity fraud remain thinly represented in
cluster structure even at higher k, consistent with docs/09's finding
that they rely on single narrow features rather than a broad behavioral
footprint — better suited to supervised modeling (docs/12) than
clustering. Account-takeover's clean separation reinforces the case for
a dedicated growth-monitoring model (docs/11).
