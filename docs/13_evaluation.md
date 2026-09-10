# Evaluation: Precision-Recall, Thresholds, Back-testing

## Precision-Recall Curve
Flat at precision=1.000 up to ~80% recall, degrading to 0.859 at 95% recall,
collapsing to 0.063 at 100% recall (structural: reaching 100% recall
requires flagging everyone, at which point precision converges to the base
fraud rate). Indicates a large set of cleanly-separable fraud cases plus a
smaller genuinely ambiguous tail.

## Cost-Based Threshold Selection
Cost model: false positive = $50 flat review cost; false negative = that
seller's actual extracted revenue (ties loss severity to real dollar
amounts rather than treating all misses equally).

| Threshold | Flagged | FP | FN | Total Cost |
|---|---|---|---|---|
| 0.22 (optimal) | 73 | 10 | 1 | $500.00 |
| 0.50 (default) | - | - | - | $2,266.87 |

Optimal threshold is far more aggressive (lower) than a naive 0.5 default,
because missed-fraud cost (avg. $1,802/case) vastly exceeds review cost
($50) -- cost-optimal behavior favors high recall given this asymmetry.

**Caveats:** assumes full loss prevention on catch, flat review cost
regardless of case complexity, and omits false-positive friction cost
beyond the review itself -- a simplified decision aid, not a complete
business case.

## Walk-Forward Back-testing
Expanding-window validation (4 folds) across the full observation period:
ROC-AUC 0.9922-0.9995, PR-AUC 0.9710-0.9924. No fold collapses or clear
drift -- confirms docs/12's single-split result reflects stable
performance across the window, not a lucky split. Same synthetic-data
caveat as docs/12 applies to the absolute values.
