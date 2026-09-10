# Evaluation: Precision-Recall, Thresholds, Back-testing

## Precision-Recall Curve
Flat at precision=1.000 up to ~90% recall, degrading to 0.910 at 95%
recall, collapsing to 0.063 at 100% recall (structural: reaching 100%
recall requires flagging everyone, at which point precision converges to
the base fraud rate). Indicates a large set of cleanly-separable fraud
cases plus a smaller genuinely ambiguous tail.

## Cost-Based Threshold Selection
Cost model: false positive = $50 flat review cost; false negative = that
seller's actual extracted revenue (ties loss severity to real dollar
amounts rather than treating all misses equally).

| Threshold | Flagged | FP | FN | Total Cost |
|---|---|---|---|---|
| 0.39 (optimal) | 68 | 6 | 2 | $343.67 |
| 0.50 (default) | - | - | - | $2,266.87 |

Optimal threshold is far more aggressive (lower) than a naive 0.5 default,
because missed-fraud cost (avg. $1,802/case) vastly exceeds review cost
($50) -- cost-optimal behavior favors high recall given this asymmetry.
Savings vs. the 0.5 default: $1,923.20.

**Note on run-to-run variance:** the exact optimal threshold shifts
slightly between training runs (observed range ~0.22-0.39 across two
runs) due to XGBoost's inherent non-determinism combined with a modest
test-set size (~64 fraud cases), where a handful of borderline sellers
crossing the threshold boundary can move the argmin. The qualitative
conclusion -- an aggressive, well-below-0.5 threshold is cost-optimal
given this loss asymmetry, with ~$1,800-1,900 in savings vs. the default
-- is stable across runs; the specific threshold value is not guaranteed
to be exactly reproducible.

**Caveats:** assumes full loss prevention on catch, flat review cost
regardless of case complexity, and omits false-positive friction cost
beyond the review itself -- a simplified decision aid, not a complete
business case.

## Walk-Forward Back-testing
Expanding-window validation (4 folds) across the full observation period:
ROC-AUC 0.9926-0.9991, PR-AUC 0.9685-0.9883. No fold collapses or clear
drift -- confirms docs/12's single-split result reflects stable
performance across the window, not a lucky split. Same synthetic-data
caveat as docs/12 applies to the absolute values.
