# Modeling Notes

This project predicts UFC fight winners from fighter names and optional betting
odds. The model is built for tabular fight data, not images or free-form text,
so the key design problem is feature engineering, leakage control, and choosing
the best model family for structured data.

## Executive Summary

- Production model: XGBoost classifier.
- Baseline model that was tried and kept for comparison: PyTorch MLP.
- Reported XGBoost test AUC on the natural fight ordering: `0.7692`.
- PyTorch test AUC: `0.7426`.
- XGBoost was selected because it produced the better ranking metric and was
  the stronger fit for this kind of tabular data.

## Data Sources

The model pipeline uses two public data sources:

- UFCStats-derived fight results and round-level statistics.
- BestFightOdds-derived opening and closing/current moneylines.

Those sources are normalized into CSVs under `server/models/data/` and then
transformed into training artifacts under `server/models/artifacts/`.

## Feature Engineering

The core modeling choice is to convert each bout into a fighter-vs-opponent
feature vector. The final feature space is a mix of:

- Fighter career attributes.
- Opponent career attributes.
- Rolling fight-history features from prior UFC bouts.
- Difference features, such as fighter minus opponent for the same stat.
- Odds features, when moneylines are available.
- An `odds_inferred` flag for rows where odds were filled or approximated.

### Rolling History

The strongest signal in the training pipeline is recent performance. The
XGBoost notebook builds lagged rolling aggregates from prior fights only, so
the current bout never leaks into the features. The rolling set includes:

- Significant strikes landed and attempted.
- Significant strike accuracy.
- Takedowns landed and attempted.
- Takedown accuracy.
- Head, body, leg, distance, clinch, and ground rates.
- Control time.
- Knockdowns.
- Submission attempts.
- Reversals.
- Recent win rate.

Rolling windows are computed over the previous 3 and 5 fights.

### Career Attributes

Career features add the static profile information that does not depend on the
most recent fights:

- Height, weight, and reach.
- Stance.
- Age at fight time.
- Record and rate-style summary stats.

These features help the model capture baseline physical and stylistic
differences that do not show up in the rolling form signal.

### Difference Features

The model also includes explicit fighter-minus-opponent differences. That makes
the "advantage" signal easier to learn and helps the model focus on relative
matchup strength rather than absolute values.

## Leakage Control

The training notebooks are careful about two forms of leakage:

1. The split is done by bout grouping so both mirrored rows for the same fight
   stay in the same train/validation/test split.
2. Rolling features are shifted before aggregation so only prior fights can
   contribute to the current row.

This matters because UFC data can be very easy to leak if the mirrored fighter
rows or future bouts are allowed across split boundaries.

## Why XGBoost

XGBoost is the production model for three reasons.

### 1. It Won The AUC Comparison

The repo contains a PyTorch baseline and an XGBoost model trained on similar
matchup data. Their held-out test metrics are:

| Model | Test Accuracy | Test AUC |
| --- | ---: | ---: |
| XGBoost | 0.7020 | 0.7692 |
| PyTorch MLP | 0.6754 | 0.7426 |

The XGBoost notebook also reports a grouped mirrored-row test AUC of `0.7715`
and a cross-validated best AUC of `0.7796`, but the `0.7692` natural-ordering
test AUC is the most representative metric for reporting because it reflects
the actual fight ordering used at inference time.

### 2. It Matches The Data Type

This is a structured tabular problem with:

- Mixed numeric and categorical signals.
- Many nonlinear interactions.
- A relatively modest dataset size compared with image or language tasks.
- A lot of engineered relative features.

Gradient-boosted trees are a strong fit for that setting. The model does not
need feature scaling, does not depend on large batch optimization, and tends to
handle tabular interactions better than a small MLP unless the neural network
is very carefully tuned.

### 3. It Is Easier To Inspect

The XGBoost model is easier to explain and debug because SHAP values work
cleanly with tree models. That makes it possible to see which matchup signals
actually drive the predictions instead of treating the network as a black box.

## Why Not PyTorch

PyTorch was tried first as a feed-forward MLP baseline. It is a reasonable
experiment, but it underperformed XGBoost on the key ranking metric.

Practical reasons it was not chosen for production:

- Lower AUC on the held-out test set.
- More sensitivity to preprocessing choices such as scaling.
- More tuning effort for architecture, regularization, and learning rate
  scheduling.
- Less transparent feature attribution than a tree model in this repo.

That does not mean the PyTorch model was bad. It simply did not beat the tree
model on the metric that matters most for this use case.

## Training Approach

The XGBoost training notebook uses:

- Grouped train/validation/test splits by bout.
- Optuna hyperparameter search.
- GroupKFold cross-validation.
- Early stopping.
- Tree SHAP for feature importance.
- Calibration plotting on the natural held-out distribution.

The optimized parameters are saved into the trained model artifact so inference
is deterministic and does not depend on the notebook environment.

## What The Plots Show

### XGBoost Training Curves

![XGBoost training curves](../server/models/data/xgb_training_curves.png)

This plot shows training and validation log loss over boosting rounds. It is
used to confirm that early stopping is finding a stable region rather than
overfitting aggressively.

### PyTorch Training Curves

![PyTorch training curves](../server/models/data/training_curves.png)

This is the MLP baseline. The plot is useful because it shows that the neural
network is learning, but not well enough to beat the boosted tree on AUC.

### Calibration

![Calibration curve](../server/models/reports/calibration.png)

The calibration plot shows whether predicted probabilities line up with actual
win frequency. For a betting-oriented app, calibration matters almost as much
as raw accuracy.

### SHAP Importance

![SHAP feature importance](../server/models/data/shap_importance.png)

This figure shows which features the XGBoost model leans on most heavily. It is
the main explanation tool for the production model.

## Inference Behavior

At prediction time the app builds one row for `fighter1 vs fighter2` and a
second row for `fighter2 vs fighter1`, then averages both perspectives. That
keeps the output symmetric, so the predicted probability does not depend on the
order in which the user entered the names.

If a fighter has career stats but lacks a usable rolling-history row, the API
falls back to zero-filled rolling features. That keeps inference available, but
the resulting prediction is less informed than for fighters with a full fight
history.

## Limitations

- The model depends on the freshness of the scraped UFCStats and BestFightOdds
  data.
- Moneyline features can lag reality if the source pages change or odds are
  missing.
- The model does not directly understand injuries, short-notice replacements,
  weigh-in surprises, or qualitative context.
- AUC is the main reported metric, but no model is perfect and calibration can
  drift over time.

## Bottom Line

XGBoost is the production model because it performs better on the held-out
data, fits the structure of the problem, and is easier to explain. PyTorch was
useful as a baseline, but it did not justify replacing the tree model.
