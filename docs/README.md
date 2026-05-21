# UFC Win Predictor Docs

This folder documents how the UFC model was built, why the production model is
XGBoost, and how the training visuals should be interpreted.

## Start Here

- [Model report](./model-report.md)
- [Modeling and feature engineering appendix](./modeling.md)

## Key Artifacts

- [XGBoost training curves](../server/models/data/xgb_training_curves.png)
- [PyTorch training curves](../server/models/data/training_curves.png)
- [Calibration curve](../server/models/reports/calibration.png)
- [SHAP feature importance](../server/models/data/shap_importance.png)

## What To Look For

- XGBoost is the production model because it achieved the best AUC on the
  held-out test set.
- The PyTorch baseline is documented so the tradeoff is explicit rather than
  implied.
- The plots show how the model learns, how well it calibrates, and which
  features matter most.
