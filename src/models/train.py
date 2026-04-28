import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.utils.logger import get_logger

# configs
DATA_PATH = "data/processed/patient_level.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "outputs/"

N_SPLITS = 5
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE
}

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = get_logger(__name__)


def load_data(path):
    logger.info("Loading dataset...")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def run_sanity_checks(df):
    logger.info("Running sanity checks...")

    assert "target" in df.columns, "Target column missing!"
    assert "eye_id" in df.columns, "eye_id column missing!"

    missing = df.isnull().sum().sum()
    logger.info(f"Total missing values: {missing}")
    logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    logger.info("Feature summary:")
    logger.info(df.describe().to_string())


def prepare_features(df):
    X = df.drop(columns=["target", "eye_id"])
    y = df["target"]
    groups = df["eye_id"]

    assert not X.isnull().any().any(), "NaNs found in features!"

    return X, y, groups


def train_model(X, y, groups):
    logger.info("Starting training with GroupKFold...")

    gkf = GroupKFold(n_splits=N_SPLITS)

    aucs = []
    models = []
    oof_preds = np.zeros(len(y))
    oof_labels = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"Fold {fold + 1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        model = XGBClassifier(**MODEL_PARAMS, scale_pos_weight=spw)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        oof_labels[val_idx] = y_val.values

        auc = roc_auc_score(y_val, preds)
        logger.info(f"Fold {fold + 1} AUC: {auc:.3f}")

        aucs.append(auc)
        models.append(model)

    logger.info(f"Mean AUC: {np.mean(aucs):.3f}")

    return models, aucs, oof_preds, oof_labels


def save_best_model(models, aucs):
    best_idx = int(np.argmax(aucs))
    best_model = models[best_idx]

    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Best model saved to {MODEL_PATH}")

    return best_model


def fit_calibrator(oof_labels, oof_preds):
    """Fit isotonic regression on OOF predictions to correct probability scale."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds, oof_labels)
    return calibrator


def save_feature_importance(model, X):
    logger.info("Saving feature importance plot...")

    order = np.argsort(model.feature_importances_)
    features = np.array(X.columns)[order]
    importances = model.feature_importances_[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.3)))
    ax.barh(features, importances)
    ax.set_title("Feature Importance (best fold)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)


def save_shap(model, X):
    logger.info("Generating SHAP values...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.title("SHAP Feature Impact (best fold)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plot saved.")


def save_pr_curve(oof_labels, oof_preds, cal_preds):
    logger.info("Saving PR curve...")

    baseline = oof_labels.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    for preds, label in [(oof_preds, "Raw"), (cal_preds, "Calibrated")]:
        prec, rec, _ = precision_recall_curve(oof_labels, preds)
        ap = average_precision_score(oof_labels, preds)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})")

    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=120)
    plt.close(fig)


def save_calibration(oof_labels, oof_preds, cal_preds):
    logger.info("Saving calibration curve...")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for preds, label in [(oof_preds, "Raw"), (cal_preds, "Calibrated")]:
        frac_pos, mean_pred = calibration_curve(oof_labels, preds, n_bins=8)
        brier = brier_score_loss(oof_labels, preds)
        ax.plot(mean_pred, frac_pos, "s-", label=f"{label} (Brier={brier:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=120)
    plt.close(fig)


def _optimal_threshold(oof_labels, oof_preds):
    """Threshold that maximises F1 on OOF predictions."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        _, _, f1, _ = precision_recall_fscore_support(
            oof_labels, (oof_preds >= t).astype(int),
            average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _section(f, oof_labels, preds, label):
    opt_t = _optimal_threshold(oof_labels, preds)
    ap = average_precision_score(oof_labels, preds)
    brier = brier_score_loss(oof_labels, preds)

    p5, r5, f5, _ = precision_recall_fscore_support(
        oof_labels, (preds >= 0.5).astype(int), average="binary", zero_division=0
    )
    p, r, f1, _ = precision_recall_fscore_support(
        oof_labels, (preds >= opt_t).astype(int), average="binary", zero_division=0
    )

    f.write(f"\n=== {label} ===\n")
    f.write(f"PR-AUC (Avg Precision): {ap:.3f}\n")
    f.write(f"Brier Score:            {brier:.3f}\n")
    f.write(f"\n  Threshold = 0.50\n")
    f.write(f"    Precision: {p5:.3f}  Recall: {r5:.3f}  F1: {f5:.3f}\n")
    f.write(f"\n  Optimal threshold (max F1) = {opt_t:.2f}\n")
    f.write(f"    Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}\n")


def save_metrics(aucs, oof_labels, oof_preds, cal_preds):
    mean_auc = np.mean(aucs)

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write("Fold AUCs:\n")
        for i, auc in enumerate(aucs):
            f.write(f"  Fold {i+1}: {auc:.3f}\n")
        f.write(f"\nMean AUC: {mean_auc:.3f}\n")

        _section(f, oof_labels, oof_preds, "Raw (uncalibrated)")
        _section(f, oof_labels, cal_preds, "Calibrated (isotonic regression)")

    logger.info("Metrics saved.")


def main():
    df = load_data(DATA_PATH)

    run_sanity_checks(df)

    X, y, groups = prepare_features(df)

    models, aucs, oof_preds, oof_labels = train_model(X, y, groups)

    best_model = save_best_model(models, aucs)

    calibrator = fit_calibrator(oof_labels, oof_preds)
    joblib.dump({"model": best_model, "calibrator": calibrator}, "models/model_calibrated.pkl")
    logger.info("Calibrated model saved to models/model_calibrated.pkl")

    cal_preds = calibrator.predict(oof_preds)

    save_feature_importance(best_model, X)
    save_shap(best_model, X)
    save_pr_curve(oof_labels, oof_preds, cal_preds)
    save_calibration(oof_labels, oof_preds, cal_preds)
    save_metrics(aucs, oof_labels, oof_preds, cal_preds)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
