import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.utils.logger import get_logger

# configs
DATA_PATH = "data/processed/patient_level.csv"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "outputs/"

N_SPLITS = 5
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE
}

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = get_logger(__name__)

# load
def load_data(path):
    logger.info("Loading dataset...")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


# sanity checks
def run_sanity_checks(df):
    logger.info("Running sanity checks...")

    assert "target" in df.columns, "Target column missing!"
    assert "eye_id" in df.columns, "eye_id column missing!"

    # missings check
    missing = df.isnull().sum().sum()
    logger.info(f"Total missing values: {missing}")

    # target distribution
    logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

    # Basic stats
    logger.info("Feature summary:")
    logger.info(df.describe().to_string())


# prepare features
def prepare_features(df):
    X = df.drop(columns=["target", "eye_id"])
    y = df["target"]
    groups = df["eye_id"]

    # final check
    assert not X.isnull().any().any(), "NaNs found in features!"

    return X, y, groups


# GroupKFold training
def train_model(X, y, groups):
    logger.info("Starting training with GroupKFold...")

    gkf = GroupKFold(n_splits=N_SPLITS)

    aucs = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"Fold {fold + 1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        logger.info(f"Fold {fold + 1} AUC: {auc:.3f}")

        aucs.append(auc)
        models.append(model)

    mean_auc = np.mean(aucs)
    logger.info(f"Mean AUC: {mean_auc:.3f}")

    return models, aucs


# saving
def save_best_model(models, aucs):
    best_idx = int(np.argmax(aucs))
    best_model = models[best_idx]

    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Best model saved to {MODEL_PATH}")

    return best_model


# feature importance
def save_feature_importance(model, X):
    logger.info("Saving feature importance plot...")

    importances = model.feature_importances_
    features = X.columns

    plt.figure()
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))


# shap explainability
def save_shap(model, X):
    logger.info("Generating SHAP values...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))

    logger.info("SHAP plot saved.")


# metric saving
def save_metrics(aucs):
    mean_auc = np.mean(aucs)

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write("Fold AUCs:\n")
        for i, auc in enumerate(aucs):
            f.write(f"Fold {i+1}: {auc:.3f}\n")
        f.write(f"\nMean AUC: {mean_auc:.3f}\n")

    logger.info("Metrics saved.")


# pipeline
def main():
    df = load_data(DATA_PATH)

    run_sanity_checks(df)

    X, y, groups = prepare_features(df)

    models, aucs = train_model(X, y, groups)

    best_model = save_best_model(models, aucs)

    save_feature_importance(best_model, X)

    save_shap(best_model, X)

    save_metrics(aucs)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()