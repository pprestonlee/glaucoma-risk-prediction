# Glaucoma Progression Risk Prediction

An end-to-end healthcare machine learning system that predicts glaucoma progression risk from longitudinal clinical and visual field data, deployed as a live REST API with a clinician-facing analytics dashboard.

---

## Overview

Glaucoma is a leading cause of irreversible blindness. Early identification of patients at high risk of disease progression enables timely intervention and personalised treatment planning.

This project builds a full production ML pipeline on the **GRAPE dataset** — from raw clinical data through feature engineering, model training, calibration, cloud deployment, and real-time monitoring — demonstrating the complete lifecycle of a healthcare data science system.

---

## Results

| Metric | Value |
|---|---|
| Mean AUC (5-fold GroupKFold) | **0.745** |
| PR-AUC | 0.307 |
| Brier Score (calibrated) | **0.112** (below null baseline of 0.129) |
| Recall at optimal threshold | 75% |
| Optimal decision threshold | 0.20 |

At threshold 0.20: the model identifies **75% of progressors** while maintaining clinically actionable precision — appropriate for a screening tool where missing a progressor carries high cost.

---

## Key Technical Decisions

**GroupKFold cross-validation** — Eyes are grouped by `eye_id` (subject + laterality) to prevent data leakage between train and validation folds. A patient's left and right eyes cannot appear on both sides of a fold boundary.

**Isotonic regression calibration** — XGBoost's raw probabilities were over-dispersed (Brier score 0.157, worse than naive baseline). Isotonic regression fitted on out-of-fold predictions corrects the probability scale, making outputs clinically interpretable.

**`scale_pos_weight` per fold** — The dataset is 85% stable / 15% progressor. Class weight is computed from each fold's training labels independently, ensuring the model learns the minority class rather than defaulting to the majority.

**Blind spot masking** — VF values of `-1` encode the physiological blind spot, not a sensitivity measurement. Naively including them in threshold-based defect counts (the original pipeline) produced defect counts of zero for 100% of eyes. Replacing with `NaN` before feature computation restored meaningful signal.

---

## Architecture

```
Raw Data (GRAPE Excel)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  clean_columns, fix_duplicate_columns,
│   data/preprocess_  │  split_vf, blind spot masking
│   grape.py          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  VF features (mean, std, defect count/ratio,
│                     │  severe loss, S/I + L/R asymmetry)
│                     │  Longitudinal IOP (mean, max, min, std, slope)
│                     │  VF progression (slope, delta, final mean)
│                     │  Interaction features (age×IOP, RNFL×VF, IOP×VF)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │  XGBoost · GroupKFold · scale_pos_weight
│  src/models/train.py│  Isotonic calibration · SHAP explainability
└────────┬────────────┘
         │
         ├──── models/model_calibrated.pkl  →  S3
         └──── outputs/ (SHAP, PR curve, calibration, metrics)
                        │
                        ▼
              ┌─────────────────────┐
              │   FastAPI REST API  │  /predict  /health  /docs
              │   src/api/app.py    │  Boto3 · Background tasks
              └────────┬────────────┘
                       │
          ┌────────────┼────────────┐────────────┐
          ▼            ▼            ▼            ▼
        AWS S3      AWS SNS    CloudWatch    Athena
    (model +      (high-risk   (metrics     (SQL over
   pred logs)     alerts)     dashboard)   pred logs)
                       │
                       ▼
              ┌─────────────────────┐
              │  Streamlit Dashboard│  Risk assessment form
              │  src/dashboard/     │  Population analytics
              │  app.py             │  Model info + SHAP plots
              └─────────────────────┘
```

---

## Feature Engineering

### Visual Field (VF)
| Feature | Description |
|---|---|
| `vf_mean` | Mean absolute sensitivity across 61 points (dB) |
| `vf_std` | Sensitivity variability |
| `vf_defect_count` | Points below 15 dB (mild+ depression) |
| `vf_defect_ratio` | Defect count normalised by valid point count |
| `vf_severe_loss_count` | Points below 5 dB (near-total loss) |
| `vf_si_asymmetry` | Superior vs inferior hemifield mean difference |
| `vf_lr_asymmetry` | Even vs odd index interleave (second spatial dimension) |

### VF Progression (longitudinal)
| Feature | Description |
|---|---|
| `vf_mean_slope` | Linear regression of VF mean over follow-up time (dB/year) |
| `vf_delta` | Last visit VF mean minus first visit VF mean |
| `vf_final_mean` | VF mean sensitivity at last follow-up visit |

### Longitudinal IOP
| Feature | Description |
|---|---|
| `iop_mean/max/min/std` | IOP summary statistics across follow-up visits |
| `iop_slope` | Linear trend of IOP over time (mmHg/year) |

### Interaction Features
`age × IOP` · `RNFL × VF mean` · `IOP × VF mean`

---

## AWS Infrastructure

| Service | Purpose |
|---|---|
| **ECR** | Container registry for the Docker image |
| **ECS Fargate** | Managed serverless container hosting (0.5 vCPU / 1 GB) |
| **S3** | Model artifact storage + Hive-partitioned prediction logs |
| **Athena** | SQL querying over prediction logs (`glaucoma_db.predictions`) |
| **CloudWatch** | Custom metrics: `PredictionCount`, `RiskScore`, `HighRiskCount` |
| **SNS** | Email alert on every high-risk prediction |
| **IAM** | Scoped task role — least-privilege access to S3, CloudWatch, SNS |

---

## Repository Structure

```
├── data/
│   ├── load_grape.py            # Raw Excel ingestion
│   ├── preprocess_grape.py      # Full preprocessing + feature engineering
│   ├── intermediate/            # Parquet cache (VF splits)
│   └── processed/               # patient_level.csv (model input)
│
├── src/
│   ├── models/train.py          # Training, calibration, SHAP, metrics
│   ├── api/app.py               # FastAPI prediction endpoint
│   ├── dashboard/app.py         # Streamlit clinician dashboard
│   └── utils/logger.py          # Shared logger
│
├── models/
│   ├── model.pkl                # Best-fold XGBoost
│   └── model_calibrated.pkl     # XGBoost + isotonic calibrator
│
├── outputs/
│   ├── shap_summary.png         # SHAP feature impact plot
│   ├── feature_importance.png   # XGBoost feature importance
│   ├── calibration.png          # Raw vs calibrated reliability diagram
│   ├── pr_curve.png             # Precision-recall curve (OOF)
│   └── metrics.txt              # Full evaluation report
│
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis (11 sections)
│
├── deploy/
│   └── deploy.sh                # One-command AWS deployment
│
├── Dockerfile                   # Production container (python:3.11-slim)
└── requirements.txt
```

---

## Running Locally

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess data
python data/preprocess_grape.py

# 3. Train model
PYTHONPATH=. python src/models/train.py

# 4. Start API
PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 5. Start dashboard (separate terminal)
PYTHONPATH=. streamlit run src/dashboard/app.py

# 6. Run EDA notebook
jupyter notebook notebooks/eda.ipynb
```

---

## Tech Stack

**ML & Data** — Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · SciPy

**API & Dashboard** — FastAPI · Uvicorn · Pydantic · Streamlit · Plotly

**Cloud** — AWS ECR · ECS Fargate · S3 · Athena · CloudWatch · SNS · IAM

**Dev** — Docker · Jupyter · Matplotlib · Seaborn