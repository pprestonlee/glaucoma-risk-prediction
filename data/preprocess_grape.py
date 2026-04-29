import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# cfg
RAW_PATH = "data/raw/clinical/vf_clinical.xlsx"
OUTPUT_PATH = "data/processed/patient_level.csv"

os.makedirs("data/intermediate", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# utils
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def create_eye_id(df: pd.DataFrame) -> pd.DataFrame:
    df["eye_id"] = (
        df["subject_number"].astype(str) + "_" +
        df["laterality"]
    )
    return df


def fix_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            cols[idx] = f"{dup}_{i}" if i != 0 else dup
    df.columns = cols
    return df


# vf handling
def split_vf(df: pd.DataFrame):
    """Robust VF column detection."""

    # search for vfs
    vf_cols = df.columns[df.columns.str.contains("vf", case=False, na=False)]

    # fallback taking last 60 columns
    if len(vf_cols) < 50:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        vf_cols = numeric_cols[-61:]

    vf = df[vf_cols].copy()
    vf.columns = [f"vf_{i}" for i in range(len(vf.columns))]

    clinical = df.drop(columns=vf_cols).copy()

    return clinical, vf


def engineer_vf_features(vf: pd.DataFrame):
    vf = vf.apply(pd.to_numeric, errors="coerce")

    # -1 encodes the physiological blind spot (always 0 sensitivity); exclude from stats
    vf_valid = vf.replace(-1, np.nan)
    n_points = vf_valid.shape[1]
    half = n_points // 2

    features = pd.DataFrame()

    features["vf_mean"] = vf_valid.mean(axis=1)
    features["vf_std"] = vf_valid.std(axis=1)
    features["vf_max"] = vf_valid.max(axis=1)

    # defect features — thresholds are for absolute sensitivity (dB), not deviation scores
    features["vf_defect_count"] = (vf_valid < 15).sum(axis=1)   # mild+ depression
    features["vf_defect_ratio"] = features["vf_defect_count"] / vf_valid.notna().sum(axis=1)
    features["vf_severe_loss_count"] = (vf_valid < 5).sum(axis=1)  # near-total loss

    # superior/inferior asymmetry (assumes top-to-bottom point ordering)
    features["vf_si_asymmetry"] = (
        vf_valid.iloc[:, :half].mean(axis=1) -
        vf_valid.iloc[:, half:].mean(axis=1)
    )

    # left/right asymmetry via even/odd index interleave (second spatial dimension)
    even_idx = list(range(0, n_points, 2))
    odd_idx = list(range(1, n_points, 2))
    features["vf_lr_asymmetry"] = (
        vf_valid.iloc[:, even_idx].mean(axis=1) -
        vf_valid.iloc[:, odd_idx].mean(axis=1)
    )

    return features


# vf progression
def compute_vf_progression(followup_df, followup_vf):
    """Per-eye linear slope of VF mean over time, plus first-to-last delta."""
    followup_vf = followup_vf.replace(-1, np.nan)
    vf_mean_per_visit = followup_vf.mean(axis=1)

    combined = followup_df[["eye_id", "interval_years"]].copy()
    combined["interval_years"] = pd.to_numeric(combined["interval_years"], errors="coerce")
    combined["vf_visit_mean"] = vf_mean_per_visit.values

    records = []
    for eye_id, grp in combined.groupby("eye_id"):
        grp = grp.sort_values("interval_years").dropna(subset=["interval_years", "vf_visit_mean"])

        if len(grp) < 2:
            records.append({"eye_id": eye_id, "vf_mean_slope": 0.0, "vf_delta": 0.0, "vf_final_mean": grp["vf_visit_mean"].iloc[-1] if len(grp) else np.nan})
            continue

        X = grp["interval_years"].values.reshape(-1, 1)
        y = grp["vf_visit_mean"].values

        slope = LinearRegression().fit(X, y).coef_[0]
        delta = float(y[-1] - y[0])
        records.append({"eye_id": eye_id, "vf_mean_slope": slope, "vf_delta": delta, "vf_final_mean": float(y[-1])})

    return pd.DataFrame(records)


# iop trend
def compute_iop_slope(followup_df):
    slopes = []

    for eye_id, group in followup_df.groupby("eye_id"):
        group = group.sort_values("interval_years")

        group["iop"] = pd.to_numeric(group["iop"], errors="coerce")
        group = group.dropna(subset=["iop", "interval_years"])

        if len(group) < 2:
            slopes.append((eye_id, 0))
            continue

        X = group["interval_years"].values.reshape(-1, 1)
        y = group["iop"].values

        model = LinearRegression()
        model.fit(X, y)

        slopes.append((eye_id, model.coef_[0]))

    return pd.DataFrame(slopes, columns=["eye_id", "iop_slope"])


# load data
def load_data(path):
    baseline = pd.read_excel(path, sheet_name="Baseline", header=0)
    followup = pd.read_excel(path, sheet_name="Follow-up", header=0)
    return baseline, followup


# preprocessing
def preprocess():
    print("Loading data...")
    baseline, followup = load_data(RAW_PATH)

    # clean columns
    baseline = clean_columns(baseline)
    followup = clean_columns(followup)

    baseline = fix_duplicate_columns(baseline)
    followup = fix_duplicate_columns(followup)

    # replace "/" with NaN
    baseline = baseline.replace("/", np.nan)
    followup = followup.replace("/", np.nan)

    # debug
    print("Total baseline columns:", len(baseline.columns))

    # split VF
    baseline_clinical, vf_baseline = split_vf(baseline)
    followup_clinical, vf_followup = split_vf(followup)

    # vf feature engineering
    print("Engineering VF features...")
    vf_features = engineer_vf_features(vf_baseline)

    baseline_clinical = pd.concat(
        [baseline_clinical.reset_index(drop=True), vf_features],
        axis=1
    )

    # eye_id 
    baseline_clinical = create_eye_id(baseline_clinical)
    followup_clinical = create_eye_id(followup_clinical)

    # target
    baseline_clinical["target"] = baseline_clinical["progression_status_plr2"]
    baseline_clinical = baseline_clinical.dropna(subset=["target"])

    # longitudinal features
    print("Building longitudinal features...")

    followup_clinical["iop"] = pd.to_numeric(followup_clinical["iop"], errors="coerce")

    iop_stats = (
        followup_clinical
        .groupby("eye_id")["iop"]
        .agg(["mean", "max", "min", "std"])
        .reset_index()
    )

    iop_stats.columns = [
        "eye_id",
        "iop_mean",
        "iop_max",
        "iop_min",
        "iop_std"
    ]

    visits = (
        followup_clinical
        .groupby("eye_id")
        .size()
        .reset_index(name="visit_count")
    )

    interval = (
        followup_clinical
        .groupby("eye_id")["interval_years"]
        .max()
        .reset_index()
    )

    interval.columns = ["eye_id", "max_interval_years"]

    # IOP slope
    iop_slope = compute_iop_slope(followup_clinical)

    # VF progression
    vf_progression = compute_vf_progression(followup_clinical, vf_followup)

    # merges
    df = baseline_clinical.merge(iop_stats, on="eye_id", how="left")
    df = df.merge(visits, on="eye_id", how="left")
    df = df.merge(interval, on="eye_id", how="left")
    df = df.merge(iop_slope, on="eye_id", how="left")
    df = df.merge(vf_progression, on="eye_id", how="left")

    # visit frequency
    df["visit_frequency"] = df["visit_count"] / (df["max_interval_years"] + 1e-5)

    # interaction features
    df["iop"] = pd.to_numeric(df["iop"], errors="coerce")
    df["oct_rnfl_thickness_mean"] = pd.to_numeric(df["oct_rnfl_thickness_mean"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["age_x_iop"] = df["age"] * df["iop"]
    df["rnfl_x_vf_mean"] = df["oct_rnfl_thickness_mean"] * df["vf_mean"]
    df["iop_x_vf_mean"] = df["iop"] * df["vf_mean"]

    # feature selection
    feature_cols = [
        "eye_id",
        "age",
        "gender",
        "iop",
        "cct",
        "category_of_glaucoma",
        "oct_rnfl_thickness_mean",
        "target",
        "iop_mean",
        "iop_max",
        "iop_min",
        "iop_std",
        "iop_slope",
        "visit_count",
        "max_interval_years",
        "visit_frequency",
        "vf_mean",
        "vf_std",
        "vf_max",
        "vf_defect_count",
        "vf_defect_ratio",
        "vf_severe_loss_count",
        "vf_si_asymmetry",
        "vf_lr_asymmetry",
        "age_x_iop",
        "rnfl_x_vf_mean",
        "iop_x_vf_mean",
        "vf_mean_slope",
        "vf_delta",
        "vf_final_mean",
    ]

    df = df[feature_cols]

    # encoding
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df["category_of_glaucoma"] = df["category_of_glaucoma"].map({
        "OAG": 0,
        "ACG": 1
    })

    # convert numeric
    numeric_cols = df.columns.drop(["eye_id"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # fill missing
    df = df.fillna(df.median(numeric_only=True))

    print(df.head())
    print(df.columns)

    # save
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["target"].value_counts(normalize=True))


# run
if __name__ == "__main__":
    preprocess()