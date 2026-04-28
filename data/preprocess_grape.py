import pandas as pd
import numpy as np
import os

# config
RAW_PATH = "./data/raw/clinical/vf_clinical.xlsx"
OUTPUT_PATH = "./data/processed/patient_level.csv"

os.makedirs("./data/intermediate", exist_ok=True)
os.makedirs("./data/processed", exist_ok=True)


# utils
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    print(f"NAMES: {df.columns}")
    return df


def create_eye_id(df: pd.DataFrame) -> pd.DataFrame:
    df["eye_id"] = (
        df["subject_number"].astype(str) + "_" +
        df["laterality"]
    )
    return df


def split_vf(df: pd.DataFrame):
    vf_cols = df.columns[-61:]
    
    # Create proper names for VF columns
    vf_names = [f"vf_{i}" for i in range(61)]
    
    vf = df[vf_cols].copy()
    vf.columns = vf_names  # 🔥 critical fix
    
    clinical = df.drop(columns=vf_cols).copy()
    
    return clinical, vf


# loading
def load_data(path):
    baseline = pd.read_excel(path, sheet_name="Baseline")
    followup = pd.read_excel(path, sheet_name="Follow-up")
    return baseline, followup


# preprocessing
def preprocess():
    print("Loading data...")
    baseline, followup = load_data(RAW_PATH)

    # clean cols
    baseline = clean_columns(baseline)
    followup = clean_columns(followup)

    # replace "/" with nans
    baseline = baseline.replace("/", np.nan)
    followup = followup.replace("/", np.nan)

    # split vf data
    baseline_clinical, vf_baseline = split_vf(baseline)
    followup_clinical, vf_followup = split_vf(followup)

    # save im data
    vf_baseline.to_parquet("data/intermediate/vf_baseline.parquet")
    vf_followup.to_parquet("data/intermediate/vf_followup.parquet")

    # creating new eye_id
    baseline_clinical = create_eye_id(baseline_clinical)
    followup_clinical = create_eye_id(followup_clinical)

    print("TESTING2: Clinical columns:")
    for col in baseline_clinical.columns:
        print(col)

    # define target var
    baseline_clinical["target"] = baseline_clinical["progression_status_plr2"]

    # drop missings in target
    baseline_clinical = baseline_clinical.dropna(subset=["target"])

    # feature engineering
    print("building longitudinal features...")

    # convert numeric cols
    followup_clinical["iop"] = pd.to_numeric(followup_clinical["iop"], errors="coerce")

    # iop stats over time
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

    # visit count
    visits = (
        followup_clinical
        .groupby("eye_id")
        .size()
        .reset_index(name="visit_count")
    )

    # interval years (disease duration proxy)
    interval = (
        followup_clinical
        .groupby("eye_id")["interval_years"]
        .max()
        .reset_index()
    )

    interval.columns = ["eye_id", "max_interval_years"]

    # merge features 

    df = baseline_clinical.merge(iop_stats, on="eye_id", how="left")
    df = df.merge(visits, on="eye_id", how="left")
    df = df.merge(interval, on="eye_id", how="left")

    # final features

    print("Selecting final features...")

    feature_cols = [
        "eye_id",
        "age",
        "gender",
        "iop",
        "cct",
        "total_visits",
        "category_of_glaucoma",
        "oct_rnfl_thickness_mean",
        "target",
        "iop_mean",
        "iop_max",
        "iop_min",
        "iop_std",
        "visit_count",
        "max_interval_years"
    ]

    df = df[feature_cols]

    # clean

    print("Cleaning final dataset...")

    # encode categorical vars
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df["category_of_glaucoma"] = df["category_of_glaucoma"].map({
        "OAG": 0,
        "ACG": 1
    })

    # convert numeric cols
    numeric_cols = df.columns.drop(["eye_id"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # handle missing vals
    df = df.fillna(df.median(numeric_only=True))

    # saving

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Final dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")



if __name__ == "__main__":
    preprocess()