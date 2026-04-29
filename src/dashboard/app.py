import json
import os

import boto3
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

# config
DEFAULT_API = os.environ.get("API_URL", "http://YOUR_API_URL")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "glaucoma-risk-YOUR_ACCOUNT_ID")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

st.set_page_config(
    page_title="Glaucoma Progression Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# sidebar
with st.sidebar:
    st.title("Glaucoma Risk Dashboard")
    st.caption("GRAPE dataset · XGBoost + Isotonic Calibration")
    st.divider()
    api_url = st.text_input("API endpoint", value=DEFAULT_API)
    s3_bucket = st.text_input("S3 bucket", value=DEFAULT_BUCKET)
    if st.button("Refresh analytics"):
        st.cache_data.clear()
    st.divider()
    st.caption("Mean AUC: 0.745 · Brier: 0.112 · PR-AUC: 0.307")


# load data
@st.cache_data(ttl=60, show_spinner="Loading prediction logs from S3...")
def load_predictions(bucket: str) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    records = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="predictions/"):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".json"):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            records.append(json.loads(body))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    return df


# tabs
tab_assess, tab_analytics, tab_model = st.tabs(
    ["Risk Assessment", "Population Analytics", "Model Info"]
)


# Risk Assessment Tab
with tab_assess:
    st.header("Patient Risk Assessment")
    st.caption("Enter clinical and visual field data to predict glaucoma progression risk.")

    with st.form("predict_form"):
        st.subheader("Clinical Baseline")
        c1, c2, c3, c4 = st.columns(4)
        age       = c1.number_input("Age (years)",          value=41,    min_value=0,   max_value=120)
        gender    = c2.selectbox("Gender",                  options=[1, 0], format_func=lambda x: "Male" if x else "Female")
        iop       = c3.number_input("IOP (mmHg)",           value=17.0,  min_value=0.0, step=0.5)
        cct       = c4.number_input("CCT (μm)",             value=540.0, min_value=300.0, max_value=700.0)
        c5, c6    = st.columns(2)
        glaucoma_cat = c5.selectbox("Glaucoma type",        options=[0, 1], format_func=lambda x: "OAG" if x == 0 else "ACG")
        rnfl      = c6.number_input("RNFL thickness (μm)",  value=80.0,  min_value=20.0, max_value=150.0)

        st.divider()

        st.subheader("Longitudinal IOP")
        c1, c2, c3, c4, c5 = st.columns(5)
        iop_mean  = c1.number_input("Mean IOP",   value=16.5, step=0.5)
        iop_max   = c2.number_input("Max IOP",    value=19.0, step=0.5)
        iop_min   = c3.number_input("Min IOP",    value=14.5, step=0.5)
        iop_std   = c4.number_input("IOP std",    value=1.8,  step=0.1)
        iop_slope = c5.number_input("IOP slope (mmHg/yr)", value=-0.06, step=0.01, format="%.2f")

        st.divider()

        st.subheader("Visit History")
        c1, c2, c3 = st.columns(3)
        visit_count         = c1.number_input("Visit count",           value=4,   min_value=1, step=1)
        max_interval_years  = c2.number_input("Follow-up duration (yr)", value=2.3, step=0.1)
        visit_frequency     = c3.number_input("Visits per year",       value=1.8, step=0.1)

        st.divider()

        st.subheader("Visual Field — Baseline")
        c1, c2, c3, c4 = st.columns(4)
        vf_mean              = c1.number_input("VF mean (dB)",          value=22.6, step=0.1)
        vf_std               = c2.number_input("VF std",                value=4.4,  step=0.1)
        vf_max               = c3.number_input("VF max (dB)",           value=31.0, step=0.5)
        vf_defect_count      = c4.number_input("Defect count (<15 dB)", value=3,    min_value=0, step=1)
        c5, c6, c7, c8 = st.columns(4)
        vf_defect_ratio      = c5.number_input("Defect ratio",          value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        vf_severe_loss_count = c6.number_input("Severe loss (<5 dB)",   value=0,    min_value=0, step=1)
        vf_si_asymmetry      = c7.number_input("S/I asymmetry",         value=3.3,  step=0.1)
        vf_lr_asymmetry      = c8.number_input("L/R asymmetry",         value=0.2,  step=0.1)

        st.divider()

        st.subheader("Visual Field — Progression")
        c1, c2, c3 = st.columns(3)
        vf_mean_slope = c1.number_input("VF slope (dB/yr)",   value=-0.19, step=0.01, format="%.2f")
        vf_delta      = c2.number_input("VF delta (last-first)", value=-0.31, step=0.1)
        vf_final_mean = c3.number_input("VF mean at last visit", value=22.1, step=0.1)

        submitted = st.form_submit_button("Predict Progression Risk", use_container_width=True)

    if submitted:
        payload = {
            "age": age, "gender": gender, "iop": iop, "cct": cct,
            "category_of_glaucoma": glaucoma_cat, "oct_rnfl_thickness_mean": rnfl,
            "iop_mean": iop_mean, "iop_max": iop_max, "iop_min": iop_min,
            "iop_std": iop_std, "iop_slope": iop_slope,
            "visit_count": visit_count, "max_interval_years": max_interval_years,
            "visit_frequency": visit_frequency,
            "vf_mean": vf_mean, "vf_std": vf_std, "vf_max": vf_max,
            "vf_defect_count": vf_defect_count, "vf_defect_ratio": vf_defect_ratio,
            "vf_severe_loss_count": vf_severe_loss_count,
            "vf_si_asymmetry": vf_si_asymmetry, "vf_lr_asymmetry": vf_lr_asymmetry,
            "vf_mean_slope": vf_mean_slope, "vf_delta": vf_delta,
            "vf_final_mean": vf_final_mean,
        }
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            category = result["risk_category"]
            score    = result["risk_score"]

            color = {"low": "success", "moderate": "warning", "high": "error"}[category]
            badge = {"low": "🟢 Low", "moderate": "🟡 Moderate", "high": "🔴 High"}[category]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Score", f"{score:.1%}")
            m2.metric("Risk Category", badge)
            m3.metric("Request ID", result["request_id"][:8] + "…")

            getattr(st, color)(f"**Recommendation:** {result['recommendation']}")

            st.progress(score, text=f"Calibrated progression probability: {score:.1%}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Check the endpoint in the sidebar.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Population Analytics Tab
with tab_analytics:
    st.header("Population Analytics")

    try:
        df = load_predictions(s3_bucket)
    except Exception as e:
        st.error(f"Could not load predictions from S3: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No predictions logged yet. Make a prediction in the Risk Assessment tab to populate this view.")
    else:
        # Summary metrics
        total = len(df)
        high_risk = (df["risk_category"] == "high").sum()
        avg_score = df["risk_score"].mean()
        cat_counts = df["risk_category"].value_counts()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Assessments", total)
        m2.metric("High Risk", int(high_risk), delta=f"{high_risk/total:.0%} of total")
        m3.metric("Mean Risk Score", f"{avg_score:.1%}")
        m4.metric("Moderate + High", int((df["risk_category"] != "low").sum()))

        st.divider()

        # charts
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Risk Category Distribution")
            color_map = {"low": "#2ecc71", "moderate": "#f39c12", "high": "#e74c3c"}
            fig_pie = px.pie(
                names=cat_counts.index,
                values=cat_counts.values,
                color=cat_counts.index,
                color_discrete_map=color_map,
                hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Risk Score Distribution")
            fig_hist = px.histogram(
                df, x="risk_score", nbins=20,
                color_discrete_sequence=["#3498db"],
                labels={"risk_score": "Calibrated Risk Score"},
            )
            fig_hist.add_vline(x=0.10, line_dash="dash", line_color="#2ecc71",
                               annotation_text="Low/Moderate")
            fig_hist.add_vline(x=0.20, line_dash="dash", line_color="#e74c3c",
                               annotation_text="Moderate/High")
            fig_hist.update_layout(margin=dict(t=0, b=0), bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        # assessment trends over time
        if df["date"].nunique() > 1:
            st.subheader("Assessments Over Time")
            daily = df.groupby(["date", "risk_category"]).size().reset_index(name="count")
            fig_time = px.bar(
                daily, x="date", y="count", color="risk_category",
                color_discrete_map=color_map,
                labels={"date": "Date", "count": "Assessments", "risk_category": "Category"},
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # clinical features vs risk score
        st.subheader("Clinical Features vs Risk Score")
        feat_col = st.selectbox(
            "Feature",
            ["vf_mean_slope", "vf_delta", "vf_final_mean", "iop_mean",
             "iop_slope", "oct_rnfl_thickness_mean", "age", "vf_defect_ratio"],
        )
        if feat_col in df.columns:
            fig_scatter = px.scatter(
                df, x=feat_col, y="risk_score",
                color="risk_category", color_discrete_map=color_map,
                hover_data=["request_id", "timestamp"],
                labels={"risk_score": "Risk Score", feat_col: feat_col},
            )
            fig_scatter.update_layout(margin=dict(t=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # high risk patients
        st.subheader("Recent High-Risk Assessments")
        high_df = (
            df[df["risk_category"] == "high"]
            .sort_values("timestamp", ascending=False)
            [["timestamp", "request_id", "risk_score", "age", "vf_mean_slope",
              "vf_delta", "iop_mean"]]
            .head(10)
        )
        if high_df.empty:
            st.success("No high-risk assessments on record.")
        else:
            st.dataframe(
                high_df.style.format({
                    "risk_score": "{:.1%}",
                    "vf_mean_slope": "{:.2f}",
                    "vf_delta": "{:.2f}",
                    "iop_mean": "{:.1f}",
                }),
                use_container_width=True,
            )

        # raw data 
        with st.expander("View all prediction logs"):
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)


# Model Info Tab
with tab_model:
    st.header("Model Information")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean AUC",    "0.745")
    m2.metric("PR-AUC",      "0.307")
    m3.metric("Brier Score", "0.112")
    m4.metric("Optimal threshold", "0.20")

    st.caption(
        "XGBoost (100 estimators, max depth 3) · 5-fold GroupKFold · "
        "Isotonic regression calibration · scale_pos_weight per fold"
    )

    st.divider()

    col_shap, col_imp = st.columns(2)

    with col_shap:
        st.subheader("SHAP Feature Impact")
        try:
            img = Image.open("outputs/shap_summary.png")
            st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.info("Run training to generate SHAP plot.")

    with col_imp:
        st.subheader("Feature Importance")
        try:
            img = Image.open("outputs/feature_importance.png")
            st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.info("Run training to generate feature importance plot.")

    st.divider()
    st.subheader("Calibration & Precision-Recall")
    col_cal, col_pr = st.columns(2)
    with col_cal:
        try:
            st.image(Image.open("outputs/calibration.png"), use_container_width=True)
        except FileNotFoundError:
            st.info("Run training to generate calibration plot.")
    with col_pr:
        try:
            st.image(Image.open("outputs/pr_curve.png"), use_container_width=True)
        except FileNotFoundError:
            st.info("Run training to generate PR curve.")
