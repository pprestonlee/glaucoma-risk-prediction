import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# config
S3_BUCKET      = os.environ.get("S3_BUCKET", "")
SNS_TOPIC_ARN  = os.environ.get("SNS_TOPIC_ARN", "")
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
MODEL_S3_KEY   = "models/model_calibrated.pkl"
LOCAL_MODEL    = "models/model_calibrated.pkl"

FEATURE_ORDER = [
    "age", "gender", "iop", "cct", "category_of_glaucoma", "oct_rnfl_thickness_mean",
    "iop_mean", "iop_max", "iop_min", "iop_std", "iop_slope",
    "visit_count", "max_interval_years", "visit_frequency",
    "vf_mean", "vf_std", "vf_max", "vf_defect_count", "vf_defect_ratio",
    "vf_severe_loss_count", "vf_si_asymmetry", "vf_lr_asymmetry",
    "age_x_iop", "rnfl_x_vf_mean", "iop_x_vf_mean",
    "vf_mean_slope", "vf_delta", "vf_final_mean",
]

_model = _calibrator = _s3 = _cw = _sns = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _calibrator, _s3, _cw, _sns

    _s3  = boto3.client("s3",          region_name=AWS_REGION)
    _cw  = boto3.client("cloudwatch",  region_name=AWS_REGION)
    _sns = boto3.client("sns",         region_name=AWS_REGION)

    if S3_BUCKET:
        logger.info("Loading model from s3://%s/%s", S3_BUCKET, MODEL_S3_KEY)
        buf = io.BytesIO()
        _s3.download_fileobj(S3_BUCKET, MODEL_S3_KEY, buf)
        buf.seek(0)
        artifacts = joblib.load(buf)
    else:
        logger.info("Loading model from %s", LOCAL_MODEL)
        artifacts = joblib.load(LOCAL_MODEL)

    _model      = artifacts["model"]
    _calibrator = artifacts["calibrator"]
    yield


app = FastAPI(
    title="Glaucoma Progression Risk API",
    description="Predicts probability of glaucoma progression from clinical and visual field features.",
    version="1.1.0",
    lifespan=lifespan,
)


# schema for request and response 
class PredictionInput(BaseModel):
    # clinical baseline
    age: float                    = Field(..., ge=0, le=120, description="Age in years")
    gender: int                   = Field(..., ge=0, le=1,   description="1=Male 0=Female")
    iop: float                    = Field(..., description="Baseline IOP (mmHg)")
    cct: float                    = Field(..., description="Central corneal thickness (um)")
    category_of_glaucoma: int     = Field(..., ge=0, le=1,   description="0=OAG 1=ACG")
    oct_rnfl_thickness_mean: float= Field(..., description="Mean RNFL thickness (um)")
    # longitudinal IOP
    iop_mean: float               = Field(..., description="Mean IOP across follow-up (mmHg)")
    iop_max: float
    iop_min: float
    iop_std: float
    iop_slope: float              = Field(..., description="IOP trend (mmHg/year)")
    # visit history
    visit_count: int              = Field(..., ge=1)
    max_interval_years: float     = Field(..., description="Total follow-up duration (years)")
    visit_frequency: float        = Field(..., description="Visits per year")
    # VF baseline
    vf_mean: float                = Field(..., description="Mean VF sensitivity (dB)")
    vf_std: float
    vf_max: float
    vf_defect_count: int          = Field(..., description="Points < 15 dB")
    vf_defect_ratio: float        = Field(..., ge=0.0, le=1.0)
    vf_severe_loss_count: int     = Field(..., description="Points < 5 dB")
    vf_si_asymmetry: float        = Field(..., description="Superior-inferior asymmetry")
    vf_lr_asymmetry: float        = Field(..., description="Left-right asymmetry")
    # VF progression
    vf_mean_slope: float          = Field(..., description="VF sensitivity trend (dB/year)")
    vf_delta: float               = Field(..., description="VF change last minus first visit")
    vf_final_mean: float          = Field(..., description="VF mean at last visit (dB)")


class PredictionOutput(BaseModel):
    request_id:     str
    risk_score:     float = Field(..., description="Calibrated progression probability (0-1)")
    risk_category:  str   = Field(..., description="low | moderate | high")
    recommendation: str


# logging
def _log_to_s3(request_id: str, inputs: dict, cal_prob: float, category: str) -> None:
    """Write one JSON record per prediction to S3 under a Hive-partitioned prefix."""
    if not S3_BUCKET:
        return
    try:
        now = datetime.now(timezone.utc)
        record = {
            "request_id": request_id,
            "timestamp": now.isoformat(),
            **inputs,
            "risk_score": cal_prob,
            "risk_category": category,
        }
        key = (
            f"predictions/year={now.year}/month={now.month:02d}"
            f"/day={now.day:02d}/{request_id}.json"
        )
        _s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(record),
            ContentType="application/json",
        )
    except Exception:
        logger.exception("S3 prediction log failed")

# push metrics to CloudWatch 
def _push_metrics(cal_prob: float, category: str) -> None:
    try:
        _cw.put_metric_data(
            Namespace="GlaucomaRiskAPI",
            MetricData=[
                {"MetricName": "PredictionCount",
                 "Value": 1.0, "Unit": "Count"},
                {"MetricName": "RiskScore",
                 "Value": cal_prob, "Unit": "None"},
                {"MetricName": "HighRiskCount",
                 "Value": 1.0 if category == "high" else 0.0, "Unit": "Count"},
            ],
        )
    except Exception:
        logger.exception("CloudWatch metric push failed")

# SNS alerting for high-risk patients
def _alert_high_risk(request_id: str, cal_prob: float) -> None:
    if not SNS_TOPIC_ARN:
        return
    try:
        _sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="[Glaucoma Risk API] High Progression Risk Detected",
            Message=(
                f"A high-risk glaucoma progression prediction was generated.\n\n"
                f"Request ID:  {request_id}\n"
                f"Risk Score:  {cal_prob:.4f}\n"
                f"Category:    high\n"
                f"Timestamp:   {datetime.now(timezone.utc).isoformat()}"
            ),
        )
    except Exception:
        logger.exception("SNS alert failed")


# endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(body: PredictionInput, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    data = body.model_dump()

    data["age_x_iop"]      = data["age"] * data["iop"]
    data["rnfl_x_vf_mean"] = data["oct_rnfl_thickness_mean"] * data["vf_mean"]
    data["iop_x_vf_mean"]  = data["iop"] * data["vf_mean"]

    X       = pd.DataFrame([data])[FEATURE_ORDER]
    raw     = float(_model.predict_proba(X)[0, 1])
    cal     = float(np.clip(_calibrator.predict([raw])[0], 0.0, 1.0))

    if cal < 0.10:
        category = "low"
        rec = "Routine monitoring; follow-up in 12 months."
    elif cal < 0.20:
        category = "moderate"
        rec = "Increased monitoring; follow-up in 6 months."
    else:
        category = "high"
        rec = "High progression risk; consider treatment adjustment and follow-up in 3 months."

    background_tasks.add_task(_log_to_s3, request_id, body.model_dump(), cal, category)
    background_tasks.add_task(_push_metrics, cal, category)
    if category == "high":
        background_tasks.add_task(_alert_high_risk, request_id, cal)

    return PredictionOutput(
        request_id=request_id,
        risk_score=round(cal, 4),
        risk_category=category,
        recommendation=rec,
    )
