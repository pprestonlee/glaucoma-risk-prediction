import os
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn  # type: ignore[import]

# config
REGION              = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
ACCOUNT_ID          = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
S3_BUCKET           = f"glaucoma-risk-{ACCOUNT_ID}"
ROLE                = os.environ.get(
    "SAGEMAKER_ROLE",
    f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole",
)
MODEL_PACKAGE_GROUP = "GlaucomaRiskModels"

boto_session = boto3.Session(region_name=REGION)
sm_client    = boto_session.client("sagemaker")
session      = sagemaker.Session(boto_session=boto_session)


def ensure_model_package_group():
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Glaucoma progression risk models (XGBoost + isotonic calibration)",
        )
        print(f"==> Created model package group: {MODEL_PACKAGE_GROUP}")
    except sm_client.exceptions.ClientError as e:
        if "already exists" in str(e).lower() or "ConflictException" in type(e).__name__:
            print(f"==> Model package group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise


def upload_training_data() -> str:
    print("==> Uploading patient_level.csv to S3...")
    uri = session.upload_data(
        "data/processed/patient_level.csv",
        bucket=S3_BUCKET,
        key_prefix="sm-training-data",
    )
    print(f"==> Data at: {uri}")
    return uri


def launch_training_job(data_uri: str) -> SKLearn:
    print("==> Launching SageMaker training job...")
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src/models",
        framework_version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
        instance_count=1,
        role=ROLE,
        output_path=f"s3://{S3_BUCKET}/sm-training/",
        base_job_name="glaucoma-risk",
        sagemaker_session=session,
    )
    estimator.fit({"training": data_uri})
    print(f"==> Training job complete: {estimator.latest_training_job.name}")
    return estimator


def register_model(estimator: SKLearn) -> str:
    print("==> Registering model in Model Registry...")
    model_package = estimator.register(
        content_types=["application/json"],
        response_types=["application/json"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        description=(
            "XGBoost (n=100, depth=3) + IsotonicRegression calibration. "
            "Trained on GRAPE dataset with 5-fold GroupKFold. "
            f"Training job: {estimator.latest_training_job.name}"
        ),
    )
    arn = model_package.model_package_arn
    print(f"\n==> Model registered (pending approval): {arn}")
    return arn


def main():
    ensure_model_package_group()
    data_uri  = upload_training_data()
    estimator = launch_training_job(data_uri)
    arn       = register_model(estimator)

    print("\n" + "=" * 60)
    print("Next step — approve the model, then deploy:")
    print()
    print("  aws sagemaker update-model-package \\")
    print(f"    --model-package-arn {arn} \\")
    print("    --model-approval-status Approved")
    print()
    print("  bash deploy/deploy.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
