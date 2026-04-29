#!/bin/bash
# deploy.sh — full AWS deployment: S3, SNS, Athena, CloudWatch, ECR, ECS Fargate.
# Run from the project root: bash deploy/deploy.sh
set -euo pipefail

# ─── Config ────────────────────────────────────────────────────────────────────
REPO_NAME="glaucoma-risk-api"
CLUSTER_NAME="glaucoma-cluster"
SERVICE_NAME="glaucoma-service"
TASK_FAMILY="glaucoma-task"
CONTAINER_NAME="glaucoma-api"
LOG_GROUP="/ecs/glaucoma-risk-api"
ALERT_EMAIL="your-alert-email@example.com"
PORT=8000
CPU=512
MEMORY=1024
# ────────────────────────────────────────────────────────────────────────────────

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
IMAGE_URI="${ECR_URI}:latest"
S3_BUCKET="glaucoma-risk-${ACCOUNT_ID}"

echo "==> Account: ${ACCOUNT_ID}  Region: ${REGION}"

# ── 1. S3 bucket ───────────────────────────────────────────────────────────────
echo "==> Ensuring S3 bucket exists..."
aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null || \
  aws s3api create-bucket \
    --bucket "${S3_BUCKET}" \
    --region "${REGION}" \
    --create-bucket-configuration LocationConstraint="${REGION}"

echo "==> Uploading model artifact to S3..."
aws s3 cp models/model_calibrated.pkl "s3://${S3_BUCKET}/models/model_calibrated.pkl"

# ── 2. SNS topic + email subscription ─────────────────────────────────────────
echo "==> Creating SNS topic..."
SNS_TOPIC_ARN=$(aws sns create-topic \
  --name "glaucoma-high-risk-alerts" \
  --query "TopicArn" --output text)

# subscribe only if not already subscribed
ALREADY=$(aws sns list-subscriptions-by-topic \
  --topic-arn "${SNS_TOPIC_ARN}" \
  --query "Subscriptions[?Endpoint=='${ALERT_EMAIL}'].SubscriptionArn" \
  --output text)
if [ -z "${ALREADY}" ]; then
  aws sns subscribe \
    --topic-arn "${SNS_TOPIC_ARN}" \
    --protocol email \
    --notification-endpoint "${ALERT_EMAIL}"
  echo "==> Subscription confirmation sent to ${ALERT_EMAIL} — check your inbox."
else
  echo "==> Email already subscribed."
fi
echo "==> SNS topic: ${SNS_TOPIC_ARN}"

# ── 3. Athena / Glue database + external table ────────────────────────────────
ATHENA_RESULTS="s3://${S3_BUCKET}/athena-results/"

echo "==> Creating Glue database..."
aws glue create-database \
  --database-input '{"Name":"glaucoma_db"}' 2>/dev/null || true

echo "==> Creating Athena predictions table..."
CREATE_TABLE_SQL="
CREATE EXTERNAL TABLE IF NOT EXISTS glaucoma_db.predictions (
  request_id             STRING,
  timestamp              STRING,
  age                    DOUBLE,
  gender                 INT,
  iop                    DOUBLE,
  cct                    DOUBLE,
  category_of_glaucoma   INT,
  oct_rnfl_thickness_mean DOUBLE,
  iop_mean               DOUBLE,
  iop_max                DOUBLE,
  iop_min                DOUBLE,
  iop_std                DOUBLE,
  iop_slope              DOUBLE,
  visit_count            INT,
  max_interval_years     DOUBLE,
  visit_frequency        DOUBLE,
  vf_mean                DOUBLE,
  vf_std                 DOUBLE,
  vf_max                 DOUBLE,
  vf_defect_count        INT,
  vf_defect_ratio        DOUBLE,
  vf_severe_loss_count   INT,
  vf_si_asymmetry        DOUBLE,
  vf_lr_asymmetry        DOUBLE,
  vf_mean_slope          DOUBLE,
  vf_delta               DOUBLE,
  vf_final_mean          DOUBLE,
  risk_score             DOUBLE,
  risk_category          STRING
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES ('ignore.malformed.json' = 'true')
LOCATION 's3://${S3_BUCKET}/predictions/'
TBLPROPERTIES ('has_encrypted_data'='false');
"

aws athena start-query-execution \
  --query-string "${CREATE_TABLE_SQL}" \
  --query-execution-context Database=glaucoma_db \
  --result-configuration OutputLocation="${ATHENA_RESULTS}" \
  > /dev/null
echo "==> Athena table registered. Query predictions with:"
echo "    SELECT * FROM glaucoma_db.predictions LIMIT 10;"

# ── 4. ECR repository ──────────────────────────────────────────────────────────
echo "==> Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" \
  > /dev/null 2>&1 || \
  aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --image-scanning-configuration scanOnPush=true \
    --region "${REGION}"

# ── 5. Docker build + push ─────────────────────────────────────────────────────
echo "==> Logging in to ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Building image (linux/amd64)..."
docker build --platform linux/amd64 -t "${REPO_NAME}:latest" .

echo "==> Pushing to ECR..."
docker tag "${REPO_NAME}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

# ── 6. CloudWatch log group ────────────────────────────────────────────────────
echo "==> Ensuring CloudWatch log group exists..."
aws logs create-log-group --log-group-name "${LOG_GROUP}" --region "${REGION}" \
  2>/dev/null || true

# ── 7. IAM execution role ──────────────────────────────────────────────────────
EXEC_ROLE_NAME="ecsTaskExecutionRole"
EXEC_ROLE_ARN=$(aws iam get-role --role-name "${EXEC_ROLE_NAME}" \
  --query "Role.Arn" --output text 2>/dev/null) || {
  EXEC_ROLE_ARN=$(aws iam create-role \
    --role-name "${EXEC_ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' --query "Role.Arn" --output text)
  aws iam attach-role-policy --role-name "${EXEC_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── 8. IAM task role (S3 + CloudWatch + SNS) ──────────────────────────────────
echo "==> Ensuring ECS task role exists..."
TASK_ROLE_NAME="glaucomaTaskRole"
TASK_ROLE_ARN=$(aws iam get-role --role-name "${TASK_ROLE_NAME}" \
  --query "Role.Arn" --output text 2>/dev/null) || {
  TASK_ROLE_ARN=$(aws iam create-role \
    --role-name "${TASK_ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' --query "Role.Arn" --output text)
}

aws iam put-role-policy \
  --role-name "${TASK_ROLE_NAME}" \
  --policy-name "GlaucomaAPIAccess" \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],
       \"Resource\":\"arn:aws:s3:::${S3_BUCKET}/models/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\"],
       \"Resource\":\"arn:aws:s3:::${S3_BUCKET}/predictions/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:PutMetricData\"],
       \"Resource\":\"*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"sns:Publish\"],
       \"Resource\":\"${SNS_TOPIC_ARN}\"}
    ]
  }"
echo "==> Task role: ${TASK_ROLE_ARN}"

# ── 9. ECS task definition ─────────────────────────────────────────────────────
echo "==> Registering task definition..."
TASK_DEF=$(cat <<EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${CPU}",
  "memory": "${MEMORY}",
  "executionRoleArn": "${EXEC_ROLE_ARN}",
  "taskRoleArn": "${TASK_ROLE_ARN}",
  "containerDefinitions": [{
    "name": "${CONTAINER_NAME}",
    "image": "${IMAGE_URI}",
    "essential": true,
    "portMappings": [{"containerPort": ${PORT}, "protocol": "tcp"}],
    "environment": [
      {"name": "S3_BUCKET",         "value": "${S3_BUCKET}"},
      {"name": "SNS_TOPIC_ARN",     "value": "${SNS_TOPIC_ARN}"},
      {"name": "AWS_DEFAULT_REGION","value": "${REGION}"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group":         "${LOG_GROUP}",
        "awslogs-region":        "${REGION}",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
EOF
)

TASK_ARN=$(aws ecs register-task-definition \
  --cli-input-json "${TASK_DEF}" \
  --query "taskDefinition.taskDefinitionArn" --output text)
echo "==> Task definition: ${TASK_ARN}"

# ── 10. ECS cluster ────────────────────────────────────────────────────────────
echo "==> Ensuring ECS cluster exists..."
aws ecs describe-clusters --clusters "${CLUSTER_NAME}" \
  --query "clusters[?status=='ACTIVE'].clusterName" --output text \
  | grep -q "${CLUSTER_NAME}" || aws ecs create-cluster --cluster-name "${CLUSTER_NAME}"

# ── 11. Networking ─────────────────────────────────────────────────────────────
echo "==> Resolving default VPC and subnets..."
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=${VPC_ID}" \
  --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

SG_NAME="glaucoma-api-sg"
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)

if [ "${SG_ID}" = "None" ] || [ -z "${SG_ID}" ]; then
  echo "==> Creating security group..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "Glaucoma Risk API - allow inbound port ${PORT}" \
    --vpc-id "${VPC_ID}" \
    --query "GroupId" --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" --protocol tcp --port "${PORT}" --cidr 0.0.0.0/0
fi
echo "==> VPC: ${VPC_ID}  SG: ${SG_ID}"

# ── 12. ECS service — create or update ────────────────────────────────────────
NETWORK_CONFIG="awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}"
SERVICE_STATUS=$(aws ecs describe-services \
  --cluster "${CLUSTER_NAME}" --services "${SERVICE_NAME}" \
  --query "services[0].status" --output text 2>/dev/null)

if [ "${SERVICE_STATUS}" = "ACTIVE" ]; then
  echo "==> Updating existing service..."
  aws ecs update-service \
    --cluster "${CLUSTER_NAME}" --service "${SERVICE_NAME}" \
    --task-definition "${TASK_ARN}" --force-new-deployment > /dev/null
else
  echo "==> Creating service..."
  aws ecs create-service \
    --cluster "${CLUSTER_NAME}" --service-name "${SERVICE_NAME}" \
    --task-definition "${TASK_ARN}" --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "${NETWORK_CONFIG}" > /dev/null
fi

# ── 13. Print endpoint ─────────────────────────────────────────────────────────
echo ""
echo "==> Deployment complete. Waiting for task to start..."
sleep 20

TASK_ARN_RUNNING=$(aws ecs list-tasks \
  --cluster "${CLUSTER_NAME}" --service-name "${SERVICE_NAME}" \
  --query "taskArns[0]" --output text)

if [ "${TASK_ARN_RUNNING}" != "None" ] && [ -n "${TASK_ARN_RUNNING}" ]; then
  ENI_ID=$(aws ecs describe-tasks \
    --cluster "${CLUSTER_NAME}" --tasks "${TASK_ARN_RUNNING}" \
    --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
    --output text)
  PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids "${ENI_ID}" \
    --query "NetworkInterfaces[0].Association.PublicIp" --output text 2>/dev/null || echo "pending")
  echo ""
  echo "  S3 bucket:     s3://${S3_BUCKET}"
  echo "  SNS topic:     ${SNS_TOPIC_ARN}"
  echo "  Athena DB:     glaucoma_db.predictions"
  echo "  CW namespace:  GlaucomaRiskAPI"
  echo ""
  echo "  API endpoint:  http://${PUBLIC_IP}:${PORT}"
  echo "  Health check:  http://${PUBLIC_IP}:${PORT}/health"
  echo "  Docs:          http://${PUBLIC_IP}:${PORT}/docs"
else
  echo "  Task not yet running — check ECS console or re-run."
fi
