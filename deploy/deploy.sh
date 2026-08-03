#!/usr/bin/env bash
# Deploy Philately Assistant to a fresh AWS EC2 instance.
#
# Not run against a live AWS account — reviewed for correctness against the
# AWS CLI docs, but untested end-to-end. Ports 8501 (app) and 3001 (Grafana)
# are opened to 0.0.0.0/0 for simplicity; restrict --cidr in production.
#
# Prerequisites:
#   - AWS CLI configured (`aws configure`) with EC2 permissions
#   - An existing EC2 key pair, with the .pem file in the current directory
#   - .env filled in (OPENAI_API_KEY etc.) — copied to the instance, never committed
#
# Usage:
#   ./deploy/deploy.sh <key-pair-name> [instance-type]
set -euo pipefail

KEY_NAME="${1:?Usage: deploy.sh <key-pair-name> [instance-type]}"
INSTANCE_TYPE="${2:-t3.small}"
REGION="${AWS_REGION:-us-east-1}"
PEM_FILE="${KEY_NAME}.pem"
SG_NAME="philately-assistant-sg"

if [ ! -f "$PEM_FILE" ]; then
  echo "Key file $PEM_FILE not found in current directory." >&2
  exit 1
fi
if [ ! -f .env ]; then
  echo ".env not found — copy .env.example to .env and fill it in first." >&2
  exit 1
fi

echo "Looking up latest Ubuntu 24.04 AMI in $REGION..."
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --region "$REGION" --output text)

SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --region "$REGION" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ]; then
  echo "Creating security group $SG_NAME..."
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "Philately Assistant: SSH + app + Grafana" \
    --region "$REGION" --query "GroupId" --output text)
  for port in 22 8501 3001; do
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
      --protocol tcp --port "$port" --cidr 0.0.0.0/0 --region "$REGION" >/dev/null
  done
fi

echo "Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --user-data "file://$(dirname "$0")/user-data.sh" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=philately-assistant}]' \
  --region "$REGION" \
  --query "Instances[0].InstanceId" --output text)

echo "Instance $INSTANCE_ID launching, waiting for it to reach 'running'..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "Instance at $PUBLIC_IP. Waiting for SSH and cloud-init (Docker install) to finish..."
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$PEM_FILE" ubuntu@"$PUBLIC_IP" \
  'cloud-init status --wait' 2>/dev/null; do
  sleep 5
done

echo "Copying project files (excluding .git, .venv, data/raw)..."
rsync -az --exclude '.git' --exclude '.venv' --exclude 'data/raw' --exclude '__pycache__' \
  -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
  ./ ubuntu@"$PUBLIC_IP":~/philately_assistant/

echo "Building the corpus and starting the stack on the instance..."
ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" bash -s <<'REMOTE'
set -e
cd ~/philately_assistant
docker compose --profile ingest run --rm ingest
docker compose up -d --build
REMOTE

echo
echo "Done."
echo "  App:     http://$PUBLIC_IP:8501"
echo "  Grafana: http://$PUBLIC_IP:3001"
echo
echo "Tear down with: ./deploy/teardown.sh"
