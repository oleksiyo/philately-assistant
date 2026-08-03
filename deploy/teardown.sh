#!/usr/bin/env bash
# Terminate the EC2 instance(s) created by deploy.sh, to stop billing.
#
# Usage:
#   ./deploy/teardown.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"

INSTANCE_IDS=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=philately-assistant" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[].Instances[].InstanceId" --output text --region "$REGION")

if [ -z "$INSTANCE_IDS" ]; then
  echo "No philately-assistant instances found in $REGION."
  exit 0
fi

echo "Terminating: $INSTANCE_IDS"
aws ec2 terminate-instances --instance-ids $INSTANCE_IDS --region "$REGION" >/dev/null
aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS --region "$REGION"
echo "Terminated."
