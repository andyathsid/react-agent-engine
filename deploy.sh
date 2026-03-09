#!/bin/bash

set -e

if [[ -z "$PROJECT_ID" || -z "$REGION" || -z "$REPO_NAME" || -z "$SERVICE_NAME" ]]; then
  echo "Please export PROJECT_ID, REGION, REPO_NAME, and SERVICE_NAME before running this script."
  exit 1
fi

# Check if Artifact Registry repo exists, create if not
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository: $REPO_NAME in $REGION"
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION"
else
  echo "Artifact Registry repository $REPO_NAME already exists in $REGION"
fi

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"

# Read .env and build key=value,... string for Cloud Run
ENV_FILE="${ENV_FILE:-.env}"
ENV_VARS=""
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value%\"}" && value="${value#\"}"
      value="${value%\'}" && value="${value#\'}"
      value="${value%$'\r'}"
      ENV_VARS+="${key}=${value},"
    fi
  done < "$ENV_FILE"
  ENV_VARS="${ENV_VARS%,}"
  echo "Environment variables to be set from $ENV_FILE"
else
  echo "No $ENV_FILE found, skipping env vars."
fi

echo "Building Docker image..."
docker build -t "$IMAGE" .

echo "Pushing Docker image to Artifact Registry..."
docker push "$IMAGE"

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 3600 \
  --min-instances 1 \
  --cpu-boost \
  $( [[ -n "$ENV_VARS" ]] && echo --set-env-vars "$ENV_VARS" )

echo "Deployment complete."
