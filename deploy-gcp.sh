#!/bin/bash
# Deploy Koda Vocal Health Coach to Google Cloud Run
# Usage: ./deploy-gcp.sh
#
# Prerequisites:
#   brew install --cask google-cloud-sdk
#   gcloud auth login
#   gcloud config set project gen-lang-client-0999911778
#
# This script:
#   1. Enables required GCP services (idempotent)
#   2. Builds the Docker image via Cloud Build
#   3. Deploys to Cloud Run with WebSocket support
#   4. Outputs the live URL

set -euo pipefail

PROJECT_ID="gen-lang-client-0999911778"
REGION="us-central1"
SERVICE_NAME="koda-vocal-coach"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Use system Python 3.14 for gcloud (avoids 3.9 warning)
export CLOUDSDK_PYTHON=/opt/homebrew/bin/python3.14

echo "=== Deploying Koda Vocal Health Coach to Google Cloud Run ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Step 1: Enable required services
echo ">>> Enabling GCP services..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

echo ">>> Services enabled."

# Step 2: Build and push Docker image via Cloud Build
echo ""
echo ">>> Building Docker image via Cloud Build (this takes 3-5 min)..."
gcloud builds submit \
    --tag "${IMAGE}" \
    --project="${PROJECT_ID}" \
    --timeout=600s \
    --quiet

echo ">>> Image built and pushed."

# Step 3: Deploy to Cloud Run
echo ""
echo ">>> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project="${PROJECT_ID}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 3 \
    --timeout 300 \
    --session-affinity \
    --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY:-}" \
    --quiet

echo ""
echo ">>> Deployment complete!"
echo ""

# Step 4: Get the live URL
URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format='value(status.url)')

echo "=== LIVE URL ==="
echo "  App:    ${URL}"
echo "  Debug:  ${URL}/debug"
echo "  Health: ${URL}/health"
echo ""
echo "Test WebSocket: open ${URL} in Chrome, grant mic access, sing!"
