# Legalify Backend

## Setup and Installation

<!-- Add setup and installation instructions here -->

## ☁️ Deployment

The backend is containerized using Docker and deployed as a serverless application on **Google Cloud Run**. The deployment process is automated via the `gcloud` CLI using the following core command:

```bash
gcloud run deploy legalify-backend \
  --image [YOUR_ARTIFACT_REGISTRY_IMAGE_URL] \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=[YOUR_PROJECT_ID]"
```

This ensures a scalable, secure, and cost-effective production environment.

```