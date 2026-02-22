# Article Topic Classifier -- Deployment & AWS Setup Documentation

*Last Updated: 2026-02-13 20:06 UTC*

------------------------------------------------------------------------

# 1️⃣ Project Overview

This document describes the complete process followed to:

-   Build a production-ready ML inference API
-   Containerize it using Docker
-   Push image to Amazon ECR
-   Configure IAM roles
-   Create an ECS cluster (Fargate)
-   Prepare for orchestrated deployment

The system supports: - Local model loading - S3-based model registry
loading - Confidence-aware classification decisions

------------------------------------------------------------------------

# 2️⃣ Local Development Steps

## Install Dependencies

``` bash
pip install -r requirements.txt
```

Production dependencies include:

-   torch (CPU version)
-   transformers
-   fastapi
-   uvicorn
-   numpy
-   boto3

------------------------------------------------------------------------

## Run Locally

PowerShell:

``` powershell
$env:MODEL_URI="artifacts/distilbert"
uvicorn app:app --reload
```

Test health endpoint:

    http://localhost:8000/health

------------------------------------------------------------------------

# 3️⃣ Dockerization

## Build Docker Image

``` bash
docker build -t article-topic-classifier .
```

Verify image:

``` bash
docker images
```

------------------------------------------------------------------------

# 4️⃣ AWS Setup

## 4.1 Create IAM User (CLI Access)

1.  Go to IAM → Users → Create user
2.  Attach policies:
    -   AmazonEC2ContainerRegistryFullAccess
    -   AmazonECSFullAccess
    -   AmazonS3FullAccess
3.  Create access key (CLI)
4.  Configure locally:

``` powershell
aws configure
```

Verify:

``` powershell
aws sts get-caller-identity
```

------------------------------------------------------------------------

# 5️⃣ Amazon ECR Setup

## Create Repository

Repository name:

    article-topic-classifier-inference

## Login to ECR

``` powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 363283722404.dkr.ecr.us-east-1.amazonaws.com
```

## Tag Image

``` powershell
docker tag article-topic-classifier 363283722404.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest
```

## Push Image

``` powershell
docker push 363283722404.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest
```

------------------------------------------------------------------------

# 6️⃣ ECS Cluster Creation (Fargate)

Due to EC2 managed instance complexity, we selected:

    Fargate only

Cluster name:

    article-topic-cluster-v2

Fargate avoids: - EC2 instance management - Capacity provider
configuration - Instance availability issues

------------------------------------------------------------------------

# 7️⃣ Model Loading Architecture

The API supports:

## Local Mode

    MODEL_URI=artifacts/distilbert

## S3 Mode

    MODEL_URI=s3://bucket/models/topicclf/latest.json

Model resolution flow:

1.  Read latest.json
2.  Extract artifact_uri
3.  Download model.tar.gz
4.  Extract locally
5.  Load with HuggingFace

------------------------------------------------------------------------

# 8️⃣ Security Practices Applied

-   No AWS keys stored in code
-   IAM roles used for ECS
-   Model versioning via S3
-   Separation of training & inference
-   Reproducible Docker builds

------------------------------------------------------------------------

# 9️⃣ Current System State

✅ Dockerized inference service\
✅ Image stored in ECR\
✅ ECS cluster created (Fargate)\
✅ IAM roles configured\
✅ S3-compatible model loading\
✅ Production-ready API structure

------------------------------------------------------------------------

# 🔜 Next Steps

-   Create Fargate Task Definition
-   Deploy ECS Service
-   Attach security group for port 8000
-   Configure CI/CD via GitHub Actions
-   Implement monthly retraining pipeline

------------------------------------------------------------------------

# 🧠 Architectural Summary

We transitioned from:

Manual EC2 Docker →\
ECR Registry →\
ECS Orchestration →\
Cloud-ready MLOps Foundation

------------------------------------------------------------------------

End of documentation.
