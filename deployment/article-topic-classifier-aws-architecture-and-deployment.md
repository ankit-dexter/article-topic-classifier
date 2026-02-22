# Article Topic Classifier -- Architecture & Deployment Documentation

*Last Updated: 2026-02-13 20:09 UTC*

------------------------------------------------------------------------

# 📌 1. System Overview

This project implements a **production-style ML inference system** for
article topic classification.

The system:

-   Accepts article title + body
-   Uses a fine‑tuned DistilBERT model
-   Produces:
    -   Predicted label
    -   Confidence score
    -   Full probability distribution
    -   Routing decision (auto_accept / needs_review / reject)

The architecture separates:

-   Model training
-   Model storage/versioning
-   Inference service
-   Container orchestration

This separation enables scalability, reproducibility, and safe model
evolution.

------------------------------------------------------------------------

# 🧠 2. Architectural Philosophy

## Why Separate Training and Inference?

In production ML systems:

-   Training is compute-heavy and periodic.
-   Inference must be lightweight and always available.

Separating them allows:

-   Independent scaling
-   Independent deployment
-   Safe rollback of models
-   Reduced operational risk

------------------------------------------------------------------------

# 🏗 3. High-Level Architecture

                     ┌──────────────────────┐
                     │     Client Request   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   ECS (Fargate)      │
                     │  FastAPI Container   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  HuggingFace Model   │
                     │  Loaded from S3      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Amazon S3 Model    │
                     │   Version Registry   │
                     └──────────────────────┘

------------------------------------------------------------------------

# ☁ 4. Why Amazon ECR?

Amazon ECR serves as a **central image registry**.

Without ECR:

-   ECS cannot pull container images.
-   Images remain tied to individual machines.
-   Deployment becomes manual and fragile.

Using ECR ensures:

-   Reproducible builds
-   Versioned container images
-   Decoupling from local machines

------------------------------------------------------------------------

# 🚀 5. Why ECS (Fargate)?

Originally, EC2-managed ECS was attempted.\
However:

-   Instance capacity matching failed.
-   Managed instances added complexity.
-   Infrastructure overhead increased.

We selected **Fargate** because:

-   No EC2 management required
-   No capacity provider complexity
-   Faster deployment
-   Lower operational risk
-   Clean serverless container execution

Fargate runs containers without managing servers.

------------------------------------------------------------------------

# 📦 6. Containerization Strategy

Docker ensures:

-   Environment reproducibility
-   Consistent dependencies
-   Isolation from host OS
-   Portable deployment

Production Docker image includes only:

-   torch (CPU)
-   transformers
-   fastapi
-   uvicorn
-   numpy
-   boto3

Training dependencies are excluded to keep image lean.

------------------------------------------------------------------------

# 📂 7. Model Loading Strategy

The API supports two modes:

## Local Mode (Development)

    MODEL_URI=artifacts/distilbert

Used for:

-   Local testing
-   EC2 prototype stage

------------------------------------------------------------------------

## S3 Mode (Production)

    MODEL_URI=s3://bucket/models/topicclf/latest.json

Process:

1.  Read latest.json
2.  Resolve artifact_uri
3.  Download model.tar.gz
4.  Extract to /tmp
5.  Load model into memory

Why this design?

-   Enables model versioning
-   No Docker rebuild required for model updates
-   Supports rollback by changing latest.json pointer

------------------------------------------------------------------------

# 🔄 8. Deployment Flow

## Step 1 -- Local Development

``` powershell
$env:MODEL_URI="artifacts/distilbert"
uvicorn app:app --reload
```

------------------------------------------------------------------------

## Step 2 -- Docker Build

``` bash
docker build -t article-topic-classifier .
```

------------------------------------------------------------------------

## Step 3 -- Push to ECR

``` powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker tag article-topic-classifier <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest

docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest
```

------------------------------------------------------------------------

## Step 4 -- ECS Cluster (Fargate)

Cluster created using:

    Fargate only

Why?

-   Simpler orchestration
-   Avoid EC2 capacity issues
-   Reduced infrastructure complexity

------------------------------------------------------------------------

# 🔐 9. Security Decisions

-   IAM user created for CLI operations
-   Access keys not stored in source code
-   ECS tasks will later use IAM roles
-   S3 model access controlled via IAM policies

------------------------------------------------------------------------

# 📊 10. Current System State

The system now has:

✔ Dockerized inference API\
✔ Image stored in Amazon ECR\
✔ ECS Fargate cluster created\
✔ IAM roles configured\
✔ S3-compatible model loading\
✔ Confidence-based routing logic

------------------------------------------------------------------------

# 🔮 11. What This Architecture Enables

This design supports:

-   Monthly automated retraining
-   Versioned model promotion
-   CI/CD container deployment
-   Horizontal scaling (future)
-   Blue/green deployments
-   Safe rollback of model versions

------------------------------------------------------------------------

# 🎯 12. Next Architectural Layer

Next steps:

1.  Create Fargate Task Definition
2.  Deploy ECS Service
3.  Attach security group for port 8000
4.  Add CI/CD automation
5.  Implement scheduled monthly retraining pipeline

------------------------------------------------------------------------

# 🏁 Conclusion

We transitioned from:

Manual EC2 container deployment\
→ Registry-backed container image\
→ Orchestrated container execution\
→ Cloud-native ML architecture foundation

This is now a scalable, production-aligned MLOps architecture.

------------------------------------------------------------------------

End of Document.
