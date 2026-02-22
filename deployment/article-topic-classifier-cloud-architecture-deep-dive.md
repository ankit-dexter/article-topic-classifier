# Article Topic Classifier

# Cloud Architecture & Infrastructure Deep Dive

*Last Updated: 2026-02-13 20:15 UTC*

------------------------------------------------------------------------

# 1️⃣ Cloud Architecture Overview

This document explains how the following AWS components interact:

-   Amazon ECR
-   Amazon ECS (Fargate)
-   IAM Roles
-   VPC & Networking
-   Security Groups
-   Amazon S3 (Model Registry)

The goal is to understand not just *what* was created --- but *how
everything connects* inside AWS.

------------------------------------------------------------------------

# 2️⃣ High-Level Cloud Architecture Diagram

                    ┌───────────────────────────┐
                    │        Internet           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Security Group (Port 8000)
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   ECS Service (Fargate)  │
                    │  ───────────────────────  │
                    │   ECS Task Definition     │
                    │        │                  │
                    │        ▼                  │
                    │   Docker Container        │
                    │   (FastAPI + Model)       │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
     │     ECR      │     │      IAM     │     │      S3      │
     │ Docker Image │     │ Task Role    │     │ Model Store  │
     └──────────────┘     └──────────────┘     └──────────────┘

------------------------------------------------------------------------

# 3️⃣ Component-by-Component Explanation

------------------------------------------------------------------------

## 🔹 Amazon ECR (Elastic Container Registry)

### What it stores:

-   The Docker image built locally
-   Tagged version: `article-topic-classifier-inference:latest`

### Why it is required:

ECS cannot run local Docker images. It must pull images from a container
registry.

### How it connects:

1.  ECS Task starts.
2.  Fargate requests image from ECR.
3.  ECR verifies IAM permissions.
4.  Image layers are downloaded into Fargate runtime.

ECR acts as the **single source of truth for container images**.

------------------------------------------------------------------------

## 🔹 Amazon ECS (Elastic Container Service)

ECS is the container orchestration service.

We created: - ECS Cluster (Fargate only) - ECS Task Definition - ECS
Service

### Relationships:

  Component         Role
  ----------------- -------------------------------
  Cluster           Logical grouping of services
  Task Definition   Blueprint of container
  Service           Keeps container running
  Task              Running instance of container

### How It Works Internally:

1.  You define Task Definition.
2.  Service launches a Task.
3.  Fargate provisions compute runtime.
4.  Container is started.
5.  Health checks monitor status.

If container crashes → ECS restarts it automatically.

------------------------------------------------------------------------

## 🔹 Fargate (Serverless Compute)

Fargate removes EC2 management.

Instead of:

-   Launching EC2
-   Managing capacity
-   Choosing instance types

Fargate:

-   Allocates CPU & memory per task
-   Runs container in isolated environment
-   Bills per vCPU + memory usage

This simplified deployment and removed EC2 capacity issues.

------------------------------------------------------------------------

## 🔹 IAM Roles (Critical Security Layer)

There are multiple IAM roles involved.

### 1️⃣ IAM User (CLI)

Used only for: - Pushing images to ECR - Creating infrastructure

Never used at runtime.

------------------------------------------------------------------------

### 2️⃣ ECS Task Execution Role

Used by ECS to: - Pull image from ECR - Write logs to CloudWatch

Without this role: ECS cannot start containers.

------------------------------------------------------------------------

### 3️⃣ ECS Task Role (Future Use)

Used by container itself to: - Access S3 model bucket - Download model
artifacts

This prevents hardcoding AWS credentials in code.

IAM roles provide secure, temporary credentials.

------------------------------------------------------------------------

## 🔹 Amazon S3 (Model Registry)

S3 stores:

    models/topicclf/
        2026-03-01/
            model.tar.gz
        latest.json

### Why S3:

-   Durable storage
-   Versioned artifacts
-   Decouples model from container image

### How it connects:

1.  Container starts.
2.  Reads MODEL_URI.
3.  Downloads model from S3.
4.  Loads into memory.

This allows model updates without rebuilding Docker image.

------------------------------------------------------------------------

## 🔹 VPC & Networking

ECS Fargate runs inside a VPC.

Each task receives:

-   Elastic Network Interface (ENI)
-   Private IP
-   Optional public IP

Security Group controls:

-   Inbound port 8000 (API access)
-   Outbound internet access (ECR & S3 access)

------------------------------------------------------------------------

# 4️⃣ End-to-End Request Flow

1.  Client sends HTTP request.
2.  Request reaches public IP.
3.  Security Group allows port 8000.
4.  Traffic reaches Fargate task.
5.  FastAPI processes request.
6.  Model performs inference.
7.  Response returned.

------------------------------------------------------------------------

# 5️⃣ Deployment Flow

### Local → Cloud

1.  Docker build locally.
2.  Authenticate with ECR.
3.  Push image to ECR.
4.  Create ECS Task Definition.
5.  Create ECS Service.
6.  ECS pulls image from ECR.
7.  Container runs on Fargate.

------------------------------------------------------------------------

# 6️⃣ Why This Architecture Is Production-Aligned

This architecture enables:

-   Stateless containers
-   Immutable infrastructure
-   Versioned deployments
-   Auto-restart on failure
-   Model version decoupling
-   Secure IAM-based access
-   Horizontal scalability (future)

------------------------------------------------------------------------

# 7️⃣ Current Cloud State

✔ Docker image stored in ECR\
✔ Fargate cluster created\
✔ IAM roles configured\
✔ Networking isolated via Security Groups\
✔ Model storage externalized to S3

------------------------------------------------------------------------

# 8️⃣ Future Extensions

This setup easily supports:

-   CI/CD via GitHub Actions
-   Blue/Green deployments
-   Canary releases
-   Auto-scaling services
-   Scheduled retraining pipelines
-   Spot training instances

------------------------------------------------------------------------

# 9️⃣ Final Architecture Summary

ECR → Stores container\
ECS → Orchestrates container\
Fargate → Runs container\
IAM → Secures access\
S3 → Stores model\
Security Groups → Control network

Everything is loosely coupled and cloud-native.

------------------------------------------------------------------------

End of Document.
