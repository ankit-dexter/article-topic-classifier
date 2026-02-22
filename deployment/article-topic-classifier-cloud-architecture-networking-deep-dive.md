# Article Topic Classifier

# Cloud Architecture, Networking & Security Deep Dive

*Last Updated: 2026-02-13 20:43 UTC*

------------------------------------------------------------------------

# 1️⃣ Architecture Overview

This document provides a detailed explanation of how all AWS components
interact inside the cloud environment, including:

-   Amazon ECR
-   Amazon ECS (Fargate)
-   IAM Roles
-   VPC & Subnets
-   Elastic Network Interfaces (ENI)
-   Security Groups
-   Amazon S3 (Model Registry)
-   CloudWatch Logs

The goal is to clearly understand not only what was created --- but how
traffic, security, and permissions flow internally within AWS.

------------------------------------------------------------------------

# 2️⃣ Updated Cloud Architecture Diagram (Accurate Networking Flow)

                       ┌──────────────────────┐
                       │       Internet       │
                       └───────────┬──────────┘
                                   │
                                   ▼
                       ┌──────────────────────┐
                       │   Internet Gateway   │
                       └───────────┬──────────┘
                                   │
                                   ▼
                       ┌──────────────────────┐
                       │     Public Subnet    │
                       └───────────┬──────────┘
                                   │
                                   ▼
                       ┌────────────────────────────┐
                       │  ENI (Fargate Task ENI)    │
                       │  Security Group Attached   │
                       └───────────┬────────────────┘
                                   │
                                   ▼
                       ┌──────────────────────┐
                       │   Docker Container   │
                       │   FastAPI (Port 8000)│
                       └───────────┬──────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
    ┌──────────────┐       ┌──────────────┐         ┌──────────────┐
    │     ECR      │       │     IAM      │         │      S3      │
    │ Container    │       │ Task Roles   │         │ Model Store  │
    │ Registry     │       │ Permissions  │         │ Registry     │
    └──────────────┘       └──────────────┘         └──────────────┘

------------------------------------------------------------------------

# 3️⃣ VPC & Networking Explained

## 🔹 VPC (Virtual Private Cloud)

All ECS Fargate tasks run inside a VPC.

The VPC provides: - Network isolation - Routing control - Internet
Gateway - Subnets - Security Groups

------------------------------------------------------------------------

## 🔹 Subnets

Your Fargate task was launched in:

-   Public Subnets

Public subnets allow: - Internet-bound traffic - Public IP assignment

------------------------------------------------------------------------

## 🔹 Internet Gateway

The Internet Gateway enables: - External internet traffic to reach the
VPC - Outbound traffic to ECR & S3

Without an Internet Gateway: - Public API access would fail - Image
pulls from ECR would fail

------------------------------------------------------------------------

## 🔹 Elastic Network Interface (ENI)

When a Fargate task starts:

1.  AWS creates an ENI
2.  Assigns private IP
3.  Optionally assigns public IP
4.  Attaches Security Group

The ENI is the actual network endpoint of your container.

------------------------------------------------------------------------

# 4️⃣ Security Group -- Where Security Is Enforced

Security Groups are attached to the ENI.

They act as stateful firewalls.

Example rule:

Inbound: - TCP 8000 from 0.0.0.0/0 (public access)

Outbound: - Allow all (default)

Traffic flow:

Internet → Internet Gateway → Subnet → ENI → Security Group → Container

The Security Group does NOT belong to ECS directly. It belongs to the
VPC networking layer.

------------------------------------------------------------------------

# 5️⃣ ECS & Fargate Architecture

## 🔹 ECS Cluster

Logical grouping of services.

## 🔹 Task Definition

Blueprint of container: - CPU - Memory - Image - Port - Environment
variables - Logging configuration

## 🔹 Service

Ensures: - 1 task is always running - Auto-restart on crash - Desired
count maintained

## 🔹 Fargate

Provides: - Serverless container runtime - Isolated compute - CPU &
Memory allocation per task

No EC2 instance management required.

------------------------------------------------------------------------

# 6️⃣ Image Pull Flow (ECR → ECS)

1.  Task starts
2.  ECS Execution Role authenticates with ECR
3.  ECR verifies IAM permissions
4.  Image layers downloaded
5.  Container launched

IAM Execution Role is required for this step.

------------------------------------------------------------------------

# 7️⃣ IAM Roles & Permission Flow

## 🔹 IAM User (CLI Only)

Used for: - Creating infrastructure - Pushing Docker image to ECR

Never used at runtime.

------------------------------------------------------------------------

## 🔹 ECS Task Execution Role

Used for: - Pulling image from ECR - Sending logs to CloudWatch

Without this role, task fails to start.

------------------------------------------------------------------------

## 🔹 ECS Task Role (Future Use)

Used by container for: - Accessing S3 model artifacts - Reading
latest.json - Downloading model.tar.gz

Provides temporary credentials to container.

------------------------------------------------------------------------

# 8️⃣ S3 Model Registry Flow

When MODEL_URI points to S3:

1.  Container reads latest.json
2.  Resolves artifact_uri
3.  Downloads model.tar.gz
4.  Extracts locally
5.  Loads into memory

This design enables: - Model versioning - Rollbacks - No container
rebuild required

------------------------------------------------------------------------

# 9️⃣ CloudWatch Logging Flow

1.  Container writes stdout/stderr
2.  awslogs driver captures logs
3.  Logs sent to CloudWatch Log Group: /ecs/article-topic-classifier-v2

This enables debugging & monitoring.

------------------------------------------------------------------------

# 🔟 End-to-End Request Lifecycle

1.  Client sends HTTP request
2.  Internet Gateway forwards traffic
3.  Public Subnet routes packet
4.  ENI receives traffic
5.  Security Group validates rule
6.  Container processes request
7.  Model performs inference
8.  JSON response returned

------------------------------------------------------------------------

# 1️⃣1️⃣ Why This Architecture Is Production-Aligned

This design provides:

-   Stateless container deployment
-   Infrastructure as configuration
-   Decoupled model & container
-   Secure IAM-based access
-   Auto-restart capability
-   Observability via CloudWatch
-   Horizontal scalability (future)
-   Easy CI/CD integration

------------------------------------------------------------------------

# 1️⃣2️⃣ Current System State

✔ Docker image stored in ECR\
✔ Fargate cluster active\
✔ ECS Task Definition created\
✔ ECS Service ready to deploy\
✔ Security Group attached to ENI\
✔ CloudWatch logging configured\
✔ S3 model registry design ready

------------------------------------------------------------------------

# 1️⃣3️⃣ Future Improvements

-   Restrict Security Group to specific IP
-   Add Application Load Balancer
-   Enable HTTPS via ACM
-   Auto-scaling policies
-   CI/CD pipeline (GitHub Actions)
-   Monthly retraining automation
-   Canary model deployments

------------------------------------------------------------------------

# Final Summary

ECR → Stores container\
ECS → Orchestrates service\
Fargate → Runs container\
IAM → Secures interactions\
VPC → Provides network isolation\
Security Group → Enforces firewall rules\
ENI → Network endpoint of task\
S3 → Stores model artifacts\
CloudWatch → Observability layer

Everything works together to create a secure, scalable, cloud-native ML
inference architecture.

------------------------------------------------------------------------

End of Document.
