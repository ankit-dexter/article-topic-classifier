# Article Topic Classifier

# Complete AWS Deployment Deep Dive (Field-by-Field Explanation)

*Last Updated: 2026-02-13 21:01 UTC*

------------------------------------------------------------------------

# 1️⃣ Objective

Deploy a production-style ML inference API using:

-   Docker
-   Amazon ECR
-   Amazon ECS (Fargate)
-   IAM Roles
-   CloudWatch Logs
-   VPC Networking
-   Security Groups

This document explains:

-   What was configured
-   Which exact fields were selected
-   Why each decision was made
-   How all AWS components connect internally

------------------------------------------------------------------------

# 2️⃣ Local Setup Summary

## Model Loading Design

Environment variable used:

    MODEL_URI=artifacts/distilbert

Why: - Allows local testing - Enables later switch to S3 without
container rebuild

------------------------------------------------------------------------

## Docker Build

    docker build -t article-topic-classifier .

Why Docker: - Immutable runtime - Reproducible builds - Portable
deployment

------------------------------------------------------------------------

# 3️⃣ Amazon ECR Setup

## Repository Name

    article-topic-classifier-inference

Why: - Clear separation of inference image - Single source of truth for
containers

## Image Tag Used

    latest

Why: - Simplicity during development - Can later switch to SHA-based
immutable tags

------------------------------------------------------------------------

# 4️⃣ ECS Cluster Configuration

Cluster Name:

    article-topic-cluster-v2

Launch Type Selected:

    Fargate only

Why Fargate:

-   Avoid EC2 capacity issues
-   No instance management
-   Simplified orchestration
-   Lower operational complexity
-   Serverless compute model

------------------------------------------------------------------------

# 5️⃣ Task Definition Configuration (Detailed)

Task Definition Family:

    article-topic-task

------------------------------------------------------------------------

## Infrastructure Requirements

Launch Type:

    AWS Fargate

Operating System:

    Linux / x86_64

Why: - Compatible with built Docker image

------------------------------------------------------------------------

## Task Size Selected

CPU:

    0.25 vCPU

Memory:

    0.5 GB

Why:

-   Sufficient for FastAPI + DistilBERT CPU inference
-   Minimizes Fargate cost
-   Scalable later if required

------------------------------------------------------------------------

## Task Execution Role

Selected:

    ecsTaskExecutionRole

Why:

Allows ECS to: - Pull image from ECR - Send logs to CloudWatch

Without this role, container would not start.

------------------------------------------------------------------------

## Container Configuration

Container Name:

    api

Image URI:

    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest

Port Mapping:

    Container Port: 8000
    Protocol: TCP

Environment Variable:

Name:

    MODEL_URI

Value:

    artifacts/distilbert

Why:

-   Keeps model local initially
-   Allows seamless switch to S3 model registry later

------------------------------------------------------------------------

## Logging Configuration

Log Driver:

    awslogs

Log Group:

    /ecs/article-topic-classifier-v2

Region:

    us-east-1

Stream Prefix:

    api

Why:

-   Centralized observability
-   Debugging container startup failures
-   Production-grade monitoring foundation

------------------------------------------------------------------------

# 6️⃣ ECS Service Configuration

Service Name:

    article-topic-service

Desired Tasks:

    1

Why: - Minimal deployment for testing - Can scale horizontally later

------------------------------------------------------------------------

# 7️⃣ Networking Configuration (Critical)

VPC:

    Default VPC

Subnets Selected: - Two subnets in different Availability Zones

Why: - Required by Fargate - Enables high availability - AZ distribution
capability

------------------------------------------------------------------------

Public IP:

    Enabled

Why: - Required for public API access - Enables direct browser testing

------------------------------------------------------------------------

Security Group Used:

    launch-wizard-1

Inbound Rules:

  Type         Port   Source
  ------------ ------ -----------
  Custom TCP   8000   0.0.0.0/0
  SSH          22     My IP

Why:

-   Port 8000 allows API traffic
-   SSH rule irrelevant for Fargate but harmless
-   0.0.0.0/0 used temporarily for testing

------------------------------------------------------------------------

# 8️⃣ Networking Flow Explained

Actual traffic flow:

Internet → Internet Gateway → Public Subnet → ENI (attached to Fargate
task) → Security Group → Container (Port 8000)

Security Group is attached to the ENI, not ECS directly.

------------------------------------------------------------------------

# 9️⃣ Image Pull Flow

1.  Task starts
2.  ECS Execution Role authenticates to ECR
3.  Image layers pulled
4.  Container launched

------------------------------------------------------------------------

# 🔟 End-to-End Request Lifecycle

1.  Client sends HTTP request
2.  Internet Gateway forwards packet
3.  Subnet routing applied
4.  ENI receives traffic
5.  Security Group validates inbound rule
6.  FastAPI processes request
7.  Model performs inference
8.  JSON response returned

------------------------------------------------------------------------

# 1️⃣1️⃣ Current Cloud State

✔ Docker image stored in ECR\
✔ ECS Fargate cluster active\
✔ Task Definition configured\
✔ ECS Service deployed\
✔ Security Group configured\
✔ CloudWatch logs enabled\
✔ Public endpoint accessible

------------------------------------------------------------------------

# 1️⃣2️⃣ Why This Architecture Is Production-Grade

This setup ensures:

-   Stateless containers
-   Immutable deployment
-   Secure IAM-based access
-   Decoupled model registry
-   Observability
-   Automatic restart on failure
-   Future scalability
-   CI/CD readiness

------------------------------------------------------------------------

# 1️⃣3️⃣ Next Logical Improvements

-   Switch MODEL_URI to S3 model registry
-   Add HTTPS via ALB + ACM
-   Add auto-scaling
-   Add CI/CD pipeline
-   Add monthly retraining automation
-   Restrict security group to specific IP
-   Introduce canary model deployments

------------------------------------------------------------------------

# Final Summary

ECR → Stores container\
ECS → Orchestrates service\
Fargate → Runs container\
IAM → Controls permissions\
VPC → Network isolation\
Security Group → Firewall enforcement\
ENI → Network endpoint of task\
CloudWatch → Observability layer\
S3 → Model version storage

This deployment now represents a cloud-native, containerized,
production-ready ML inference architecture.

------------------------------------------------------------------------

End of Document.
