# Article Topic Classifier — Production Architecture Master Document (v1.0)
_Last Updated: 2026-02-13 21:40 UTC_

> This is a comprehensive, end-to-end engineering document capturing **what you built**, **how you deployed it to AWS**, **exact console fields/options to select**, **why each choice was made**, **common failure modes**, **how to debug**, and **how to evolve it into an industry-grade MLOps system**.

---

## Table of Contents
1. Executive summary  
2. System goals and non-goals  
3. High-level architecture (cloud + runtime)  
4. Component responsibilities and how they connect  
5. Local-to-cloud workflow (source → container → registry → runtime)  
6. AWS resources created (inventory)  
7. Step-by-step AWS build guide (console, field-by-field)  
   - IAM & CLI access  
   - ECR repository and push  
   - CloudWatch logs  
   - ECS cluster (Fargate)  
   - Task definition  
   - Service + networking  
8. Verification checklist (what “healthy” looks like)  
9. Debugging runbook (symptoms → causes → fixes)  
10. Incident postmortem: “Timeout after deploy” (root cause + learnings)  
11. Cost model and free-tier-friendly practices  
12. Security model (IAM, networking, secrets)  
13. Production hardening roadmap (ALB/HTTPS/DNS/Autoscaling/CI/CD)  
14. MLOps roadmap: monthly retraining from S3 XML, continual learning, model registry, safe rollout  
15. Appendices (useful commands & scripts)

---

# 1) Executive Summary

You deployed an ML inference API that performs topic classification for articles (title + body) using a fine-tuned DistilBERT model. The API returns:

- Predicted label  
- Confidence score  
- Full probability distribution  
- Decision label: `auto_accept`, `needs_review`, `reject` (risk-aware routing)

The service is containerized and deployed on **AWS ECS (Fargate)**, pulling its image from **Amazon ECR**, using **CloudWatch Logs** for observability, and networking through **VPC subnets + ENI + Security Groups**.

You also developed real deployment/debugging skill: IAM/ECR login, ECS configuration, networking troubleshooting, and the key insight that **Fargate public IP changes on redeploy** unless a load balancer/DNS is used.

---

# 2) System Goals and Non-Goals

## Goals
- Production-style inference API (FastAPI) with correct model loading lifecycle
- Containerized deployment
- Cloud deployment with minimal operational burden
- Logging and visibility (CloudWatch)
- Cost-minimized approach (small CPU/mem, no ALB initially)
- Foundation for future: model registry + monthly retraining + safe model rollout

## Non-goals (for v1.0)
- Stable hostname / custom domain / HTTPS (will be added later via ALB + ACM)
- Autoscaling and multi-AZ high availability beyond baseline requirement
- Full CI/CD pipeline (recommended next)
- Full monitoring/alerting (recommended next)

---

# 3) High-Level Architecture

## 3.1 Cloud runtime architecture (accurate networking)

```
Client (Browser/Curl)
   |
   |  HTTP :8000
   v
Internet
   |
   v
Internet Gateway (VPC)
   |
   v
Public Subnet (Route 0.0.0.0/0 -> IGW)
   |
   v
ENI (Elastic Network Interface) created for the Fargate Task
   |
   |  Security Group rules enforced here (stateful firewall)
   v
Container (FastAPI/Uvicorn listening on 0.0.0.0:8000)
   |
   +--> CloudWatch Logs (stdout/stderr via awslogs driver)
   +--> ECR (image pulled at task start using Execution Role)
   +--> (Future) S3 (model registry + artifacts using Task Role)
```

## 3.2 Core AWS service connections (mental model)
- **ECR**: Stores the Docker image
- **ECS**: Orchestrates running containers (services/tasks)
- **Fargate**: Serverless compute that actually runs the container
- **IAM Execution Role**: Allows ECS/Fargate to pull the image from ECR + push logs to CloudWatch
- **VPC/Subnet/IGW**: Allows internet traffic to reach the task ENI
- **Security Group**: Firewall that allows inbound `TCP 8000`
- **CloudWatch Logs**: Central place to see container logs

---

# 4) Component Responsibilities and How They Connect

## 4.1 FastAPI + Model (inside container)
- Loads model/tokenizer at startup (FastAPI lifespan)
- Runs inference on request
- Returns structured JSON
- Prints logs to stdout (captured by CloudWatch)

## 4.2 Amazon ECR (image registry)
- Stores `article-topic-classifier-inference:latest`
- ECS uses the **execution role** to authenticate and pull layers

## 4.3 ECS (orchestrator)
- **Task definition**: blueprint (image, env vars, ports, cpu/mem, logging)
- **Service**: desired state manager (“keep 1 task running”)
- **Task**: the running container instance

## 4.4 Fargate (compute)
- Provisions isolated runtime for the task
- Creates ENI and attaches security group
- Assigns public IP if enabled
- Bills based on vCPU/memory seconds

## 4.5 IAM roles (security)
- **IAM User** (your CLI identity): used only for provisioning and pushing images
- **Task Execution Role**: used by ECS/Fargate *to start the task* (ECR pull, CloudWatch logs)
- **Task Role** (future): used by your application *at runtime* (S3 model downloads, etc.)

## 4.6 VPC/Subnets + Security group (network)
- Subnet must be public for direct internet access
- Route table must include `0.0.0.0/0 -> igw-...`
- SG must allow inbound `TCP 8000` from your IP (recommended) or `0.0.0.0/0` (temporary)

---

# 5) Local-to-Cloud Workflow

1. Develop locally (FastAPI + model artifacts)
2. Build Docker image
3. Push image to ECR
4. Create ECS task definition referencing ECR image
5. Create ECS service to run 1 task
6. ECS pulls image from ECR, runs in Fargate
7. Test via public IP + port 8000

---

# 6) AWS Resources Created (Inventory)

> Names are based on your current setup; adjust if you renamed anything.

### Core
- ECS Cluster: `article-topic-cluster-v2` (Fargate only)
- ECR Repository: `article-topic-classifier-inference`
- Task Definition family: `article-topic-task`
- ECS Service: `article-topic-service`
- CloudWatch Log Group: `/ecs/article-topic-classifier-v2`
- Security Group: `launch-wizard-1` (inbound 8000 open) — ideally rename later

### IAM
- IAM User: e.g., `mlops-admin` (CLI)
- Role: `ecsTaskExecutionRole` (execution role)

---

# 7) Step-by-Step AWS Build Guide (Field-by-Field)

## 7.1 IAM & AWS CLI Setup (Local)

### A) Install AWS CLI
You already verified:
```bash
aws --version
```

### B) Configure credentials
```bash
aws configure
```
Provide:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `us-east-1`
- Default output: `json`

### C) Verify identity
```bash
aws sts get-caller-identity
```
You used this to confirm your real account ID (important).

**Why this matters**: You must use the correct account ID in ECR URLs. A wrong ID causes `403 Forbidden` on docker push.

---

## 7.2 ECR Setup (Console + Commands)

### A) Create ECR repo (Console)
ECR → Repositories → Create repository
- Visibility: Private
- Repository name: `article-topic-classifier-inference`
- Leave defaults

### B) Login Docker to ECR (PowerShell)
```powershell
aws ecr get-login-password --region us-east-1 `
| docker login --username AWS --password-stdin 363283722404.dkr.ecr.us-east-1.amazonaws.com
```

### C) Tag and push
```powershell
docker tag article-topic-classifier 363283722404.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest
docker push 363283722404.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest
```

**Why ECR**: ECS needs a registry to pull images; local images are not accessible by Fargate.

---

## 7.3 CloudWatch Log Group (Console)

CloudWatch → Log groups → Create log group
- Name: `/ecs/article-topic-classifier-v2`

**Why**: Centralized container logs for debugging and observability.

---

## 7.4 ECS Cluster (Fargate only)

ECS → Clusters → Create cluster
- Cluster name: `article-topic-cluster-v2`
- Infrastructure: **Fargate only**
- Create

**Why Fargate**:
- No EC2 capacity/provider problems
- No server management
- Minimal operational overhead

---

## 7.5 Task Definition (Fargate)

ECS → Task definitions → Create new task definition

### A) Basics
- Task definition family: `article-topic-task`
- Launch type: **AWS Fargate**
- OS/Architecture: Linux / x86_64

### B) Task size (cost-controlled)
- CPU: `0.25 vCPU`
- Memory: `0.5 GB` (if OOM, raise to 1GB)

### C) Task execution role
- Choose: `ecsTaskExecutionRole`
- If not present: create default

**Why**: Allows image pull from ECR and log delivery to CloudWatch.

### D) Task role
- Leave blank for now (add later for S3 model access)

### E) Container definition
Add container:
- Name: `api`
- Image URI: `363283722404.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest`
- Essential: Yes

#### Port mappings
- Container port: `8000`
- Protocol: TCP
- App protocol: HTTP

#### Environment variables
- `MODEL_URI = artifacts/distilbert`

#### Logging
- Log driver: awslogs
- Log group: `/ecs/article-topic-classifier-v2`
- Region: `us-east-1`
- Stream prefix: `api`

Create task definition.

**Why these choices**:
- Port 8000 is your FastAPI port
- env var keeps model location configurable
- logs enable debugging without SSH

---

## 7.6 ECS Service (Runs the task)

ECS → Clusters → `article-topic-cluster-v2` → Services → Create

### A) Service config
- Launch type: Fargate
- Task definition: `article-topic-task` (latest revision)
- Desired tasks: `1`
- Deployment: Rolling (default)
- Load balancer: None (for v1.0)

### B) Networking (most important)
- VPC: Default VPC
- Subnets: Select **2 subnets** in different AZs (public preferred)
- Auto-assign public IP: **Enabled**
- Security group: select SG that has inbound `TCP 8000` open

**Security group inbound rule (temporary)**:
- Type: Custom TCP
- Port: 8000
- Source: `0.0.0.0/0` (temporary) or your IP (recommended)

Create service.

---

# 8) Verification Checklist

## 8.1 ECS Health
- Service shows Desired = 1, Running = 1
- Task status: RUNNING
- Task has Public IP

## 8.2 Logs
CloudWatch → Log group → `/ecs/article-topic-classifier-v2`
You should see:
- Startup logs
- Uvicorn running on `0.0.0.0:8000`

## 8.3 Endpoint Test
Use the **current** task public IP:
```bash
curl http://<PUBLIC_IP>:8000/health
```

Expected:
```json
{"status":"ok"}
```

---

# 9) Debugging Runbook (Symptoms → Causes → Fixes)

## A) `curl: (28) Could not connect` / Browser timeout
**Most common causes**
1) You are hitting an **old IP** (task replaced)
2) Public IP is not enabled
3) SG doesn’t allow inbound 8000
4) Task is in private subnet (no IGW route)
5) Wrong port

**Fix**
- Re-check ECS Task → Networking → Public IP (copy fresh)
- Ensure service networking has auto-assign public IP enabled
- SG inbound: TCP 8000 from your IP or 0.0.0.0/0
- Confirm subnet route: `0.0.0.0/0 -> igw-...`

## B) Logs show `Uvicorn running on http://127.0.0.1:8000`
**Cause**: Uvicorn bound to localhost  
**Fix**: Ensure Docker CMD uses:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## C) Task keeps restarting / stops
**Cause**: Crash, OOM, bad env var, missing model files  
**Fix**:
- Check CloudWatch logs for stack trace
- Increase memory to 1GB
- Ensure model directory exists inside image or switch to S3 model loading

## D) `403 Forbidden` on `docker push`
**Cause**: wrong account ID, not logged into ECR, missing permissions  
**Fix**:
- `aws sts get-caller-identity` to confirm account id
- re-run ECR login command with correct account id
- ensure IAM user has ECR permissions

---

# 10) Incident Postmortem: Timeout After Deploy (What Happened)

## Symptom
- Browser timed out
- curl could not connect

## Investigation steps taken
- Verified SG inbound rule for 8000
- Verified subnet route table includes IGW route
- Verified Uvicorn binds to 0.0.0.0
- Verified task is RUNNING and has public IP
- Verified port mapping exists

## Root cause
- The public IP changed after redeployment (Fargate tasks are ephemeral).
- You tested the old IP (dead task) instead of the new running task IP.

## Fix
- Re-copied the current RUNNING task public IP and retested.
- Endpoint worked immediately.

## Key learning
- Without an ALB/DNS, the “URL” changes on every replacement.
- To prevent this, use an ALB + DNS for a stable endpoint.

---

# 11) Cost Model and Free-Tier-Friendly Practices

## What costs money in this setup
- Fargate compute (vCPU + memory per second)
- CloudWatch logs ingestion & storage
- ECR image storage (GB-month)
- Data transfer (small for basic testing)

## Cost-minimizing decisions you made
- CPU 0.25 vCPU, memory 0.5GB
- Desired tasks = 1
- No ALB (ALB adds cost)
- No autoscaling (prevents runaway spend)

## Practical budget tips
- Stop the service when not using (set desired tasks to 0)
- Add log retention (e.g., 7 days) in CloudWatch
- Avoid large images (keep Docker image lean)

---

# 12) Security Model

## Networking
- Prefer inbound 8000 from your IP, not 0.0.0.0/0
- Keep outbound open for ECR/S3/CloudWatch

## IAM
- Do not embed keys in code
- Use Task Role for S3 access later

## Secrets
- Use ECS secrets integration or SSM Parameter Store later (if needed)

---

# 13) Production Hardening Roadmap

## Stable URL + HTTPS (recommended)
Add Application Load Balancer (ALB) + ACM certificate
- ALB provides stable DNS
- ACM provides TLS
- Route 53 maps your domain

## Autoscaling
- CPU-based or request-based scaling
- Requires ALB or metrics strategy

## Observability
- CloudWatch alarms (task restarts, 5xx)
- Structured logs
- Tracing (optional)

## CI/CD
- GitHub Actions builds/pushes image
- ECS service update & force deployment

---

# 14) MLOps Roadmap (Monthly Training From S3 XML)

> Your stated goal: every month, new XML articles in S3 are used to retrain, creating a new versioned model without forgetting old knowledge, then deployed safely.

## 14.1 Data ingestion
- Store XML per article in S3 prefix like:
  - `s3://<bucket>/articles/YYYY/MM/*.xml`
- Monthly job reads “last month” prefix, parses XML, extracts fields

## 14.2 Model versioning + registry (simple, strong)
Store models in S3 like:
- `s3://<bucket>/models/topicclf/2026-03-01/model.tar.gz`
- `s3://<bucket>/models/topicclf/latest.json` (pointer)

`latest.json` example:
```json
{"version":"2026-03-01","artifact_uri":"s3://<bucket>/models/topicclf/2026-03-01/model.tar.gz"}
```

Inference container reads `MODEL_URI=s3://.../latest.json`, downloads artifact, loads model.

## 14.3 Continual learning without forgetting (industry approaches)
- **Baseline**: retrain from scratch using *all* data (simplest, safest; higher compute)
- **Incremental fine-tune**: fine-tune existing model on new data (risk of catastrophic forgetting)
- **Replay buffer**: mix new data + sampled old data to preserve performance (common)
- **Regularization**: EWC, LwF (advanced)
- **Distillation**: new model learns from old model + new labels (common at scale)

Free-tier-friendly recommendation:
- Use incremental fine-tune + replay sampling, limited epochs, spot if possible (later)

## 14.4 Safe rollout
- Validate accuracy + calibration on held-out test
- Promote only if metrics pass thresholds
- Keep previous model version for rollback
- Update `latest.json` only after passing gates

## 14.5 Scheduler
- EventBridge schedule monthly
- Training job could run on:
  - ECS task (batch) OR
  - EC2 spot instance (cheaper for training) OR
  - SageMaker (managed but costlier)

---

# 15) Appendices

## A) Get current public IP quickly (CLI)
```bash
aws ecs describe-tasks   --cluster article-topic-cluster-v2   --tasks $(aws ecs list-tasks --cluster article-topic-cluster-v2 --query 'taskArns[0]' --output text)   --query 'tasks[0].attachments[0].details[?name==`publicIPv4Address`].value'   --output text
```

## B) PowerShell script to print full URL
Create `get-url.ps1`:
```powershell
$task = aws ecs list-tasks --cluster article-topic-cluster-v2 --query 'taskArns[0]' --output text
$ip = aws ecs describe-tasks --cluster article-topic-cluster-v2 --tasks $task --query "tasks[0].attachments[0].details[?name=='publicIPv4Address'].value" --output text
Write-Host "http://$ip`:8000/health"
```

Run:
```powershell
./get-url.ps1
```

---

## Final note
This v1.0 deployment is already “real-world”: containerized, orchestrated, logged, and debugged in AWS.
The next leap to “industry-grade” is **stable endpoint (ALB/DNS/HTTPS)** and **automated retraining + safe model promotion**.

