# 🚀 Production Deployment Guide

## Article Topic Classification API --- AWS EC2 (Free Tier, Production-Safe)

This document describes the complete end-to-end deployment of the
Article Topic Classification system to AWS EC2 using Docker.

It includes:

-   Architecture overview\
-   Production vs development environment separation\
-   AWS configuration\
-   Docker setup\
-   Real issues encountered\
-   Root cause analysis\
-   Resolutions\
-   Diagnostics used\
-   Security considerations\
-   Lessons learned

This deployment intentionally uses AWS Free Tier infrastructure while
maintaining production-aligned engineering practices.

------------------------------------------------------------------------

# 🧠 System Architecture

Local Development (Training + Evaluation) ↓ Docker Image (Production
Build) ↓ AWS EC2 (t2.micro - Free Tier) ↓ Public IP + Port 8000 ↓
FastAPI Inference Service

------------------------------------------------------------------------

# 📦 Environment Separation

## Development Environment (`requirements-dev.txt`)

Used for: - Model training - Dataset processing - Evaluation -
Calibration analysis

Includes heavy ML and evaluation dependencies.

------------------------------------------------------------------------

## Production Environment (`requirements-prod.txt`)

Used only for: - Running FastAPI inference service

Includes: - CPU-only torch - transformers - fastapi - uvicorn - numpy

This separation: - Prevents CUDA installation in production - Reduces
Docker image size - Avoids disk overflow - Reduces attack surface -
Speeds up container startup

------------------------------------------------------------------------

# ☁️ AWS Infrastructure Setup

## EC2 Configuration

-   Instance type: t2.micro (Free Tier eligible)
-   OS: Amazon Linux 2023
-   Root volume increased from 8GB → 20GB
-   SSH restricted to current public IP
-   Port 8000 open publicly for API

------------------------------------------------------------------------

# 🔐 Billing Protection

AWS Budget configured: - Monthly limit: ₹500 - Email alert at 100%
threshold

Prevents unexpected billing.

------------------------------------------------------------------------

# 🐳 Production Dockerfile

``` dockerfile
FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

------------------------------------------------------------------------

# 🚀 Deployment Steps

## Install Docker on EC2

``` bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```

Reconnect after usermod command.

------------------------------------------------------------------------

## Upload Project

``` bash
scp -i article-key.pem -r * ec2-user@<public-ip>:~/article-topic-classifier
```

Remove private key from EC2 after upload:

``` bash
rm article-key.pem
```

------------------------------------------------------------------------

## Build Image

``` bash
docker build -t article-topic-classifier .
```

------------------------------------------------------------------------

## Run Container

``` bash
docker run -d -p 8000:8000 article-topic-classifier
```

------------------------------------------------------------------------

## Access Public API

http://`<public-ip>`{=html}:8000/docs

------------------------------------------------------------------------

# 🛠 Issues Faced & Resolutions

## SSH Timeout

Cause: Dynamic public IP changed.\
Fix: Updated security group SSH rule to "My IP".

------------------------------------------------------------------------

## Docker Build Failed --- No Space Left on Device

Cause: GPU-enabled dependencies + 8GB root volume.\
Fix: - Increased volume to 20GB\
- Removed training dependencies from production\
- Used CPU-only torch

------------------------------------------------------------------------

## Transformers Not Found

Cause: Overriding pip index URL.\
Fix: Installed torch separately using CPU index.

------------------------------------------------------------------------

## Windows SSH Permission Error

Fix:

``` powershell
icacls article-key.pem /inheritance:r
icacls article-key.pem /grant:r "$($env:USERNAME):R"
```

------------------------------------------------------------------------

# 📈 Final Outcome

✔ Model trained & validated\
✔ Confidence-aware decision logic\
✔ Calibration & coverage analysis\
✔ Dockerized inference service\
✔ Secure EC2 deployment\
✔ Public API accessible\
✔ Free-tier safe configuration

------------------------------------------------------------------------

# 🔐 Security Practices

-   SSH restricted to current IP\
-   No password authentication\
-   Billing alert enabled\
-   Private key removed from server\
-   Minimal production dependencies\
-   No unnecessary open ports

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Assign Elastic IP\
-   Add HTTPS via Let's Encrypt\
-   Add reverse proxy (Nginx)\
-   Move to ECS Fargate\
-   Add CI/CD pipeline\
-   Add monitoring & logging

------------------------------------------------------------------------

This deployment represents a production-aligned ML inference system
deployed securely on AWS Free Tier infrastructure.
