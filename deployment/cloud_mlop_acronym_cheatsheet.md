# Cloud & MLOps Acronym Cheat Sheet

## 🌐 Networking & Infrastructure

**VPC --- Virtual Private Cloud**\
Your isolated virtual network inside AWS.

**IGW --- Internet Gateway**\
Allows traffic between your VPC and the public internet.

**ENI --- Elastic Network Interface**\
Virtual network card attached to your ECS task.

**SG --- Security Group**\
Virtual firewall attached to ENIs.

**NACL --- Network Access Control List**\
Subnet-level firewall (stateless).

**CIDR --- Classless Inter-Domain Routing**\
IP range notation (e.g., 0.0.0.0/0 = entire internet).

------------------------------------------------------------------------

## 🚀 Compute & Containers

**ECS --- Elastic Container Service**\
AWS container orchestration service.

**Fargate --- Serverless container compute for ECS**\
Run containers without managing servers.

**EC2 --- Elastic Compute Cloud**\
Virtual machines in AWS.

**ECR --- Elastic Container Registry**\
Private Docker image registry.

**vCPU --- Virtual CPU**\
Compute allocation for containers.

------------------------------------------------------------------------

## 🌍 Load Balancing & Edge

**ALB --- Application Load Balancer**\
Layer 7 (HTTP) traffic router.

**NLB --- Network Load Balancer**\
Layer 4 (TCP) load balancer.

**ACM --- AWS Certificate Manager**\
Manages SSL/TLS certificates.

**DNS --- Domain Name System**\
Maps domain names to IP addresses.

**TLS/SSL --- Transport Layer Security**\
Encryption for HTTPS.

------------------------------------------------------------------------

## 📊 Monitoring & Logging

**CloudWatch**\
AWS monitoring and logging service.

**Metrics**\
CPU %, memory %, request count, etc.

------------------------------------------------------------------------

## 🤖 ML & MLOps

**NLP --- Natural Language Processing**\
ML on text.

**API --- Application Programming Interface**\
Your /predict endpoint.

**JSON --- JavaScript Object Notation**\
Data exchange format.

**ECE --- Expected Calibration Error**\
Measures confidence calibration quality.

**S3 --- Simple Storage Service**\
Object storage (future model registry).

**CI/CD --- Continuous Integration / Continuous Deployment**\
Automated build and deploy pipeline.

**WAF --- Web Application Firewall**\
Protects against malicious traffic.

**IAM --- Identity and Access Management**\
Controls permissions and roles.
