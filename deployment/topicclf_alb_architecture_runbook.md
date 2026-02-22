# Topic Classification Inference on AWS (ECS Fargate + ALB) — Architecture & Runbook

_Last updated: 2026-02-15 19:35 UTC_

This document captures the **current deployed architecture**, the **options selected and why**, how components are **linked together**, and the **problems encountered + fixes** during setup.

---

## 1) What you built (high level)

You have a FastAPI-based NLP inference service running in **ECS Fargate** behind an **Application Load Balancer (ALB)**.

Users call the ALB URL (stable DNS), and the ALB forwards traffic to the ECS task (ephemeral IPs), while also performing health checks.

You can access the interactive FastAPI docs via:

- `http://<ALB-DNS>/docs`

---

## 2) Why ALB was needed

Before ALB:
- You accessed the service directly via the **public IP of the running Fargate task**.
- Fargate tasks are **ephemeral**: redeploying creates a new ENI → **new public IP**.
- Result: old IPs time out even though the service is healthy.

With ALB:
- You get a **stable DNS name** (ALB DNS), while the ALB automatically routes to the current running task(s).
- This is the standard pattern for “stable endpoint” on ECS.

---

## 3) Final architecture (components + links)

### Core components
1. **Client (Browser / curl / app)**
2. **ALB (Internet-facing)**
3. **ALB Listener**
   - HTTP :80
4. **Target Group**
   - Target type: **IP**
   - Port: **8000**
   - Health check path: **/health**
5. **ECS Cluster**
   - `article-topic-cluster-v2`
6. **ECS Service**
   - `article-topic-service`
7. **ECS Task (Fargate)**
   - Container exposes port **8000**
   - FastAPI listens on `0.0.0.0:8000`

### How traffic flows
```
Internet
  → ALB (DNS name)
      → Listener :80
          → Target Group (IP targets, port 8000)
              → ECS Task ENI private IP (e.g. 172.31.x.x)
                  → FastAPI container :8000
```

### Why “Target type = IP” matters on Fargate
- Fargate does not expose a classic “instance” to register like EC2.
- You register the **task ENI IP** into the target group.
- ECS service integration handles this automatically once load balancing is configured.

---

## 4) The exact options you selected (and why)

### 4.1 ALB
- **Type:** Application Load Balancer  
- **Scheme:** Internet-facing  
- **Listener:** HTTP :80  
- **Subnets:** Two public subnets in different AZs (multi-AZ)  
- **Security Group:** `topicclf-alb-sg`  

**Why**
- Application LB is best for HTTP routing and health checks.
- Internet-facing enables public access (you can later tighten to private-only patterns).
- Multi-AZ improves availability (and is the recommended baseline).

---

### 4.2 Target Group
- **Name:** `topicclf-tg`
- **Target type:** IP addresses
- **Protocol/Port:** HTTP :8000
- **Health check path:** `/health`
- **Success codes:** 200

**Why**
- Fargate tasks register via IP.
- Port 8000 matches the container’s listening port.
- `/health` is a lightweight health endpoint.

---

### 4.3 ECS Service Load Balancing integration
In the ECS Service update:
- **Use load balancing:** enabled
- **Load balancer:** `topicclf-alb`
- **Listener:** HTTP :80
- **Target group:** `topicclf-tg`
- **Container mapping:** `api 8000:8000`

**Why**
- This is what makes ECS automatically:
  - register/deregister task IPs into the target group on deploys
  - keep the target group healthy
  - reduce “manual IP management”

---

### 4.4 Security Groups (current choice)
#### ALB Security Group: `topicclf-alb-sg`
- Inbound:
  - **HTTP :80** allowed only from **Your public IP (/32)**

**Why**
- Locks down the public endpoint so only you can access it.
- Prevents random internet traffic from discovering/using `/docs` and your API endpoints.

> Note: If your ISP public IP changes, you will need to update this rule (choose “My IP” again).

---

## 5) What changed vs. “direct task public IP”
### Before
- You used: `http://<task-public-ip>:8000/health`
- Worked only while that exact task was running.

### After
- You use: `http://<alb-dns>/health`
- DNS stays stable; ALB routes to the current task.

This is the correct “production-style” stabilization step.

---

## 6) Problems faced & fixes (debugging log)

### Incident A — “Nothing shows in Target group dropdown”
**Cause**
- There were **no target groups created yet** in the region/VPC.

**Fix**
- Created a target group first (`topicclf-tg`) with:
  - Target type: IP
  - Port: 8000
  - Health check: /health

---

### Incident B — Targets not registering / 0 targets
**Cause**
- Target group existed, but ECS service wasn’t yet configured to use it.

**Fix**
- Updated ECS service:
  - enabled load balancing
  - connected the ALB listener and the target group
  - ensured container port mapping 8000 is correct

---

### Incident C — Target unhealthy: “Availability Zone not enabled for load balancer”
**Cause**
- Task got placed in an AZ/subnet that wasn’t enabled on the ALB,
  or subnet/AZ mismatch between service networking and ALB.

**Fix**
- Ensured ALB is enabled in the same AZs/subnets as ECS service networking.
- After alignment, target became **Healthy**.

---

### Incident D — Public IP confusion (old IP tested)
**Cause**
- Old public IPs were tested after redeploy (Fargate task changed).

**Fix**
- Move to ALB (stable DNS).
- Stop using raw task IPs for “primary endpoint”.

---

## 7) Current “what’s left” around load balancing (Phase 4 remaining, optional)

You’re operational now. Remaining improvements (optional, industry-grade hardening):

1. **HTTPS**
   - Add an ACM certificate
   - Add HTTPS listener :443
   - Redirect HTTP → HTTPS

2. **Public exposure hardening**
   - Keep ALB locked to your IP (current), or
   - Add authentication to docs/API, or
   - Put ALB in private subnets + use VPN (more advanced)

3. **Disable public IP on tasks (stronger posture)**
   - Tasks in private subnets
   - ALB in public subnets
   - ECS SG allows inbound only from ALB SG

4. **Custom domain**
   - Route53 record to ALB
   - Easier to remember than raw ALB DNS

---

## 8) Quick operational notes

- **Costs still accrue while running**, even with zero traffic:
  - ECS Fargate compute for running tasks
  - ALB hourly + LCU (LCU depends on usage; idle is mostly hourly)
  - CloudWatch logs
  - ECR storage

- To reduce cost quickly:
  - set ECS desired tasks = **0** when idle (service stays, tasks stop)
  - ALB cost still continues if ALB remains running

---

## 9) “At a glance” summary

**Stable endpoint:** ALB DNS  
**Security:** ALB SG locked to your current public IP (/32)  
**Health checks:** Target group checks `/health` on port 8000  
**Registration:** ECS service automatically registers task ENI IPs to target group  
**Key lesson:** Task IPs are ephemeral; ALB provides stability

---

_End of document._
