
# Article Topic Classifier
# ECS Deployment Incident Debug Report

Last Updated: 2026-02-13 21:31 UTC

---

## 1. Incident Summary

After deploying the FastAPI ML inference container to AWS ECS (Fargate),
the public endpoint:

    http://<public-ip>:8000/health

returned:

    ERR_CONNECTION_TIMED_OUT

Even though:

- Task status = RUNNING
- Public IP assigned
- Security Group allowed port 8000
- Subnet routed to Internet Gateway
- Uvicorn bound to 0.0.0.0

---

## 2. Infrastructure Context

ECS Cluster: article-topic-cluster-v2 (Fargate)
Task Definition: article-topic-task
CPU: 0.25 vCPU
Memory: 0.5 GB
Container Port: 8000/TCP
Public IP: Enabled
Security Group: launch-wizard-1
Inbound Rule: TCP 8000 from 0.0.0.0/0
Log Group: /ecs/article-topic-classifier-v2

---

## 3. Investigation Steps

### Step 1 – Security Group Verification

Confirmed inbound rule:

    Custom TCP | Port 8000 | 0.0.0.0/0

Result: Correct

---

### Step 2 – Subnet Routing Verification

Confirmed route table:

    0.0.0.0/0 -> Internet Gateway

Result: Public subnet

---

### Step 3 – Uvicorn Binding Verification

CloudWatch logs showed:

    Uvicorn running on http://0.0.0.0:8000

Result: Application listening on all interfaces

---

### Step 4 – Port Mapping Verification

Confirmed task definition included:

    Container port: 8000 (TCP)

Result: Correct

---

### Step 5 – Subnet Distribution Verification

Confirmed two subnets across different AZs selected.

Result: Correct

---

## 4. Root Cause

The public IP being tested was outdated.

When the ECS Service was updated and a new deployment forced:

- Old task stopped
- New ENI created
- New Public IP assigned

Browser was still pointing to the previous task IP.

---

## 5. Evidence

Old IP tested:

    54.237.118.48

Current running task IP:

    54.237.218.48

Testing updated IP resolved the issue:

    http://54.237.218.48:8000/health
    -> {{"status":"ok"}}

---

## 6. Technical Explanation

Fargate tasks are ephemeral:

- Each task receives a new ENI
- Each ENI receives a new public IP
- On redeployment, IP changes
- Old IP becomes unreachable

Public IPs are not persistent unless using:

- Application Load Balancer
- Network Load Balancer
- Elastic IP (EC2 only)

---

## 7. Lessons Learned

1. Always verify the current Public IP after redeployment.
2. Fargate does not guarantee stable public IPs.
3. CloudWatch logs help isolate application-level issues.
4. Systematic debugging prevents unnecessary infrastructure changes.

---

## 8. Preventive Improvements

- Deploy behind Application Load Balancer (ALB)
- Use Route 53 DNS
- Add HTTPS via ACM
- Avoid direct IP-based access
- Implement health checks

---

## 9. Final Status

Deployment successful.
Service accessible via current Public IP.

---

End of Incident Report.
