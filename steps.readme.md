PHASE 1 — Push Image to ECR

step 1: Create ECR Repository

In AWS Console:
ECR → Create Repository

Name:

article-topic-classifier-inference




🚀 Click “Create Repository”

Once created:

Copy the full URI. It will look like:

363283724240.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference



---------------------------------------------------- Adding use in IAM ----------------------------------------------------

🚨 Never Use Root Account for CLI

If you're using the root account email login, do NOT create access keys for root.

Industry rule:

Root account is for billing and account-level management only.

We will create a proper IAM user.

✅ Proper Setup (Industry Standard Way)
Step 1 — Go to IAM in AWS Console

Search:

IAM


Click:

Users

Step 2 — Create New User

Click:

Create user


Name it:

mlops-admin


(or anything clean)

Click Next.

Step 3 — Permissions

Choose:

Attach policies directly


Attach these (for now):

AmazonEC2ContainerRegistryFullAccess

AmazonECSFullAccess

AmazonEC2FullAccess

AmazonS3FullAccess

⚠️ This is temporary for learning.
Later we reduce permissions.

Click Next → Create user.

Step 4 — Create Access Key

After user is created:

Click the new user

Go to Security credentials

Click Create access key

Choose:

Command Line Interface (CLI)


Confirm

Copy:

Access key ID

Secret access key

✅ Step 5 — Configure CLI

Back in PowerShell:

aws configure


Enter:

Access Key ID

Secret Access Key

Region: us-east-1

Output: json

✅ Step 6 — Verify

Run:

aws sts get-caller-identity


---------------------------------------------------- Adding use in IAM ----------------------------------------------------


login ECR : aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 363283724240.dkr.ecr.us-east-1.amazonaws.com

Tag local image in ERC :  docker tag article-topic-classifier 363283724240.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest

PUSH to ECR : docker push 363283724240.dkr.ecr.us-east-1.amazonaws.com/article-topic-classifier-inference:latest




-------------------------------------------------------- know API ------------------

Perfect — since you’re okay with IP changing and just want to know how to get the updated URL each time, here’s the clean way to do it.

🔁 How To Get The Updated URL (After Redeployment)

Every time you:

Update service

Force new deployment

Change task definition

Scale service

You must fetch the new Public IP.

✅ Method 1 — From AWS Console (Manual Way)
Step 1

Go to:

ECS → Clusters → article-topic-cluster-v2

Step 2

Click:

Tasks

Step 3

Click the RUNNING task.

Step 4

Go to:

Networking


You’ll see:

Public IP: xx.xx.xx.xx


That is your new URL.

Then Access:
http://18.207.141.215:8000/health
topicclf-alb-1900064463.us-east-1.elb.amazonaws.com


Steps to update the IP (AWS Console)

Go to EC2

Left menu → Security Groups

Search and open: topicclf-alb-sg

Go to the Inbound rules tab

Click Edit inbound rules

Find the rule:

HTTP | Port 80

In the Source column:

Click the dropdown and select My IP

AWS will auto-fill something like x.x.x.x/32

Click Save rules















------------------------------------------------------------------------------

Phase 1 — ML System Build
1.1 Data + split discipline

Train/val/test split

No leakage checks

1.2 Model training

DistilBERT fine-tune (HF Transformers + PyTorch)

GPU training (RTX 3060)

3 epochs, converged cleanly

1.3 Evaluation

Accuracy ~90%

Balanced class performance

Confusion matrix review

✅ Status: Done

Phase 2 — Risk-aware Inference (Selective Classification)
2.1 Confidence routing

auto_accept (≥0.85 and gap ≥0.20)

needs_review (≥0.60 but ambiguous)

reject (<0.60)

2.2 Calibration + coverage

ECE implemented

Coverage vs accuracy curves

Reliability analysis

Selected threshold = 0.85

✅ Status: Done

Phase 3 — API + Containerization + Local Prod Readiness
3.1 API layer

FastAPI service

/predict, /batch_predict, /health

Pydantic schemas

Model load once at startup (lifespan)

MODEL_URI support (local + future S3)

3.2 Dockerization

Separate dev vs prod deps

CPU-only inference image

Solved CUDA bloat / disk overflow issues

✅ Status: Done

Phase 4 — Stabilization & Cloud Hardening (AWS)
4.1 Deploy on ECS Fargate

ECR push

ECS cluster/service running

CloudWatch logging

Public IP testing initially

✅ Done

4.2 Stable endpoint via ALB

Created Target Group (IP, port 8000, health /health)

Created ALB (listener HTTP:80 → target group)

Attached ECS service to target group

Fixed AZ/subnet mismatch (“AZ not enabled” issue)

Verified target healthy + /docs works via ALB

✅ Done

4.3 Access control (zero code)

ALB SG inbound restricted to My IP /32

✅ Done
⚠️ Manual maintenance needed when your ISP IP changes (you accepted this)

4.4 Backend lock-down (no more “backdoor”)

Created ECS task SG allowing 8000 only from ALB SG

Attached ECS service to that SG

✅ Done

4.5 HTTPS (ACM + 443 listener + redirect)

Request ACM cert

Add HTTPS listener

Redirect 80→443

⏭️ Skipped for now

4.6 Private subnets / no public IP on tasks (strongest posture)

Move tasks to private subnets

Disable auto-assign public IP

(Often needs NAT gateway or VPC endpoints)

⏭️ Skipped for now (you chose “Desired tasks = 0 when idle” instead)

✅ Phase 4 status: Core done, hardening partially skipped (by choice)




{
  "title": "Further sanctions on Russian banks impair Venezuela's access to hard currency",
  "body": "Context: Venezuela's export earnings stand to benefit from relatively high oil prices in the near term, partly driven by global geopolitical uncertainty. However, a moderate risk remains that the country will struggle to access its oil revenue, as a portion of these funds is channelled through the Russian financial system. The removal of several Russian banks from the SWIFT global payments network following the war in Ukraine could further restrict the Maduro regime's ability to access these proceeds. Trigger: Venezuela's access to oil revenue could face greater restrictions if Western powers impose further sanctions on Russia's financial system in an effort to intensify its economic isolation and end the war in Ukraine. Impact: Stronger sanctions on Russian banks would be likely to limit Venezuela's access to foreign exchange, leading to a drop in international reserves and tighter capital controls. This could trigger a new currency crisis and a possible return to recession, driving up exchange-rate volatility and fuelling renewed inflation. Mitigation: Businesses reliant on imported goods should factor in the risk of higher input costs when planning budgets and pricing strategies, particularly if tighter capital controls and currency depreciation materialise."
}

http://topicclf-alb-1900064463.us-east-1.elb.amazonaws.com/docs#