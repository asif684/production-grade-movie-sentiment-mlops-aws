# 🎬 Production-Grade Movie Sentiment Analysis using MLOps on AWS

This repository demonstrates an **end-to-end production-grade MLOps pipeline** for movie sentiment classification — fully automated, containerized, deployed, and monitored using **AWS cloud services** and **DevOps tools**.  
It follows a modern MLOps architecture integrating **DVC**, **MLflow**, **Docker**, **GitHub Actions (CI/CD)**, **Kubernetes (EKS)**, **Prometheus**, and **Grafana**.

---

## 🧠 Project Overview

### 🚀 Objective
To build, train, deploy, and monitor a sentiment analysis model using an **MLOps pipeline** on **AWS** infrastructure with complete automation — from data ingestion to production monitoring.

### 🔁 End-to-End Pipeline
1. **Data Ingestion** – Fetch data from AWS S3 Bucket.  
2. **Data Preprocessing** – Clean and prepare raw text data.  
3. **Feature Engineering** – Transform data for ML models.  
4. **Model Building** – Train ML model using scikit-learn / deep learning models.  
5. **Model Evaluation** – Evaluate metrics and log results to MLflow.  
6. **Model Registry** – Store and version models in MLflow model registry.  
7. **Deployment** – Containerized Flask app deployed on AWS EKS (Kubernetes).  
8. **Monitoring** – Metrics scraped by Prometheus and visualized on Grafana.

---

## ☁️ Tech Stack & Tools

| Category | Tools Used |
|-----------|-------------|
| **Programming** | Python 3.10 |
| **MLOps Frameworks** | DVC, MLflow, Dagshub |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Orchestration** | Kubernetes (EKS) |
| **Monitoring** | Prometheus, Grafana |
| **AWS Cloud Services** | EC2, S3, ECR, EKS, IAM, CloudFormation |
| **Web Framework** | Flask |
| **Version Control** | Git & GitHub |

---
---

##  Flow-Chart
---
![imag alt](https://github.com/asif684/production-grade-movie-sentiment-mlops-aws/blob/e55e702242d4de7cdd8a74a06ce820d912bd4730/movie_mlops.png)
---

## 🏗️ Project Structure Setup

```bash
# 1️⃣ Create repo and clone locally
git clone <repo-url>
cd production-grade-movie-sentiment-mlops-aws

# 2️⃣ Create virtual environment
conda create -n atlas python=3.10
conda activate atlas

# 3️⃣ Install cookiecutter
pip install cookiecutter

# 4️⃣ Generate base project
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

# 5️⃣ Rename directory
mv src/models src/model

# 6️⃣ Commit setup
git add .
git commit -m "Initial project setup"
git push
```
---
## 🧾 MLflow Setup (Using Dagshub)

- Go to Dagshub  → Create a new repo and connect to GitHub.

- Copy your MLflow tracking URL and credentials.

- Install and initialize MLflow:
```bash
pip install dagshub mlflow
```
- Run experiments and push results — MLflow will automatically log to Dagshub.

---
## 📦 DVC Setup (Data Version Control)
---
```bash
dvc init
mkdir local_s3
dvc remote add -d mylocal local_s3
```
---
### Add all pipeline scripts inside src/:
---
- data_ingestion.py

- data_preprocessing.py

- feature_engineering.py

- model_building.py

- model_evaluation.py

- register_model.py
---
### Create:
- `dvc.yaml` (pipeline stages)

- `params.yaml` (model parameters)
---
### Then run:
```bash
dvc repro
dvc status
git add .
git commit -m "Added DVC pipeline"
git push
```
---
## ☁️ AWS S3 Integration (for DVC Remote Storage)
---
```bash
pip install 'dvc[s3]' awscli
aws configure
dvc remote add -d myremote s3://<bucket-name>
```
---
## 🔥 Flask App Setup
---
```bash
mkdir flask_app
pip install flask
dvc push   # Push data to S3
pip freeze > requirements.txt
```
![img alt](https://github.com/asif684/production-grade-movie-sentiment-mlops-aws/blob/5d9c1d08e5e9bd1ac7ad8cd09c3a8e1edf3df69d/movie-app-img.png)
---
## ⚙️ CI/CD with GitHub Actions
---
- Create `.github/workflows/ci.yaml` for automated testing, build, and deployment.
---
### Required GitHub Secrets:
- AWS_ACCESS_KEY_ID

- AWS_SECRET_ACCESS_KEY

- AWS_REGION

- AWS_ACCOUNT_ID

- ECR_REPOSITORY

- CAPSTONE_TEST (Dagshub token)
---
## 🐳 Docker Setup
---
```bash
cd flask_app
pip install pipreqs
pipreqs . --force

docker build -t capstone-app:latest .
docker run -p 8888:5000 -e CAPSTONE_TEST=<token> capstone-app:latest
```
---
### Push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker tag capstone-app:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/capstone-proj:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/capstone-proj:latest
```
---
### ☸️ AWS EKS Setup (Kubernetes Cluster)
---
### Install CLI Tools

- `AWS CLI v2`

- `kubectl`

- `eksctl`
---
### Verify Installation
```bash
aws --version
kubectl version --client
eksctl version
```
---
### Create Cluster
```bash
eksctl create cluster \
--name flask-app-cluster \
--region us-east-1 \
--nodegroup-name flask-app-nodes \
--node-type t3.small \
--nodes 1 --nodes-min 1 --nodes-max 1 --managed
```
---
### Verify:
```bash
aws eks list-clusters
kubectl get nodes
```
---
### Deploy App
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl get svc
```
---
### Access app:
```bash
http://<external-ip>:5000
```
---
### 📊 Monitoring with Prometheus & Grafana
---
### 🔹 Prometheus Setup (on EC2)
---
```bash
# Allow inbound 9090
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz
mv prometheus-2.46.0.linux-amd64 /etc/prometheus
```
Edit` /etc/prometheus/prometheus.yml`:
```bash
scrape_configs:
  - job_name: "flask-app"
    static_configs:
      - targets: ["<EKS-load-balancer-endpoint>:5000"]
```
---
### Run Prometheus:
---
```bash
/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml
```
---
### Access:
---
```bash
http://<ec2-public-ip>:9090
```
---
🔹 Grafana Setup (on EC2)
---
```bash
http://<ec2-public-ip>:3000
```
---
### (Default credentials: admin/admin)
- Add Prometheus as a data source and create dashboards for metrics.
---
## 🏁 Final Architecture
---
                 ┌──────────────────────────┐
                 │         GitHub CI/CD     │
                 │ (Build → Test → Deploy)  │
                 └────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │      Docker       │
                    │  Container Image  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    AWS ECR        │
                    │  Image Registry   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    AWS EKS        │
                    │ (Flask Deployment)│
                    └─────────┬─────────┘
                              │
             ┌────────────────▼────────────────┐
             │  Prometheus  +  Grafana (EC2)   │
             │   Real-time monitoring setup    │
             └────────────────────────────────┘
