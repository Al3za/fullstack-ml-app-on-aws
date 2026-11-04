# ML Fullstack App

A fullstack machine learning project for customer churn prediction, combining a trained ML model with a web application and AWS cloud infrastructure.

The project brings together data analysis, machine learning, backend inference, frontend development, and infrastructure as code.

## Project Architecture

The planned inference flow is:

```text
Next.js Frontend
       |
       v
AWS API Gateway
       |
       v
AWS Lambda
       |
       v
Amazon SageMaker Endpoint
```

The ML workflow uses a customer churn dataset to develop, evaluate, and prepare a classification model for deployment.

## Technology Stack

- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **API:** Amazon API Gateway
- **Backend:** AWS Lambda
- **Machine Learning:** Python, pandas, scikit-learn, XGBoost, Random Forest, Logistic Regression
- **Model Hosting:** Amazon SageMaker
- **Data Storage:** Amazon S3
- **Infrastructure as Code:** Terraform

## Machine Learning Workflow

The local ML development workflow includes:

- Exploratory Data Analysis (EDA) and data quality checks
- Feature preprocessing and selection
- Model comparison and hyperparameter optimization
- Cross-validation and out-of-fold (OOF) predictions
- Classification threshold optimization
- Final evaluation on a held-out test set
- Model interpretation using permutation importance and SHAP

Detailed methodology, experiments, and evaluation results are available in [`docs/machine_learning.md`](docs/machine_learning.md).

## AWS Infrastructure

The project uses **Terraform** to provision and manage AWS infrastructure, with resources deployed in the `eu-central-1` (Frankfurt) region.

### Amazon S3

An S3 bucket stores the original **Telco Customer Churn dataset** and will also be used for **SageMaker training artifacts**.

- **Bucket:** `ml-project-dataset-alessandro-2026`
- **Region:** `eu-central-1`
- **Dataset:** `raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The bucket is provisioned using Terraform, with **S3 Block Public Access enabled**. Dataset uploads and transfers are managed separately using the AWS CLI or boto3.

### Infrastructure as Code

Terraform configuration is maintained in `infra/aws/` and currently includes:

- **S3 bucket provisioning and public access restrictions**.
- **SageMaker IAM execution role and S3 access policy configuration**.
- **Terraform outputs for AWS resource identifiers**.

The IAM execution role is intended to provide SageMaker with access to the training dataset and model artifact storage.

### SageMaker Training Preparation

The local XGBoost classification pipeline has been adapted for execution in an Amazon SageMaker training container.

The training implementation includes:

- Reusable feature definitions and preprocessing.
- Reproducible train/test splitting.
- Model training with hyperparameters selected during local experimentation.
- Model artifact serialization using joblib.
- Preservation of the optimized classification threshold (`0.35`).

The project uses **SageMaker Python SDK V3** and plans to orchestrate training jobs using `ModelTrainer`.

## Project Status

- [x] Project structure and local development environment
- [x] Local ML analysis, model development, and evaluation
- [x] Local model serialization and inference prototype
- [x] S3 infrastructure provisioning with Terraform
- [x] Dataset migration to AWS Frankfurt (`eu-central-1`)
- [x] SageMaker training script and dependency preparation
- [x] SageMaker IAM role and policy configuration in Terraform
- [ ] SageMaker IAM role deployment and verification
- [ ] SageMaker training job execution and artifact validation
- [ ] SageMaker inference endpoint deployment
- [ ] AWS Lambda inference integration
- [ ] API Gateway integration
- [ ] Frontend integration
- [ ] End-to-end testing and deployment
- [ ] Production-oriented security and infrastructure improvements

## Getting Started

The project is currently under active development.

### Prerequisites

- Python 3.11
- Node.js and npm
- AWS CLI configured with valid credentials
- Terraform
- An AWS account with the permissions required to provision infrastructure and run SageMaker training jobs

### Project Components

- `frontend/` — Next.js frontend application.
- `ml/` — Local machine learning development, preprocessing, evaluation, and inference.
- `sagemaker/training/` — SageMaker training source code and dependencies.
- `infra/aws/` — Terraform infrastructure configuration.
- `docs/` — Machine learning methodology and project documentation.

Detailed setup, training, deployment, and testing instructions will be added as the cloud integration is completed.
