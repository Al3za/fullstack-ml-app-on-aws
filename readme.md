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

The cloud training workflow has been successfully implemented:

```text
Amazon S3 (Raw Dataset)
       |
       v
Amazon SageMaker Training Job
       |
       +----> Amazon CloudWatch Logs
       |
       v
Amazon S3 (Model Artifacts)
```

## Technology Stack

- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **API:** Amazon API Gateway
- **Backend:** AWS Lambda
- **Machine Learning:** Python, pandas, scikit-learn, XGBoost, Random Forest, Logistic Regression
- **Model Training and Hosting:** Amazon SageMaker
- **Data Storage:** Amazon S3
- **Monitoring and Logging:** Amazon CloudWatch
- **Infrastructure as Code:** Terraform
- **AWS Integration:** boto3, AWS CLI, SageMaker Python SDK V3

## Machine Learning Workflow

The local ML development workflow includes:

- Exploratory Data Analysis (EDA) and data quality checks
- Feature preprocessing and selection
- Model comparison and hyperparameter optimization
- Cross-validation and out-of-fold (OOF) predictions
- Classification threshold optimization
- Final evaluation on a held-out test set
- Model interpretation using permutation importance and SHAP
- Final model serialization using joblib

XGBoost was selected as the final classification model, with an optimized classification threshold of `0.35`.

Detailed methodology, experiments, and evaluation results are available in [`docs/machine_learning.md`](docs/machine_learning.md).

## AWS Infrastructure

The project uses **Terraform** to provision and manage AWS infrastructure, with resources deployed in the `eu-central-1` (Frankfurt) region.

### Amazon S3

An S3 bucket stores the original **Telco Customer Churn dataset** and the **SageMaker training artifacts**.

- **Bucket:** `ml-project-dataset-alessandro-2026`
- **Region:** `eu-central-1`
- **Dataset:** `raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Training output:** `sagemaker/output/`

The bucket is provisioned using Terraform, with **S3 Block Public Access enabled**. Dataset uploads and transfers are managed separately using the AWS CLI or boto3.

Model artifacts are automatically uploaded to S3 by SageMaker after successful training.

### Infrastructure as Code

Terraform configuration is maintained in `infra/aws/` and currently includes:

- **S3 bucket provisioning and public access restrictions.**
- **SageMaker IAM execution role and trust relationship.**
- **IAM policies for dataset access and model artifact storage.**
- **IAM permissions required for SageMaker training operations.**
- **CloudWatch Logs permissions for training monitoring and debugging.**
- **Terraform outputs for AWS resource identifiers.**

The SageMaker execution role has been successfully deployed and validated through actual cloud training execution.

IAM permissions are managed through dedicated policies, following the principle of least privilege where applicable.

### Amazon CloudWatch

Amazon CloudWatch Logs is configured to collect SageMaker training job logs.

The training log group is:

```text
/aws/sagemaker/TrainingJobs
```

CloudWatch provides visibility into:

- Training container initialization.
- Python dependency installation.
- Training script execution.
- Runtime errors and Python tracebacks.
- Training diagnostics and troubleshooting.

CloudWatch logging has been successfully validated during cloud training development and debugging.

### SageMaker Cloud Training

The local XGBoost classification pipeline has been adapted for execution in an Amazon SageMaker training container.

The training implementation includes:

- Reusable feature definitions and preprocessing.
- Reproducible train/test splitting.
- Model training with hyperparameters selected during local experimentation.
- Model artifact serialization using joblib.
- Preservation of the optimized classification threshold (`0.35`).
- Automatic model artifact upload to Amazon S3.

The project uses **SageMaker Python SDK V3** to orchestrate training jobs through `ModelTrainer`.

The training workflow separates infrastructure orchestration from machine learning execution:

- `run_training.py` — Initializes the SageMaker session, configures the training environment, and submits the training job.
- `sagemaker/training/train.py` — Executes data loading, preprocessing, model training, and artifact serialization inside the training container.
- `sagemaker/training/requirements.txt` — Defines additional Python dependencies required by the training environment.

#### Training Configuration

| Parameter                | Value           |
| ------------------------ | --------------- |
| AWS Region               | `eu-central-1`  |
| Training Instance        | `ml.m5.large`   |
| Instance Count           | 1               |
| Training Framework       | Scikit-learn    |
| Container Version        | `1.9-0-cpu-py3` |
| XGBoost Version          | `3.2.0`         |
| Classification Threshold | `0.35`          |
| Maximum Runtime          | 1800 seconds    |
| Training Input           | Amazon S3       |
| Artifact Storage         | Amazon S3       |

The required SageMaker training instance quota has been successfully approved.

#### Cloud Training Results

The first successful cloud training job was completed on November 24, 2026.

| Metric                | Result      |
| --------------------- | ----------- |
| Training Job Status   | Completed   |
| Training Time         | 84 seconds  |
| Billable Time         | 84 seconds  |
| Model Artifact Upload | Successful  |
| CloudWatch Logging    | Operational |

The generated model archive is stored in Amazon S3 using the following structure:

```text
sagemaker/output/
└── <training-job-name>/
    └── output/
        └── model.tar.gz
```

The archive is expected to contain:

```text
model.tar.gz
├── xgb_final.joblib
└── threshold.json
```

The cloud training workflow is operational. Model archive inspection and validation of the cloud-trained model are the next development steps.

### SageMaker Inference

A dedicated inference script has been implemented in `sagemaker/inference/inference.py` to prepare the trained model for deployment to a SageMaker inference endpoint.

The inference implementation follows the SageMaker model serving lifecycle:

| Function       | Responsibility                                                                   |
| -------------- | -------------------------------------------------------------------------------- |
| `model_fn()`   | Loads the serialized model and classification threshold                          |
| `input_fn()`   | Deserializes JSON requests into a pandas DataFrame                               |
| `predict_fn()` | Generates churn probabilities and applies the optimized classification threshold |
| `output_fn()`  | Serializes prediction results into a JSON response                               |

The inference workflow accepts a JSON object containing the 19 customer features required by the trained classification pipeline.

An example inference response is:

```json
{
  "churn_probability": 0.18,
  "churn_prediction": 0,
  "threshold": 0.35
}
```

The probability shown above is illustrative.

The inference implementation has been successfully validated through an end-to-end local test using the locally trained model artifacts.

Cloud-trained model validation and SageMaker endpoint deployment are pending.

## Project Status

### Machine Learning

- [x] Project structure and local development environment
- [x] Local ML analysis, model development, and evaluation
- [x] Hyperparameter and classification threshold optimization
- [x] Local model serialization
- [x] SageMaker inference script implementation
- [x] End-to-end local inference testing

### AWS Infrastructure and Cloud Training

- [x] S3 infrastructure provisioning with Terraform
- [x] Dataset migration to AWS Frankfurt (`eu-central-1`)
- [x] SageMaker training script and dependency preparation
- [x] SageMaker IAM role and policy configuration in Terraform
- [x] SageMaker IAM role deployment and verification
- [x] SageMaker training instance quota approval
- [x] CloudWatch Logs configuration and validation
- [x] Successful SageMaker cloud training job execution
- [x] Automatic model artifact upload to Amazon S3
- [ ] Cloud model artifact inspection and validation
- [ ] Local inference testing using cloud-trained model artifacts
- [ ] Local and cloud model prediction comparison

### Cloud Deployment and Fullstack Integration

- [ ] SageMaker inference artifact packaging
- [ ] SageMaker inference endpoint deployment
- [ ] Remote inference testing
- [ ] AWS Lambda inference integration
- [ ] API Gateway integration
- [ ] Frontend integration
- [ ] End-to-end testing and deployment
- [ ] Production-oriented security and infrastructure improvements

## Getting Started

The project is currently under active development.

### Prerequisites

- Python 3.11 for local development
- Node.js and npm
- AWS CLI configured with valid credentials
- Terraform
- An AWS account with the permissions required to provision infrastructure and run SageMaker training jobs
- An available SageMaker training instance quota in the target AWS region

### Project Components

- `frontend/` — Next.js frontend application.
- `ml/` — Local machine learning development, preprocessing, evaluation, and inference.
- `ml/models/artifacts/` — Locally generated model artifacts.
- `sagemaker/training/` — SageMaker training source code and dependencies.
- `sagemaker/inference/` — SageMaker inference implementation and local inference testing.
- `run_training.py` — SageMaker cloud training orchestration script.
- `infra/aws/` — Terraform infrastructure configuration.
- `docs/` — Machine learning methodology and project documentation.

### Running Cloud Training

Once the AWS infrastructure has been provisioned, the local Python environment has been configured, and the required SageMaker quota is available, a cloud training job can be submitted from the project root:

```bash
python run_training.py
```

Training logs are available through Amazon CloudWatch, and successfully generated model artifacts are automatically uploaded to Amazon S3.

Running SageMaker training jobs incurs AWS charges.

### Running Local Inference Tests

The SageMaker inference implementation can be tested locally using the existing model artifacts:

```bash
python sagemaker/inference/test_inference.py
```

The test validates model loading, JSON input processing, prediction generation, and JSON response serialization.

Detailed setup, deployment, and testing instructions will be added as the cloud integration is completed.
