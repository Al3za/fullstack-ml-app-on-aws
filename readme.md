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

<!-- Additional data exploration and dataset findings are documented in [`docs/data_understanding.md`](docs/data_understanding.md). -->

## Project Status

- [x] Local ML analysis, model development, and evaluation
- [ ] Model packaging and inference preparation
- [ ] SageMaker model deployment
- [ ] Lambda inference integration
- [ ] API Gateway integration
- [ ] Frontend integration
- [ ] Terraform infrastructure completion
- [ ] End-to-end testing and deployment

## Getting Started

Setup instructions for the ML environment, frontend, AWS resources, and deployment will be added as the corresponding components are completed.
