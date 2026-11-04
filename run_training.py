
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    SourceCode,
    Compute,
    InputData
)


# --------------------------------------------------
# AWS CONFIGURATION
# --------------------------------------------------

REGION = "eu-central-1"

# Inserire il nome effettivo del bucket S3
BUCKET_NAME = "INSERISCI-IL-TUO-BUCKET"

# URI del dataset originale
DATASET_URI = (
    f"s3://{BUCKET_NAME}/raw/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Inserire l'ARN del ruolo IAM di SageMaker
ROLE_ARN = "INSERISCI-ARN-DEL-RUOLO"


# --------------------------------------------------
# SAGEMAKER SESSION
# --------------------------------------------------

session = Session()

print("SageMaker session initialized.")


# --------------------------------------------------
# TRAINING CONTAINER
# --------------------------------------------------

# already checked if existed
training_image = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.9-0",
    image_scope="training",
    instance_type="ml.m5.large"
)

print("Training image:")
print(training_image)


# --------------------------------------------------
# SOURCE CODE
# --------------------------------------------------

source_code = SourceCode(
    source_dir="sagemaker/training",
    entry_script="train.py",
    requirements="requirements.txt"
)


# --------------------------------------------------
# COMPUTE CONFIGURATION
# --------------------------------------------------

compute = Compute(
    instance_type="ml.m5.large",
    instance_count=1
)


# --------------------------------------------------
# TRAINING DATA. Comunicare a SageMaker dove si trova il dataset e come deve renderlo disponibile al container di training
# --------------------------------------------------

train_input = InputData(
    channel_name="train",
    data_source=DATASET_URI # dove e' salvato il dataset Telco che abbiamo caricato manualmente prima del traning di sagemaker
)


# --------------------------------------------------
# MODEL TRAINER
# --------------------------------------------------

trainer = ModelTrainer(
    training_image=training_image,
    role=ROLE_ARN,
    sagemaker_session=session,
    source_code=source_code,
    compute=compute,
    base_job_name="telco-churn-training"
)


# --------------------------------------------------
# START TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting SageMaker training job...")

    trainer.train(
        input_data_config=[train_input],
        wait=True,
        logs=True,
        dry_run=True
    )

    print("Dry run completed successfully.")
