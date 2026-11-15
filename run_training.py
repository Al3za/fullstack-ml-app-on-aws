
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes.shapes import OutputDataConfig
from sagemaker.train import ModelTrainer
from sagemaker.core.shapes.shapes import StoppingCondition # Per evitare che un qualunque problema lasci il job attivo troppo a lungo (e consumare possibilmente risorse e denaro), aggiungiamo un limite di esecuzione.

import boto3
from sagemaker.core.training.configs import (
    SourceCode,
    Compute,
    InputData,
)


# --------------------------------------------------
# AWS CONFIGURATION
# --------------------------------------------------

REGION = "eu-central-1"

# Nome del bucket S3
BUCKET_NAME = "ml-project-dataset-alessandro-2026"

# URI del dataset originale
DATASET_URI = (
    f"s3://{BUCKET_NAME}/raw/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Invochiamo l'ARN del ruolo IAM di SageMaker con boto3 per evitare di scrivere l'arn nel file
account_id = boto3.client("sts").get_caller_identity()["Account"]

ROLE_ARN = (
    f"arn:aws:iam::{account_id}:role/"
    "ml-fullstack-sagemaker-execution-role"
)


# --------------------------------------------------
# SAGEMAKER SESSION
# --------------------------------------------------

# Assicuriamo che i file preparatori del training vengano caricati nel nostro bucket S3 a Francoforte, 
# non in un bucket "predefinito" diverso
boto_session = boto3.Session(region_name=REGION)

session = Session(
    boto_session=boto_session,
    default_bucket=BUCKET_NAME,
)

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
    requirements="requirements.txt",
       ignore_patterns=[
        ".env",
        ".git",
        "__pycache__",
        ".DS_Store",
        ".cache",
        ".ipynb_checkpoints",
        "sagemaker_tasks",
    ], # Ignore unusefull files and docs i wrote like sagemaker_tasks, .cache
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

# SageMaker rende disponibile il dataset all'interno del container nella directory **/opt/ml/input/data/train/** definita nel file train.py
train_input = InputData(
    channel_name="train",
    data_source=DATASET_URI # dove e' salvato il dataset Telco che abbiamo caricato manualmente prima del traning di sagemaker. Sagemaker lo usera in train.py per leggerlo dalla cartella di input del container docker che crea prima di avviare il training.
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
    base_job_name="telco-churn-training",
    output_data_config=OutputDataConfig( # Specifichiamo dove salvare gli artefatti. (OutputDataConfig serve per compatibilita' della tipizazzione sdk ed evitare problemi di validazione)
    s3_output_path=f"s3://{BUCKET_NAME}/sagemaker/output"
    ),
    stopping_condition=StoppingCondition(
    max_runtime_in_seconds=1800 # limite massimo di 30 minuti per il training job
    )
)


# --------------------------------------------------
# START TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting SageMaker training job...")

    trainer.train(
        input_data_config=[train_input], # inizializiamo il training
        wait=True,
        logs=True,
        dry_run=True # serve a verificare come SageMaker SDK prepara il training job, senza avviare effettivamente un'istanza ml.m5.large. (ottimo per avviare un test e vedere se ci sono problemi di compatibilita' durante il training)
    )

    print("Dry run completed successfully.")
