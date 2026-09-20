provider "aws" {
  region = "eu-west-1"
}

resource "aws_s3_bucket" "ml_bucket" {
  bucket = "ml-project-dataset-nome-alessandro-2026"

  tags = {
    Project = "ml-fullstack-app"
    Environment = "dev"
  }
}

