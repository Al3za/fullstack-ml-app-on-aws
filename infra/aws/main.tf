
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "eu-central-1"
}

resource "aws_s3_bucket" "ml_bucket" {
  bucket = "ml-project-dataset-alessandro-2026"

  tags = {
    Project     = "ml-fullstack-app"
    Environment = "dev"
  }
}

resource "aws_s3_bucket_public_access_block" "ml_bucket" {
  bucket = aws_s3_bucket.ml_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "bucket_name" {
  value = aws_s3_bucket.ml_bucket.bucket
}

output "bucket_region" {
  value = "eu-central-1"
}
