output "sagemaker_execution_role_arn" {
  description = "IAM role used by SageMaker training jobs"
  value       = aws_iam_role.sagemaker_execution_role.arn
}