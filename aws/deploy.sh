#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f ../.env ]; then # Check for .env in parent directory
    export $(cat ../.env | grep -v '^#' | xargs)
elif [ -f .env ]; then # Check for .env in current directory
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found in current or parent directory"
    exit 1
fi

# Configuration
AWS_REGION=${AWS_REGION:-us-east-2} # Use environment variable or default
STACK_NAME="wallstreetbets-sentiment"
# DOMAIN_NAME="synthetixsphere.com" # Replaced by .env variable
ECR_REPOSITORY_NAME="${STACK_NAME}-repo"

# AWS_ACCOUNT_ID is expected to be in .env
# CERTIFICATE_ARN is expected to be in .env
# DOMAIN_NAME is expected to be in .env

ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"


# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install it first."
    exit 1
fi

# Check if AWS_ACCOUNT_ID is set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: AWS_ACCOUNT_ID is not set in the .env file or environment"
    exit 1
fi

# Check if DOMAIN_NAME is set
if [ -z "$DOMAIN_NAME" ]; then
    echo "Error: DOMAIN_NAME is not set in the .env file or environment"
    exit 1
fi

# Check if CERTIFICATE_ARN is set
if [ -z "$CERTIFICATE_ARN" ]; then
    echo "Error: CERTIFICATE_ARN is not set in the .env file or environment"
    exit 1
fi

echo "Deploying WallStreetBets Sentiment Analysis to AWS ECS..."
echo "AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "AWS Region: ${AWS_REGION}"
echo "Stack Name: ${STACK_NAME}"
echo "Domain Name: ${DOMAIN_NAME}"

# Create ECR repository if it doesn't exist
echo "Creating ECR repository if it doesn't exist..."
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} || \
aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build Docker imag
 
 e
echo "Building Docker image..."
docker build --platform linux/amd64 -t ${ECR_REPOSITORY_NAME}:latest .

# Tag and push Docker image
echo "Tagging and pushing Docker image to ECR..."
docker tag ${ECR_REPOSITORY_NAME}:latest ${ECR_REPOSITORY_URI}:latest
docker push ${ECR_REPOSITORY_URI}:latest

# Deploy CloudFormation stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file aws/cloudformation.yml \
  --stack-name ${STACK_NAME} \
  --parameter-overrides \
    DomainName=${DOMAIN_NAME} \
    Environment=production \
    CertificateArn=${CERTIFICATE_ARN} \
  --capabilities CAPABILITY_IAM \
  --region ${AWS_REGION}

# Get the task definition from the CloudFormation stack
echo "Getting task definition from CloudFormation stack..."
TASK_DEFINITION=$(aws cloudformation describe-stacks \
  --stack-name ${STACK_NAME} \
  --query "Stacks[0].Outputs[?OutputKey=='TaskDefinition'].OutputValue" \
  --output text \
  --region ${AWS_REGION})

# Update the ECS service
echo "Updating ECS service..."
aws ecs update-service \
  --cluster ${STACK_NAME}-cluster \
  --service ${STACK_NAME}-service \
  --force-new-deployment \
  --region ${AWS_REGION}

echo "Deployment completed successfully!"
echo "Your application is now available at https://${DOMAIN_NAME}"