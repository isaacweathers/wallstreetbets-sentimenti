#!/bin/bash
set -e

# Source environment variables
if [ -f ../.env ]; then
  export $(grep -v '^#' ../.env | xargs)
elif [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Configuration
AWS_REGION="us-east-2"
# DOMAIN_NAME="synthetixsphere.com" # Replaced by .env variable

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS credentials are not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Looking for SSL certificate for ${DOMAIN_NAME} in AWS Certificate Manager..."
echo "AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "AWS Region: ${AWS_REGION}"

# List all certificates
CERTIFICATES=$(aws acm list-certificates --region ${AWS_REGION} --query "CertificateSummaryList[?DomainName=='${DOMAIN_NAME}' || DomainName=='*.${DOMAIN_NAME}'].CertificateArn" --output text)

if [ -z "$CERTIFICATES" ]; then
    echo "No certificate found for ${DOMAIN_NAME}."
    echo "Would you like to request a new certificate? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Requesting new certificate for ${DOMAIN_NAME}..."
        CERTIFICATE_ARN=$(aws acm request-certificate \
            --domain-name "${DOMAIN_NAME}" \
            --validation-method DNS \
            --region ${AWS_REGION} \
            --query "CertificateArn" \
            --output text)
        
        echo "Certificate requested successfully!"
        echo "Certificate ARN: ${CERTIFICATE_ARN}"
        echo ""
        echo "IMPORTANT: You need to validate this certificate by adding DNS records to your domain."
        echo "Run the following command to get the validation records:"
        echo "aws acm describe-certificate --certificate-arn ${CERTIFICATE_ARN} --region ${AWS_REGION} --query 'Certificate.DomainValidationOptions[0].ResourceRecord'"
    else
        echo "Certificate request cancelled."
        exit 1
    fi
else
    echo "Found certificate(s) for ${DOMAIN_NAME}:"
    echo "${CERTIFICATES}"
    echo ""
    echo "To use this certificate in your deployment, update the CERTIFICATE_ARN variable in aws/deploy.sh:"
    echo "CERTIFICATE_ARN=\"${CERTIFICATES}\""
fi