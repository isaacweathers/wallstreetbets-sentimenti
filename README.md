# WallStreetBets Sentiment Analysis

A web application that analyzes sentiment and topics from r/wallstreetbets posts using natural language processing and machine learning techniques.

## Features

- **Sentiment Analysis**: Analyzes the sentiment of posts using VADER sentiment analysis
- **Topic Modeling**: Identifies main topics in posts using LDA (Latent Dirichlet Allocation)
- **Ticker Extraction**: Automatically identifies stock tickers mentioned in posts
- **Trend Analysis**: Visualizes trends in ticker mentions, sentiment, and topics over time
- **Interactive UI**: Modern, responsive interface with dark/light mode toggle

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Reddit API account (for accessing r/wallstreetbets data)
- Docker (for containerization)
- AWS CLI (for AWS deployment)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wallstreetbets-sentiment.git
   cd wallstreetbets-sentiment
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

   To obtain these credentials:
   1. Go to https://www.reddit.com/prefs/apps
   2. Click "Create another app..."
   3. Select "script" as the application type
   4. Fill in the required information
   5. After creation, note the client ID (under the app name) and client secret

### Running the Application Locally

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t wallstreetbets-sentiment .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 --env-file .env wallstreetbets-sentiment
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## AWS Deployment

### Pre-Deployment Checklist & Setup

Before running the AWS deployment script (`aws/deploy.sh`), ensure you have completed the following steps:

1.  **AWS Account and CLI Configuration**:
    *   Ensure you have an AWS account with the necessary permissions to create and manage resources like ECS, ECR, VPC, Load Balancers, Route53, S3, and IAM roles.
    *   Install and configure the AWS CLI with your credentials. You can test your configuration by running `aws sts get-caller-identity`.

2.  **Domain Name and SSL Certificate**:
    *   You must own or have control over a registered domain name (e.g., `example.com`).
    *   This domain must be configured as a Hosted Zone in AWS Route53.
    *   Request or import an SSL/TLS certificate for your domain (and `*.yourdomain.com` for wildcards if needed) in AWS Certificate Manager (ACM) in the **us-east-1 (N. Virginia) region**. This is crucial because Application Load Balancers require certificates from this region.
    *   Ensure the certificate is validated (e.g., via DNS validation through Route53). Note the **ARN** of this certificate.

3.  **`.env` File Configuration**:
    *   Create a `.env` file in the root directory of the project (`/workspaces/WALLSTREETBETS/.env`).
    *   Populate this file with the following values, replacing placeholders with your actual information:
        ```env
        REDDIT_CLIENT_ID=your_reddit_client_id
        REDDIT_CLIENT_SECRET=your_reddit_client_secret
        REDDIT_USER_AGENT=your_reddit_user_agent_string (e.g., MyWSBApp/0.1 by u/yourusername)
        AWS_REGION=your_target_aws_region (e.g., us-east-2)
        AWS_ACCOUNT_ID=your_12_digit_aws_account_id
        DOMAIN_NAME=your_domain_name (e.g., example.com)
        CERTIFICATE_ARN=arn_of_your_acm_certificate_in_us-east-1
        ```
    *   **Important**: The `DOMAIN_NAME` in `.env` must match the domain for which your SSL certificate was issued.
    *   The `AWS_ACCOUNT_ID` can be found by running `aws sts get-caller-identity --query Account --output text`.

4.  **Reddit API Credentials (Optional - AWS Systems Manager Parameter Store)**:
    *   While the `.env` file is used by `deploy.sh` for some parameters, the application itself (when deployed to ECS) is designed to fetch Reddit credentials from AWS Systems Manager Parameter Store for better security.
    *   Store your Reddit API credentials as SecureString parameters:
        ```bash
        aws ssm put-parameter --name "/wallstreetbets/reddit/client_id" --value "your_reddit_client_id" --type SecureString --region your_target_aws_region
        aws ssm put-parameter --name "/wallstreetbets/reddit/client_secret" --value "your_reddit_client_secret" --type SecureString --region your_target_aws_region
        aws ssm put-parameter --name "/wallstreetbets/reddit/user_agent" --value "your_reddit_user_agent" --type SecureString --region your_target_aws_region
        ```
    *   Ensure the ECS Task Execution Role created by the CloudFormation template has permissions to read these parameters.

5.  **Docker Installation**:
    *   Ensure Docker is installed and running on the machine where you will execute the `deploy.sh` script. The script uses Docker to build and push the application image to Amazon ECR.

### Deployment Steps

1. Make sure you have AWS credentials configured:
   ```bash
   aws configure
   ```

2. Create an SSL certificate in AWS Certificate Manager (ACM):
   - Go to the ACM console
   - Request a certificate for your domain (e.g., example.com)
   - Validate the certificate using DNS validation
   - Note the certificate ARN for the deployment

3. ~~Store your Reddit API credentials in AWS Systems Manager Parameter Store:~~ (This step is now covered in the Pre-Deployment Checklist)
   ```bash
   aws ssm put-parameter --name "/wallstreetbets/reddit/client_id" --value "your_client_id" --type SecureString
   aws ssm put-parameter --name "/wallstreetbets/reddit/client_secret" --value "your_client_secret" --type SecureString
   aws ssm put-parameter --name "/wallstreetbets/reddit/user_agent" --value "your_user_agent" --type SecureString
   ```

4. ~~Update the `aws/deploy.sh` script with your certificate ARN:~~ (This is now handled by the .env file)
   ```bash
   # Edit the script to add your certificate ARN
   CERTIFICATE_ARN="arn:aws:acm:us-east-1:123456789012:certificate/your-certificate-id"
   ```

5. Make the deployment script executable:
   ```bash
   chmod +x aws/deploy.sh
   ```

6. Run the deployment script:
   ```bash
   ./aws/deploy.sh
   ```

7. The script will:
   - Create an ECR repository
   - Build and push the Docker image
   - Deploy the CloudFormation stack
   - Create the ECS cluster and service
   - Set up the Application Load Balancer
   - Configure Route53 DNS records

8. Once deployment is complete, your application will be available at:
   ```
   https://example.com
   ```

### AWS Infrastructure

The deployment creates the following AWS resources:

- **VPC**: Virtual Private Cloud with public subnets
- **ECS Cluster**: Container orchestration service
- **ECR Repository**: Container registry for Docker images
- **Application Load Balancer**: For routing traffic to the ECS tasks
- **Route53**: DNS configuration for your domain
- **CloudWatch Logs**: For application logging
- **IAM Roles**: For ECS task execution and permissions

## Usage Guide

### Main Dashboard

The main dashboard displays:
- Recent posts from r/wallstreetbets
- Sentiment analysis results for each post
- Identified stock tickers
- Topic modeling results

### Sentiment Analysis

- Each post is analyzed for sentiment (positive, negative, or neutral)
- Sentiment scores are displayed with color-coded badges
- The sentiment analysis uses VADER, which is specifically tuned for social media text

### Topic Modeling

- The application automatically identifies main topics in the posts
- Topics are displayed with their associated keywords
- Each post is assigned to the most relevant topic

### Ticker Extraction

- Stock tickers are automatically extracted from post content
- Tickers are displayed as badges on each post
- The system recognizes common stock symbols and filters out false positives

### Trend Analysis

- Visualizes trends in ticker mentions over time
- Shows sentiment trends for specific tickers
- Displays topic popularity trends
- Provides interactive plots and data tables

### UI Features

- **Dark/Light Mode**: Toggle between dark and light themes using the switch in the header
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Elements**: Hover over elements for additional information
- **Data Tables**: Sort and filter data in the trend analysis tables

## Troubleshooting

### Common Issues

1. **Reddit API Authentication Errors**:
   - Verify your Reddit API credentials in the `.env` file or AWS Parameter Store
   - Ensure your Reddit account has the necessary permissions

2. **NLTK Data Missing**:
   - The application will attempt to download required NLTK data automatically
   - If issues persist, manually download the data:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('averaged_perceptron_tagger')
     nltk.download('wordnet')
     nltk.download('vader_lexicon')
     ```

3. **Application Not Starting**:
   - Check that all dependencies are installed correctly
   - Verify that the Flask application is running on the correct port
   - Check the console for error messages

4. **AWS Deployment Issues**:
   - Check CloudFormation stack events for errors
   - Verify that your AWS credentials have the necessary permissions
   - Ensure your SSL certificate is valid and properly configured
   - Check ECS task logs in CloudWatch for application errors

## Project Structure

- `app.py`: Main Flask application
- `text_processor.py`: Text processing and sentiment analysis
- `topic_modeler.py`: Topic modeling using LDA
- `trend_analyzer.py`: Trend analysis and visualization
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static files
- `aws/`: AWS deployment configuration files
  - `cloudformation.yml`: CloudFormation template
  - `task-definition.json`: ECS task definition
  - `deploy.sh`: Deployment script

## Technologies Used

- **Flask**: Web framework
- **PRAW**: Reddit API wrapper
- **NLTK**: Natural language processing
- **VADER**: Sentiment analysis
- **Gensim**: Topic modeling
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive plots
- **Docker**: Containerization
- **AWS ECS**: Container orchestration
- **AWS CloudFormation**: Infrastructure as code
- **AWS Route53**: DNS management

## Future Enhancements

- Add more advanced trend analysis features
- Implement machine learning for stock price prediction
- Add user authentication for personalized dashboards
- Expand data sources beyond Reddit
- Implement real-time analysis with WebSockets
- Add CI/CD pipeline for automated deployments

## License

This project is licensed under the MIT License - see the LICENSE file for details.