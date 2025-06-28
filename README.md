## Lambda Function Deployment

## Automatic Deployment

1. Configure AWS secrets in GitHub Actions

2. Pushes to `master` trigger automatic deployment.

3. Workflow:
   - Install dependencies
   - Creates a ZIP package
   - Updates existing Lambda function

## Requirements
- AWS Lambda previously configured
- IAM credentials with permissions to update Lambda