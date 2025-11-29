#!/bin/bash
# Create Kubernetes secrets for MailFind

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Create Kubernetes Secrets for MailFind${NC}"
echo "======================================="

# Default values
NAMESPACE="${NAMESPACE:-mailfind}"
ENV_FILE="${ENV_FILE:-.env}"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NAMESPACE   Kubernetes namespace (default: mailfind)"
    echo "  -f, --env-file FILE        Environment file (default: .env)"
    echo "  -h, --help                 Display this help message"
    echo ""
    echo "This script will prompt for required secrets if not found in env file."
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use .env file"
    echo "  $0 --env-file .env.production        # Use custom env file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -f|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Load environment file if it exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}Loading environment from: ${ENV_FILE}${NC}"
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}Warning: Environment file not found: ${ENV_FILE}${NC}"
    echo "You will be prompted to enter values manually."
fi

# Function to prompt for value
prompt_for_value() {
    local var_name=$1
    local description=$2
    local current_value=${!var_name}
    
    if [ -z "$current_value" ]; then
        echo -n -e "${YELLOW}Enter ${description} (${var_name}): ${NC}"
        read -r value
        export ${var_name}="$value"
    else
        echo -e "${GREEN}✓ ${var_name} loaded from environment${NC}"
    fi
}

echo ""
echo -e "${BLUE}Collecting required secrets...${NC}"
echo ""

# Prompt for all required secrets
prompt_for_value "SUPABASE_PUBLIC_URL" "Supabase URL"
prompt_for_value "SERVICE_ROLE" "Supabase Service Role Key"
prompt_for_value "SUPABASE_ANON_KEY" "Supabase Anon Key (optional, press enter to skip)"
prompt_for_value "GOOGLE_CLIENT_ID" "Google OAuth Client ID"
prompt_for_value "GOOGLE_CLIENT_SECRET" "Google OAuth Client Secret"
prompt_for_value "OPENAI_API_KEY" "OpenAI API Key"
prompt_for_value "ENCRYPTION_KEY" "Fernet Encryption Key"

# Validate required fields
if [ -z "$SUPABASE_PUBLIC_URL" ] || [ -z "$SERVICE_ROLE" ] || \
   [ -z "$GOOGLE_CLIENT_ID" ] || [ -z "$GOOGLE_CLIENT_SECRET" ] || \
   [ -z "$OPENAI_API_KEY" ] || [ -z "$ENCRYPTION_KEY" ]; then
    echo ""
    echo -e "${RED}Error: Missing required secrets${NC}"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Namespace '${NAMESPACE}' does not exist. Creating...${NC}"
    kubectl create namespace "$NAMESPACE"
fi

# Check if secret already exists
if kubectl get secret mailfind-secrets -n "$NAMESPACE" &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Secret 'mailfind-secrets' already exists in namespace '${NAMESPACE}'${NC}"
    echo -n "Do you want to delete and recreate it? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        kubectl delete secret mailfind-secrets -n "$NAMESPACE"
    else
        echo "Aborted"
        exit 0
    fi
fi

# Create secret
echo ""
echo -e "${BLUE}Creating Kubernetes secret...${NC}"

SECRET_ARGS="--from-literal=SUPABASE_PUBLIC_URL=$SUPABASE_PUBLIC_URL \
--from-literal=SERVICE_ROLE=$SERVICE_ROLE \
--from-literal=GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID \
--from-literal=GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET \
--from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
--from-literal=ENCRYPTION_KEY=$ENCRYPTION_KEY"

# Add optional fields
if [ -n "$SUPABASE_ANON_KEY" ]; then
    SECRET_ARGS="$SECRET_ARGS --from-literal=SUPABASE_ANON_KEY=$SUPABASE_ANON_KEY"
fi

kubectl create secret generic mailfind-secrets \
    --namespace="$NAMESPACE" \
    $SECRET_ARGS

echo ""
echo -e "${GREEN}✓ Secret created successfully!${NC}"
echo ""
echo "Verify with:"
echo "  kubectl get secret mailfind-secrets -n ${NAMESPACE}"
echo "  kubectl describe secret mailfind-secrets -n ${NAMESPACE}"
echo ""
echo -e "${YELLOW}Note: Secret values are base64 encoded but NOT encrypted at rest by default.${NC}"
echo "Consider using a secrets management solution for production (Vault, Sealed Secrets, etc.)"

