#!/bin/bash
# Deploy MailFind to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying MailFind to Kubernetes${NC}"
echo "======================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
K8S_DIR="$PROJECT_ROOT/k8s"

# Parse arguments
NAMESPACE="${NAMESPACE:-mailfind}"
ACTION="${ACTION:-apply}"
USE_KUSTOMIZE="${USE_KUSTOMIZE:-true}"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NAMESPACE   Kubernetes namespace (default: mailfind)"
    echo "  -a, --action ACTION         Action: apply or delete (default: apply)"
    echo "  -k, --kustomize             Use kustomize (default: true)"
    echo "  --no-kustomize              Don't use kustomize"
    echo "  -h, --help                  Display this help message"
    echo ""
    echo "Environment variables:"
    echo "  NAMESPACE                   Same as --namespace"
    echo "  ACTION                      Same as --action"
    echo "  USE_KUSTOMIZE               Same as --kustomize"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy with kustomize"
    echo "  $0 --no-kustomize                     # Deploy without kustomize"
    echo "  $0 --action delete                    # Delete all resources"
    echo "  NAMESPACE=mailfind-dev $0             # Deploy to different namespace"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -k|--kustomize)
            USE_KUSTOMIZE="true"
            shift
            ;;
        --no-kustomize)
            USE_KUSTOMIZE="false"
            shift
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

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    echo "Please configure kubectl to access your cluster"
    exit 1
fi

echo -e "${BLUE}Connected to cluster:${NC}"
kubectl cluster-info | head -n 1

# Function to apply or delete resources
apply_resources() {
    if [ "$USE_KUSTOMIZE" = "true" ]; then
        echo ""
        echo -e "${YELLOW}Using Kustomize...${NC}"
        kubectl ${ACTION} -k "$K8S_DIR/"
    else
        echo ""
        echo -e "${YELLOW}Applying manifests...${NC}"
        
        # Apply in order
        kubectl ${ACTION} -f "$K8S_DIR/namespace.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/configmap.yaml"
        
        # Check if secrets.yaml exists
        if [ -f "$K8S_DIR/secrets.yaml" ]; then
            kubectl ${ACTION} -f "$K8S_DIR/secrets.yaml"
        else
            echo -e "${YELLOW}Warning: secrets.yaml not found. You need to create it from secrets.yaml.template${NC}"
            echo "Run: cp $K8S_DIR/secrets.yaml.template $K8S_DIR/secrets.yaml"
            echo "Then edit secrets.yaml with your actual base64-encoded values"
        fi
        
        kubectl ${ACTION} -f "$K8S_DIR/backend-deployment.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/backend-service.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/frontend-deployment.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/frontend-service.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/ingress.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/hpa.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/network-policy.yaml"
        kubectl ${ACTION} -f "$K8S_DIR/persistent-volume.yaml"
    fi
}

# Deploy or delete
if [ "$ACTION" = "apply" ]; then
    echo ""
    echo -e "${YELLOW}Deploying to namespace: ${NAMESPACE}${NC}"
    apply_resources
    
    echo ""
    echo -e "${GREEN}Deployment initiated!${NC}"
    echo ""
    echo -e "${BLUE}Checking deployment status...${NC}"
    
    # Wait a bit for resources to be created
    sleep 5
    
    # Show pod status
    echo ""
    echo -e "${YELLOW}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE}
    
    # Show service status
    echo ""
    echo -e "${YELLOW}Services:${NC}"
    kubectl get svc -n ${NAMESPACE}
    
    # Show ingress status
    echo ""
    echo -e "${YELLOW}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    echo ""
    echo -e "${GREEN}Useful commands:${NC}"
    echo "  kubectl get pods -n ${NAMESPACE}"
    echo "  kubectl logs -f deployment/mailfind-backend -n ${NAMESPACE}"
    echo "  kubectl logs -f deployment/mailfind-frontend -n ${NAMESPACE}"
    echo "  kubectl describe pod <pod-name> -n ${NAMESPACE}"
    echo "  kubectl port-forward svc/mailfind-backend 8000:8000 -n ${NAMESPACE}"
    
elif [ "$ACTION" = "delete" ]; then
    echo ""
    echo -e "${RED}Deleting resources from namespace: ${NAMESPACE}${NC}"
    echo -e "${YELLOW}Are you sure? This will delete all MailFind resources. (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        apply_resources
        echo ""
        echo -e "${GREEN}Resources deleted${NC}"
    else
        echo "Aborted"
        exit 0
    fi
else
    echo -e "${RED}Unknown action: ${ACTION}${NC}"
    echo "Valid actions: apply, delete"
    exit 1
fi

