#!/bin/bash
# Build Docker images for MailFind

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MailFind Docker Images${NC}"
echo "======================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Parse arguments
REGISTRY="${REGISTRY:-localhost}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-false}"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --registry REGISTRY   Container registry (default: localhost)"
    echo "  -t, --tag TAG            Image tag (default: latest)"
    echo "  -p, --push               Push images to registry after building"
    echo "  -h, --help               Display this help message"
    echo ""
    echo "Environment variables:"
    echo "  REGISTRY                 Same as --registry"
    echo "  TAG                      Same as --tag"
    echo "  PUSH                     Same as --push (set to 'true')"
    echo ""
    echo "Examples:"
    echo "  $0 --registry docker.io/myuser --tag v1.0.0 --push"
    echo "  REGISTRY=gcr.io/myproject TAG=latest $0 --push"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH="true"
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

# Build backend image
echo ""
echo -e "${YELLOW}Building backend image...${NC}"
cd "$PROJECT_ROOT/packages/backend"
docker build -t mailfind-backend:${TAG} .
docker tag mailfind-backend:${TAG} ${REGISTRY}/mailfind-backend:${TAG}
echo -e "${GREEN}✓ Backend image built: ${REGISTRY}/mailfind-backend:${TAG}${NC}"

# Build frontend image
echo ""
echo -e "${YELLOW}Building frontend image...${NC}"
cd "$PROJECT_ROOT/packages/chrome-extension"
docker build -t mailfind-frontend:${TAG} .
docker tag mailfind-frontend:${TAG} ${REGISTRY}/mailfind-frontend:${TAG}
echo -e "${GREEN}✓ Frontend image built: ${REGISTRY}/mailfind-frontend:${TAG}${NC}"

# Push images if requested
if [ "$PUSH" = "true" ]; then
    echo ""
    echo -e "${YELLOW}Pushing images to registry...${NC}"
    
    docker push ${REGISTRY}/mailfind-backend:${TAG}
    echo -e "${GREEN}✓ Backend image pushed${NC}"
    
    docker push ${REGISTRY}/mailfind-frontend:${TAG}
    echo -e "${GREEN}✓ Frontend image pushed${NC}"
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Images:"
echo "  ${REGISTRY}/mailfind-backend:${TAG}"
echo "  ${REGISTRY}/mailfind-frontend:${TAG}"

if [ "$PUSH" != "true" ]; then
    echo ""
    echo -e "${YELLOW}Note: Images not pushed. Use --push flag to push to registry.${NC}"
fi

