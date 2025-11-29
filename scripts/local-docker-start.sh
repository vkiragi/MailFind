#!/bin/bash
# Quick start script for local Docker development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MailFind with Docker Compose${NC}"
echo "======================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${RED}Please edit .env with your actual values before continuing${NC}"
        exit 1
    else
        echo -e "${RED}Error: .env.example not found${NC}"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Parse arguments
BUILD=false
DETACH=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --attach)
            DETACH=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--build] [--attach]"
            echo "  --build   Rebuild images before starting"
            echo "  --attach  Run in foreground (show logs)"
            exit 1
            ;;
    esac
done

# Build images if requested
if [ "$BUILD" = true ]; then
    echo ""
    echo -e "${YELLOW}Building images...${NC}"
    docker-compose build
fi

# Start services
echo ""
echo -e "${YELLOW}Starting services...${NC}"

if [ "$DETACH" = true ]; then
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}✓ Services started!${NC}"
    echo ""
    echo "Access:"
    echo "  Backend API:  http://localhost:8000"
    echo "  Frontend:     http://localhost:3000"
    echo "  Health Check: http://localhost:8000/"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs -f"
    echo "  docker-compose logs -f backend"
    echo "  docker-compose logs -f frontend"
    echo ""
    echo "Stop services:"
    echo "  docker-compose down"
else
    echo ""
    echo -e "${YELLOW}Running in foreground. Press Ctrl+C to stop.${NC}"
    echo ""
    docker-compose up
fi

