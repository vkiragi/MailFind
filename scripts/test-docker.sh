#!/bin/bash
# Quick Docker Compose test script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "MailFind Docker Compose Test"
echo -e "========================================${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

test_command() {
    local name="$1"
    local cmd="$2"
    local expected="$3"
    
    echo -n "Testing: $name... "
    
    result=$(eval "$cmd" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        if [ -n "$expected" ]; then
            if echo "$result" | grep -q "$expected"; then
                echo -e "${GREEN}✓ PASSED${NC}"
                PASSED=$((PASSED + 1))
                return 0
            else
                echo -e "${YELLOW}⚠ WARNING${NC} (unexpected output)"
                WARNINGS=$((WARNINGS + 1))
                return 1
            fi
        else
            echo -e "${GREEN}✓ PASSED${NC}"
            PASSED=$((PASSED + 1))
            return 0
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Check prerequisites
echo -e "${YELLOW}Phase 1: Prerequisites${NC}"
test_command "Docker installed" "docker --version"
test_command "Docker Compose installed" "docker-compose --version"
test_command "Docker daemon running" "docker info"
test_command ".env file exists" "test -f .env"
test_command "docker-compose.yml exists" "test -f docker-compose.yml"
echo ""

# Check file structure
echo -e "${YELLOW}Phase 2: File Structure${NC}"
test_command "Backend Dockerfile exists" "test -f packages/backend/Dockerfile"
test_command "Frontend Dockerfile exists" "test -f packages/chrome-extension/Dockerfile"
test_command "Backend .dockerignore exists" "test -f packages/backend/.dockerignore"
test_command "Frontend .dockerignore exists" "test -f packages/chrome-extension/.dockerignore"
echo ""

# Build images
echo -e "${YELLOW}Phase 3: Building Images${NC}"
echo "This may take a few minutes..."
if test_command "Build backend image" "cd packages/backend && docker build -t mailfind-backend:test . -q"; then
    test_command "Backend image exists" "docker images mailfind-backend:test | grep -q mailfind-backend"
fi

if test_command "Build frontend image" "cd packages/chrome-extension && docker build -t mailfind-frontend:test . -q"; then
    test_command "Frontend image exists" "docker images mailfind-frontend:test | grep -q mailfind-frontend"
fi
echo ""

# Check image sizes
echo -e "${YELLOW}Phase 4: Image Sizes${NC}"
BACKEND_SIZE=$(docker images mailfind-backend:test --format "{{.Size}}" | head -1)
FRONTEND_SIZE=$(docker images mailfind-frontend:test --format "{{.Size}}" | head -1)

echo "  Backend image size:  $BACKEND_SIZE"
echo "  Frontend image size: $FRONTEND_SIZE"
echo ""

# Start services
echo -e "${YELLOW}Phase 5: Starting Services${NC}"
echo "Starting docker-compose..."
docker-compose up -d > /dev/null 2>&1 || {
    echo -e "${RED}Failed to start services${NC}"
    echo "Checking logs..."
    docker-compose logs --tail=20
    exit 1
}

echo "Waiting for services to be ready..."
sleep 10

test_command "Backend container running" "docker-compose ps | grep backend | grep -q Up"
test_command "Frontend container running" "docker-compose ps | grep frontend | grep -q Up"
echo ""

# Test endpoints
echo -e "${YELLOW}Phase 6: Testing Endpoints${NC}"
test_command "Backend health endpoint" "curl -f -s http://localhost:8000/" '{"status":"ok"}'
test_command "Frontend responds" "curl -f -s -I http://localhost:3000/ | head -1 | grep -q 200"
echo ""

# Check logs
echo -e "${YELLOW}Phase 7: Checking Logs${NC}"
echo "Backend logs (last 5 lines):"
docker-compose logs backend --tail=5
echo ""
echo "Frontend logs (last 5 lines):"
docker-compose logs frontend --tail=5
echo ""

# Check for errors
BACKEND_ERRORS=$(docker-compose logs backend | grep -i "error" | grep -v "0 error" | wc -l)
FRONTEND_ERRORS=$(docker-compose logs frontend | grep -i "error" | wc -l)

if [ "$BACKEND_ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No errors in backend logs"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC} Found $BACKEND_ERRORS error(s) in backend logs"
    WARNINGS=$((WARNINGS + 1))
fi

if [ "$FRONTEND_ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No errors in frontend logs"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC} Found $FRONTEND_ERRORS error(s) in frontend logs"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Resource usage
echo -e "${YELLOW}Phase 8: Resource Usage${NC}"
echo "Container stats:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep mailfind
echo ""

# Test restart
echo -e "${YELLOW}Phase 9: Testing Restart${NC}"
echo "Restarting services..."
docker-compose restart > /dev/null 2>&1
sleep 5
test_command "Backend restarted successfully" "curl -f -s http://localhost:8000/" '{"status":"ok"}'
test_command "Frontend restarted successfully" "curl -f -s -I http://localhost:3000/ | head -1 | grep -q 200"
echo ""

# Summary
echo -e "${BLUE}========================================"
echo "Test Summary"
echo -e "========================================${NC}"
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical tests passed!${NC}"
    echo ""
    echo "Services are running:"
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    echo ""
    echo "Commands:"
    echo "  View logs:    docker-compose logs -f"
    echo "  Stop:         docker-compose stop"
    echo "  Stop & Clean: docker-compose down"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  Check logs:   docker-compose logs"
    echo "  Check status: docker-compose ps"
    echo "  Clean up:     docker-compose down"
    exit 1
fi

