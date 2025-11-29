# MailFind - Docker & Kubernetes Testing Guide

This document provides comprehensive test cases to verify your Docker and Kubernetes deployment is working correctly.

## Table of Contents

1. [Pre-Deployment Tests](#pre-deployment-tests)
2. [Docker Compose Tests](#docker-compose-tests)
3. [Docker Image Tests](#docker-image-tests)
4. [Kubernetes Tests](#kubernetes-tests)
5. [Integration Tests](#integration-tests)
6. [Performance Tests](#performance-tests)
7. [Troubleshooting Tests](#troubleshooting-tests)

---

## Pre-Deployment Tests

### Test 1: Check Dependencies

**Purpose**: Verify all required tools are installed

```bash
make check-deps
```

**Expected Output**:
```
✓ Docker
✓ Docker Compose
✓ kubectl
✓ Python 3
✓ Node.js
✓ .env file exists
```

**Manual Check**:
```bash
docker --version          # Should be 20.10+
docker-compose --version  # Should be 2.0+
kubectl version --client  # Should be 1.20+
python3 --version        # Should be 3.10+
node --version           # Should be 18+
```

### Test 2: Verify File Structure

**Purpose**: Ensure all deployment files exist

```bash
# Check Docker files
test -f docker-compose.yml && echo "✓ docker-compose.yml exists"
test -f packages/backend/Dockerfile && echo "✓ Backend Dockerfile exists"
test -f packages/chrome-extension/Dockerfile && echo "✓ Frontend Dockerfile exists"

# Check Kubernetes files
test -d k8s && echo "✓ k8s directory exists"
test -f k8s/backend-deployment.yaml && echo "✓ K8s backend deployment exists"

# Check scripts
test -x scripts/build-images.sh && echo "✓ Build script is executable"
test -x scripts/deploy-k8s.sh && echo "✓ Deploy script is executable"
```

### Test 3: Validate Environment Configuration

**Purpose**: Ensure .env file has all required variables

```bash
# Create test script
cat > /tmp/test_env.sh << 'EOF'
#!/bin/bash
REQUIRED_VARS=(
    "SUPABASE_PUBLIC_URL"
    "SERVICE_ROLE"
    "GOOGLE_CLIENT_ID"
    "GOOGLE_CLIENT_SECRET"
    "OPENAI_API_KEY"
    "ENCRYPTION_KEY"
)

if [ ! -f .env ]; then
    echo "❌ .env file not found"
    exit 1
fi

source .env

echo "Checking required environment variables..."
MISSING=0
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ $var is not set"
        MISSING=$((MISSING + 1))
    else
        echo "✓ $var is set"
    fi
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "✓ All required variables are set"
else
    echo ""
    echo "❌ $MISSING variable(s) missing"
    exit 1
fi
EOF

chmod +x /tmp/test_env.sh
./tmp/test_env.sh
```

---

## Docker Compose Tests

### Test 4: Build Docker Images

**Purpose**: Verify images build successfully

```bash
# Build without cache to test from scratch
docker-compose build --no-cache
```

**Expected Output**: Both backend and frontend should build without errors

**Verify**:
```bash
docker images | grep mailfind
```

**Expected**:
```
mailfind-backend     latest
mailfind-frontend    latest
```

### Test 5: Start Services

**Purpose**: Verify services start correctly

```bash
# Start in detached mode
docker-compose up -d

# Wait for services to be ready
sleep 10

# Check status
docker-compose ps
```

**Expected Output**: Both services should be "Up" and healthy

### Test 6: Backend Health Check

**Purpose**: Verify backend API is responding

```bash
# Test health endpoint
curl -f http://localhost:8000/

# Test with verbose output
curl -v http://localhost:8000/
```

**Expected Output**:
```json
{"status":"ok"}
```

**Status Code**: 200

### Test 7: Frontend Health Check

**Purpose**: Verify frontend is serving

```bash
# Test frontend
curl -I http://localhost:3000/

# Check for HTML content
curl -s http://localhost:3000/ | head -n 20
```

**Expected Output**: HTTP 200, HTML content with proper headers

### Test 8: Backend Logs

**Purpose**: Verify backend is running without errors

```bash
# Check logs
docker-compose logs backend | tail -n 50

# Look for errors
docker-compose logs backend | grep -i error
```

**Expected**: No critical errors, server started successfully

### Test 9: Frontend Logs

**Purpose**: Verify frontend nginx is running

```bash
# Check logs
docker-compose logs frontend | tail -n 50
```

**Expected**: Nginx started successfully, no errors

### Test 10: Container Resource Usage

**Purpose**: Verify containers aren't consuming excessive resources

```bash
# Check resource usage
docker stats --no-stream
```

**Expected**: 
- Backend: < 500MB memory under no load
- Frontend: < 50MB memory

### Test 11: Environment Variables in Containers

**Purpose**: Verify env vars are passed correctly

```bash
# Check backend env vars
docker-compose exec backend env | grep -E "SUPABASE|GOOGLE|OPENAI"

# Verify they're not empty
docker-compose exec backend bash -c 'echo "SUPABASE_PUBLIC_URL=$SUPABASE_PUBLIC_URL"'
```

**Expected**: All variables should have values (not empty)

### Test 12: Network Connectivity

**Purpose**: Verify containers can communicate

```bash
# Frontend should be able to reach backend
docker-compose exec frontend wget -O- http://backend:8000/

# Check network
docker network ls | grep mailfind
docker network inspect mailfind_mailfind-network
```

**Expected**: Frontend can reach backend, network exists

### Test 13: Volume Persistence

**Purpose**: Verify volumes are working

```bash
# Check volumes
docker volume ls | grep mailfind

# Inspect volume
docker volume inspect mailfind_backend-logs
```

**Expected**: Volumes exist and are mounted

### Test 14: Stop and Restart

**Purpose**: Verify graceful shutdown and restart

```bash
# Stop services
docker-compose stop

# Verify stopped
docker-compose ps

# Start again
docker-compose start

# Wait and check
sleep 5
curl -f http://localhost:8000/
```

**Expected**: Services stop cleanly and restart successfully

### Test 15: Clean Teardown

**Purpose**: Verify cleanup works

```bash
# Stop and remove
docker-compose down

# Verify removed
docker-compose ps
docker ps -a | grep mailfind
```

**Expected**: All containers removed

---

## Docker Image Tests

### Test 16: Build Backend Image

**Purpose**: Verify backend image builds independently

```bash
cd packages/backend
docker build -t test-backend:test .
```

**Expected**: Build completes without errors

### Test 17: Build Frontend Image

**Purpose**: Verify frontend image builds independently

```bash
cd packages/chrome-extension
docker build -t test-frontend:test .
```

**Expected**: Build completes without errors

### Test 18: Image Size Verification

**Purpose**: Ensure images aren't too large

```bash
docker images | grep -E "mailfind|test-"
```

**Expected Sizes**:
- Backend: < 2GB
- Frontend: < 200MB

### Test 19: Image Security Scan

**Purpose**: Check for vulnerabilities (if you have docker scan)

```bash
# Scan backend
docker scan mailfind-backend:latest || echo "Docker scan not available"

# Scan frontend
docker scan mailfind-frontend:latest || echo "Docker scan not available"
```

### Test 20: Run Backend Container Standalone

**Purpose**: Verify backend runs without compose

```bash
# Run backend with minimal env vars
docker run -d \
  --name test-backend \
  -p 8001:8000 \
  -e SUPABASE_PUBLIC_URL=test \
  -e SERVICE_ROLE=test \
  -e GOOGLE_CLIENT_ID=test \
  -e GOOGLE_CLIENT_SECRET=test \
  -e OPENAI_API_KEY=test \
  -e ENCRYPTION_KEY=test \
  mailfind-backend:latest

# Check logs
sleep 5
docker logs test-backend

# Cleanup
docker stop test-backend
docker rm test-backend
```

**Expected**: Container starts, logs show startup (may fail on actual API calls without valid credentials)

---

## Kubernetes Tests

### Test 21: Kubectl Access

**Purpose**: Verify kubectl can access cluster

```bash
# Check connection
kubectl cluster-info

# Check current context
kubectl config current-context

# List nodes
kubectl get nodes
```

**Expected**: Connected to cluster, nodes visible

### Test 22: Validate Kubernetes Manifests

**Purpose**: Verify YAML syntax is correct

```bash
# Validate all manifests
for file in k8s/*.yaml; do
    echo "Validating $file..."
    kubectl apply --dry-run=client -f "$file" 2>&1 | grep -v "Warning" || echo "Error in $file"
done
```

**Expected**: No syntax errors

### Test 23: Validate Kustomization

**Purpose**: Verify kustomize configuration

```bash
# Build kustomization (don't apply)
kubectl kustomize k8s/

# Check output
kubectl kustomize k8s/ | grep -E "kind:|name:"
```

**Expected**: All resources rendered correctly

### Test 24: Create Namespace

**Purpose**: Verify namespace creation

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Verify
kubectl get namespace mailfind

# Check labels
kubectl describe namespace mailfind
```

**Expected**: Namespace created with proper labels

### Test 25: Create ConfigMap

**Purpose**: Verify configmap creation

```bash
# Apply configmap
kubectl apply -f k8s/configmap.yaml

# Verify
kubectl get configmap -n mailfind

# Check contents
kubectl describe configmap mailfind-config -n mailfind
```

**Expected**: ConfigMap created with correct data

### Test 26: Create Secrets

**Purpose**: Verify secrets creation (use script)

```bash
# Create secrets using script
./scripts/create-k8s-secrets.sh --namespace mailfind

# Verify
kubectl get secret mailfind-secrets -n mailfind

# Check keys (not values)
kubectl describe secret mailfind-secrets -n mailfind
```

**Expected**: Secret created with all required keys

### Test 27: Deploy Backend

**Purpose**: Verify backend deployment

```bash
# Apply backend deployment
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml

# Wait for pods
kubectl wait --for=condition=ready pod -l app=mailfind,component=backend -n mailfind --timeout=120s

# Check status
kubectl get pods -n mailfind
kubectl get svc -n mailfind
```

**Expected**: Backend pods running, service created

### Test 28: Backend Pod Health

**Purpose**: Verify backend pods are healthy

```bash
# Check pod status
kubectl get pods -n mailfind -l component=backend

# Describe pod
POD_NAME=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n mailfind

# Check logs
kubectl logs $POD_NAME -n mailfind
```

**Expected**: Pod running, no crash loops, logs show startup

### Test 29: Backend Service Health

**Purpose**: Verify backend service is working

```bash
# Port forward
kubectl port-forward svc/mailfind-backend 8002:8000 -n mailfind &
PF_PID=$!

sleep 3

# Test endpoint
curl -f http://localhost:8002/

# Cleanup
kill $PF_PID
```

**Expected**: Service responds with {"status":"ok"}

### Test 30: Deploy Frontend

**Purpose**: Verify frontend deployment

```bash
# Apply frontend deployment
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml

# Wait for pods
kubectl wait --for=condition=ready pod -l app=mailfind,component=frontend -n mailfind --timeout=120s

# Check status
kubectl get pods -n mailfind -l component=frontend
```

**Expected**: Frontend pods running

### Test 31: Frontend Pod Health

**Purpose**: Verify frontend pods are healthy

```bash
# Check pod status
kubectl get pods -n mailfind -l component=frontend

# Check logs
POD_NAME=$(kubectl get pods -n mailfind -l component=frontend -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD_NAME -n mailfind
```

**Expected**: Nginx running, serving files

### Test 32: HPA Deployment

**Purpose**: Verify Horizontal Pod Autoscaler

```bash
# Apply HPA
kubectl apply -f k8s/hpa.yaml

# Check HPA status
kubectl get hpa -n mailfind

# Describe HPA
kubectl describe hpa mailfind-backend-hpa -n mailfind
```

**Expected**: HPA created, monitoring deployments

### Test 33: Network Policies

**Purpose**: Verify network policies are applied

```bash
# Apply network policies
kubectl apply -f k8s/network-policy.yaml

# Check policies
kubectl get networkpolicy -n mailfind

# Describe policy
kubectl describe networkpolicy mailfind-backend-policy -n mailfind
```

**Expected**: Network policies created

### Test 34: Ingress Deployment

**Purpose**: Verify ingress configuration

```bash
# Apply ingress (may need ingress controller installed)
kubectl apply -f k8s/ingress.yaml

# Check ingress
kubectl get ingress -n mailfind

# Describe ingress
kubectl describe ingress mailfind-ingress -n mailfind
```

**Expected**: Ingress created (may not have external IP without controller)

### Test 35: Pod-to-Pod Communication

**Purpose**: Verify backend and frontend can communicate

```bash
# Get pod names
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')
FRONTEND_POD=$(kubectl get pods -n mailfind -l component=frontend -o jsonpath='{.items[0].metadata.name}')

# Test frontend -> backend
kubectl exec $FRONTEND_POD -n mailfind -- wget -O- http://mailfind-backend:8000/

# Test with curl from a test pod
kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n mailfind -- \
  curl -f http://mailfind-backend:8000/
```

**Expected**: Pods can reach each other

### Test 36: Resource Usage in K8s

**Purpose**: Verify resource limits are working

```bash
# Check resource usage
kubectl top pods -n mailfind

# Check resource quotas
kubectl describe deployment mailfind-backend -n mailfind | grep -A 10 "Limits\|Requests"
```

**Expected**: Pods within limits

### Test 37: Rolling Update Test

**Purpose**: Verify rolling updates work

```bash
# Trigger a rolling update
kubectl set image deployment/mailfind-backend backend=mailfind-backend:latest -n mailfind

# Watch rollout
kubectl rollout status deployment/mailfind-backend -n mailfind

# Check history
kubectl rollout history deployment/mailfind-backend -n mailfind
```

**Expected**: Rollout completes successfully, zero downtime

### Test 38: Scaling Test

**Purpose**: Verify manual scaling works

```bash
# Scale up
kubectl scale deployment mailfind-backend --replicas=3 -n mailfind

# Verify
kubectl get pods -n mailfind -l component=backend

# Scale down
kubectl scale deployment mailfind-backend --replicas=2 -n mailfind

# Verify
kubectl get pods -n mailfind -l component=backend
```

**Expected**: Pods scale up and down correctly

### Test 39: Persistent Volume Test

**Purpose**: Verify PVC is working

```bash
# Apply PVC
kubectl apply -f k8s/persistent-volume.yaml

# Check PVC
kubectl get pvc -n mailfind

# Check if bound
kubectl describe pvc mailfind-backend-logs -n mailfind
```

**Expected**: PVC created and bound (if storage class available)

### Test 40: Complete Deployment Script Test

**Purpose**: Test automated deployment script

```bash
# Run deployment script
./scripts/deploy-k8s.sh --namespace mailfind-test

# Verify
kubectl get all -n mailfind-test

# Cleanup
./scripts/deploy-k8s.sh --namespace mailfind-test --action delete
```

**Expected**: Script deploys all resources successfully

---

## Integration Tests

### Test 41: End-to-End Backend Test

**Purpose**: Test actual backend functionality

```bash
# Port forward backend
kubectl port-forward svc/mailfind-backend 8003:8000 -n mailfind &
PF_PID=$!

sleep 3

# Test health
curl http://localhost:8003/

# Test auth status (will fail without credentials, but should return proper error)
curl -v http://localhost:8003/auth/status

# Cleanup
kill $PF_PID
```

**Expected**: Endpoints respond (even if with auth errors)

### Test 42: Environment Variables in Pods

**Purpose**: Verify secrets are loaded correctly

```bash
# Check env vars in backend pod
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')

kubectl exec $BACKEND_POD -n mailfind -- env | grep -E "SUPABASE|GOOGLE|OPENAI|ENCRYPTION"
```

**Expected**: All env vars present (values masked)

### Test 43: External Service Connectivity

**Purpose**: Verify pods can reach external services

```bash
# Test from backend pod
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')

# Test Supabase connectivity (use your actual URL)
kubectl exec $BACKEND_POD -n mailfind -- curl -I https://supabase.co

# Test OpenAI connectivity
kubectl exec $BACKEND_POD -n mailfind -- curl -I https://api.openai.com
```

**Expected**: Can reach external services

### Test 44: DNS Resolution

**Purpose**: Verify service discovery works

```bash
# Test from backend pod
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')

# Test service DNS
kubectl exec $BACKEND_POD -n mailfind -- nslookup mailfind-backend
kubectl exec $BACKEND_POD -n mailfind -- nslookup mailfind-frontend
```

**Expected**: Services resolve correctly

### Test 45: Load Test (Basic)

**Purpose**: Test basic load handling

```bash
# Port forward
kubectl port-forward svc/mailfind-backend 8004:8000 -n mailfind &
PF_PID=$!

sleep 3

# Simple load test (if you have ab or hey installed)
ab -n 100 -c 10 http://localhost:8004/ 2>/dev/null || \
hey -n 100 -c 10 http://localhost:8004/ 2>/dev/null || \
echo "Install ab (apache-bench) or hey for load testing"

# Cleanup
kill $PF_PID
```

**Expected**: Handles concurrent requests

---

## Performance Tests

### Test 46: Startup Time

**Purpose**: Measure pod startup time

```bash
# Delete and recreate a pod
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')

echo "Deleting pod..."
kubectl delete pod $BACKEND_POD -n mailfind

echo "Waiting for new pod..."
time kubectl wait --for=condition=ready pod -l app=mailfind,component=backend -n mailfind --timeout=120s
```

**Expected**: Backend pod ready in < 60 seconds

### Test 47: Resource Limits Test

**Purpose**: Verify resource limits are enforced

```bash
# Check current usage
kubectl top pod -n mailfind

# Compare with limits
kubectl describe deployment mailfind-backend -n mailfind | grep -A 4 "Limits:"
```

**Expected**: Usage within limits

### Test 48: HPA Trigger Test

**Purpose**: Test autoscaling triggers (requires metrics-server)

```bash
# Check HPA metrics
kubectl get hpa -n mailfind

# Generate load to trigger scaling
kubectl port-forward svc/mailfind-backend 8005:8000 -n mailfind &
PF_PID=$!

# Run load for 2 minutes
for i in {1..120}; do 
    curl -s http://localhost:8005/ > /dev/null & 
done

sleep 30

# Check if HPA triggered
kubectl get hpa -n mailfind

kill $PF_PID
```

**Expected**: HPA shows metrics, may trigger scaling under load

---

## Troubleshooting Tests

### Test 49: Crash Recovery

**Purpose**: Verify pods recover from crashes

```bash
# Get backend pod
BACKEND_POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')

# Kill the process inside (simulate crash)
kubectl exec $BACKEND_POD -n mailfind -- kill 1 || true

# Wait and check if recovered
sleep 10
kubectl get pods -n mailfind -l component=backend
```

**Expected**: Pod restarts automatically

### Test 50: Log Collection

**Purpose**: Verify logs are accessible

```bash
# Get logs from all backend pods
kubectl logs -l component=backend -n mailfind --tail=50

# Get logs from all frontend pods
kubectl logs -l component=frontend -n mailfind --tail=50

# Check for errors
kubectl logs -l component=backend -n mailfind | grep -i error | tail -20
```

**Expected**: Logs accessible, minimal errors

---

## Quick Test Script

Create a comprehensive test runner:

```bash
cat > test-deployment.sh << 'EOF'
#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
}

echo "========================================"
echo "MailFind Deployment Tests"
echo "========================================"
echo ""

# Pre-deployment tests
echo "Pre-deployment Tests:"
run_test "Docker installed" "docker --version"
run_test "Docker Compose installed" "docker-compose --version"
run_test "kubectl installed" "kubectl version --client"
run_test ".env file exists" "test -f .env"
run_test "docker-compose.yml exists" "test -f docker-compose.yml"
run_test "Backend Dockerfile exists" "test -f packages/backend/Dockerfile"
run_test "Frontend Dockerfile exists" "test -f packages/chrome-extension/Dockerfile"

echo ""
echo "Docker Tests:"
run_test "Backend image exists" "docker images mailfind-backend:latest | grep -q mailfind-backend"
run_test "Frontend image exists" "docker images mailfind-frontend:latest | grep -q mailfind-frontend"

echo ""
echo "Kubernetes Tests (if deployed):"
run_test "K8s namespace exists" "kubectl get namespace mailfind"
run_test "K8s backend deployment exists" "kubectl get deployment mailfind-backend -n mailfind"
run_test "K8s frontend deployment exists" "kubectl get deployment mailfind-frontend -n mailfind"
run_test "K8s backend service exists" "kubectl get svc mailfind-backend -n mailfind"

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC}"
echo "========================================"

if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
EOF

chmod +x test-deployment.sh
```

Run all tests:
```bash
./test-deployment.sh
```

---

## Summary Checklist

Use this checklist to track your testing progress:

### Docker Compose
- [ ] Images build successfully
- [ ] Services start without errors
- [ ] Backend health endpoint responds
- [ ] Frontend serves content
- [ ] Containers communicate
- [ ] Logs show no critical errors
- [ ] Environment variables loaded
- [ ] Services restart cleanly

### Kubernetes
- [ ] Manifests are valid YAML
- [ ] Namespace creates successfully
- [ ] ConfigMap applies correctly
- [ ] Secrets create successfully
- [ ] Backend deployment runs
- [ ] Frontend deployment runs
- [ ] Services are accessible
- [ ] HPA is configured
- [ ] Network policies apply
- [ ] Pods can communicate
- [ ] Rolling updates work
- [ ] Scaling works

### Integration
- [ ] Backend API endpoints work
- [ ] Frontend serves properly
- [ ] External services reachable
- [ ] DNS resolution works
- [ ] Logs are accessible
- [ ] Metrics are available

---

## Next Steps After Testing

1. If all tests pass: ✅ Ready for production
2. If tests fail: 📋 Check logs and troubleshoot
3. Document any issues: 📝 Update troubleshooting guide
4. Performance tune: ⚡ Adjust resources based on metrics

Need help with any failing tests? Check:
- Container logs: `docker-compose logs` or `kubectl logs`
- Events: `kubectl get events -n mailfind --sort-by='.lastTimestamp'`
- Describe resources: `kubectl describe pod <pod-name> -n mailfind`

