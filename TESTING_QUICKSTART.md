# Testing Quick Start Guide

Quick reference for testing your MailFind Docker and Kubernetes deployment.

## 🚀 Quick Test Commands

### Test Everything (Recommended)

```bash
# Run comprehensive Docker tests
make test-docker

# Run comprehensive Kubernetes tests
make test-k8s

# Run all tests
make test-all
```

### Individual Component Tests

```bash
# Test Docker backend
make test-backend

# Test Docker frontend
make test-frontend

# Test Kubernetes backend
make test-k8s-backend
```

---

## 📋 Manual Testing Checklist

### Docker Compose Testing (5 minutes)

```bash
# 1. Check prerequisites
make check-deps

# 2. Start services
make docker-start

# 3. Run automated tests
./scripts/test-docker.sh

# 4. Manual verification
curl http://localhost:8000/           # Should return {"status":"ok"}
curl -I http://localhost:3000/        # Should return HTTP 200

# 5. Check logs
docker-compose logs backend | tail -20
docker-compose logs frontend | tail -20

# 6. Stop services
make docker-down
```

### Kubernetes Testing (10 minutes)

```bash
# 1. Verify kubectl access
kubectl cluster-info

# 2. Validate manifests
kubectl apply --dry-run=client -f k8s/

# 3. Deploy (if not already deployed)
make k8s-deploy

# 4. Run automated tests
./scripts/test-k8s.sh

# 5. Check pod status
kubectl get pods -n mailfind

# 6. Test backend endpoint
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind &
curl http://localhost:8000/
# Press Ctrl+C to stop port forward

# 7. Check logs
kubectl logs -f deployment/mailfind-backend -n mailfind --tail=20
```

---

## 🔍 Common Test Scenarios

### Scenario 1: First Time Setup

```bash
# Step 1: Create environment
cp .env.example .env
# Edit .env with your values

# Step 2: Test Docker locally
make test-docker

# Step 3: If Docker works, test Kubernetes
make build-push REGISTRY=your-registry TAG=v1.0.0
make k8s-secrets
make k8s-deploy
make test-k8s
```

### Scenario 2: After Code Changes

```bash
# Test locally first
make docker-down
make docker-build-start
make test-backend

# If OK, deploy to Kubernetes
make build-push REGISTRY=your-registry TAG=v1.0.1
kubectl set image deployment/mailfind-backend backend=your-registry/mailfind-backend:v1.0.1 -n mailfind
kubectl rollout status deployment/mailfind-backend -n mailfind
make test-k8s
```

### Scenario 3: Production Health Check

```bash
# Quick health check
kubectl get pods -n mailfind
kubectl get hpa -n mailfind
make test-k8s-backend

# Full check
./scripts/test-k8s.sh

# Check resource usage
kubectl top pods -n mailfind
kubectl top nodes
```

---

## 🧪 Specific Test Cases

### Test 1: Backend API Endpoints

```bash
# Health check
curl http://localhost:8000/

# Auth status (will fail without credentials, but should return proper error)
curl http://localhost:8000/auth/status

# Check all endpoints return proper errors (not 500)
curl -I http://localhost:8000/sync-inbox
curl -I http://localhost:8000/chat
curl -I http://localhost:8000/analytics
```

### Test 2: Environment Variables

```bash
# Docker
docker-compose exec backend env | grep -E "SUPABASE|GOOGLE|OPENAI"

# Kubernetes
POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD -n mailfind -- env | grep -E "SUPABASE|GOOGLE|OPENAI"
```

### Test 3: Container/Pod Restart

```bash
# Docker
docker-compose restart backend
sleep 5
curl http://localhost:8000/

# Kubernetes
kubectl rollout restart deployment/mailfind-backend -n mailfind
kubectl rollout status deployment/mailfind-backend -n mailfind
make test-k8s-backend
```

### Test 4: Load Testing (Basic)

```bash
# Install hey if not available: go install github.com/rakyll/hey@latest

# Port forward backend
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind &

# Run load test
hey -n 1000 -c 10 http://localhost:8000/

# Check if HPA scaled
kubectl get hpa -n mailfind
kubectl get pods -n mailfind

# Kill port forward
pkill -f "port-forward"
```

### Test 5: Network Connectivity

```bash
# Kubernetes pod-to-pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n mailfind -- sh
# Inside pod:
curl http://mailfind-backend:8000/
curl http://mailfind-frontend:80/
exit

# External connectivity
POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD -n mailfind -- curl -I https://supabase.co
kubectl exec $POD -n mailfind -- curl -I https://api.openai.com
```

### Test 6: Logs and Debugging

```bash
# Docker - Show last 50 lines
docker-compose logs backend --tail=50

# Docker - Follow logs
docker-compose logs -f backend

# Kubernetes - Show last 50 lines
kubectl logs deployment/mailfind-backend -n mailfind --tail=50

# Kubernetes - Follow logs
kubectl logs -f deployment/mailfind-backend -n mailfind

# Kubernetes - Previous container (if crashed)
kubectl logs deployment/mailfind-backend -n mailfind --previous

# Search for errors
docker-compose logs backend | grep -i error
kubectl logs deployment/mailfind-backend -n mailfind | grep -i error
```

### Test 7: Resource Usage

```bash
# Docker
docker stats --no-stream

# Kubernetes
kubectl top pods -n mailfind
kubectl top nodes

# Compare with limits
kubectl describe deployment mailfind-backend -n mailfind | grep -A 5 "Limits"
```

### Test 8: Scaling

```bash
# Manual scale up
kubectl scale deployment mailfind-backend --replicas=5 -n mailfind

# Wait for pods
kubectl wait --for=condition=ready pod -l component=backend -n mailfind --timeout=120s

# Check
kubectl get pods -n mailfind

# Manual scale down
kubectl scale deployment mailfind-backend --replicas=2 -n mailfind
```

---

## ❌ Testing Failures and Fixes

### Issue: Docker containers won't start

```bash
# Check logs
docker-compose logs

# Check for port conflicts
lsof -i :8000
lsof -i :3000

# Clean start
docker-compose down -v
docker-compose up -d --build
```

### Issue: Backend returns errors

```bash
# Check environment variables
docker-compose exec backend env | grep SUPABASE

# Check if .env is loaded
cat .env | grep SUPABASE

# Restart with clean state
docker-compose down
docker-compose up -d
```

### Issue: Kubernetes pods in CrashLoopBackOff

```bash
# Describe pod
POD=$(kubectl get pods -n mailfind -l component=backend -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD -n mailfind

# Check logs
kubectl logs $POD -n mailfind

# Check secrets
kubectl get secret mailfind-secrets -n mailfind
kubectl describe secret mailfind-secrets -n mailfind

# Recreate secrets if needed
./scripts/create-k8s-secrets.sh
kubectl rollout restart deployment/mailfind-backend -n mailfind
```

### Issue: Service not accessible

```bash
# Check service exists
kubectl get svc -n mailfind

# Check endpoints
kubectl get endpoints -n mailfind

# Check if pods are ready
kubectl get pods -n mailfind

# Port forward to test directly
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind
curl http://localhost:8000/
```

### Issue: Images not pulling

```bash
# Check image name
kubectl describe deployment mailfind-backend -n mailfind | grep Image

# Verify image exists in registry
docker pull your-registry/mailfind-backend:latest

# Create image pull secret if using private registry
kubectl create secret docker-registry regcred \
  --docker-server=your-registry \
  --docker-username=your-username \
  --docker-password=your-password \
  -n mailfind
```

---

## 📊 Expected Test Results

### Successful Docker Test Output

```
✓ Docker installed
✓ Docker Compose installed
✓ Docker daemon running
✓ .env file exists
✓ docker-compose.yml exists
✓ Backend container running
✓ Frontend container running
✓ Backend health endpoint
✓ Frontend responds
✓ No errors in backend logs
✓ No errors in frontend logs

Test Summary
============
Passed:   12
Warnings: 0
Failed:   0

✓ All critical tests passed!
```

### Successful Kubernetes Test Output

```
✓ kubectl installed
✓ kubectl can connect
✓ Namespace exists
✓ Backend deployment exists
✓ Frontend deployment exists
✓ Backend pods are running
✓ Frontend pods are running
✓ Backend pod is ready
✓ Backend service is accessible
✓ HPA is configured

Test Summary
============
Passed:   10
Warnings: 0
Failed:   0

✓ All tests passed!
```

---

## 🎯 Daily Health Check

Quick daily check for production:

```bash
# 1-minute health check
kubectl get pods -n mailfind
kubectl get hpa -n mailfind
make test-k8s-backend

# 5-minute comprehensive check
./scripts/test-k8s.sh
kubectl top pods -n mailfind
kubectl get events -n mailfind --sort-by='.lastTimestamp' | tail -10
```

---

## 📚 See Also

- [TESTING.md](./TESTING.md) - Comprehensive testing guide with 50+ test cases
- [DOCKER_KUBERNETES_GUIDE.md](./DOCKER_KUBERNETES_GUIDE.md) - Full deployment guide
- [DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md) - Docker quick start

---

## 🆘 Getting Help

If tests are failing:

1. **Check logs first**: `docker-compose logs` or `kubectl logs`
2. **Verify environment**: Ensure all env vars are set
3. **Check external services**: Verify Supabase, OpenAI, Google APIs are accessible
4. **Clean restart**: `docker-compose down && docker-compose up -d`
5. **Review documentation**: Check TESTING.md for specific error scenarios

**Quick debug commands:**

```bash
# Docker
docker-compose ps
docker-compose logs backend --tail=100
make info

# Kubernetes  
kubectl get all -n mailfind
kubectl describe pod <pod-name> -n mailfind
kubectl get events -n mailfind --sort-by='.lastTimestamp'
```

