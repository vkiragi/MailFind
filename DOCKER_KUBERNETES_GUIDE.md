# Docker and Kubernetes Deployment Guide

This guide covers deploying MailFind using Docker and Kubernetes.

## Table of Contents

1. [Docker Setup](#docker-setup)
2. [Docker Compose (Local Development)](#docker-compose-local-development)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Production Considerations](#production-considerations)
5. [Troubleshooting](#troubleshooting)

---

## Docker Setup

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- For Kubernetes: kubectl and a cluster (minikube, kind, or cloud provider)

### Building Images

#### Backend

```bash
cd packages/backend
docker build -t mailfind-backend:latest .
```

#### Frontend (Chrome Extension)

```bash
cd packages/chrome-extension
docker build -t mailfind-frontend:latest .
```

### Running Individual Containers

#### Backend

```bash
docker run -d \
  --name mailfind-backend \
  -p 8000:8000 \
  -e SUPABASE_PUBLIC_URL=your-url \
  -e SERVICE_ROLE=your-key \
  -e GOOGLE_CLIENT_ID=your-client-id \
  -e GOOGLE_CLIENT_SECRET=your-secret \
  -e OPENAI_API_KEY=your-api-key \
  -e ENCRYPTION_KEY=your-fernet-key \
  mailfind-backend:latest
```

#### Frontend

```bash
docker run -d \
  --name mailfind-frontend \
  -p 3000:80 \
  mailfind-frontend:latest
```

---

## Docker Compose (Local Development)

### Setup

1. **Create environment file**:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
SUPABASE_PUBLIC_URL=https://your-project.supabase.co
SERVICE_ROLE=your-service-role-key
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
OPENAI_API_KEY=sk-your-openai-key
ENCRYPTION_KEY=your-fernet-key
```

2. **Start all services**:

```bash
docker-compose up -d
```

3. **View logs**:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

4. **Stop services**:

```bash
docker-compose down
```

5. **Rebuild and restart**:

```bash
docker-compose up -d --build
```

### Accessing Services

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- Backend Health Check: http://localhost:8000/

---

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes cluster** (choose one):
   - Local: minikube, kind, Docker Desktop
   - Cloud: GKE, EKS, AKS

2. **kubectl** configured to access your cluster

3. **Container registry** (choose one):
   - Docker Hub
   - Google Container Registry (GCR)
   - Amazon ECR
   - Azure ACR
   - GitHub Container Registry (GHCR)

### Step 1: Push Images to Registry

#### Tag images

```bash
# Replace 'your-registry' with your actual registry
docker tag mailfind-backend:latest your-registry/mailfind-backend:latest
docker tag mailfind-frontend:latest your-registry/mailfind-frontend:latest
```

#### Push to registry

```bash
docker push your-registry/mailfind-backend:latest
docker push your-registry/mailfind-frontend:latest
```

### Step 2: Create Kubernetes Secrets

**Method 1: Using kubectl (Recommended)**

```bash
kubectl create secret generic mailfind-secrets \
  --namespace=mailfind \
  --from-literal=SUPABASE_PUBLIC_URL='https://your-project.supabase.co' \
  --from-literal=SERVICE_ROLE='your-service-role-key' \
  --from-literal=SUPABASE_ANON_KEY='your-anon-key' \
  --from-literal=GOOGLE_CLIENT_ID='your-client-id' \
  --from-literal=GOOGLE_CLIENT_SECRET='your-client-secret' \
  --from-literal=OPENAI_API_KEY='sk-your-key' \
  --from-literal=ENCRYPTION_KEY='your-fernet-key'
```

**Method 2: Using YAML file**

```bash
# Copy template and edit
cp k8s/secrets.yaml.template k8s/secrets.yaml

# Base64 encode each value
echo -n "https://your-project.supabase.co" | base64

# Edit k8s/secrets.yaml with encoded values
# Apply
kubectl apply -f k8s/secrets.yaml
```

### Step 3: Update Image References

Edit `k8s/kustomization.yaml`:

```yaml
images:
- name: mailfind-backend
  newName: your-registry/mailfind-backend
  newTag: latest
- name: mailfind-frontend
  newName: your-registry/mailfind-frontend
  newTag: latest
```

### Step 4: Deploy to Kubernetes

#### Using kubectl

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply all manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml  # if using YAML method
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/network-policy.yaml
```

#### Using Kustomize (Recommended)

```bash
kubectl apply -k k8s/
```

### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n mailfind

# Check services
kubectl get svc -n mailfind

# Check ingress
kubectl get ingress -n mailfind

# View logs
kubectl logs -f deployment/mailfind-backend -n mailfind
kubectl logs -f deployment/mailfind-frontend -n mailfind
```

### Step 6: Configure Ingress (Production)

1. **Update domain names** in `k8s/ingress.yaml`:

```yaml
spec:
  rules:
  - host: mailfind.yourdomain.com    # Update this
  - host: api.mailfind.yourdomain.com  # Update this
```

2. **Set up TLS/SSL** (if using cert-manager):

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f k8s/cert-issuer.yaml  # You may need to create this
```

3. **Apply updated ingress**:

```bash
kubectl apply -f k8s/ingress.yaml
```

---

## Production Considerations

### Security

1. **Use Kubernetes Secrets** properly:
   - Never commit secrets to git
   - Use external secret managers (AWS Secrets Manager, HashiCorp Vault)
   - Rotate secrets regularly

2. **Network Policies**: Already configured in `k8s/network-policy.yaml`

3. **Pod Security**:
   - Run as non-root user (already configured)
   - Use read-only root filesystems where possible
   - Enable security contexts

### Scaling

1. **Horizontal Pod Autoscaler**: Already configured in `k8s/hpa.yaml`

2. **Resource Limits**: Configured in deployment files
   - Backend: 512Mi-2Gi memory, 250m-1000m CPU
   - Frontend: 64Mi-128Mi memory, 50m-200m CPU

3. **Vertical Pod Autoscaler** (optional):

```bash
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vertical-pod-autoscaler.yaml
```

### Monitoring

1. **Install Prometheus + Grafana**:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

2. **Add ServiceMonitor** for your app (create `k8s/service-monitor.yaml`):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mailfind-backend
  namespace: mailfind
spec:
  selector:
    matchLabels:
      app: mailfind
      component: backend
  endpoints:
  - port: http
    path: /metrics
```

### Logging

1. **Install EFK/ELK stack** or use cloud provider logging

2. **Log aggregation** options:
   - Fluentd
   - Fluent Bit
   - Loki

### Backup and Disaster Recovery

1. **Backup Supabase data** regularly
2. **Version control all Kubernetes manifests**
3. **Use GitOps** (ArgoCD, Flux) for deployment automation

### CI/CD Integration

Example GitHub Actions workflow (`.github/workflows/deploy.yml`):

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Backend
        run: |
          docker build -t ${{ secrets.REGISTRY }}/mailfind-backend:${{ github.sha }} packages/backend
          docker push ${{ secrets.REGISTRY }}/mailfind-backend:${{ github.sha }}
      
      - name: Build Frontend
        run: |
          docker build -t ${{ secrets.REGISTRY }}/mailfind-frontend:${{ github.sha }} packages/chrome-extension
          docker push ${{ secrets.REGISTRY }}/mailfind-frontend:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/mailfind-backend mailfind-backend=${{ secrets.REGISTRY }}/mailfind-backend:${{ github.sha }} -n mailfind
          kubectl set image deployment/mailfind-frontend mailfind-frontend=${{ secrets.REGISTRY }}/mailfind-frontend:${{ github.sha }} -n mailfind
```

---

## Troubleshooting

### Docker Issues

**Container won't start:**

```bash
# Check logs
docker logs mailfind-backend

# Inspect container
docker inspect mailfind-backend

# Access container shell
docker exec -it mailfind-backend /bin/bash
```

**Port already in use:**

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Kubernetes Issues

**Pods in CrashLoopBackOff:**

```bash
# Check pod status
kubectl describe pod <pod-name> -n mailfind

# View logs
kubectl logs <pod-name> -n mailfind

# Check events
kubectl get events -n mailfind --sort-by='.lastTimestamp'
```

**ImagePullBackOff:**

```bash
# Check if image exists in registry
# Verify image pull secrets if using private registry
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email> \
  -n mailfind

# Update deployment to use secret
```

**Service not accessible:**

```bash
# Check service
kubectl get svc -n mailfind

# Check endpoints
kubectl get endpoints -n mailfind

# Port forward for testing
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind
```

**Resource quota exceeded:**

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n mailfind

# Adjust resource requests/limits in deployments
```

### Database Connection Issues

1. **Check network policies** allow egress to Supabase
2. **Verify SUPABASE_PUBLIC_URL** is correct
3. **Test connection** from pod:

```bash
kubectl exec -it <backend-pod> -n mailfind -- curl https://your-project.supabase.co
```

### Performance Issues

1. **Check HPA status**:

```bash
kubectl get hpa -n mailfind
```

2. **Monitor resource usage**:

```bash
kubectl top pods -n mailfind
```

3. **Scale manually** if needed:

```bash
kubectl scale deployment mailfind-backend --replicas=5 -n mailfind
```

---

## Useful Commands

### Docker

```bash
# Remove all containers
docker rm -f $(docker ps -aq)

# Remove all images
docker rmi -f $(docker images -q)

# Clean up system
docker system prune -a --volumes

# View resource usage
docker stats
```

### Kubernetes

```bash
# Get all resources
kubectl get all -n mailfind

# Delete all resources
kubectl delete all --all -n mailfind

# Restart deployment
kubectl rollout restart deployment/mailfind-backend -n mailfind

# View deployment history
kubectl rollout history deployment/mailfind-backend -n mailfind

# Rollback deployment
kubectl rollout undo deployment/mailfind-backend -n mailfind

# Shell into pod
kubectl exec -it <pod-name> -n mailfind -- /bin/bash

# Copy files from pod
kubectl cp mailfind/<pod-name>:/app/logs ./logs

# Apply changes
kubectl apply -f k8s/

# Delete resources
kubectl delete -f k8s/
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kustomize Documentation](https://kustomize.io/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

---

## Support

For issues specific to MailFind deployment, please check:
1. Main README.md
2. Backend logs: `kubectl logs -f deployment/mailfind-backend -n mailfind`
3. Environment variables are correctly set
4. All external services (Supabase, OpenAI, Google OAuth) are accessible

