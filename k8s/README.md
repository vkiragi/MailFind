# Kubernetes Manifests

This directory contains Kubernetes manifests for deploying MailFind to a Kubernetes cluster.

## Files Overview

| File | Description |
|------|-------------|
| `namespace.yaml` | Creates the `mailfind` namespace |
| `configmap.yaml` | Application configuration (non-sensitive) |
| `secrets.yaml.template` | Template for sensitive credentials (DO NOT COMMIT actual secrets) |
| `backend-deployment.yaml` | Backend API deployment |
| `backend-service.yaml` | Backend service (ClusterIP) |
| `frontend-deployment.yaml` | Frontend deployment |
| `frontend-service.yaml` | Frontend service (ClusterIP) |
| `ingress.yaml` | Ingress for external access |
| `hpa.yaml` | Horizontal Pod Autoscalers for auto-scaling |
| `network-policy.yaml` | Network policies for security |
| `persistent-volume.yaml` | PVC for logs (optional) |
| `kustomization.yaml` | Kustomize configuration |

## Quick Start

### Prerequisites

1. Kubernetes cluster (minikube, kind, GKE, EKS, AKS, etc.)
2. kubectl configured to access cluster
3. Docker images built and pushed to a registry
4. Environment variables ready

### Deploy

**Method 1: Using deployment script (recommended)**

```bash
# Create secrets
../scripts/create-k8s-secrets.sh

# Deploy all resources
../scripts/deploy-k8s.sh
```

**Method 2: Using kubectl directly**

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Create secrets (see instructions below)
kubectl create secret generic mailfind-secrets \
  --namespace=mailfind \
  --from-literal=SUPABASE_PUBLIC_URL='your-url' \
  --from-literal=SERVICE_ROLE='your-key' \
  --from-literal=GOOGLE_CLIENT_ID='your-id' \
  --from-literal=GOOGLE_CLIENT_SECRET='your-secret' \
  --from-literal=OPENAI_API_KEY='your-key' \
  --from-literal=ENCRYPTION_KEY='your-key'

# 3. Apply all manifests
kubectl apply -f configmap.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f frontend-service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
kubectl apply -f network-policy.yaml
```

**Method 3: Using Kustomize**

```bash
# Update kustomization.yaml with your registry/tags
# Then apply
kubectl apply -k .
```

## Configuration

### 1. Update Image References

Edit `kustomization.yaml`:

```yaml
images:
- name: mailfind-backend
  newName: your-registry/mailfind-backend
  newTag: v1.0.0
- name: mailfind-frontend
  newName: your-registry/mailfind-frontend
  newTag: v1.0.0
```

Or update directly in deployment files:

```yaml
# backend-deployment.yaml
spec:
  containers:
  - name: backend
    image: your-registry/mailfind-backend:v1.0.0
```

### 2. Configure Ingress

Edit `ingress.yaml`:

```yaml
spec:
  rules:
  - host: mailfind.yourdomain.com      # Update this
  - host: api.mailfind.yourdomain.com  # Update this
```

### 3. Adjust Resource Limits

Edit deployment files:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### 4. Configure Autoscaling

Edit `hpa.yaml`:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

## Secrets Management

### Option 1: kubectl create secret (Recommended)

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

### Option 2: YAML file

```bash
# Copy template
cp secrets.yaml.template secrets.yaml

# Base64 encode values
echo -n "your-value" | base64

# Edit secrets.yaml with base64-encoded values
# Apply
kubectl apply -f secrets.yaml
```

**IMPORTANT: Never commit `secrets.yaml` to git!**

### Option 3: External Secrets Manager

For production, use:
- AWS Secrets Manager + External Secrets Operator
- HashiCorp Vault
- Google Secret Manager
- Azure Key Vault

## Monitoring

### Check Deployment Status

```bash
# Pods
kubectl get pods -n mailfind

# Deployments
kubectl get deployments -n mailfind

# Services
kubectl get svc -n mailfind

# Ingress
kubectl get ingress -n mailfind

# HPA
kubectl get hpa -n mailfind

# All resources
kubectl get all -n mailfind
```

### View Logs

```bash
# Backend logs
kubectl logs -f deployment/mailfind-backend -n mailfind

# Frontend logs
kubectl logs -f deployment/mailfind-frontend -n mailfind

# Specific pod
kubectl logs -f <pod-name> -n mailfind

# Previous container logs
kubectl logs --previous <pod-name> -n mailfind
```

### Describe Resources

```bash
# Describe pod
kubectl describe pod <pod-name> -n mailfind

# Describe deployment
kubectl describe deployment mailfind-backend -n mailfind

# Events
kubectl get events -n mailfind --sort-by='.lastTimestamp'
```

## Testing

### Port Forwarding

```bash
# Backend
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind

# Frontend
kubectl port-forward svc/mailfind-frontend 3000:80 -n mailfind

# Access at http://localhost:8000 and http://localhost:3000
```

### Execute Commands in Pod

```bash
# Shell into backend pod
kubectl exec -it deployment/mailfind-backend -n mailfind -- /bin/bash

# Run command
kubectl exec deployment/mailfind-backend -n mailfind -- env

# Check connectivity to Supabase
kubectl exec deployment/mailfind-backend -n mailfind -- \
  curl https://your-project.supabase.co
```

## Updating Deployment

### Update Images

```bash
# Update backend
kubectl set image deployment/mailfind-backend \
  backend=your-registry/mailfind-backend:v2.0.0 \
  -n mailfind

# Update frontend
kubectl set image deployment/mailfind-frontend \
  frontend=your-registry/mailfind-frontend:v2.0.0 \
  -n mailfind

# Monitor rollout
kubectl rollout status deployment/mailfind-backend -n mailfind
```

### Update ConfigMap or Secrets

```bash
# Apply changes
kubectl apply -f configmap.yaml

# Restart deployments to pick up changes
kubectl rollout restart deployment/mailfind-backend -n mailfind
kubectl rollout restart deployment/mailfind-frontend -n mailfind
```

### Rollback

```bash
# View history
kubectl rollout history deployment/mailfind-backend -n mailfind

# Rollback to previous version
kubectl rollout undo deployment/mailfind-backend -n mailfind

# Rollback to specific revision
kubectl rollout undo deployment/mailfind-backend --to-revision=2 -n mailfind
```

## Scaling

### Manual Scaling

```bash
# Scale backend
kubectl scale deployment mailfind-backend --replicas=5 -n mailfind

# Scale frontend
kubectl scale deployment mailfind-frontend --replicas=3 -n mailfind
```

### Auto-scaling

HPA is configured in `hpa.yaml`. Monitor with:

```bash
kubectl get hpa -n mailfind
kubectl describe hpa mailfind-backend-hpa -n mailfind
```

## Cleanup

### Delete All Resources

```bash
# Using script
../scripts/deploy-k8s.sh --action delete

# Or manually
kubectl delete -f .

# Delete namespace (removes everything)
kubectl delete namespace mailfind
```

## Security Best Practices

1. **Secrets**: Use external secrets manager in production
2. **Network Policies**: Configured in `network-policy.yaml`
3. **RBAC**: Consider creating service accounts with limited permissions
4. **Pod Security**: Deployments run as non-root users
5. **TLS**: Configure TLS certificates for ingress
6. **Image Scanning**: Scan images for vulnerabilities before deployment
7. **Resource Limits**: Set appropriate limits to prevent resource exhaustion

## Troubleshooting

See [DOCKER_KUBERNETES_GUIDE.md](../DOCKER_KUBERNETES_GUIDE.md#troubleshooting) for detailed troubleshooting guide.

### Common Issues

**ImagePullBackOff**
```bash
# Check image name and tag
kubectl describe pod <pod-name> -n mailfind

# Create image pull secret for private registries
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n mailfind
```

**CrashLoopBackOff**
```bash
# Check logs
kubectl logs <pod-name> -n mailfind

# Check events
kubectl get events -n mailfind --sort-by='.lastTimestamp'
```

**Service Not Accessible**
```bash
# Check endpoints
kubectl get endpoints -n mailfind

# Check service
kubectl describe svc mailfind-backend -n mailfind

# Port forward to test
kubectl port-forward svc/mailfind-backend 8000:8000 -n mailfind
```

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kustomize Documentation](https://kustomize.io/)
- [DOCKER_KUBERNETES_GUIDE.md](../DOCKER_KUBERNETES_GUIDE.md)

