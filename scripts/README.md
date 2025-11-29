# Deployment Scripts

This directory contains helper scripts for deploying MailFind using Docker and Kubernetes.

## Available Scripts

### 1. `local-docker-start.sh`

Quick start script for local Docker Compose development.

**Usage:**
```bash
./scripts/local-docker-start.sh [OPTIONS]

Options:
  --build     Rebuild images before starting
  --attach    Run in foreground (show logs)
```

**Examples:**
```bash
# Start with existing images
./scripts/local-docker-start.sh

# Rebuild and start
./scripts/local-docker-start.sh --build

# Start and show logs
./scripts/local-docker-start.sh --attach
```

**What it does:**
- Checks if .env file exists
- Verifies Docker is running
- Starts services with docker-compose
- Displays access URLs and helpful commands

---

### 2. `build-images.sh`

Build Docker images for backend and frontend.

**Usage:**
```bash
./scripts/build-images.sh [OPTIONS]

Options:
  -r, --registry REGISTRY   Container registry (default: localhost)
  -t, --tag TAG            Image tag (default: latest)
  -p, --push               Push images to registry after building
  -h, --help               Display help message
```

**Examples:**
```bash
# Build locally
./scripts/build-images.sh

# Build and tag for Docker Hub
./scripts/build-images.sh --registry docker.io/myuser --tag v1.0.0

# Build, tag, and push to GCR
./scripts/build-images.sh --registry gcr.io/myproject --tag latest --push

# Using environment variables
REGISTRY=ghcr.io/myuser TAG=v2.0.0 ./scripts/build-images.sh --push
```

**What it does:**
- Builds backend Docker image
- Builds frontend Docker image
- Tags images with specified registry and tag
- Optionally pushes images to registry

---

### 3. `create-k8s-secrets.sh`

Create Kubernetes secrets from environment variables.

**Usage:**
```bash
./scripts/create-k8s-secrets.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE   Kubernetes namespace (default: mailfind)
  -f, --env-file FILE        Environment file (default: .env)
  -h, --help                 Display help message
```

**Examples:**
```bash
# Create secrets from .env file
./scripts/create-k8s-secrets.sh

# Use custom env file
./scripts/create-k8s-secrets.sh --env-file .env.production

# Create in different namespace
./scripts/create-k8s-secrets.sh --namespace mailfind-staging

# Interactive mode (will prompt for values)
./scripts/create-k8s-secrets.sh  # if .env doesn't exist
```

**What it does:**
- Loads environment variables from specified file
- Prompts for missing required values
- Creates Kubernetes secret with all credentials
- Handles secret replacement if it already exists

**Required environment variables:**
- SUPABASE_PUBLIC_URL
- SERVICE_ROLE
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- OPENAI_API_KEY
- ENCRYPTION_KEY

**Optional:**
- SUPABASE_ANON_KEY

---

### 4. `deploy-k8s.sh`

Deploy MailFind to Kubernetes cluster.

**Usage:**
```bash
./scripts/deploy-k8s.sh [OPTIONS]

Options:
  -n, --namespace NAMESPACE   Kubernetes namespace (default: mailfind)
  -a, --action ACTION         Action: apply or delete (default: apply)
  -k, --kustomize             Use kustomize (default: true)
  --no-kustomize              Don't use kustomize
  -h, --help                  Display help message
```

**Examples:**
```bash
# Deploy using kustomize
./scripts/deploy-k8s.sh

# Deploy without kustomize
./scripts/deploy-k8s.sh --no-kustomize

# Deploy to different namespace
./scripts/deploy-k8s.sh --namespace mailfind-staging

# Delete all resources
./scripts/deploy-k8s.sh --action delete

# Using environment variables
NAMESPACE=mailfind-dev ./scripts/deploy-k8s.sh
```

**What it does:**
- Verifies kubectl is installed and connected
- Deploys all Kubernetes manifests
- Shows deployment status
- Displays helpful commands for monitoring

**Prerequisites:**
- kubectl configured to access cluster
- Secrets created (use `create-k8s-secrets.sh`)
- Images pushed to registry (use `build-images.sh --push`)

---

## Typical Workflow

### Local Development with Docker

```bash
# 1. Create .env file
cp .env.example .env
# Edit .env with your values

# 2. Start services
./scripts/local-docker-start.sh

# 3. View logs
docker-compose logs -f

# 4. Stop services
docker-compose down
```

### Production Deployment to Kubernetes

```bash
# 1. Build and push images
./scripts/build-images.sh \
  --registry gcr.io/my-project \
  --tag v1.0.0 \
  --push

# 2. Update k8s/kustomization.yaml with your registry and tag

# 3. Create secrets
./scripts/create-k8s-secrets.sh \
  --namespace mailfind

# 4. Deploy to cluster
./scripts/deploy-k8s.sh \
  --namespace mailfind

# 5. Verify deployment
kubectl get pods -n mailfind
kubectl logs -f deployment/mailfind-backend -n mailfind

# 6. Check ingress
kubectl get ingress -n mailfind
```

### Update Deployment

```bash
# 1. Build new images
./scripts/build-images.sh \
  --registry gcr.io/my-project \
  --tag v1.1.0 \
  --push

# 2. Update deployment
kubectl set image deployment/mailfind-backend \
  backend=gcr.io/my-project/mailfind-backend:v1.1.0 \
  -n mailfind

kubectl set image deployment/mailfind-frontend \
  frontend=gcr.io/my-project/mailfind-frontend:v1.1.0 \
  -n mailfind

# 3. Monitor rollout
kubectl rollout status deployment/mailfind-backend -n mailfind
kubectl rollout status deployment/mailfind-frontend -n mailfind
```

---

## Environment Variables

All scripts support environment variables as an alternative to command-line flags:

| Script | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| All | - | - | See individual scripts |
| build-images.sh | REGISTRY | localhost | Container registry |
| build-images.sh | TAG | latest | Image tag |
| build-images.sh | PUSH | false | Push to registry |
| create-k8s-secrets.sh | NAMESPACE | mailfind | K8s namespace |
| create-k8s-secrets.sh | ENV_FILE | .env | Environment file |
| deploy-k8s.sh | NAMESPACE | mailfind | K8s namespace |
| deploy-k8s.sh | ACTION | apply | K8s action |
| deploy-k8s.sh | USE_KUSTOMIZE | true | Use kustomize |

---

## Troubleshooting

### Script Permission Denied

```bash
chmod +x scripts/*.sh
```

### Docker Not Running

```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

### kubectl Not Found

Install kubectl:
```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### Cannot Connect to Cluster

```bash
# Verify cluster access
kubectl cluster-info

# Check context
kubectl config current-context

# Switch context
kubectl config use-context my-cluster
```

### Image Push Failed

```bash
# Login to registry
docker login <registry>

# For GCR
gcloud auth configure-docker

# For ECR
aws ecr get-login-password --region region | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# For GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

---

## CI/CD Integration

These scripts can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
- name: Build and Push Images
  run: |
    ./scripts/build-images.sh \
      --registry ghcr.io/${{ github.repository_owner }} \
      --tag ${{ github.sha }} \
      --push

- name: Deploy to Kubernetes
  run: |
    ./scripts/deploy-k8s.sh --namespace production
```

### GitLab CI Example

```yaml
deploy:
  script:
    - ./scripts/build-images.sh --registry $CI_REGISTRY_IMAGE --tag $CI_COMMIT_SHA --push
    - ./scripts/deploy-k8s.sh --namespace production
```

---

## Additional Resources

- [DOCKER_QUICKSTART.md](../DOCKER_QUICKSTART.md) - Quick start with Docker
- [DOCKER_KUBERNETES_GUIDE.md](../DOCKER_KUBERNETES_GUIDE.md) - Comprehensive K8s guide
- [README.md](../README.md) - Main project documentation

