# Makefile for MailFind - Docker and Kubernetes deployment

.PHONY: help build start stop clean docker-build docker-push k8s-deploy k8s-delete k8s-secrets

# Variables
REGISTRY ?= localhost
TAG ?= latest
NAMESPACE ?= mailfind

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[1;33m
NC     := \033[0m # No Color

help: ## Show this help message
	@echo '$(GREEN)MailFind Deployment Commands$(NC)'
	@echo '=============================='
	@echo ''
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ''
	@echo 'Variables:'
	@echo '  REGISTRY     Container registry (default: localhost)'
	@echo '  TAG          Image tag (default: latest)'
	@echo '  NAMESPACE    Kubernetes namespace (default: mailfind)'
	@echo ''
	@echo 'Examples:'
	@echo '  make docker-start'
	@echo '  make docker-build REGISTRY=docker.io/myuser TAG=v1.0.0'
	@echo '  make k8s-deploy NAMESPACE=production'

# Docker Compose Commands
docker-start: ## Start services with Docker Compose
	@echo "$(GREEN)Starting services with Docker Compose...$(NC)"
	./scripts/local-docker-start.sh

docker-build-start: ## Build and start services with Docker Compose
	@echo "$(GREEN)Building and starting services...$(NC)"
	./scripts/local-docker-start.sh --build

docker-stop: ## Stop Docker Compose services
	@echo "$(YELLOW)Stopping Docker Compose services...$(NC)"
	docker-compose stop

docker-down: ## Stop and remove Docker Compose services
	@echo "$(YELLOW)Stopping and removing Docker Compose services...$(NC)"
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

docker-ps: ## Show running containers
	docker-compose ps

# Docker Build Commands
build: ## Build Docker images locally
	@echo "$(GREEN)Building Docker images...$(NC)"
	./scripts/build-images.sh --registry $(REGISTRY) --tag $(TAG)

build-push: ## Build and push Docker images to registry
	@echo "$(GREEN)Building and pushing Docker images...$(NC)"
	./scripts/build-images.sh --registry $(REGISTRY) --tag $(TAG) --push

docker-build-backend: ## Build backend Docker image only
	@echo "$(GREEN)Building backend image...$(NC)"
	cd packages/backend && docker build -t $(REGISTRY)/mailfind-backend:$(TAG) .

docker-build-frontend: ## Build frontend Docker image only
	@echo "$(GREEN)Building frontend image...$(NC)"
	cd packages/chrome-extension && docker build -t $(REGISTRY)/mailfind-frontend:$(TAG) .

# Kubernetes Commands
k8s-secrets: ## Create Kubernetes secrets
	@echo "$(GREEN)Creating Kubernetes secrets...$(NC)"
	./scripts/create-k8s-secrets.sh --namespace $(NAMESPACE)

k8s-deploy: ## Deploy to Kubernetes
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	./scripts/deploy-k8s.sh --namespace $(NAMESPACE)

k8s-delete: ## Delete Kubernetes resources
	@echo "$(YELLOW)Deleting Kubernetes resources...$(NC)"
	./scripts/deploy-k8s.sh --namespace $(NAMESPACE) --action delete

k8s-status: ## Show Kubernetes deployment status
	@echo "$(GREEN)Kubernetes Status$(NC)"
	@echo "=================="
	@echo ""
	@echo "$(YELLOW)Pods:$(NC)"
	@kubectl get pods -n $(NAMESPACE)
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@kubectl get svc -n $(NAMESPACE)
	@echo ""
	@echo "$(YELLOW)Ingress:$(NC)"
	@kubectl get ingress -n $(NAMESPACE)
	@echo ""
	@echo "$(YELLOW)HPA:$(NC)"
	@kubectl get hpa -n $(NAMESPACE)

k8s-logs-backend: ## View backend logs in Kubernetes
	kubectl logs -f deployment/mailfind-backend -n $(NAMESPACE)

k8s-logs-frontend: ## View frontend logs in Kubernetes
	kubectl logs -f deployment/mailfind-frontend -n $(NAMESPACE)

k8s-restart-backend: ## Restart backend deployment
	kubectl rollout restart deployment/mailfind-backend -n $(NAMESPACE)

k8s-restart-frontend: ## Restart frontend deployment
	kubectl rollout restart deployment/mailfind-frontend -n $(NAMESPACE)

k8s-scale-backend: ## Scale backend (use: make k8s-scale-backend REPLICAS=5)
	@echo "$(GREEN)Scaling backend to $(REPLICAS) replicas...$(NC)"
	kubectl scale deployment mailfind-backend --replicas=$(REPLICAS) -n $(NAMESPACE)

k8s-scale-frontend: ## Scale frontend (use: make k8s-scale-frontend REPLICAS=3)
	@echo "$(GREEN)Scaling frontend to $(REPLICAS) replicas...$(NC)"
	kubectl scale deployment mailfind-frontend --replicas=$(REPLICAS) -n $(NAMESPACE)

k8s-port-forward-backend: ## Port forward backend service
	@echo "$(GREEN)Port forwarding backend to localhost:8000$(NC)"
	kubectl port-forward svc/mailfind-backend 8000:8000 -n $(NAMESPACE)

k8s-port-forward-frontend: ## Port forward frontend service
	@echo "$(GREEN)Port forwarding frontend to localhost:3000$(NC)"
	kubectl port-forward svc/mailfind-frontend 3000:80 -n $(NAMESPACE)

# Complete Deployment Workflows
deploy-local: ## Complete local deployment with Docker Compose
	@echo "$(GREEN)Complete Local Deployment$(NC)"
	@echo "========================="
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env from .env.example...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "$(YELLOW)Please create .env file manually$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Starting services...$(NC)"
	@$(MAKE) docker-start
	@echo ""
	@echo "$(GREEN)✓ Deployment complete!$(NC)"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"

deploy-k8s: ## Complete Kubernetes deployment
	@echo "$(GREEN)Complete Kubernetes Deployment$(NC)"
	@echo "=============================="
	@echo ""
	@echo "$(YELLOW)Step 1: Building images...$(NC)"
	@$(MAKE) build-push
	@echo ""
	@echo "$(YELLOW)Step 2: Creating secrets...$(NC)"
	@$(MAKE) k8s-secrets
	@echo ""
	@echo "$(YELLOW)Step 3: Deploying to cluster...$(NC)"
	@$(MAKE) k8s-deploy
	@echo ""
	@echo "$(GREEN)✓ Deployment complete!$(NC)"
	@$(MAKE) k8s-status

# Cleanup Commands
clean: ## Clean up all Docker resources
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker-compose down -v
	docker system prune -f

clean-all: ## Clean up all Docker resources including images
	@echo "$(YELLOW)Cleaning up all Docker resources...$(NC)"
	docker-compose down -v --rmi all
	docker system prune -af --volumes

# Testing Commands
test-docker: ## Run comprehensive Docker tests
	@echo "$(GREEN)Running Docker Compose tests...$(NC)"
	./scripts/test-docker.sh

test-k8s: ## Run comprehensive Kubernetes tests
	@echo "$(GREEN)Running Kubernetes tests...$(NC)"
	./scripts/test-k8s.sh

test-backend: ## Test backend health (Docker)
	@echo "$(GREEN)Testing backend health...$(NC)"
	@curl -f http://localhost:8000/ && echo "$(GREEN)✓ Backend is healthy$(NC)" || echo "$(YELLOW)✗ Backend is not responding$(NC)"

test-frontend: ## Test frontend health (Docker)
	@echo "$(GREEN)Testing frontend health...$(NC)"
	@curl -f -I http://localhost:3000/ && echo "$(GREEN)✓ Frontend is healthy$(NC)" || echo "$(YELLOW)✗ Frontend is not responding$(NC)"

test-k8s-backend: ## Test backend in Kubernetes
	@echo "$(GREEN)Testing backend in Kubernetes...$(NC)"
	@kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n $(NAMESPACE) -- \
		curl -f http://mailfind-backend:8000/ && \
		echo "$(GREEN)✓ Backend is healthy$(NC)" || \
		echo "$(YELLOW)✗ Backend is not responding$(NC)"

test-all: test-docker test-k8s ## Run all tests (Docker and Kubernetes)

# Development Commands
dev-backend: ## Run backend in development mode
	@echo "$(GREEN)Starting backend in development mode...$(NC)"
	cd packages/backend && \
		. .venv/bin/activate 2>/dev/null || python3 -m venv .venv && . .venv/bin/activate && \
		pip install -r requirements.txt && \
		python start_server.py

dev-frontend: ## Run frontend in development mode
	@echo "$(GREEN)Starting frontend in development mode...$(NC)"
	cd packages/chrome-extension && \
		npm install && \
		npm run dev

# Information Commands
info: ## Show deployment information
	@echo "$(GREEN)MailFind Deployment Information$(NC)"
	@echo "==============================="
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  Registry:  $(REGISTRY)"
	@echo "  Tag:       $(TAG)"
	@echo "  Namespace: $(NAMESPACE)"
	@echo ""
	@echo "$(YELLOW)Docker Images:$(NC)"
	@echo "  Backend:   $(REGISTRY)/mailfind-backend:$(TAG)"
	@echo "  Frontend:  $(REGISTRY)/mailfind-frontend:$(TAG)"
	@echo ""
	@echo "$(YELLOW)Local Services:$(NC)"
	@echo "  Backend:   http://localhost:8000"
	@echo "  Frontend:  http://localhost:3000"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  README.md                    - Main documentation"
	@echo "  DOCKER_QUICKSTART.md         - Docker quick start"
	@echo "  DOCKER_KUBERNETES_GUIDE.md   - Kubernetes guide"
	@echo "  DEPLOYMENT_SUMMARY.md        - Deployment summary"

version: ## Show versions
	@echo "$(GREEN)Version Information$(NC)"
	@echo "==================="
	@echo "Docker:      $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Compose:     $$(docker-compose --version 2>/dev/null || echo 'Not installed')"
	@echo "kubectl:     $$(kubectl version --client --short 2>/dev/null || echo 'Not installed')"
	@echo "Python:      $$(python3 --version 2>/dev/null || echo 'Not installed')"
	@echo "Node:        $$(node --version 2>/dev/null || echo 'Not installed')"

check-deps: ## Check if required dependencies are installed
	@echo "$(GREEN)Checking Dependencies$(NC)"
	@echo "===================="
	@which docker >/dev/null 2>&1 && echo "$(GREEN)✓ Docker$(NC)" || echo "$(YELLOW)✗ Docker not found$(NC)"
	@which docker-compose >/dev/null 2>&1 && echo "$(GREEN)✓ Docker Compose$(NC)" || echo "$(YELLOW)✗ Docker Compose not found$(NC)"
	@which kubectl >/dev/null 2>&1 && echo "$(GREEN)✓ kubectl$(NC)" || echo "$(YELLOW)✗ kubectl not found$(NC)"
	@which python3 >/dev/null 2>&1 && echo "$(GREEN)✓ Python 3$(NC)" || echo "$(YELLOW)✗ Python 3 not found$(NC)"
	@which node >/dev/null 2>&1 && echo "$(GREEN)✓ Node.js$(NC)" || echo "$(YELLOW)✗ Node.js not found$(NC)"
	@[ -f .env ] && echo "$(GREEN)✓ .env file exists$(NC)" || echo "$(YELLOW)✗ .env file not found (copy from .env.example)$(NC)"

# Default target
.DEFAULT_GOAL := help

