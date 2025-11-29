#!/bin/bash
# Quick Kubernetes deployment test script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="${NAMESPACE:-mailfind}"

echo -e "${BLUE}========================================"
echo "MailFind Kubernetes Test"
echo "Namespace: $NAMESPACE"
echo -e "========================================${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

test_command() {
    local name="$1"
    local cmd="$2"
    
    echo -n "Testing: $name... "
    
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Prerequisites
echo -e "${YELLOW}Phase 1: Prerequisites${NC}"
test_command "kubectl installed" "kubectl version --client"
test_command "kubectl can connect" "kubectl cluster-info"
test_command "k8s directory exists" "test -d k8s"
echo ""

# Validate manifests
echo -e "${YELLOW}Phase 2: Validating Manifests${NC}"
test_command "Namespace manifest valid" "kubectl apply --dry-run=client -f k8s/namespace.yaml"
test_command "ConfigMap manifest valid" "kubectl apply --dry-run=client -f k8s/configmap.yaml"
test_command "Backend deployment valid" "kubectl apply --dry-run=client -f k8s/backend-deployment.yaml"
test_command "Frontend deployment valid" "kubectl apply --dry-run=client -f k8s/frontend-deployment.yaml"
test_command "Backend service valid" "kubectl apply --dry-run=client -f k8s/backend-service.yaml"
test_command "Frontend service valid" "kubectl apply --dry-run=client -f k8s/frontend-service.yaml"
echo ""

# Check if namespace exists
echo -e "${YELLOW}Phase 3: Checking Deployment${NC}"
if kubectl get namespace "$NAMESPACE" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Namespace exists: $NAMESPACE"
    PASSED=$((PASSED + 1))
    
    # Check resources
    test_command "ConfigMap exists" "kubectl get configmap mailfind-config -n $NAMESPACE"
    test_command "Secrets exist" "kubectl get secret mailfind-secrets -n $NAMESPACE"
    test_command "Backend deployment exists" "kubectl get deployment mailfind-backend -n $NAMESPACE"
    test_command "Frontend deployment exists" "kubectl get deployment mailfind-frontend -n $NAMESPACE"
    test_command "Backend service exists" "kubectl get svc mailfind-backend -n $NAMESPACE"
    test_command "Frontend service exists" "kubectl get svc mailfind-frontend -n $NAMESPACE"
    echo ""
    
    # Check pod status
    echo -e "${YELLOW}Phase 4: Checking Pods${NC}"
    
    BACKEND_PODS=$(kubectl get pods -n $NAMESPACE -l component=backend --no-headers 2>/dev/null | wc -l)
    FRONTEND_PODS=$(kubectl get pods -n $NAMESPACE -l component=frontend --no-headers 2>/dev/null | wc -l)
    
    echo "Backend pods:  $BACKEND_PODS"
    echo "Frontend pods: $FRONTEND_PODS"
    
    if [ "$BACKEND_PODS" -gt 0 ]; then
        RUNNING_BACKEND=$(kubectl get pods -n $NAMESPACE -l component=backend --no-headers 2>/dev/null | grep Running | wc -l)
        echo "  Running: $RUNNING_BACKEND"
        
        if [ "$RUNNING_BACKEND" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Backend pods are running"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗${NC} Backend pods not running"
            FAILED=$((FAILED + 1))
        fi
    fi
    
    if [ "$FRONTEND_PODS" -gt 0 ]; then
        RUNNING_FRONTEND=$(kubectl get pods -n $NAMESPACE -l component=frontend --no-headers 2>/dev/null | grep Running | wc -l)
        echo "  Running: $RUNNING_FRONTEND"
        
        if [ "$RUNNING_FRONTEND" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Frontend pods are running"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗${NC} Frontend pods not running"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
    
    # Check pod health
    echo -e "${YELLOW}Phase 5: Checking Pod Health${NC}"
    
    if [ "$BACKEND_PODS" -gt 0 ]; then
        BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l component=backend -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$BACKEND_POD" ]; then
            READY=$(kubectl get pod "$BACKEND_POD" -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
            if [ "$READY" = "True" ]; then
                echo -e "${GREEN}✓${NC} Backend pod is ready"
                PASSED=$((PASSED + 1))
            else
                echo -e "${YELLOW}⚠${NC} Backend pod not ready"
                WARNINGS=$((WARNINGS + 1))
            fi
            
            # Check restarts
            RESTARTS=$(kubectl get pod "$BACKEND_POD" -n $NAMESPACE -o jsonpath='{.status.containerStatuses[0].restartCount}')
            if [ "$RESTARTS" -eq 0 ]; then
                echo -e "${GREEN}✓${NC} No restarts"
                PASSED=$((PASSED + 1))
            else
                echo -e "${YELLOW}⚠${NC} Pod has restarted $RESTARTS times"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    fi
    echo ""
    
    # Test connectivity
    echo -e "${YELLOW}Phase 6: Testing Connectivity${NC}"
    
    if [ "$BACKEND_PODS" -gt 0 ] && [ "$RUNNING_BACKEND" -gt 0 ]; then
        echo "Testing backend service..."
        if kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -n $NAMESPACE -- \
            curl -f -s http://mailfind-backend:8000/ > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Backend service is accessible"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗${NC} Backend service not accessible"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
    
    # Check HPA
    echo -e "${YELLOW}Phase 7: Checking Autoscaling${NC}"
    if kubectl get hpa -n $NAMESPACE > /dev/null 2>&1; then
        HPA_COUNT=$(kubectl get hpa -n $NAMESPACE --no-headers | wc -l)
        echo "HPAs configured: $HPA_COUNT"
        
        if [ "$HPA_COUNT" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} HPA is configured"
            PASSED=$((PASSED + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} No HPA found (may not be deployed yet)"
        WARNINGS=$((WARNINGS + 1))
    fi
    echo ""
    
    # Show recent events
    echo -e "${YELLOW}Phase 8: Recent Events${NC}"
    echo "Last 5 events:"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -n 5
    echo ""
    
else
    echo -e "${YELLOW}⚠${NC} Namespace does not exist: $NAMESPACE"
    echo "   Deploy first with: ./scripts/deploy-k8s.sh"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo -e "${BLUE}========================================"
echo "Test Summary"
echo -e "========================================${NC}"
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
    else
        echo -e "${YELLOW}⚠ Tests passed with warnings${NC}"
    fi
    echo ""
    echo "Useful commands:"
    echo "  View pods:    kubectl get pods -n $NAMESPACE"
    echo "  View logs:    kubectl logs -f deployment/mailfind-backend -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward svc/mailfind-backend 8000:8000 -n $NAMESPACE"
    echo "  Delete:       ./scripts/deploy-k8s.sh --namespace $NAMESPACE --action delete"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  Check pods:   kubectl get pods -n $NAMESPACE"
    echo "  Describe pod: kubectl describe pod <pod-name> -n $NAMESPACE"
    echo "  View logs:    kubectl logs <pod-name> -n $NAMESPACE"
    echo "  Events:       kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
    exit 1
fi

