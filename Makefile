# KubeAnalyzer Makefile

.PHONY: help build test lint clean run deploy

# Docker image info
IMAGE_NAME := kubeanalyzer/kube-analyzer
IMAGE_TAG := 0.1.0

# Helm chart info
CHART_PATH := ./deploy/kube-analyzer
RELEASE_NAME := kube-analyzer
NAMESPACE := monitoring

# Default target
help:
	@echo "KubeAnalyzer Make Commands:"
	@echo "  make build       - Build Docker image"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run linting"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make run         - Run local test analysis"
	@echo "  make deploy      - Deploy to Kubernetes using Helm"
	@echo "  make undeploy    - Remove from Kubernetes"

# Build the Docker image
build:
	@echo "Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Run the tests
test:
	@echo "Running tests"
	python -m unittest discover -s tests

# Run linting
lint:
	@echo "Running linting"
	flake8 src/ tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts"
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf reports/
	rm -rf .pytest_cache/

# Run local analysis
run:
	@echo "Running local analysis"
	python scripts/run_local.py --analysis-type cluster-health

# Deploy to Kubernetes
deploy:
	@echo "Deploying to Kubernetes with Helm"
	helm upgrade --install $(RELEASE_NAME) $(CHART_PATH) \
		--namespace $(NAMESPACE) \
		--create-namespace

# Undeploy from Kubernetes
undeploy:
	@echo "Removing from Kubernetes"
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE)

# Show logs from deployed application
logs:
	@echo "Showing logs from deployed application"
	kubectl logs -n $(NAMESPACE) -l app.kubernetes.io/name=kube-analyzer -f

# Port-forward to access the application locally
port-forward:
	@echo "Setting up port-forwarding to access the application locally on port 8080"
	kubectl port-forward -n $(NAMESPACE) svc/$(RELEASE_NAME) 8080:8080
