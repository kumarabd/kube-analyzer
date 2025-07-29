# KubeAnalyzer Installation Guide

This document provides instructions for installing and configuring KubeAnalyzer in your Kubernetes cluster.

## Prerequisites

Before installing KubeAnalyzer, ensure you have the following:

- Kubernetes cluster (version 1.20+)
- Helm v3 installed on your local machine
- kubectl configured to communicate with your cluster
- Appropriate RBAC permissions to create resources in your cluster

## Installation Methods

### Option 1: Helm Chart (Recommended)

KubeAnalyzer is packaged as a Helm chart for easy deployment and management.

```bash
# Add the KubeAnalyzer Helm repository
helm repo add kube-analyzer https://kube-analyzer.github.io/charts
helm repo update

# Install KubeAnalyzer with default configuration
helm install kube-analyzer kube-analyzer/kube-analyzer

# Or install with custom values
helm install kube-analyzer kube-analyzer/kube-analyzer -f values.yaml
```

#### Customizing the Installation

Create a `values.yaml` file to override default settings:

```yaml
# Example custom values.yaml
resources:
  requests:
    cpu: "200m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"

config:
  scanSchedule: "0 */6 * * *"  # Run every 6 hours
  reportRetention: 30          # Store reports for 30 days
  
monitoring:
  enabled: true                # Enable built-in Prometheus metrics
  
persistence:
  enabled: true                # Enable persistent storage for reports
  size: 10Gi                   # Storage size
```

### Option 2: Manual Deployment

For manual deployment, you can build the Docker image and apply the Kubernetes manifests directly.

1. Build the Docker image:
   ```bash
   docker build -t your-registry/kube-analyzer:latest .
   docker push your-registry/kube-analyzer:latest
   ```

2. Modify the deployment files in the `deploy/kube-analyzer/templates` directory.

3. Apply the manifests:
   ```bash
   kubectl apply -f deploy/kube-analyzer/templates/
   ```

## Post-Installation Verification

Verify that KubeAnalyzer is running correctly:

```bash
# Check that pods are running
kubectl get pods -l app=kube-analyzer

# Access the web interface
kubectl port-forward svc/kube-analyzer-service 8080:8080
```

Then visit http://localhost:8080 in your browser.

## Configuration

### Environment Variables

KubeAnalyzer can be configured using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity (debug, info, warning, error) | `info` |
| `SCAN_INTERVAL` | Interval between scans in seconds | `3600` |
| `REPORT_FORMAT` | Default report format (json, yaml, html) | `html` |
| `ENABLE_AI` | Enable AI-powered analysis | `true` |

### Custom Rules

You can provide custom analysis rules by mounting a ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kube-analyzer-rules
data:
  custom-rules.yaml: |
    observability:
      required_metrics:
        - http_request_duration_seconds
        - memory_usage
    vulnerability:
      critical_cves:
        - CVE-2021-44228
        - CVE-2022-0778
```

Then update your Helm values to mount this ConfigMap:

```yaml
extraVolumes:
  - name: custom-rules
    configMap:
      name: kube-analyzer-rules

extraVolumeMounts:
  - name: custom-rules
    mountPath: /app/config/rules
    readOnly: true
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Ensure the ServiceAccount has appropriate RBAC permissions
   - Check logs for specific permission issues: `kubectl logs deploy/kube-analyzer`

2. **Pod Stuck in Pending State**
   - Check resource constraints: `kubectl describe pod kube-analyzer`
   - Ensure PVCs are bound if persistence is enabled

3. **Analysis Not Running**
   - Verify the application is properly configured
   - Check logs for errors: `kubectl logs deploy/kube-analyzer`

### Getting Support

If you encounter issues that you cannot resolve, please:
1. Check the [GitHub Issues](https://github.com/your-org/kube-analyzer/issues)
2. Join our [Slack community](https://slack.kube-analyzer.io)
3. File a bug report with detailed information about your environment
