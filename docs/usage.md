# KubeAnalyzer Usage Guide

This document explains how to use KubeAnalyzer to analyze your Kubernetes cluster and interpret the results.

## Getting Started

After [installing KubeAnalyzer](installation.md), you can access it through its web interface or API.

### Accessing the Web Interface

```bash
# Port-forward the KubeAnalyzer service to your local machine
kubectl port-forward svc/kube-analyzer-service 8080:8080
```

Then open your browser and navigate to http://localhost:8080

### Using the API

```bash
# Get the API endpoint
API_ENDPOINT=$(kubectl get svc kube-analyzer-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Fetch the latest analysis report
curl http://$API_ENDPOINT:8080/api/v1/reports/latest
```

## Running an Analysis

### On-Demand Analysis

From the web interface:
1. Navigate to the "Analysis" tab
2. Select the types of analysis to run (observability, vulnerability, cluster health)
3. Click "Start Analysis"

Using the API:
```bash
# Start a new analysis
curl -X POST http://$API_ENDPOINT:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"types": ["observability", "vulnerability", "cluster-health"]}'
```

### Scheduled Analysis

KubeAnalyzer runs scheduled analyses based on its configuration. By default, it runs a full analysis every hour. You can modify the schedule in the configuration settings.

## Understanding Analysis Reports

### Report Structure

Each KubeAnalyzer report includes:

1. **Summary**: High-level overview of findings with severity distribution
2. **Observability Coverage**:
   - Services lacking monitoring
   - Missing key metrics
   - Logging gaps
3. **Vulnerability Assessment**:
   - Containers with vulnerabilities
   - Severity classification (Critical, High, Medium, Low)
   - CVE references and remediation steps
4. **Cluster Health**:
   - Resource utilization issues
   - Configuration problems
   - Performance bottlenecks
5. **Recommendations**:
   - AI-generated suggestions for improvement
   - Best practices applicable to your environment
   - Prioritized action items

### Interpreting Results

#### Observability Coverage

KubeAnalyzer checks for:
- Prometheus metrics endpoints
- Logging configuration
- Tracing instrumentation
- Alert rules

Services are classified as:
- **Complete**: All observability components present
- **Partial**: Some components missing
- **Minimal**: Only basic monitoring
- **None**: No observability components detected

#### Vulnerability Analysis

Vulnerabilities are reported with:
- CVE identifier
- Affected package
- Current version
- Fixed version (if available)
- CVSS score and vector
- Description and impact
- Remediation steps

#### Cluster Health

Health issues are categorized as:
- **Resource Issues**: CPU, memory, or storage problems
- **Network Issues**: Communication or DNS problems
- **Control Plane Issues**: API server, etcd, or scheduler problems
- **Configuration Issues**: Suboptimal or risky configurations

### Sample Report

Here's what a sample finding might look like:

```json
{
  "finding_id": "OBS-2023-07-12-001",
  "type": "observability",
  "severity": "medium",
  "service": "payment-processor",
  "namespace": "ecommerce",
  "description": "Service lacks Prometheus metrics endpoint",
  "details": "The payment-processor deployment does not expose metrics that can be scraped by Prometheus, resulting in limited visibility into its performance and health.",
  "impact": "Difficult to monitor service performance and detect issues proactively.",
  "recommendation": "Add a Prometheus metrics endpoint by implementing the /metrics HTTP endpoint and exposing relevant business and technical metrics.",
  "remediation_example": {
    "code": "https://github.com/your-org/kube-analyzer/examples/prometheus-metrics.py",
    "docs": "https://prometheus.io/docs/instrumenting/writing_exporters/"
  }
}
```

## Working with AI Insights

KubeAnalyzer's AI agent provides context-aware insights based on:
- Historical data from your cluster
- Industry best practices
- Common patterns and anti-patterns
- Vulnerability databases

### Training the AI

The AI improves its recommendations over time as it:
1. Observes your environment and workload patterns
2. Learns from your feedback on recommendations
3. Correlates issues across different analysis dimensions

### Providing Feedback

You can improve the AI's recommendations by providing feedback:
1. In the web interface, use the thumbs up/down buttons on recommendations
2. Add comments to explain why a recommendation was helpful or not
3. Mark false positives to help the system learn

## Exporting and Integrating

### Export Formats

Reports can be exported in multiple formats:
- JSON (for programmatic processing)
- YAML (for configuration management)
- HTML (for human-readable reports)
- PDF (for formal documentation)

### Integration Options

KubeAnalyzer can integrate with:
- **Slack/Teams**: Send notifications for critical findings
- **Jira/GitHub**: Create tickets for issues that need remediation
- **Grafana**: Visualize trends and status dashboards
- **CI/CD Pipelines**: Include analysis in deployment validation

To configure integrations, see the [Integration Guide](integrations.md).
