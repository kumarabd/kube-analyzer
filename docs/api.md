# KubeAnalyzer API Documentation

## Overview

KubeAnalyzer provides a comprehensive REST API that allows you to programmatically interact with the system, trigger analyses, retrieve reports, and integrate with other tools and services.

## API Basics

- **Base URL**: `http://<kube-analyzer-service>:8080/api/v1`
- **Authentication**: API key authentication using the `X-API-Key` header
- **Response Format**: JSON is the default format for all responses
- **Status Codes**: Standard HTTP status codes are used to indicate success/failure

## Authentication

To use the API, you need to obtain an API key:

```bash
# Generate an API key
kubectl exec deploy/kube-analyzer -- kube-analyzer-cli generate-api-key

# Use the key in requests
curl -H "X-API-Key: your-api-key" http://<kube-analyzer-service>:8080/api/v1/status
```

## Endpoints

### System Status

#### Get System Status

```
GET /status
```

Returns the current status of the KubeAnalyzer system.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.2.3",
  "uptime": "3d 4h 12m",
  "last_analysis": "2023-07-26T15:30:45Z",
  "next_scheduled_analysis": "2023-07-26T16:30:45Z",
  "components": {
    "analyzer": "running",
    "database": "healthy",
    "ai_engine": "ready"
  }
}
```

### Analysis Management

#### List Available Analyzers

```
GET /analyzers
```

Returns a list of available analyzers and their status.

**Response**:
```json
{
  "analyzers": [
    {
      "id": "observability",
      "name": "Observability Coverage Analyzer",
      "description": "Analyzes services for proper monitoring, logging, and tracing",
      "enabled": true
    },
    {
      "id": "vulnerability",
      "name": "Container Vulnerability Scanner",
      "description": "Scans containers for known vulnerabilities",
      "enabled": true
    },
    {
      "id": "cluster-health",
      "name": "Cluster Health Analyzer",
      "description": "Detects potential issues in the cluster",
      "enabled": true
    }
  ]
}
```

#### Start Analysis

```
POST /analyze
```

Triggers a new analysis run.

**Request Body**:
```json
{
  "types": ["observability", "vulnerability", "cluster-health"],
  "namespaces": ["default", "kube-system"],
  "report_format": "json",
  "notify": true
}
```

**Response**:
```json
{
  "analysis_id": "a1b2c3d4",
  "status": "started",
  "estimated_completion": "2023-07-26T15:45:30Z"
}
```

#### Get Analysis Status

```
GET /analyze/{analysis_id}
```

Checks the status of a running analysis.

**Response**:
```json
{
  "analysis_id": "a1b2c3d4",
  "status": "in-progress",
  "progress": 65,
  "started_at": "2023-07-26T15:30:45Z",
  "estimated_completion": "2023-07-26T15:45:30Z"
}
```

### Reports

#### List Reports

```
GET /reports
```

Lists all available reports.

**Parameters**:
- `limit`: Maximum number of reports to return (default: 10)
- `offset`: Pagination offset (default: 0)
- `type`: Filter by report type (e.g., "observability")

**Response**:
```json
{
  "total": 45,
  "reports": [
    {
      "id": "rep123",
      "analysis_id": "a1b2c3d4",
      "created_at": "2023-07-26T15:45:30Z",
      "type": "full",
      "summary": {
        "critical_issues": 2,
        "high_issues": 5,
        "medium_issues": 12,
        "low_issues": 20
      }
    },
    {
      "id": "rep122",
      "analysis_id": "e5f6g7h8",
      "created_at": "2023-07-25T15:45:30Z",
      "type": "full",
      "summary": {
        "critical_issues": 3,
        "high_issues": 7,
        "medium_issues": 14,
        "low_issues": 22
      }
    }
  ]
}
```

#### Get Report

```
GET /reports/{report_id}
```

Retrieves a specific report.

**Parameters**:
- `format`: Response format (json, yaml, html, pdf) (default: json)

**Response**:
```json
{
  "id": "rep123",
  "analysis_id": "a1b2c3d4",
  "created_at": "2023-07-26T15:45:30Z",
  "type": "full",
  "summary": {
    "critical_issues": 2,
    "high_issues": 5,
    "medium_issues": 12,
    "low_issues": 20
  },
  "findings": [
    {
      "id": "finding1",
      "type": "vulnerability",
      "severity": "critical",
      "service": "payment-service",
      "container": "payment-processor",
      "image": "payment-processor:1.2.3",
      "vulnerability": {
        "cve": "CVE-2023-12345",
        "description": "Remote code execution vulnerability in library X",
        "affected_package": "openssl",
        "affected_version": "1.1.1k",
        "fixed_version": "1.1.1l",
        "cvss_score": 9.8,
        "link": "https://nvd.nist.gov/vuln/detail/CVE-2023-12345"
      },
      "recommendation": "Update openssl to version 1.1.1l or later"
    }
  ]
}
```

#### Get Latest Report

```
GET /reports/latest
```

Retrieves the most recent report.

### AI Recommendations

#### Get AI Insights

```
GET /ai/insights
```

Get AI-generated insights about your cluster.

**Response**:
```json
{
  "insights": [
    {
      "id": "ins1",
      "category": "performance",
      "title": "Resource allocation optimization opportunities",
      "description": "Several deployments have consistently low CPU utilization while requesting high resources",
      "affected_resources": [
        "deployment/auth-service",
        "deployment/notification-service"
      ],
      "recommendation": "Consider reducing CPU requests to improve cluster efficiency",
      "potential_impact": "Medium",
      "confidence": 0.85
    }
  ]
}
```

#### Submit Feedback on Recommendation

```
POST /ai/feedback
```

Submit feedback on an AI recommendation.

**Request Body**:
```json
{
  "recommendation_id": "rec123",
  "helpful": true,
  "applied": true,
  "comment": "This recommendation helped us optimize our cluster resources"
}
```

### Configuration

#### Get Current Configuration

```
GET /config
```

Retrieves the current KubeAnalyzer configuration.

#### Update Configuration

```
PUT /config
```

Updates KubeAnalyzer configuration.

**Request Body**:
```json
{
  "scan_interval": 3600,
  "report_retention_days": 30,
  "notifications": {
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/..."
    }
  },
  "ai": {
    "enabled": true,
    "focus_areas": {
      "security": "high",
      "performance": "medium"
    }
  }
}
```

## Webhooks

KubeAnalyzer can send webhook notifications when analyses complete or issues are detected.

### Register Webhook

```
POST /webhooks
```

Registers a new webhook endpoint.

**Request Body**:
```json
{
  "url": "https://your-service.example.com/webhook",
  "events": ["analysis.complete", "issue.critical", "issue.high"],
  "secret": "your-webhook-secret"
}
```

## Error Handling

All API errors follow a standard format:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid report format specified",
    "details": "Supported formats are: json, yaml, html, pdf"
  }
}
```

Common error codes:
- `unauthorized`: Authentication failed
- `forbidden`: Insufficient permissions
- `not_found`: Resource not found
- `invalid_request`: Invalid request parameters
- `internal_error`: Server-side error

## Rate Limiting

API requests are rate limited to protect the service. Current limits are:

- 60 requests per minute for standard operations
- 10 requests per minute for resource-intensive operations (like starting analyses)

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1627312768
```

## API Versioning

The API version is included in the URL path (e.g., `/api/v1/`). When breaking changes are introduced, a new version will be made available, and older versions will be supported for a deprecation period.

## Client Libraries

Official client libraries are available for:
- Python: `pip install kube-analyzer-client`
- Go: `go get github.com/kube-analyzer/client-go`
- JavaScript: `npm install kube-analyzer-client`

Example Python usage:

```python
from kube_analyzer import Client

# Initialize client
client = Client(api_key="your-api-key", base_url="http://kube-analyzer-service:8080/api/v1")

# Start an analysis
analysis = client.start_analysis(types=["vulnerability", "cluster-health"])
print(f"Analysis started with ID: {analysis['analysis_id']}")

# Get latest report
latest_report = client.get_latest_report()
print(f"Found {latest_report['summary']['critical_issues']} critical issues")
```
