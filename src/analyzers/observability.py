"""
Observability Analyzer for KubeAnalyzer

This module analyzes Kubernetes services for observability coverage,
including metrics endpoints, logging, and tracing configurations.
"""

import logging
from typing import Dict, List, Optional

from ..utils.k8s_client import KubernetesClient
from ..utils.prometheus import PrometheusClient

logger = logging.getLogger(__name__)


class ObservabilityAnalyzer:
    """Analyzes Kubernetes services for observability coverage."""

    def __init__(self, k8s_client: KubernetesClient, prometheus_client: Optional[PrometheusClient] = None):
        """
        Initialize the ObservabilityAnalyzer.
        
        Args:
            k8s_client: Kubernetes client instance
            prometheus_client: Optional Prometheus client for metrics discovery
        """
        self.k8s_client = k8s_client
        self.prometheus_client = prometheus_client
        
    def analyze_service(self, namespace: str, service_name: str) -> Dict:
        """
        Analyze a specific service for observability coverage.
        
        Args:
            namespace: Kubernetes namespace
            service_name: Service name
            
        Returns:
            Dict with analysis results
        """
        logger.info(f"Analyzing observability for service {service_name} in namespace {namespace}")
        
        results = {
            "service": service_name,
            "namespace": namespace,
            "metrics": self._check_metrics(namespace, service_name),
            "logging": self._check_logging(namespace, service_name),
            "tracing": self._check_tracing(namespace, service_name),
            "alerts": self._check_alerts(namespace, service_name),
            "dashboard": self._check_dashboard(namespace, service_name),
        }
        
        # Calculate overall score
        coverage = self._calculate_coverage(results)
        results["coverage_score"] = coverage
        results["coverage_level"] = self._get_coverage_level(coverage)
        
        return results
    
    def analyze(self, namespaces=None) -> Dict:
        """
        Analyze observability coverage across specified namespaces.
        
        Args:
            namespaces: List of namespaces to analyze. If None, all namespaces will be analyzed.
            
        Returns:
            Dict with analysis results including coverage metrics and recommendations
        """
        logger.info(f"Analyzing observability coverage for namespaces: {namespaces if namespaces else 'all'}")        
        
        # Analyze each namespace separately if specified, otherwise analyze all
        results = []
        if namespaces:
            for namespace in namespaces:
                namespace_results = self.analyze_all_services(namespace)
                results.extend(namespace_results)
        else:
            results = self.analyze_all_services()
        
        # Calculate overall statistics
        coverage_scores = [r.get("coverage_score", 0) for r in results]
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        
        # Generate summary
        summary = {
            "services_analyzed": len(results),
            "namespaces_analyzed": len(set(r.get("namespace") for r in results)) if results else 0,
            "average_coverage_score": round(avg_coverage, 2),
            "coverage_level": self._get_coverage_level(avg_coverage),
            "service_results": results
        }
        
        return summary
        
    def analyze_all_services(self, namespace: Optional[str] = None) -> List[Dict]:
        """
        Analyze all services in the specified namespace or across all namespaces.
        
        Args:
            namespace: Optional namespace to restrict analysis to
            
        Returns:
            List of service analysis results
        """
        if namespace:
            services = self.k8s_client.list_services(namespace)
            namespaces = [namespace]
        else:
            namespaces = self.k8s_client.list_namespaces()
            services = []
            for ns in namespaces:
                services.extend([(ns, svc) for svc in self.k8s_client.list_services(ns)])
        
        results = []
        for ns, service in services:
            results.append(self.analyze_service(ns, service))
        
        return results
    
    def _check_metrics(self, namespace: str, service_name: str) -> Dict:
        """Check if service has Prometheus metrics endpoint."""
        has_metrics = False
        metrics_path = "/metrics"  # Default path
        
        # Check if service has prometheus.io/scrape annotation
        service = self.k8s_client.get_service(namespace, service_name)
        annotations = service.get("metadata", {}).get("annotations", {})
        
        if annotations.get("prometheus.io/scrape") == "true":
            has_metrics = True
            if "prometheus.io/path" in annotations:
                metrics_path = annotations["prometheus.io/path"]
        
        # If Prometheus client is available, check for actual metrics
        metrics_count = 0
        if self.prometheus_client and has_metrics:
            try:
                metrics = self.prometheus_client.get_metrics_for_service(namespace, service_name)
                metrics_count = len(metrics)
            except Exception as e:
                logger.warning(f"Error checking metrics for {service_name}: {str(e)}")
        
        return {
            "has_metrics_endpoint": has_metrics,
            "metrics_path": metrics_path,
            "metrics_count": metrics_count,
            "score": 1.0 if has_metrics else 0.0
        }
    
    def _check_logging(self, namespace: str, service_name: str) -> Dict:
        """Check if service has proper logging configured."""
        # This would check for log configuration in the pod spec
        # For now, using a simplified approach
        pods = self.k8s_client.list_pods_for_service(namespace, service_name)
        
        has_logging = False
        has_structured_logging = False
        has_log_level = False
        
        for pod in pods:
            containers = pod.get("spec", {}).get("containers", [])
            for container in containers:
                env = container.get("env", [])
                for env_var in env:
                    if env_var.get("name") == "LOG_LEVEL":
                        has_log_level = True
                    if env_var.get("name") == "LOG_FORMAT" and env_var.get("value") == "json":
                        has_structured_logging = True
            
            # Check for volume mounts that might indicate logging
            for container in containers:
                volume_mounts = container.get("volumeMounts", [])
                for mount in volume_mounts:
                    if "log" in mount.get("name", "").lower():
                        has_logging = True
        
        # Simple scoring
        score = 0.0
        if has_logging:
            score += 0.4
        if has_structured_logging:
            score += 0.4
        if has_log_level:
            score += 0.2
        
        return {
            "has_logging": has_logging or has_structured_logging or has_log_level,
            "has_structured_logging": has_structured_logging,
            "has_log_level": has_log_level,
            "score": score
        }
    
    def _check_tracing(self, namespace: str, service_name: str) -> Dict:
        """Check if service has distributed tracing configured."""
        # Check for common tracing environment variables or annotations
        pods = self.k8s_client.list_pods_for_service(namespace, service_name)
        
        has_tracing = False
        tracing_type = None
        
        for pod in pods:
            containers = pod.get("spec", {}).get("containers", [])
            for container in containers:
                env = container.get("env", [])
                for env_var in env:
                    name = env_var.get("name", "").lower()
                    if "jaeger" in name or "zipkin" in name or "opentracing" in name or "opentelemetry" in name:
                        has_tracing = True
                        if "jaeger" in name:
                            tracing_type = "jaeger"
                        elif "zipkin" in name:
                            tracing_type = "zipkin"
                        elif "opentelemetry" in name:
                            tracing_type = "opentelemetry"
                        elif "opentracing" in name:
                            tracing_type = "opentracing"
        
        return {
            "has_tracing": has_tracing,
            "tracing_type": tracing_type,
            "score": 1.0 if has_tracing else 0.0
        }
    
    def _check_alerts(self, namespace: str, service_name: str) -> Dict:
        """Check if service has alerts configured."""
        # This would ideally query Prometheus for PrometheusRules targeting this service
        # For now, provide a placeholder
        has_alerts = False
        alerts_count = 0
        
        if self.prometheus_client:
            try:
                alerts = self.prometheus_client.get_alerts_for_service(namespace, service_name)
                has_alerts = len(alerts) > 0
                alerts_count = len(alerts)
            except Exception as e:
                logger.warning(f"Error checking alerts for {service_name}: {str(e)}")
        
        return {
            "has_alerts": has_alerts,
            "alerts_count": alerts_count,
            "score": 1.0 if has_alerts else 0.0
        }
    
    def _check_dashboard(self, namespace: str, service_name: str) -> Dict:
        """Check if service has a dashboard configured."""
        # This would ideally query Grafana API for dashboards related to this service
        # For now, provide a placeholder implementation
        return {
            "has_dashboard": False,
            "dashboard_url": None,
            "score": 0.0
        }
    
    def _calculate_coverage(self, results: Dict) -> float:
        """Calculate overall observability coverage score."""
        # Define weights for each component
        weights = {
            "metrics": 0.3,
            "logging": 0.3,
            "tracing": 0.2,
            "alerts": 0.15,
            "dashboard": 0.05
        }
        
        weighted_score = 0.0
        for component, weight in weights.items():
            weighted_score += results[component]["score"] * weight
        
        return round(weighted_score, 2)
    
    def _get_coverage_level(self, score: float) -> str:
        """Convert numerical score to coverage level."""
        if score >= 0.8:
            return "complete"
        elif score >= 0.5:
            return "partial"
        elif score > 0:
            return "minimal"
        else:
            return "none"
