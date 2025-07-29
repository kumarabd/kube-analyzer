#!/usr/bin/env python3
"""
Local test script for KubeAnalyzer.

This script allows for testing the KubeAnalyzer functionality
without deploying to a cluster, using local kubeconfig
and mocked services when necessary.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.analyzers.cluster_health import ClusterHealthAnalyzer
from src.analyzers.observability import ObservabilityAnalyzer
from src.analyzers.vulnerability import VulnerabilityAnalyzer
from src.utils.k8s_client import KubernetesClient
from src.utils.prometheus import PrometheusClient
from src.utils.reporting import Report, ReportFormatter


def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KubeAnalyzer Local Test Script"
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus server URL (default: http://localhost:9090 or PROMETHEUS_URL env var)"
    )
    parser.add_argument(
        "--analysis-type",
        choices=["all", "observability", "vulnerability", "cluster-health"],
        default="all",
        help="Type of analysis to run (default: all)"
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        help="Namespaces to analyze (default: all)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "yaml", "markdown", "html"],
        default="markdown",
        help="Output format for reports (default: markdown)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "reports"),
        help="Directory to save reports (default: ./reports)"
    )
    parser.add_argument(
        "--mock-prometheus",
        action="store_true",
        help="Use mocked Prometheus data when Prometheus is unavailable"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


class MockPrometheusClient:
    """Mock Prometheus client for local testing."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock client."""
        logging.info("Using mock Prometheus client")
    
    def check_connection(self) -> bool:
        """Mock successful connection."""
        return True
    
    def get_metrics_for_service(self, namespace: str, service_name: str):
        """Return mock metrics for a service."""
        return [
            {"name": "http_requests_total", "labels": {"service": service_name}},
            {"name": "http_request_duration_seconds", "labels": {"service": service_name}},
            {"name": "http_request_size_bytes", "labels": {"service": service_name}},
            {"name": "http_response_size_bytes", "labels": {"service": service_name}}
        ]
    
    def get_alerts_for_service(self, namespace: str, service_name: str):
        """Return mock alerts for a service."""
        return [
            {
                "name": "HighErrorRate",
                "annotations": {
                    "namespace": namespace,
                    "service": service_name,
                    "summary": "High error rate detected",
                    "description": "Service is returning a high number of 5xx errors"
                },
                "state": "inactive"
            }
        ]
    
    def get_resource_usage(self, namespace=None):
        """Return mock resource usage metrics."""
        return {
            "cpu": {
                "total": 1.5,  # 1.5 cores used
                "details": []
            },
            "memory": {
                "total": 2 * 1024 * 1024 * 1024,  # 2GB used
                "details": []
            }
        }
    
    def query(self, query, time=None):
        """Mock query execution."""
        return {"resultType": "vector", "result": []}
    
    def query_range(self, query, start, end, step):
        """Mock range query execution."""
        return {"resultType": "matrix", "result": []}


def main():
    """Main function for local testing."""
    args = parse_args()
    setup_logging(args.debug)
    
    logging.info("Starting KubeAnalyzer local test run")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create K8s client using local config
    try:
        k8s_client = KubernetesClient(in_cluster=False)
        logging.info("Successfully created Kubernetes client using local config")
    except Exception as e:
        logging.error(f"Failed to create Kubernetes client: {str(e)}")
        sys.exit(1)
    
    # Create Prometheus client (real or mock)
    if args.mock_prometheus:
        prometheus_client = MockPrometheusClient()
    else:
        prometheus_client = PrometheusClient(base_url=args.prometheus_url)
        if not prometheus_client.check_connection():
            logging.warning("Cannot connect to Prometheus, using mock client instead")
            prometheus_client = MockPrometheusClient()
    
    # Initialize analyzers
    observability_analyzer = ObservabilityAnalyzer(k8s_client, prometheus_client)
    vulnerability_analyzer = VulnerabilityAnalyzer(k8s_client)
    cluster_health_analyzer = ClusterHealthAnalyzer(k8s_client)
    
    start_time = datetime.now()
    
    # Create a report
    report = Report(name="KubeAnalyzer Local Test Report")
    
    # Determine namespaces to analyze
    namespaces = args.namespaces
    if not namespaces:
        try:
            namespaces = k8s_client.list_namespaces()
            logging.info(f"Found {len(namespaces)} namespaces in the cluster")
        except Exception as e:
            logging.error(f"Failed to list namespaces: {str(e)}")
            namespaces = ["default"]
            logging.info("Falling back to 'default' namespace")
    
    # Run analyses based on type
    if args.analysis_type in ["all", "cluster-health"]:
        logging.info("Running cluster health analysis")
        try:
            cluster_health_results = cluster_health_analyzer.analyze()
            report.add_section("cluster_health", cluster_health_results)
            logging.info("Cluster health analysis completed")
        except Exception as e:
            logging.error(f"Failed to run cluster health analysis: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    if args.analysis_type in ["all", "vulnerability"]:
        logging.info("Running vulnerability analysis")
        try:
            vulnerability_results = vulnerability_analyzer.analyze(namespaces)
            report.add_section("vulnerability", vulnerability_results)
            logging.info("Vulnerability analysis completed")
        except Exception as e:
            logging.error(f"Failed to run vulnerability analysis: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    if args.analysis_type in ["all", "observability"]:
        logging.info("Running observability analysis")
        try:
            observability_results = observability_analyzer.analyze(namespaces)
            report.add_section("observability", observability_results)
            logging.info("Observability analysis completed")
        except Exception as e:
            logging.error(f"Failed to run observability analysis: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Add summary to report
    summary = {
        "analyzed_namespaces": len(namespaces),
        "analysis_type": args.analysis_type,
        "analysis_duration_seconds": duration
    }
    report.add_summary(summary)
    
    # Print the report to the console
    try:
        print("\n" + "=" * 50)
        print(f"KubeAnalyzer Report ({args.output_format.upper()})")
        print("=" * 50)
        
        if args.output_format == "json":
            print(report.to_json(pretty=True))
        elif args.output_format == "yaml":
            print(report.to_yaml())
        elif args.output_format == "markdown":
            print(ReportFormatter.format_markdown(report.get_report()))
        elif args.output_format == "html":
            print(ReportFormatter.format_html(report.get_report()))
        else:
            logging.error(f"Unsupported output format: {args.output_format}")
            
        print("\n" + "=" * 50)
    except Exception as e:
        logging.error(f"Failed to generate report: {str(e)}")
    
    # Print summary
    print("\nKubeAnalyzer Local Test Summary:")
    print(f"- Analysis Type: {args.analysis_type}")
    print(f"- Analyzed Namespaces: {len(namespaces)}")
    print(f"- Namespaces: {', '.join(namespaces)}")
    print(f"- Analysis Duration: {duration:.2f} seconds")
    
    logging.info("KubeAnalyzer local test completed")


if __name__ == "__main__":
    main()
