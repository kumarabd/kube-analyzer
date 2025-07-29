#!/usr/bin/env python3
"""
KubeAnalyzer - Kubernetes Cluster Analyzer AI Agent

This module is the main entry point for the KubeAnalyzer application.
It orchestrates the various analyzers to provide a comprehensive analysis
of the Kubernetes cluster.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from analyzers.cluster_health import ClusterHealthAnalyzer
from analyzers.observability import ObservabilityAnalyzer
from analyzers.vulnerability import VulnerabilityAnalyzer
from utils.k8s_client import KubernetesClient
from utils.prometheus import PrometheusClient
from utils.reporting import Report, ReportFormatter

# Set up logging
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class KubeAnalyzer:
    """Main KubeAnalyzer application class."""
    
    def __init__(self, config: Dict):
        """
        Initialize KubeAnalyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Initializing KubeAnalyzer")
        
        # Extract configuration values
        self.prometheus_url = config.get('prometheus_url', 'http://prometheus-server:9090')
        self.in_cluster = config.get('in_cluster', True)
        self.output_format = config.get('output_format', 'json')
        self.output_dir = config.get('output_dir', './reports')
        
        # Initialize clients
        self.k8s_client = KubernetesClient(in_cluster=self.in_cluster)
        self.prometheus_client = PrometheusClient(base_url=self.prometheus_url)
        
        # Initialize analyzers
        self.observability_analyzer = ObservabilityAnalyzer(
            self.k8s_client, 
            self.prometheus_client
        )
        self.vulnerability_analyzer = VulnerabilityAnalyzer(self.k8s_client)
        self.cluster_health_analyzer = ClusterHealthAnalyzer(
            self.k8s_client
        )
        
        logger.info("KubeAnalyzer initialized successfully")
    
    def run_analysis(self, namespaces: Optional[List[str]] = None) -> Report:
        """
        Run all analyzers to produce a comprehensive analysis report.
        
        Args:
            namespaces: Optional list of namespaces to analyze
                       If None, all namespaces will be analyzed
        
        Returns:
            Report object containing analysis results
        """
        logger.info("Starting comprehensive analysis")
        start_time = datetime.now()
        
        # Create a new report
        report = Report(name="KubeAnalyzer Comprehensive Report")
        
        # Determine which namespaces to analyze
        if not namespaces:
            namespaces = self.k8s_client.list_namespaces()
            logger.info(f"Analyzing all {len(namespaces)} namespaces")
        else:
            logger.info(f"Analyzing specified namespaces: {', '.join(namespaces)}")
        
        # Run cluster health analysis
        logger.info("Running cluster health analysis")
        cluster_health_results = self.cluster_health_analyzer.analyze()
        report.add_section("cluster_health", cluster_health_results)
        
        # Run vulnerability analysis
        logger.info("Running vulnerability analysis")
        vulnerability_results = self.vulnerability_analyzer.analyze(namespaces)
        report.add_section("vulnerability", vulnerability_results)
        
        # Run observability analysis for each namespace
        logger.info("Running observability analysis")
        observability_results = self.observability_analyzer.analyze(namespaces)
        report.add_section("observability", observability_results)
        
        # Generate report summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count issues by severity
        total_issues = 0
        issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        
        # Process findings from each section
        for section_name, section_data in report.data["sections"].items():
            findings = section_data.get("findings", [])
            total_issues += len(findings)
            
            for finding in findings:
                severity = finding.get("severity", "info").lower()
                if severity in issues_by_severity:
                    issues_by_severity[severity] += 1
        
        # Generate AI-powered recommendations summary
        recommendation_count = sum(
            len(section_data.get("recommendations", []))
            for section_data in report.data["sections"].values()
        )
        
        # Add summary to report
        summary = {
            "analyzed_namespaces": len(namespaces),
            "services_analyzed": sum(
                len(ns_data.get("services", {}))
                for ns_data in observability_results.get("namespaces", {}).values()
            ),
            "total_issues": total_issues,
            "issues_by_severity": issues_by_severity,
            "recommendation_count": recommendation_count,
            "analysis_duration_seconds": duration
        }
        report.add_summary(summary)
        
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {total_issues} issues")
        
        return report
    
    def run_observability_analysis(self, namespaces: Optional[List[str]] = None) -> Report:
        """
        Run only observability analysis.
        
        Args:
            namespaces: Optional list of namespaces to analyze
        
        Returns:
            Report object containing observability analysis results
        """
        logger.info("Starting observability analysis")
        
        # Create a new report
        report = Report(name="KubeAnalyzer Observability Report")
        
        # Determine which namespaces to analyze
        if not namespaces:
            namespaces = self.k8s_client.list_namespaces()
        
        # Run observability analysis
        observability_results = self.observability_analyzer.analyze(namespaces)
        report.add_section("observability", observability_results)
        
        return report
    
    def run_vulnerability_analysis(self, namespaces: Optional[List[str]] = None) -> Report:
        """
        Run only vulnerability analysis.
        
        Args:
            namespaces: Optional list of namespaces to analyze
        
        Returns:
            Report object containing vulnerability analysis results
        """
        logger.info("Starting vulnerability analysis")
        
        # Create a new report
        report = Report(name="KubeAnalyzer Vulnerability Report")
        
        # Determine which namespaces to analyze
        if not namespaces:
            namespaces = self.k8s_client.list_namespaces()
        
        # Run vulnerability analysis
        vulnerability_results = self.vulnerability_analyzer.analyze(namespaces)
        report.add_section("vulnerability", vulnerability_results)
        
        return report
    
    def run_cluster_health_analysis(self) -> Report:
        """
        Run only cluster health analysis.
        
        Returns:
            Report object containing cluster health analysis results
        """
        logger.info("Starting cluster health analysis")
        
        # Create a new report
        report = Report(name="KubeAnalyzer Cluster Health Report")
        
        # Run cluster health analysis
        cluster_health_results = self.cluster_health_analyzer.analyze()
        report.add_section("cluster_health", cluster_health_results)
        
        return report
    
    def save_report(self, report: Report, output_format: Optional[str] = None) -> str:
        """
        Print report output to console instead of saving to file.
        
        Args:
            report: Report object to display
            output_format: Optional output format override
        
        Returns:
            String representation of the report
        """
        format_to_use = output_format or self.output_format
        
        # Get the appropriate string representation based on format
        if format_to_use.lower() == 'json':
            output = report.to_json(pretty=True)
        elif format_to_use.lower() == 'yaml':
            output = report.to_yaml()
        elif format_to_use.lower() == 'markdown':
            output = ReportFormatter.format_markdown(report.get_report())
        elif format_to_use.lower() == 'html':
            output = ReportFormatter.format_html(report.get_report())
        else:
            logger.error(f"Unsupported output format: {format_to_use}")
            return ""
            
        # Print the output to console
        print(f"\n--- KubeAnalyzer Report ({format_to_use}) ---\n")
        print(output)
        print("\n--- End of Report ---\n")
        
        return output


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KubeAnalyzer - Kubernetes Cluster Analyzer AI Agent"
    )
    parser.add_argument(
        "--in-cluster", 
        action="store_true",
        help="Run in-cluster using service account"
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://prometheus-server:9090",
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "yaml", "markdown", "html"],
        default="json",
        help="Output format for reports"
    )
    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        help="Namespaces to analyze (default: all)"
    )
    parser.add_argument(
        "--analysis-type",
        choices=["all", "observability", "vulnerability", "cluster-health"],
        default="all",
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create config from args
    config = {
        "in_cluster": args.in_cluster,
        "prometheus_url": args.prometheus_url,
        "output_format": args.output_format,
        "output_dir": args.output_dir
    }
    
    try:
        # Initialize KubeAnalyzer
        analyzer = KubeAnalyzer(config)
        
        # Run requested analysis
        if args.analysis_type == "all":
            report = analyzer.run_analysis(args.namespaces)
        elif args.analysis_type == "observability":
            report = analyzer.run_observability_analysis(args.namespaces)
        elif args.analysis_type == "vulnerability":
            report = analyzer.run_vulnerability_analysis(args.namespaces)
        elif args.analysis_type == "cluster-health":
            report = analyzer.run_cluster_health_analysis()
        
        # Save report
        report_path = analyzer.save_report(report)
        if report_path:
            logger.info(f"Report saved to {report_path}")
        
        # Print report summary to console
        summary = report.get_report().get("summary", {})
        print("\nKubeAnalyzer Summary:")
        print(f"- Analyzed Namespaces: {summary.get('analyzed_namespaces', 0)}")
        print(f"- Services Analyzed: {summary.get('services_analyzed', 0)}")
        print(f"- Total Issues: {summary.get('total_issues', 0)}")
        
        # Print issues by severity
        issues_by_severity = summary.get("issues_by_severity", {})
        for severity, count in issues_by_severity.items():
            if count > 0:
                print(f"  - {severity.upper()}: {count}")
        
        print(f"- Recommendations: {summary.get('recommendation_count', 0)}")
        print(f"- Analysis Duration: {summary.get('analysis_duration_seconds', 0):.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error running KubeAnalyzer: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
