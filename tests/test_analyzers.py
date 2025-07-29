#!/usr/bin/env python3
"""
Unit tests for KubeAnalyzer analyzers.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.analyzers.observability import ObservabilityAnalyzer
from src.analyzers.vulnerability import VulnerabilityAnalyzer
from src.analyzers.cluster_health import ClusterHealthAnalyzer


class TestObservabilityAnalyzer(unittest.TestCase):
    """Tests for ObservabilityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.k8s_client = MagicMock()
        self.prometheus_client = MagicMock()
        self.analyzer = ObservabilityAnalyzer(self.k8s_client, self.prometheus_client)
    
    def test_check_metrics_coverage(self):
        """Test metrics coverage check."""
        # Mock necessary data
        namespace = "test-namespace"
        service_name = "test-service"
        
        # Mock Prometheus response with metrics
        self.prometheus_client.get_metrics_for_service.return_value = [
            {"name": "http_requests_total", "labels": {"service": service_name}},
            {"name": "http_request_duration_seconds", "labels": {"service": service_name}}
        ]
        
        # Call the method
        coverage = self.analyzer._check_metrics_coverage(namespace, service_name)
        
        # Assertions
        self.assertGreater(coverage["score"], 0)
        self.assertEqual(len(coverage["available_metrics"]), 2)
        self.prometheus_client.get_metrics_for_service.assert_called_once_with(namespace, service_name)
    
    def test_check_logging_coverage(self):
        """Test logging coverage check."""
        # Mock necessary data
        namespace = "test-namespace"
        service_name = "test-service"
        
        # Mock K8s response with pods that have logging
        pods = [
            {
                "metadata": {
                    "name": "test-pod-1"
                },
                "spec": {
                    "containers": [
                        {
                            "name": "test-container",
                            "env": [{"name": "LOG_LEVEL", "value": "info"}]
                        }
                    ]
                }
            }
        ]
        self.k8s_client.list_pods_for_service.return_value = pods
        
        # Call the method
        coverage = self.analyzer._check_logging_coverage(namespace, service_name)
        
        # Assertions
        self.assertGreater(coverage["score"], 0)
        self.assertEqual(len(coverage["findings"]), 0)  # No findings means good coverage
        self.k8s_client.list_pods_for_service.assert_called_once_with(namespace, service_name)


class TestVulnerabilityAnalyzer(unittest.TestCase):
    """Tests for VulnerabilityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.k8s_client = MagicMock()
        self.analyzer = VulnerabilityAnalyzer(self.k8s_client)
    
    @patch('src.analyzers.vulnerability.subprocess.run')
    def test_scan_image(self, mock_run):
        """Test image scanning."""
        # Mock subprocess.run response
        mock_process = MagicMock()
        mock_process.stdout = """
        {
          "Results": [
            {
              "Target": "nginx:latest",
              "Vulnerabilities": [
                {
                  "VulnerabilityID": "CVE-2023-12345",
                  "Severity": "HIGH",
                  "Title": "Test vulnerability"
                }
              ]
            }
          ]
        }
        """
        mock_run.return_value = mock_process
        
        # Call the method
        image = "nginx:latest"
        results = self.analyzer._scan_image(image)
        
        # Assertions
        self.assertEqual(len(results.get("vulnerabilities", [])), 1)
        self.assertEqual(results["vulnerabilities"][0]["severity"], "HIGH")
        mock_run.assert_called_once()


class TestClusterHealthAnalyzer(unittest.TestCase):
    """Tests for ClusterHealthAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.k8s_client = MagicMock()
        self.prometheus_client = MagicMock()
        self.analyzer = ClusterHealthAnalyzer(self.k8s_client, self.prometheus_client)
    
    def test_analyze_node_status(self):
        """Test node status analysis."""
        # Mock node data
        nodes = [
            {
                "metadata": {"name": "node1"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "True"
                        }
                    ],
                    "allocatable": {
                        "cpu": "4",
                        "memory": "16Gi"
                    },
                    "capacity": {
                        "cpu": "4",
                        "memory": "16Gi"
                    }
                }
            }
        ]
        self.k8s_client.list_nodes.return_value = nodes
        
        # Call the method
        results = self.analyzer._analyze_node_status()
        
        # Assertions
        self.assertEqual(results["healthy_nodes"], 1)
        self.assertEqual(results["unhealthy_nodes"], 0)
        self.k8s_client.list_nodes.assert_called_once()
    
    def test_analyze_resource_usage(self):
        """Test resource usage analysis."""
        # Mock Prometheus response
        self.prometheus_client.get_resource_usage.return_value = {
            "cpu": {
                "total": 2.5,  # 2.5 cores used
                "details": []
            },
            "memory": {
                "total": 4 * 1024 * 1024 * 1024,  # 4GB used
                "details": []
            }
        }
        
        # Mock node capacities
        nodes = [
            {
                "metadata": {"name": "node1"},
                "status": {
                    "allocatable": {
                        "cpu": "4",
                        "memory": "16Gi"
                    }
                }
            }
        ]
        self.k8s_client.list_nodes.return_value = nodes
        
        # Call the method
        results = self.analyzer._analyze_resource_usage()
        
        # Assertions
        self.assertIn("cpu_usage_percentage", results)
        self.assertIn("memory_usage_percentage", results)
        self.prometheus_client.get_resource_usage.assert_called_once()


if __name__ == '__main__':
    unittest.main()
