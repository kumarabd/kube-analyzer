"""
Prometheus Client for KubeAnalyzer

This module provides a client for interacting with Prometheus
to retrieve metrics and alerts for Kubernetes services.
"""

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for interacting with Prometheus API."""

    def __init__(self, base_url: str = "http://prometheus-server:9090"):
        """
        Initialize the Prometheus client.
        
        Args:
            base_url: Base URL for the Prometheus API
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"
        logger.info(f"Initialized Prometheus client with URL: {self.base_url}")
        
    def check_connection(self) -> bool:
        """
        Check if the connection to Prometheus is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/status/buildinfo", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to Prometheus")
                return True
            else:
                logger.warning(f"Failed to connect to Prometheus: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Prometheus: {str(e)}")
            return False
    
    def get_metrics_for_service(self, namespace: str, service_name: str) -> List[Dict]:
        """
        Get metrics for a specific service.
        
        Args:
            namespace: Namespace the service is in
            service_name: Name of the service
            
        Returns:
            List of metrics associated with the service
        """
        query = f'{{namespace="{namespace}", service="{service_name}"}}'
        try:
            response = requests.get(
                f"{self.api_url}/series",
                params={"match[]": query},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get metrics for service {service_name}: Status {response.status_code}")
                return []
            
            data = response.json()
            if data.get("status") != "success" or "data" not in data:
                logger.warning(f"Invalid response from Prometheus for service {service_name}")
                return []
            
            return data["data"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting metrics for service {service_name}: {str(e)}")
            return []
    
    def get_alerts_for_service(self, namespace: str, service_name: str) -> List[Dict]:
        """
        Get alerts for a specific service.
        
        Args:
            namespace: Namespace the service is in
            service_name: Name of the service
            
        Returns:
            List of alerts associated with the service
        """
        try:
            # Get all alerts from Prometheus
            response = requests.get(f"{self.api_url}/rules", timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get alerts: Status {response.status_code}")
                return []
            
            data = response.json()
            if data.get("status") != "success" or "data" not in data:
                logger.warning("Invalid response from Prometheus for alerts")
                return []
            
            # Filter alerts related to the service
            service_alerts = []
            groups = data["data"].get("groups", [])
            for group in groups:
                for rule in group.get("rules", []):
                    if rule.get("type") != "alerting":
                        continue
                    
                    # Check if alert is related to the service
                    alert_namespace = rule.get("annotations", {}).get("namespace", "")
                    alert_service = rule.get("annotations", {}).get("service", "")
                    alert_labels = rule.get("labels", {})
                    
                    if ((alert_namespace == namespace and alert_service == service_name) or
                            (alert_labels.get("namespace") == namespace and alert_labels.get("service") == service_name) or
                            (service_name in rule.get("name", "").lower())):
                        service_alerts.append(rule)
            
            return service_alerts
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting alerts for service {service_name}: {str(e)}")
            return []
    
    def query(self, query: str, time: Optional[str] = None) -> Dict:
        """
        Execute a PromQL query.
        
        Args:
            query: PromQL query string
            time: Optional time for query (rfc3339 or Unix timestamp)
            
        Returns:
            Query results
        """
        params = {"query": query}
        if time:
            params["time"] = time
        
        try:
            response = requests.get(f"{self.api_url}/query", params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to execute query: Status {response.status_code}")
                return {}
            
            data = response.json()
            if data.get("status") != "success":
                logger.warning(f"Query failed: {data.get('error', 'Unknown error')}")
                return {}
            
            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing query: {str(e)}")
            return {}
    
    def query_range(self, query: str, start: str, end: str, step: str) -> Dict:
        """
        Execute a PromQL range query.
        
        Args:
            query: PromQL query string
            start: Start time (rfc3339 or Unix timestamp)
            end: End time (rfc3339 or Unix timestamp)
            step: Query resolution step width
            
        Returns:
            Query results
        """
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        
        try:
            response = requests.get(f"{self.api_url}/query_range", params=params, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to execute range query: Status {response.status_code}")
                return {}
            
            data = response.json()
            if data.get("status") != "success":
                logger.warning(f"Range query failed: {data.get('error', 'Unknown error')}")
                return {}
            
            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing range query: {str(e)}")
            return {}
    
    def get_resource_usage(self, namespace: Optional[str] = None) -> Dict:
        """
        Get resource usage metrics for the cluster or a specific namespace.
        
        Args:
            namespace: Optional namespace to filter metrics
            
        Returns:
            Dictionary with resource usage metrics
        """
        # Get current time
        now = int(time.time())
        one_hour_ago = now - 3600
        
        # Build namespace filter
        namespace_filter = f'namespace="{namespace}"' if namespace else ""
        
        # CPU usage query
        cpu_query = "sum(rate(container_cpu_usage_seconds_total{container!='POD',container!=''}[5m]))"
        if namespace_filter:
            cpu_query += f" by (pod) * on (pod) group_left() kube_pod_info{{{namespace_filter}}}"
        
        # Memory usage query
        mem_query = "sum(container_memory_working_set_bytes{container!='POD',container!=''})"
        if namespace_filter:
            mem_query += f" by (pod) * on (pod) group_left() kube_pod_info{{{namespace_filter}}}"
        
        # Execute queries
        cpu_usage = self.query(cpu_query)
        mem_usage = self.query(mem_query)
        
        # Process results
        result = {
            "cpu": self._process_resource_metrics(cpu_usage),
            "memory": self._process_resource_metrics(mem_usage)
        }
        
        return result
    
    def _process_resource_metrics(self, metrics: Dict) -> Dict:
        """Process resource metrics from Prometheus query results."""
        result = {
            "total": 0.0,
            "details": []
        }
        
        result_type = metrics.get("resultType")
        if result_type != "vector":
            return result
        
        for item in metrics.get("result", []):
            metric = item.get("metric", {})
            value = item.get("value", [0, "0"])
            
            # Extract the value (second element in the value array)
            try:
                numeric_value = float(value[1])
            except (ValueError, IndexError):
                numeric_value = 0.0
            
            result["total"] += numeric_value
            result["details"].append({
                "metric": metric,
                "value": numeric_value
            })
        
        return result
