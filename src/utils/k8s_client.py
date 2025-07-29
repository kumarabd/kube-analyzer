"""
Kubernetes Client for KubeAnalyzer

This module provides a wrapper around the Kubernetes Python client
to interact with the Kubernetes API.
"""

import logging
from typing import Dict, List, Optional

from kubernetes import client, config

logger = logging.getLogger(__name__)


class KubernetesClient:
    """Wrapper around the Kubernetes Python client for KubeAnalyzer."""

    def __init__(self, in_cluster: bool = True):
        """
        Initialize the Kubernetes client.
        
        Args:
            in_cluster: Whether the client is running inside a Kubernetes cluster
        """
        try:
            if in_cluster:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
        except Exception as e:
            logger.error(f"Failed to load Kubernetes configuration: {str(e)}")
            raise
        
        # Initialize API clients
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.rbac_v1 = client.RbacAuthorizationV1Api()
        self.custom_objects = client.CustomObjectsApi()
    
    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the cluster.
        
        Returns:
            List of namespace names
        """
        try:
            namespaces = self.core_v1.list_namespace()
            return [ns.metadata.name for ns in namespaces.items]
        except Exception as e:
            logger.error(f"Failed to list namespaces: {str(e)}")
            return []
    
    def list_nodes(self) -> List[Dict]:
        """
        List all nodes in the cluster.
        
        Returns:
            List of node objects as dictionaries
        """
        try:
            nodes = self.core_v1.list_node()
            return [self._serialize_object(node) for node in nodes.items]
        except Exception as e:
            logger.error(f"Failed to list nodes: {str(e)}")
            return []
    
    def list_pods(self, namespace: Optional[str] = None) -> List[Dict]:
        """
        List pods in the specified namespace or across all namespaces.
        
        Args:
            namespace: Optional namespace to filter by
            
        Returns:
            List of pod objects as dictionaries
        """
        try:
            if namespace:
                pods = self.core_v1.list_namespaced_pod(namespace)
            else:
                pods = self.core_v1.list_pod_for_all_namespaces()
            return [self._serialize_object(pod) for pod in pods.items]
        except Exception as e:
            logger.error(f"Failed to list pods: {str(e)}")
            return []
    
    def list_services(self, namespace: str) -> List[str]:
        """
        List services in the specified namespace.
        
        Args:
            namespace: Namespace to list services from
            
        Returns:
            List of service names
        """
        try:
            services = self.core_v1.list_namespaced_service(namespace)
            return [service.metadata.name for service in services.items]
        except Exception as e:
            logger.error(f"Failed to list services in namespace {namespace}: {str(e)}")
            return []
    
    def get_service(self, namespace: str, name: str) -> Dict:
        """
        Get a specific service by name.
        
        Args:
            namespace: Namespace the service is in
            name: Name of the service
            
        Returns:
            Service object as dictionary
        """
        try:
            service = self.core_v1.read_namespaced_service(name, namespace)
            return self._serialize_object(service)
        except client.rest.ApiException as e:
            if e.status == 404:
                logger.warning(f"Service {name} not found in namespace {namespace}")
            else:
                logger.error(f"API error getting service {name} in namespace {namespace}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get service {name} in namespace {namespace}: {str(e)}")
            return {}
    
    def list_pods_for_service(self, namespace: str, service_name: str) -> List[Dict]:
        """
        List pods that belong to a specific service.
        
        Args:
            namespace: Namespace the service is in
            service_name: Name of the service
            
        Returns:
            List of pod objects as dictionaries
        """
        try:
            service = self.core_v1.read_namespaced_service(service_name, namespace)
            selector = service.spec.selector
            
            if not selector:
                logger.warning(f"Service {service_name} in namespace {namespace} has no selector")
                return []
            
            # Create label selector string
            label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
            
            # List pods matching the service selector
            pods = self.core_v1.list_namespaced_pod(namespace, label_selector=label_selector)
            return [self._serialize_object(pod) for pod in pods.items]
        except client.rest.ApiException as e:
            if e.status == 404:
                logger.warning(f"Service {service_name} not found in namespace {namespace}")
            else:
                logger.error(f"API error listing pods for service {service_name}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Failed to list pods for service {service_name}: {str(e)}")
            return []
    
    def list_deployments(self, namespace: str) -> List[str]:
        """
        List deployments in the specified namespace.
        
        Args:
            namespace: Namespace to list deployments from
            
        Returns:
            List of deployment names
        """
        try:
            deployments = self.apps_v1.list_namespaced_deployment(namespace)
            return [deployment.metadata.name for deployment in deployments.items]
        except Exception as e:
            logger.error(f"Failed to list deployments in namespace {namespace}: {str(e)}")
            return []
    
    def get_deployment(self, namespace: str, name: str) -> Dict:
        """
        Get a specific deployment by name.
        
        Args:
            namespace: Namespace the deployment is in
            name: Name of the deployment
            
        Returns:
            Deployment object as dictionary
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            return self._serialize_object(deployment)
        except client.rest.ApiException as e:
            if e.status == 404:
                logger.warning(f"Deployment {name} not found in namespace {namespace}")
            else:
                logger.error(f"API error getting deployment {name}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get deployment {name}: {str(e)}")
            return {}
    
    def get_pod(self, namespace: str, name: str) -> Dict:
        """
        Get a specific pod by name.
        
        Args:
            namespace: Namespace the pod is in
            name: Name of the pod
            
        Returns:
            Pod object as dictionary
        """
        try:
            pod = self.core_v1.read_namespaced_pod(name, namespace)
            return self._serialize_object(pod)
        except client.rest.ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {name} not found in namespace {namespace}")
            else:
                logger.error(f"API error getting pod {name}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get pod {name}: {str(e)}")
            return {}
    
    def list_pods_on_node(self, node_name: str) -> List[Dict]:
        """
        List pods running on a specific node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            List of pod objects as dictionaries
        """
        try:
            field_selector = f"spec.nodeName={node_name}"
            pods = self.core_v1.list_pod_for_all_namespaces(field_selector=field_selector)
            return [self._serialize_object(pod) for pod in pods.items]
        except Exception as e:
            logger.error(f"Failed to list pods on node {node_name}: {str(e)}")
            return []
    
    def list_persistent_volumes(self) -> List[Dict]:
        """
        List all persistent volumes in the cluster.
        
        Returns:
            List of persistent volume objects as dictionaries
        """
        try:
            pvs = self.core_v1.list_persistent_volume()
            return [self._serialize_object(pv) for pv in pvs.items]
        except Exception as e:
            logger.error(f"Failed to list persistent volumes: {str(e)}")
            return []
    
    def list_persistent_volume_claims(self, namespace: str) -> List[Dict]:
        """
        List persistent volume claims in the specified namespace.
        
        Args:
            namespace: Namespace to list PVCs from
            
        Returns:
            List of PVC objects as dictionaries
        """
        try:
            pvcs = self.core_v1.list_namespaced_persistent_volume_claim(namespace)
            return [self._serialize_object(pvc) for pvc in pvcs.items]
        except Exception as e:
            logger.error(f"Failed to list PVCs in namespace {namespace}: {str(e)}")
            return []
    
    def _serialize_object(self, obj) -> Dict:
        """
        Serialize a Kubernetes object to a dictionary.
        
        Args:
            obj: Kubernetes object to serialize
            
        Returns:
            Dictionary representation of the object
        """
        return client.ApiClient().sanitize_for_serialization(obj)
