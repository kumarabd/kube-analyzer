"""
Cluster Health Analyzer for KubeAnalyzer

This module analyzes Kubernetes cluster health, including resource utilization,
node conditions, control plane status, and potential optimization areas.
"""

import logging
from typing import Dict, List, Optional

from ..utils.k8s_client import KubernetesClient

logger = logging.getLogger(__name__)


class ClusterHealthAnalyzer:
    """Analyzes Kubernetes cluster health and identifies potential issues."""

    def __init__(self, k8s_client: KubernetesClient):
        """
        Initialize the ClusterHealthAnalyzer.
        
        Args:
            k8s_client: Kubernetes client instance
        """
        self.k8s_client = k8s_client
        
    def analyze(self, namespaces=None) -> Dict:
        """
        Perform a comprehensive cluster health analysis.
        
        Args:
            namespaces: Optional list of namespaces to analyze (not used for cluster-wide analysis)
            
        Returns:
            Dict with analysis results
        """
        return self.analyze_cluster()
        
    def analyze_cluster(self) -> Dict:
        """
        Perform a comprehensive cluster health analysis.
        
        Returns:
            Dict with analysis results
        """
        logger.info("Starting comprehensive cluster health analysis")
        
        results = {
            "nodes": self.analyze_nodes(),
            "control_plane": self.analyze_control_plane(),
            "resource_usage": self.analyze_resource_usage(),
            "pod_distribution": self.analyze_pod_distribution(),
            "networking": self.analyze_networking(),
            "storage": self.analyze_storage(),
            "recommendations": []
        }
        
        # Generate overall health score and recommendations
        results["health_score"] = self._calculate_health_score(results)
        results["health_status"] = self._get_health_status(results["health_score"])
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def analyze_nodes(self) -> Dict:
        """
        Analyze cluster node health and status.
        
        Returns:
            Dict with node analysis results
        """
        nodes = self.k8s_client.list_nodes()
        node_statuses = []
        issues = []
        
        # Track aggregate statistics
        total_nodes = len(nodes)
        ready_nodes = 0
        cordoned_nodes = 0
        
        for node in nodes:
            node_name = node.get("metadata", {}).get("name", "unknown")
            status = node.get("status", {})
            spec = node.get("spec", {})
            
            # Check if node is ready
            is_ready = False
            for condition in status.get("conditions", []):
                if condition.get("type") == "Ready":
                    is_ready = condition.get("status") == "True"
            
            if is_ready:
                ready_nodes += 1
            
            # Check if node is cordoned (unschedulable)
            is_cordoned = spec.get("unschedulable", False)
            if is_cordoned:
                cordoned_nodes += 1
                issues.append({
                    "node": node_name,
                    "issue": "node_cordoned",
                    "description": "Node is cordoned (marked as unschedulable)",
                    "severity": "medium"
                })
            
            # Check node conditions
            for condition in status.get("conditions", []):
                condition_type = condition.get("type")
                condition_status = condition.get("status")
                
                # Skip Ready condition as we already checked it
                if condition_type == "Ready":
                    continue
                
                # Check for issues in other conditions
                if (condition_type in ["DiskPressure", "MemoryPressure", "PIDPressure"] 
                        and condition_status == "True"):
                    issues.append({
                        "node": node_name,
                        "issue": f"node_{condition_type.lower()}",
                        "description": f"Node has {condition_type}",
                        "severity": "high"
                    })
                elif condition_type == "NetworkUnavailable" and condition_status == "True":
                    issues.append({
                        "node": node_name,
                        "issue": "node_network_unavailable",
                        "description": "Node has network connectivity issues",
                        "severity": "high"
                    })
            
            # Add node status to results
            node_statuses.append({
                "name": node_name,
                "ready": is_ready,
                "cordoned": is_cordoned,
                "conditions": status.get("conditions", []),
                "allocatable": status.get("allocatable", {}),
                "capacity": status.get("capacity", {}),
                "kubelet_version": status.get("nodeInfo", {}).get("kubeletVersion", "")
            })
        
        return {
            "total_nodes": total_nodes,
            "ready_nodes": ready_nodes,
            "cordoned_nodes": cordoned_nodes,
            "health_percentage": (ready_nodes / total_nodes) * 100 if total_nodes > 0 else 0,
            "nodes": node_statuses,
            "issues": issues
        }
    
    def analyze_control_plane(self) -> Dict:
        """
        Analyze control plane component health.
        
        Returns:
            Dict with control plane analysis results
        """
        # Get control plane pods (API server, scheduler, controller manager, etcd)
        cp_pods = self.k8s_client.list_pods(namespace="kube-system")
        
        # Filter for control plane components
        control_plane_components = {
            "api_server": [],
            "scheduler": [],
            "controller_manager": [],
            "etcd": []
        }
        
        for pod in cp_pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            
            if "kube-apiserver" in pod_name:
                control_plane_components["api_server"].append(pod)
            elif "kube-scheduler" in pod_name:
                control_plane_components["scheduler"].append(pod)
            elif "kube-controller-manager" in pod_name:
                control_plane_components["controller_manager"].append(pod)
            elif "etcd" in pod_name:
                control_plane_components["etcd"].append(pod)
        
        # Analyze each component
        component_status = {}
        issues = []
        
        for component, pods in control_plane_components.items():
            total = len(pods)
            ready = sum(1 for pod in pods if self._is_pod_ready(pod))
            
            component_status[component] = {
                "total": total,
                "ready": ready,
                "health_percentage": (ready / total) * 100 if total > 0 else 0
            }
            
            if total > 0 and ready < total:
                issues.append({
                    "component": component,
                    "issue": "control_plane_component_not_ready",
                    "description": f"{component.replace('_', ' ').title()} has {total - ready} unhealthy instances",
                    "severity": "critical"
                })
        
        return {
            "components": component_status,
            "issues": issues
        }
    
    def analyze_resource_usage(self) -> Dict:
        """
        Analyze cluster-wide resource utilization using an LLM agent.
        The agent performs an iterative analysis session to provide insights
        about cluster resource usage.
        
        Returns:
            Dict with resource usage analysis from the LLM agent
        """
        import os
        import logging
        import json
        from typing import List, Optional, Dict, Any, Callable
        from pydantic import BaseModel, Field
        import instructor
        from openai import OpenAI
        from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
        from atomic_agents.lib.components.agent_memory import AgentMemory

        # Define our own Tool class since it's not directly available in atomic-agents
        class Tool:
            """Simple Tool class for atomic-agents"""
            def __init__(self, name: str, description: str, function: Callable):
                self.name = name
                self.description = description
                self.function = function
                
        # Define NodeResource and ClusterResourceData models
        class NodeResource(BaseModel):
            """Model representing a Kubernetes node's resources"""
            name: str
            cpu_capacity: float
            cpu_allocatable: float
            memory_capacity: float
            memory_allocatable: float
            pods_capacity: int
            pods_allocatable: int
            conditions: List[Dict[str, Any]]
            
        class ClusterResourceData(BaseModel):
            """Model representing overall cluster resource data"""
            nodes: List[NodeResource]
            total_capacity: Dict[str, Any]
            total_allocatable: Dict[str, Any]
            usage_metrics: Optional[Dict[str, Any]] = None
        
        # Set up OpenAI client with instructor
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logging.warning("OPENAI_API_KEY not found. Using mock LLM analysis.")
            # Return a minimal compatible structure when no API key is available
            return {
                "summary": "No OpenAI API key available for resource analysis",
                "insights": ["Configure OPENAI_API_KEY environment variable for LLM-based analysis"],
                "recommendations": ["Set up proper API key for detailed resource analysis"],
                "risks": [],
                "metrics": {},
                "utilization": {
                    "cpu_request": 0.0,
                    "cpu_limit": 0.0,
                    "memory_request": 0.0,
                    "memory_limit": 0.0,
                    "pods": 0.0,
                    "cpu_request_percentage": 0.0,
                    "memory_request_percentage": 0.0,
                    "pod_percentage": 0.0
                }
            }
        
        # Collect node data
        nodes_data = []
        total_capacity = {"cpu": 0, "memory": 0, "pods": 0}
        total_allocatable = {"cpu": 0, "memory": 0, "pods": 0}
        
        try:
            nodes = self.k8s_client.list_nodes()
            
            for node in nodes:
                name = node.get("metadata", {}).get("name", "unknown")
                capacity = node.get("status", {}).get("capacity", {})
                allocatable = node.get("status", {}).get("allocatable", {})
                conditions = node.get("status", {}).get("conditions", [])
                
                # Process CPU
                cpu_capacity = self._parse_cpu(capacity.get("cpu", "0"))
                cpu_allocatable = self._parse_cpu(allocatable.get("cpu", "0"))
                total_capacity["cpu"] += cpu_capacity
                total_allocatable["cpu"] += cpu_allocatable
                
                # Process memory
                memory_capacity = self._parse_memory(capacity.get("memory", "0"))
                memory_allocatable = self._parse_memory(allocatable.get("memory", "0"))
                total_capacity["memory"] += memory_capacity
                total_allocatable["memory"] += memory_allocatable
                
                # Process pods
                pods_capacity = int(capacity.get("pods", "0"))
                pods_allocatable = int(allocatable.get("pods", "0"))
                total_capacity["pods"] += pods_capacity
                total_allocatable["pods"] += pods_allocatable
                
                # Add node data
                nodes_data.append(
                    NodeResource(
                        name=name,
                        cpu_capacity=cpu_capacity,
                        cpu_allocatable=cpu_allocatable,
                        memory_capacity=memory_capacity,
                        memory_allocatable=memory_allocatable,
                        pods_capacity=pods_capacity,
                        pods_allocatable=pods_allocatable,
                        conditions=conditions
                    )
                )
            
            # Try to get current usage metrics if available
            usage_metrics = None
            try:
                # This would be replaced with actual metrics from Prometheus or similar
                # For now, we'll leave it as None
                pass
            except Exception as e:
                logging.warning(f"Failed to get usage metrics: {e}")
                
            # Create cluster resource data
            cluster_data = ClusterResourceData(
                nodes=nodes_data,
                total_capacity=total_capacity,
                total_allocatable=total_allocatable,
                usage_metrics=usage_metrics
            )
            
            # Define prompt for the resource analysis
            resource_prompt = f"""You are a Kubernetes resource optimization expert. 
            Analyze the provided cluster resource data and generate insights.
            Focus on resource efficiency, potential bottlenecks, and optimization opportunities.
            
            Here is the cluster data:\n{cluster_data.model_dump_json(indent=2)}
            """
            
            # Define input/output schemas for the agent
            class ResourceAgentInput(BaseIOSchema):
                """Input for the Kubernetes resource analysis agent"""
                cluster_data: dict = Field(description="Initial cluster resource data")
                query: Optional[str] = Field(None, description="Specific query or follow-up question")
            
            class ResourceAgentOutput(BaseIOSchema):
                """Output from the Kubernetes resource analysis agent"""
                summary: str = Field(description="Summary of resource usage analysis")
                insights: List[str] = Field(description="Key insights about resource usage")
                recommendations: List[str] = Field(description="Recommendations for resource optimization")
                risks: List[dict] = Field(description="Potential risks or issues identified")
                metrics: dict = Field(description="Calculated metrics including utilization percentages")
                follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions for deeper analysis")
            
            # Define tool output schemas
            class K8sNamespacesResult(BaseModel):
                """Result of get_namespaces tool"""
                namespaces: List[dict] = Field(description="List of namespace objects")
                error: Optional[str] = Field(None, description="Error message if any")
            
            class K8sPodCountResult(BaseModel):
                """Result of get_pod_count_by_namespace tool"""
                counts: Dict[str, int] = Field(description="Pod counts by namespace")
                error: Optional[str] = Field(None, description="Error message if any")
            
            class K8sResourceRequestsResult(BaseModel):
                """Result of get_resource_requests_by_namespace tool"""
                requests: Dict[str, Dict[str, Any]] = Field(description="Resource requests by namespace")
                error: Optional[str] = Field(None, description="Error message if any")
                
            # Define tools for the agent to interact with Kubernetes
            def get_namespaces(self) -> K8sNamespacesResult:
                """Get list of all namespaces in the cluster"""
                try:
                    namespaces = self.k8s_client.list_namespaces()
                    return K8sNamespacesResult(namespaces=namespaces)
                except Exception as e:
                    return K8sNamespacesResult(
                        namespaces=[], 
                        error=f"Failed to get namespaces: {str(e)}"
                    )
                    
            def get_pod_count_by_namespace(self) -> K8sPodCountResult:
                """Get count of pods by namespace"""
                try:
                    namespaces = self.k8s_client.list_namespaces()
                    counts = {}
                    for ns in namespaces:
                        ns_name = ns.get("metadata", {}).get("name")
                        if ns_name:
                            pods = self.k8s_client.list_pods(namespace=ns_name)
                            counts[ns_name] = len(pods)
                    return K8sPodCountResult(counts=counts)
                except Exception as e:
                    return K8sPodCountResult(
                        counts={}, 
                        error=f"Failed to get pod counts: {str(e)}"
                    )
                    
            def get_resource_requests_by_namespace(self) -> K8sResourceRequestsResult:
                """Get resource requests (CPU/memory) by namespace"""
                try:
                    namespaces = self.k8s_client.list_namespaces()
                    results = {}
                    for ns in namespaces:
                        ns_name = ns.get("metadata", {}).get("name")
                        if ns_name:
                            ns_results = {
                                "cpu_requests": 0,
                                "memory_requests": 0,
                                "pod_count": 0
                            }
                            pods = self.k8s_client.list_pods(namespace=ns_name)
                            for pod in pods:
                                containers = pod.get("spec", {}).get("containers", [])
                                for container in containers:
                                    resources = container.get("resources", {})
                                    requests = resources.get("requests", {})
                                    cpu = requests.get("cpu", "0")
                                    memory = requests.get("memory", "0")
                                    ns_results["cpu_requests"] += self._parse_cpu(cpu) / 1000  # Convert to cores
                                    ns_results["memory_requests"] += self._parse_memory(memory)
                            ns_results["pod_count"] = len(pods)
                            results[ns_name] = ns_results
                    return K8sResourceRequestsResult(requests=results)
                except Exception as e:
                    return K8sResourceRequestsResult(
                        requests={}, 
                        error=f"Failed to get resource requests: {str(e)}"
                    )
            
            # Create agent tools
            k8s_tools = [
                Tool(
                    name="get_namespaces",
                    description="Get list of all namespaces in the cluster",
                    function=get_namespaces.__get__(self, self.__class__)
                ),
                Tool(
                    name="get_pod_count_by_namespace",
                    description="Get count of pods by namespace",
                    function=get_pod_count_by_namespace.__get__(self, self.__class__)
                ),
                Tool(
                    name="get_resource_requests_by_namespace",
                    description="Get resource requests (CPU/memory) by namespace",
                    function=get_resource_requests_by_namespace.__get__(self, self.__class__)
                )
            ]
            
            # Generate system prompt for the agent
            system_prompt = f"""You are a Kubernetes resource optimization expert.
            Analyze the provided cluster resource data and generate insights.
            Focus on resource efficiency, potential bottlenecks, and optimization opportunities.
            
            You can use tools to fetch additional data about the cluster to enhance your analysis.
            When calculating utilization percentages, be sure to include:
            - cpu_request_percentage: Percentage of total CPU requested vs available
            - memory_request_percentage: Percentage of total memory requested vs available
            - pod_percentage: Percentage of total pods used vs maximum allowed
            
            These metrics are critical for the overall cluster health analysis.
            """
            
            # Initialize OpenAI client with instructor
            client = instructor.from_openai(OpenAI(api_key=openai_api_key))
            
            # Create the agent with tools
            agent = BaseAgent(
                config=BaseAgentConfig(
                    client=client,
                    model="gpt-4o",  # Using GPT-4o for high-quality analysis
                    memory=AgentMemory(),
                    tools=k8s_tools,
                    system_prompt=system_prompt,
                    input_schema=ResourceAgentInput,
                    output_schema=ResourceAgentOutput,
                )
            )
            
            logging.info("Running agentic resource usage analysis")
            
            # Execute the agent with initial cluster data and query
            agent_input = ResourceAgentInput(
                cluster_data=json.loads(cluster_data.model_dump_json()),
                query="Perform a comprehensive analysis of the cluster resources. Calculate resource utilization percentages and identify optimization opportunities."
            )
            
            # Run the analysis
            try:
                # Execute the agent with the input
                agent_result = agent.run(agent_input)
                logging.info("Agentic LLM resource analysis completed successfully")
                
                # Convert agent result to format expected by the application
                result = {
                    "summary": agent_result.summary,
                    "insights": agent_result.insights,
                    "recommendations": agent_result.recommendations,
                    "risks": agent_result.risks,
                    "metrics": agent_result.metrics
                }
                
                # Add follow-up questions if available
                if hasattr(agent_result, 'follow_up_questions') and agent_result.follow_up_questions:
                    result["follow_up_questions"] = agent_result.follow_up_questions
                
                # Add the 'utilization' key that the rest of the app expects
                # with default values derived from the LLM analysis
                result['utilization'] = {
                    'cpu_request': 0.0,
                    'cpu_limit': 0.0,
                    'memory_request': 0.0,
                    'memory_limit': 0.0,
                    'pods': 0.0,
                    'cpu_request_percentage': 0.0,
                    'memory_request_percentage': 0.0,
                    'pod_percentage': 0.0
                }
                
                # If the LLM provided metrics, use them to update utilization values
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    metrics = result['metrics']
                    for key in ['cpu_request', 'cpu_limit', 'memory_request', 'memory_limit', 'pods']:
                        if key in metrics:
                            result['utilization'][key] = metrics[key]
                    
                    # Add percentage values if available or calculate them
                    if 'cpu_request_percentage' in metrics:
                        result['utilization']['cpu_request_percentage'] = metrics['cpu_request_percentage']
                    elif 'cpu_allocation_ratio' in metrics:
                        result['utilization']['cpu_request_percentage'] = metrics['cpu_allocation_ratio'] * 100
                    
                    if 'memory_request_percentage' in metrics:
                        result['utilization']['memory_request_percentage'] = metrics['memory_request_percentage']
                    elif 'memory_allocation_ratio' in metrics:
                        result['utilization']['memory_request_percentage'] = metrics['memory_allocation_ratio'] * 100
                    
                    if 'pod_percentage' in metrics:
                        result['utilization']['pod_percentage'] = metrics['pod_percentage']
                    elif 'pod_allocation_ratio' in metrics:
                        result['utilization']['pod_percentage'] = metrics['pod_allocation_ratio'] * 100
                
                return result
                
            except Exception as e:
                logging.error(f"Error during LLM resource analysis: {e}")
                # Return a minimal compatible structure when LLM analysis fails
                return {
                    "summary": "LLM analysis failed: " + str(e),
                    "insights": ["Error occurred during LLM resource analysis"],
                    "recommendations": ["Check logs for detailed error information"],
                    "risks": [],
                    "metrics": {},
                    "utilization": {
                        "cpu_request": 0.0,
                        "cpu_limit": 0.0,
                        "memory_request": 0.0,
                        "memory_limit": 0.0,
                        "pods": 0.0,
                        "cpu_request_percentage": 0.0,
                        "memory_request_percentage": 0.0,
                        "pod_percentage": 0.0
                    }
                }
                
        except Exception as e:
            logging.error(f"Failed to collect node resource data: {e}")
            # Return a minimal compatible structure when node data collection fails
            return {
                "summary": "Failed to collect node resource data: " + str(e),
                "insights": ["Error occurred during node data collection"],
                "recommendations": ["Check Kubernetes cluster connectivity", "Verify permissions to access node data"],
                "risks": [],
                "metrics": {},
                "utilization": {
                    "cpu_request": 0.0,
                    "cpu_limit": 0.0,
                    "memory_request": 0.0,
                    "memory_limit": 0.0,
                    "pods": 0.0,
                    "cpu_request_percentage": 0.0,
                    "memory_request_percentage": 0.0,
                    "pod_percentage": 0.0
                }
            }
    
    def analyze_pod_distribution(self) -> Dict:
        """
        Analyze pod distribution across nodes.
        
        Returns:
            Dict with pod distribution analysis
        """
        # Get all nodes
        nodes = self.k8s_client.list_nodes()
        
        # Get pods per node
        pods_per_node = {}
        for node in nodes:
            node_name = node.get("metadata", {}).get("name", "unknown")
            pods = self.k8s_client.list_pods_on_node(node_name)
            pods_per_node[node_name] = len(pods)
        
        # Calculate statistics
        total_pods = sum(pods_per_node.values())
        nodes_count = len(pods_per_node)
        
        if nodes_count > 0:
            avg_pods = total_pods / nodes_count
            max_pods = max(pods_per_node.values()) if pods_per_node else 0
            min_pods = min(pods_per_node.values()) if pods_per_node else 0
            
            # Calculate standard deviation to measure balance
            variance = sum((pods - avg_pods) ** 2 for pods in pods_per_node.values()) / nodes_count
            std_dev = variance ** 0.5
            
            # Calculate balance score (lower std_dev/avg_pods ratio means better balance)
            if avg_pods > 0:
                balance_score = 1 - min(1, std_dev / avg_pods)
            else:
                balance_score = 1
        else:
            avg_pods = 0
            max_pods = 0
            min_pods = 0
            std_dev = 0
            balance_score = 1
        
        # Identify issues
        issues = []
        
        if nodes_count > 1 and balance_score < 0.7:
            issues.append({
                "issue": "pod_distribution_imbalance",
                "description": "Pods are not evenly distributed across nodes",
                "severity": "medium",
                "balance_score": balance_score
            })
        
        return {
            "total_pods": total_pods,
            "nodes_count": nodes_count,
            "pods_per_node": pods_per_node,
            "avg_pods": avg_pods,
            "max_pods": max_pods,
            "min_pods": min_pods,
            "std_deviation": std_dev,
            "balance_score": balance_score,
            "issues": issues
        }
    
    def analyze_networking(self) -> Dict:
        """
        Analyze cluster networking.
        
        Returns:
            Dict with networking analysis
        """
        # Placeholder for actual network analysis
        # This would involve checking network policies, service meshes, etc.
        return {
            "issues": []
        }
    
    def analyze_storage(self) -> Dict:
        """
        Analyze cluster storage.
        
        Returns:
            Dict with storage analysis
        """
        # Get persistent volumes
        pvs = self.k8s_client.list_persistent_volumes()
        
        # Get persistent volume claims
        pv_claims = []
        for namespace in self.k8s_client.list_namespaces():
            pv_claims.extend(self.k8s_client.list_persistent_volume_claims(namespace))
        
        # Analyze volume status
        volume_statuses = {
            "available": 0,
            "bound": 0,
            "released": 0,
            "failed": 0
        }
        
        for pv in pvs:
            status = pv.get("status", {}).get("phase")
            if status in volume_statuses:
                volume_statuses[status.lower()] += 1
        
        # Identify issues
        issues = []
        
        if volume_statuses["failed"] > 0:
            issues.append({
                "issue": "failed_persistent_volumes",
                "description": f"{volume_statuses['failed']} persistent volumes are in Failed state",
                "severity": "high"
            })
        
        return {
            "persistent_volumes": len(pvs),
            "persistent_volume_claims": len(pv_claims),
            "volume_statuses": volume_statuses,
            "issues": issues
        }
    
    def _is_pod_ready(self, pod: Dict) -> bool:
        """Check if a pod is in the Ready state."""
        status = pod.get("status", {})
        
        # Check pod phase
        phase = status.get("phase")
        if phase != "Running":
            return False
        
        # Check ready condition
        for condition in status.get("conditions", []):
            if condition.get("type") == "Ready":
                return condition.get("status") == "True"
        
        return False
    
    def _parse_cpu(self, cpu_str: str) -> int:
        """Parse CPU string to millicores."""
        try:
            if cpu_str.endswith("m"):
                return int(cpu_str[:-1])
            else:
                return int(float(cpu_str) * 1000)
        except (ValueError, TypeError):
            return 0
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes."""
        try:
            if memory_str.endswith("Ki"):
                return int(memory_str[:-2]) * 1024
            elif memory_str.endswith("Mi"):
                return int(memory_str[:-2]) * 1024 * 1024
            elif memory_str.endswith("Gi"):
                return int(memory_str[:-2]) * 1024 * 1024 * 1024
            elif memory_str.endswith("Ti"):
                return int(memory_str[:-2]) * 1024 * 1024 * 1024 * 1024
            elif memory_str.endswith("K") or memory_str.endswith("k"):
                return int(memory_str[:-1]) * 1000
            elif memory_str.endswith("M"):
                return int(memory_str[:-1]) * 1000 * 1000
            elif memory_str.endswith("G"):
                return int(memory_str[:-1]) * 1000 * 1000 * 1000
            elif memory_str.endswith("T"):
                return int(memory_str[:-1]) * 1000 * 1000 * 1000 * 1000
            else:
                return int(memory_str)
        except (ValueError, TypeError):
            return 0
    
    def _calculate_health_score(self, results: Dict) -> float:
        """Calculate overall cluster health score."""
        # Define weights for different components
        weights = {
            "nodes": 0.3,
            "control_plane": 0.3,
            "resource_usage": 0.2,
            "pod_distribution": 0.1,
            "networking": 0.05,
            "storage": 0.05
        }
        
        scores = {
            # Node score based on percentage of ready nodes
            "nodes": min(100, results["nodes"]["health_percentage"]) / 100,
            
            # Control plane score
            "control_plane": self._calculate_control_plane_score(results["control_plane"]),
            
            # Resource usage score (lower utilization is better, but not too low)
            "resource_usage": self._calculate_resource_usage_score(results["resource_usage"]),
            
            # Pod distribution score
            "pod_distribution": results["pod_distribution"].get("balance_score", 1.0),
            
            # Networking score (placeholder)
            "networking": 1.0 if not results["networking"]["issues"] else 0.5,
            
            # Storage score
            "storage": self._calculate_storage_score(results["storage"])
        }
        
        # Calculate weighted average
        weighted_score = sum(scores[component] * weight for component, weight in weights.items())
        
        # Convert to 0-100 scale
        return round(weighted_score * 100, 1)
    
    def _calculate_control_plane_score(self, control_plane_results: Dict) -> float:
        """Calculate health score for control plane components."""
        components = control_plane_results["components"]
        
        # If any component is missing entirely, score is 0
        for component, status in components.items():
            if status["total"] == 0:
                return 0
        
        # Average health percentage across components
        health_percentages = [status["health_percentage"] for status in components.values()]
        return sum(health_percentages) / len(health_percentages) / 100
    
    def _calculate_resource_usage_score(self, resource_results: Dict) -> float:
        """Calculate health score for resource usage."""
        utilization = resource_results["utilization"]
        
        # Ideal CPU utilization is between 50-80%
        cpu_score = 1.0
        cpu_util = utilization["cpu_request_percentage"]
        if cpu_util > 90:
            cpu_score = 0.3
        elif cpu_util > 80:
            cpu_score = 0.6
        elif cpu_util < 20:
            cpu_score = 0.7  # Underutilization is also not ideal
        
        # Ideal memory utilization is between 60-85%
        memory_score = 1.0
        mem_util = utilization["memory_request_percentage"]
        if mem_util > 90:
            memory_score = 0.3
        elif mem_util > 85:
            memory_score = 0.7
        elif mem_util < 30:
            memory_score = 0.8  # Underutilization is also not ideal
        
        # Pod utilization should ideally be below 80%
        pod_score = 1.0
        pod_util = utilization["pod_percentage"]
        if pod_util > 90:
            pod_score = 0.3
        elif pod_util > 80:
            pod_score = 0.7
        
        # Weighted average (CPU and memory are more important)
        return 0.4 * cpu_score + 0.4 * memory_score + 0.2 * pod_score
    
    def _calculate_storage_score(self, storage_results: Dict) -> float:
        """Calculate health score for storage."""
        # If there are failed volumes, reduce score
        failed = storage_results["volume_statuses"].get("failed", 0)
        if failed > 0:
            return max(0, 1 - (failed * 0.2))
        
        return 1.0
    
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string."""
        if health_score >= 90:
            return "healthy"
        elif health_score >= 70:
            return "warning"
        elif health_score >= 50:
            return "degraded"
        else:
            return "critical"
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Collect all issues
        all_issues = []
        all_issues.extend(results["nodes"].get("issues", []))
        all_issues.extend(results["control_plane"].get("issues", []))
        all_issues.extend(results["resource_usage"].get("issues", []))
        all_issues.extend(results["pod_distribution"].get("issues", []))
        all_issues.extend(results["networking"].get("issues", []))
        all_issues.extend(results["storage"].get("issues", []))
        
        # Generate recommendations for issues
        for issue in all_issues:
            issue_type = issue.get("issue", "")
            recommendation = self._get_recommendation_for_issue(issue_type, issue)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort recommendations by priority
        priority_levels = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_levels.get(x.get("priority", "low"), 4))
        
        return recommendations
    
    def _get_recommendation_for_issue(self, issue_type: str, issue_details: Dict) -> Optional[Dict]:
        """Get recommendation for a specific issue type."""
        recommendations = {
            "node_cordoned": {
                "title": "Investigate cordoned node",
                "description": f"Node {issue_details.get('node')} is cordoned. Investigate why it was manually marked as unschedulable.",
                "priority": "medium",
                "action_items": [
                    f"Check node {issue_details.get('node')} for issues",
                    "Uncordon if it's ready to receive workloads: `kubectl uncordon {}`"
                ]
            },
            "node_diskpressure": {
                "title": "Resolve disk pressure on node",
                "description": f"Node {issue_details.get('node')} is experiencing disk pressure.",
                "priority": "high",
                "action_items": [
                    "Clean up unused container images: `kubectl exec -it ... -- crictl rmi --prune`",
                    "Check for large log files or core dumps",
                    "Consider adding more disk space to the node"
                ]
            },
            "node_memorypressure": {
                "title": "Resolve memory pressure on node",
                "description": f"Node {issue_details.get('node')} is experiencing memory pressure.",
                "priority": "high",
                "action_items": [
                    "Review memory requests/limits for pods on this node",
                    "Check for memory leaks in applications",
                    "Consider adding more memory to the node or migrating workloads"
                ]
            },
            "control_plane_component_not_ready": {
                "title": "Restore control plane health",
                "description": issue_details.get("description", "Control plane component is not ready"),
                "priority": "critical",
                "action_items": [
                    f"Check logs for {issue_details.get('component')} component",
                    "Ensure etcd is healthy if applicable",
                    "Verify certificates haven't expired"
                ]
            },
            "high_cpu_request_utilization": {
                "title": "Address high CPU utilization",
                "description": "Cluster CPU request utilization is high at " + 
                               f"{round(issue_details.get('value', 0), 1)}%.",
                "priority": "high",
                "action_items": [
                    "Review workload CPU requests and adjust if overprovisioned",
                    "Consider adding more nodes to the cluster",
                    "Evaluate workload scheduling to optimize distribution"
                ]
            },
            "high_memory_request_utilization": {
                "title": "Address high memory utilization",
                "description": "Cluster memory request utilization is high at " + 
                               f"{round(issue_details.get('value', 0), 1)}%.",
                "priority": "high",
                "action_items": [
                    "Review workload memory requests and adjust if overprovisioned",
                    "Consider adding more nodes to the cluster",
                    "Check for memory leaks in applications"
                ]
            },
            "pod_distribution_imbalance": {
                "title": "Improve pod distribution",
                "description": "Pods are not evenly distributed across nodes (balance score: " + 
                               f"{round(issue_details.get('balance_score', 0), 2)}).",
                "priority": "medium",
                "action_items": [
                    "Review node selectors and affinities in workloads",
                    "Check for taints on nodes that might be preventing scheduling",
                    "Consider using topology spread constraints for better distribution"
                ]
            },
            "failed_persistent_volumes": {
                "title": "Address failed persistent volumes",
                "description": issue_details.get("description", "Some persistent volumes have failed"),
                "priority": "high",
                "action_items": [
                    "Check the status of storage provider/backend",
                    "Review PV/PVC events for error details: `kubectl describe pv <name>`",
                    "Ensure storage classes are correctly configured"
                ]
            }
        }
        
        if issue_type in recommendations:
            return {
                **recommendations[issue_type],
                "issue_type": issue_type,
                "issue_details": issue_details
            }
        
        return None
