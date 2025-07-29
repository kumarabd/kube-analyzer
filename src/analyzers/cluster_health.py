"""
Cluster Health Analyzer for KubeAnalyzer

This module uses an agentic LLM approach to analyze Kubernetes cluster health, 
resource utilization, node conditions, control plane status, and provides 
actionable recommendations for optimizing cluster performance and reliability.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field

import instructor
from openai import OpenAI
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, AgentMemory, BaseIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory

from ..utils.k8s_client import KubernetesClient

logger = logging.getLogger(__name__)


class ClusterHealthAnalyzer:
    """Analyzes Kubernetes cluster health using agentic LLM analysis."""

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
            namespaces: Optional list of namespaces to analyze
            
        Returns:
            Dict with analysis results
        """
        return self.analyze_cluster(namespaces)
        
    def analyze_cluster(self, namespaces=None) -> Dict:
        """
        Perform a comprehensive cluster health analysis using agentic LLM.
        
        Args:
            namespaces: Optional list of namespaces to analyze
            
        Returns:
            Dict with comprehensive analysis results
        """
        logger.info("Starting comprehensive cluster health analysis")
        
        # Perform agentic cluster analysis
        analysis_result = self.analyze_cluster_with_agent(namespaces)
        
        # Ensure we're returning a dictionary, not a string
        if isinstance(analysis_result, str):
            try:
                # Try to parse it as JSON if it's a string
                analysis_result = json.loads(analysis_result)
                logger.info("Successfully parsed analysis result string as JSON")
            except json.JSONDecodeError:
                logger.error("Analysis result is a string but not valid JSON")
                # Return a minimal compatible dictionary
                analysis_result = {
                    "summary": analysis_result,  # Use the string as summary
                    "insights": [],
                    "recommendations": [],
                    "risks": [],
                    "metrics": {},
                    "utilization": {
                        "cpu_request": 0.0,
                        "memory_request": 0.0,
                        "pods": 0.0
                    }
                }
                
        # Return the results directly from the agent's analysis
        return analysis_result
    
    def analyze_cluster_with_agent(self, namespaces=None) -> Dict:
        """
        Perform comprehensive cluster health analysis using an LLM agent.
        The agent performs an iterative analysis session to provide insights
        about overall cluster health, resource utilization, node conditions,
        and other critical aspects of Kubernetes cluster operations.
        
        Args:
            namespaces: Optional list of namespaces to analyze
            
        Returns:
            Dict with comprehensive cluster analysis from the LLM agent
        """

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
                risks: List[Any] = Field(description="Potential risks or issues identified - can be strings or dicts")
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
            system_prompt = f"""You are a Kubernetes cluster health analysis expert.
            Perform a comprehensive analysis of the Kubernetes cluster based on the provided data.
            
            Focus on these key areas:
            1. Node health and conditions (Ready status, pressure conditions, etc.)
            2. Resource utilization and efficiency (CPU, memory, pods)
            3. Control plane component health
            4. Pod distribution and scheduling
            5. Networking health and configuration
            6. Storage health and potential issues
            7. Overall cluster stability and reliability
            
            You can use tools to fetch additional data about the cluster to enhance your analysis.
            When calculating resource utilization percentages, be sure to include:
            - cpu_request_percentage: Percentage of total CPU requested vs available
            - memory_request_percentage: Percentage of total memory requested vs available
            - pod_percentage: Percentage of total pods used vs maximum allowed
            
            Provide actionable insights and recommendations to improve cluster health,
            optimize resource utilization, and address any identified issues or risks.
            
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
                query="Perform a comprehensive analysis of the entire cluster health. Evaluate node status, control plane health, resource utilization, pod distribution, networking, and storage. Identify any issues, provide actionable recommendations, and calculate key health metrics."
            )
            
            # Run the analysis
            try:
                # Execute the agent with the input
                agent_result = agent.run(agent_input)
                logging.info("Agentic LLM resource analysis completed successfully")
                
                # Convert agent result to format expected by the application
                # Normalize risks - if any risks are strings, convert them to dictionaries
                normalized_risks = []
                for risk in agent_result.risks:
                    if isinstance(risk, str):
                        normalized_risks.append({
                            "description": risk,
                            "severity": "medium", # Default severity
                            "affected_resources": ["cluster"]
                        })
                    else:
                        normalized_risks.append(risk)
                
                result = {
                    "summary": agent_result.summary,
                    "insights": agent_result.insights,
                    "recommendations": agent_result.recommendations,
                    "risks": normalized_risks,
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
                
                print(result)
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
