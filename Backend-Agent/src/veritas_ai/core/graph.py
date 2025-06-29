"""
LangGraph foundation for the Veritas AI fact-checking engine.

This module provides the core graph setup, node registration patterns,
conditional edge management, and compilation utilities.
"""

import uuid
import asyncio
from typing import Dict, Any, List, Callable, Optional, Literal, Union
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send

from .state import GraphState, MinimalGraphState, ClaimStatus
from .validation import StateValidator, ValidationError
from .serialization import safe_update_state


class GraphError(Exception):
    """Custom exception for graph-related errors."""
    pass


class NodeRegistrationError(Exception):
    """Custom exception for node registration errors."""
    pass


class GraphTimeoutError(Exception):
    """Custom exception for graph execution timeouts."""
    pass


class GraphLoopError(Exception):
    """Custom exception for infinite loop detection."""
    pass


# === CONDITIONAL EDGE ROUTING FUNCTIONS ===

def should_continue_processing(state: GraphState) -> Literal["continue", "complete", "error"]:
    """
    Central routing function to determine if workflow should continue.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node to route to: continue, complete, or error
    """
    try:
        # Check for critical errors
        if state.get("error_message"):
            return "error"
        
        # Check if we've reached completion criteria
        if state.get("workflow_stage") == "completed":
            return "complete"
        
        # Check maximum processing attempts
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        
        if revision_count >= max_revisions:
            return "complete"
        
        # Check for processing timeouts
        if _check_processing_timeout(state):
            return "error"
        
        return "continue"
        
    except Exception:
        return "error"


def route_content_processing(state: GraphState) -> Literal["identify_claims", "error", "human_interaction"]:
    """
    Route after content ingestion based on extraction success.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node: identify_claims, error, or human_interaction
    """
    try:
        raw_content = state.get("raw_content", {})
        
        # Check if we have any extractable content
        has_transcript = bool(raw_content.get("transcript", "").strip())
        has_ocr = bool(raw_content.get("ocr_text", "").strip())
        has_comment = bool(raw_content.get("selected_comment_text", "").strip())
        
        # If no content was extracted, route to human interaction for manual input
        if not (has_transcript or has_ocr or has_comment):
            return "human_interaction"
        
        # Check for extraction errors
        extraction_errors = raw_content.get("extraction_errors", [])
        if extraction_errors and len(extraction_errors) > 2:  # Too many errors
            return "error"
        
        return "identify_claims"
        
    except Exception:
        return "error"


def route_claims_analysis(state: GraphState) -> Literal["research_claims", "human_interaction", "generate_response"]:
    """
    Route after claim identification based on claims found.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node: research_claims, human_interaction, or generate_response
    """
    try:
        claims = state.get("claims", [])
        
        # No claims found - need human input to identify claims manually
        if not claims:
            return "human_interaction"
        
        # Check if all claims are already verified or unverifiable
        verified_statuses = {ClaimStatus.VERIFIED, ClaimStatus.DEBUNKED, ClaimStatus.UNVERIFIABLE}
        all_processed = all(claim.get("status") in verified_statuses for claim in claims)
        
        if all_processed:
            return "generate_response"
        
        return "research_claims"
        
    except Exception:
        return "human_interaction"


def route_after_research(state: GraphState) -> Literal["generate_response", "human_interaction", "research_claims"]:
    """
    Route after evidence research based on research quality.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node: generate_response, human_interaction, or research_claims
    """
    try:
        claims = state.get("claims", [])
        
        if not claims:
            return "human_interaction"
        
        # Check research quality
        well_researched_count = 0
        needs_more_research = 0
        
        for claim in claims:
            sources = claim.get("sources", [])
            if len(sources) >= 2:  # At least 2 sources
                avg_credibility = sum(s.get("credibility_score", 0.5) for s in sources) / len(sources)
                if avg_credibility >= 0.7:
                    well_researched_count += 1
                else:
                    needs_more_research += 1
            else:
                needs_more_research += 1
        
        # If most claims need more research, continue research
        if needs_more_research > well_researched_count:
            # But don't loop forever - check attempt count
            research_attempts = state.get("debug_info", {}).get("research_attempts", 0)
            if research_attempts >= 2:
                return "human_interaction"  # Ask for human guidance
            return "research_claims"
        
        return "generate_response"
        
    except Exception:
        return "generate_response"  # Proceed with what we have


def route_after_generation(state: GraphState) -> Literal["critique_response", "human_interaction", "complete"]:
    """
    Route after response generation based on content quality.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node: critique_response, human_interaction, or complete
    """
    try:
        draft_response = state.get("draft_response", "")
        
        # No response generated - need human help
        if not draft_response.strip():
            return "human_interaction"
        
        # Check if response is too short (likely incomplete)
        if len(draft_response.strip()) < 50:
            return "human_interaction"
        
        # Check revision count
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        
        # If we've already done maximum revisions, complete
        if revision_count >= max_revisions:
            return "complete"
        
        return "critique_response"
        
    except Exception:
        return "complete"


def route_after_critique(state: GraphState) -> Literal["generate_response", "human_interaction", "complete"]:
    """
    Route after adversarial review based on critique results.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node: generate_response, human_interaction, or complete
    """
    try:
        critique = state.get("critique", {})
        
        # No critique - something went wrong, complete with what we have
        if not critique:
            return "complete"
        
        revision_needed = critique.get("is_revision_needed", False)
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        
        # Check if we've hit revision limit
        if revision_count >= max_revisions:
            return "complete"
        
        # If revision needed and we can still revise
        if revision_needed:
            # Check quality assessment to decide if human input is needed
            quality = critique.get("quality_assessment", {})
            overall_score = quality.get("overall_score", 0.5)
            
            # If quality is very low, get human help
            if overall_score < 0.3:
                return "human_interaction"
            
            return "generate_response"
        
        return "complete"
        
    except Exception:
        return "complete"


def route_after_human_interaction(state: GraphState) -> Literal["identify_claims", "research_claims", "generate_response", "complete", "error"]:
    """
    Route after human interaction based on user input and current stage.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node based on workflow stage and user input
    """
    try:
        workflow_stage = state.get("workflow_stage", "initialized")
        user_feedback_pending = state.get("user_feedback_pending", False)
        
        # If user feedback is still pending, stay in human interaction
        if user_feedback_pending:
            return "error"  # Shouldn't happen, but fallback
        
        # Route based on current workflow stage
        if workflow_stage in ["initialized", "content_extracted"]:
            return "identify_claims"
        elif workflow_stage == "claims_identified":
            return "research_claims"
        elif workflow_stage in ["evidence_gathered", "response_drafted", "response_reviewed"]:
            return "generate_response"
        elif workflow_stage == "completed":
            return "complete"
        else:
            return "error"
        
    except Exception:
        return "error"


# === LOOP CONTROL AND TERMINATION ===

def _check_processing_timeout(state: GraphState) -> bool:
    """Check if processing has exceeded timeout limits."""
    try:
        created_at = state.get("created_at")
        if not created_at:
            return False
        
        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        current_time = datetime.utcnow().replace(tzinfo=created_time.tzinfo)
        
        # Default timeout: 10 minutes for complete workflow
        timeout_minutes = state.get("config", {}).get("timeout_minutes", 10)
        timeout_delta = timedelta(minutes=timeout_minutes)
        
        return (current_time - created_time) > timeout_delta
        
    except Exception:
        return False


def _check_infinite_loop(state: GraphState) -> bool:
    """Check for potential infinite loops in processing."""
    try:
        debug_info = state.get("debug_info", {})
        
        # Track node visit counts
        node_visits = debug_info.get("node_visits", {})
        current_step = state.get("status", {}).get("current_step", "")
        
        if current_step:
            visit_count = node_visits.get(current_step, 0)
            return visit_count > 5  # More than 5 visits to same node indicates loop
        
        return False
        
    except Exception:
        return False


# === ERROR HANDLING AND RETRY PATTERNS ===

class GraphExecutionManager:
    """Enhanced execution manager with retry and fallback capabilities."""
    
    def __init__(self, max_retries: int = 3, timeout_seconds: float = 600):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.retry_delays = [1, 2, 4]  # Exponential backoff
    
    async def execute_with_timeout(self, coro, timeout_seconds: Optional[float] = None):
        """Execute a coroutine with timeout."""
        timeout = timeout_seconds or self.timeout_seconds
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise GraphTimeoutError(f"Operation timed out after {timeout} seconds")
    
    def execute_node_with_retry(self, node_func: Callable, state: GraphState, node_name: str) -> Union[Dict[str, Any], Command]:
        """Execute a node function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check for infinite loop before execution
                if _check_infinite_loop(state):
                    raise GraphLoopError(f"Infinite loop detected in node '{node_name}'")
                
                # Track node visits
                debug_info = state.get("debug_info", {})
                node_visits = debug_info.get("node_visits", {})
                node_visits[node_name] = node_visits.get(node_name, 0) + 1
                
                # Update state with visit tracking
                state = safe_update_state(state, {
                    "debug_info": {
                        **debug_info,
                        "node_visits": node_visits
                    }
                }, validate=False)
                
                # Execute the node
                result = node_func(state)
                
                # Successful execution - reset retry count for this node
                if isinstance(result, dict) and "debug_info" in result:
                    result["debug_info"]["retry_count"] = 0
                
                return result
                
            except (ValidationError, GraphLoopError, GraphTimeoutError):
                # Don't retry these exceptions
                raise
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Add delay before retry
                    if attempt < len(self.retry_delays):
                        import time
                        time.sleep(self.retry_delays[attempt])
                else:
                    # Final attempt failed
                    break
        
        # All retries exhausted
        return {
            "error_message": f"Node '{node_name}' failed after {self.max_retries + 1} attempts: {str(last_exception)}",
            "workflow_stage": "failed",
            "debug_info": {
                "failed_node": node_name,
                "retry_count": self.max_retries + 1,
                "last_error": str(last_exception)
            }
        }


# === ENHANCED GRAPH CLASS WITH ORCHESTRATION ===

class VeritasGraph:
    """
    Core LangGraph wrapper for the Veritas AI fact-checking engine.
    
    Provides a high-level interface for building, configuring, and executing
    the fact-checking workflow graph.
    """
    
    def __init__(self, use_checkpointer: bool = True):
        """
        Initialize the Veritas graph.
        
        Args:
            use_checkpointer: Whether to enable state persistence with checkpoints
        """
        self.builder = StateGraph(GraphState)
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[Dict[str, Any]] = []
        self.conditional_edges: List[Dict[str, Any]] = []
        self.compiled_graph = None
        
        # Set up checkpointer for state persistence
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        # Enhanced execution management
        self.execution_manager = GraphExecutionManager()
        
        # Graph metadata
        self.metadata = {
            "graph_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat() + 'Z',
            "version": "1.0.0",
            "nodes_count": 0,
            "edges_count": 0
        }
    
    def add_node(
        self, 
        name: str, 
        func: Callable[[GraphState], Union[Dict[str, Any], Command]], 
        description: Optional[str] = None
    ) -> 'VeritasGraph':
        """
        Register a node function with the graph.
        
        Args:
            name: Unique node identifier
            func: Node function that processes state
            description: Optional description for documentation
        
        Returns:
            Self for method chaining
        
        Raises:
            NodeRegistrationError: If node name already exists or function is invalid
        """
        if name in self.nodes:
            raise NodeRegistrationError(f"Node '{name}' already registered")
        
        if not callable(func):
            raise NodeRegistrationError(f"Node function for '{name}' must be callable")
        
        # Validate node function signature
        if not self._validate_node_function(func):
            raise NodeRegistrationError(
                f"Node function for '{name}' must accept GraphState and return Dict or Command"
            )
        
        # Wrap the function with error handling and logging
        wrapped_func = self._wrap_node_function(name, func, description)
        
        # Register with LangGraph builder
        self.builder.add_node(name, wrapped_func)
        
        # Track in our registry
        self.nodes[name] = {
            "function": func,
            "wrapped_function": wrapped_func,
            "description": description,
            "registered_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        self.metadata["nodes_count"] += 1
        
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'VeritasGraph':
        """
        Add a direct edge between two nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name
        
        Returns:
            Self for method chaining
        """
        self.builder.add_edge(from_node, to_node)
        
        self.edges.append({
            "from": from_node,
            "to": to_node,
            "type": "direct",
            "added_at": datetime.utcnow().isoformat() + 'Z'
        })
        
        self.metadata["edges_count"] += 1
        
        return self
    
    def add_conditional_edge(
        self,
        from_node: str,
        condition_func: Callable[[GraphState], str],
        path_map: Dict[str, str],
        description: Optional[str] = None
    ) -> 'VeritasGraph':
        """
        Add a conditional edge that routes based on state.
        
        Args:
            from_node: Source node name
            condition_func: Function that returns the next node name
            path_map: Mapping of condition results to node names
            description: Optional description for documentation
        
        Returns:
            Self for method chaining
        """
        # Wrap condition function with error handling
        wrapped_condition = self._wrap_condition_function(from_node, condition_func)
        
        self.builder.add_conditional_edges(
            from_node,
            wrapped_condition,
            path_map
        )
        
        self.conditional_edges.append({
            "from": from_node,
            "condition": condition_func,
            "path_map": path_map,
            "description": description,
            "added_at": datetime.utcnow().isoformat() + 'Z'
        })
        
        self.metadata["edges_count"] += 1
        
        return self
    
    def set_entry_point(self, node_name: str) -> 'VeritasGraph':
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the starting node
        
        Returns:
            Self for method chaining
        """
        self.builder.add_edge(START, node_name)
        
        # Track the edge in our metadata
        edge_info = {
            "from": START,
            "to": node_name,
            "added_at": datetime.utcnow().isoformat() + 'Z'
        }
        self.edges.append(edge_info)
        
        return self
    
    def set_exit_point(self, node_name: str) -> 'VeritasGraph':
        """
        Set an exit point for the graph.
        
        Args:
            node_name: Name of the ending node
        
        Returns:
            Self for method chaining
        """
        self.builder.add_edge(node_name, END)
        
        # Track the edge in our metadata
        edge_info = {
            "from": node_name,
            "to": END,
            "added_at": datetime.utcnow().isoformat() + 'Z'
        }
        self.edges.append(edge_info)
        
        return self
    
    def compile(self, validate: bool = True) -> 'CompiledVeritasGraph':
        """
        Compile the graph for execution.
        
        Args:
            validate: Whether to validate the graph structure
        
        Returns:
            Compiled graph ready for execution
        
        Raises:
            GraphError: If compilation fails or validation errors
        """
        if validate:
            validation_errors = self._validate_graph_structure()
            if validation_errors:
                raise GraphError(f"Graph validation failed: {'; '.join(validation_errors)}")
        
        try:
            # Compile the LangGraph
            if self.checkpointer:
                compiled = self.builder.compile(checkpointer=self.checkpointer)
            else:
                compiled = self.builder.compile()
            
            self.compiled_graph = compiled
            
            # Update metadata
            self.metadata["compiled_at"] = datetime.utcnow().isoformat() + 'Z'
            self.metadata["status"] = "compiled"
            
            return CompiledVeritasGraph(compiled, self.metadata, self.nodes, self.checkpointer, self.execution_manager)
            
        except Exception as e:
            raise GraphError(f"Failed to compile graph: {str(e)}")
    
    def _validate_node_function(self, func: Callable) -> bool:
        """Validate that a function can be used as a node."""
        import inspect
        
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Should accept at least one parameter (state)
            return len(params) >= 1
        except Exception:
            return False
    
    def _wrap_node_function(
        self, 
        name: str, 
        func: Callable, 
        description: Optional[str]
    ) -> Callable:
        """Wrap a node function with error handling and state validation."""
        
        def wrapped_node(state: GraphState) -> Union[Dict[str, Any], Command]:
            try:
                # Use execution manager for retry logic
                return self.execution_manager.execute_node_with_retry(func, state, name)
                
            except (GraphLoopError, GraphTimeoutError) as e:
                return {
                    "error_message": str(e),
                    "workflow_stage": "failed",
                    "debug_info": {
                        "failed_node": name,
                        "error_type": type(e).__name__
                    }
                }
            except Exception as e:
                return {
                    "error_message": f"Unexpected error in node '{name}': {str(e)}",
                    "workflow_stage": "failed",
                    "debug_info": {
                        "failed_node": name,
                        "error_type": type(e).__name__
                    }
                }
        
        # Preserve original function metadata
        wrapped_node.__name__ = f"wrapped_{name}"
        wrapped_node.__doc__ = description or func.__doc__
        
        return wrapped_node
    
    def _wrap_condition_function(
        self, 
        from_node: str, 
        func: Callable
    ) -> Callable:
        """Wrap a condition function with error handling."""
        
        def wrapped_condition(state: GraphState) -> str:
            try:
                result = func(state)
                if not isinstance(result, str):
                    # Fallback to error state
                    return "error"
                return result
            except Exception:
                # Fallback to error state
                return "error"
        
        return wrapped_condition
    
    def _validate_graph_structure(self) -> List[str]:
        """Validate the overall graph structure."""
        errors = []
        
        # Check if we have at least one node
        if len(self.nodes) == 0:
            errors.append("Graph must have at least one node")
        
        # Check for entry point
        has_entry = any(edge["from"] == START for edge in self.edges)
        if not has_entry:
            errors.append("Graph must have an entry point (use set_entry_point)")
        
        # Check for exit point
        has_exit = any(edge["to"] == END for edge in self.edges)
        if not has_exit:
            errors.append("Graph must have an exit point (use set_exit_point)")
        
        # Validate that all edge references exist
        for edge in self.edges:
            if edge["from"] != START and edge["from"] not in self.nodes:
                errors.append(f"Edge references unknown node: {edge['from']}")
            if edge["to"] != END and edge["to"] not in self.nodes:
                errors.append(f"Edge references unknown node: {edge['to']}")
        
        return errors
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the current graph configuration."""
        return {
            "metadata": self.metadata,
            "nodes": list(self.nodes.keys()),
            "edges": len(self.edges),
            "conditional_edges": len(self.conditional_edges),
            "is_compiled": self.compiled_graph is not None
        }


class CompiledVeritasGraph:
    """
    Wrapper for compiled LangGraph with Veritas-specific functionality.
    
    Provides execution methods, state management, and monitoring capabilities.
    """
    
    def __init__(
        self, 
        compiled_graph, 
        metadata: Dict[str, Any], 
        nodes: Dict[str, Any],
        checkpointer,
        execution_manager: GraphExecutionManager
    ):
        self.graph = compiled_graph
        self.metadata = metadata
        self.nodes = nodes
        self.checkpointer = checkpointer
        self.execution_manager = execution_manager
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
    
    def invoke(
        self, 
        initial_state: GraphState, 
        config: Optional[Dict[str, Any]] = None
    ) -> GraphState:
        """
        Execute the graph with an initial state.
        
        Args:
            initial_state: Starting state for the execution
            config: Optional configuration for execution
        
        Returns:
            Final state after graph execution
        """
        try:
            # Validate initial state
            validation_errors = StateValidator.validate_graph_state(initial_state)
            if validation_errors:
                raise GraphError(f"Invalid initial state: {'; '.join(validation_errors)}")
            
            # Record execution start
            execution_id = str(uuid.uuid4())
            execution_record = {
                "execution_id": execution_id,
                "started_at": datetime.utcnow().isoformat() + 'Z',
                "initial_stage": initial_state.get("workflow_stage"),
                "config": config
            }
            
            # Execute the graph
            result = self.graph.invoke(initial_state, config=config)
            
            # Record execution completion
            execution_record.update({
                "completed_at": datetime.utcnow().isoformat() + 'Z',
                "final_stage": result.get("workflow_stage"),
                "status": "completed"
            })
            
            self.execution_history.append(execution_record)
            
            return result
            
        except Exception as e:
            # Record execution failure
            if 'execution_record' in locals():
                execution_record.update({
                    "completed_at": datetime.utcnow().isoformat() + 'Z',
                    "status": "failed",
                    "error": str(e)
                })
                self.execution_history.append(execution_record)
            
            raise GraphError(f"Graph execution failed: {str(e)}")
    
    def stream(
        self, 
        initial_state: GraphState, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Stream graph execution with real-time updates.
        
        Args:
            initial_state: Starting state for the execution
            config: Optional configuration for execution
        
        Yields:
            State updates as they occur
        """
        try:
            # Validate initial state
            validation_errors = StateValidator.validate_graph_state(initial_state)
            if validation_errors:
                raise GraphError(f"Invalid initial state: {'; '.join(validation_errors)}")
            
            # Stream execution
            for chunk in self.graph.stream(initial_state, config=config):
                yield chunk
                
        except Exception as e:
            raise GraphError(f"Graph streaming failed: {str(e)}")
    
    def get_state(self, config: Dict[str, Any]) -> Optional[GraphState]:
        """
        Get the current state for a given configuration.
        
        Args:
            config: Configuration containing thread_id
        
        Returns:
            Current state or None if not found
        """
        if not self.checkpointer:
            return None
        
        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception:
            return None
    
    def update_state(
        self, 
        config: Dict[str, Any], 
        values: Dict[str, Any]
    ) -> None:
        """
        Update the state for a given configuration.
        
        Args:
            config: Configuration containing thread_id
            values: State updates to apply
        """
        if not self.checkpointer:
            raise GraphError("Cannot update state without checkpointer")
        
        try:
            self.graph.update_state(config, values)
        except Exception as e:
            raise GraphError(f"Failed to update state: {str(e)}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history for this graph."""
        return self.execution_history.copy()
    
    def visualize(self) -> Optional[str]:
        """
        Generate a visual representation of the graph.
        
        Returns:
            Graph visualization as Mermaid syntax or None if not available
        """
        try:
            # Generate Mermaid diagram
            mermaid = "graph TD;\n"
            
            # Add nodes
            for node_name in self.nodes.keys():
                safe_name = node_name.replace("-", "_")
                mermaid += f"    {safe_name}[{node_name}];\n"
            
            # Add START and END
            mermaid += "    START([START]);\n"
            mermaid += "    END([END]);\n"
            
            # Add edges (this would need to be implemented based on stored edge info)
            # For now, just return basic structure
            
            return mermaid
            
        except Exception:
            return None


# === COMPLETE WORKFLOW SETUP FUNCTION ===

def setup_complete_veritas_workflow() -> VeritasGraph:
    """
    Set up the complete Veritas fact-checking workflow with all conditional edges.
    
    Returns:
        Configured VeritasGraph ready for node registration and compilation
    """
    graph = VeritasGraph(use_checkpointer=True)
    
    # This will be used by the implementation to set up all the conditional edges
    # once all nodes are registered. The actual node registration will happen
    # in the calling code.
    
    return graph


def add_complete_workflow_edges(graph: VeritasGraph) -> VeritasGraph:
    """
    Add all conditional edges for the complete Veritas workflow.
    
    Args:
        graph: VeritasGraph with all nodes already registered
    
    Returns:
        Graph with all edges configured
    """
    # Entry point
    graph.set_entry_point("ingest_content")
    
    # Content processing flow
    graph.add_conditional_edge(
        "ingest_content",
        route_content_processing,
        {
            "identify_claims": "identify_claims",
            "human_interaction": "human_interaction",
            "error": "error_handler"
        },
        description="Route based on content extraction success"
    )
    
    # Claims analysis flow
    graph.add_conditional_edge(
        "identify_claims",
        route_claims_analysis,
        {
            "research_claims": "research_claims",
            "human_interaction": "human_interaction",
            "generate_response": "generate_response"
        },
        description="Route based on claims found"
    )
    
    # Research flow
    graph.add_conditional_edge(
        "research_claims",
        route_after_research,
        {
            "generate_response": "generate_response",
            "human_interaction": "human_interaction",
            "research_claims": "research_claims"  # Retry research
        },
        description="Route based on research quality"
    )
    
    # Response generation flow
    graph.add_conditional_edge(
        "generate_response",
        route_after_generation,
        {
            "critique_response": "critique_response",
            "human_interaction": "human_interaction",
            "complete": END
        },
        description="Route based on response quality"
    )
    
    # Quality control flow
    graph.add_conditional_edge(
        "critique_response",
        route_after_critique,
        {
            "generate_response": "generate_response",
            "human_interaction": "human_interaction",
            "complete": END
        },
        description="Route based on critique results"
    )
    
    # Human interaction flow
    graph.add_conditional_edge(
        "human_interaction",
        route_after_human_interaction,
        {
            "identify_claims": "identify_claims",
            "research_claims": "research_claims",
            "generate_response": "generate_response",
            "complete": END,
            "error": "error_handler"
        },
        description="Route based on user input and workflow stage"
    )
    
    # Error handling
    graph.add_edge("error_handler", END)
    
    return graph


# === ERROR HANDLER NODE ===

def error_handler_node(state: GraphState) -> Dict[str, Any]:
    """
    Simple error handler node that processes errors and sets final state.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with error handling
    """
    error_message = state.get("error_message", "Unknown error occurred")
    debug_info = state.get("debug_info", {})
    
    return {
        "workflow_stage": "failed",
        "final_response": f"Sorry, an error occurred during processing: {error_message}",
        "error_message": error_message,
        "debug_info": {
            **debug_info,
            "handled_at": datetime.utcnow().isoformat() + 'Z',
            "error_handler_triggered": True
        }
    }


# === UTILITY FUNCTIONS ===

# Utility functions for common graph patterns
def create_basic_veritas_graph() -> VeritasGraph:
    """Create a basic Veritas graph with common configuration."""
    graph = VeritasGraph(use_checkpointer=True)
    return graph


def create_minimal_graph() -> VeritasGraph:
    """Create a minimal graph for testing purposes."""
    graph = VeritasGraph(use_checkpointer=False)
    return graph 