"""
LangGraph foundation for the Veritas AI fact-checking engine.

Provides the :class:`VeritasGraph` builder, the :class:`CompiledVeritasGraph`
wrapper, the :class:`GraphExecutionManager` retry layer, and the shared
``error_handler_node``. The runtime workflow is wired up in
:mod:`.autonomous_graph`; this module only exposes the building blocks.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .serialization import safe_update_state
from .state import GraphState
from .validation import StateValidator, ValidationError


class GraphError(Exception):
    """Raised for graph build/compile/execution failures."""


class NodeRegistrationError(Exception):
    """Raised when a node fails to register."""


class GraphTimeoutError(Exception):
    """Raised when a graph operation exceeds its timeout."""


class GraphLoopError(Exception):
    """Raised when the execution manager detects an infinite loop."""


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _check_infinite_loop(state: GraphState) -> bool:
    """Return True if the current node has been visited more than five times."""
    try:
        debug_info = state.get("debug_info", {})
        node_visits = debug_info.get("node_visits", {})
        current_step = state.get("status", {}).get("current_step", "")

        if current_step:
            return node_visits.get(current_step, 0) > 5
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Execution manager
# ---------------------------------------------------------------------------

class GraphExecutionManager:
    """Runs node callables with retries, loop detection, and timeouts."""

    def __init__(self, max_retries: int = 3, timeout_seconds: float = 600):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.retry_delays = [1, 2, 4]  # Exponential backoff (seconds)

    async def execute_with_timeout(self, coro, timeout_seconds: Optional[float] = None):
        """Await ``coro`` with a hard timeout, raising :class:`GraphTimeoutError`."""
        timeout = timeout_seconds or self.timeout_seconds
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise GraphTimeoutError(f"Operation timed out after {timeout} seconds")

    def execute_node_with_retry(
        self,
        node_func: Callable,
        state: GraphState,
        node_name: str,
    ) -> Union[Dict[str, Any], Command]:
        """Execute ``node_func`` with retry, loop detection, and visit tracking."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if _check_infinite_loop(state):
                    raise GraphLoopError(f"Infinite loop detected in node '{node_name}'")

                debug_info = state.get("debug_info", {})
                node_visits = debug_info.get("node_visits", {})
                node_visits[node_name] = node_visits.get(node_name, 0) + 1

                state = safe_update_state(
                    state,
                    {"debug_info": {**debug_info, "node_visits": node_visits}},
                    validate=False,
                )

                result = node_func(state)

                if isinstance(result, dict) and "debug_info" in result:
                    result["debug_info"]["retry_count"] = 0

                return result

            except (ValidationError, GraphLoopError, GraphTimeoutError):
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries and attempt < len(self.retry_delays):
                    import time
                    time.sleep(self.retry_delays[attempt])
                else:
                    break

        return {
            "error_message": (
                f"Node '{node_name}' failed after {self.max_retries + 1} attempts: "
                f"{last_exception}"
            ),
            "workflow_stage": "failed",
            "debug_info": {
                "failed_node": node_name,
                "retry_count": self.max_retries + 1,
                "last_error": str(last_exception),
            },
        }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class VeritasGraph:
    """High-level builder around :class:`langgraph.graph.StateGraph`."""

    def __init__(self, use_checkpointer: bool = True):
        self.builder = StateGraph(GraphState)
        self.nodes: Dict[str, Any] = {}
        self.edges: List[Dict[str, Any]] = []
        self.conditional_edges: List[Dict[str, Any]] = []
        self.compiled_graph = None

        self.checkpointer = MemorySaver() if use_checkpointer else None
        self.execution_manager = GraphExecutionManager()

        self.metadata: Dict[str, Any] = {
            "graph_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
            "nodes_count": 0,
            "edges_count": 0,
        }

    def add_node(
        self,
        name: str,
        func: Callable[[GraphState], Union[Dict[str, Any], Command]],
        description: Optional[str] = None,
    ) -> "VeritasGraph":
        """Register a node, wrapping it with retry/error handling."""
        if name in self.nodes:
            raise NodeRegistrationError(f"Node '{name}' already registered")
        if not callable(func):
            raise NodeRegistrationError(f"Node function for '{name}' must be callable")
        if not self._validate_node_function(func):
            raise NodeRegistrationError(
                f"Node function for '{name}' must accept GraphState and return Dict or Command"
            )

        wrapped_func = self._wrap_node_function(name, func, description)
        self.builder.add_node(name, wrapped_func)

        self.nodes[name] = {
            "function": func,
            "wrapped_function": wrapped_func,
            "description": description,
            "registered_at": datetime.utcnow().isoformat() + "Z",
        }
        self.metadata["nodes_count"] += 1
        return self

    def add_edge(self, from_node: str, to_node: str) -> "VeritasGraph":
        """Add an unconditional edge between two registered nodes."""
        self.builder.add_edge(from_node, to_node)
        self.edges.append({
            "from": from_node,
            "to": to_node,
            "type": "direct",
            "added_at": datetime.utcnow().isoformat() + "Z",
        })
        self.metadata["edges_count"] += 1
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition_func: Callable[[GraphState], str],
        path_map: Dict[str, str],
        description: Optional[str] = None,
    ) -> "VeritasGraph":
        """Add a conditional edge that dispatches based on ``condition_func``."""
        wrapped_condition = self._wrap_condition_function(from_node, condition_func)
        self.builder.add_conditional_edges(from_node, wrapped_condition, path_map)

        self.conditional_edges.append({
            "from": from_node,
            "condition": condition_func,
            "path_map": path_map,
            "description": description,
            "added_at": datetime.utcnow().isoformat() + "Z",
        })
        self.metadata["edges_count"] += 1
        return self

    def set_entry_point(self, node_name: str) -> "VeritasGraph":
        """Mark ``node_name`` as the START -> entry edge."""
        self.builder.add_edge(START, node_name)
        self.edges.append({
            "from": START,
            "to": node_name,
            "added_at": datetime.utcnow().isoformat() + "Z",
        })
        return self

    def set_exit_point(self, node_name: str) -> "VeritasGraph":
        """Mark ``node_name`` -> END as a terminal edge."""
        self.builder.add_edge(node_name, END)
        self.edges.append({
            "from": node_name,
            "to": END,
            "added_at": datetime.utcnow().isoformat() + "Z",
        })
        return self

    def compile(self, validate: bool = True) -> "CompiledVeritasGraph":
        """Compile the graph and wrap it as a :class:`CompiledVeritasGraph`."""
        if validate:
            errors = self._validate_graph_structure()
            if errors:
                raise GraphError(f"Graph validation failed: {'; '.join(errors)}")

        try:
            if self.checkpointer:
                compiled = self.builder.compile(checkpointer=self.checkpointer)
            else:
                compiled = self.builder.compile()

            self.compiled_graph = compiled
            self.metadata["compiled_at"] = datetime.utcnow().isoformat() + "Z"
            self.metadata["status"] = "compiled"

            return CompiledVeritasGraph(
                compiled,
                self.metadata,
                self.nodes,
                self.checkpointer,
                self.execution_manager,
            )
        except Exception as e:
            raise GraphError(f"Failed to compile graph: {e}")

    @staticmethod
    def _validate_node_function(func: Callable) -> bool:
        """Cheap arity check for node callables."""
        import inspect
        try:
            sig = inspect.signature(func)
            return len(sig.parameters) >= 1
        except Exception:
            return False

    def _wrap_node_function(
        self,
        name: str,
        func: Callable,
        description: Optional[str],
    ) -> Callable:
        """Wrap ``func`` so failures resolve into a structured error state."""

        def wrapped_node(state: GraphState) -> Union[Dict[str, Any], Command]:
            try:
                return self.execution_manager.execute_node_with_retry(func, state, name)
            except (GraphLoopError, GraphTimeoutError) as e:
                return {
                    "error_message": str(e),
                    "workflow_stage": "failed",
                    "debug_info": {"failed_node": name, "error_type": type(e).__name__},
                }
            except Exception as e:
                return {
                    "error_message": f"Unexpected error in node '{name}': {e}",
                    "workflow_stage": "failed",
                    "debug_info": {"failed_node": name, "error_type": type(e).__name__},
                }

        wrapped_node.__name__ = f"wrapped_{name}"
        wrapped_node.__doc__ = description or func.__doc__
        return wrapped_node

    @staticmethod
    def _wrap_condition_function(from_node: str, func: Callable) -> Callable:
        """Wrap a routing function so any failure routes to ``"error"``."""

        def wrapped_condition(state: GraphState) -> str:
            try:
                result = func(state)
                return result if isinstance(result, str) else "error"
            except Exception:
                return "error"

        return wrapped_condition

    def _validate_graph_structure(self) -> List[str]:
        """Return a list of structural problems with the current graph."""
        errors: List[str] = []

        if not self.nodes:
            errors.append("Graph must have at least one node")

        has_entry = any(edge["from"] == START for edge in self.edges)
        if not has_entry:
            errors.append("Graph must have an entry point (use set_entry_point)")

        has_exit = any(edge["to"] == END for edge in self.edges)
        if not has_exit:
            errors.append("Graph must have an exit point (use set_exit_point)")

        for edge in self.edges:
            if edge["from"] != START and edge["from"] not in self.nodes:
                errors.append(f"Edge references unknown node: {edge['from']}")
            if edge["to"] != END and edge["to"] not in self.nodes:
                errors.append(f"Edge references unknown node: {edge['to']}")

        return errors

    def get_graph_info(self) -> Dict[str, Any]:
        """Return a snapshot of the current graph configuration."""
        return {
            "metadata": self.metadata,
            "nodes": list(self.nodes.keys()),
            "edges": len(self.edges),
            "conditional_edges": len(self.conditional_edges),
            "is_compiled": self.compiled_graph is not None,
        }


# ---------------------------------------------------------------------------
# Compiled graph wrapper
# ---------------------------------------------------------------------------

class CompiledVeritasGraph:
    """Thin wrapper around a compiled LangGraph adding execution bookkeeping."""

    def __init__(
        self,
        compiled_graph,
        metadata: Dict[str, Any],
        nodes: Dict[str, Any],
        checkpointer,
        execution_manager: GraphExecutionManager,
    ):
        self.graph = compiled_graph
        self.metadata = metadata
        self.nodes = nodes
        self.checkpointer = checkpointer
        self.execution_manager = execution_manager
        self.execution_history: List[Dict[str, Any]] = []

    def invoke(
        self,
        initial_state: GraphState,
        config: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """Run the graph synchronously and record an execution entry."""
        execution_record: Dict[str, Any] = {}
        try:
            errors = StateValidator.validate_graph_state(initial_state)
            if errors:
                raise GraphError(f"Invalid initial state: {'; '.join(errors)}")

            execution_record = {
                "execution_id": str(uuid.uuid4()),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "initial_stage": initial_state.get("workflow_stage"),
                "config": config,
            }

            result = self.graph.invoke(initial_state, config=config)

            execution_record.update({
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "final_stage": result.get("workflow_stage"),
                "status": "completed",
            })
            self.execution_history.append(execution_record)
            return result

        except Exception as e:
            if execution_record:
                execution_record.update({
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "status": "failed",
                    "error": str(e),
                })
                self.execution_history.append(execution_record)
            raise GraphError(f"Graph execution failed: {e}")

    def stream(
        self,
        initial_state: GraphState,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Stream LangGraph chunks, validating the initial state first."""
        try:
            errors = StateValidator.validate_graph_state(initial_state)
            if errors:
                raise GraphError(f"Invalid initial state: {'; '.join(errors)}")
            for chunk in self.graph.stream(initial_state, config=config):
                yield chunk
        except Exception as e:
            raise GraphError(f"Graph streaming failed: {e}")

    def get_state(self, config: Dict[str, Any]) -> Optional[GraphState]:
        """Return the checkpointed state for ``config``, or ``None``."""
        if not self.checkpointer:
            return None
        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception:
            return None

    def update_state(self, config: Dict[str, Any], values: Dict[str, Any]) -> None:
        """Apply a state update against the checkpointer."""
        if not self.checkpointer:
            raise GraphError("Cannot update state without checkpointer")
        try:
            self.graph.update_state(config, values)
        except Exception as e:
            raise GraphError(f"Failed to update state: {e}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the execution history."""
        return self.execution_history.copy()


# ---------------------------------------------------------------------------
# Shared error-handler node
# ---------------------------------------------------------------------------

def error_handler_node(state: GraphState) -> Dict[str, Any]:
    """Terminal node that surfaces a user-facing failure response."""
    error_message = state.get("error_message", "Unknown error occurred")
    debug_info = state.get("debug_info", {})

    return {
        "workflow_stage": "failed",
        "final_response": f"Sorry, an error occurred during processing: {error_message}",
        "error_message": error_message,
        "debug_info": {
            **debug_info,
            "handled_at": datetime.utcnow().isoformat() + "Z",
            "error_handler_triggered": True,
        },
    }
