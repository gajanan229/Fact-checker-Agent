"""
Core components for the Veritas AI fact-checking engine.

This package contains:
- GraphState definitions and validation
- StateGraph setup and configuration
- Core utilities and base classes
"""

from .state import (
    GraphState, MinimalGraphState, Claim, Source, UserInput, RawContent,
    ClaimStatus, SourceType, StateUpdate
)
from .validation import (
    StateValidator, ValidationError, create_initial_state, validate_state_update
)
from .serialization import (
    StateSerializer, StateMinimizer, StateArchiver, 
    safe_update_state, extract_public_state
)
from .graph import (
    VeritasGraph, CompiledVeritasGraph, GraphError, 
    create_basic_veritas_graph, create_minimal_graph
)
from .visualization import (
    GraphVisualizer, visualize_veritas_workflow, export_graph_documentation
)

__all__ = [
    # State definitions
    "GraphState", "MinimalGraphState", "Claim", "Source", "UserInput", "RawContent",
    "ClaimStatus", "SourceType", "StateUpdate",
    
    # Validation
    "StateValidator", "ValidationError", "create_initial_state", "validate_state_update",
    
    # Serialization
    "StateSerializer", "StateMinimizer", "StateArchiver", 
    "safe_update_state", "extract_public_state",
    
    # Graph foundation
    "VeritasGraph", "CompiledVeritasGraph", "GraphError",
    "create_basic_veritas_graph", "create_minimal_graph",
    
    # Visualization
    "GraphVisualizer", "visualize_veritas_workflow", "export_graph_documentation"
] 