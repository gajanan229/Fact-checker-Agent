"""
Core building blocks for the Veritas AI fact-checking engine: state types,
validation, the safe state-update helper, and the LangGraph wrappers.
"""

from .graph import (
    CompiledVeritasGraph,
    GraphError,
    VeritasGraph,
)
from .serialization import safe_update_state
from .state import (
    Claim,
    ClaimStatus,
    GraphState,
    RawContent,
    Source,
    SourceType,
    StateUpdate,
    UserInput,
)
from .validation import (
    StateValidator,
    ValidationError,
    create_initial_state,
    validate_state_update,
)

__all__ = [
    # State definitions
    "GraphState", "Claim", "Source", "UserInput", "RawContent",
    "ClaimStatus", "SourceType", "StateUpdate",

    # Validation
    "StateValidator", "ValidationError", "create_initial_state", "validate_state_update",

    # State updates
    "safe_update_state",

    # Graph foundation
    "VeritasGraph", "CompiledVeritasGraph", "GraphError",
]
