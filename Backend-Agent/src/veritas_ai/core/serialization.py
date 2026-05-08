"""
State update helpers for the Veritas AI graph.

Historically this module also contained a JSON serializer, archiver, and
state minimizer. None of those were used by the running pipeline, so the
module has been pared back to the single helper that *is* used:
:func:`safe_update_state`.
"""

from datetime import datetime

from .state import GraphState, StateUpdate
from .validation import ValidationError, validate_state_update


def safe_update_state(
    current_state: GraphState,
    updates: StateUpdate,
    validate: bool = True,
) -> GraphState:
    """
    Merge ``updates`` into ``current_state`` and stamp ``last_updated``.

    Args:
        current_state: The current graph state.
        updates: Partial state to merge in.
        validate: When True (the default), reject updates that violate the
            state-validation rules in :mod:`.validation`.

    Returns:
        A new :class:`GraphState` dict with the updates applied.

    Raises:
        ValidationError: If ``validate`` is True and the update is invalid.
    """
    if validate:
        errors = validate_state_update(current_state, updates)
        if errors:
            raise ValidationError("state_update", "; ".join(errors))

    updates["last_updated"] = datetime.utcnow().isoformat() + "Z"
    return {**current_state, **updates}
