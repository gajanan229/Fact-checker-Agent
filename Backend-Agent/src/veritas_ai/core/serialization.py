"""
State serialization utilities for the Veritas AI fact-checking engine.

This module provides functions to serialize and deserialize GraphState objects
for persistence, API responses, and debugging.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
from dataclasses import asdict, is_dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .state import GraphState, MinimalGraphState, Claim, Source, StateUpdate
from .validation import ValidationError, StateValidator


class StateSerializationError(Exception):
    """Custom exception for state serialization errors."""
    pass


class StateSerializer:
    """Handles serialization and deserialization of GraphState objects."""
    
    @staticmethod
    def serialize_message(message: BaseMessage) -> Dict[str, Any]:
        """Serialize a LangChain message to dict."""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "id": getattr(message, 'id', None),
            "additional_kwargs": getattr(message, 'additional_kwargs', {}),
        }
    
    @staticmethod
    def deserialize_message(data: Dict[str, Any]) -> BaseMessage:
        """Deserialize a dict to LangChain message."""
        message_type = data.get("type", "HumanMessage")
        content = data.get("content", "")
        message_id = data.get("id")
        additional_kwargs = data.get("additional_kwargs", {})
        
        message_classes = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage,
        }
        
        MessageClass = message_classes.get(message_type, HumanMessage)
        
        kwargs = {
            "content": content,
            "additional_kwargs": additional_kwargs,
        }
        
        if message_id:
            kwargs["id"] = message_id
        
        return MessageClass(**kwargs)
    
    @staticmethod
    def to_dict(state: GraphState, include_debug: bool = False) -> Dict[str, Any]:
        """
        Convert GraphState to a serializable dictionary.
        
        Args:
            state: GraphState to serialize
            include_debug: Whether to include debug information
        
        Returns:
            Serializable dictionary representation
        """
        try:
            # Create a deep copy to avoid modifying original
            serialized = {}
            
            for key, value in state.items():
                if key == "chat_history":
                    # Handle LangChain messages specially
                    serialized[key] = [
                        StateSerializer.serialize_message(msg) for msg in value
                    ]
                elif key == "debug_info" and not include_debug:
                    # Skip debug info unless requested
                    continue
                elif isinstance(value, datetime):
                    # Handle datetime objects
                    serialized[key] = value.isoformat() + 'Z'
                elif isinstance(value, uuid.UUID):
                    # Handle UUID objects
                    serialized[key] = str(value)
                elif is_dataclass(value):
                    # Handle dataclass objects
                    serialized[key] = asdict(value)
                else:
                    # Handle all other types
                    serialized[key] = value
            
            return serialized
            
        except Exception as e:
            raise StateSerializationError(f"Failed to serialize state: {str(e)}")
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> GraphState:
        """
        Create GraphState from a dictionary.
        
        Args:
            data: Dictionary representation of GraphState
        
        Returns:
            GraphState object
        
        Raises:
            StateSerializationError: If deserialization fails
            ValidationError: If resulting state is invalid
        """
        try:
            # Handle special fields
            if "chat_history" in data:
                data["chat_history"] = [
                    StateSerializer.deserialize_message(msg_data)
                    for msg_data in data["chat_history"]
                ]
            
            # Validate the resulting state
            validation_errors = StateValidator.validate_graph_state(data)
            if validation_errors:
                raise ValidationError(
                    "deserialized_state",
                    "; ".join(validation_errors)
                )
            
            return data
            
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise StateSerializationError(f"Failed to deserialize state: {str(e)}")
    
    @staticmethod
    def to_json(state: GraphState, indent: Optional[int] = None, include_debug: bool = False) -> str:
        """
        Convert GraphState to JSON string.
        
        Args:
            state: GraphState to serialize
            indent: JSON indentation (None for compact)
            include_debug: Whether to include debug information
        
        Returns:
            JSON string representation
        """
        try:
            serialized = StateSerializer.to_dict(state, include_debug=include_debug)
            return json.dumps(serialized, indent=indent, ensure_ascii=False)
        except Exception as e:
            raise StateSerializationError(f"Failed to serialize state to JSON: {str(e)}")
    
    @staticmethod
    def from_json(json_str: str) -> GraphState:
        """
        Create GraphState from JSON string.
        
        Args:
            json_str: JSON string representation
        
        Returns:
            GraphState object
        """
        try:
            data = json.loads(json_str)
            return StateSerializer.from_dict(data)
        except json.JSONDecodeError as e:
            raise StateSerializationError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise StateSerializationError(f"Failed to deserialize state from JSON: {str(e)}")


class StateMinimizer:
    """Utilities for creating minimal state representations."""
    
    @staticmethod
    def to_minimal(state: GraphState) -> MinimalGraphState:
        """
        Extract minimal state for lightweight operations.
        
        Args:
            state: Full GraphState
        
        Returns:
            MinimalGraphState with essential fields only
        """
        minimal: MinimalGraphState = {
            "session_id": state["session_id"],
            "user_input": state["user_input"],
            "claims": state["claims"],
            "revision_count": state["revision_count"],
            "workflow_stage": state["workflow_stage"],
        }
        
        if "draft_response" in state:
            minimal["draft_response"] = state["draft_response"]
        
        return minimal
    
    @staticmethod
    def extract_summary(state: GraphState) -> Dict[str, Any]:
        """
        Extract a summary of the investigation for display purposes.
        
        Args:
            state: GraphState to summarize
        
        Returns:
            Summary dictionary with key metrics and status
        """
        summary = {
            "investigation_id": state["investigation_id"],
            "video_url": state["user_input"]["video_url"],
            "target_type": state["user_input"]["target_type"],
            "workflow_stage": state["workflow_stage"],
            "claims_count": len(state["claims"]),
            "revision_count": state["revision_count"],
            "created_at": state["created_at"],
            "last_updated": state["last_updated"],
        }
        
        # Add claim status breakdown
        claim_statuses = {}
        for claim in state["claims"]:
            status = claim["status"]
            claim_statuses[status] = claim_statuses.get(status, 0) + 1
        summary["claim_statuses"] = claim_statuses
        
        # Add response status
        if "final_response" in state:
            summary["has_final_response"] = True
            summary["response_length"] = len(state["final_response"])
        elif "draft_response" in state:
            summary["has_draft_response"] = True
            summary["draft_length"] = len(state["draft_response"])
        else:
            summary["has_response"] = False
        
        # Add current status
        if "status" in state:
            summary["current_step"] = state["status"].get("current_step")
            summary["step_progress"] = state["status"].get("step_progress", 0.0)
        
        return summary


class StateArchiver:
    """Utilities for archiving and compressing state data."""
    
    @staticmethod
    def archive_state(state: GraphState, compression_level: int = 6) -> bytes:
        """
        Archive state to compressed bytes for long-term storage.
        
        Args:
            state: GraphState to archive
            compression_level: Compression level (1-9, higher = more compression)
        
        Returns:
            Compressed state data as bytes
        """
        import gzip
        
        try:
            # Serialize to JSON
            json_data = StateSerializer.to_json(state, include_debug=True)
            
            # Compress
            compressed = gzip.compress(
                json_data.encode('utf-8'),
                compresslevel=compression_level
            )
            
            return compressed
            
        except Exception as e:
            raise StateSerializationError(f"Failed to archive state: {str(e)}")
    
    @staticmethod
    def restore_state(archived_data: bytes) -> GraphState:
        """
        Restore state from archived bytes.
        
        Args:
            archived_data: Compressed state data
        
        Returns:
            Restored GraphState
        """
        import gzip
        
        try:
            # Decompress
            json_data = gzip.decompress(archived_data).decode('utf-8')
            
            # Deserialize
            return StateSerializer.from_json(json_data)
            
        except Exception as e:
            raise StateSerializationError(f"Failed to restore state from archive: {str(e)}")


def create_state_checkpoint(state: GraphState) -> Dict[str, Any]:
    """
    Create a checkpoint of the current state for rollback purposes.
    
    Args:
        state: Current GraphState
    
    Returns:
        Checkpoint data including metadata
    """
    checkpoint = {
        "checkpoint_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + 'Z',
        "workflow_stage": state["workflow_stage"],
        "revision_count": state["revision_count"],
        "state_data": StateSerializer.to_dict(state, include_debug=False),
    }
    
    return checkpoint


def restore_from_checkpoint(checkpoint_data: Dict[str, Any]) -> GraphState:
    """
    Restore state from a checkpoint.
    
    Args:
        checkpoint_data: Checkpoint data from create_state_checkpoint
    
    Returns:
        Restored GraphState
    """
    return StateSerializer.from_dict(checkpoint_data["state_data"])


# Utility functions for common operations
def safe_update_state(
    current_state: GraphState,
    updates: StateUpdate,
    validate: bool = True
) -> GraphState:
    """
    Safely update state with validation.
    
    Args:
        current_state: Current GraphState
        updates: Updates to apply
        validate: Whether to validate the resulting state
    
    Returns:
        Updated GraphState
    
    Raises:
        ValidationError: If validation fails
    """
    if validate:
        from .validation import validate_state_update
        errors = validate_state_update(current_state, updates)
        if errors:
            raise ValidationError("state_update", "; ".join(errors))
    
    # Update timestamp
    updates["last_updated"] = datetime.utcnow().isoformat() + 'Z'
    
    # Merge updates
    updated_state = {**current_state, **updates}
    
    return updated_state


def extract_public_state(state: GraphState) -> Dict[str, Any]:
    """
    Extract public-facing state information (no sensitive debug data).
    
    Args:
        state: GraphState to extract from
    
    Returns:
        Public state information suitable for API responses
    """
    public_state = StateSerializer.to_dict(state, include_debug=False)
    
    # Remove potentially sensitive fields
    sensitive_fields = ["debug_info", "performance_metrics", "config"]
    for field in sensitive_fields:
        public_state.pop(field, None)
    
    return public_state 