"""
State validation utilities for the Veritas AI fact-checking engine.

This module provides validation functions to ensure GraphState integrity
and consistency throughout the workflow.
"""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .state import (
    GraphState, MinimalGraphState, Claim, Source, UserInput, 
    RawContent, ClaimStatus, SourceType, StateUpdate
)


class ValidationError(Exception):
    """Custom exception for state validation errors."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error in '{field}': {message}")


class StateValidator:
    """Comprehensive state validation utilities."""
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_tiktok_url(url: str) -> bool:
        """Validate TikTok URL format by checking the domain."""
        if not StateValidator.validate_url(url):
            return False
        
        try:
            domain = urlparse(url).netloc.lower()
            valid_domains = ['tiktok.com', 'vm.tiktok.com', 'vt.tiktok.com']
            return any(d in domain for d in valid_domains)
        except Exception:
            return False
    
    @staticmethod
    def validate_iso_timestamp(timestamp: str) -> bool:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID format."""
        try:
            uuid.UUID(uuid_string)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """Validate score is within expected range."""
        return isinstance(score, (int, float)) and min_val <= score <= max_val
    
    @staticmethod
    def validate_user_input(user_input: UserInput) -> List[str]:
        """Validate UserInput structure and values."""
        errors = []
        
        # Required fields
        if 'video_url' not in user_input:
            errors.append("Missing required field: video_url")
        elif not StateValidator.validate_tiktok_url(user_input['video_url']):
            errors.append("Invalid TikTok URL format")
        
        if 'target_type' not in user_input:
            errors.append("Missing required field: target_type")
        elif user_input['target_type'] not in ['video', 'comment']:
            errors.append("target_type must be 'video' or 'comment'")
        
        # Conditional validation
        if user_input.get('target_type') == 'comment' and not user_input.get('target_content'):
            errors.append("target_content is required when target_type is 'comment'")
        
        return errors
    
    @staticmethod
    def validate_claim(claim: Claim) -> List[str]:
        """Validate Claim structure and values."""
        errors = []
        
        # Required fields
        required_fields = ['id', 'text', 'status']
        for field in required_fields:
            if field not in claim:
                errors.append(f"Missing required field in claim: {field}")
        
        # Validate claim ID
        if 'id' in claim and not StateValidator.validate_uuid(claim['id']):
            errors.append("Claim ID must be a valid UUID")
        
        # Validate claim text
        if 'text' in claim:
            if not isinstance(claim['text'], str) or len(claim['text'].strip()) < 10:
                errors.append("Claim text must be at least 10 characters long")
        
        # Validate status
        if 'status' in claim:
            try:
                ClaimStatus(claim['status'])
            except ValueError:
                errors.append(f"Invalid claim status: {claim['status']}")
        
        # Validate optional scores
        for score_field in ['confidence_score']:
            if score_field in claim:
                if not StateValidator.validate_score(claim[score_field]):
                    errors.append(f"Invalid {score_field}: must be between 0.0 and 1.0")
        
        # Validate timestamps
        for timestamp_field in ['created_at', 'last_updated']:
            if timestamp_field in claim:
                if not StateValidator.validate_iso_timestamp(claim[timestamp_field]):
                    errors.append(f"Invalid {timestamp_field}: must be ISO 8601 format")
        
        # Validate sources
        if 'sources' in claim:
            for i, source in enumerate(claim['sources']):
                source_errors = StateValidator.validate_source(source)
                for error in source_errors:
                    errors.append(f"Source {i}: {error}")
        
        return errors
    
    @staticmethod
    def validate_source(source: Source) -> List[str]:
        """Validate Source structure and values."""
        errors = []
        
        # Required fields
        required_fields = ['url', 'title', 'domain', 'content_snippet']
        for field in required_fields:
            if field not in source:
                errors.append(f"Missing required field in source: {field}")
        
        # Validate URL
        if 'url' in source and not StateValidator.validate_url(source['url']):
            errors.append("Invalid source URL format")
        
        # Validate domain matches URL
        if 'url' in source and 'domain' in source:
            try:
                url_domain = urlparse(source['url']).netloc.lower()
                if source['domain'].lower() not in url_domain:
                    errors.append("Domain does not match URL")
            except Exception:
                pass  # URL validation will catch URL issues
        
        # Validate scores
        for score_field in ['credibility_score', 'relevance_score']:
            if score_field in source:
                if not StateValidator.validate_score(source[score_field]):
                    errors.append(f"Invalid {score_field}: must be between 0.0 and 1.0")
        
        # Validate source type
        if 'source_type' in source:
            try:
                SourceType(source['source_type'])
            except ValueError:
                errors.append(f"Invalid source_type: {source['source_type']}")
        
        return errors
    
    @staticmethod
    def validate_graph_state(state: GraphState) -> List[str]:
        """Validate complete GraphState structure and consistency."""
        errors = []
        
        # Required fields
        required_fields = [
            'session_id', 'investigation_id', 'created_at', 'last_updated',
            'user_input', 'raw_content', 'claims', 'revision_count',
            'chat_history', 'status', 'workflow_stage'
        ]
        
        for field in required_fields:
            if field not in state:
                errors.append(f"Missing required field: {field}")
        
        # Validate IDs
        for id_field in ['session_id', 'investigation_id']:
            if id_field in state and not StateValidator.validate_uuid(state[id_field]):
                errors.append(f"Invalid {id_field}: must be a valid UUID")
        
        # Validate timestamps
        for timestamp_field in ['created_at', 'last_updated']:
            if timestamp_field in state:
                if not StateValidator.validate_iso_timestamp(state[timestamp_field]):
                    errors.append(f"Invalid {timestamp_field}: must be ISO 8601 format")
        
        # Validate user input
        if 'user_input' in state:
            input_errors = StateValidator.validate_user_input(state['user_input'])
            errors.extend(input_errors)
        
        # Validate claims
        if 'claims' in state:
            for i, claim in enumerate(state['claims']):
                claim_errors = StateValidator.validate_claim(claim)
                for error in claim_errors:
                    errors.append(f"Claim {i}: {error}")
        
        # Validate revision count
        if 'revision_count' in state:
            if not isinstance(state['revision_count'], int) or state['revision_count'] < 0:
                errors.append("revision_count must be a non-negative integer")
        
        # Validate max revisions
        if 'max_revisions' in state:
            if not isinstance(state['max_revisions'], int) or state['max_revisions'] < 1:
                errors.append("max_revisions must be a positive integer")
        
        # Validate workflow stage
        valid_stages = [
            "initialized", "content_extracted", "claims_identified",
            "evidence_gathered", "response_drafted", "response_reviewed",
            "user_collaborating", "completed", "failed"
        ]
        if 'workflow_stage' in state and state['workflow_stage'] not in valid_stages:
            errors.append(f"Invalid workflow_stage: {state['workflow_stage']}")
        
        # Validate status consistency
        if 'status' in state and 'workflow_stage' in state:
            status_errors = StateValidator._validate_status_consistency(
                state['status'], state['workflow_stage']
            )
            errors.extend(status_errors)
        
        return errors
    
    @staticmethod
    def _validate_status_consistency(status: Dict[str, Any], workflow_stage: str) -> List[str]:
        """Validate consistency between status and workflow_stage."""
        errors = []
        
        # Check if current_step aligns with workflow_stage
        stage_to_step_mapping = {
            "initialized": "ingesting",
            "content_extracted": "identifying_claims",
            "claims_identified": "researching",
            "evidence_gathered": "generating_response",
            "response_drafted": "reviewing",
            "response_reviewed": "awaiting_user",
            "user_collaborating": "awaiting_user",
            "completed": "awaiting_user",
            "failed": "awaiting_user"
        }
        
        expected_step = stage_to_step_mapping.get(workflow_stage)
        current_step = status.get('current_step')
        
        if expected_step and current_step and current_step != expected_step:
            # Allow some flexibility for in-progress states
            if not (workflow_stage in ["user_collaborating", "completed", "failed"]):
                errors.append(
                    f"Status current_step '{current_step}' inconsistent with "
                    f"workflow_stage '{workflow_stage}'"
                )
        
        return errors


def create_initial_state(
    video_url: str,
    target_type: str,
    target_content: Optional[str] = None,
    session_id: Optional[str] = None,
    investigation_id: Optional[str] = None
) -> GraphState:
    """
    Create a valid initial GraphState for a new investigation.
    
    Args:
        video_url: TikTok video URL to investigate
        target_type: Either 'video' or 'comment'
        target_content: Username if targeting a comment
        session_id: Optional session ID (generated if not provided)
        investigation_id: Optional investigation ID (generated if not provided)
    
    Returns:
        Valid GraphState ready for workflow execution
    
    Raises:
        ValidationError: If input parameters are invalid
    """
    now = datetime.utcnow().isoformat() + 'Z'
    
    # Generate IDs if not provided
    session_id = session_id or str(uuid.uuid4())
    investigation_id = investigation_id or str(uuid.uuid4())
    
    # Create user input
    user_input: UserInput = {
        'video_url': video_url,
        'target_type': target_type,
    }
    
    if target_content:
        user_input['target_content'] = target_content
    
    # Validate user input
    validation_errors = StateValidator.validate_user_input(user_input)
    if validation_errors:
        raise ValidationError("user_input", "; ".join(validation_errors))
    
    # Create initial state
    initial_state: GraphState = {
        'session_id': session_id,
        'investigation_id': investigation_id,
        'created_at': now,
        'last_updated': now,
        'user_input': user_input,
        'raw_content': {},
        'claims': [],
        'revision_count': 0,
        'chat_history': [],
        'status': {
            'current_step': 'ingesting',
            'step_progress': 0.0,
        },
        'workflow_stage': 'initialized'
    }
    
    return initial_state


def validate_state_update(current_state: GraphState, update: StateUpdate) -> List[str]:
    """
    Validate a proposed state update before applying it.
    
    Args:
        current_state: Current GraphState
        update: Proposed state updates
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Create a temporary updated state for validation
    temp_state = {**current_state, **update}
    
    # Update timestamp
    temp_state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
    
    # Validate the resulting state
    state_errors = StateValidator.validate_graph_state(temp_state)
    errors.extend(state_errors)
    
    return errors 