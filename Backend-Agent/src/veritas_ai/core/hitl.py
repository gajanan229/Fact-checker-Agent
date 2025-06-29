"""
Human-in-the-Loop (HITL) Infrastructure for Veritas AI.

This module provides comprehensive infrastructure for human-AI collaboration including:
- Interrupt/resume patterns with Command objects
- Session management and state checkpointing
- Real-time collaboration features
- Advanced workflow orchestration
- State recovery and resilience
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Union, Literal, Callable, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, field_validator
from langgraph.types import interrupt, Command, Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from .state import GraphState, ChatMessage
from .validation import ValidationError

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


class InterruptType(str, Enum):
    """Types of workflow interrupts."""
    HUMAN_REVIEW = "human_review"
    APPROVAL_REQUEST = "approval_request"
    FEEDBACK_REQUEST = "feedback_request"
    CLARIFICATION_NEEDED = "clarification_needed"
    COLLABORATION_INVITE = "collaboration_invite"
    EMERGENCY_STOP = "emergency_stop"


class SessionStatus(str, Enum):
    """Status of user sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class CollaborationMode(str, Enum):
    """Modes of human-AI collaboration."""
    FULL_CONTROL = "full_control"        # Human drives the process
    GUIDED_ASSISTANCE = "guided_assistance"  # AI suggests, human decides
    AUTONOMOUS_WITH_OVERSIGHT = "autonomous_with_oversight"  # AI drives, human reviews
    COLLABORATIVE = "collaborative"      # Shared control


@dataclass
class InterruptPayload:
    """Payload for workflow interrupts."""
    interrupt_type: InterruptType
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    requires_approval: bool = False
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumePayload:
    """Payload for workflow resume operations."""
    action: str
    data: Any = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserSession(BaseModel):
    """User session management."""
    
    session_id: str = Field(description="Unique session identifier")
    user_id: str = Field(description="User identifier")
    thread_id: str = Field(description="LangGraph thread identifier")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    collaboration_mode: CollaborationMode = Field(default=CollaborationMode.COLLABORATIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))
    checkpoints: List[str] = Field(default_factory=list, description="List of checkpoint IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def extend_session(self, hours: int = 24) -> None:
        """Extend session expiry."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.last_activity = datetime.now(timezone.utc)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        import time
        # Small delay to ensure timestamp difference
        time.sleep(0.001)
        self.last_activity = datetime.now(timezone.utc)


class HITLManager:
    """Core Human-in-the-Loop workflow manager."""
    
    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """Initialize HITL manager."""
        self.checkpointer = checkpointer or MemorySaver()
        self.sessions: Dict[str, UserSession] = {}
        self.active_interrupts: Dict[str, InterruptPayload] = {}
        self.collaboration_handlers: Dict[str, Callable] = {}
        self.session_cleanup_interval = 3600  # 1 hour in seconds
        
        logger.info("HITLManager initialized")
    
    def create_session(
        self, 
        user_id: str, 
        thread_id: Optional[str] = None,
        collaboration_mode: CollaborationMode = CollaborationMode.COLLABORATIVE,
        session_duration_hours: int = 24
    ) -> UserSession:
        """Create a new user session."""
        
        session_id = str(uuid.uuid4())
        if thread_id is None:
            thread_id = f"thread_{session_id}"
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            thread_id=thread_id,
            collaboration_mode=collaboration_mode,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=session_duration_hours)
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            self.terminate_session(session_id, reason="expired")
            return None
        return session
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[UserSession]:
        """Get all sessions for a user."""
        sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        if active_only:
            sessions = [s for s in sessions if s.status == SessionStatus.ACTIVE and not s.is_expired()]
        return sessions
    
    def terminate_session(self, session_id: str, reason: str = "user_requested") -> bool:
        """Terminate a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.TERMINATED
        session.metadata["termination_reason"] = reason
        session.metadata["terminated_at"] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Terminated session {session_id}: {reason}")
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired_count = 0
        current_time = datetime.now(timezone.utc)
        
        for session_id, session in list(self.sessions.items()):
            if session.is_expired() or (
                session.status in [SessionStatus.COMPLETED, SessionStatus.TERMINATED] and
                (current_time - session.last_activity).total_seconds() > self.session_cleanup_interval
            ):
                del self.sessions[session_id]
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    async def create_interrupt(
        self,
        session_id: str,
        interrupt_type: InterruptType,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None,
        priority: Literal["low", "medium", "high", "critical"] = "medium"
    ) -> str:
        """Create a workflow interrupt."""
        
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("session_not_found", f"Session {session_id} not found or expired")
        
        interrupt_id = str(uuid.uuid4())
        payload = InterruptPayload(
            interrupt_type=interrupt_type,
            question=question,
            context=context or {},
            options=options or [],
            timeout_seconds=timeout_seconds,
            priority=priority,
            metadata={
                "session_id": session_id,
                "user_id": session.user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "interrupt_id": interrupt_id
            }
        )
        
        self.active_interrupts[interrupt_id] = payload
        session.update_activity()
        
        logger.info(f"Created interrupt {interrupt_id} for session {session_id}")
        
        # Use LangGraph's interrupt function
        response = interrupt({
            "interrupt_id": interrupt_id,
            "interrupt_type": interrupt_type.value,
            "question": question,
            "context": context or {},
            "options": options or [],
            "timeout_seconds": timeout_seconds,
            "priority": priority,
            "session_id": session_id,
            "user_id": session.user_id
        })
        
        return response
    
    def create_resume_command(
        self,
        session_id: str,
        action: str,
        data: Any = None,
        goto_node: Optional[str] = None,
        update_state: Optional[Dict[str, Any]] = None
    ) -> Command:
        """Create a resume command for workflow continuation."""
        
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("session_not_found", f"Session {session_id} not found or expired")
        
        resume_payload = {
            "action": action,
            "data": data,
            "user_id": session.user_id,
            "session_id": session_id,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "collaboration_mode": session.collaboration_mode.value
            }
        }
        
        session.update_activity()
        
        # Create appropriate Command object based on action type
        if goto_node and update_state:
            return Command(goto=goto_node, update=update_state)
        elif goto_node:
            return Command(goto=goto_node)
        elif update_state:
            return Command(resume=resume_payload, update=update_state)
        else:
            return Command(resume=resume_payload)
    
    async def handle_collaboration_request(
        self,
        session_id: str,
        request_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle real-time collaboration requests."""
        
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("session_not_found", f"Session {session_id} not found")
        
        handler = self.collaboration_handlers.get(request_type)
        if not handler:
            raise ValidationError("handler_not_found", f"No handler for request type: {request_type}")
        
        session.update_activity()
        result = await handler(session, data)
        
        return {
            "status": "success",
            "result": result,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def register_collaboration_handler(
        self,
        request_type: str,
        handler: Callable[[UserSession, Dict[str, Any]], Any]
    ) -> None:
        """Register a collaboration request handler."""
        self.collaboration_handlers[request_type] = handler
        logger.info(f"Registered collaboration handler for: {request_type}")


class CheckpointManager:
    """Advanced checkpoint management for state persistence."""
    
    def __init__(self, checkpointer: BaseCheckpointSaver):
        """Initialize checkpoint manager."""
        self.checkpointer = checkpointer
        self.checkpoint_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def create_checkpoint(
        self,
        thread_id: str,
        state: GraphState,
        checkpoint_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a named checkpoint."""
        
        checkpoint_id = str(uuid.uuid4())
        
        # Store checkpoint metadata
        self.checkpoint_metadata[checkpoint_id] = {
            "name": checkpoint_name or f"checkpoint_{checkpoint_id[:8]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "thread_id": thread_id,
            "metadata": metadata or {},
            "state_summary": self._create_state_summary(state)
        }
        
        logger.info(f"Created checkpoint {checkpoint_id} for thread {thread_id}")
        return checkpoint_id
    
    async def restore_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str
    ) -> Optional[GraphState]:
        """Restore state from a checkpoint."""
        
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        
        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple:
                logger.info(f"Restored checkpoint {checkpoint_id} for thread {thread_id}")
                return checkpoint_tuple.checkpoint.get("channel_values", {})
            return None
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a thread."""
        
        checkpoints = []
        
        # Filter checkpoints by thread_id from our metadata
        for checkpoint_id, metadata in self.checkpoint_metadata.items():
            if metadata.get("thread_id") == thread_id:
                checkpoints.append({
                    "checkpoint_id": checkpoint_id,
                    "created_at": metadata.get("created_at"),
                    "name": metadata.get("name", f"checkpoint_{checkpoint_id[:8]}"),
                    "metadata": metadata.get("metadata", {}),
                    "state_summary": metadata.get("state_summary", {})
                })
        
        # Sort by creation time (most recent first)
        return sorted(checkpoints, key=lambda x: x["created_at"] or "", reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            if checkpoint_id in self.checkpoint_metadata:
                del self.checkpoint_metadata[checkpoint_id]
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def _create_state_summary(self, state: GraphState) -> Dict[str, Any]:
        """Create a summary of the current state."""
        return {
            "workflow_stage": state.get("workflow_stage", "unknown"),
            "claims_count": len(state.get("claims", [])),
            "has_draft_response": bool(state.get("draft_response")),
            "chat_messages_count": len(state.get("chat_history", [])),
            "last_updated": state.get("last_updated"),
            "status": state.get("status", {})
        }


class CollaborationHub:
    """Real-time collaboration features hub."""
    
    def __init__(self, hitl_manager: HITLManager):
        """Initialize collaboration hub."""
        self.hitl_manager = hitl_manager
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        
        # Register default collaboration handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default collaboration request handlers."""
        
        self.hitl_manager.register_collaboration_handler(
            "review_draft", self._handle_review_draft
        )
        self.hitl_manager.register_collaboration_handler(
            "suggest_edit", self._handle_suggest_edit
        )
        self.hitl_manager.register_collaboration_handler(
            "request_clarification", self._handle_request_clarification
        )
        self.hitl_manager.register_collaboration_handler(
            "approve_response", self._handle_approve_response
        )
    
    async def _handle_review_draft(
        self, 
        session: UserSession, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle draft review requests."""
        
        draft_text = data.get("draft_text", "")
        review_type = data.get("review_type", "general")
        
        return {
            "action": "review_requested",
            "draft_text": draft_text,
            "review_type": review_type,
            "session_id": session.session_id,
            "requires_human_input": True
        }
    
    async def _handle_suggest_edit(
        self, 
        session: UserSession, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle edit suggestions."""
        
        suggested_text = data.get("suggested_text", "")
        original_text = data.get("original_text", "")
        reason = data.get("reason", "")
        
        return {
            "action": "edit_suggested",
            "original_text": original_text,
            "suggested_text": suggested_text,
            "reason": reason,
            "session_id": session.session_id
        }
    
    async def _handle_request_clarification(
        self, 
        session: UserSession, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle clarification requests."""
        
        question = data.get("question", "")
        context = data.get("context", {})
        
        return {
            "action": "clarification_requested",
            "question": question,
            "context": context,
            "session_id": session.session_id,
            "requires_human_input": True
        }
    
    async def _handle_approve_response(
        self, 
        session: UserSession, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle response approval."""
        
        approved = data.get("approved", False)
        feedback = data.get("feedback", "")
        
        return {
            "action": "approval_received",
            "approved": approved,
            "feedback": feedback,
            "session_id": session.session_id
        }
    
    async def initiate_collaboration(
        self,
        session_id: str,
        collaboration_type: str,
        participants: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Initiate a collaborative session."""
        
        collaboration_id = str(uuid.uuid4())
        
        self.active_collaborations[collaboration_id] = {
            "id": collaboration_id,
            "type": collaboration_type,
            "session_id": session_id,
            "participants": participants,
            "context": context,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "messages": []
        }
        
        logger.info(f"Initiated collaboration {collaboration_id} for session {session_id}")
        return collaboration_id
    
    async def send_collaboration_message(
        self,
        collaboration_id: str,
        sender_id: str,
        message: str,
        message_type: str = "message"
    ) -> bool:
        """Send a message in a collaborative session."""
        
        collaboration = self.active_collaborations.get(collaboration_id)
        if not collaboration:
            return False
        
        message_data = {
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "message": message,
            "message_type": message_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        collaboration["messages"].append(message_data)
        logger.info(f"Message sent in collaboration {collaboration_id}")
        
        return True


# HITL Workflow Integration Functions

async def hitl_interrupt_node(state: GraphState) -> Union[GraphState, Command]:
    """
    Node function that creates an interrupt for human input.
    
    This function is designed to be used as a LangGraph node that
    pauses execution and waits for human input.
    """
    
    session_id = state.get("session_id")
    if not session_id:
        raise ValidationError("missing_session", "Session ID required for HITL operations")
    
    # Create interrupt payload based on current state
    interrupt_type = InterruptType.HUMAN_REVIEW
    question = "Please review the current analysis and provide your feedback:"
    
    context = {
        "workflow_stage": state.get("workflow_stage", "unknown"),
        "claims_summary": [
            {"text": claim.get("text", ""), "status": claim.get("status", "")}
            for claim in state.get("claims", [])
        ],
        "draft_response": state.get("draft_response", ""),
        "current_step": state.get("status", {}).get("current_step", "")
    }
    
    # Use interrupt to pause workflow
    response = interrupt({
        "interrupt_type": interrupt_type.value,
        "question": question,
        "context": context,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Process the response and update state
    updated_state = {
        **state,
        "last_human_interaction": {
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": interrupt_type.value
        },
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    return updated_state


def create_hitl_workflow_nodes(hitl_manager: HITLManager) -> Dict[str, Callable]:
    """
    Create workflow nodes that integrate with the HITL manager.
    
    Returns a dictionary of node functions that can be added to a LangGraph.
    """
    
    async def human_review_node(state: GraphState) -> Union[GraphState, Command]:
        """Node for human review with approval workflow."""
        session_id = state.get("session_id")
        if not session_id:
            raise ValidationError("missing_session", "Session ID required")
        
        response = await hitl_manager.create_interrupt(
            session_id=session_id,
            interrupt_type=InterruptType.APPROVAL_REQUEST,
            question="Please review and approve this fact-checking response:",
            context={
                "draft_response": state.get("draft_response", ""),
                "claims": state.get("claims", []),
                "sources": state.get("sources", [])
            },
            options=["approve", "reject", "request_changes"],
            priority="high"
        )
        
        return {
            **state,
            "workflow_stage": "awaiting_approval",
            "last_human_interaction": {
                "type": "approval_request",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    async def collaboration_node(state: GraphState) -> Union[GraphState, Command]:
        """Node for collaborative editing and feedback."""
        session_id = state.get("session_id")
        if not session_id:
            raise ValidationError("missing_session", "Session ID required")
        
        response = await hitl_manager.create_interrupt(
            session_id=session_id,
            interrupt_type=InterruptType.COLLABORATION_INVITE,
            question="Would you like to collaborate on improving this response?",
            context={
                "draft_response": state.get("draft_response", ""),
                "current_claims": state.get("claims", [])
            },
            options=["collaborate", "approve_as_is", "request_revision"],
            priority="medium"
        )
        
        return {
            **state,
            "workflow_stage": "collaborating",
            "collaboration_active": True
        }
    
    return {
        "human_review": human_review_node,
        "collaboration": collaboration_node,
        "hitl_interrupt": hitl_interrupt_node
    }


# Export main classes and functions
__all__ = [
    "InterruptType",
    "SessionStatus", 
    "CollaborationMode",
    "InterruptPayload",
    "ResumePayload",
    "UserSession",
    "HITLManager",
    "CheckpointManager",
    "CollaborationHub",
    "hitl_interrupt_node",
    "create_hitl_workflow_nodes"
] 