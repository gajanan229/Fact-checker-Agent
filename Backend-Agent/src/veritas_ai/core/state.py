"""
Core state definitions for the Veritas AI fact-checking engine.

This module defines the GraphState and supporting TypedDict classes that represent
the complete investigation case file as it flows through the LangGraph workflow.
"""

from typing import Dict, List, Literal, Optional, Union, Any
from typing_extensions import TypedDict, NotRequired, Annotated
from datetime import datetime
from enum import Enum

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ClaimStatus(str, Enum):
    """Status of a claim during verification process."""
    PENDING = "pending"
    RESEARCHING = "researching"
    VERIFIED = "verified"
    DEBUNKED = "debunked"
    MISLEADING = "misleading"
    LACKS_CONTEXT = "lacks_context"
    UNVERIFIABLE = "unverifiable"


class SourceType(str, Enum):
    """Type of evidence source."""
    NEWS_ARTICLE = "news_article"
    ACADEMIC_PAPER = "academic_paper"
    GOVERNMENT_DOCUMENT = "government_document"
    FACT_CHECK_SITE = "fact_check_site"
    SCIENTIFIC_JOURNAL = "scientific_journal"
    EXPERT_OPINION = "expert_opinion"
    OFFICIAL_STATEMENT = "official_statement"
    DATA_REPOSITORY = "data_repository"
    OTHER = "other"


class Source(TypedDict):
    """Evidence source information."""
    url: str
    title: str
    domain: str
    content_snippet: str
    publication_date: NotRequired[Optional[str]]
    author: NotRequired[Optional[str]]
    source_type: NotRequired[SourceType]
    credibility_score: NotRequired[float]  # 0.0 to 1.0
    relevance_score: NotRequired[float]    # 0.0 to 1.0


class NumberedSourceRef(TypedDict):
    """A numbered source reference for display in the frontend."""
    number: int
    domain: str
    title: NotRequired[str]
    url: NotRequired[str]


class Claim(TypedDict):
    """Individual claim to be fact-checked."""
    id: str
    text: str
    status: ClaimStatus
    confidence_score: NotRequired[float]  # 0.0 to 1.0
    verification_summary: NotRequired[str]
    evidence_summary: NotRequired[str]
    sources: NotRequired[List[Source]]
    extracted_from: NotRequired[Literal["video", "comment"]]  # Where claim was found
    created_at: NotRequired[str]  # ISO timestamp
    last_updated: NotRequired[str]  # ISO timestamp


class UserInput(TypedDict):
    """User input and targeting information."""
    video_url: str
    target_type: Literal["video", "comment"]
    target_content: NotRequired[str]  # Username for comment targeting
    user_preferences: NotRequired[Dict[str, Any]]


class RawContent(TypedDict):
    """Raw extracted content from TikTok."""
    transcript: NotRequired[str]
    ocr_text: NotRequired[str]
    selected_comment_text: NotRequired[str]
    video_metadata: NotRequired[Dict[str, Any]]
    extraction_timestamp: NotRequired[str]
    extraction_errors: NotRequired[List[str]]


class ResponseQuality(TypedDict):
    """Quality metrics for generated responses."""
    accuracy_score: NotRequired[float]
    tone_score: NotRequired[float]
    citation_score: NotRequired[float]
    clarity_score: NotRequired[float]
    overall_score: NotRequired[float]


class Critique(TypedDict):
    """Adversarial review feedback."""
    is_revision_needed: bool
    feedback_text: str
    suggested_improvements: NotRequired[List[str]]
    quality_assessment: NotRequired[ResponseQuality]
    critique_timestamp: NotRequired[str]


class ChatMessage(TypedDict):
    """Human-AI collaborative chat message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    message_type: NotRequired[Literal["feedback", "suggestion", "question", "approval"]]
    referenced_claim_ids: NotRequired[List[str]]


class ProcessingStatus(TypedDict):
    """Current processing status and progress."""
    current_step: Literal[
        "ingesting", "identifying_claims", "researching", 
        "generating_response", "reviewing", "awaiting_user"
    ]
    step_progress: NotRequired[float]  # 0.0 to 1.0
    estimated_completion: NotRequired[str]  # ISO timestamp
    error_count: NotRequired[int]
    warnings: NotRequired[List[str]]


class GraphState(TypedDict):
    """
    Complete state of the Veritas AI fact-checking investigation.
    
    This represents the "case file" that flows through the LangGraph workflow,
    containing all information from initial input to final response.
    """
    
    # Session and tracking
    session_id: str
    investigation_id: str
    created_at: str  # ISO timestamp
    last_updated: str  # ISO timestamp
    
    # User input and targeting
    user_input: UserInput
    
    # Raw extracted content
    raw_content: RawContent
    
    # Claims and evidence
    claims: List[Claim]
    
    # Response generation
    draft_response: NotRequired[str]
    final_response: NotRequired[str]
    response_metadata: NotRequired[Dict[str, Any]]
    response_sources: NotRequired[List[NumberedSourceRef]]
    
    # Quality control
    critique: NotRequired[Critique]
    revision_count: int
    max_revisions: NotRequired[int]  # Default: 3
    
    # Human-in-the-loop collaboration
    chat_history: Annotated[List[BaseMessage], add_messages]
    user_feedback_pending: NotRequired[bool]
    collaboration_notes: NotRequired[List[ChatMessage]]
    
    # Processing status and control
    status: ProcessingStatus
    workflow_stage: Literal[
        "initialized", "content_extracted", "claims_identified", 
        "evidence_gathered", "response_drafted", "response_reviewed", 
        "user_collaborating", "completed", "failed"
    ]
    
    # Error handling and debugging
    error_message: NotRequired[str]
    debug_info: NotRequired[Dict[str, Any]]
    performance_metrics: NotRequired[Dict[str, float]]
    
    # Configuration and preferences
    config: NotRequired[Dict[str, Any]]
    user_preferences: NotRequired[Dict[str, Any]]


class MinimalGraphState(TypedDict):
    """
    Minimal state for basic operations and testing.
    
    This is a simplified version of GraphState for development and testing
    purposes, containing only the essential fields.
    """
    session_id: str
    user_input: UserInput
    claims: List[Claim]
    draft_response: NotRequired[str]
    revision_count: int
    workflow_stage: Literal[
        "initialized", "content_extracted", "claims_identified", 
        "evidence_gathered", "response_drafted", "response_reviewed", 
        "completed", "failed"
    ]


# Type aliases for common state operations
StateUpdate = Dict[str, Any]
ClaimUpdate = Dict[str, Any]
SourceUpdate = Dict[str, Any] 