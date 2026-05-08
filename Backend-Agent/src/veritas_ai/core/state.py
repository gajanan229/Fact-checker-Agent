"""
Core state definitions for the Veritas AI fact-checking engine.

This module defines :class:`GraphState` and its supporting TypedDicts. The
state object is the "case file" that flows through the LangGraph workflow,
carrying everything from the original TikTok URL to the final published
response.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict


class ClaimStatus(str, Enum):
    """Lifecycle status of an individual claim."""
    PENDING = "pending"
    RESEARCHING = "researching"
    VERIFIED = "verified"
    DEBUNKED = "debunked"
    MISLEADING = "misleading"
    LACKS_CONTEXT = "lacks_context"
    UNVERIFIABLE = "unverifiable"


class SourceType(str, Enum):
    """Classification of an evidence source."""
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
    """An evidence source attached to a claim."""
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
    """A numbered source reference rendered in the frontend response."""
    number: int
    domain: str
    title: NotRequired[str]
    url: NotRequired[str]


class Claim(TypedDict):
    """An individual claim under fact-check."""
    id: str
    text: str
    status: ClaimStatus
    confidence_score: NotRequired[float]  # 0.0 to 1.0
    verification_summary: NotRequired[str]
    evidence_summary: NotRequired[str]
    sources: NotRequired[List[Source]]
    extracted_from: NotRequired[Literal["video", "comment"]]
    created_at: NotRequired[str]
    last_updated: NotRequired[str]


class UserInput(TypedDict):
    """User input and target selection for an investigation."""
    video_url: str
    target_type: Literal["video", "comment"]
    target_content: NotRequired[str]
    user_preferences: NotRequired[Dict[str, Any]]


class RawContent(TypedDict):
    """Raw content extracted from TikTok before claim identification."""
    transcript: NotRequired[str]
    ocr_text: NotRequired[str]
    selected_comment_text: NotRequired[str]
    video_metadata: NotRequired[Dict[str, Any]]
    extraction_timestamp: NotRequired[str]
    extraction_errors: NotRequired[List[str]]


class ResponseQuality(TypedDict):
    """Quality scores assigned by the critique stage."""
    accuracy_score: NotRequired[float]
    tone_score: NotRequired[float]
    citation_score: NotRequired[float]
    clarity_score: NotRequired[float]
    overall_score: NotRequired[float]


class Critique(TypedDict):
    """Adversarial review feedback on a draft response."""
    is_revision_needed: bool
    feedback_text: str
    suggested_improvements: NotRequired[List[str]]
    quality_assessment: NotRequired[ResponseQuality]
    critique_timestamp: NotRequired[str]


class ProcessingStatus(TypedDict):
    """Snapshot of the current processing step."""
    current_step: Literal[
        "ingesting", "identifying_claims", "researching",
        "generating_response", "reviewing", "awaiting_user",
    ]
    step_progress: NotRequired[float]  # 0.0 to 1.0
    estimated_completion: NotRequired[str]
    error_count: NotRequired[int]
    warnings: NotRequired[List[str]]


class GraphState(TypedDict):
    """
    Complete state of a Veritas AI fact-checking investigation.

    The same dict instance is mutated as the LangGraph workflow progresses
    through ingestion -> claim identification -> research -> response
    drafting -> critique.
    """

    # Session and tracking
    session_id: str
    investigation_id: str
    created_at: str
    last_updated: str

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
    max_revisions: NotRequired[int]

    # Processing status and control
    status: ProcessingStatus
    workflow_stage: Literal[
        "initialized", "content_extracted", "claims_identified",
        "evidence_gathered", "response_drafted", "response_reviewed",
        "user_collaborating", "completed", "failed",
    ]

    # Error handling and debugging
    error_message: NotRequired[str]
    debug_info: NotRequired[Dict[str, Any]]
    performance_metrics: NotRequired[Dict[str, float]]

    # Configuration and preferences
    config: NotRequired[Dict[str, Any]]
    user_preferences: NotRequired[Dict[str, Any]]


# Type aliases for common state operations
StateUpdate = Dict[str, Any]
ClaimUpdate = Dict[str, Any]
SourceUpdate = Dict[str, Any]
