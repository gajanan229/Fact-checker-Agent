"""
Autonomous Graph Definition for Veritas AI

This module defines the graph structure for a fully autonomous fact-checking workflow,
where human-in-the-loop interactions are bypassed.
"""

from typing import Literal
from langgraph.graph import END

from .state import GraphState, ClaimStatus
from .graph import VeritasGraph, error_handler_node
from ..nodes.content_ingestion import ingest_content_sync
from ..nodes.claim_identification import identify_claims_sync
from ..nodes.evidence_research import research_claims_sync
from ..nodes.response_generation import generate_response_sync
from ..nodes.adversarial_review import critique_response_sync

# === AUTONOMOUS ROUTING FUNCTIONS (BYPASS HUMAN INTERACTION) ===

def route_after_ingestion(state: GraphState) -> Literal["identify_claims", "error"]:
    """Autonomous routing after content ingestion."""
    try:
        raw_content = state.get("raw_content", {})
        if not raw_content or state.get("workflow_stage") == "failed":
            return "error"
        has_transcript = bool(raw_content.get("transcript", "").strip())
        if not has_transcript:
            return "error"
        return "identify_claims"
    except Exception:
        return "error"

def route_claims_analysis_autonomous(state: GraphState) -> Literal["research_claims", "generate_response"]:
    """Autonomous routing after claim identification."""
    try:
        claims = state.get("claims", [])
        if not claims:
            return "generate_response"
        verified_statuses = {ClaimStatus.VERIFIED, ClaimStatus.DEBUNKED, ClaimStatus.UNVERIFIABLE}
        all_processed = all(claim.get("status") in verified_statuses for claim in claims)
        if all_processed:
            return "generate_response"
        return "research_claims"
    except Exception:
        return "generate_response"

def route_after_research_autonomous(state: GraphState) -> Literal["generate_response"]:
    """Autonomous routing after evidence research."""
    return "generate_response"

def route_after_generation_autonomous(state: GraphState) -> Literal["critique_response", "complete"]:
    """Autonomous routing after response generation."""
    try:
        draft_response = state.get("draft_response", "")
        if not draft_response.strip() or len(draft_response.strip()) < 50:
            return "complete"
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        if revision_count >= max_revisions:
            return "complete"
        return "critique_response"
    except Exception:
        return "complete"

def route_after_critique_autonomous(state: GraphState) -> Literal["generate_response", "complete"]:
    """Autonomous routing after adversarial review."""
    try:
        critique = state.get("critique", {})
        if not critique:
            return "complete"
        revision_needed = critique.get("is_revision_needed", False)
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        if revision_count >= max_revisions:
            return "complete"
        if revision_needed:
            quality = critique.get("quality_assessment", {})
            overall_score = quality.get("overall_score", 0.5)
            if overall_score < 0.3:
                return "complete"
            return "generate_response"
        return "complete"
    except Exception:
        return "complete"

# === AUTONOMOUS GRAPH CONSTRUCTION ===

def create_autonomous_graph() -> VeritasGraph:
    """
    Creates and configures the complete autonomous fact-checking graph.
    
    Returns:
        A compiled VeritasGraph instance ready for execution.
    """
    # 1. Initialize the graph
    graph = VeritasGraph(use_checkpointer=False)

    # 2. Add all the nodes
    graph.add_node("ingest_content", ingest_content_sync, "Content Ingestion")
    graph.add_node("identify_claims", identify_claims_sync, "Claim Identification")
    graph.add_node("research_claims", research_claims_sync, "Evidence Research")
    graph.add_node("generate_response", generate_response_sync, "Response Generation")
    graph.add_node("critique_response", critique_response_sync, "Quality Review")
    graph.add_node("error_handler", error_handler_node, "Error Handling")

    # 3. Define the workflow edges
    graph.set_entry_point("ingest_content")
    graph.add_conditional_edge(
        "ingest_content",
        route_after_ingestion,
        {"identify_claims": "identify_claims", "error": "error_handler"}
    )
    graph.add_conditional_edge(
        "identify_claims",
        route_claims_analysis_autonomous,
        {"research_claims": "research_claims", "generate_response": "generate_response"}
    )
    graph.add_edge("research_claims", "generate_response")
    graph.add_conditional_edge(
        "generate_response",
        route_after_generation_autonomous,
        {"critique_response": "critique_response", "complete": END}
    )
    graph.add_conditional_edge(
        "critique_response",
        route_after_critique_autonomous,
        {"generate_response": "generate_response", "complete": END}
    )
    graph.add_edge("error_handler", END)

    # 4. Compile and return the graph
    return graph.compile()
