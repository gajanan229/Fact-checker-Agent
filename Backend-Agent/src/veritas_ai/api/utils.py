"""
This module contains utility functions for the API, including data transformation.
"""
from typing import Dict, Any, List
from ..core.state import GraphState, Claim, Source

def transform_state_for_frontend(graph_state: GraphState) -> Dict[str, Any]:
    """
    Transforms the final GraphState into the caseFile format expected by the frontend.
    """
    # Get raw data from the graph state, with defaults
    backend_claims: List[Claim] = graph_state.get("claims", [])
    backend_sources: List[Source] = graph_state.get("sources", [])
    
    # Create a quick-access map for sources by their ID
    source_map = {source["id"]: source for source in backend_sources}
    
    # 1. Transform claims and create the dossier simultaneously
    frontend_claims = []
    dossier = {}
    
    for claim in backend_claims:
        claim_id_str = str(claim["id"])
        
        # Format the claim for the frontend 'claims' list
        frontend_claims.append({
            "claim_id": claim_id_str,
            "text": claim["text"],
            "status": claim["status"],
        })
        
        # Find the sources for the current claim
        claim_sources = [
            source_map[source_id] for source_id in claim.get("source_ids", []) 
            if source_id in source_map
        ]
        
        # Create the dossier entry for this claim
        dossier[claim_id_str] = {
            "verdict": claim.get("status", "unknown"),
            "summary": claim.get("summary", "No summary available."),
            "sources": claim_sources
        }

    # 2. Assemble the final caseFile payload
    case_file = {
        "claims": frontend_claims,
        "dossier": dossier,
        "critique": graph_state.get("critique", {}),
        "draft_response": graph_state.get("draft_response", ""),
        "final_response": graph_state.get("final_response", ""),
        "response_sources": graph_state.get("response_sources", []),
    }
    
    return case_file 