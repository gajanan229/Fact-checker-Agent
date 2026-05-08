"""
API utility functions.

Currently exposes :func:`transform_state_for_frontend`, which projects the
backend ``GraphState`` into the simpler ``caseFile`` shape consumed by the
React Workbench (claims list, dossier, draft response, sources).
"""

from typing import Any, Dict, List

from ..core.state import Claim, ClaimStatus, GraphState


_VERDICT_LABELS: Dict[str, str] = {
    ClaimStatus.VERIFIED.value: "Verified",
    ClaimStatus.DEBUNKED.value: "False",
    ClaimStatus.MISLEADING.value: "Misleading",
    ClaimStatus.LACKS_CONTEXT.value: "Lacks Context",
    ClaimStatus.UNVERIFIABLE.value: "Unverifiable",
    ClaimStatus.PENDING.value: "Pending",
    ClaimStatus.RESEARCHING.value: "Researching",
}


def _normalize_status(status: Any) -> str:
    """Return a ``ClaimStatus`` value as its raw string, defaulting to PENDING."""
    if isinstance(status, ClaimStatus):
        return status.value
    if status is None:
        return ClaimStatus.PENDING.value
    return str(status)


def _verdict_label(status: str) -> str:
    """Map a status string to the human-friendly label rendered in the UI."""
    return _VERDICT_LABELS.get(status, status.replace("_", " ").title())


def _format_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project backend ``Source`` records into the shape the EvidenceLocker expects."""
    formatted: List[Dict[str, Any]] = []
    for index, source in enumerate(sources or [], start=1):
        url = source.get("url", "")
        title = source.get("title") or source.get("domain") or url or "Untitled source"
        formatted.append({
            "id": index,
            "url": url,
            "title": title,
            "domain": source.get("domain", ""),
            "snippet": source.get("content_snippet", ""),
        })
    return formatted


def transform_state_for_frontend(graph_state: GraphState) -> Dict[str, Any]:
    """
    Transform the final ``GraphState`` into the ``caseFile`` payload.

    Sources live inside each claim on the backend, and analysis text is
    stored as ``verification_summary`` / ``evidence_summary`` rather than a
    generic ``summary`` field. This projection flattens those into the
    ``{claims, dossier}`` shape the Workbench renders.
    """
    backend_claims: List[Claim] = graph_state.get("claims", []) or []

    frontend_claims: List[Dict[str, Any]] = []
    dossier: Dict[str, Dict[str, Any]] = {}

    for claim in backend_claims:
        claim_id = str(claim.get("id", "")).strip()
        if not claim_id:
            continue

        status = _normalize_status(claim.get("status"))
        summary = (
            claim.get("verification_summary")
            or claim.get("evidence_summary")
            or "Analysis is not yet available for this claim."
        )

        frontend_claims.append({
            "claim_id": claim_id,
            "text": claim.get("text", ""),
            "status": status,
        })

        dossier[claim_id] = {
            "verdict": _verdict_label(status),
            "summary": summary,
            "sources": _format_sources(claim.get("sources", [])),
        }

    return {
        "claims": frontend_claims,
        "dossier": dossier,
        "critique": graph_state.get("critique", {}),
        "draft_response": graph_state.get("draft_response", ""),
        "final_response": graph_state.get("final_response", ""),
        "response_sources": graph_state.get("response_sources", []),
    }
