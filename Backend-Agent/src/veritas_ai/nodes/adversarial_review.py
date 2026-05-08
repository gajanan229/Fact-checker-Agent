"""
Adversarial Review Node for Veritas AI.

Three-stage critique pipeline that genuinely red-teams the draft response:

  1. extract_response_claims  - LLM pulls verifiable factual assertions out of
                                the draft text itself
  2. verify_response_claims   - tool-using agent web-searches each one and
                                returns a verdict (reuses the evidence research
                                agent from evidence_research.py)
  3. judge                    - LLM scores the response on factual accuracy,
                                tone/respectfulness, and citation quality, and
                                produces a structured Critique

Replaces the previous "two LLM self-grades against the response's own claims"
implementation, which had no way to catch hallucinations introduced by the
response generator.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..core.state import (
    Claim,
    Critique,
    CritiqueDimensionScores,
    GraphState,
    ResponseClaimVerification,
)
from ..utils.api_usage import APIUsageError, api_usage_manager
from .evidence_research import EvidenceResearchError, _ClaimResearchAgent

load_dotenv()
logger = logging.getLogger(__name__)


class AdversarialReviewError(Exception):
    """Raised for unrecoverable adversarial review failures."""


# === Structured outputs =================================================

class ExtractedResponseClaim(BaseModel):
    """A factual assertion lifted from the draft response."""
    text: str = Field(
        description="The exact factual assertion as it appears in the response."
    )


class ExtractedResponseClaims(BaseModel):
    """Set of factual assertions extracted from the draft response."""
    claims: List[ExtractedResponseClaim] = Field(
        default_factory=list,
        description="Up to 4 most load-bearing factual assertions in the response.",
    )


class CritiqueScores(BaseModel):
    """Dimension scores assigned by the judge."""
    factual_accuracy: float = Field(
        ge=0.0, le=1.0,
        description="How well the response's factual claims hold up to verification.",
    )
    tone_respectfulness: float = Field(
        ge=0.0, le=1.0,
        description="How constructive, neutral, and educational the tone is.",
    )
    citation_quality: float = Field(
        ge=0.0, le=1.0,
        description="How well the response cites credible sources for its claims.",
    )


class CritiqueJudgement(BaseModel):
    """Final judgement produced by the critique judge."""
    scores: CritiqueScores
    is_revision_needed: bool = Field(
        description="True only if at least one dimension is < 0.6 OR a verification flagged a debunked claim."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Concrete strengths of the response (max 4 items, one short sentence each).",
        max_length=4,
    )
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Concrete issues that warrant a revision (max 4 items).",
        max_length=4,
    )
    revision_recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions for the next draft (max 5 items, name what to change).",
        max_length=5,
    )


# === Pipeline ===========================================================

_EXTRACT_CLAIMS_PROMPT = """Extract every distinct, externally verifiable factual claim from a fact-checking response.

Include:
- Statistics, numbers, percentages, dates
- Specific historical, scientific, or biographical facts
- Definite assertions that could be confirmed against external sources

Skip:
- Opinions, hedged language ("may", "might", "some say"), calls to action
- General context that doesn't make a verifiable claim
- Citation tokens like [1], [2]

Return at most 4 claims. Prioritize the most load-bearing assertions. Keep each
claim's text faithful to how the response stated it."""


_JUDGE_PROMPT = """You are the final judge for a fact-checking response.

Score the response on three dimensions, each 0.0 to 1.0:
- factual_accuracy: do the response's factual assertions match the verification results below? Penalize debunked or misleading verifications heavily.
- tone_respectfulness: is the tone constructive and educational, free of partisanship, condescension, or inflammatory language?
- citation_quality: does the response cite credible sources for non-trivial claims, and are citations placed where they belong?

Set is_revision_needed = true only if at least one dimension is below 0.6 OR a
verification status is "debunked".

Provide concrete strengths, critical issues, and revision recommendations.
Recommendations must name what to change — never write generic advice like
"improve quality"."""


class CritiquePipeline:
    """Three-stage critique: extract response claims -> web-verify them -> judge."""

    def __init__(self, max_response_claims: int = 3, verification_concurrency: int = 2):
        self._llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            temperature=0.1,
            max_output_tokens=2048,
        )
        self._verifier = _ClaimResearchAgent(self._llm, recursion_limit=10)
        self._verification_semaphore = asyncio.Semaphore(verification_concurrency)
        self._max_response_claims = max_response_claims

    async def critique(
        self, draft_response: str, original_claims: List[Claim]
    ) -> Critique:
        """Run the full pipeline and return a structured Critique."""
        if not draft_response or not draft_response.strip():
            return _empty_critique("Draft response is empty; nothing to review.")

        extracted = await self._extract_response_claims(draft_response)
        logger.info(f"Extracted {len(extracted)} claim(s) from draft response")

        verifications = await self._verify_response_claims(extracted)
        logger.info(
            f"Verified {len(verifications)} response claim(s); "
            f"{sum(1 for v in verifications if v['status'] == 'debunked')} debunked"
        )

        judgement = await self._judge(draft_response, original_claims, verifications)

        return _build_critique(judgement, verifications)

    async def _extract_response_claims(self, draft_response: str) -> List[str]:
        try:
            api_usage_manager.check_and_increment_gemini()
            structured = self._llm.with_structured_output(ExtractedResponseClaims)
            result = await structured.ainvoke([
                ("system", _EXTRACT_CLAIMS_PROMPT),
                ("human", f"Response to analyze:\n\n{draft_response}"),
            ])
            claims = [c.text.strip() for c in result.claims if c.text.strip()]
            return claims[: self._max_response_claims]
        except APIUsageError:
            raise
        except Exception as e:
            logger.warning(f"Response claim extraction failed: {e}")
            return []

    async def _verify_response_claims(
        self, claim_texts: List[str]
    ) -> List[ResponseClaimVerification]:
        if not claim_texts:
            return []

        async def _verify(index: int, text: str) -> ResponseClaimVerification:
            async with self._verification_semaphore:
                synthetic_claim: Claim = {  # type: ignore[typeddict-item]
                    "id": f"resp-{index}",
                    "text": text,
                }
                try:
                    api_usage_manager.check_and_increment_gemini()
                    verdict = await self._verifier.adjudicate(synthetic_claim)
                    return ResponseClaimVerification(
                        claim=text,
                        status=verdict.status,
                        verification_summary=verdict.verification_summary,
                    )
                except APIUsageError as e:
                    logger.warning(f"API limit hit verifying response claim: {e}")
                    return ResponseClaimVerification(
                        claim=text,
                        status="unverifiable",
                        verification_summary="Verification blocked by API quota limit.",
                    )
                except (EvidenceResearchError, Exception) as e:
                    logger.warning(f"Failed to verify response claim '{text[:60]}': {e}")
                    return ResponseClaimVerification(
                        claim=text,
                        status="unverifiable",
                        verification_summary=f"Verification failed: {e}",
                    )

        return await asyncio.gather(
            *(_verify(i, t) for i, t in enumerate(claim_texts))
        )

    async def _judge(
        self,
        draft_response: str,
        original_claims: List[Claim],
        verifications: List[ResponseClaimVerification],
    ) -> CritiqueJudgement:
        try:
            api_usage_manager.check_and_increment_gemini()
            structured = self._llm.with_structured_output(CritiqueJudgement)
            user_prompt = (
                f"DRAFT RESPONSE:\n{draft_response}\n\n"
                f"{_format_original_claims(original_claims)}\n\n"
                f"{_format_response_verifications(verifications)}"
            )
            return await structured.ainvoke([
                ("system", _JUDGE_PROMPT),
                ("human", user_prompt),
            ])
        except APIUsageError:
            raise
        except Exception as e:
            logger.error(f"Critique judgement failed: {e}", exc_info=True)
            raise AdversarialReviewError(f"Failed to produce critique judgement: {e}")


# === Helpers ============================================================

def _format_original_claims(claims: List[Claim]) -> str:
    if not claims:
        return "ORIGINAL CLAIMS BEING ADDRESSED: (none)"
    lines = ["ORIGINAL CLAIMS BEING ADDRESSED:"]
    for i, claim in enumerate(claims, 1):
        status = claim.get("status")
        status_value = status.value if hasattr(status, "value") else str(status)
        verification = (claim.get("verification_summary") or "").strip()
        lines.append(f"{i}. [{status_value}] {claim.get('text', '')}")
        if verification:
            lines.append(f"   Adjudication: {verification}")
    return "\n".join(lines)


def _format_response_verifications(
    verifications: List[ResponseClaimVerification],
) -> str:
    if not verifications:
        return (
            "RESPONSE-CLAIM VERIFICATIONS: (no factual assertions were extracted from the response — "
            "score factual_accuracy based on consistency with the original claim adjudications)"
        )
    lines = ["RESPONSE-CLAIM VERIFICATIONS:"]
    for i, v in enumerate(verifications, 1):
        lines.append(f'{i}. [{v["status"]}] "{v["claim"]}"')
        if v.get("verification_summary"):
            lines.append(f"   {v['verification_summary']}")
    return "\n".join(lines)


def _build_critique(
    judgement: CritiqueJudgement,
    verifications: List[ResponseClaimVerification],
) -> Critique:
    scores = judgement.scores
    overall = (
        scores.factual_accuracy
        + scores.tone_respectfulness
        + scores.citation_quality
    ) / 3.0

    return Critique(
        is_revision_needed=judgement.is_revision_needed,
        overall_quality_score=round(overall, 3),
        quality_scores=CritiqueDimensionScores(
            factual_accuracy=scores.factual_accuracy,
            tone_respectfulness=scores.tone_respectfulness,
            citation_quality=scores.citation_quality,
        ),
        strengths=list(judgement.strengths),
        critical_issues=list(judgement.critical_issues),
        revision_recommendations=list(judgement.revision_recommendations),
        response_claim_verifications=list(verifications),
        critique_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _empty_critique(reason: str) -> Critique:
    """Build a passing critique for cases where there's nothing to review."""
    return Critique(
        is_revision_needed=False,
        overall_quality_score=0.0,
        quality_scores=CritiqueDimensionScores(
            factual_accuracy=0.0,
            tone_respectfulness=0.0,
            citation_quality=0.0,
        ),
        strengths=[],
        critical_issues=[reason],
        revision_recommendations=[],
        response_claim_verifications=[],
        critique_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# === LangGraph node functions ==========================================

async def critique_response(
    state: GraphState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """LangGraph node: red-team the draft response with web verification + judgement."""
    session_id = state.get("session_id")
    logger.info(f"Starting adversarial review for session {session_id}")

    draft_response = state.get("draft_response", "") or ""
    claims = state.get("claims", []) or []
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    if not draft_response.strip():
        logger.warning("No draft response to review.")
        return {
            "critique": _empty_critique("No draft response was provided."),
            "workflow_stage": "response_reviewed",
            "status": {"current_step": "reviewing", "step_progress": 1.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    try:
        pipeline = CritiquePipeline()
        critique = await pipeline.critique(draft_response, claims)
    except APIUsageError as e:
        logger.error(f"API limit hit during critique: {e}")
        return {
            "critique": _empty_critique(f"Quality review skipped: {e}"),
            "workflow_stage": "response_reviewed",
            "status": {"current_step": "reviewing", "step_progress": 1.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    except AdversarialReviewError as e:
        logger.error(f"Adversarial review failed: {e}")
        return {
            "critique": _empty_critique(f"Quality review failed: {e}"),
            "workflow_stage": "response_reviewed",
            "status": {"current_step": "reviewing", "step_progress": 1.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    revise = critique["is_revision_needed"] and revision_count < max_revisions
    next_stage = "response_drafted" if revise else "response_reviewed"
    new_revision_count = revision_count + 1 if revise else revision_count

    logger.info(
        f"Critique complete. revision_needed={critique['is_revision_needed']}, "
        f"overall={critique['overall_quality_score']:.2f}, "
        f"revising={revise} (count {revision_count}->{new_revision_count})"
    )

    return {
        "critique": critique,
        "workflow_stage": next_stage,
        "revision_count": new_revision_count,
        "status": {"current_step": "reviewing", "step_progress": 1.0},
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def critique_response_sync(
    state: GraphState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for the adversarial review node."""
    try:
        return asyncio.run(critique_response(state, config))
    except Exception as e:
        logger.error(f"Synchronous adversarial review failed: {e}", exc_info=True)
        return {
            "critique": _empty_critique(f"Quality review failed: {e}"),
            "workflow_stage": "response_reviewed",
            "status": {"current_step": "reviewing", "step_progress": 1.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
