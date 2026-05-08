"""
Claim Identification Node for Veritas AI.

Pulls verifiable factual claims out of the cleaned transcript using a single
structured-output Gemini call. The extractor is intentionally permissive --
recall over precision -- because the downstream evidence research stage is the
real filter (it adjudicates each claim and emits a terminal status).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from ..core.state import Claim, ClaimStatus, GraphState
from ..core.validation import ValidationError
from ..utils.api_usage import APIUsageError, api_usage_manager


logger = logging.getLogger(__name__)


# Safety rail: stop the LLM from generating a runaway list of claims for a long
# transcript. Anything past this is almost certainly noise or near-duplicates.
MAX_CLAIMS_PER_VIDEO = 20


class ExtractedClaim(BaseModel):
    """A single verifiable factual claim lifted from the transcript."""

    text: str = Field(
        description="The exact factual claim, rewritten as a complete standalone sentence."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are this is a verifiable factual claim (not opinion or speculation).",
    )
    context: str = Field(
        default="",
        description="One short sentence of surrounding context that helps a fact-checker understand the claim.",
    )

    @field_validator("text")
    @classmethod
    def _strip_and_check_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Claim text too short - must be at least 10 characters")
        return v


class ClaimExtractionResult(BaseModel):
    """Structured output for the extraction call."""

    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="Every distinct, externally verifiable claim found in the content.",
    )


_SYSTEM_PROMPT = """You extract verifiable factual claims from short-form video transcripts for a fact-checking pipeline.

Extract every distinct claim that could be checked against external sources -- statistics, dates, named events, scientific or historical assertions, specific cause/effect statements, quotes attributed to people. Prefer recall over precision; the next stage will adjudicate each claim and discard weak ones.

Skip only obvious non-claims: pure opinion ("I love it"), rhetorical questions, calls to action, and vague platitudes with no checkable content.

Rewrite each claim as a complete standalone sentence (resolve pronouns, name entities) so it can be researched without the surrounding transcript."""


class ClaimIdentifier:
    """Single-call structured-output claim extractor."""

    def __init__(self, model_name: str | None = None):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
                temperature=0.1,
                max_output_tokens=4096,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ClaimIdentifier: {e}")
            raise ValidationError("model_initialization", f"ClaimIdentifier setup failed: {e}")

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "CONTENT SOURCE: {content_source}\n\nTRANSCRIPT:\n{content_text}",
            ),
        ])
        self._chain = self._prompt | self.llm.with_structured_output(ClaimExtractionResult)

    async def extract_claims_async(
        self,
        content_text: str,
        content_source: str = "unknown",
    ) -> ClaimExtractionResult:
        if not content_text or not content_text.strip():
            logger.warning("Empty content provided for claim extraction")
            return ClaimExtractionResult(claims=[])

        try:
            api_usage_manager.check_and_increment_gemini()
            result: ClaimExtractionResult = await self._chain.ainvoke({
                "content_text": content_text.strip(),
                "content_source": content_source,
            })
        except APIUsageError as e:
            logger.error(f"API limit reached for Gemini in claim identification: {e}")
            raise ValidationError("api_limit", str(e))
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}", exc_info=True)
            return ClaimExtractionResult(claims=[])

        if len(result.claims) > MAX_CLAIMS_PER_VIDEO:
            logger.info(
                f"Truncating extracted claims from {len(result.claims)} to "
                f"{MAX_CLAIMS_PER_VIDEO} (safety rail)"
            )
            result.claims = result.claims[:MAX_CLAIMS_PER_VIDEO]

        logger.info(f"Extracted {len(result.claims)} claim(s) from {content_source}")
        return result

    def extract_claims_sync(
        self,
        content_text: str,
        content_source: str = "unknown",
    ) -> ClaimExtractionResult:
        return asyncio.run(self.extract_claims_async(content_text, content_source))


class ClaimProcessor:
    """Converts extracted claims into Claim TypedDicts and dedupes them."""

    @staticmethod
    def create_claim_objects(
        extracted_claims: List[ExtractedClaim],
        source_type: Literal["video", "comment"] = "video",
    ) -> List[Claim]:
        """Project ExtractedClaim models into Claim dicts ready for the graph."""
        current_time = datetime.now(timezone.utc).isoformat()
        claims: List[Claim] = []
        for extracted_claim in extracted_claims:
            claim = Claim(
                id=str(uuid.uuid4()),
                text=extracted_claim.text.strip(),
                status=ClaimStatus.PENDING,
                confidence_score=extracted_claim.confidence,
                verification_summary="",
                evidence_summary="",
                sources=[],
                extracted_from=source_type,
                created_at=current_time,
                last_updated=current_time,
            )
            claims.append(claim)
        return claims

    @staticmethod
    def filter_duplicate_claims(claims: List[Claim]) -> List[Claim]:
        """Remove exact and near-duplicate claims.

        Uses Jaccard similarity over word sets; collapses pairs only when the
        overlap is high (>= 0.8) so that legitimately distinct claims that
        happen to share common stopwords aren't merged.
        """
        filtered: List[Claim] = []
        seen_word_sets: List[set] = []

        for claim in claims:
            text = claim["text"].lower().strip()
            words = {w for w in text.split() if w}
            if not words:
                continue

            is_dup = False
            for prev in seen_word_sets:
                union = words | prev
                if not union:
                    continue
                jaccard = len(words & prev) / len(union)
                if jaccard >= 0.8:
                    is_dup = True
                    break

            if is_dup:
                logger.debug(f"Filtered near-duplicate claim: {text[:60]}...")
                continue

            filtered.append(claim)
            seen_word_sets.append(words)

        removed = len(claims) - len(filtered)
        if removed:
            logger.info(f"Filtered {removed} duplicate/near-duplicate claim(s)")
        return filtered

    @staticmethod
    def validate_claims(claims: List[Claim]) -> List[Claim]:
        """Drop malformed claims and backfill missing metadata."""
        valid_claims: List[Claim] = []
        current_time = datetime.now(timezone.utc).isoformat()

        for claim in claims:
            if not claim.get("text") or len(claim["text"].strip()) < 10:
                logger.warning("Skipping invalid claim with insufficient text")
                continue
            claim.setdefault("id", str(uuid.uuid4()))
            claim.setdefault("status", ClaimStatus.PENDING)
            claim.setdefault("created_at", current_time)
            claim.setdefault("last_updated", current_time)
            valid_claims.append(claim)

        return valid_claims


# === LangGraph node ====================================================

async def identify_claims(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: extract verifiable claims from raw_content."""
    session_id = state.get("session_id")
    logger.info(f"Starting claim identification for session {session_id}")

    raw_content = state.get("raw_content") or {}
    user_input = state.get("user_input") or {}
    target_type: Literal["video", "comment"] = user_input.get("target_type", "video")

    content_parts: List[str] = []
    content_sources: List[str] = []
    if raw_content.get("transcript"):
        content_parts.append(raw_content["transcript"])
        content_sources.append("video_transcript")
    if raw_content.get("ocr_text"):
        content_parts.append(raw_content["ocr_text"])
        content_sources.append("video_text_overlay")
    if raw_content.get("selected_comment_text"):
        content_parts.append(raw_content["selected_comment_text"])
        content_sources.append("comment_text")

    status = state.get("status") or {}

    if not content_parts:
        logger.warning("No content available for claim identification")
        return {
            "claims": [],
            "workflow_stage": "claims_identified",
            "status": {
                **status,
                "current_step": "identifying_claims",
                "warnings": [
                    *(status.get("warnings") or []),
                    "No content available for claim identification",
                ],
            },
        }

    combined_content = "\n\n".join(
        f"[{src.upper()}]: {body}" for src, body in zip(content_sources, content_parts)
    )

    try:
        identifier = ClaimIdentifier()
        extraction_result = await identifier.extract_claims_async(
            combined_content, f"{target_type}_content"
        )
    except ValidationError as e:
        logger.error(f"Claim identification blocked: {e}")
        return {
            "workflow_stage": "failed",
            "error_message": f"Claim identification failed: {e}",
            "status": {
                **status,
                "current_step": "identifying_claims",
                "error_count": (status.get("error_count") or 0) + 1,
            },
        }

    claim_objects = ClaimProcessor.create_claim_objects(
        extraction_result.claims, target_type
    )
    deduped = ClaimProcessor.filter_duplicate_claims(claim_objects)
    validated = ClaimProcessor.validate_claims(deduped)

    logger.info(f"Successfully identified {len(validated)} claim(s)")
    return {
        "claims": validated,
        "workflow_stage": "claims_identified",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "status": {
            **status,
            "current_step": "identifying_claims",
            "step_progress": 1.0,
        },
    }


def identify_claims_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous wrapper for the claim identification node."""
    try:
        return asyncio.run(identify_claims(state))
    except Exception as e:
        logger.error(f"Synchronous claim identification failed: {e}", exc_info=True)
        return {
            "workflow_stage": "failed",
            "error_message": f"Claim identification failed: {e}",
        }
