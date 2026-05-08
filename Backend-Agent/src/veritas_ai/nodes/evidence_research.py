"""
Evidence Research Node for Veritas AI.

Each pending claim is run through a deterministic two-call pipeline:

  1. The LLM proposes one precise web search query (structured output).
  2. We execute that query against Tavily directly.
  3. The LLM reads the results and commits a structured verdict.

Single-shot calls only, so we never need to round-trip thought_signatures
back to Gemini 3.x — sidestepping a known langchain-google-genai limitation
with multi-turn tool-call loops on thinking models.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from ..core.state import Claim, ClaimStatus, GraphState, Source, SourceType
from ..utils.api_usage import APIUsageError, api_usage_manager

load_dotenv()
logger = logging.getLogger(__name__)


class EvidenceResearchError(Exception):
    """Raised for unrecoverable evidence research failures."""


# === Structured agent output ============================================

class VerdictCitation(BaseModel):
    """A single source the agent used to support its verdict."""

    url: str = Field(description="Full URL of the source page.")
    title: str = Field(description="Title of the page or article.")
    domain: str = Field(description="Bare domain, e.g. 'nytimes.com'.")
    excerpt: str = Field(
        description="Short quote (1-3 sentences) from the page that bears on the claim."
    )


class ClaimVerdict(BaseModel):
    """Final adjudication of a single claim."""

    status: Literal[
        "verified", "debunked", "misleading", "lacks_context", "unverifiable"
    ] = Field(
        description=(
            "Verdict for the claim:\n"
            "- verified: well-supported by credible sources\n"
            "- debunked: directly contradicted by credible sources\n"
            "- misleading: technically accurate but framed deceptively\n"
            "- lacks_context: incomplete; missing critical context\n"
            "- unverifiable: insufficient credible evidence either way"
        )
    )
    verification_summary: str = Field(
        description=(
            "2-4 sentence reasoned conclusion explaining the verdict. "
            "Reference specific source domains."
        )
    )
    evidence_summary: str = Field(
        description="2-4 sentence summary of what the gathered evidence actually shows."
    )
    citations: List[VerdictCitation] = Field(
        default_factory=list,
        description="Sources that most directly support the verdict (1-5 items).",
    )


# === Rate-limited Tavily tool ===========================================

class _RateLimitedTavilySearch(TavilySearch):
    """TavilySearch that increments our shared Tavily quota counter per call."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return super()._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return await super()._arun(*args, **kwargs)


# === Agent ==============================================================

class _SearchQuery(BaseModel):
    """A targeted web search query for fact-checking a single claim."""

    query: str = Field(
        description=(
            "A precise web search query that would surface authoritative evidence "
            "for or against the claim. Name specific entities, numbers, or events."
        )
    )


_QUERY_SYSTEM_PROMPT = """You are preparing to fact-check a single claim.

Output ONE short, precise web search query that would surface authoritative
evidence for or against the claim. Prefer queries that name the specific
entities, numbers, or events involved. Avoid generic queries.
"""


_VERDICT_SYSTEM_PROMPT = """You are a fact-checking research agent adjudicating ONE claim against web search results.

Read the search results carefully and decide a verdict status:
- verified: well-supported by credible sources
- debunked: directly contradicted by credible sources
- misleading: technically accurate but framed deceptively
- lacks_context: incomplete; missing critical context
- unverifiable: insufficient credible evidence either way

Prefer reputable sources (major news outlets, government publications, peer-reviewed
research, established fact-checkers) over blogs and social media. If sources disagree,
weigh credibility and recency.

Citations must come from the provided search results — pick 1-5 of the most
directly supportive items. The excerpt for each citation must be a short quote
drawn from that result's content.
"""


def _format_search_results(payload: Any) -> str:
    """Render a Tavily search payload into a compact prompt-friendly form."""
    if not isinstance(payload, dict):
        return str(payload) if payload else "(no results returned)"

    results = payload.get("results") or []
    if not results:
        return "(no results returned)"

    lines: List[str] = []
    answer = (payload.get("answer") or "").strip()
    if answer:
        lines.append(f"Tavily summary: {answer}")
        lines.append("")

    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip() or "(untitled)"
        url = (r.get("url") or "").strip()
        content = (r.get("content") or r.get("raw_content") or "").strip()
        if len(content) > 1500:
            content = content[:1500] + "..."
        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    Excerpt: {content}")
        lines.append("")

    return "\n".join(lines).rstrip()


class _ClaimResearchAgent:
    """Deterministic two-call research pipeline for a single claim.

    Stage 1: ask the LLM to propose one precise search query.
    Stage 2: run Tavily search directly.
    Stage 3: ask the LLM for a verdict given the search results.

    Both LLM calls are single-shot structured outputs, so we never round-trip
    function-call messages back to the model — this avoids Gemini 3.x's
    thought_signature requirement, which langchain-google-genai 2.1.6 does
    not currently preserve across turns.
    """

    def __init__(self, llm: Any, recursion_limit: int = 14):
        # recursion_limit retained for backwards-compatible call sites; unused
        # in the deterministic pipeline.
        del recursion_limit
        self._llm = llm
        self._search_tool = _RateLimitedTavilySearch(
            max_results=5,
            search_depth="advanced",
            include_raw_content="text",
        )

    async def adjudicate(self, claim: Claim) -> ClaimVerdict:
        """Run the pipeline on a single claim and return its structured verdict."""
        claim_text = claim["text"]
        claim_id = claim.get("id", "?")

        try:
            query_model = self._llm.with_structured_output(_SearchQuery)
            query: _SearchQuery = await query_model.ainvoke([
                ("system", _QUERY_SYSTEM_PROMPT),
                ("human", f"Claim to fact-check: {claim_text}"),
            ])
        except APIUsageError:
            raise
        except Exception as e:
            raise EvidenceResearchError(
                f"Failed to generate search query for claim {claim_id}: {e}"
            ) from e

        try:
            search_payload = await self._search_tool.ainvoke({"query": query.query})
        except APIUsageError:
            raise
        except Exception as e:
            raise EvidenceResearchError(
                f"Tavily search failed for claim {claim_id}: {e}"
            ) from e

        formatted_results = _format_search_results(search_payload)

        try:
            api_usage_manager.check_and_increment_gemini()
            verdict_model = self._llm.with_structured_output(ClaimVerdict)
            verdict: ClaimVerdict = await verdict_model.ainvoke([
                ("system", _VERDICT_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        f"Claim: {claim_text}\n\n"
                        f"Search query used: {query.query}\n\n"
                        f"Search results:\n{formatted_results}"
                    ),
                ),
            ])
        except APIUsageError:
            raise
        except Exception as e:
            raise EvidenceResearchError(
                f"Failed to produce verdict for claim {claim_id}: {e}"
            ) from e

        if not isinstance(verdict, ClaimVerdict):
            raise EvidenceResearchError(
                f"LLM did not produce a structured verdict for claim {claim_id}"
            )
        return verdict


# === Pipeline ===========================================================

class EvidenceResearcher:
    """Coordinates evidence research across multiple claims with bounded parallelism."""

    def __init__(self, max_concurrent_claims: int = 2):
        if not os.getenv("TAVILY_API_KEY"):
            raise EvidenceResearchError("TAVILY_API_KEY environment variable is required")

        self._llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            temperature=0.1,
            max_output_tokens=2048,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_claims)
        self._agent = _ClaimResearchAgent(self._llm)

    async def research_claims(self, claims: List[Claim]) -> List[Claim]:
        """Adjudicate every PENDING claim concurrently with bounded parallelism."""

        async def _process(claim: Claim) -> Claim:
            if claim.get("status") != ClaimStatus.PENDING:
                return claim
            async with self._semaphore:
                return await self._adjudicate_one(claim)

        return await asyncio.gather(*(_process(c) for c in claims))

    async def _adjudicate_one(self, claim: Claim) -> Claim:
        """Run the agent on one claim and merge its verdict back into the claim."""
        updated = claim.copy()
        try:
            api_usage_manager.check_and_increment_gemini()
            verdict = await self._agent.adjudicate(claim)

            updated["status"] = ClaimStatus(verdict.status)
            updated["verification_summary"] = verdict.verification_summary
            updated["evidence_summary"] = verdict.evidence_summary
            updated["sources"] = [_citation_to_source(c) for c in verdict.citations]

            logger.info(
                f"Claim {claim['id']} adjudicated as {verdict.status} "
                f"with {len(verdict.citations)} citation(s)"
            )
        except APIUsageError as e:
            logger.error(f"API limit hit while researching claim {claim['id']}: {e}")
            updated["status"] = ClaimStatus.UNVERIFIABLE
            updated["verification_summary"] = "Research blocked by API quota limit."
            updated["evidence_summary"] = str(e)
            updated["sources"] = []
        except Exception as e:
            logger.error(
                f"Failed to adjudicate claim {claim['id']}: {e}", exc_info=True
            )
            updated["status"] = ClaimStatus.UNVERIFIABLE
            updated["verification_summary"] = "Research failed before reaching a verdict."
            updated["evidence_summary"] = f"Error: {e}"
            updated["sources"] = []

        updated["last_updated"] = datetime.now(timezone.utc).isoformat()
        return updated


def _citation_to_source(citation: VerdictCitation) -> Source:
    """Project a VerdictCitation produced by the agent into a backend Source dict."""
    domain = (citation.domain or "").strip().lower().replace("www.", "")
    if not domain and citation.url:
        try:
            domain = urlparse(citation.url).netloc.lower().replace("www.", "")
        except Exception:
            domain = ""

    return Source(
        url=citation.url,
        title=citation.title or domain or "Untitled source",
        domain=domain,
        content_snippet=citation.excerpt[:1000],
        source_type=SourceType.OTHER,
        credibility_score=0.5,
        relevance_score=0.5,
        publication_date=None,
        author=None,
    )


# === LangGraph node functions ==========================================

async def research_claims(
    state: GraphState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """LangGraph node: adjudicate every pending claim with a tool-using agent."""
    logger.info(f"Starting evidence research for session {state['session_id']}")
    claims = state.get("claims", []) or []

    pending = [c for c in claims if c.get("status") == ClaimStatus.PENDING]
    if not pending:
        logger.info("No pending claims to research.")
        return {
            "workflow_stage": "evidence_gathered",
            "status": {"current_step": "researching", "step_progress": 1.0},
        }

    try:
        researcher = EvidenceResearcher()
    except EvidenceResearchError as e:
        logger.error(f"Failed to initialize EvidenceResearcher: {e}")
        return {
            "workflow_stage": "failed",
            "error_message": str(e),
        }

    logger.info(f"Researching {len(pending)} claim(s)")
    updated_claims = await researcher.research_claims(claims)

    terminal_statuses = {
        ClaimStatus.VERIFIED,
        ClaimStatus.DEBUNKED,
        ClaimStatus.MISLEADING,
        ClaimStatus.LACKS_CONTEXT,
        ClaimStatus.UNVERIFIABLE,
    }
    adjudicated = sum(
        1 for c in updated_claims
        if c.get("id") in {p["id"] for p in pending}
        and c.get("status") in terminal_statuses
    )
    total_sources = sum(len(c.get("sources", []) or []) for c in updated_claims)
    logger.info(
        f"Research complete: {adjudicated}/{len(pending)} adjudicated, "
        f"{total_sources} source(s) collected"
    )

    return {
        "claims": updated_claims,
        "workflow_stage": "evidence_gathered",
        "status": {"current_step": "researching", "step_progress": 1.0},
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "performance_metrics": {
            "research_success_rate": adjudicated / len(pending) if pending else 1.0,
            "total_sources_found": float(total_sources),
        },
    }


def research_claims_sync(
    state: GraphState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for the evidence research node."""
    try:
        return asyncio.run(research_claims(state, config))
    except Exception as e:
        logger.error(f"Synchronous evidence research failed: {e}", exc_info=True)
        return {
            "workflow_stage": "failed",
            "error_message": f"Evidence research failed: {e}",
        }
