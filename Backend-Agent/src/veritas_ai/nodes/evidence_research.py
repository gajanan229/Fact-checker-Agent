"""
Evidence Research Node for Veritas AI.

Each pending claim is handed to a tool-using ReAct agent that may issue
multiple Tavily searches, optionally extract full pages, and finally
commits a structured verdict (status + reasoning + supporting sources).

Replaces the previous "one fixed query per claim, never adjudicate" path.
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
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.prebuilt import create_react_agent
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


# === Rate-limited Tavily tools ==========================================

class _RateLimitedTavilySearch(TavilySearch):
    """TavilySearch that increments our shared Tavily quota counter per call."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return super()._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return await super()._arun(*args, **kwargs)


class _RateLimitedTavilyExtract(TavilyExtract):
    """TavilyExtract that increments our shared Tavily quota counter per call."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return super()._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        api_usage_manager.check_and_increment_tavily()
        return await super()._arun(*args, **kwargs)


# === Agent ==============================================================

_AGENT_SYSTEM_PROMPT = """You are a fact-checking research agent adjudicating ONE claim.

Tools available to you:
- tavily_search(query): web search; returns ranked snippets
- tavily_extract(urls): pull full text from specific URLs when a snippet looks promising

Strategy: plan an effective search query, review results, optionally extract the most
relevant page, optionally do one follow-up search if a key angle is missing, then commit
a verdict. Stop within 3 searches total.

Prefer reputable sources (major news outlets, government publications, peer-reviewed
research, established fact-checkers) over blogs and social media. If sources disagree,
weigh credibility and recency.

Your final response MUST conform to the ClaimVerdict schema. Choose status:
- verified: well-supported by credible sources
- debunked: directly contradicted by credible sources
- misleading: technically accurate but framed deceptively
- lacks_context: incomplete; missing critical context
- unverifiable: insufficient credible evidence either way
"""


class _ClaimResearchAgent:
    """Tool-using ReAct agent that adjudicates a single claim."""

    def __init__(self, llm: Any, recursion_limit: int = 14):
        self._search_tool = _RateLimitedTavilySearch(
            max_results=5,
            search_depth="advanced",
            include_raw_content="text",
        )
        self._extract_tool = _RateLimitedTavilyExtract(extract_depth="basic")
        self._agent = create_react_agent(
            model=llm,
            tools=[self._search_tool, self._extract_tool],
            prompt=_AGENT_SYSTEM_PROMPT,
            response_format=ClaimVerdict,
        )
        self._recursion_limit = recursion_limit

    async def adjudicate(self, claim: Claim) -> ClaimVerdict:
        """Run the agent on a single claim and return its structured verdict."""
        user_message = (
            f"Claim to adjudicate: {claim['text']}\n\n"
            "Search for evidence, then commit a verdict in the ClaimVerdict format."
        )
        result = await self._agent.ainvoke(
            {"messages": [("user", user_message)]},
            config={"recursion_limit": self._recursion_limit},
        )
        verdict = result.get("structured_response")
        if not isinstance(verdict, ClaimVerdict):
            raise EvidenceResearchError(
                f"Agent did not produce a structured verdict for claim {claim['id']}"
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
