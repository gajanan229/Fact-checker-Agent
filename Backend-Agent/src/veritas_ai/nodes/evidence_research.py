"""
Evidence Research Node for Veritas AI (Simplified)

This module handles the research and evidence gathering phase of fact-checking using the Tavily API's integrated LLM features.
- Performs a single, comprehensive search per claim.
- Extracts a direct summary answer for the claim.
- Gathers supporting sources and their content.
- Updates claims with the summary and sourced evidence.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Third-party imports
from tavily import TavilyClient
from langchain_core.runnables import RunnableConfig

# Internal imports
from ..core.state import GraphState, Claim, Source, ClaimStatus, SourceType
from ..core.validation import ValidationError

logger = logging.getLogger(__name__)


class EvidenceResearchError(Exception):
    """Custom exception for evidence research errors"""
    pass


class EvidenceResearcher:
    """Main class for researching claims using Tavily's search API."""

    def __init__(self):
        """Initialize the evidence researcher with the TavilyClient."""
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not self.tavily_api_key:
            raise EvidenceResearchError("TAVILY_API_KEY environment variable is required")
        
        self.client = TavilyClient(api_key=self.tavily_api_key)
        self.max_sources_per_claim = 3

    async def research_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Research multiple claims concurrently and gather evidence.
        
        Args:
            claims: List of claims to research.
            
        Returns:
            Updated claims with research sources and summaries.
        """
        tasks = []
        for claim in claims:
            if claim['status'] == ClaimStatus.PENDING:
                tasks.append(self._research_single_claim(claim))
            else:
                # If claim doesn't need research, wrap it in a coroutine to keep types consistent
                async def no_research(c):
                    return c
                tasks.append(no_research(claim))
        
        updated_claims = await asyncio.gather(*tasks)
        return updated_claims

    async def _research_single_claim(self, claim: Claim) -> Claim:
        """Research a single claim and gather evidence sources."""
        updated_claim = claim.copy()
        
        try:
            query = f"{claim['text']} fact check"
            logger.info(f"Researching claim: {claim['id']} with query: '{query}'")

            # Use Tavily's search with answer and raw content
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="advanced",
                max_results=self.max_sources_per_claim,
                include_answer=True,
                include_raw_content=True
            )

            # Process the response
            if response.get('answer'):
                updated_claim['evidence_summary'] = response['answer']
                updated_claim['sources'] = self._process_search_results(response.get('results', []))
                updated_claim['status'] = ClaimStatus.RESEARCHING
            else:
                updated_claim['evidence_summary'] = "No direct answer found. Sources require manual review."
                updated_claim['sources'] = self._process_search_results(response.get('results', []))
                if not updated_claim['sources']:
                     updated_claim['status'] = ClaimStatus.UNVERIFIABLE
                else:
                    updated_claim['status'] = ClaimStatus.RESEARCHING

        except Exception as e:
            logger.error(f"Failed to research claim {claim['id']}: {e}", exc_info=True)
            updated_claim['status'] = ClaimStatus.UNVERIFIABLE
            updated_claim['verification_summary'] = f"Research failed: {str(e)}"
            updated_claim['evidence_summary'] = "An error occurred during evidence research."
        
        updated_claim['last_updated'] = datetime.utcnow().isoformat()
        return updated_claim

    def _process_search_results(self, search_results: List[Dict]) -> List[Source]:
        """Process raw search results from Tavily into Source objects."""
        sources = []
        for result in search_results:
            try:
                domain = urlparse(result['url']).netloc.lower().replace('www.', '')
                
                source = Source(
                    url=result['url'],
                    title=result['title'],
                    domain=domain,
                    content_snippet=(result.get('raw_content') or result.get('content') or "")[:1000] + "...",
                    source_type=SourceType.OTHER,  # Simplified typing
                    credibility_score=0.5,  # Neutral default, as direct credibility is not assessed here
                    relevance_score=result.get('score', 0.5),
                    publication_date=None, # Tavily response format doesn't guarantee this
                    author=None
                )
                sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to process search result '{result.get('url')}': {e}")
                continue
        return sources


async def research_claims(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, any]:
    """
    LangGraph node function for researching claims and gathering evidence.
    
    This function uses the simplified EvidenceResearcher to:
    1. Extract claims needing research from the state.
    2. Use the Tavily API to get a direct answer and supporting sources.
    3. Update claims with the evidence and new status.
    
    Args:
        state: Current graph state containing claims to research.
        config: Optional LangGraph configuration.
        
    Returns:
        A dictionary with the updated state fields.
        
    Raises:
        EvidenceResearchError: If the research process encounters a critical failure.
    """
    try:
        logger.info(f"Starting evidence research for session {state['session_id']}")
        
        researcher = EvidenceResearcher()
        
        claims_to_research = [
            claim for claim in state.get('claims', [])
            if claim.get('status') == ClaimStatus.PENDING
        ]
        
        if not claims_to_research:
            logger.info("No claims found that need research.")
            return {
                'workflow_stage': 'evidence_gathered',
                'status': {'current_step': 'researching', 'step_progress': 1.0}
            }
        
        logger.info(f"Researching {len(claims_to_research)} claims.")
        
        # Perform research for all pending claims
        researched_claims = await researcher.research_claims(claims_to_research)
        
        # Create a map of researched claims by ID
        researched_claims_map = {claim['id']: claim for claim in researched_claims}

        # Update the full list of claims in the state
        updated_claims_list = [
            researched_claims_map.get(claim['id'], claim) for claim in state.get('claims', [])
        ]
        
        # Calculate research statistics
        total_sources = sum(len(claim.get('sources', [])) for claim in researched_claims)
        successful_research = sum(
            1 for claim in researched_claims 
            if claim.get('status') not in [ClaimStatus.UNVERIFIABLE, ClaimStatus.PENDING]
        )
        
        logger.info(
            f"Research completed: {successful_research}/{len(claims_to_research)} claims "
            f"successfully researched, {total_sources} total sources found."
        )
        
        return {
            'claims': updated_claims_list,
            'workflow_stage': 'evidence_gathered',
            'status': {
                'current_step': 'researching',
                'step_progress': 1.0
            },
            'last_updated': datetime.utcnow().isoformat(),
            'performance_metrics': {
                'research_success_rate': (successful_research / len(claims_to_research)) if claims_to_research else 1.0,
                'total_sources_found': total_sources
            }
        }
        
    except Exception as e:
        logger.error(f"Evidence research failed: {e}", exc_info=True)
        error_message = f"Evidence research failed: {str(e)}"
        
        return {
            'workflow_stage': 'failed',
            'error_message': error_message,
            'status': {
                'current_step': 'researching',
                'error_count': state.get('status', {}).get('error_count', 0) + 1
            }
        }


def research_claims_sync(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, any]:
    """
    Synchronous wrapper for the evidence research node.
    """
    try:
        return asyncio.run(research_claims(state, config))
    except Exception as e:
        logger.error(f"Synchronous evidence research failed: {e}")
        return {
            'workflow_stage': 'failed',
            'error_message': f"Evidence research failed: {str(e)}"
        } 