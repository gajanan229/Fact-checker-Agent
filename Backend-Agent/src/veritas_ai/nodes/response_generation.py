"""
Response Generation Node for Veritas AI

This module handles the generation of fact-checking responses including:
- LLM-powered response drafting with advanced prompt engineering
- Citation integration and source formatting
- Tone and style control for constructive dialogue
- Response length optimization and template system
- Quality assessment and metadata generation
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Literal, Tuple
from datetime import datetime, timezone
from enum import Enum

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Third-party imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

# Internal imports
from ..core.state import GraphState, Claim, Source, ClaimStatus, ResponseQuality
from ..utils.api_usage import api_usage_manager, APIUsageError

# Configure logging
logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    """Base exception for response generation errors"""
    pass


class ResponseTone(str, Enum):
    """Available response tones for generation"""
    EDUCATIONAL = "educational"
    EMPATHETIC = "empathetic" 
    DIRECT = "direct"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"


class ResponseLength(str, Enum):
    """Target response length categories"""
    BRIEF = "brief"          # 50-100 words
    STANDARD = "standard"    # 100-200 words
    DETAILED = "detailed"    # 200-400 words


class CitationStyle(str, Enum):
    """Citation formatting styles"""
    INLINE = "inline"        # (Source: domain.com)
    NUMBERED = "numbered"    # [1], [2], etc.
    EMBEDDED = "embedded"    # According to BBC News...


class NumberedSource(BaseModel):
    """A numbered source for citation display"""
    number: int = Field(description="Sequential number for citation reference")
    domain: str = Field(description="Source domain for display")
    title: Optional[str] = Field(default=None, description="Source title if available")
    url: Optional[str] = Field(default=None, description="Source URL if available")


class GeneratedResponse(BaseModel):
    """Structured output model for LLM-generated responses"""
    
    response_text: str = Field(
        description="The main response text addressing the claims with proper citations"
    )
    
    tone_assessment: ResponseTone = Field(
        description="The assessed tone of the generated response"
    )
    
    key_points: List[str] = Field(
        description="Main factual points addressed in the response",
        max_length=15
    )
    
    sources_cited: List[str] = Field(
        description="List of source URLs or domains referenced in the response"
    )
    
    numbered_sources: List[NumberedSource] = Field(
        description="List of numbered sources for separate display",
        default_factory=list
    )
    
    confidence_level: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the accuracy and completeness of the response"
    )
    
    @field_validator('response_text')
    @classmethod
    def validate_response_quality(cls, v):
        if len(v.strip()) < 30:  # Reduced from 50 to 30 for more reasonable minimum
            raise ValueError(f"Response too short - must be at least 30 characters, got {len(v.strip())}")
        if len(v) > 2000:
            raise ValueError("Response too long - must be under 2000 characters")
        return v.strip()


class LLMManager:
    """Manages multiple LLM providers with fallback support"""
    
    def __init__(self):
        """Initialize LLM manager with provider configuration"""
        self.primary_provider = os.getenv('PRIMARY_LLM_PROVIDER', 'gemini')
        self.llm_instances = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Google Gemini
        if os.getenv('GOOGLE_API_KEY'):
            try:
                self.llm_instances['gemini'] = ChatGoogleGenerativeAI(
                    model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite-preview-06-17'),
                    temperature=float(os.getenv('GEMINI_TEMPERATURE', '0.1')),
                    max_output_tokens=int(os.getenv('GEMINI_MAX_TOKENS', '32048'))
                )
                logger.info("Initialized Google Gemini LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # OpenAI GPT
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.llm_instances['openai'] = ChatOpenAI(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                    max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '22048'))
                )
                logger.info("Initialized OpenAI GPT LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        if not self.llm_instances:
            raise ResponseGenerationError("No LLM providers successfully initialized")
    
    def get_llm(self):
        """Get primary LLM instance"""
        if self.primary_provider in self.llm_instances:
            return self.llm_instances[self.primary_provider]
        else:
            # Return first available LLM
            return next(iter(self.llm_instances.values()))


class ResponseGenerator:
    """Main class for generating fact-checking responses"""
    
    def __init__(self):
        """Initialize response generator"""
        self.llm_manager = LLMManager()
        
    def _create_system_prompt(self, tone: ResponseTone) -> str:
        """Create system prompt based on desired tone"""
        base_prompt = """You are a helpful fact-checking assistant that provides accurate, well-sourced information to counter misinformation. Your goal is to educate and inform, not to attack or shame."""
        
        tone_additions = {
            ResponseTone.EDUCATIONAL: " Focus on clear explanations and learning opportunities.",
            ResponseTone.EMPATHETIC: " Show empathy and understanding. Acknowledge concerns while providing facts.",
            ResponseTone.DIRECT: " Be clear and straightforward with the facts.",
            ResponseTone.CONVERSATIONAL: " Use a friendly, approachable tone.",
            ResponseTone.FORMAL: " Maintain a professional, academic tone."
        }
        
        return base_prompt + tone_additions.get(tone, "")
    
    def _prepare_claims_summary(self, claims: List[Claim]) -> str:
        """Prepare structured summary of claims for LLM"""
        if not claims:
            return "No specific claims identified to fact-check."
        
        summaries = []
        for i, claim in enumerate(claims, 1):
            summary = f"{i}. CLAIM: {claim['text']}\n"
            summary += f"   STATUS: {claim['status'].value}\n"
            
            if claim.get('evidence_summary'):
                summary += f"   EVIDENCE: {claim['evidence_summary']}\n"
            
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _prepare_evidence_summary(self, claims: List[Claim]) -> str:
        """Prepare structured evidence summary from all sources"""
        all_sources = []
        source_index = 0
        
        for claim in claims:
            if claim.get('sources'):
                for source in claim['sources']:
                    all_sources.append({
                        **source,
                        'source_id': source_index
                    })
                    source_index += 1
        
        if not all_sources:
            return "No research sources available."
        
        evidence_parts = []
        for source in all_sources:
            evidence = f"[SOURCE_{source['source_id']}] {source['title']}\n"
            evidence += f"Domain: {source['domain']}\n" 
            evidence += f"Content: {source['content_snippet']}\n"
            evidence_parts.append(evidence)
        
        return "\n".join(evidence_parts)
    
    async def generate_response(
        self, 
        claims: List[Claim],
        content_summary: str,
        user_preferences: Dict = None
    ) -> GeneratedResponse:
        """Generate fact-checking response using LLM"""
        
        user_preferences = user_preferences or {}
        tone = ResponseTone(user_preferences.get('response_tone', 'educational'))
        
        # Validate inputs before proceeding
        if not claims:
            logger.warning("No claims provided for response generation")
            return self._create_fallback_response("No verifiable claims were found to address.")
        
        try:
            # Check Gemini API usage limits
            api_usage_manager.check_and_increment_gemini()
            
            # Create prompts
            system_prompt = self._create_system_prompt(tone)
            claims_summary = self._prepare_claims_summary(claims)
            evidence_summary = self._prepare_evidence_summary(claims)
            
            # Debug logging to track what's being passed to LLM
            logger.info(f"Generating response for {len(claims)} claims")
            logger.info(f"Content summary length: {len(content_summary)} chars")
            logger.info(f"Claims summary length: {len(claims_summary)} chars")
            logger.info(f"Evidence summary length: {len(evidence_summary)} chars")
            
            # Enhanced user prompt with stronger length guidance
            user_prompt = f"""Content Summary:
{content_summary}

Claims to address:
{claims_summary}

Available evidence:
{evidence_summary}

IMPORTANT: Please provide a constructive, detailed fact-checking response that:
1. Addresses each claim with specific evidence
2. Uses the provided sources for citations (reference as [SOURCE_X])
3. Maintains a respectful, educational tone
4. Must be AT LEAST 100 words long - provide detailed explanations
5. Include specific facts and context to help readers understand the truth

Your response should be comprehensive and informative, not just a brief statement."""
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            # Get LLM
            llm = self.llm_manager.get_llm()
            
            # Try structured output first, with fallback
            try:
                # Set up structured output
                structured_llm = llm.with_structured_output(GeneratedResponse)
                chain = prompt | structured_llm
                
                logger.info("Generating response with structured output")
                response = await chain.ainvoke({})
                
                # Validate response length
                if len(response.response_text.strip()) < 30:
                    logger.warning(f"Structured output too short ({len(response.response_text)} chars), trying fallback")
                    raise ValueError("Response too short, trying fallback")
                
                # Apply citation formatting
                response = self._apply_numbered_citations(response, claims)
                logger.info("Successfully generated structured response")
                return response
                
            except Exception as structured_error:
                logger.warning(f"Structured output failed: {structured_error}, trying plain text fallback")
                
                # Fallback to plain text generation
                plain_chain = prompt | llm | StrOutputParser()
                plain_response = await plain_chain.ainvoke({})
                
                # Create structured response from plain text
                fallback_response = self._create_structured_from_plain(plain_response, claims, tone)
                logger.info("Successfully generated fallback response")
                return fallback_response
            
        except APIUsageError as e:
            logger.error(f"API limit reached for Gemini in response generation: {e}")
            raise ResponseGenerationError(str(e))
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            # Return a basic fallback response instead of failing completely
            return self._create_fallback_response(
                "An error occurred during response generation. Please try again or contact support."
            )
    
    def _create_fallback_response(self, message: str) -> GeneratedResponse:
        """Create a basic fallback response when generation fails"""
        return GeneratedResponse(
            response_text=message,
            tone_assessment=ResponseTone.EDUCATIONAL,
            key_points=["Response generation encountered an issue"],
            sources_cited=[],
            confidence_level=0.1
        )
    
    def _create_structured_from_plain(self, plain_text: str, claims: List[Claim], tone: ResponseTone) -> GeneratedResponse:
        """Convert plain text response to structured format"""
        # Apply numbered citation formatting to plain text
        formatted_text = plain_text
        all_sources = []
        source_domains = set()
        
        # Collect unique sources
        for claim in claims:
            if claim.get('sources'):
                for source in claim['sources']:
                    domain = source['domain']
                    if domain not in source_domains:
                        all_sources.append(source)
                        source_domains.add(domain)
        
        # Create numbered sources and replace placeholders
        numbered_sources = []
        for i, source in enumerate(all_sources, 1):
            # Create numbered source for display
            numbered_source = NumberedSource(
                number=i,
                domain=source['domain'],
                title=source.get('title'),
                url=source.get('url')
            )
            numbered_sources.append(numbered_source)
            
            # Replace placeholders with numbered references
            placeholder = f"[SOURCE_{i-1}]"  # LLM uses 0-based indexing
            citation = f"[{i}]"
            formatted_text = formatted_text.replace(placeholder, citation)
        
        # Extract key points (simple heuristic)
        sentences = [s.strip() for s in formatted_text.split('.') if s.strip()]
        key_points = sentences[:5] if sentences else ["Response generated"]
        
        # Extract cited sources
        cited_domains = [source['domain'] for source in all_sources]
        
        return GeneratedResponse(
            response_text=formatted_text,
            tone_assessment=tone,
            key_points=key_points,
            sources_cited=cited_domains,
            numbered_sources=numbered_sources,
            confidence_level=0.8
        )
    
    def _apply_numbered_citations(
        self, 
        response: GeneratedResponse, 
        claims: List[Claim]
    ) -> GeneratedResponse:
        """Apply numbered citation formatting to response and create sources list"""
        
        # Collect all sources and remove duplicates
        all_sources = []
        source_domains = set()
        
        for claim in claims:
            if claim.get('sources'):
                for source in claim['sources']:
                    domain = source['domain']
                    # Avoid duplicate sources by domain
                    if domain not in source_domains:
                        all_sources.append(source)
                        source_domains.add(domain)
        
        if not all_sources:
            return response
        
        # Create numbered sources list
        numbered_sources = []
        text = response.response_text
        
        for i, source in enumerate(all_sources, 1):
            # Create numbered source for display
            numbered_source = NumberedSource(
                number=i,
                domain=source['domain'],
                title=source.get('title'),
                url=source.get('url')
            )
            numbered_sources.append(numbered_source)
            
            # Replace placeholders with numbered references
            placeholder = f"[SOURCE_{i-1}]"  # LLM uses 0-based indexing
            citation = f"[{i}]"
            text = text.replace(placeholder, citation)
        
        # Update response with numbered citations and sources
        response.response_text = text
        response.numbered_sources = numbered_sources
        
        return response


# Main node functions for LangGraph integration
async def generate_response(state: GraphState) -> GraphState:
    """
    Main LangGraph node for generating fact-checking responses.
    """
    logger.info(f"Starting response generation for session {state['session_id']}")
    
    try:
        claims = state.get('claims', [])
        
        # If no verifiable claims were found, we cannot generate a response.
        if not claims or all(c.get('status') == ClaimStatus.UNVERIFIABLE for c in claims):
            logger.warning("No verifiable claims available for response generation. Ending workflow.")
            return {
                "workflow_stage": "failed",
                "error_message": "No verifiable claims were found in the video. Please try another one."
            }

        content_summary = state.get('raw_content', {}).get('summary', '')
        
        generator = ResponseGenerator()
        # For simplicity, we'll just use the first set of preferences found
        user_preferences = state.get('user_preferences', {})

        # Generate response
        try:
            logger.info("Generating response with LLM")
            response = await generator.generate_response(
                claims=claims,
                content_summary=content_summary,
                user_preferences=user_preferences
            )
            
            logger.info(f"Successfully generated draft response with confidence: {response.confidence_level}")
            
            # Convert NumberedSource objects to dict format for state
            response_sources = [
                {
                    "number": source.number,
                    "domain": source.domain,
                    "title": source.title,
                    "url": source.url
                }
                for source in response.numbered_sources
            ]
            
            # Update state
            return {
                "draft_response": response.response_text,
                "response_sources": response_sources,
                "response_quality": {
                    "confidence_level": response.confidence_level,
                    "tone_assessment": response.tone_assessment.value,
                    "key_points": response.key_points
                },
                "workflow_stage": "response_generated"
            }
            
        except ResponseGenerationError as e:
            logger.error(f"Response generation failed: {e}")
            return {"error_message": str(e), "workflow_stage": "failed"}
            
    except Exception as e:
        logger.error(f"An unexpected error occurred in generate_response: {e}", exc_info=True)
        return {"error_message": "An unexpected error occurred during response generation.", "workflow_stage": "failed"}


def generate_response_sync(state: GraphState) -> GraphState:
    """Synchronous wrapper for response generation"""
    try:
        return asyncio.run(generate_response(state))
    except Exception as e:
        logger.error(f"Synchronous response generation failed: {e}")
        return {
            **state,
            'workflow_stage': 'failed',
            'error_message': f"Response generation failed: {str(e)}"
        } 