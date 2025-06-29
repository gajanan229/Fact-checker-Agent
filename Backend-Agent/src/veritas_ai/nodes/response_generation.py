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
    
    confidence_level: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the accuracy and completeness of the response"
    )
    
    @field_validator('response_text')
    @classmethod
    def validate_response_quality(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Response too short - must be at least 50 characters")
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
        
        try:
            # Check Gemini API usage limits
            api_usage_manager.check_and_increment_gemini()
            
            # Create prompts
            system_prompt = self._create_system_prompt(tone)
            claims_summary = self._prepare_claims_summary(claims)
            evidence_summary = self._prepare_evidence_summary(claims)
            
            user_prompt = f"""Content Summary:
{content_summary}

Claims to address:
{claims_summary}

Available evidence:
{evidence_summary}

Please provide a constructive response that:
1. Addresses each claim with evidence
2. Uses the provided sources for citations (reference as [SOURCE_X])
3. Maintains a respectful, educational tone
4. Is approximately 100-200 words"""
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            # Get LLM
            llm = self.llm_manager.get_llm()
            
            # Set up structured output
            structured_llm = llm.with_structured_output(GeneratedResponse)
            
            # Create chain
            chain = prompt | structured_llm
            
            # Generate response
            logger.info("Generating response with LLM")
            response = await chain.ainvoke({})
            
            # Apply citation formatting
            response = self._apply_citation_formatting(response, claims)
            logger.info("Successfully generated response")
            return response
            
        except APIUsageError as e:
            logger.error(f"API limit reached for Gemini in response generation: {e}")
            raise ResponseGenerationError(str(e))
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise ResponseGenerationError(f"Failed to generate response: {e}")
    
    def _apply_citation_formatting(
        self, 
        response: GeneratedResponse, 
        claims: List[Claim]
    ) -> GeneratedResponse:
        """Apply inline citation formatting to response"""
        
        # Collect all sources
        all_sources = []
        for claim in claims:
            if claim.get('sources'):
                all_sources.extend(claim['sources'])
        
        if not all_sources:
            return response
        
        # Format inline citations (Source: domain.com)
        text = response.response_text
        for i, source in enumerate(all_sources):
            placeholder = f"[SOURCE_{i}]"
            citation = f"(Source: {source['domain']})"
            text = text.replace(placeholder, citation)
        
        response.response_text = text
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
            
            # Update state
            return {
                "draft_response": response.response_text,
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