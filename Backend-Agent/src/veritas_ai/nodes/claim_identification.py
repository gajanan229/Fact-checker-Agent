"""
Claim Identification Node for Veritas AI.

This module implements the claim identification system that extracts verifiable
factual statements from raw content using LLM-powered analysis with structured output.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field, model_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

from ..core.state import GraphState, Claim, ClaimStatus
from ..core.validation import ValidationError


# Configure logging
logger = logging.getLogger(__name__)


class ExtractedClaim(BaseModel):
    """Pydantic model for individual claim extraction."""
    
    text: str = Field(
        description="The exact factual claim as stated in the content. Should be a complete, standalone statement."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this is a verifiable factual claim (0.0 to 1.0)"
    )
    claim_type: Literal["factual", "statistical", "historical", "scientific", "other"] = Field(
        description="Category of the claim for better processing"
    )
    context: str = Field(
        description="Surrounding context or explanation that helps understand the claim"
    )
    is_verifiable: bool = Field(
        description="Whether this claim can be fact-checked with external sources"
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_claim_quality(cls, values: dict) -> dict:
        """Validate claim meets quality standards."""
        text = values.get('text', '')
        confidence = values.get('confidence', 0.0)
        
        # Minimum text length
        if len(text.strip()) < 10:
            raise ValueError("Claim text too short - must be at least 10 characters")
        
        # Opinion indicators - lower confidence for these
        opinion_words = ['think', 'believe', 'feel', 'opinion', 'seems', 'appears', 'probably']
        if any(word in text.lower() for word in opinion_words):
            values['confidence'] = min(confidence, 0.6)
            values['is_verifiable'] = False
        
        return values


class ClaimExtractionResult(BaseModel):
    """Complete result of claim extraction process."""
    
    claims: List[ExtractedClaim] = Field(
        description="List of factual claims identified in the content"
    )
    total_claims_found: int = Field(
        description="Total number of claims identified"
    )
    content_analysis_summary: str = Field(
        description="Brief summary of the content analysis process and findings"
    )
    has_verifiable_claims: bool = Field(
        description="Whether any verifiable factual claims were found"
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_extraction_result(cls, values: dict) -> dict:
        """Validate the extraction results are consistent."""
        claims = values.get('claims', [])
        total_found = values.get('total_claims_found', 0)
        
        # Ensure count matches
        values['total_claims_found'] = len(claims)
        
        # Check if any claims are verifiable
        has_verifiable = any(
            getattr(claim, 'is_verifiable', False) if hasattr(claim, 'is_verifiable') 
            else claim.get('is_verifiable', False) if hasattr(claim, 'get')
            else False
            for claim in claims
        )
        values['has_verifiable_claims'] = has_verifiable
        
        return values


class ClaimIdentifier:
    """Core claim identification system using LLM analysis."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        """Initialize the claim identifier with LLM."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=32768
            )
            self.parser = PydanticOutputParser(pydantic_object=ClaimExtractionResult)
            self._setup_prompt_template()
            logger.info(f"ClaimIdentifier initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ClaimIdentifier: {e}")
            raise ValidationError("model_initialization", f"ClaimIdentifier setup failed: {e}")
    
    def _setup_prompt_template(self) -> None:
        """Set up the prompt template for claim extraction."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert fact-checking analyst specializing in identifying verifiable factual claims.

Your task is to extract ONLY factual claims that can be verified through external sources. 

FACTUAL CLAIMS include:
- Specific statistics, numbers, dates, or measurements
- Historical events or facts
- Scientific statements or data
- Specific assertions about people, places, or events
- Verifiable statements about cause and effect

DO NOT extract:
- Personal opinions or subjective statements
- Emotional expressions or feelings
- Hypothetical scenarios or speculation
- General advice or recommendations
- Rhetorical questions
- Vague or ambiguous statements

For each claim:
1. Extract the exact text as stated
2. Assess confidence based on specificity and verifiability
3. Categorize the type of claim
4. Provide context to understand the claim
5. Determine if it's actually verifiable

Be conservative - only extract claims that are clearly factual and verifiable.

{format_instructions}"""
            ),
            (
                "human",
                """Analyze the following content for factual claims:

CONTENT SOURCE: {content_source}
CONTENT TEXT:
{content_text}

Extract all verifiable factual claims from this content."""
            )
        ]).partial(format_instructions=self.parser.get_format_instructions())
    
    async def extract_claims_async(
        self, 
        content_text: str, 
        content_source: str = "unknown"
    ) -> ClaimExtractionResult:
        """Extract claims from content asynchronously."""
        try:
            if not content_text or not content_text.strip():
                logger.warning("Empty content provided for claim extraction")
                return ClaimExtractionResult(
                    claims=[],
                    total_claims_found=0,
                    content_analysis_summary="No content provided for analysis",
                    has_verifiable_claims=False
                )
            
            # Create the chain and invoke
            chain = self.prompt_template | self.llm | self.parser
            result = await chain.ainvoke({
                "content_text": content_text.strip(),
                "content_source": content_source
            })
            
            logger.info(f"Extracted {len(result.claims)} claims from {content_source}")
            return result
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return ClaimExtractionResult(
                claims=[],
                total_claims_found=0,
                content_analysis_summary=f"Extraction failed: {str(e)}",
                has_verifiable_claims=False
            )
    
    def extract_claims_sync(
        self, 
        content_text: str, 
        content_source: str = "unknown"
    ) -> ClaimExtractionResult:
        """Extract claims from content synchronously."""
        try:
            return asyncio.run(self.extract_claims_async(content_text, content_source))
        except Exception as e:
            logger.error(f"Synchronous claim extraction failed: {e}")
            return ClaimExtractionResult(
                claims=[],
                total_claims_found=0,
                content_analysis_summary=f"Extraction failed: {str(e)}",
                has_verifiable_claims=False
            )


class ClaimProcessor:
    """Processes and validates extracted claims."""
    
    @staticmethod
    def create_claim_objects(
        extracted_claims: List[ExtractedClaim],
        source_type: Literal["video", "comment"] = "video"
    ) -> List[Claim]:
        """Convert extracted claims to Claim objects with proper metadata."""
        claims = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        for extracted_claim in extracted_claims:
            # Only process verifiable claims with reasonable confidence
            if not extracted_claim.is_verifiable or extracted_claim.confidence < 0.3:
                logger.debug(f"Skipping low-confidence or unverifiable claim: {extracted_claim.text[:50]}...")
                continue
            
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
                last_updated=current_time
            )
            claims.append(claim)
        
        return claims
    
    @staticmethod
    def filter_duplicate_claims(claims: List[Claim]) -> List[Claim]:
        """Remove duplicate or very similar claims."""
        filtered_claims = []
        seen_texts = set()
        
        for claim in claims:
            # Simple deduplication based on text similarity
            claim_text = claim['text'].lower().strip()
            
            # Check for exact duplicates
            if claim_text in seen_texts:
                continue
            
            # Check for very similar claims using simple word overlap
            claim_words = set(claim_text.split())
            is_similar = any(
                len(claim_words.intersection(set(seen_text.split()))) >= max(3, len(claim_words) * 0.6)
                for seen_text in seen_texts
            )
            
            if not is_similar:
                filtered_claims.append(claim)
                seen_texts.add(claim_text)
            else:
                logger.debug(f"Filtered similar claim: {claim_text[:50]}...")
        
        logger.info(f"Filtered {len(claims) - len(filtered_claims)} duplicate/similar claims")
        return filtered_claims
    
    @staticmethod
    def validate_claims(claims: List[Claim]) -> List[Claim]:
        """Validate and clean up claim objects."""
        valid_claims = []
        
        for claim in claims:
            try:
                # Basic validation
                if not claim.get('text') or len(claim['text'].strip()) < 10:
                    logger.warning(f"Skipping invalid claim with insufficient text")
                    continue
                
                if not claim.get('id'):
                    claim['id'] = str(uuid.uuid4())
                
                if not claim.get('status'):
                    claim['status'] = ClaimStatus.PENDING
                
                # Ensure timestamps
                current_time = datetime.now(timezone.utc).isoformat()
                if not claim.get('created_at'):
                    claim['created_at'] = current_time
                if not claim.get('last_updated'):
                    claim['last_updated'] = current_time
                
                valid_claims.append(claim)
                
            except Exception as e:
                logger.warning(f"Failed to validate claim: {e}")
                continue
        
        return valid_claims


# Main node functions for LangGraph integration
async def identify_claims(state: GraphState) -> GraphState:
    """
    LangGraph node function to identify claims from raw content.
    
    Args:
        state: Current graph state containing raw content
        
    Returns:
        Updated state with identified claims
    """
    try:
        logger.info(f"Starting claim identification for session {state.get('session_id')}")
        
        # Extract content from state
        raw_content = state.get('raw_content', {})
        user_input = state.get('user_input', {})
        target_type = user_input.get('target_type', 'video')
        
        # Combine all available text content
        content_parts = []
        content_sources = []
        
        if raw_content.get('transcript'):
            content_parts.append(raw_content['transcript'])
            content_sources.append("video_transcript")
        
        if raw_content.get('ocr_text'):
            content_parts.append(raw_content['ocr_text'])
            content_sources.append("video_text_overlay")
        
        if raw_content.get('selected_comment_text'):
            content_parts.append(raw_content['selected_comment_text'])
            content_sources.append("comment_text")
        
        if not content_parts:
            logger.warning("No content available for claim identification")
            return {
                **state,
                'claims': [],
                'workflow_stage': 'claims_identified',
                'status': {
                    **state.get('status', {}),
                    'current_step': 'identifying_claims',
                    'warnings': [*state.get('status', {}).get('warnings', []), 
                               "No content available for claim identification"]
                }
            }
        
        # Combine content with source labels
        combined_content = "\n\n".join([
            f"[{source.upper()}]: {content}"
            for source, content in zip(content_sources, content_parts)
        ])
        
        # Initialize claim identifier
        identifier = ClaimIdentifier()
        
        # Extract claims
        extraction_result = await identifier.extract_claims_async(
            combined_content, 
            f"{target_type}_content"
        )
        
        # Process claims
        processor = ClaimProcessor()
        
        # Convert to Claim objects
        claim_objects = processor.create_claim_objects(
            extraction_result.claims,
            target_type
        )
        
        # Filter duplicates and validate
        filtered_claims = processor.filter_duplicate_claims(claim_objects)
        validated_claims = processor.validate_claims(filtered_claims)
        
        # Update state
        current_time = datetime.now(timezone.utc).isoformat()
        
        updated_state = {
            **state,
            'claims': validated_claims,
            'workflow_stage': 'claims_identified',
            'last_updated': current_time,
            'status': {
                **state.get('status', {}),
                'current_step': 'identifying_claims',
                'step_progress': 1.0,
            }
        }
        
        logger.info(f"Successfully identified {len(validated_claims)} claims")
        return updated_state
        
    except Exception as e:
        logger.error(f"Claim identification failed: {e}")
        return {
            **state,
            'workflow_stage': 'failed',
            'error_message': f"Claim identification failed: {str(e)}",
            'status': {
                **state.get('status', {}),
                'current_step': 'identifying_claims',
                'error_count': state.get('status', {}).get('error_count', 0) + 1
            }
        }


def identify_claims_sync(state: GraphState) -> GraphState:
    """
    Synchronous wrapper for claim identification.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with identified claims
    """
    try:
        return asyncio.run(identify_claims(state))
    except Exception as e:
        logger.error(f"Synchronous claim identification failed: {e}")
        return {
            **state,
            'workflow_stage': 'failed',
            'error_message': f"Claim identification failed: {str(e)}"
        }


def create_test_claims(content_text: str, count: int = 3) -> List[Claim]:
    """
    Create test claims for development and testing.
    
    Args:
        content_text: Content to create claims from
        count: Number of test claims to create
        
    Returns:
        List of test claims
    """
    test_claims = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for i in range(min(count, 5)):  # Max 5 test claims
        claim = Claim(
            id=str(uuid.uuid4()),
            text=f"Test claim {i+1} extracted from: {content_text[:50]}...",
            status=ClaimStatus.PENDING,
            confidence_score=0.8,
            verification_summary="",
            evidence_summary="",
            sources=[],
            extracted_from="video",
            created_at=current_time,
            last_updated=current_time
        )
        test_claims.append(claim)
    
    return test_claims 