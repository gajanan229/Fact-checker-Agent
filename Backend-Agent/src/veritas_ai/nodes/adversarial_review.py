"""
Adversarial Review Node for Veritas AI

This module handles the internal quality assurance and critique of generated responses including:
- LLM-powered response critique and quality assessment
- Bias detection and fairness evaluation
- Revision recommendation engine with specific improvement suggestions
- Quality scoring across multiple dimensions
- Self-correction loops and iterative improvement
- Safety guardrails against misinformation injection
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Literal, Tuple, Any
from datetime import datetime, timezone
from enum import Enum

# Third-party imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

# Internal imports
from ..core.state import GraphState, Claim, Source, ClaimStatus, ResponseQuality, Critique
from ..nodes.response_generation import LLMManager, ResponseGenerationError

# Configure logging
logger = logging.getLogger(__name__)


class AdversarialReviewError(Exception):
    """Base exception for adversarial review errors"""
    pass


class BiasDetectionError(AdversarialReviewError):
    """Exception for bias detection failures"""
    pass


class QualityDimension(str, Enum):
    """Dimensions for quality assessment"""
    ACCURACY = "accuracy"
    OBJECTIVITY = "objectivity"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    TONE_APPROPRIATENESS = "tone_appropriateness"
    CITATION_QUALITY = "citation_quality"
    SAFETY = "safety"
    BIAS_ABSENCE = "bias_absence"


class RevisionPriority(str, Enum):
    """Priority levels for revision recommendations"""
    CRITICAL = "critical"      # Must fix before proceeding
    HIGH = "high"             # Should fix, major impact
    MEDIUM = "medium"         # Good to fix, moderate impact  
    LOW = "low"               # Minor improvement opportunity
    OPTIONAL = "optional"     # Style/preference only


class BiasType(str, Enum):
    """Types of bias that can be detected"""
    POLITICAL = "political"
    CULTURAL = "cultural"
    GENDER = "gender"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    SOCIOECONOMIC = "socioeconomic"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    ANCHORING = "anchoring"


class QualityScore(BaseModel):
    """Individual quality score for a specific dimension"""
    dimension: QualityDimension
    score: float = Field(ge=0.0, le=1.0, description="Quality score from 0.0 to 1.0")
    justification: str = Field(description="Explanation for the score")
    improvement_suggestions: List[str] = Field(default_factory=list, max_length=5)


class BiasDetection(BaseModel):
    """Bias detection result"""
    bias_type: BiasType
    severity: float = Field(ge=0.0, le=1.0, description="Bias severity from 0.0 to 1.0")
    evidence: str = Field(description="Specific evidence of bias in the text")
    mitigation_strategy: str = Field(description="How to address this bias")


class RevisionRecommendation(BaseModel):
    """Specific revision recommendation"""
    priority: RevisionPriority
    category: QualityDimension
    description: str = Field(description="What needs to be changed")
    specific_suggestion: str = Field(description="Concrete suggestion for improvement")
    rationale: str = Field(description="Why this change is needed")


class CritiqueAssessment(BaseModel):
    """Comprehensive critique assessment of a response"""
    
    overall_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall response quality score"
    )
    
    quality_scores: List[QualityScore] = Field(
        description="Detailed scores for each quality dimension"
    )
    
    detected_biases: List[BiasDetection] = Field(
        default_factory=list,
        description="Any biases detected in the response"
    )
    
    revision_recommendations: List[RevisionRecommendation] = Field(
        description="Specific recommendations for improvement"
    )
    
    is_revision_needed: bool = Field(
        description="Whether response needs revision before proceeding"
    )
    
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Any critical issues that must be addressed"
    )
    
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive aspects of the response"
    )
    
    confidence_in_assessment: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this critique assessment"
    )

    @field_validator('overall_quality_score')
    @classmethod
    def validate_overall_score(cls, v, info):
        """Validate that overall score aligns with individual scores"""
        if hasattr(info, 'data') and 'quality_scores' in info.data:
            individual_scores = [qs.score for qs in info.data['quality_scores']]
            if individual_scores:
                avg_score = sum(individual_scores) / len(individual_scores)
                if abs(v - avg_score) > 0.3:  # Allow some variance but not too much
                    logger.warning(f"Overall score {v} differs significantly from average {avg_score}")
        return v


class AdversarialReviewer:
    """Main class for adversarial review and critique of responses"""
    
    def __init__(self):
        """Initialize the adversarial reviewer"""
        self.llm_manager = LLMManager()
        
        # Quality thresholds
        self.min_quality_threshold = float(os.getenv('MIN_QUALITY_THRESHOLD', '0.7'))
        self.bias_severity_threshold = float(os.getenv('BIAS_SEVERITY_THRESHOLD', '0.3'))
        self.max_revisions = int(os.getenv('MAX_REVISIONS', '3'))
        
    def _create_critique_system_prompt(self) -> str:
        """Create comprehensive system prompt for response critique"""
        return """You are an expert fact-checking quality assurance reviewer. Your role is to rigorously evaluate fact-checking responses for accuracy, objectivity, clarity, and potential biases.

You must assess responses across these dimensions:
1. ACCURACY: Are the facts correct and properly sourced?
2. OBJECTIVITY: Is the tone neutral and evidence-based?
3. CLARITY: Is the response clear and well-structured?
4. COMPLETENESS: Does it address all key claims adequately?
5. TONE_APPROPRIATENESS: Is the tone constructive and educational?
6. CITATION_QUALITY: Are sources properly cited and credible?
7. SAFETY: Does it avoid harmful misinformation?
8. BIAS_ABSENCE: Is it free from political, cultural, or other biases?

Be especially vigilant for:
- Subtle confirmation bias or leading language
- Political partisanship or cultural insensitivity  
- Incomplete or misleading citations
- Overly aggressive or dismissive tone
- Missing context or nuance
- Technical inaccuracies or oversimplifications

Provide specific, actionable feedback with concrete suggestions for improvement. Your goal is to ensure the highest quality fact-checking responses that build trust and understanding."""

    def _create_bias_detection_prompt(self) -> str:
        """Create specialized prompt for bias detection"""
        return """You are a bias detection specialist. Analyze the given text for various types of bias including:

POLITICAL BIAS: Favoring one political viewpoint, party language, partisan framing
CULTURAL BIAS: Assumptions about cultural norms, stereotypes, ethnocentrism
GENDER BIAS: Gendered language, assumptions, stereotypes
RACIAL/ETHNIC BIAS: Racial stereotypes, assumptions, insensitive language
RELIGIOUS BIAS: Favoritism toward/against religious views, assumptions
SOCIOECONOMIC BIAS: Class assumptions, privilege assumptions
CONFIRMATION BIAS: Cherry-picking evidence, ignoring contradictory information
SELECTION BIAS: Unrepresentative examples, biased source selection
ANCHORING BIAS: Over-reliance on first information, insufficient adjustment

For each bias detected, provide:
1. The specific type of bias
2. Severity level (0.0 to 1.0)
3. Concrete evidence from the text
4. Specific mitigation strategy

Be thorough but fair - not every difference in perspective constitutes harmful bias."""

    def _prepare_response_context(self, draft_response: str, claims: List[Claim], sources: List[Source]) -> str:
        """Prepare context for response evaluation"""
        context_parts = [
            f"RESPONSE TO EVALUATE:\n{draft_response}\n",
            
            f"ORIGINAL CLAIMS BEING ADDRESSED:\n"
        ]
        
        for i, claim in enumerate(claims, 1):
            context_parts.append(f"{i}. {claim['text']} (Status: {claim['status'].value})")
        
        context_parts.append(f"\nAVAILABLE EVIDENCE SOURCES:\n")
        
        all_sources = []
        for claim in claims:
            if claim.get('sources'):
                all_sources.extend(claim['sources'])
        
        for i, source in enumerate(all_sources[:10], 1):  # Limit to first 10 sources
            context_parts.append(f"{i}. {source['title']} ({source['domain']})")
        
        return "\n".join(context_parts)

    async def conduct_quality_assessment(
        self, 
        draft_response: str, 
        claims: List[Claim],
        sources: List[Source] = None
    ) -> CritiqueAssessment:
        """
        Conduct comprehensive quality assessment of a draft response
        
        Args:
            draft_response: The response text to evaluate
            claims: Original claims being addressed
            sources: Available evidence sources
            
        Returns:
            CritiqueAssessment: Comprehensive evaluation results
        """
        try:
            logger.info("Starting quality assessment of draft response")
            
            # Prepare evaluation context
            response_context = self._prepare_response_context(draft_response, claims, sources or [])
            
            # Create critique prompt
            critique_prompt = ChatPromptTemplate.from_messages([
                ("system", self._create_critique_system_prompt()),
                ("human", """Please evaluate this fact-checking response comprehensively:

{response_context}

Provide a thorough assessment including:
1. Quality scores for each dimension (0.0 to 1.0)
2. Specific improvement recommendations
3. Overall quality assessment
4. Critical issues (if any)
5. Response strengths

Be thorough but constructive in your critique.""")
            ])
            
            # Set up LLM chain
            llm = self.llm_manager.get_llm()
            parser = PydanticOutputParser(pydantic_object=CritiqueAssessment)
            
            # Add format instructions to prompt
            critique_prompt = critique_prompt.partial(format_instructions=parser.get_format_instructions())
            
            # Execute critique
            chain = critique_prompt | llm | parser
            assessment = chain.invoke({
                "response_context": response_context
            })
            
            logger.info(f"Quality assessment completed with overall score: {assessment.overall_quality_score}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            raise AdversarialReviewError(f"Quality assessment failed: {e}")

    async def detect_biases(self, draft_response: str) -> List[BiasDetection]:
        """
        Detect potential biases in the response
        
        Args:
            draft_response: Response text to analyze for bias
            
        Returns:
            List[BiasDetection]: Detected biases and mitigation strategies
        """
        try:
            logger.info("Starting bias detection analysis")
            
            # Create bias detection prompt
            bias_prompt = ChatPromptTemplate.from_messages([
                ("system", self._create_bias_detection_prompt()),
                ("human", """Analyze this fact-checking response for potential biases:

TEXT TO ANALYZE:
{response_text}

Identify any biases present, their severity, and how to address them. Be thorough but fair in your analysis.""")
            ])
            
            # Set up LLM for bias detection
            llm = self.llm_manager.get_llm()
            
            # Create a simpler model for bias detection results
            class BiasDetectionResults(BaseModel):
                detected_biases: List[BiasDetection] = Field(default_factory=list)
            
            parser = PydanticOutputParser(pydantic_object=BiasDetectionResults)
            bias_prompt = bias_prompt.partial(format_instructions=parser.get_format_instructions())
            
            # Execute bias detection
            chain = bias_prompt | llm | parser
            results = chain.invoke({"response_text": draft_response})
            
            significant_biases = [
                bias for bias in results.detected_biases 
                if bias.severity >= self.bias_severity_threshold
            ]
            
            logger.info(f"Bias detection completed. Found {len(significant_biases)} significant biases")
            return significant_biases
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            raise BiasDetectionError(f"Bias detection failed: {e}")

    def generate_revision_recommendations(self, assessment: CritiqueAssessment) -> List[RevisionRecommendation]:
        """
        Generate prioritized revision recommendations based on assessment
        
        Args:
            assessment: The critique assessment results
            
        Returns:
            List[RevisionRecommendation]: Prioritized recommendations
        """
        recommendations = list(assessment.revision_recommendations)
        
        # Add recommendations based on quality scores
        for quality_score in assessment.quality_scores:
            if quality_score.score < 0.6:  # Below acceptable threshold
                for suggestion in quality_score.improvement_suggestions:
                    recommendation = RevisionRecommendation(
                        priority=RevisionPriority.HIGH if quality_score.score < 0.4 else RevisionPriority.MEDIUM,
                        category=quality_score.dimension,
                        description=f"Improve {quality_score.dimension.value}",
                        specific_suggestion=suggestion,
                        rationale=quality_score.justification
                    )
                    recommendations.append(recommendation)
        
        # Add recommendations for detected biases
        for bias in assessment.detected_biases:
            if bias.severity >= self.bias_severity_threshold:
                recommendation = RevisionRecommendation(
                    priority=RevisionPriority.CRITICAL if bias.severity > 0.7 else RevisionPriority.HIGH,
                    category=QualityDimension.BIAS_ABSENCE,
                    description=f"Address {bias.bias_type.value} bias",
                    specific_suggestion=bias.mitigation_strategy,
                    rationale=f"Detected {bias.bias_type.value} bias: {bias.evidence}"
                )
                recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {
            RevisionPriority.CRITICAL: 0,
            RevisionPriority.HIGH: 1,
            RevisionPriority.MEDIUM: 2,
            RevisionPriority.LOW: 3,
            RevisionPriority.OPTIONAL: 4
        }
        
        recommendations.sort(key=lambda r: priority_order[r.priority])
        
        return recommendations

    def should_require_revision(self, assessment: CritiqueAssessment) -> bool:
        """
        Determine if response requires revision based on assessment
        
        Args:
            assessment: The critique assessment
            
        Returns:
            bool: Whether revision is required
        """
        # Critical issues always require revision
        if assessment.critical_issues:
            return True
        
        # Overall quality below threshold requires revision
        if assessment.overall_quality_score < self.min_quality_threshold:
            return True
        
        # High-severity biases require revision
        high_severity_biases = [
            bias for bias in assessment.detected_biases
            if bias.severity >= self.bias_severity_threshold
        ]
        if high_severity_biases:
            return True
        
        # Critical or high priority recommendations require revision
        critical_recommendations = [
            rec for rec in assessment.revision_recommendations
            if rec.priority in [RevisionPriority.CRITICAL, RevisionPriority.HIGH]
        ]
        if critical_recommendations:
            return True
        
        return False

    async def critique_response(
        self, 
        draft_response: str, 
        claims: List[Claim],
        revision_count: int = 0
    ) -> Critique:
        """
        Perform comprehensive critique of a draft response
        
        Args:
            draft_response: The response to critique
            claims: Original claims being addressed
            revision_count: Current revision iteration count
            
        Returns:
            Critique: Complete critique with recommendations
        """
        try:
            logger.info(f"Starting comprehensive critique (revision #{revision_count})")
            
            # Gather all sources from claims
            all_sources = []
            for claim in claims:
                if claim.get('sources'):
                    all_sources.extend(claim['sources'])
            
            # Conduct quality assessment
            assessment = await self.conduct_quality_assessment(draft_response, claims, all_sources)
            
            # Detect additional biases
            detected_biases = await self.detect_biases(draft_response)
            assessment.detected_biases.extend(detected_biases)
            
            # Generate comprehensive revision recommendations
            all_recommendations = self.generate_revision_recommendations(assessment)
            
            # Determine if revision is needed
            needs_revision = self.should_require_revision(assessment)
            
            # Create comprehensive feedback text
            feedback_parts = []
            
            if assessment.strengths:
                feedback_parts.append("STRENGTHS:")
                for strength in assessment.strengths:
                    feedback_parts.append(f"• {strength}")
                feedback_parts.append("")
            
            if assessment.critical_issues:
                feedback_parts.append("CRITICAL ISSUES:")
                for issue in assessment.critical_issues:
                    feedback_parts.append(f"• {issue}")
                feedback_parts.append("")
            
            feedback_parts.append(f"OVERALL QUALITY SCORE: {assessment.overall_quality_score:.2f}")
            feedback_parts.append("")
            
            if all_recommendations:
                feedback_parts.append("REVISION RECOMMENDATIONS:")
                for rec in all_recommendations[:5]:  # Top 5 recommendations
                    feedback_parts.append(f"• [{rec.priority.value.upper()}] {rec.description}: {rec.specific_suggestion}")
            
            feedback_text = "\n".join(feedback_parts)
            
            # Convert quality scores to ResponseQuality format
            quality_metrics = ResponseQuality()
            for qs in assessment.quality_scores:
                if qs.dimension == QualityDimension.ACCURACY:
                    quality_metrics['accuracy_score'] = qs.score
                elif qs.dimension == QualityDimension.TONE_APPROPRIATENESS:
                    quality_metrics['tone_score'] = qs.score
                elif qs.dimension == QualityDimension.CITATION_QUALITY:
                    quality_metrics['citation_score'] = qs.score
                elif qs.dimension == QualityDimension.CLARITY:
                    quality_metrics['clarity_score'] = qs.score
            
            quality_metrics['overall_score'] = assessment.overall_quality_score
            
            # Create final critique
            critique = Critique(
                is_revision_needed=needs_revision,
                feedback_text=feedback_text,
                suggested_improvements=[rec.specific_suggestion for rec in all_recommendations[:10]],
                quality_assessment=quality_metrics,
                critique_timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            logger.info(f"Critique completed. Revision needed: {needs_revision}")
            return critique
            
        except Exception as e:
            logger.error(f"Error in response critique: {e}")
            raise AdversarialReviewError(f"Response critique failed: {e}")


# Node function for LangGraph integration
async def critique_response(state: GraphState) -> GraphState:
    """
    LangGraph node function for adversarial review of generated responses
    
    Args:
        state: Current graph state containing draft response and claims
        
    Returns:
        GraphState: Updated state with critique results
    """
    try:
        logger.info("Starting adversarial review node")
        
        # Extract required data from state
        draft_response = state.get('draft_response')
        claims = state.get('claims', [])
        revision_count = state.get('revision_count', 0)
        
        if not draft_response:
            raise AdversarialReviewError("No draft response found for critique")
        
        if not claims:
            logger.warning("No claims found for context in critique")
        
        # Initialize reviewer
        reviewer = AdversarialReviewer()
        
        # Perform critique
        critique = await reviewer.critique_response(
            draft_response=draft_response,
            claims=claims,
            revision_count=revision_count
        )
        
        # Update state with critique results
        updated_state = state.copy()
        updated_state['critique'] = critique
        
        # Update workflow stage based on results
        if critique['is_revision_needed'] and revision_count < reviewer.max_revisions:
            updated_state['workflow_stage'] = 'response_drafted'  # Send back for revision
            updated_state['revision_count'] = revision_count + 1
            logger.info(f"Revision needed. Iteration {revision_count + 1}")
        else:
            updated_state['workflow_stage'] = 'response_reviewed'
            logger.info("Response approved or max revisions reached")
        
        # Update status
        updated_state['status'] = {
            'current_step': 'reviewing',
            'step_progress': 1.0
        }
        
        updated_state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        logger.info("Adversarial review completed successfully")
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in adversarial review node: {e}")
        
        # Update state with error information
        error_state = state.copy()
        error_state['error_message'] = f"Adversarial review failed: {str(e)}"
        error_state['workflow_stage'] = 'failed'
        error_state['status'] = {
            'current_step': 'reviewing',
            'step_progress': 0.0
        }
        error_state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return error_state


def critique_response_sync(state: GraphState) -> GraphState:
    """
    Synchronous wrapper for the adversarial review node
    
    Args:
        state: Current graph state
        
    Returns:
        GraphState: Updated state with critique results
    """
    try:
        # Run the async function in the current event loop or create new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, critique_response(state))
                    return future.result()
            else:
                return loop.run_until_complete(critique_response(state))
        except RuntimeError:
            # No event loop in current thread, create a new one
            return asyncio.run(critique_response(state))
            
    except Exception as e:
        logger.error(f"Error in synchronous adversarial review wrapper: {e}")
        
        # Return error state
        error_state = state.copy()
        error_state['error_message'] = f"Adversarial review synchronous wrapper failed: {str(e)}"
        error_state['workflow_stage'] = 'failed'
        error_state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return error_state 