"""
Human Interaction Node for Veritas AI.

This module implements the human-in-the-loop system that processes user feedback,
classifies intent, validates input, and manages collaborative chat interactions
with safety guardrails against misinformation injection.

Updated to integrate with the comprehensive HITL infrastructure.
"""

import asyncio
import logging
import uuid
import re
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any, Union, Tuple
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.types import Command, interrupt

from ..core.state import GraphState, ChatMessage, Claim, ClaimStatus
from ..core.validation import ValidationError
from ..core.hitl import HITLManager, InterruptType, SessionStatus, hitl_interrupt_node


# Configure logging
logger = logging.getLogger(__name__)


class FeedbackIntent(str, Enum):
    """Types of user feedback intent."""
    STYLISTIC = "stylistic"  # Tone, length, format changes
    FACTUAL = "factual"      # Content accuracy, new evidence
    APPROVAL = "approval"    # Accept/reject current version  
    CLARIFICATION = "clarification"  # Questions about process
    NEW_CLAIM = "new_claim"  # User suggesting additional claims
    GENERAL = "general"      # Other feedback


class FeedbackPriority(str, Enum):
    """Priority levels for feedback processing."""
    CRITICAL = "critical"    # Must address immediately
    HIGH = "high"           # Should address before finalizing
    MEDIUM = "medium"       # Nice to have improvements
    LOW = "low"            # Optional suggestions


class SafetyRisk(str, Enum):
    """Safety risk levels for user input."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IntentClassification(BaseModel):
    """Classification of user feedback intent."""
    
    primary_intent: FeedbackIntent = Field(
        description="Primary intent category of the user feedback"
    )
    secondary_intents: List[FeedbackIntent] = Field(
        default_factory=list,
        description="Additional intent categories if applicable"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the intent classification (0.0 to 1.0)"
    )
    reasoning: str = Field(
        description="Explanation of why this intent was assigned"
    )
    requires_fact_check: bool = Field(
        description="Whether this feedback requires fact-checking validation"
    )
    
    @field_validator('secondary_intents')
    @classmethod
    def validate_secondary_intents(cls, v: List[FeedbackIntent]) -> List[FeedbackIntent]:
        """Ensure secondary intents don't duplicate primary."""
        return list(set(v))  # Remove duplicates


class SafetyAssessment(BaseModel):
    """Safety assessment of user input."""
    
    risk_level: SafetyRisk = Field(
        description="Overall safety risk level of the input"
    )
    contains_misinformation: bool = Field(
        description="Whether input contains potential misinformation"
    )
    contains_harmful_content: bool = Field(
        description="Whether input contains harmful or inappropriate content"
    )
    manipulation_attempt: bool = Field(
        description="Whether this appears to be an attempt to manipulate the system"
    )
    safety_concerns: List[str] = Field(
        default_factory=list,
        description="Specific safety concerns identified"
    )
    mitigation_needed: bool = Field(
        description="Whether mitigation measures are needed"
    )


class ProcessedFeedback(BaseModel):
    """Fully processed and validated user feedback."""
    
    original_feedback: str = Field(
        description="Original user feedback text"
    )
    sanitized_feedback: str = Field(
        description="Sanitized and cleaned feedback text"
    )
    intent_classification: IntentClassification = Field(
        description="Classification of feedback intent"
    )
    safety_assessment: SafetyAssessment = Field(
        description="Safety assessment of the feedback"
    )
    priority: FeedbackPriority = Field(
        description="Processing priority for this feedback"
    )
    actionable_items: List[str] = Field(
        default_factory=list,
        description="Specific actionable items extracted from feedback"
    )
    suggested_claims: List[str] = Field(
        default_factory=list,
        description="Any new claims suggested by the user (requires validation)"
    )
    approval_status: Optional[Literal["approved", "rejected", "needs_revision"]] = Field(
        default=None,
        description="User approval status if applicable"
    )


class HumanInteractionManager:
    """Core human interaction processing system."""
    
    def __init__(self, primary_llm: str = "gemini-2.5-flash-lite-preview-06-17", hitl_manager: Optional[HITLManager] = None):
        """Initialize the human interaction manager."""
        try:
            # Primary LLM for intent classification and safety assessment
            self.llm = ChatGoogleGenerativeAI(
                model=primary_llm,
                temperature=0.1,
                max_tokens=2048
            )
            
            # Fallback LLM
            try:
                self.fallback_llm = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0.1,
                    max_tokens=2048
                )
            except:
                self.fallback_llm = None
                logger.warning("Fallback LLM not available")
            
            # HITL Manager integration
            self.hitl_manager = hitl_manager
            
            self._setup_prompts()
            logger.info(f"HumanInteractionManager initialized with {primary_llm}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HumanInteractionManager: {e}")
            raise ValidationError("model_initialization", f"HumanInteractionManager setup failed: {e}")
    
    def _setup_prompts(self) -> None:
        """Set up prompt templates for different processing tasks."""
        
        # Intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at classifying user feedback intent in a fact-checking context.

Analyze user feedback and classify its primary intent:

STYLISTIC: Changes to tone, length, format, writing style
- "Make it shorter", "Use simpler language", "Be more formal"

FACTUAL: Content accuracy, evidence quality, claims verification
- "This source is wrong", "You missed this evidence", "Check this claim too"

APPROVAL: Accept/reject current response version
- "This looks good", "Publish this", "Start over", "I don't approve"

CLARIFICATION: Questions about the process or findings
- "Why did you conclude this?", "How confident are you?", "What does this mean?"

NEW_CLAIM: User suggesting additional claims to fact-check
- "Also check if X is true", "What about this other statement?"

GENERAL: Other feedback not fitting above categories

Respond in JSON format with: primary_intent, secondary_intents, confidence, reasoning, requires_fact_check"""
            ),
            (
                "human",
                """Context: {context}

User feedback: "{feedback}"

Classify this feedback intent:"""
            )
        ])
        
        # Safety assessment prompt
        self.safety_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a safety assessment expert for fact-checking systems.

Analyze user input for safety risks:

RISK LEVELS:
- NONE: Safe, normal feedback
- LOW: Minor concerns, proceed with caution
- MEDIUM: Moderate risk, needs attention
- HIGH: Significant risk, limit functionality
- CRITICAL: Immediate threat, block operation

CHECK FOR:
1. Misinformation injection attempts
2. Harmful or inappropriate content
3. System manipulation attempts
4. Coordinated inauthentic behavior

Respond in JSON: risk_level, contains_misinformation, contains_harmful_content, manipulation_attempt, safety_concerns, mitigation_needed"""
            ),
            (
                "human",
                """Assess this user input for safety risks:

"{user_input}"

Provide safety assessment:"""
            )
        ])
        
        # Setup parsers
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentClassification)
        self.safety_parser = PydanticOutputParser(pydantic_object=SafetyAssessment)
        
        # Add format instructions
        self.intent_prompt = self.intent_prompt.partial(
            format_instructions=self.intent_parser.get_format_instructions()
        )
        self.safety_prompt = self.safety_prompt.partial(
            format_instructions=self.safety_parser.get_format_instructions()
        )
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input by removing potentially harmful content."""
        if not user_input:
            return ""
        
        sanitized = user_input.strip()
        
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]*>', '', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove potentially malicious patterns
        malicious_patterns = [
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'on\w+\s*=',
        ]
        for pattern in malicious_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 5000:
            sanitized = sanitized[:4997] + "..."
            logger.warning("User input truncated due to length")
        
        return sanitized
    
    async def classify_intent(
        self, 
        user_feedback: str, 
        context_summary: str = ""
    ) -> IntentClassification:
        """Classify the intent of user feedback."""
        try:
            chain = self.intent_prompt | self.llm | self.intent_parser
            result = await chain.ainvoke({
                "user_feedback": user_feedback,
                "context_summary": context_summary
            })
            
            logger.info(f"Classified feedback intent: {result.primary_intent}")
            return result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback classification
            return IntentClassification(
                primary_intent=FeedbackIntent.GENERAL,
                secondary_intents=[],
                confidence=0.3,
                reasoning=f"Classification failed, defaulting to general: {e}",
                requires_fact_check=True  # Conservative approach
            )
    
    async def assess_safety(self, user_input: str) -> SafetyAssessment:
        """Assess safety risks in user input."""
        try:
            chain = self.safety_prompt | self.llm | self.safety_parser
            result = await chain.ainvoke({
                "user_input": user_input
            })
            
            logger.info(f"Safety assessment completed: {result.risk_level}")
            return result
            
        except Exception as e:
            logger.error(f"Safety assessment failed: {e}")
            # Conservative fallback
            return SafetyAssessment(
                risk_level=SafetyRisk.MEDIUM,
                contains_misinformation=True,  # Conservative
                contains_harmful_content=False,
                manipulation_attempt=False,
                safety_concerns=[f"Assessment failed: {e}"],
                mitigation_needed=True
            )
    
    def determine_priority(
        self, 
        intent: IntentClassification, 
        safety: SafetyAssessment
    ) -> FeedbackPriority:
        """Determine processing priority based on intent and safety."""
        
        # Critical safety issues take highest priority
        if safety.risk_level in [SafetyRisk.CRITICAL, SafetyRisk.HIGH]:
            return FeedbackPriority.CRITICAL
        
        # Approval/rejection feedback is high priority
        if intent.primary_intent == FeedbackIntent.APPROVAL:
            return FeedbackPriority.HIGH
        
        # Factual feedback requiring fact-checking is high priority
        if intent.primary_intent == FeedbackIntent.FACTUAL or intent.requires_fact_check:
            return FeedbackPriority.HIGH
        
        # New claims need validation - medium priority
        if intent.primary_intent == FeedbackIntent.NEW_CLAIM:
            return FeedbackPriority.MEDIUM
        
        # Stylistic and clarification feedback is lower priority
        if intent.primary_intent in [FeedbackIntent.STYLISTIC, FeedbackIntent.CLARIFICATION]:
            return FeedbackPriority.MEDIUM
        
        return FeedbackPriority.LOW
    
    def extract_actionable_items(
        self, 
        feedback: str, 
        intent: IntentClassification
    ) -> List[str]:
        """Extract specific actionable items from feedback."""
        actionable_items = []
        
        # Simple pattern matching for common feedback types
        if intent.primary_intent == FeedbackIntent.STYLISTIC:
            if "shorter" in feedback.lower():
                actionable_items.append("Reduce response length")
            if "longer" in feedback.lower():
                actionable_items.append("Expand response with more detail")
            if "formal" in feedback.lower():
                actionable_items.append("Use more formal tone")
            if "simple" in feedback.lower() or "easier" in feedback.lower():
                actionable_items.append("Simplify language and explanations")
        
        elif intent.primary_intent == FeedbackIntent.FACTUAL:
            if "source" in feedback.lower():
                actionable_items.append("Review and verify sources")
            if "evidence" in feedback.lower():
                actionable_items.append("Add additional evidence")
            if "wrong" in feedback.lower() or "incorrect" in feedback.lower():
                actionable_items.append("Fact-check and correct information")
        
        elif intent.primary_intent == FeedbackIntent.APPROVAL:
            if any(word in feedback.lower() for word in ["approve", "good", "publish", "ready"]):
                actionable_items.append("User approved - ready to finalize")
            elif any(word in feedback.lower() for word in ["reject", "wrong", "start over", "redo"]):
                actionable_items.append("User rejected - needs revision")
        
        # If no specific items found, add general item
        if not actionable_items:
            actionable_items.append("Review and address user feedback")
        
        return actionable_items
    
    def extract_suggested_claims(self, feedback: str) -> List[str]:
        """Extract any new claims suggested by the user."""
        suggested_claims = []
        
        # Look for patterns indicating new claims
        claim_patterns = [
            r"check if (.+?)(?:\.|$|\?)",
            r"what about (.+?)(?:\.|$|\?)",
            r"also verify (.+?)(?:\.|$|\?)",
            r"is it true that (.+?)(?:\.|$|\?)",
        ]
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, feedback, re.IGNORECASE)
            for match in matches:
                claim = match.strip()
                if len(claim) > 10 and len(claim) < 500:  # Reasonable claim length
                    suggested_claims.append(claim)
        
        return suggested_claims
    
    def detect_approval_status(self, feedback: str) -> Optional[Literal["approved", "rejected", "needs_revision"]]:
        """Detect if feedback contains approval/rejection status."""
        feedback_lower = feedback.lower()
        
        # Approval indicators
        approval_words = ["approve", "approved", "good", "looks good", "publish", "ready", "accept", "fine"]
        if any(word in feedback_lower for word in approval_words):
            return "approved"
        
        # Rejection indicators
        rejection_words = ["reject", "rejected", "wrong", "incorrect", "start over", "redo", "bad"]
        if any(word in feedback_lower for word in rejection_words):
            return "rejected"
        
        # Revision indicators
        revision_words = ["revise", "change", "modify", "improve", "fix", "update"]
        if any(word in feedback_lower for word in revision_words):
            return "needs_revision"
        
        return None
    
    async def process_feedback(
        self, 
        user_feedback: str, 
        context_summary: str = ""
    ) -> ProcessedFeedback:
        """Process user feedback comprehensively."""
        try:
            # Sanitize input
            sanitized_feedback = self.sanitize_input(user_feedback)
            
            if not sanitized_feedback:
                raise ValidationError("empty_feedback", "No valid feedback provided")
            
            # Run intent classification and safety assessment in parallel
            intent_task = self.classify_intent(sanitized_feedback, context_summary)
            safety_task = self.assess_safety(sanitized_feedback)
            
            intent, safety = await asyncio.gather(intent_task, safety_task)
            
            # Determine priority
            priority = self.determine_priority(intent, safety)
            
            # Extract actionable items and suggestions
            actionable_items = self.extract_actionable_items(sanitized_feedback, intent)
            suggested_claims = self.extract_suggested_claims(sanitized_feedback)
            approval_status = self.detect_approval_status(sanitized_feedback)
            
            result = ProcessedFeedback(
                original_feedback=user_feedback,
                sanitized_feedback=sanitized_feedback,
                intent_classification=intent,
                safety_assessment=safety,
                priority=priority,
                actionable_items=actionable_items,
                suggested_claims=suggested_claims,
                approval_status=approval_status
            )
            
            logger.info(f"Processed feedback with priority: {priority} and intent: {intent.primary_intent}")
            return result
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            raise ValidationError("feedback_processing", f"Failed to process feedback: {e}")


class CollaborationManager:
    """Manages collaborative chat history and human-AI interactions."""
    
    @staticmethod
    def create_chat_message(
        role: Literal["user", "assistant", "system"],
        content: str,
        message_type: Optional[Literal["feedback", "suggestion", "question", "approval"]] = None,
        referenced_claim_ids: Optional[List[str]] = None
    ) -> ChatMessage:
        """Create a structured chat message."""
        return ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message_type=message_type,
            referenced_claim_ids=referenced_claim_ids or []
        )
    
    @staticmethod
    def update_chat_history(
        state: GraphState,
        processed_feedback: ProcessedFeedback
    ) -> List[ChatMessage]:
        """Update collaboration notes with processed feedback."""
        
        existing_notes = state.get('collaboration_notes', [])
        
        # Add user feedback message
        user_message = CollaborationManager.create_chat_message(
            role="user",
            content=processed_feedback.original_feedback,
            message_type=processed_feedback.intent_classification.primary_intent.value
        )
        
        # Add system analysis message
        analysis_content = f"Feedback analysis: Intent={processed_feedback.intent_classification.primary_intent.value}, Priority={processed_feedback.priority.value}, Safety={processed_feedback.safety_assessment.risk_level.value}"
        
        system_message = CollaborationManager.create_chat_message(
            role="system",
            content=analysis_content,
            message_type="suggestion"
        )
        
        updated_notes = existing_notes + [user_message, system_message]
        
        return updated_notes


async def process_user_feedback(state: GraphState) -> Union[GraphState, Command]:
    """
    Enhanced human-in-the-loop node using modern HITL infrastructure.
    
    This function:
    1. Uses HITLManager for session management
    2. Creates proper interrupts with Command objects
    3. Processes feedback with safety assessment
    4. Manages collaborative workflows
    5. Supports resume with different actions
    """
    
    try:
        logger.info("Starting enhanced user feedback processing")
        
        # Get or create session information
        session_id = state.get("session_id")
        user_id = state.get("user_id", "anonymous")
        
        # Initialize HITL manager if not provided
        hitl_manager = state.get("hitl_manager")
        if not hitl_manager:
            from langgraph.checkpoint.memory import MemorySaver
            hitl_manager = HITLManager(checkpointer=MemorySaver())
            logger.info("Created new HITLManager for session")
        
        # Create or retrieve session
        if session_id:
            session = hitl_manager.get_session(session_id)
            if not session:
                # Session expired or invalid, create new one
                session = hitl_manager.create_session(user_id=user_id)
                logger.info(f"Created new session {session.session_id} (previous expired)")
        else:
            session = hitl_manager.create_session(user_id=user_id)
            logger.info(f"Created new session {session.session_id}")
        
        # Update state with session information
        state = {
            **state,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "thread_id": session.thread_id
        }
        
        # Initialize interaction manager with HITL integration
        interaction_manager = HumanInteractionManager(hitl_manager=hitl_manager)
        
        # Create context for the interrupt
        context = {
            "workflow_stage": state.get("workflow_stage", "unknown"),
            "claims_summary": [
                {"text": claim.get('text', ''), "status": claim.get('status', '')} 
                for claim in state.get('claims', [])
            ],
            "draft_response": state.get('draft_response', ''),
            "current_step": state.get('status', {}).get('current_step', ''),
            "collaboration_mode": session.collaboration_mode.value
        }
        
        # Determine interrupt type based on workflow stage
        workflow_stage = state.get("workflow_stage", "user_collaborating")
        if workflow_stage == "response_drafted":
            interrupt_type = InterruptType.APPROVAL_REQUEST
            question = "Please review and approve this fact-checking response:"
            options = ["approve", "reject", "request_changes", "collaborate"]
        elif workflow_stage == "claims_identified":
            interrupt_type = InterruptType.HUMAN_REVIEW
            question = "Please review the identified claims and provide feedback:"
            options = ["continue", "add_claims", "modify_claims", "reject"]
        else:
            interrupt_type = InterruptType.FEEDBACK_REQUEST
            question = "Please provide your feedback on the current analysis:"
            options = ["continue", "modify", "restart"]
        
        # Create interrupt using HITL manager
        user_response = await hitl_manager.create_interrupt(
            session_id=session.session_id,
            interrupt_type=interrupt_type,
            question=question,
            context=context,
            options=options,
            priority="high" if interrupt_type == InterruptType.APPROVAL_REQUEST else "medium"
        )
        
        # Handle different response types
        if isinstance(user_response, dict):
            action = user_response.get("action", "continue")
            feedback_text = user_response.get("feedback", "")
            data = user_response.get("data", {})
        elif isinstance(user_response, str):
            # Simple text response
            feedback_text = user_response
            action = "continue"
            data = {}
        else:
            logger.warning(f"Unexpected response type: {type(user_response)}")
            feedback_text = str(user_response)
            action = "continue"
            data = {}
        
        # Process feedback if provided
        processed_feedback = None
        if feedback_text and feedback_text.strip():
            context_summary = f"Workflow stage: {workflow_stage}, Claims: {len(state.get('claims', []))}"
            processed_feedback = await interaction_manager.process_feedback(
                feedback_text, 
                context_summary
            )
            
            # Check for critical safety issues
            if processed_feedback.safety_assessment.risk_level == SafetyRisk.CRITICAL:
                logger.error(f"Critical safety risk detected: {processed_feedback.safety_assessment.safety_concerns}")
                return {
                    **state,
                    "workflow_stage": "failed",
                    "error_message": f"Critical safety risk: {processed_feedback.safety_assessment.safety_concerns}",
                    "session_id": session.session_id,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
        
        # Update collaboration history
        if processed_feedback:
            updated_notes = CollaborationManager.update_chat_history(state, processed_feedback)
            chat_messages = state.get('chat_history', [])
            chat_messages.append(HumanMessage(content=processed_feedback.original_feedback))
        else:
            updated_notes = state.get('collaboration_notes', [])
            chat_messages = state.get('chat_history', [])
        
        # Determine next action based on user response
        if action == "approve":
            # User approved - proceed to completion
            next_stage = "completed"
            chat_messages.append(AIMessage(content="Response approved by user. Ready for finalization."))
            
            return hitl_manager.create_resume_command(
                session_id=session.session_id,
                action="approve",
                goto_node="finalize_response",
                update_state={
                    "workflow_stage": next_stage,
                    "collaboration_notes": updated_notes,
                    "chat_history": chat_messages,
                    "user_approval": True,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            )
            
        elif action == "reject":
            # User rejected - go back to regenerate
            next_stage = "response_generation"
            chat_messages.append(AIMessage(content="Response rejected. Will regenerate with feedback."))
            
            return hitl_manager.create_resume_command(
                session_id=session.session_id,
                action="reject",
                goto_node="generate_response",
                update_state={
                    "workflow_stage": next_stage,
                    "collaboration_notes": updated_notes,
                    "chat_history": chat_messages,
                    "user_feedback": processed_feedback.model_dump() if processed_feedback else {},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            )
            
        elif action == "request_changes" or action == "collaborate":
            # User wants to collaborate on changes
            next_stage = "user_collaborating"
            chat_messages.append(AIMessage(content="Entering collaborative mode for response refinement."))
            
            return hitl_manager.create_resume_command(
                session_id=session.session_id,
                action="collaborate",
                update_state={
                    "workflow_stage": next_stage,
                    "collaboration_notes": updated_notes,
                    "chat_history": chat_messages,
                    "collaboration_active": True,
                    "user_feedback": processed_feedback.model_dump() if processed_feedback else {},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            )
            
        elif action == "add_claims" and processed_feedback:
            # User wants to add new claims
            if processed_feedback.suggested_claims:
                next_stage = "claims_identification"
                chat_messages.append(AIMessage(content="New claims suggested. Will research additional evidence."))
                
                return hitl_manager.create_resume_command(
                    session_id=session.session_id,
                    action="add_claims",
                    goto_node="identify_claims",
                    update_state={
                        "workflow_stage": next_stage,
                        "suggested_claims": processed_feedback.suggested_claims,
                        "collaboration_notes": updated_notes,
                        "chat_history": chat_messages,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                )
        
        # Default: continue with current workflow
        next_stage = state.get("workflow_stage", "user_collaborating")
        chat_messages.append(AIMessage(content="Continuing with workflow based on user feedback."))
        
        return hitl_manager.create_resume_command(
            session_id=session.session_id,
            action="continue",
            update_state={
                "workflow_stage": next_stage,
                "collaboration_notes": updated_notes,
                "chat_history": chat_messages,
                "user_feedback": processed_feedback.model_dump() if processed_feedback else {},
                "session_id": session.session_id,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced feedback processing failed: {e}")
        
        # Create error recovery command
        return Command(
            update={
                **state,
                "workflow_stage": "failed",
                "error_message": f"Feedback processing failed: {e}",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        )


def process_user_feedback_sync(state: GraphState) -> GraphState:
    """Synchronous wrapper for the user feedback processing node."""
    try:
        return asyncio.run(process_user_feedback(state))
    except Exception as e:
        logger.error(f"Sync wrapper failed: {e}")
        return {
            **state,
            "workflow_stage": "failed",
            "error_message": f"User feedback processing failed: {e}"
        }


# Export for testing
__all__ = [
    "FeedbackIntent",
    "FeedbackPriority", 
    "SafetyRisk",
    "IntentClassification",
    "SafetyAssessment",
    "ProcessedFeedback",
    "HumanInteractionManager",
    "CollaborationManager",
    "process_user_feedback",
    "process_user_feedback_sync"
] 