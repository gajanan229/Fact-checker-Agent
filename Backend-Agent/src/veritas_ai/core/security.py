"""
Security and Safety Systems for Veritas AI.

This module implements comprehensive security measures including:
- Input validation and sanitization
- Content filtering for harmful inputs
- Rate limiting for user interactions
- Audit logging for all decisions
- Anti-manipulation safeguards
- Fact-checking for user-suggested claims
"""

import re
import time
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from pydantic import BaseModel, Field, validator
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets


# Configure security logger
security_logger = logging.getLogger("veritas_ai.security")
security_logger.setLevel(logging.INFO)

if not security_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)


class SecurityThreatLevel(Enum):
    """Security threat levels for classification."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class SecurityEventType(Enum):
    """Types of security events for logging."""
    INPUT_VALIDATION = "input_validation"
    CONTENT_FILTER = "content_filter"
    RATE_LIMIT = "rate_limit"
    PROMPT_INJECTION = "prompt_injection"
    FACT_CHECK_FAILURE = "fact_check_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    AUDIT_LOG = "audit_log"
    ANTI_MANIPULATION = "anti_manipulation"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    description: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary for logging."""
        threat_level_names = {1: "low", 2: "medium", 3: "high", 4: "critical"}
        return {
            "event_type": self.event_type.value,
            "threat_level": threat_level_names[self.threat_level.value],
            "description": self.description,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class InputValidationResult(BaseModel):
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    threat_level: SecurityThreatLevel
    violations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContentFilterResult(BaseModel):
    """Result of content filtering."""
    is_safe: bool
    filtered_content: str
    threat_level: SecurityThreatLevel
    filter_reasons: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RateLimitResult(BaseModel):
    """Result of rate limit check."""
    is_allowed: bool
    remaining_requests: int
    reset_time: datetime
    current_usage: int
    limit: int


class InputValidator:
    """Validates and sanitizes user inputs against security threats."""
    
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?",
            r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|rules?)",
            r"you\s+are\s+now\s+(?:a|an|\w+)",
            r"new\s+instructions?:",
            r"admin\s+mode",
            r"sudo\s+",
            r"bypass\s+security",
            r"access\s+all\s+(?:tools|data|files)",
            r"unlimited\s+permissions?",
            r"override\s+(?:security|safety|restrictions?)",
            r"forget\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|rules?)",
            r"act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:an?\s+)?(?:admin|root|system)",
            r"enable\s+(?:debug|developer|admin)\s+mode",
            r"\\x[0-9a-fA-F]{2}",  # Hex escape sequences
        ]
        
        # System prompt exposure patterns
        self.system_prompt_patterns = [
            r"show\s+(?:me\s+)?your\s+(?:system\s+)?(?:prompt|instructions?)",
            r"what\s+are\s+your\s+(?:system\s+)?instructions?",
            r"repeat\s+your\s+(?:system\s+)?(?:message|prompt|instructions?)",
            r"display\s+your\s+(?:system\s+)?(?:configuration|setup)",
            r"reveal\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        ]
        
        # Data exfiltration patterns
        self.exfiltration_patterns = [
            r"export\s+(?:all\s+)?(?:data|users?|emails?|passwords?)",
            r"download\s+(?:all\s+)?(?:data|files?|database)",
            r"list\s+(?:all\s+)?(?:users?|emails?|accounts?|data)",
            r"dump\s+(?:database|data|table)",
            r"select\s+\*\s+from\s+\w+",
            r"show\s+(?:tables?|databases?|schemas?)",
        ]
        
        # Maximum input lengths
        self.max_input_length = 5000
        self.max_claim_length = 1000
        
        # Compiled patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.injection_patterns]
        self.compiled_system_prompt = [re.compile(p, re.IGNORECASE) for p in self.system_prompt_patterns]
        self.compiled_exfiltration = [re.compile(p, re.IGNORECASE) for p in self.exfiltration_patterns]
    
    def validate_input(self, user_input: str, input_type: str = "general") -> InputValidationResult:
        """
        Validate and sanitize user input against various security threats.
        
        Args:
            user_input: Raw user input to validate
            input_type: Type of input (general, claim, url, etc.)
            
        Returns:
            InputValidationResult with validation details
        """
        violations = []
        threat_level = SecurityThreatLevel.LOW
        sanitized_input = user_input
        
        # Basic validation
        if not user_input or not user_input.strip():
            return InputValidationResult(
                is_valid=False,
                sanitized_input="",
                threat_level=SecurityThreatLevel.LOW,
                violations=["Empty input"]
            )
        
        # Length validation
        max_length = self.max_claim_length if input_type == "claim" else self.max_input_length
        if len(user_input) > max_length:
            violations.append(f"Input exceeds maximum length ({max_length})")
            threat_level = SecurityThreatLevel.MEDIUM
            sanitized_input = user_input[:max_length]
        else:
            sanitized_input = user_input
        
        # Check for prompt injection
        injection_found = False
        for pattern in self.compiled_injection:
            if pattern.search(user_input):
                violations.append(f"Potential prompt injection detected")
                threat_level = SecurityThreatLevel.HIGH
                injection_found = True
                break
        
        # Check for system prompt exposure attempts
        for pattern in self.compiled_system_prompt:
            if pattern.search(user_input):
                violations.append("System prompt exposure attempt detected")
                threat_level = SecurityThreatLevel.HIGH
                return InputValidationResult(
                    is_valid=False,
                    sanitized_input="I cannot share internal system information.",
                    threat_level=threat_level,
                    violations=violations
                )
        
        # Check for data exfiltration attempts
        for pattern in self.compiled_exfiltration:
            if pattern.search(user_input):
                violations.append("Potential data exfiltration attempt detected")
                threat_level = SecurityThreatLevel.CRITICAL
                injection_found = True
                break
        
        # Check for suspicious character sequences
        if self._contains_suspicious_chars(user_input):
            violations.append("Suspicious character sequences detected")
            threat_level = max(threat_level, SecurityThreatLevel.MEDIUM)
        
        # Check for excessive special characters
        if self._excessive_special_chars(user_input):
            violations.append("Excessive special characters detected")
            threat_level = max(threat_level, SecurityThreatLevel.MEDIUM)
        
        # Sanitize input if needed
        if injection_found or threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            sanitized_input = self._sanitize_malicious_input(user_input)
        else:
            sanitized_input = self._basic_sanitization(user_input)
        
        is_valid = threat_level not in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
        
        return InputValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input,
            threat_level=threat_level,
            violations=violations,
            metadata={
                "original_length": len(user_input),
                "sanitized_length": len(sanitized_input),
                "input_type": input_type
            }
        )
    
    def _contains_suspicious_chars(self, text: str) -> bool:
        """Check for suspicious character sequences that might indicate attacks."""
        suspicious_patterns = [
            r"\\[ux][0-9a-fA-F]+",  # Unicode/hex escapes
            r"\\[rnt]",  # Escape sequences
            r"\x00-\x1f",  # Control characters
            r"[{}()[\]<>\"'`].*[{}()[\]<>\"'`]",  # Multiple quotes/brackets
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _excessive_special_chars(self, text: str) -> bool:
        """Check if input contains excessive special characters."""
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_char_count > len(text) * 0.3  # More than 30% special chars
    
    def _sanitize_malicious_input(self, text: str) -> str:
        """Sanitize potentially malicious input by extracting safe parts."""
        # Split by sentences and keep only safe ones
        safe_parts = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains injection patterns
            is_safe = True
            for pattern in self.compiled_injection + self.compiled_exfiltration:
                if pattern.search(sentence):
                    is_safe = False
                    break
            
            if is_safe and len(sentence) > 5:  # Minimum length for meaningful content
                safe_parts.append(sentence)
        
        if safe_parts:
            return '. '.join(safe_parts)
        else:
            return "I can help you with legitimate fact-checking requests."
    
    def _basic_sanitization(self, text: str) -> str:
        """Basic sanitization for normal inputs."""
        # Remove control characters (except basic whitespace)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1f\x7f-\x9f]', '', text)
        
        # Convert newlines and tabs to spaces
        text = re.sub(r'[\n\r\t]+', ' ', text)
        
        # Normalize multiple spaces
        text = re.sub(r' +', ' ', text).strip()
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)
        
        return text


class ContentFilter:
    """Filters content for harmful, inappropriate, or malicious material."""
    
    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = [
            # Violence and threats
            r"\b(?:kill|murder|assault|attack|harm|hurt|violence|weapon|bomb|explosive)\b",
            # Hate speech indicators
            r"\b(?:hate|racist|discrimination|prejudice|supremacy)\b",
            # Misinformation indicators
            r"\b(?:conspiracy|hoax|fake\s+news|propaganda|disinformation)\b",
            # Personal information
            r"\b(?:ssn|social\s+security|credit\s+card|password|private\s+key)\b",
            # Explicit content
            r"\b(?:explicit|nsfw|adult\s+content|inappropriate)\b",
        ]
        
        # Toxic language patterns
        self.toxic_patterns = [
            r"\b(?:stupid|idiot|moron|dumb|retard)\b",
            r"\b(?:shut\s+up|screw\s+you|go\s+to\s+hell)\b",
            r"\b(?:worthless|pathetic|disgusting|revolting)\b",
        ]
        
        # Compile patterns
        self.compiled_harmful = [re.compile(p, re.IGNORECASE) for p in self.harmful_patterns]
        self.compiled_toxic = [re.compile(p, re.IGNORECASE) for p in self.toxic_patterns]
    
    def filter_content(self, content: str, strict_mode: bool = False) -> ContentFilterResult:
        """
        Filter content for harmful or inappropriate material.
        
        Args:
            content: Content to filter
            strict_mode: Whether to apply stricter filtering rules
            
        Returns:
            ContentFilterResult with filtering details
        """
        filter_reasons = []
        threat_level = SecurityThreatLevel.LOW
        confidence_score = 0.0
        
        # Check for harmful content
        harmful_matches = 0
        for pattern in self.compiled_harmful:
            matches = pattern.findall(content)
            if matches:
                harmful_matches += len(matches)
                filter_reasons.append(f"Harmful content detected: {matches[0]}")
        
        # Check for toxic language
        toxic_matches = 0
        for pattern in self.compiled_toxic:
            matches = pattern.findall(content)
            if matches:
                toxic_matches += len(matches)
                if strict_mode:
                    filter_reasons.append(f"Toxic language detected: {matches[0]}")
        
        # Calculate threat level and confidence
        total_words = len(content.split())
        harmful_ratio = harmful_matches / max(total_words, 1)
        toxic_ratio = toxic_matches / max(total_words, 1)
        
        if harmful_ratio > 0.1:  # More than 10% harmful words
            threat_level = SecurityThreatLevel.CRITICAL
            confidence_score = 0.9
        elif harmful_ratio > 0.05:  # More than 5% harmful words
            threat_level = SecurityThreatLevel.HIGH
            confidence_score = 0.7
        elif toxic_ratio > 0.15 and strict_mode:  # More than 15% toxic words in strict mode
            threat_level = SecurityThreatLevel.MEDIUM
            confidence_score = 0.5
        elif harmful_matches > 0 or (toxic_matches > 0 and strict_mode):
            threat_level = SecurityThreatLevel.LOW
            confidence_score = 0.3
        
        # Filter content if necessary
        is_safe = threat_level not in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
        
        if not is_safe:
            filtered_content = self._filter_harmful_content(content)
        elif strict_mode and toxic_matches > 0:
            filtered_content = self._filter_toxic_content(content)
        else:
            filtered_content = content
        
        return ContentFilterResult(
            is_safe=is_safe,
            filtered_content=filtered_content,
            threat_level=threat_level,
            filter_reasons=filter_reasons,
            confidence_score=confidence_score
        )
    
    def _filter_harmful_content(self, content: str) -> str:
        """Remove or replace harmful content."""
        filtered = content
        
        # Replace harmful words with placeholders
        for pattern in self.compiled_harmful:
            filtered = pattern.sub("[FILTERED]", filtered)
        
        # Clean up multiple consecutive filters
        filtered = re.sub(r'\[FILTERED\](\s*\[FILTERED\])+', '[FILTERED]', filtered)
        
        return filtered.strip()
    
    def _filter_toxic_content(self, content: str) -> str:
        """Remove or replace toxic language."""
        filtered = content
        
        # Replace toxic words with milder alternatives
        replacements = {
            r"\bstupid\b": "misguided",
            r"\bidiot\b": "person",
            r"\bdumb\b": "uninformed",
            r"\bshut\s+up\b": "please stop",
        }
        
        for pattern, replacement in replacements.items():
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        
        return filtered.strip()


class RateLimiter:
    """Rate limiting for user interactions to prevent abuse."""
    
    def __init__(self):
        # Rate limit storage (in production, use Redis or database)
        self.user_requests: Dict[str, List[datetime]] = {}
        self.global_requests: List[datetime] = []
        
        # Rate limit configurations
        self.limits = {
            "user_per_minute": 10,
            "user_per_hour": 100,
            "global_per_minute": 1000,
            "global_per_hour": 10000,
        }
        
        # Cleanup interval
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
    
    def check_rate_limit(self, user_id: str, action: str = "general") -> RateLimitResult:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: Unique identifier for the user
            action: Type of action (can have different limits)
            
        Returns:
            RateLimitResult with rate limit information
        """
        now = datetime.now()
        
        # Cleanup old requests periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
            self.last_cleanup = now
        
        # Get user's recent requests
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        user_reqs = self.user_requests[user_id]
        
        # Count requests in different time windows
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        user_minute_count = sum(1 for req_time in user_reqs if req_time > minute_ago)
        user_hour_count = sum(1 for req_time in user_reqs if req_time > hour_ago)
        
        global_minute_count = sum(1 for req_time in self.global_requests if req_time > minute_ago)
        global_hour_count = sum(1 for req_time in self.global_requests if req_time > hour_ago)
        
        # Check limits
        is_allowed = (
            user_minute_count < self.limits["user_per_minute"] and
            user_hour_count < self.limits["user_per_hour"] and
            global_minute_count < self.limits["global_per_minute"] and
            global_hour_count < self.limits["global_per_hour"]
        )
        
        # Calculate remaining requests (most restrictive limit)
        remaining_minute = self.limits["user_per_minute"] - user_minute_count
        remaining_hour = self.limits["user_per_hour"] - user_hour_count
        remaining = min(remaining_minute, remaining_hour)
        
        # Calculate reset time (next minute)
        reset_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Record the request if allowed
        if is_allowed:
            user_reqs.append(now)
            self.global_requests.append(now)
            user_minute_count += 1  # Update count for return value
        
        return RateLimitResult(
            is_allowed=is_allowed,
            remaining_requests=max(0, remaining - (1 if is_allowed else 0)),
            reset_time=reset_time,
            current_usage=user_minute_count,
            limit=self.limits["user_per_minute"]
        )
    
    def _cleanup_old_requests(self):
        """Remove old request records to prevent memory growth."""
        cutoff = datetime.now() - timedelta(hours=2)
        
        # Clean user requests
        for user_id in list(self.user_requests.keys()):
            old_reqs = self.user_requests[user_id]
            new_reqs = [req for req in old_reqs if req > cutoff]
            
            if new_reqs:
                self.user_requests[user_id] = new_reqs
            else:
                del self.user_requests[user_id]
        
        # Clean global requests
        self.global_requests = [req for req in self.global_requests if req > cutoff]


class FactChecker:
    """Fact-checking system for user-suggested claims to prevent misinformation injection."""
    
    def __init__(self):
        # Known misinformation patterns
        self.misinformation_patterns = [
            r"covid.*(?:fake|hoax|not\s+real)",
            r"coronavirus.*(?:fake|hoax|not\s+real)",
            r"vaccin\w*.*autism",
            r"earth.*flat",
            r"world.*flat",
            r"climate.*(?:fake|hoax)",
            r"global.*warming.*(?:fake|hoax)",
            r"election.*(?:rigged|stolen)",
        ]
        
        # Suspicious claim indicators
        self.suspicious_indicators = [
            r"\b(?:they|government|elite|media)\s+(?:don't\s+want\s+you\s+to\s+know|are\s+hiding)\b",
            r"\b(?:big\s+pharma|mainstream\s+media|deep\s+state)\b",
            r"\b(?:wake\s+up|sheeple|do\s+your\s+research)\b",
            r"\b(?:secret|hidden|cover[-\s]?up|conspiracy)\b",
        ]
        
        # Compile patterns
        self.compiled_misinfo = [re.compile(p, re.IGNORECASE) for p in self.misinformation_patterns]
        self.compiled_suspicious = [re.compile(p, re.IGNORECASE) for p in self.suspicious_indicators]
    
    async def check_claim(self, claim: str) -> Tuple[bool, str, List[str]]:
        """
        Check if a user-suggested claim contains potential misinformation.
        
        Args:
            claim: The claim to fact-check
            
        Returns:
            Tuple of (is_safe, explanation, flags)
        """
        flags = []
        
        # Check for known misinformation patterns
        for pattern in self.compiled_misinfo:
            if pattern.search(claim):
                flags.append("Contains known misinformation pattern")
                return False, "This claim contains patterns associated with known misinformation", flags
        
        # Check for suspicious indicators
        suspicious_count = 0
        for pattern in self.compiled_suspicious:
            if pattern.search(claim):
                suspicious_count += 1
                flags.append("Contains suspicious language patterns")
        
        # Check claim structure and language
        if self._is_overly_emotional(claim):
            flags.append("Uses overly emotional language")
            suspicious_count += 1
        
        if self._lacks_specificity(claim):
            flags.append("Lacks specific, verifiable details")
            suspicious_count += 1
        
        if self._contains_absolutes(claim):
            flags.append("Uses absolute language without evidence")
            suspicious_count += 1
        
        # Determine safety based on flags
        if suspicious_count >= 3:
            return False, "This claim shows multiple indicators of potential misinformation", flags
        elif suspicious_count >= 2:
            explanation = "This claim requires additional verification due to suspicious indicators"
            return True, explanation, flags  # Allow but flag for human review
        else:
            return True, "Claim appears suitable for fact-checking", flags
    
    def _is_overly_emotional(self, claim: str) -> bool:
        """Check if claim uses overly emotional language."""
        emotional_patterns = [
            r"\b(?:shocking|incredible|unbelievable|amazing|terrifying|outrageous)\b",
            r"\b(?:must\s+(?:see|know|watch)|you\s+won't\s+believe)\b",
            r"!{2,}",  # Multiple exclamation marks
        ]
        
        for pattern in emotional_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                return True
        return False
    
    def _lacks_specificity(self, claim: str) -> bool:
        """Check if claim lacks specific, verifiable details."""
        # Claims should have specific elements
        has_specifics = any([
            re.search(r"\b\d{4}\b", claim),  # Years
            re.search(r"\b(?:study|research|report|survey)\b", claim, re.IGNORECASE),  # Research terms
            re.search(r"\b(?:according\s+to|source|published)\b", claim, re.IGNORECASE),  # Attribution
            re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", claim),  # Proper names
        ])
        
        # Vague language indicators
        vague_patterns = [
            r"\b(?:some\s+people|many\s+believe|it\s+is\s+said|rumor\s+has\s+it)\b",
            r"\b(?:they\s+say|word\s+is|apparently|allegedly)\b",
        ]
        
        has_vague = any(re.search(p, claim, re.IGNORECASE) for p in vague_patterns)
        
        return has_vague and not has_specifics
    
    def _contains_absolutes(self, claim: str) -> bool:
        """Check if claim uses absolute language without evidence."""
        absolute_patterns = [
            r"\b(?:always|never|all|none|every|every\s+single|absolutely|definitely)\b",
            r"\b(?:proves?|proof|undeniable|irrefutable|fact)\b",
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                return True
        return False


class AuditLogger:
    """Comprehensive audit logging for all security and decision events."""
    
    def __init__(self, log_file: str = "logs/security_audit.log"):
        self.log_file = log_file
        self.events: List[SecurityEvent] = []
        
        # Setup file logger
        self.file_logger = logging.getLogger("veritas_ai.audit")
        self.file_logger.setLevel(logging.INFO)
        
        if not self.file_logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
    
    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        self.events.append(event)
        
        # Log to file
        event_data = event.to_dict()
        self.file_logger.info(json.dumps(event_data, default=str))
        
        # Log to console for high/critical threats
        if event.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            security_logger.warning(f"Security Alert: {event.description}")
    
    def log_decision(self, decision_type: str, decision_data: Dict[str, Any], 
                    user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Log system decisions for audit trail."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUDIT_LOG,
            threat_level=SecurityThreatLevel.LOW,
            description=f"System decision: {decision_type}",
            user_id=user_id,
            session_id=session_id,
            metadata=decision_data
        )
        self.log_event(event)
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events for a specific user."""
        user_events = [e for e in self.events if e.user_id == user_id]
        return sorted(user_events, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, int]:
        """Get summary of threats in the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp > cutoff]
        
        threat_level_names = {1: "low", 2: "medium", 3: "high", 4: "critical"}
        summary = {}
        for level in SecurityThreatLevel:
            summary[threat_level_names[level.value]] = sum(1 for e in recent_events if e.threat_level == level)
        
        return summary


class SecurityManager:
    """Main security manager that coordinates all security systems."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        
        # Initialize security components
        self.input_validator = InputValidator()
        self.content_filter = ContentFilter()
        self.rate_limiter = RateLimiter()
        self.fact_checker = FactChecker()
        self.audit_logger = AuditLogger()
        
        # Security configuration
        self.blocked_users: Set[str] = set()
        self.suspicious_users: Set[str] = set()
        
        # CSRF secret key (in production, load from secure config)
        self.csrf_secret = "veritas_ai_secret_key_2024_security"
        
        security_logger.info("Security Manager initialized")
    
    async def process_user_input(self, user_input: str, user_id: str, 
                                session_id: str, input_type: str = "general") -> Dict[str, Any]:
        """
        Process user input through all security checks.
        
        Args:
            user_input: Raw user input
            user_id: Unique user identifier
            session_id: Session identifier
            input_type: Type of input for specific validation rules
            
        Returns:
            Dictionary with security check results and processed input
        """
        start_time = time.time()
        
        # Check if user is blocked
        if user_id in self.blocked_users:
            event = SecurityEvent(
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                threat_level=SecurityThreatLevel.CRITICAL,
                description="Blocked user attempted access",
                user_id=user_id,
                session_id=session_id
            )
            self.audit_logger.log_event(event)
            
            return {
                "is_allowed": False,
                "reason": "Access denied",
                "processed_input": None,
                "security_flags": ["blocked_user"]
            }
        
        # Rate limiting check
        rate_result = self.rate_limiter.check_rate_limit(user_id)
        if not rate_result.is_allowed:
            event = SecurityEvent(
                event_type=SecurityEventType.RATE_LIMIT,
                threat_level=SecurityThreatLevel.MEDIUM,
                description="Rate limit exceeded",
                user_id=user_id,
                session_id=session_id,
                metadata={"current_usage": rate_result.current_usage, "limit": rate_result.limit}
            )
            self.audit_logger.log_event(event)
            
            return {
                "is_allowed": False,
                "reason": f"Rate limit exceeded. Try again after {rate_result.reset_time}",
                "processed_input": None,
                "security_flags": ["rate_limited"],
                "rate_limit": rate_result.model_dump()
            }
        
        # Input validation
        validation_result = self.input_validator.validate_input(user_input, input_type)
        
        if not validation_result.is_valid:
            event = SecurityEvent(
                event_type=SecurityEventType.INPUT_VALIDATION,
                threat_level=validation_result.threat_level,
                description="Input validation failed",
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "violations": validation_result.violations,
                    "original_input": user_input[:200] + "..." if len(user_input) > 200 else user_input
                }
            )
            self.audit_logger.log_event(event)
            
            # Track suspicious users
            if validation_result.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
                self.suspicious_users.add(user_id)
        
        # Content filtering
        content_result = self.content_filter.filter_content(
            validation_result.sanitized_input, 
            strict_mode=self.strict_mode
        )
        
        if not content_result.is_safe:
            event = SecurityEvent(
                event_type=SecurityEventType.CONTENT_FILTER,
                threat_level=content_result.threat_level,
                description="Content filtering triggered",
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "filter_reasons": content_result.filter_reasons,
                    "confidence_score": content_result.confidence_score
                }
            )
            self.audit_logger.log_event(event)
        
        # Additional fact-checking for claims
        fact_check_result = None
        if input_type == "claim":
            is_safe, explanation, flags = await self.fact_checker.check_claim(
                content_result.filtered_content
            )
            fact_check_result = {
                "is_safe": is_safe,
                "explanation": explanation,
                "flags": flags
            }
            
            if not is_safe:
                event = SecurityEvent(
                    event_type=SecurityEventType.FACT_CHECK_FAILURE,
                    threat_level=SecurityThreatLevel.HIGH,
                    description="Fact-check failed for user claim",
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"claim": content_result.filtered_content, "flags": flags}
                )
                self.audit_logger.log_event(event)
        
        # Determine overall result
        is_allowed = (
            validation_result.is_valid and
            content_result.is_safe and
            (fact_check_result is None or fact_check_result["is_safe"])
        )
        
        # Log successful processing
        if is_allowed:
            self.audit_logger.log_decision(
                "user_input_processed",
                {
                    "input_type": input_type,
                    "validation_threat_level": validation_result.threat_level.value,
                    "content_threat_level": content_result.threat_level.value,
                    "processing_time": time.time() - start_time
                },
                user_id=user_id,
                session_id=session_id
            )
        
        # Compile security flags
        security_flags = []
        if validation_result.violations:
            security_flags.extend(validation_result.violations)
        if content_result.filter_reasons:
            security_flags.extend(content_result.filter_reasons)
        if fact_check_result and fact_check_result["flags"]:
            security_flags.extend(fact_check_result["flags"])
        
        return {
            "is_allowed": is_allowed,
            "reason": "Processed successfully" if is_allowed else "Security checks failed",
            "processed_input": content_result.filtered_content if is_allowed else None,
            "security_flags": security_flags,
            "validation_result": validation_result.model_dump(),
            "content_result": content_result.model_dump(),
            "fact_check_result": fact_check_result,
            "rate_limit": rate_result.model_dump(),
            "processing_time": time.time() - start_time
        }
    
    def add_security_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add security headers to HTTP responses."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Permitted-Cross-Domain-Policies": "none",
        }
        
        # Merge with existing headers
        result_headers = headers.copy()
        result_headers.update(security_headers)
        
        return result_headers
    
    def generate_csrf_token(self, user_id: str, session_id: str) -> str:
        """Generate CSRF token for user session."""
        # Create token from user_id, session_id, and current time
        timestamp = str(int(time.time()))
        data = f"{user_id}:{session_id}:{timestamp}"
        
        # Use HMAC for secure token generation
        token = hashlib.sha256(f"{self.csrf_secret}:{data}".encode()).hexdigest()
        
        return f"{timestamp}:{token}"
    
    def validate_csrf_token(self, token: str, user_id: str, session_id: str, 
                           max_age: int = 3600) -> bool:
        """Validate CSRF token."""
        try:
            timestamp_str, token_hash = token.split(":", 1)
            timestamp = int(timestamp_str)
            
            # Check if token is expired
            if time.time() - timestamp > max_age:
                return False
            
            # Regenerate expected token with same timestamp
            data = f"{user_id}:{session_id}:{timestamp_str}"
            expected_hash = hashlib.sha256(f"{self.csrf_secret}:{data}".encode()).hexdigest()
            
            # Use constant-time comparison
            return secrets.compare_digest(token_hash, expected_hash)
        
        except (ValueError, IndexError):
            return False
    
    def block_user(self, user_id: str, reason: str):
        """Block a user from the system."""
        self.blocked_users.add(user_id)
        
        event = SecurityEvent(
            event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
            threat_level=SecurityThreatLevel.CRITICAL,
            description=f"User blocked: {reason}",
            user_id=user_id,
            metadata={"reason": reason}
        )
        self.audit_logger.log_event(event)
        
        security_logger.warning(f"User {user_id} has been blocked: {reason}")
    
    def get_security_status(self, user_id: str) -> Dict[str, Any]:
        """Get security status for a user."""
        recent_events = self.audit_logger.get_events_by_user(user_id, limit=10)
        threat_summary = self.audit_logger.get_threat_summary(hours=24)
        
        return {
            "is_blocked": user_id in self.blocked_users,
            "is_suspicious": user_id in self.suspicious_users,
            "recent_events": [e.to_dict() for e in recent_events],
            "threat_summary": threat_summary,
            "rate_limit_status": self.rate_limiter.check_rate_limit(user_id).model_dump()
        }


# Export main components
__all__ = [
    "SecurityManager",
    "InputValidator", 
    "ContentFilter",
    "RateLimiter",
    "FactChecker",
    "AuditLogger",
    "SecurityEvent",
    "SecurityEventType",
    "SecurityThreatLevel",
    "InputValidationResult",
    "ContentFilterResult",
    "RateLimitResult"
]