"""
Content Ingestion Node for Veritas AI

This module handles the ingestion and processing of TikTok content.
It uses Apify to extract the audio transcript from a video URL and then
leverages a Google Gemini model to clean the transcript for downstream processing.
"""

import os
import re
import logging
import asyncio
import requests
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

# Third-party imports
from apify_client import ApifyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain/LangGraph imports
from langchain_core.runnables import RunnableConfig

# Internal imports
from ..core.state import GraphState, RawContent
from ..core.validation import ValidationError
from ..utils.api_usage import api_usage_manager, APIUsageError


logger = logging.getLogger(__name__)


# Exception Classes
class ContentIngestionError(Exception):
    """Raised when content ingestion fails."""
    pass

class TranscriptExtractionError(ContentIngestionError):
    """Raised specifically when transcript extraction fails."""
    pass


class TikTokURLParser:
    """Handles TikTok URL parsing and validation."""
    
    TIKTOK_DOMAINS = [
        'tiktok.com',
        'www.tiktok.com',
        'm.tiktok.com',
        'vm.tiktok.com',
        'vt.tiktok.com'
    ]
    
    VIDEO_URL_PATTERN = re.compile(
        r'https?://(?:www\.|m\.|vm\.)?tiktok\.com/(?:@[\w.-]+/video/|v/|embed/|t/)(\d+)/?'
    )
    
    @classmethod
    def resolve_short_url(cls, url: str) -> str:
        """Resolves shortened TikTok URLs (vm.tiktok.com, vt.tiktok.com)."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.url
        except requests.RequestException as e:
            logger.error(f"Could not resolve shortened URL {url}: {e}")
            raise ValidationError("url", f"Failed to resolve shortened URL: {e}")

    @classmethod
    def parse_url(cls, url: str) -> Dict[str, Optional[str]]:
        """
        Parse TikTok URL and extract relevant information.
        
        Args:
            url: TikTok URL to parse.
            
        Returns:
            Dictionary containing parsed URL information.
            
        Raises:
            ValidationError: If URL is invalid or not a TikTok URL.
        """
        if not url or not isinstance(url, str):
            raise ValidationError("url", "URL must be a non-empty string")
            
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError("url", f"Invalid URL format: {e}")
            
        domain = parsed.netloc.lower()
        if not any(tiktok_domain in domain for tiktok_domain in cls.TIKTOK_DOMAINS):
            raise ValidationError("url", f"URL must be from a TikTok domain, got: {domain}")
            
        # Resolve shortened URL if necessary
        if 'vt.tiktok.com' in domain or 'vm.tiktok.com' in domain:
            logger.info(f"Resolving shortened TikTok URL: {url}")
            url = cls.resolve_short_url(url)
            logger.info(f"Resolved to: {url}")
            
        match = cls.VIDEO_URL_PATTERN.match(url)
        if not match:
            raise ValidationError("url", "URL does not appear to be a valid TikTok video URL after resolving.")
            
        return {
            'original_url': url,
            'video_id': match.group(1),
            'content_type': 'video',
            'domain': domain,
        }


class ApifyTranscriber:
    """Handles audio transcription using the Apify TikTok Scraper Actor."""
    
    ACTOR_ID = "scrape-creators/best-tiktok-transcripts-scraper"
    
    def __init__(self):
        """Initialize with Apify client."""
        api_key = os.getenv('APIFY_API_KEY')
        if not api_key:
            raise ContentIngestionError("APIFY_API_KEY environment variable is required.")
        
        self.client = ApifyClient(api_key)
        
    def get_transcript(self, video_url: str) -> str:
        """
        Get transcript from a TikTok video URL using an Apify Actor.
        
        Args:
            video_url: The URL of the TikTok video.
            
        Returns:
            The raw transcript text.
            
        Raises:
            ContentIngestionError: If transcription fails.
        """
        logger.info(f"Requesting transcript for {video_url} from Apify.")
        try:
            # Check API usage limit before making the call
            api_usage_manager.check_and_increment_apify()
            
            actor_run = self.client.actor(self.ACTOR_ID).call(
                run_input={"videos": [video_url]}
            )
            
            # Fetch results from the dataset
            dataset_items = []
            for item in self.client.dataset(actor_run["defaultDatasetId"]).iterate_items():
                dataset_items.append(item)

            if not dataset_items:
                raise ContentIngestionError("Apify actor returned no items.")
            
            transcript = dataset_items[0].get("transcript")
            if not transcript:
                raise TranscriptExtractionError("Transcript not found in Apify actor result.")
            
            logger.info(f"Successfully retrieved transcript for {video_url}.")
            return transcript
        
        except APIUsageError as e:
            logger.error(f"API limit reached for Apify: {e}")
            raise ContentIngestionError(str(e))
        except TranscriptExtractionError:
            # Re-raise transcript-specific errors without wrapping them
            raise
        except Exception as e:
            logger.error(f"Apify transcription failed for {video_url}: {e}")
            raise ContentIngestionError(f"Failed to get transcript from Apify: {e}")


class TranscriptCleaner:
    """Uses Gemini to clean and format the raw transcript."""

    def __init__(self):
        """Initialize the cleaner with a Gemini model."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"),
                temperature=0.0
            )
            self._setup_chain()
            logger.info("TranscriptCleaner initialized with Gemini model.")
        except Exception as e:
            logger.error(f"Failed to initialize TranscriptCleaner: {e}")
            raise ContentIngestionError(f"TranscriptCleaner setup failed: {e}")

    def _setup_chain(self):
        """Set up the LangChain prompt and chain for cleaning."""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert text processing assistant. Your task is to clean a raw video transcript. "
                "You must remove all timestamps (e.g., [00:00:00.000 --> 00:00:05.000]), speaker labels, special characters, and extra newline characters. "
                "Format the final output as a single, coherent paragraph. Do not add any commentary or explanation."
            ),
            (
                "human",
                "Please clean the following transcript:\n\n---\n\n{raw_transcript}"
            )
        ])
        
        self.chain = prompt | self.llm | StrOutputParser()

    async def clean_transcript_async(self, raw_transcript: str) -> str:
        """
        Cleans the raw transcript using the configured Gemini model.
        
        Args:
            raw_transcript: The raw transcript text from Apify.
            
        Returns:
            A single-paragraph, cleaned version of the transcript.
        """
        if not raw_transcript or not raw_transcript.strip():
            logger.warning("Empty raw transcript provided for cleaning.")
            return ""
        
        logger.info("Sending raw transcript to Gemini for cleaning.")
        try:
            # Check Gemini API usage limits
            api_usage_manager.check_and_increment_gemini()
            cleaned_transcript = await self.chain.ainvoke({"raw_transcript": raw_transcript})
            logger.info("Successfully cleaned transcript.")
            return cleaned_transcript.strip()
        except APIUsageError as e:
            logger.error(f"API limit reached for Gemini: {e}")
            raise ContentIngestionError(str(e))


# Main node function for LangGraph
async def ingest_content(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, any]:
    """
    LangGraph node for content ingestion.
    
    Args:
        state: Current graph state.
        config: Optional runtime configuration.
        
    Returns:
        Dictionary with state updates.
    """
    logger.info("Starting content ingestion node.")
    
    try:
        if not state.get('user_input') or not state['user_input'].get('video_url'):
            raise ContentIngestionError("No video URL provided in state for ingestion.")
        
        target_url = state['user_input']['video_url']
        
        # 1. Parse and validate URL
        url_info = TikTokURLParser.parse_url(target_url)
        
        # 2. Get raw transcript from Apify
        transcriber = ApifyTranscriber()
        raw_transcript = await asyncio.to_thread(transcriber.get_transcript, target_url)
        
        # 3. Clean the transcript with Gemini
        cleaner = TranscriptCleaner()
        cleaned_transcript = await cleaner.clean_transcript_async(raw_transcript)
        
        # 4. Populate state objects
        raw_content = RawContent(
            transcript=cleaned_transcript,
            video_metadata={
                'url': target_url,
                'content_type': url_info['content_type'],
                'video_id': url_info.get('video_id'),
                'domain': url_info['domain'],
            },
            extraction_errors=[]
        )
        
        processing_status = state.get('processing_status', {}).copy()
        processing_status.update({
            'content_ingestion_complete': True,
            'content_type': url_info['content_type'],
            'has_audio': bool(cleaned_transcript),
            'has_visual_text': False, # We no longer process OCR
        })
        
        logger.info(f"Content ingestion completed successfully for {target_url}.")
        
        return {
            'raw_content': raw_content,
            'processing_status': processing_status,
            'workflow_stage': 'content_extracted'
        }
        
    except TranscriptExtractionError:
        # Re-raise transcript extraction errors so they can be handled by the runner
        # with user-friendly messages
        raise
    except Exception as e:
        logger.error(f"Content ingestion node failed: {e}", exc_info=True)
        
        processing_status = state.get('processing_status', {}).copy()
        processing_status.update({
            'content_ingestion_complete': False,
            'content_ingestion_error': str(e),
        })
        
        return {
            'processing_status': processing_status,
            'error_message': f"Content ingestion failed: {e}",
            'workflow_stage': 'failed'
        }


# Synchronous wrapper for environments that need it
def ingest_content_sync(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, any]:
    """
    Synchronous wrapper for the content ingestion node.
    
    Args:
        state: Current graph state.
        config: Optional runtime configuration.
        
    Returns:
        Dictionary with state updates.
    """
    try:
        return asyncio.run(ingest_content(state, config))
    except TranscriptExtractionError:
        # Re-raise transcript extraction errors so they can be handled by the runner
        raise
    except Exception as e:
        logger.error(f"Sync wrapper for content ingestion failed: {e}")
        processing_status = state.get('processing_status', {}).copy()
        processing_status.update({
            'content_ingestion_complete': False,
            'content_ingestion_error': str(e),
        })
        return {
            'processing_status': processing_status,
            'error_message': f"Content ingestion wrapper failed: {e}",
            'workflow_stage': 'failed'
        } 