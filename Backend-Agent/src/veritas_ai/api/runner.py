"""
This module contains the core graph execution logic.
"""
import logging
from queue import Queue
from typing import TYPE_CHECKING

from .utils import transform_state_for_frontend
from ..core.autonomous_graph import create_autonomous_graph
from ..core.validation import create_initial_state
from ..nodes.content_ingestion import TranscriptExtractionError

if TYPE_CHECKING:
    from .manager import AnalysisManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_graph_in_thread(manager: "AnalysisManager", session_id: str, video_url: str, event_queue: Queue):
    """
    Executes the full fact-checking graph for a given URL and puts status
    updates into a queue. This function is designed to be run in a background thread.
    """
    final_state = {}
    try:
        # 1. Initialize Graph and State
        event_queue.put({'type': 'update', 'payload': {
            "step": "graph_setup", "status": "in_progress", "displayText": "Initializing agent..."
        }})
        compiled_graph = create_autonomous_graph()
        initial_state = create_initial_state(video_url=video_url, target_type="video")
        final_state = initial_state.copy()

        config = {"thread_id": session_id, "recursion_limit": 15}

        # 2. Stream the graph execution
        event_queue.put({'type': 'update', 'payload': {
            "step": "graph_setup", "status": "completed", "displayText": "Agent initialized."
        }})

        node_to_step_map = {
            "ingest_content": ("ingest_content", "Ingesting Content..."),
            "identify_claims": ("identify_claims", "Identifying Verifiable Claims..."),
            "research_claims": ("research_claims", "Gathering Evidence..."),
            "generate_response": ("generate_response", "Drafting Response..."),
            "critique_response": ("critique_response", "Performing Quality Review..."),
        }

        for chunk in compiled_graph.stream(initial_state, config):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            final_state.update(node_output)

            if node_name in node_to_step_map:
                step_key, step_text = node_to_step_map[node_name]
                logging.info(f"Session {session_id}: Completed step {step_key}")
                event_queue.put({'type': 'update', 'payload': {
                    "step": step_key, "status": "completed", "displayText": step_text
                }})

        # 3. Process and send the final result
        logging.info(f"Session {session_id}: Graph execution complete. Formatting final state.")
        
        logging.debug(f"[{session_id}] Final Draft Response from Backend: %s", final_state.get("draft_response", "DRAFT RESPONSE NOT FOUND"))

        frontend_payload = transform_state_for_frontend(final_state)
        event_queue.put({'type': 'complete', 'payload': { "caseFile": frontend_payload }})
        manager.mark_session_complete(session_id)

    except TranscriptExtractionError as e:
        logging.error(f"Transcript extraction failed for session {session_id}: {e}", exc_info=True)
        user_friendly_message = (
            "Unable to extract transcript from the video. This may occur if the video "
            "does not contain clear audio or if the content cannot be processed. "
            "Please try again with a different video that has clear spoken content, "
            "or ensure the video URL is accessible and contains audio."
        )
        event_queue.put({'type': 'error', 'payload': {"message": user_friendly_message}})
        manager.mark_session_error(session_id)
    except Exception as e:
        logging.error(f"Error during graph execution for session {session_id}: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {str(e)}"
        event_queue.put({'type': 'error', 'payload': {"message": error_message}})
        manager.mark_session_error(session_id)

    finally:
        # 4. Signal the end of the stream
        logging.info(f"Session {session_id}: Signaling end of stream.")
        event_queue.put({"type": "end_stream"}) 