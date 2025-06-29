"""
This module provides the AnalysisManager class for orchestrating the fact-checking workflow.
"""
import uuid
import json
import threading
from queue import Queue

# A simple in-memory store for session data.
# In a production environment, you would replace this with Redis or a database.
SESSIONS: dict[str, Queue] = {}

from .runner import run_graph_in_thread

class AnalysisManager:
    """Manages the lifecycle of analysis sessions."""

    def start_analysis(self, video_url: str) -> str:
        """
        Starts a new analysis workflow in a background thread.

        Args:
            video_url: The URL of the TikTok video to analyze.

        Returns:
            A unique session ID for this analysis job.
        """
        session_id = str(uuid.uuid4())
        event_queue = Queue()
        SESSIONS[session_id] = event_queue

        # The main graph execution happens in a separate thread
        # to avoid blocking the API server.
        thread = threading.Thread(
            target=run_graph_in_thread,
            args=(session_id, video_url, event_queue)
        )
        thread.start()

        return session_id

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session ID is valid."""
        return session_id in SESSIONS

    def get_event_stream(self, session_id: str):
        """
        A generator function that yields Server-Sent Events for a session.
        This function will block and wait for new events from the graph runner.
        """
        event_queue = SESSIONS.get(session_id)
        if not event_queue:
            return

        while True:
            # Blocks until an item is available in the queue
            message = event_queue.get()

            event_type = message.get("type")
            if not event_type or event_type == "end_stream":
                break
            
            payload = json.dumps(message.get("payload", {}))
            
            # Correctly format as a named Server-Sent Event
            # The 'event:' line specifies the event name for the browser's EventSource listener.
            # The 'data:' line contains the JSON payload.
            # Two newlines are required to signal the end of the message.
            formatted_event = f"event: {event_type}\ndata: {payload}\n\n"
            yield formatted_event
        
        # Session cleanup is removed to prevent a race condition with client-side
        # EventSource retries. In-memory sessions will be cleared on server restart.
        # del SESSIONS[session_id] 