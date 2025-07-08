"""
This module provides the AnalysisManager class for orchestrating the fact-checking workflow.
"""
import uuid
import json
import threading
import time
from queue import Queue
from datetime import datetime, timedelta

# A simple in-memory store for session data.
# In a production environment with multiple server instances, you would replace this
# with a distributed store like Redis. For a single-instance deployment, this
# includes a garbage collection mechanism to prevent memory leaks.
SESSIONS: dict[str, dict] = {}

from .runner import run_graph_in_thread

class AnalysisManager:
    """Manages the lifecycle of analysis sessions."""

    def __init__(self, cleanup_interval_seconds=600, session_ttl_minutes=60):
        self._cleanup_interval = cleanup_interval_seconds
        self._session_ttl = timedelta(minutes=session_ttl_minutes)
        self._stop_cleanup_event = threading.Event()
        self.start_cleanup_thread()

    def start_analysis(self, video_url: str) -> str:
        """
        Starts a new analysis workflow in a background thread.
        """
        session_id = str(uuid.uuid4())
        
        SESSIONS[session_id] = {
            "queue": Queue(),
            "status": "running", # running, completed, error
            "last_updated": datetime.utcnow()
        }

        # The main graph execution happens in a separate thread
        # to avoid blocking the API server.
        thread = threading.Thread(
            target=run_graph_in_thread,
            # Pass the manager instance to the thread for callbacks
            args=(self, session_id, video_url, SESSIONS[session_id]["queue"])
        )
        thread.start()

        return session_id

    def mark_session_complete(self, session_id: str):
        """Marks a session as completed for later cleanup."""
        if session_id in SESSIONS:
            SESSIONS[session_id]["status"] = "completed"
            SESSIONS[session_id]["last_updated"] = datetime.utcnow()

    def mark_session_error(self, session_id: str):
        """Marks a session as errored for later cleanup."""
        if session_id in SESSIONS:
            SESSIONS[session_id]["status"] = "error"
            SESSIONS[session_id]["last_updated"] = datetime.utcnow()

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session ID is valid."""
        return session_id in SESSIONS

    def get_event_stream(self, session_id: str):
        """
        A generator function that yields Server-Sent Events for a session.
        """
        event_queue = SESSIONS.get(session_id, {}).get("queue")
        if not event_queue:
            return

        while True:
            message = event_queue.get()
            event_type = message.get("type")

            if not event_type or event_type == "end_stream":
                break
            
            payload = json.dumps(message.get("payload", {}))
            formatted_event = f"event: {event_type}\ndata: {payload}\n\n"
            yield formatted_event

    def _cleanup_sessions(self):
        """Periodically cleans up old, completed sessions to prevent memory leaks."""
        while not self._stop_cleanup_event.is_set():
            time.sleep(self._cleanup_interval)
            
            now = datetime.utcnow()
            sessions_to_delete = []
            
            for session_id, data in SESSIONS.items():
                is_finished = data.get("status") in ["completed", "error"]
                last_updated = data.get("last_updated", now)
                
                if is_finished and (now - last_updated > self._session_ttl):
                    sessions_to_delete.append(session_id)
            
            for session_id in sessions_to_delete:
                if session_id in SESSIONS:
                    del SESSIONS[session_id]

    def start_cleanup_thread(self):
        """Starts the background cleanup thread."""
        cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        cleanup_thread.start()

    def stop_cleanup(self):
        """Stops the cleanup thread gracefully."""
        self._stop_cleanup_event.set() 