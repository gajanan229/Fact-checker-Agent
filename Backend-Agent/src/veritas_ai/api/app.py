"""
This module contains the Flask application factory and the API route definitions.
"""
import sys
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# This is a temporary solution to the pathing issue.
# A better solution would be to structure the project as a proper package.
sys.path.append(str(Path(__file__).parent.parent.parent))

from .manager import AnalysisManager

# Initialize the analysis manager singleton
manager = AnalysisManager()

def create_app():
    """Creates and configures a Flask application."""
    app = Flask(__name__)
    
    # Configure CORS to allow requests from the frontend development server
    CORS(app, resources={r"/api/*": {"origins": "*"}}) # Loosened for development

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint to confirm the server is running."""
        return jsonify({"status": "healthy"}), 200

    @app.route('/api/analyze', methods=['POST'])
    def analyze_video():
        """
        Starts a new analysis job for a given TikTok URL.
        Expects a JSON payload: {"video_url": "..."}
        Returns a session ID for the job: {"session_id": "..."}
        """
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({"error": "Missing 'video_url' in request body"}), 400

        video_url = data['video_url']
        
        try:
            session_id = manager.start_analysis(video_url)
            return jsonify({"session_id": session_id}), 202 # 202 Accepted
        except Exception as e:
            # In a real app, you'd want more specific error handling
            return jsonify({"error": f"Failed to start analysis: {str(e)}"}), 500

    @app.route('/api/status/<session_id>', methods=['GET'])
    def stream_status(session_id: str):
        """
        Streams status updates for a given analysis session using Server-Sent Events.
        """
        if not manager.session_exists(session_id):
            return jsonify({"error": "Session not found"}), 404
        
        # The 'text/event-stream' mimetype is crucial for SSE
        return Response(manager.get_event_stream(session_id), mimetype='text/event-stream')

    return app 