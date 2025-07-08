"""
This module contains the Flask application factory and the API route definitions.
"""
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging

from .manager import AnalysisManager
from ..utils.api_usage import api_usage_manager

# Initialize the analysis manager singleton
manager = AnalysisManager()

def create_app():
    """Creates and configures a Flask application."""
    app = Flask(__name__)
    
    # --- Secret Key Configuration ---
    # Crucial for session security. Must be set in production.
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
    is_debug = app.config.get("DEBUG")

    if not is_debug and not app.config['SECRET_KEY']:
        raise ValueError("No SECRET_KEY set for production. Please set the SECRET_KEY environment variable.")

    # --- Environment-based CORS Configuration ---
    # Load allowed origins from environment variable, defaulting to local dev server
    allowed_origins_str = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173"
    )
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]

    CORS(app,
         resources={r"/api/*": {"origins": allowed_origins}},
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"],
         supports_credentials=True # Often needed for more complex auth
    )

    # --- Security Headers ---
    @app.after_request
    def add_security_headers(response):
        # Prevent browsers from incorrectly sniffing MIME types
        response.headers['X-Content-Type-Options'] = 'nosniff'
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        # Enforce HTTPS
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

    # --- Global Error Handler ---
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Log all unhandled exceptions and return a generic error message."""
        # Log the exception with stack trace
        logging.error(f"Unhandled exception caught: {e}", exc_info=True)
        # Return a generic error response to the client
        response = {
            "error": "An internal server error occurred. Please try again later."
        }
        return jsonify(response), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint to confirm the server is running."""
        return jsonify({"status": "healthy"}), 200

    @app.route('/api/limits/status', methods=['GET'])
    def limits_status():
        """Returns the current status of API usage limits."""
        return jsonify(api_usage_manager.get_limits_status()), 200

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
            # This specific exception is now handled by the global handler
            # We can re-raise it to be caught and logged centrally
            raise e

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