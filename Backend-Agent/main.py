import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.veritas_ai.api.app import create_app
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

app = create_app()

if __name__ == '__main__':
    # Note: Use '0.0.0.0' to make the server accessible on your network
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 8000))
    debug = os.environ.get("API_DEBUG", "True").lower() == "true"
    
    print(f"🚀 Starting Veritas AI Backend Server at http://{host}:{port}")
    print(f"   Debug mode: {'on' if debug else 'off'}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)
