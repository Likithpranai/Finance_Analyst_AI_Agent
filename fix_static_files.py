#!/usr/bin/env python3
"""
Script to fix the static file serving in the API server
"""

import os
import re

# Define paths
API_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_server.py')

def fix_static_files_serving():
    """
    Add proper static file serving to the API server
    """
    print("Fixing static file serving in API server...")
    
    with open(API_SERVER_PATH, 'r') as f:
        content = f.read()
    
    # Check if the static folder is already properly configured
    if "static_folder='static'" not in content:
        # Add static folder configuration to Flask app initialization
        content = content.replace(
            "app = Flask(__name__, static_folder='frontend/build')",
            "app = Flask(__name__, static_folder='static', static_url_path='/static')"
        )
        
        # Add a route to serve static files
        if "def serve_static_files(filename):" not in content:
            static_route = """
# Serve static files (visualizations)
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)
"""
            # Insert before the serve function
            content = re.sub(
                r'(@app.route\(\'/\', defaults=\{\'path\': \'\'\}\)\n@app.route\(\'/\<path:path\>\'\)\ndef serve\(path\):)',
                f'{static_route}\n\\1',
                content
            )
    
    with open(API_SERVER_PATH, 'w') as f:
        f.write(content)
    
    print("Static file serving fixed in API server!")

if __name__ == "__main__":
    fix_static_files_serving()
