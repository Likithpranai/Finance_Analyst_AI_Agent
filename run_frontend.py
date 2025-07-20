#!/usr/bin/env python3
"""
Run Frontend Script

This script starts both the API server and the React development server
to provide a complete Finance Analyst AI Agent frontend experience.
"""

import os
import sys
import subprocess
import time
import signal
import atexit

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# Check if Node.js is installed
def check_nodejs():
    try:
        subprocess.run(['node', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Check if npm is installed
def check_npm():
    try:
        subprocess.run(['npm', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Install dependencies
def install_dependencies():
    print("Installing required Python packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'], check=True)
    
    if not os.path.exists(os.path.join(FRONTEND_DIR, 'node_modules')):
        print("Installing React dependencies (this may take a few minutes)...")
        subprocess.run(['npm', 'install'], cwd=FRONTEND_DIR, check=True)

# Start API server
def start_api_server():
    print("Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, 
        os.path.join(PROJECT_ROOT, 'api_server.py')
    ], env=os.environ.copy())
    return api_process

# Start React development server
def start_react_server():
    print("Starting React development server...")
    react_process = subprocess.Popen([
        'npm', 'start'
    ], cwd=FRONTEND_DIR, env=os.environ.copy())
    return react_process

# Clean up processes on exit
def cleanup_processes(processes):
    print("\nShutting down servers...")
    for process in processes:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":
    # Check for required API keys
    required_keys = ['GEMINI_API_KEY', 'ALPHA_VANTAGE_API_KEY']
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        print(f"Error: Missing required API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables before starting the frontend.")
        sys.exit(1)
    
    # Check for Node.js and npm
    if not check_nodejs() or not check_npm():
        print("Error: Node.js and npm are required to run the frontend.")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Start servers
        processes = []
        api_process = start_api_server()
        processes.append(api_process)
        
        # Wait a bit for API server to start
        time.sleep(2)
        
        react_process = start_react_server()
        processes.append(react_process)
        
        # Register cleanup function
        atexit.register(cleanup_processes, processes)
        
        print("\nFinance Analyst AI Frontend is starting...")
        print("API Server running at: http://localhost:5002")
        print("React Frontend running at: http://localhost:3000")
        print("\nPress Ctrl+C to stop all servers.")
        
        # Keep the script running
        while all(p.poll() is None for p in processes):
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cleanup_processes(processes)
