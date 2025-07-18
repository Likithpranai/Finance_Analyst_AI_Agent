#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd "$(dirname "$0")/backend"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2
echo "Backend server running on http://localhost:8000"

# Start the frontend server
echo "Starting frontend server..."
cd "$(dirname "$0")/frontend"
npm start &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
  echo "Shutting down servers..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit 0
}

# Register the cleanup function for termination signals
trap cleanup SIGINT SIGTERM

echo "Development servers are running."
echo "Access the web UI at http://localhost:3000"
echo "Press Ctrl+C to stop both servers."

# Keep the script running
wait
