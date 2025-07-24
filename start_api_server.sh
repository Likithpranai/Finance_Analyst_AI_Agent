#!/bin/bash
# Script to start the API server with the required API keys

export GEMINI_API_KEY="AIzaSyDcl9MVfMuzATS6PVQTuqbCDbPlFoOKiJ8"
export ALPHA_VANTAGE_API_KEY="OQKDAJNX96G1BPVM"
export NEWS_API_KEY="10e18c5c9a00420cb65e979a17f9ae2f"

echo "Starting API server with the provided API keys..."
python3 api_server.py
