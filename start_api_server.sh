#!/bin/bash
# Script to start the API server with the required API keys

export GEMINI_API_KEY="AIzaSyDcl9MVfMuzATS6PVQTuqbCDbPlFoOKiJ8"
export ALPHA_VANTAGE_API_KEY="OQKDAJNX96G1BPVM"
export NEWS_API_KEY="10e18c5c9a00420cb65e979a17f9ae2f"
export FINNHUB_API_KEY="d21it51r01qpst758020d21it51r01qpst75802g"
export NEWSAPI_API_KEY="b8ee3e0b-e75c-48cf-9804-51df4a7092af"
export TIINGO_API_KEY="5708819432a980457086066dc1ab70e1d0016ab1"

echo "Starting API server with the provided API keys..."
python3 api_server.py
