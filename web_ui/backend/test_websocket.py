#!/usr/bin/env python3
"""
WebSocket Connection Test Script for Finance Analyst AI Agent

This script tests the WebSocket connection to the backend server and helps diagnose
connection issues. It attempts to connect to the WebSocket server, send a test message,
and receive a response.
"""

import asyncio
import json
import logging
import sys
import websockets
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# WebSocket connection parameters
WS_URLS = [
    "ws://localhost:8000/ws/test-client",
    "ws://127.0.0.1:8000/ws/test-client",
    "ws://0.0.0.0:8000/ws/test-client"
]

# Test message to send
TEST_MESSAGE = {
    "query": "What is the current price of Apple stock?",
    "message_id": "test-message-1"
}

async def test_http_connection(url="http://localhost:8000/health"):
    """Test HTTP connection to the backend server"""
    try:
        logger.info(f"Testing HTTP connection to {url}")
        response = requests.get(url, timeout=5)
        logger.info(f"HTTP Response: {response.status_code} - {response.text}")
        return True
    except Exception as e:
        logger.error(f"HTTP connection failed: {str(e)}")
        return False

async def test_websocket_connection(ws_url):
    """Test WebSocket connection to the backend server"""
    try:
        logger.info(f"Attempting to connect to WebSocket at {ws_url}")
        
        async with websockets.connect(ws_url, ping_interval=None) as websocket:
            logger.info(f"Connected to WebSocket at {ws_url}")
            
            # Send a test message
            await websocket.send(json.dumps(TEST_MESSAGE))
            logger.info(f"Sent test message: {TEST_MESSAGE}")
            
            # Wait for a response with timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                logger.info(f"Received response: {response}")
                
                # Try to parse the response as JSON
                try:
                    parsed_response = json.loads(response)
                    logger.info(f"Parsed response: {json.dumps(parsed_response, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning("Response is not valid JSON")
                
                return True
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for response")
                return False
    except Exception as e:
        logger.error(f"WebSocket connection failed: {str(e)}")
        return False

async def test_ping_pong(ws_url):
    """Test simple ping/pong with the WebSocket server"""
    try:
        logger.info(f"Testing ping/pong with {ws_url}")
        
        async with websockets.connect(ws_url, ping_interval=None) as websocket:
            # Send a ping message
            ping_message = json.dumps({"type": "ping"})
            await websocket.send(ping_message)
            logger.info(f"Sent ping message: {ping_message}")
            
            # Wait for pong response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"Received response to ping: {response}")
                return True
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for pong response")
                return False
    except Exception as e:
        logger.error(f"Ping/pong test failed: {str(e)}")
        return False

async def run_tests():
    """Run all connection tests"""
    # First test HTTP connection
    http_success = await test_http_connection()
    
    if not http_success:
        logger.error("HTTP connection failed. Backend server may not be running.")
        return
    
    # Test each WebSocket URL
    for ws_url in WS_URLS:
        logger.info(f"\n--- Testing WebSocket URL: {ws_url} ---")
        
        # Test ping/pong first
        ping_success = await test_ping_pong(ws_url)
        if ping_success:
            logger.info(f"✅ Ping/pong test succeeded for {ws_url}")
        else:
            logger.error(f"❌ Ping/pong test failed for {ws_url}")
        
        # Test full WebSocket connection
        ws_success = await test_websocket_connection(ws_url)
        if ws_success:
            logger.info(f"✅ WebSocket test succeeded for {ws_url}")
        else:
            logger.error(f"❌ WebSocket test failed for {ws_url}")
        
        logger.info(f"--- Finished testing {ws_url} ---\n")

if __name__ == "__main__":
    logger.info("Starting WebSocket connection tests")
    asyncio.run(run_tests())
    logger.info("WebSocket connection tests completed")
