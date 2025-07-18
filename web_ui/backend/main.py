#!/usr/bin/env python3
"""
FastAPI backend for Finance Analyst AI Agent Web UI.
This server connects the React frontend with the Finance Analyst AI Agent.
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to import Finance Analyst AI Agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Finance Analyst AI Agent
try:
    from finance_analyst_agent import FinanceAnalystReActAgent
except ImportError:
    print("Error: Could not import FinanceAnalystReActAgent. Make sure the Finance Analyst AI Agent is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finance_analyst_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Finance Analyst AI API",
    description="API for Finance Analyst AI Agent",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Finance Analyst AI Agent
try:
    finance_agent = FinanceAnalystReActAgent()
    logger.info("Finance Analyst AI Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Finance Analyst AI Agent: {str(e)}")
    finance_agent = None

# Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: Dict[str, Any]
    session_id: str
    timestamp: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# Routes
@app.get("/")
async def root():
    return {"message": "Finance Analyst AI API is running"}

@app.get("/health")
async def health_check():
    if finance_agent:
        return {"status": "healthy", "agent": "initialized"}
    return {"status": "unhealthy", "agent": "not initialized"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not finance_agent:
        raise HTTPException(status_code=503, detail="Finance Analyst AI Agent not initialized")
    
    try:
        # Process query with Finance Analyst AI Agent
        logger.info(f"Processing query: {request.query}")
        response = finance_agent.process_query(request.query)
        
        # Generate session ID if not provided
        session_id = request.session_id or datetime.now().strftime("%Y%m%d%H%M%S")
        
        return QueryResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse incoming message
                message = json.loads(data)
                query = message.get("query")
                
                if not query:
                    await manager.send_message(client_id, json.dumps({
                        "error": "Invalid message format. Expected 'query' field."
                    }))
                    continue
                
                # Process query with Finance Analyst AI Agent
                logger.info(f"Processing query from WebSocket: {query}")
                
                # Send thinking status
                await manager.send_message(client_id, json.dumps({
                    "status": "thinking",
                    "message": "Analyzing your query..."
                }))
                
                # Process the query
                response = finance_agent.process_query(query)
                
                # Stream the response in chunks to simulate typing
                response_text = json.dumps(response)
                
                # Send the complete response
                await manager.send_message(client_id, json.dumps({
                    "status": "complete",
                    "response": response
                }))
                
            except json.JSONDecodeError:
                await manager.send_message(client_id, json.dumps({
                    "error": "Invalid JSON message"
                }))
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_message(client_id, json.dumps({
                    "error": f"Error processing query: {str(e)}"
                }))
    except WebSocketDisconnect:
        manager.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
