"""
Stream handler for Finance Analyst AI Agent.
This module provides streaming capabilities for the agent's responses.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

class StreamingResponseHandler:
    """
    Handler for streaming responses from the Finance Analyst AI Agent.
    This allows for real-time updates as the agent processes queries.
    """
    
    def __init__(self, send_func: Callable[[str], Coroutine]):
        """
        Initialize the streaming response handler.
        
        Args:
            send_func: Async function to send messages to the client
        """
        self.send_func = send_func
    
    async def stream_thinking(self, tools_needed: list):
        """
        Stream thinking status with tools being used.
        
        Args:
            tools_needed: List of tools the agent is using
        """
        await self.send_func(json.dumps({
            "status": "thinking",
            "message": "Analyzing your query...",
            "tools": tools_needed
        }))
    
    async def stream_tool_execution(self, tool_name: str, status: str = "running"):
        """
        Stream tool execution status.
        
        Args:
            tool_name: Name of the tool being executed
            status: Status of the tool execution (running, complete, error)
        """
        await self.send_func(json.dumps({
            "status": "tool_execution",
            "tool": tool_name,
            "execution_status": status
        }))
    
    async def stream_partial_response(self, partial_response: Dict[str, Any]):
        """
        Stream partial response from the agent.
        
        Args:
            partial_response: Partial response from the agent
        """
        await self.send_func(json.dumps({
            "status": "partial",
            "response": partial_response
        }))
    
    async def stream_complete_response(self, response: Dict[str, Any]):
        """
        Stream complete response from the agent.
        
        Args:
            response: Complete response from the agent
        """
        await self.send_func(json.dumps({
            "status": "complete",
            "response": response
        }))
    
    async def stream_error(self, error_message: str):
        """
        Stream error message.
        
        Args:
            error_message: Error message
        """
        await self.send_func(json.dumps({
            "status": "error",
            "message": error_message
        }))
    
    async def simulate_typing(self, text: str, delay: float = 0.01, chunk_size: int = 5):
        """
        Simulate typing by streaming text in chunks.
        
        Args:
            text: Text to stream
            delay: Delay between chunks in seconds
            chunk_size: Number of characters per chunk
        """
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            await self.send_func(json.dumps({
                "status": "typing",
                "chunk": chunk
            }))
            await asyncio.sleep(delay)


class StreamingAgentWrapper:
    """
    Wrapper for Finance Analyst AI Agent that adds streaming capabilities.
    """
    
    def __init__(self, agent, stream_handler: StreamingResponseHandler):
        """
        Initialize the streaming agent wrapper.
        
        Args:
            agent: Finance Analyst AI Agent instance
            stream_handler: StreamingResponseHandler instance
        """
        self.agent = agent
        self.stream_handler = stream_handler
    
    async def process_query_with_streaming(self, query: str) -> Dict[str, Any]:
        """
        Process a query with streaming updates.
        
        Args:
            query: Query to process
            
        Returns:
            Response from the agent
        """
        try:
            # Extract symbol and determine tools
            symbol = self.agent.extract_stock_symbol(query)
            tools_needed = self.agent.determine_tools_needed(query)
            
            # Stream thinking status
            await self.stream_handler.stream_thinking(tools_needed)
            
            # Process tools one by one with streaming updates
            results = {}
            for tool_name in tools_needed:
                if tool_name in self.agent.tools:
                    # Stream tool execution start
                    await self.stream_handler.stream_tool_execution(tool_name, "running")
                    
                    try:
                        # Execute the tool
                        if tool_name in ["calculate_obv", "calculate_adline", "calculate_adx"]:
                            results[tool_name] = self.agent.tools[tool_name](symbol)
                        else:
                            results[tool_name] = self.agent.tools[tool_name](symbol)
                        
                        # Stream tool execution complete
                        await self.stream_handler.stream_tool_execution(tool_name, "complete")
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        # Stream tool execution error
                        await self.stream_handler.stream_tool_execution(tool_name, "error")
                        results[tool_name] = {"error": str(e)}
            
            # Format the response
            response = self.agent.format_response(query, symbol, tools_needed, results)
            
            # Stream complete response
            await self.stream_handler.stream_complete_response(response)
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            await self.stream_handler.stream_error(f"Error processing query: {str(e)}")
            return {"error": str(e)}
