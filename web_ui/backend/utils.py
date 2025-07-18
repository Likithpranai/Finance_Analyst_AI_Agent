"""
Utility functions for the Finance Analyst AI Agent Web UI backend.
"""

import json
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Union

from config import get_config

# Configure logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format
)
logger = logging.getLogger("finance_agent_web_ui")

if config.logging.log_to_file:
    import os
    os.makedirs(os.path.dirname(config.logging.log_file), exist_ok=True)
    file_handler = logging.FileHandler(config.logging.log_file)
    file_handler.setFormatter(logging.Formatter(config.logging.format))
    logger.addHandler(file_handler)

def generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())

def format_tool_execution(tool_name: str, tool_input: Any, tool_output: Optional[Any] = None, status: str = "started") -> Dict:
    """Format tool execution data for WebSocket messages."""
    return {
        "tool": tool_name,
        "input": tool_input,
        "output": tool_output,
        "status": status
    }

def format_websocket_message(message_type: str, message_id: str, content: Optional[str] = None, 
                            tool_execution: Optional[Dict] = None) -> str:
    """Format a message for WebSocket transmission."""
    message = {
        "type": message_type,
        "message_id": message_id
    }
    
    if content is not None:
        message["content"] = content
        
    if tool_execution is not None:
        message["tool_execution"] = tool_execution
        
    return json.dumps(message)

def calculate_typing_delay(text: str) -> float:
    """Calculate a realistic typing delay based on text length."""
    config = get_config()
    chars_per_second = config.typing_speed_chars_per_second
    
    # Add some randomness to make it feel more natural
    import random
    variation = random.uniform(0.8, 1.2)
    
    # Calculate delay with a minimum value
    delay = max(0.5, len(text) / (chars_per_second * variation))
    
    # Cap the delay to a reasonable maximum
    return min(delay, 5.0)

def split_response_for_streaming(response: str, chunk_size: int = 100) -> List[str]:
    """Split a response into chunks for streaming."""
    # Split by sentences if possible
    sentence_delimiters = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    
    chunks = []
    current_chunk = ""
    
    # Try to split by sentences first
    sentences = []
    last_end = 0
    
    for i in range(len(response)):
        for delimiter in sentence_delimiters:
            if response[i:i+len(delimiter)] == delimiter:
                sentences.append(response[last_end:i+1])
                last_end = i + len(delimiter) - 1
                break
    
    if last_end < len(response):
        sentences.append(response[last_end:])
    
    # Group sentences into chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we couldn't split by sentences (or got very long sentences),
    # fall back to splitting by chunk size
    if not chunks:
        for i in range(0, len(response), chunk_size):
            chunks.append(response[i:i+chunk_size])
    
    return chunks

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown text."""
    import re
    pattern = r'```(\w*)\n([\s\S]*?)```'
    matches = re.findall(pattern, text)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language.strip() or "text",
            "code": code.strip()
        })
    
    return code_blocks

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Basic sanitization - remove any script tags
    import re
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    
    # Remove other potentially dangerous HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    return text.strip()

def log_request(client_id: str, request_data: Dict) -> None:
    """Log incoming requests."""
    logger.info(f"Request from client {client_id}: {json.dumps(request_data)}")

def log_response(client_id: str, response_data: Union[Dict, str]) -> None:
    """Log outgoing responses."""
    if isinstance(response_data, dict):
        logger.info(f"Response to client {client_id}: {json.dumps(response_data)}")
    else:
        logger.info(f"Response to client {client_id}: {response_data[:100]}...")

def log_error(client_id: str, error: Exception) -> None:
    """Log errors."""
    logger.error(f"Error for client {client_id}: {str(error)}", exc_info=True)
