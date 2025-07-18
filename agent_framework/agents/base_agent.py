"""
Base Agent Class for Finance Analyst AI Agent Framework
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseAgent:
    """
    Base Agent class that provides core functionality for all specialized agents
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: Dict[str, Callable] = None,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 64
    ):
        """
        Initialize the base agent
        
        Args:
            name: Name of the agent
            description: Description of the agent's role and capabilities
            tools: Dictionary of tools available to the agent
            model_name: Name of the Gemini model to use
            temperature: Temperature for text generation (higher = more creative)
            max_output_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self.name = name
        self.description = description
        self.tools = tools or {}
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.model = self._initialize_model()
        self.memory = []  # Simple memory for conversation history
        
    def _initialize_model(self):
        """Initialize the Gemini model with fallback options"""
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Try to initialize the model with fallbacks
        model_names = [self.model_name, "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                        "top_p": self.top_p,
                        "top_k": self.top_k
                    }
                )
                print(f"Successfully initialized Gemini model: {model_name}")
                return model
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {str(e)}")
                continue
        
        raise ValueError("Failed to initialize any Gemini model")
    
    def add_tool(self, name: str, tool_function: Callable):
        """Add a tool to the agent's toolkit"""
        self.tools[name] = tool_function
        
    def remove_tool(self, name: str):
        """Remove a tool from the agent's toolkit"""
        if name in self.tools:
            del self.tools[name]
            
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with provided arguments"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in agent's toolkit")
        
        try:
            start_time = time.time()
            result = self.tools[tool_name](**kwargs)
            end_time = time.time()
            
            # Log tool execution
            execution_info = {
                "tool": tool_name,
                "args": kwargs,
                "execution_time": end_time - start_time,
                "success": True
            }
            self.memory.append(execution_info)
            
            return result
        except Exception as e:
            # Log failed execution
            execution_info = {
                "tool": tool_name,
                "args": kwargs,
                "error": str(e),
                "success": False
            }
            self.memory.append(execution_info)
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Gemini model"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Process a user query - to be implemented by specialized agents
        
        Args:
            query: User query string
            
        Returns:
            Response string
        """
        raise NotImplementedError("Subclasses must implement process_query method")
