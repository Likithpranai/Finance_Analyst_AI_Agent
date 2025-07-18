#!/usr/bin/env python3
"""
Enhanced Finance Analyst AI Agent
A professional-grade financial analysis and decision-making platform with multi-agent architecture
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Import agent framework components
from agent_framework.orchestration.agent_orchestrator import AgentOrchestrator
from agent_framework.memory.agent_memory import AgentMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FinanceAnalystAI")

# Load environment variables
load_dotenv()

class EnhancedFinanceAnalystAI:
    """
    Enhanced Finance Analyst AI Agent with multi-agent architecture
    """
    
    def __init__(self):
        """Initialize the Enhanced Finance Analyst AI Agent"""
        logger.info("Initializing Enhanced Finance Analyst AI Agent")
        
        # Check for required API keys
        self._check_api_keys()
        
        # Initialize agent memory
        self.memory = AgentMemory()
        
        # Initialize agent orchestrator
        self.orchestrator = AgentOrchestrator()
        
        logger.info("Enhanced Finance Analyst AI Agent initialized successfully")
    
    def _check_api_keys(self):
        """Check for required API keys and log warnings if missing"""
        required_keys = {
            "GEMINI_API_KEY": "Google Gemini AI API",
            "POLYGON_API_KEY": "Polygon.io API",
            "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API (fallback)"
        }
        
        missing_keys = []
        for key, service in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(f"{service} ({key})")
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            logger.warning("Some functionality may be limited or unavailable")
        else:
            logger.info("All required API keys found")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using the agent orchestrator
        
        Args:
            query: User query string
            
        Returns:
            Response from the agent orchestrator
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Process query with agent orchestrator
            response = self.orchestrator.process_query(query)
            
            # Store conversation in memory
            symbols = self.orchestrator.extract_symbols(query)
            analysis_types = self.orchestrator.classify_query(query)
            self.memory.add_conversation_entry(query, response, symbols, analysis_types)
            
            logger.info("Query processed successfully")
            return response
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """
        Get recent conversations from memory
        
        Args:
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation entries
        """
        return self.memory.get_recent_conversations(limit)
    
    def get_watched_symbols(self) -> List[str]:
        """
        Get the list of watched symbols
        
        Returns:
            List of watched stock symbols
        """
        return self.memory.get_watched_symbols()
    
    def add_watched_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the watched symbols list
        
        Args:
            symbol: Stock symbol to watch
        """
        self.memory.add_watched_symbol(symbol)
        logger.info(f"Added {symbol} to watched symbols")
    
    def remove_watched_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the watched symbols list
        
        Args:
            symbol: Stock symbol to remove
        """
        self.memory.remove_watched_symbol(symbol)
        logger.info(f"Removed {symbol} from watched symbols")

def main():
    """Main entry point for the Enhanced Finance Analyst AI Agent"""
    parser = argparse.ArgumentParser(description="Enhanced Finance Analyst AI Agent")
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize the agent
    agent = EnhancedFinanceAnalystAI()
    
    if args.interactive:
        # Run in interactive mode
        print("Enhanced Finance Analyst AI Agent")
        print("Type 'exit' or 'quit' to exit")
        print("Type 'watch SYMBOL' to add a symbol to your watchlist")
        print("Type 'unwatch SYMBOL' to remove a symbol from your watchlist")
        print("Type 'watchlist' to see your watched symbols")
        print("Type 'history' to see your recent conversation history")
        print()
        
        while True:
            try:
                query = input("Query: ")
                
                if query.lower() in ["exit", "quit"]:
                    break
                elif query.lower() == "watchlist":
                    symbols = agent.get_watched_symbols()
                    if symbols:
                        print(f"Watched symbols: {', '.join(symbols)}")
                    else:
                        print("No watched symbols")
                elif query.lower() == "history":
                    conversations = agent.get_recent_conversations()
                    if conversations:
                        print("\nRecent conversations:")
                        for i, conv in enumerate(conversations):
                            print(f"{i+1}. Query: {conv['user_query']}")
                            print(f"   Symbols: {', '.join(conv['symbols']) if conv['symbols'] else 'None'}")
                            print(f"   Time: {conv['timestamp']}")
                            print()
                    else:
                        print("No conversation history")
                elif query.lower().startswith("watch "):
                    symbol = query.split(" ")[1].upper()
                    agent.add_watched_symbol(symbol)
                    print(f"Added {symbol} to watched symbols")
                elif query.lower().startswith("unwatch "):
                    symbol = query.split(" ")[1].upper()
                    agent.remove_watched_symbol(symbol)
                    print(f"Removed {symbol} from watched symbols")
                else:
                    response = agent.process_query(query)
                    print("\nResponse:")
                    print(response)
                    print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("Goodbye!")
    elif args.query:
        # Process a single query
        response = agent.process_query(args.query)
        print(response)
    else:
        # No query or interactive mode specified
        parser.print_help()

if __name__ == "__main__":
    main()
