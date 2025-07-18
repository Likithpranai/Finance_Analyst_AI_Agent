"""
Demo script for showcasing the professional-grade real-time data capabilities
of the Finance Analyst AI Agent with Polygon.io integration, WebSocket streaming,
and Redis caching.
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from finance_analyst_agent import FinanceAnalystReActAgent
from tools.real_time_data_integration import RealTimeDataTools
from tools.websocket_manager import websocket_manager
from tools.cache_manager import CacheManager

# Load environment variables
load_dotenv()

# Check for required API keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    print("Warning: POLYGON_API_KEY not found in environment variables.")
    print("Please add your Polygon.io API key to the .env file.")
    print("You can get a free API key at https://polygon.io/")

# Initialize the agent
agent = FinanceAnalystReActAgent()

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def real_time_quote_demo():
    """Demo real-time quotes with sub-second latency"""
    print_section("Real-Time Quotes Demo (Sub-Second Latency)")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print("Fetching real-time quotes for multiple symbols...")
    start_time = time.time()
    
    for symbol in symbols:
        quote = RealTimeDataTools.get_real_time_quote(symbol)
        print(f"\n{symbol} Quote (Source: {quote.get('source', 'Unknown')}):")
        
        if "data" in quote:
            data = quote["data"]
            if isinstance(data, dict):
                # Format the output nicely
                print(f"  Price: ${data.get('last', data.get('price', 'N/A'))}")
                print(f"  Change: {data.get('change', 'N/A')} ({data.get('percent_change', 'N/A')}%)")
                print(f"  Volume: {data.get('volume', 'N/A')}")
                print(f"  Time: {data.get('timestamp', 'N/A')}")
            else:
                print(data)
        else:
            print(f"  Error: {quote.get('error', 'Unknown error')}")
    
    end_time = time.time()
    print(f"\nTotal time for {len(symbols)} quotes: {end_time - start_time:.4f} seconds")
    print(f"Average time per quote: {(end_time - start_time) / len(symbols):.4f} seconds")

def intraday_data_demo():
    """Demo intraday data with caching"""
    print_section("Intraday Data Demo (With Caching)")
    
    symbol = "AAPL"
    intervals = ["1min", "5min", "15min"]
    
    for interval in intervals:
        print(f"\nFetching {interval} intraday data for {symbol}...")
        
        # First request - should hit the API
        start_time = time.time()
        data = RealTimeDataTools.get_intraday_data(symbol, interval, 10)
        first_request_time = time.time() - start_time
        
        print(f"First request (API call) - Time: {first_request_time:.4f} seconds")
        print(f"Data source: {data.attrs.get('source', 'Unknown')}")
        print(data.head())
        
        # Second request - should hit the cache
        start_time = time.time()
        cached_data = RealTimeDataTools.get_intraday_data(symbol, interval, 10)
        second_request_time = time.time() - start_time
        
        print(f"\nSecond request (Cached) - Time: {second_request_time:.4f} seconds")
        print(f"Cache speedup: {first_request_time / second_request_time:.1f}x faster")
        
        # Verify data is the same
        print(f"Data identical: {cached_data.equals(data)}")

def websocket_streaming_demo():
    """Demo WebSocket streaming for real-time updates"""
    print_section("WebSocket Streaming Demo (Real-Time Updates)")
    
    # Define a callback function to handle incoming messages
    def handle_stock_message(message):
        if isinstance(message, dict):
            if message.get('ev') == 'T':  # Trade event
                symbol = message.get('sym', 'Unknown')
                price = message.get('p', 0)
                size = message.get('s', 0)
                timestamp = message.get('t', 0)
                
                print(f"Trade: {symbol} - ${price:.2f} - Size: {size} - Time: {timestamp}")
    
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    print(f"Starting real-time WebSocket stream for: {', '.join(symbols)}")
    
    try:
        # Start the WebSocket stream
        stream_result = RealTimeDataTools.start_real_time_stream(symbols, handle_stock_message, "stocks")
        
        if "error" in stream_result:
            print(f"Error starting stream: {stream_result['error']}")
            return
        
        stream_id = stream_result.get("stream_id")
        print(f"Stream started with ID: {stream_id}")
        
        # Let it run for a few seconds to collect some data
        print("Listening for real-time trades (press Ctrl+C to stop)...")
        try:
            time.sleep(30)  # Listen for 30 seconds
        except KeyboardInterrupt:
            print("\nStream interrupted by user")
        
        # Stop the stream
        stop_result = RealTimeDataTools.stop_real_time_stream(stream_id)
        print(f"Stream stopped: {stop_result}")
        
    except Exception as e:
        print(f"Error in WebSocket demo: {str(e)}")

def cache_management_demo():
    """Demo cache management capabilities"""
    print_section("Cache Management Demo")
    
    # Get cache statistics
    stats = RealTimeDataTools.get_cache_stats()
    print("Cache Statistics:")
    print_json(stats)
    
    # Clear specific cache entries
    clear_result = RealTimeDataTools.clear_cache("finance_agent:get_real_time_quote:*")
    print("\nCleared real-time quote cache entries:")
    print_json(clear_result)
    
    # Get updated statistics
    stats = RealTimeDataTools.get_cache_stats()
    print("\nUpdated Cache Statistics:")
    print_json(stats)

def agent_query_demo():
    """Demo the agent processing queries with the new tools"""
    print_section("Agent Query Demo")
    
    queries = [
        "What's the current price of Apple stock?",
        "Show me the intraday chart for Tesla today",
        "Compare the performance of MSFT and GOOGL over the past week",
        "What's the latest news about Amazon?",
        "Give me a technical analysis of NVDA with RSI and MACD"
    ]
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 80)
        
        try:
            # Process the query
            start_time = time.time()
            response = agent.process_query(query)
            end_time = time.time()
            
            print(f"Response (in {end_time - start_time:.2f} seconds):")
            print(response)
        except Exception as e:
            print(f"Error processing query: {str(e)}")

def main():
    """Run all demos"""
    print_section("Finance Analyst AI Agent - Professional-Grade Real-Time Data Demo")
    
    # Check if Redis is available
    redis_available = False
    try:
        redis_available = CacheManager.is_redis_available()
    except Exception as e:
        print(f"Error checking Redis availability: {str(e)}")
    
    print(f"Redis Cache Available: {redis_available}")
    print("Note: Redis is not required but enables caching for better performance")
    
    # Check if Polygon API key is available
    polygon_key_available = bool(POLYGON_API_KEY)
    print(f"Polygon.io API Key Available: {polygon_key_available}")
    
    if not polygon_key_available:
        print("Warning: Polygon.io API key not found. Will fall back to Alpha Vantage.")
    
    # Run demos
    try:
        real_time_quote_demo()
    except Exception as e:
        print(f"Error in real-time quote demo: {str(e)}")
    
    try:
        intraday_data_demo()
    except Exception as e:
        print(f"Error in intraday data demo: {str(e)}")
    
    # Only run WebSocket demo if Polygon API key is available
    if polygon_key_available:
        try:
            websocket_streaming_demo()
        except Exception as e:
            print(f"Error in WebSocket streaming demo: {str(e)}")
    else:
        print_section("WebSocket Streaming Demo (Skipped - No API Key)")
        print("Skipping WebSocket demo because Polygon.io API key is not available.")
    
    # Only run cache demo if Redis is available
    if redis_available:
        try:
            cache_management_demo()
        except Exception as e:
            print(f"Error in cache management demo: {str(e)}")
    else:
        print_section("Cache Management Demo (Skipped - No Redis)")
        print("Skipping cache management demo because Redis is not available.")
    
    try:
        agent_query_demo()
    except Exception as e:
        print(f"Error in agent query demo: {str(e)}")
        
    print("\nDemo completed. The Finance Analyst AI Agent is now enhanced with professional-grade real-time data capabilities.")
    print("Even without Redis, the system will function with direct API calls.")
    print("For optimal performance, consider setting up a Redis server locally or in the cloud.")
    print("\nYou can now use the agent with the new capabilities!")


if __name__ == "__main__":
    main()
