#!/usr/bin/env python3
"""
Test script to verify API keys for Finance Analyst AI Agent
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = "I9H30JO7WPUD9ECS"
POLYGON_API_KEY = "HGaV6oKbB4mbuVEzhfhaKJPeEi8blmjn"
NEWSAPI_API_KEY = "b8ee3e0b-e75c-48cf-9804-51df4a7092af"
GEMINI_API_KEY = "AIzaSyDcl9MVfMuzATS6PVQTuqbCDbPlFoOKiJ8"

# Set environment variables
os.environ["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY
os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY
os.environ["NEWSAPI_API_KEY"] = NEWSAPI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

def test_alpha_vantage():
    """Test Alpha Vantage API key"""
    print("\n=== Testing Alpha Vantage API Key ===")
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=MSFT&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if "Symbol" in data:
        print("✅ Alpha Vantage API key is valid")
        print(f"Company: {data.get('Name', 'N/A')}")
        print(f"P/E Ratio: {data.get('PERatio', 'N/A')}")
        print(f"Market Cap: {data.get('MarketCapitalization', 'N/A')}")
    else:
        print("❌ Alpha Vantage API key is invalid or has reached its limit")
        print(f"Response: {data}")

def test_polygon():
    """Test Polygon API key"""
    print("\n=== Testing Polygon API Key ===")
    url = f"https://api.polygon.io/v2/aggs/ticker/MSFT/range/1/day/2023-01-01/2023-01-10?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") == "OK":
        print("✅ Polygon API key is valid")
        print(f"Results count: {len(data.get('results', []))}")
    else:
        print("❌ Polygon API key is invalid")
        print(f"Response: {data}")

def test_newsapi():
    """Test NewsAPI key"""
    print("\n=== Testing NewsAPI Key ===")
    url = f"https://newsapi.org/v2/everything?q=Microsoft&apiKey={NEWSAPI_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") == "ok":
        print("✅ NewsAPI key is valid")
        print(f"Articles count: {len(data.get('articles', []))}")
    else:
        print("❌ NewsAPI key is invalid")
        print(f"Response: {data}")

def test_gemini():
    """Test Gemini API key"""
    print("\n=== Testing Gemini API Key ===")
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.get_model('gemini-1.5-pro')
        print("✅ Gemini API key is valid")
    except Exception as e:
        print("❌ Gemini API key is invalid")
        print(f"Error: {str(e)}")

def main():
    """Main function to test all API keys"""
    print("Testing API keys for Finance Analyst AI Agent...")
    
    # Test all API keys
    test_alpha_vantage()
    test_polygon()
    test_newsapi()
    test_gemini()
    
    print("\nAPI key testing complete.")

if __name__ == "__main__":
    main()
