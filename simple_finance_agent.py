"""
Simple Finance AI Agent using yfinance and Gemini AI
"""
import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model - use the correct model name
try:
    # First try with the standard model name
    model = genai.GenerativeModel('gemini-pro')
    print("Using gemini-pro model")
except Exception as e:
    # If that fails, try with gemini-1.0-pro
    try:
        model = genai.GenerativeModel('gemini-1.0-pro')
        print("Using gemini-1.0-pro model")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini model: {str(e)}")
        print("Will use raw data without AI analysis")
        model = None

# Define stock analysis tools
class StockTools:
    @staticmethod
    def get_stock_price(symbol):
        """Get current stock price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if data.empty:
                return f"Could not find data for {symbol}"
            
            current_price = data['Close'].iloc[-1]
            return f"{symbol} current price: ${current_price:.2f}"
        except Exception as e:
            return f"Error fetching stock price for {symbol}: {str(e)}"
    
    @staticmethod
    def get_stock_history(symbol, period="1mo"):
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                return f"Could not find historical data for {symbol}"
            
            # Format the data for display
            result = f"Historical data for {symbol} (last {period}):\n"
            result += f"Start: ${data['Close'].iloc[0]:.2f}, End: ${data['Close'].iloc[-1]:.2f}\n"
            result += f"High: ${data['High'].max():.2f}, Low: ${data['Low'].min():.2f}\n"
            result += f"Volume: {data['Volume'].mean():.0f} (avg)"
            return result
        except Exception as e:
            return f"Error fetching historical data for {symbol}: {str(e)}"
    
    @staticmethod
    def calculate_rsi(symbol, period=14):
        """Calculate RSI for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo")
            if data.empty:
                return f"Could not find data for {symbol}"
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Interpret RSI
            if current_rsi > 70:
                interpretation = "overbought"
            elif current_rsi < 30:
                interpretation = "oversold"
            else:
                interpretation = "neutral"
                
            return f"RSI(14) for {symbol}: {current_rsi:.2f} - {interpretation}"
        except Exception as e:
            return f"Error calculating RSI for {symbol}: {str(e)}"
    
    @staticmethod
    def get_company_info(symbol):
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            name = info.get('longName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap/1e9:.2f}B"
            
            pe_ratio = info.get('trailingPE', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
                
            dividend_yield = info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield*100:.2f}%"
            
            result = f"Company Information for {symbol} ({name}):\n"
            result += f"Sector: {sector}\n"
            result += f"Industry: {industry}\n"
            result += f"Market Cap: {market_cap}\n"
            result += f"P/E Ratio: {pe_ratio}\n"
            result += f"Dividend Yield: {dividend_yield}"
            return result
        except Exception as e:
            return f"Error fetching company info for {symbol}: {str(e)}"

# Define the main agent class
class SimpleFinanceAgent:
    def __init__(self):
        self.tools = {
            "get_stock_price": StockTools.get_stock_price,
            "get_stock_history": StockTools.get_stock_history,
            "calculate_rsi": StockTools.calculate_rsi,
            "get_company_info": StockTools.get_company_info
        }
        
        self.system_prompt = """
        You are a Finance Analyst AI that helps users analyze stocks and provide financial insights.
        You have access to the following tools:
        
        1. get_stock_price(symbol): Get the current price of a stock
        2. get_stock_history(symbol, period): Get historical data for a stock (period can be 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        3. calculate_rsi(symbol, period): Calculate the RSI technical indicator (default period is 14)
        4. get_company_info(symbol): Get company information including sector, industry, market cap, P/E ratio, and dividend yield
        
        When a user asks a question about a stock, you should:
        1. Identify the stock symbol in the query
        2. Determine which tool to use based on the query
        3. Call the appropriate tool with the correct parameters
        4. Analyze the results and provide insights
        
        Always format your response in a clear, professional manner with sections for:
        - Summary: A brief overview of the findings
        - Analysis: Detailed explanation of the data
        - Recommendation: Suggested actions or insights based on the data
        
        DO NOT make up information. Only use the data provided by the tools.
        """
    
    def extract_stock_symbol(self, query):
        """Extract stock symbol from user query"""
        # Simple regex-like approach to find uppercase ticker symbols
        words = query.split()
        for word in words:
            # Look for uppercase words that might be stock symbols (1-5 characters)
            if word.isupper() and 1 <= len(word) <= 5 and word.isalpha():
                return word
        
        # Default to AAPL if no symbol found
        return "AAPL"
    
    def determine_tool(self, query):
        """Determine which tool to use based on the query"""
        query = query.lower()
        
        if any(term in query for term in ["price", "worth", "value", "cost", "current"]):
            return "get_stock_price"
        elif any(term in query for term in ["history", "historical", "trend", "past"]):
            return "get_stock_history"
        elif any(term in query for term in ["rsi", "relative strength", "overbought", "oversold"]):
            return "calculate_rsi"
        elif any(term in query for term in ["info", "information", "about", "company", "details"]):
            return "get_company_info"
        
        # Default to stock price if we can't determine
        return "get_stock_price"
    
    def process_query(self, query):
        """Process a user query and return a response"""
        try:
            # Extract stock symbol
            symbol = self.extract_stock_symbol(query)
            
            # Determine which tool to use
            tool_name = self.determine_tool(query)
            
            # Call the appropriate tool
            tool_function = self.tools[tool_name]
            if tool_name == "get_stock_history":
                # Extract period if specified, otherwise use default
                period = "1mo"  # default
                if "year" in query or "1y" in query:
                    period = "1y"
                elif "month" in query or "1mo" in query:
                    period = "1mo"
                elif "week" in query or "1w" in query:
                    period = "1w"
                elif "day" in query or "1d" in query:
                    period = "1d"
                
                result = tool_function(symbol, period)
            else:
                result = tool_function(symbol)
            
            # Use Gemini to analyze the result and provide insights if available
            if model is not None:
                try:
                    prompt = f"""
                    As a financial analyst, analyze the following data about {symbol} stock:
                    
                    {result}
                    
                    Provide a concise analysis with:
                    1. Summary: Brief overview
                    2. Analysis: Key insights from the data
                    3. Recommendation: Suggested actions or insights
                    
                    Keep your response professional and fact-based.
                    """
                    
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    print(f"Warning: Gemini API error: {str(e)}")
                    print("Falling back to raw data")
                    # Fall back to raw data
                    return f"Analysis for {symbol}:\n\n{result}"
            else:
                # If no model is available, just return the raw data
                return f"Analysis for {symbol}:\n\n{result}"
            
        except Exception as e:
            return f"Error processing your query: {str(e)}"

# Main function to run the agent
def main():
    print("=" * 80)
    print("Simple Finance AI Agent".center(80))
    print("=" * 80)
    
    agent = SimpleFinanceAgent()
    print("Agent initialized! Ask me about any stock.")
    print("\nExample queries:")
    print("1. What's the current price of AAPL?")
    print("2. Tell me about MSFT")
    print("3. Is TSLA overbought or oversold?")
    print("4. Show me the history of AMZN for the past month")
    
    while True:
        try:
            query = input("\nEnter your question (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Thank you for using the Simple Finance AI Agent. Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("Analyzing...")
            response = agent.process_query(query)
            print("\nResponse:")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
