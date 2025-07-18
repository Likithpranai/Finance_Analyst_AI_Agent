"""
Agent Orchestrator for Finance Analyst AI Agent Framework
Coordinates between specialized agents to provide comprehensive financial analysis
"""
import os
import re
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
import google.generativeai as genai
from dotenv import load_dotenv

# Import specialized agents
from agent_framework.agents.technical_analysis_agent import TechnicalAnalysisAgent
from agent_framework.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from agent_framework.agents.risk_analysis_agent import RiskAnalysisAgent
from agent_framework.agents.trading_agent import TradingAgent
from agent_framework.agents.sentiment_agent import SentimentAgent

# Load environment variables
load_dotenv()

class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents to provide comprehensive financial analysis
    """
    
    def __init__(self):
        """Initialize the Agent Orchestrator with specialized agents"""
        # Initialize specialized agents
        self.technical_agent = TechnicalAnalysisAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.trading_agent = TradingAgent()
        self.sentiment_agent = SentimentAgent()
        
        # Initialize Gemini model for orchestration
        self.model = self._initialize_model()
        
        # Define query classifiers
        self.query_classifiers = {
            "technical": [
                "technical analysis", "chart", "pattern", "trend", "support", "resistance",
                "moving average", "rsi", "macd", "bollinger", "indicator", "oscillator",
                "volume", "price action", "candlestick", "momentum", "overbought", "oversold"
            ],
            "fundamental": [
                "fundamental analysis", "financial statement", "balance sheet", "income statement",
                "cash flow", "earnings", "revenue", "profit", "margin", "growth", "valuation",
                "pe ratio", "eps", "dividend", "debt", "assets", "liabilities", "equity",
                "industry", "sector", "competitor", "market share", "business model"
            ],
            "risk": [
                "risk", "volatility", "var", "value at risk", "drawdown", "sharpe", "sortino",
                "beta", "correlation", "diversification", "hedge", "exposure", "stress test",
                "scenario", "portfolio risk", "downside", "tail risk", "risk-adjusted"
            ],
            "trading": [
                "trade", "buy", "sell", "entry", "exit", "position", "stop loss", "take profit",
                "order", "execution", "signal", "strategy", "backtest", "performance", "return",
                "profit target", "breakout", "reversal", "swing", "day trade", "scalp"
            ],
            "sentiment": [
                "sentiment", "news", "media", "social", "twitter", "reddit", "article", "headline",
                "press", "release", "announcement", "opinion", "mood", "feeling", "perception",
                "bullish", "bearish", "positive", "negative", "neutral", "coverage", "mention",
                "finbert", "nlp", "text analysis", "news sentiment", "market sentiment"
            ]
        }
    
    def _initialize_model(self):
        """Initialize the Gemini model for orchestration"""
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Try to initialize the model with fallbacks
        model_names = ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                        "top_p": 0.95,
                        "top_k": 64
                    }
                )
                print(f"Successfully initialized Gemini model for orchestration: {model_name}")
                return model
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {str(e)}")
                continue
        
        raise ValueError("Failed to initialize any Gemini model for orchestration")
    
    def classify_query(self, query: str) -> List[str]:
        """
        Classify the query to determine which specialized agents to use
        
        Args:
            query: User query string
            
        Returns:
            List of agent types to use (technical, fundamental, risk, trading)
        """
        query_lower = query.lower()
        agent_types = []
        
        # Check for each agent type
        for agent_type, keywords in self.query_classifiers.items():
            if any(keyword in query_lower for keyword in keywords):
                agent_types.append(agent_type)
        
        # If no specific agent types detected, use Gemini to classify
        if not agent_types:
            try:
                classification_prompt = f"""
                Classify the following financial query into one or more of these categories:
                - technical (technical analysis, charts, patterns, indicators)
                - fundamental (company financials, valuation, business analysis)
                - risk (risk assessment, volatility, portfolio risk)
                - trading (trading signals, entry/exit points, orders)
                - sentiment (sentiment analysis, news, media, social)
                
                Query: {query}
                
                Return only the category names separated by commas, nothing else.
                """
                
                response = self.model.generate_content(classification_prompt)
                categories = response.text.strip().lower().split(',')
                agent_types = [category.strip() for category in categories if category.strip() in self.query_classifiers]
            except Exception as e:
                print(f"Error classifying query with Gemini: {str(e)}")
                # Default to technical analysis if classification fails
                agent_types = ["technical"]
        
        # Ensure at least one agent type is selected
        if not agent_types:
            agent_types = ["technical"]
        
        return agent_types
    
    def extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from user query"""
        # Look for stock symbols in the query (typically 1-5 uppercase letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        if matches:
            return matches
        
        return []
    
    def process_query(self, query: str) -> str:
        """
        Process a user query by orchestrating specialized agents
        
        Args:
            query: User query string
            
        Returns:
            Comprehensive analysis response
        """
        # Extract symbols from query
        symbols = self.extract_symbols(query)
        if not symbols:
            # For sentiment analysis of general market, we don't always need a symbol
            if any(keyword in query.lower() for keyword in self.query_classifiers["sentiment"]):
                agent_types = ["sentiment"]
            else:
                return "I couldn't identify any stock symbols in your query. Please specify valid stock symbols (e.g., AAPL, MSFT, GOOGL)."
        else:
            # Classify query to determine which agents to use
            agent_types = self.classify_query(query)
        
        print(f"Query classified as: {', '.join(agent_types)}")
        
        # Process query with appropriate agents
        results = {}
        
        try:
            # Process with technical analysis agent if needed
            if "technical" in agent_types:
                print(f"Processing with Technical Analysis Agent")
                results["technical"] = self.technical_agent.process_query(query)
            
            # Process with fundamental analysis agent if needed
            if "fundamental" in agent_types:
                print(f"Processing with Fundamental Analysis Agent")
                results["fundamental"] = self.fundamental_agent.process_query(query)
            
            # Process with risk analysis agent if needed
            if "risk" in agent_types:
                print(f"Processing with Risk Analysis Agent")
                results["risk"] = self.risk_agent.process_query(query)
            
            # Process with trading agent if needed
            if "trading" in agent_types:
                print(f"Processing with Trading Agent")
                results["trading"] = self.trading_agent.process_query(query)
            
            # Process with sentiment analysis agent if needed
            if "sentiment" in agent_types:
                print(f"Processing with Sentiment Analysis Agent")
                results["sentiment"] = self.sentiment_agent.process_query(query)
            
            # If only one agent type was used, return its result directly
            if len(agent_types) == 1:
                return results[agent_types[0]]
            
            # Otherwise, synthesize results from multiple agents
            print(f"Synthesizing results from multiple agents: {', '.join(agent_types)}")
            
            # Create prompt for Gemini to synthesize results
            synthesis_prompt = f"""
            Synthesize the following analyses into a comprehensive response for the query: "{query}"
            
            """
            
            for agent_type in agent_types:
                if agent_type in results:
                    synthesis_prompt += f"\n{agent_type.capitalize()} Analysis:\n{results[agent_type]}\n"
            
            synthesis_prompt += """
            Create a well-structured, comprehensive analysis that integrates insights from all the specialized analyses.
            Format your response in markdown with clear sections.
            Avoid redundancy while preserving key insights from each analysis.
            Provide a concise executive summary at the beginning.
            """
            
            # Generate synthesized response
            response = self.model.generate_content(synthesis_prompt)
            return response.text
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            
            # Return any partial results if available
            if results:
                partial_results = "I encountered an error while processing your query, but here are the partial results:\n\n"
                for agent_type, result in results.items():
                    partial_results += f"\n## {agent_type.capitalize()} Analysis\n{result}\n"
                return partial_results
            
            return error_message
