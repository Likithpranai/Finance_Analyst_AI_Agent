"""
Finance Analyst AI Agent - ReAct Agent Implementation
"""
import json
import re
from typing import List, Dict, Any, Annotated, TypedDict, Sequence
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from tools.stock_data import (
    GetStockPriceTool, 
    GetStockHistoryTool,
    PlotStockPriceTool,
    GetCompanyInfoTool
)
from tools.technical_indicators import (
    CalculateRSITool,
    CalculateMovingAveragesTool,
    CalculateMACDTool,
    CalculateBollingerBandsTool
)
from tools.market_news import (
    GetStockNewsTool,
    AnalyzeMarketSentimentTool
)
# Import new data sources
from tools.alpha_vantage import (
    AlphaVantageStockTool,
    AlphaVantageForexTool
)
from tools.financial_modeling_prep import (
    CompanyFinancialsTool,
    CompanyValuationTool,
    IndustryComparisonTool
)

# Import fundamental analysis tools
from tools.fundamental_analysis import (
    FinancialRatiosTool,
    CompetitiveAnalysisTool,
    DCFValuationTool
)

# Import SEC filings analysis tools
from tools.sec_filings import (
    SECFilingsAnalyzerTool,
    SECFinancialStatementsTool
)

# Import economic indicators tools
from tools.economic_indicators import (
    FREDEconomicDataTool,
    ForexDataTool,
    GlobalMarketIndicesTool
)

from config import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE, DATA_SOURCES, PRIMARY_DATA_SOURCE


# Set up the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class AgentState(TypedDict):
    """Type definition for the agent state."""
    messages: Annotated[Sequence[Any], "The message history"]
    tools_output: Annotated[Dict, "The output from tools"]


def create_finance_react_agent():
    """Create and return a finance analyst ReAct agent."""
    
    # Initialize all tools
    tools = [
        # Yahoo Finance Tools
        GetStockPriceTool(),
        GetStockHistoryTool(),
        PlotStockPriceTool(),
        GetCompanyInfoTool(),
        
        # Technical Indicators
        CalculateRSITool(),
        CalculateMovingAveragesTool(),
        CalculateMACDTool(),
        CalculateBollingerBandsTool(),
        
        # Market News
        GetStockNewsTool(),
        AnalyzeMarketSentimentTool(),
        
        # Alpha Vantage Tools
        AlphaVantageStockTool(),
        AlphaVantageForexTool(),
        
        # Financial Modeling Prep Tools
        CompanyFinancialsTool(),
        CompanyValuationTool(),
        IndustryComparisonTool(),
        
        # Fundamental Analysis Tools
        FinancialRatiosTool(),
        CompetitiveAnalysisTool(),
        DCFValuationTool(),
        
        # SEC Filings Analysis Tools
        SECFilingsAnalyzerTool(),
        SECFinancialStatementsTool(),
        
        # Economic Indicators Tools
        FREDEconomicDataTool(),
        ForexDataTool(),
        GlobalMarketIndicesTool()
    ]
    
    # Create the LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
        convert_system_message_to_human=True
    )
    
    # ReAct agent system prompt
    system_prompt = """You are a professional financial analyst AI agent that provides comprehensive market insights, stock analysis, and investment recommendations.
    
    You have access to tools for:
    1. Getting current and historical stock prices
    2. Calculating technical indicators (RSI, Moving Averages, MACD, Bollinger Bands)
    3. Tracking market news and sentiment
    4. Visualizing stock data
    5. Fundamental Analysis (financial ratios, competitive analysis, DCF valuation)
    6. SEC Filings Analysis (recent filings, financial statements extraction)
    7. Economic Indicators (GDP, inflation, interest rates, global indices, forex)

    IMPORTANT: For financial queries, you MUST use the appropriate tools to gather real data before providing an answer.
    DO NOT provide a final answer immediately without using tools first.
    
    Follow the ReAct pattern to solve user queries:
    1. REASON: Carefully think about what information you need and which tools to use
    2. ACT: Use the appropriate tools to gather data and perform analysis
    3. OBSERVE: Process the results from tools
    4. Repeat until you have enough information to provide a comprehensive answer
    
    Available tools:
    - StockPriceTool: Get current and historical stock prices
    - TechnicalIndicatorsTool: Calculate RSI, MACD, Moving Averages, Bollinger Bands
    - MarketNewsTool: Get latest news about stocks and markets
    - FundamentalAnalysisTool: Get company financial data
    - SECFilingsTool: Analyze recent SEC filings
    - EconomicIndicatorsTool: Get economic data like GDP, inflation
    - StockVisualizationTool: Create charts of stock data
    - StockComparisonTool: Compare multiple stocks
    - StockPredictionTool: Generate forecasts
    
    For example:
    - For "Calculate the RSI for TSLA" → Use TechnicalIndicatorsTool
    - For "Compare moving averages for MSFT" → Use TechnicalIndicatorsTool
    - For "Latest news about AMZN" → Use MarketNewsTool
    
    Format your final response as follows:
    1. Key Insights: Brief bullet points summarizing main findings
    2. Analysis: Detailed explanation with data to support your conclusions
    3. Recommendation (if applicable): Suggested actions based on your analysis
    
    IMPORTANT: YOUR RESPONSE MUST BE IN THE FOLLOWING JSON FORMAT:
    ```json
    {
        "thoughts": "Your reasoning process (not shown to user)",
        "tool": "The name of the tool to use (or 'final_answer' if providing final response)",
        "tool_input": "The input to the tool as a string (omit if tool is 'final_answer')",
        "final_answer": "Your comprehensive answer to the user's question (only include if tool is 'final_answer')"
    }
    ```
    
    Do not include any text outside of the JSON structure. Make sure your JSON is properly formatted and can be parsed by Python's json.loads() function.
    """

    # Create prompt template for agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    # Create JSON output parser
    class AgentAction(TypedDict):
        thoughts: str
        tool: str
        tool_input: str
        final_answer: str
    
    output_parser = JsonOutputParser(pydantic_object=AgentAction)

    # Define the agent function
    def agent_node(state: AgentState) -> dict:
        """
        Agent node for the ReAct workflow.
        Takes the current state and decides on the next action.
        """
        # Debug: Print the state structure
        print("\nDEBUG - Agent Node State Structure:")
        print(f"State keys: {list(state.keys())}")
        
        # Format the tool-use history for the prompt
        tools_output = state.get("tools_output", {})
        agent_scratchpad = []
        
        # Debug: Print tools output
        print(f"Tools output: {tools_output}")
        
        # Add tool outputs to scratchpad if available
        if tools_output:
            result = tools_output.get("result", "")
            tool_name = tools_output.get("tool", "")
            agent_scratchpad.append(HumanMessage(content=f"Tool {tool_name} returned: {result}"))
        
        # Invoke the LLM with explicit response structure instructions
        try:
            # Format the prompt with messages and scratchpad
            formatted_prompt = prompt.format_messages(
                messages=state["messages"],
                agent_scratchpad=agent_scratchpad
            )
            
            # Print the formatted prompt for debugging
            print("\nDEBUG - Formatted Prompt:")
            for msg in formatted_prompt[-2:]:  # Print just the last two messages to avoid clutter
                print(f"[{msg.type}]: {msg.content[:100]}...")
            
            # Invoke the LLM
            response = llm.invoke(formatted_prompt)
            
            # Get the content from the response
            content = response.content
            
            # Print the raw response for debugging
            print("\nDEBUG - Raw LLM Response:")
            print(f"{content[:500]}..." if len(content) > 500 else content)
            
            # First try to extract JSON from code blocks
            json_patterns = [
                r"```json\n(\{.*?\})\n```",  # JSON in code block with json tag
                r"```\n(\{.*?\})\n```",     # JSON in code block without tag
                r"```(\{.*?\})```",         # JSON in code block without newlines
                r"\n(\{.*?\})\n"           # JSON with newlines around it
            ]
            
            parsed_json = None
            for pattern in json_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        json_str = match.group(1).strip()
                        parsed_json = json.loads(json_str)
                        break
                    except:
                        continue
            
            # If no JSON found in code blocks, try to find any JSON object
            if not parsed_json:
                # Look for any JSON-like structure
                json_match = re.search(r'\{[^\{\}]*"tool"[^\{\}]*\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0).strip()
                        parsed_json = json.loads(json_str)
                    except:
                        pass
            
            # If we found valid JSON, use it
            if parsed_json and isinstance(parsed_json, dict) and "tool" in parsed_json:
                # Ensure all required keys are present
                if parsed_json["tool"] != "final_answer" and "tool_input" not in parsed_json:
                    parsed_json["tool_input"] = "{}"
                
                # Make sure thoughts are present
                if "thoughts" not in parsed_json:
                    parsed_json["thoughts"] = "Processing the user query"
                    
                return parsed_json
            
            # If we couldn't parse JSON, use heuristics
            return extract_action_from_text(content)
            
        except Exception as e:
            # If there's any error in the process, return a safe fallback
            return {
                "thoughts": f"Error processing response: {str(e)}",
                "tool": "final_answer",
                "final_answer": f"I encountered an issue while analyzing your request. Please try rephrasing your question or try a different query."
            }
    
    def extract_action_from_text(content):
        """
        Extract action information from non-JSON text using heuristics.
        """
        # Check if the content mentions a final answer
        
        # First, try to identify the query type to determine the appropriate tool
        query = query.lower() if isinstance(query, str) else ""
        content_lower = content.lower()
        
        # Map common query patterns to tools
        stock_symbol_match = re.search(r'\b([A-Z]{1,5})\b', query)
        stock_symbol = stock_symbol_match.group(1) if stock_symbol_match else "AAPL"  # Default to AAPL if no symbol found
        
        # Force tool selection based on query keywords
        if any(term in query for term in ["price", "current price", "stock price", "value"]):
            print("DEBUG - Query suggests StockPriceTool")
            return {
                "thoughts": f"Using StockPriceTool to get price data for {stock_symbol}",
                "tool": "StockPriceTool",
                "tool_input": f"{{\"symbol\": \"{stock_symbol}\"}}"
            }
        elif any(term in query for term in ["rsi", "relative strength", "technical", "indicator", "macd", "moving average", "bollinger"]):
            print("DEBUG - Query suggests TechnicalIndicatorsTool")
            indicator = "RSI"
            if "macd" in query:
                indicator = "MACD"
            elif "moving average" in query or "ma" in query:
                indicator = "MA"
            elif "bollinger" in query:
                indicator = "BBANDS"
                
            return {
                "thoughts": f"Using TechnicalIndicatorsTool to calculate {indicator} for {stock_symbol}",
                "tool": "TechnicalIndicatorsTool",
                "tool_input": f"{{\"symbol\": \"{stock_symbol}\", \"indicator\": \"{indicator}\"}}"
            }
        elif any(term in query for term in ["news", "latest news", "recent news", "headlines"]):
            print("DEBUG - Query suggests MarketNewsTool")
            return {
                "thoughts": f"Using MarketNewsTool to get news for {stock_symbol}",
                "tool": "MarketNewsTool",
                "tool_input": f"{{\"symbol\": \"{stock_symbol}\"}}"
            }
        elif any(term in query for term in ["compare", "comparison", "versus", "vs"]):
            print("DEBUG - Query suggests StockComparisonTool")
            # Try to find a second stock symbol
            symbols = re.findall(r'\b([A-Z]{1,5})\b', query)
            second_symbol = symbols[1] if len(symbols) > 1 else "MSFT"  # Default to MSFT as second symbol
            return {
                "thoughts": f"Using StockComparisonTool to compare {stock_symbol} and {second_symbol}",
                "tool": "StockComparisonTool",
                "tool_input": f"{{\"symbols\": [\"{stock_symbol}\", \"{second_symbol}\"]}}"
            }
        elif any(term in query for term in ["plot", "chart", "graph", "visualize", "visualization"]):
            print("DEBUG - Query suggests StockVisualizationTool")
            return {
                "thoughts": f"Using StockVisualizationTool to create chart for {stock_symbol}",
                "tool": "StockVisualizationTool",
                "tool_input": f"{{\"symbol\": \"{stock_symbol}\"}}"
            }
        
        # If we couldn't determine from the query, try to extract from the LLM response
        # Patterns to extract tool names from text
        tool_patterns = [
            (r"use\s+(?:the\s+)?([\w]+(?:Tool|Analysis|Visualization))", 1),  # "use StockPriceTool"
            (r"using\s+(?:the\s+)?([\w]+(?:Tool|Analysis|Visualization))", 1),  # "using StockPriceTool"
            (r"with\s+(?:the\s+)?([\w]+(?:Tool|Analysis|Visualization))", 1),  # "with StockPriceTool"
            (r"([\w]+(?:Tool|Analysis|Visualization))\s+to", 1),  # "StockPriceTool to get"
            (r"need\s+to\s+(?:use\s+)?(?:the\s+)?([\w]+(?:Tool|Analysis|Visualization))", 1),  # "need to use StockPriceTool"
            (r"([\w]+(?:Tool|Analysis|Visualization))\s+(?:can|will|should)", 1),  # "StockPriceTool can provide"
            (r"tool:\s*['\"]?([\w]+(?:Tool|Analysis|Visualization))['\"]?", 1),  # "tool: StockPriceTool"
        ]
        
        for pattern, group in tool_patterns:
            tool_match = re.search(pattern, content, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1)
                # Check if the extracted name ends with 'Tool'
                if not tool_name.endswith('Tool'):
                    tool_name = tool_name + 'Tool'
                    
                # Try to extract tool input if available
                input_match = re.search(r"input[\s:]*['\"]?([^'\"\n]+)['\"]?", content, re.IGNORECASE)
                tool_input = input_match.group(1) if input_match else f"{{\"symbol\": \"{stock_symbol}\"}}"
                
                print(f"DEBUG - Extracted tool: {tool_name}, input: {tool_input}")
                return {
                    "thoughts": f"Using {tool_name} to process the query",
                    "tool": tool_name,
                    "tool_input": tool_input
                }
        
        # Default to StockPriceTool if we can't determine a specific tool
        # This ensures we always use a tool rather than giving a direct final answer
        print("DEBUG - Defaulting to StockPriceTool")
        return {
            "thoughts": f"Using StockPriceTool as default to get information about {stock_symbol}",
            "tool": "StockPriceTool",
            "tool_input": f"{{\"symbol\": \"{stock_symbol}\"}}"
        }
        
    # Create the LangGraph workflow

    # Create a very simple workflow with basic structure
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Define edges
    workflow.add_edge(START, "agent")
    
    # Define a router function with debugging
    def debug_router(state):
        print("\nDEBUG - Router Function:")
        print(f"State keys: {list(state.keys())}")
        
        # The state structure is different than expected
        # The tool is directly in the state, not nested under 'agent'
        if "tool" in state:
            tool = state.get("tool")
            print(f"Tool selected: {tool}")
            if tool == "final_answer":
                return END
            return tool
        
        print("WARNING: Could not find tool in state")
        # Default to final_answer if we can't determine the tool
        return END
    
    # Add conditional edges with the debug router
    workflow.add_conditional_edges(
        "agent",
        debug_router,
        {tool.name: "tools" for tool in tools}
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the workflow
    return workflow.compile()


def create_agent_executor():
    """Creates and returns the agent executor."""
    finance_agent = create_finance_react_agent()
    return finance_agent


def run_agent(agent, query: str, chat_history: List = None) -> Dict[str, Any]:
    """
    Run the finance agent on a user query.
    
    Args:
        agent: The compiled agent workflow
        query: User's query as a string
        chat_history: Optional list of previous messages
        
    Returns:
        Dict with response information
    """
    if chat_history is None:
        chat_history = []
    
    # Create initial state with the user query
    state = {"messages": chat_history + [HumanMessage(content=query)]}
    result = None  # Initialize result to avoid UnboundLocalError
    
    try:
        # Execute the agent with a timeout to prevent hanging
        result = agent.invoke(state)
        
        # Debug the result structure
        print("\nDEBUG - Final Result Structure:")
        print(f"Result keys: {list(result.keys() if isinstance(result, dict) else [])}")
        
        # Check if we have a valid result
        if not result or not isinstance(result, dict):
            return {"answer": "I couldn't process your request. Please try again with a different question.", "full_result": None}
        
        # Look for the final answer in the result
        if "final_answer" in result:
            return {"answer": result["final_answer"], "full_result": result}
        
        # If we have thoughts, use that as context
        if "thoughts" in result:
            thoughts = result["thoughts"]
            print(f"DEBUG - Thoughts: {thoughts}")
        
        # Extract the final message from the conversation
        final_messages = result.get("messages", [])
        if final_messages:
            final_message = final_messages[-1]
            
            # If it's an AI message, try to extract the answer
            if isinstance(final_message, AIMessage):
                content = final_message.content
                print(f"DEBUG - Final AI Message: {content[:100]}...")
                
                # Try to find a structured response in the content
                structured_patterns = [
                    # Look for Key Insights/Analysis/Recommendation sections
                    r"Key Insights:(.*?)(?:Analysis:|Recommendation:|$)",
                    r"Analysis:(.*?)(?:Recommendation:|$)",
                    r"Recommendation:(.*?)$",
                    # Look for any final answer or conclusion
                    r"(?:final answer|conclusion|summary):\s*(.+)$",
                    # Look for direct statements about stock prices or indicators
                    r"((?:the )?(?:current )?price of [A-Z]+ is [\$\d\.]+)",
                    r"((?:the )?RSI (?:for|of) [A-Z]+ is [\d\.]+)"
                ]
                
                for pattern in structured_patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        return {"answer": match.group(1).strip(), "full_result": result}
                
                # If we couldn't extract a specific section, return the full content
                return {"answer": content, "full_result": result}
            else:
                # If it's not an AI message, return it as a string
                return {"answer": str(final_message), "full_result": result}
        
        # If we get here, we have a result but couldn't extract a good answer
        # Use the thoughts as a fallback if available
        if "thoughts" in result:
            return {"answer": f"Based on my analysis: {result['thoughts']}", "full_result": result}
            
        # Last resort fallback
        return {"answer": "I processed your request but couldn't generate a specific answer. Please try a different question.", "full_result": result}
    
    except Exception as e:
        # Provide a helpful error message
        error_msg = str(e)
        if error_msg == "'__end__'":
            # This is actually not an error, but the END state in the workflow
            # Return the final answer or thoughts from the result if available
            if result is not None and isinstance(result, dict):
                if "final_answer" in result:
                    return {"answer": result["final_answer"], "full_result": result}
                elif "thoughts" in result:
                    return {"answer": f"Based on my analysis: {result['thoughts']}", "full_result": result}
            
            # If we can't extract anything useful, provide a generic response
            return {"answer": "I've analyzed your request but couldn't generate a specific answer. The model determined this was a final answer scenario.", "full_result": None}
        elif "thoughts" in error_msg:
            return {"answer": "I'm analyzing your request, but I'm having trouble formatting the response. Let me try a simpler approach: Please ask a more specific financial question about a particular stock or indicator.", "full_result": None}
        else:
            return {"answer": f"I encountered an error while processing your request: {error_msg}. Please try a different question.", "full_result": None}
