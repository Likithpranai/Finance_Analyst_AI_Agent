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

    Follow the ReAct pattern to solve user queries:
    - REASON: Carefully think about what information you need and which tools to use
    - ACT: Use the appropriate tools to gather data and perform analysis
    - OBSERVE: Process the results from tools
    - Repeat until you have enough information to provide a comprehensive answer
    
    Format your final response as follows:
    1. Key Insights: Brief bullet points summarizing main findings
    2. Analysis: Detailed explanation with data to support your conclusions
    3. Recommendation (if applicable): Suggested actions based on your analysis
    
    YOUR RESPONSE MUST BE IN JSON FORMAT with these keys:
    {
        "thoughts": "Your reasoning process (not shown to user)",
        "tool": "The name of the tool to use (or 'final_answer' if providing final response)",
        "tool_input": "The input to the tool as a string (omit if tool is 'final_answer')",
        "final_answer": "Your comprehensive answer to the user's question (only include if tool is 'final_answer')"
    }
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
        # Format the tool-use history for the prompt
        tools_output = state.get("tools_output", {})
        agent_scratchpad = []
        
        # Add tool outputs to scratchpad if available
        if tools_output:
            result = tools_output.get("result", "")
            tool_name = tools_output.get("tool", "")
            agent_scratchpad.append(HumanMessage(content=f"Tool {tool_name} returned: {result}"))
        
        # Invoke the LLM
        response = llm.invoke(
            prompt.format_messages(
                messages=state["messages"],
                agent_scratchpad=agent_scratchpad
            )
        )
        
        # Parse the JSON response
        # Handle potential issues with JSON parsing
        try:
            content = response.content
            # Check if the content is surrounded by markdown code fence
            match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if match:
                content = match.group(1)
            
            parsed_response = json.loads(content)
            
        except Exception as e:
            # If JSON parsing fails, create a structured output from the text
            content = response.content
            if "final_answer" in content.lower():
                parsed_response = {
                    "thoughts": "Providing final answer to user query",
                    "tool": "final_answer",
                    "final_answer": content
                }
            else:
                # Try to extract a tool name from the content
                tool_match = re.search(r"I should use the (\w+) tool", content)
                if tool_match:
                    tool_name = tool_match.group(1)
                    parsed_response = {
                        "thoughts": content,
                        "tool": tool_name,
                        "tool_input": "{}"
                    }
                else:
                    # Default to final answer if we can't parse properly
                    parsed_response = {
                        "thoughts": "JSON parsing failed, providing best response",
                        "tool": "final_answer",
                        "final_answer": content
                    }
        
        return parsed_response

    # Create the LangGraph workflow
    workflow = StateGraph(AgentState)
    
    # Add the agent node
    workflow.add_node("agent", agent_node)
    
    # Add the tools node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        lambda state, result: result.get("tool") if result.get("tool") != "final_answer" else END,
        {
            tool.name: "tools" for tool in tools
        }
    )
    
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
        
    # Create initial state
    state = {"messages": chat_history + [HumanMessage(content=query)]}
    
    # Execute the agent
    result = agent.invoke(state)
    
    # Extract final answer
    final_message = result["messages"][-1]
    
    # Parse the final answer from the message
    if isinstance(final_message, AIMessage):
        try:
            content = final_message.content
            # Check if the content is surrounded by markdown code fence
            match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if match:
                content = match.group(1)
                
            parsed_content = json.loads(content)
            final_answer = parsed_content.get("final_answer", content)
        except:
            final_answer = final_message.content
    else:
        final_answer = str(final_message)
    
    # Return the final response
    return {
        "answer": final_answer,
        "full_result": result
    }
