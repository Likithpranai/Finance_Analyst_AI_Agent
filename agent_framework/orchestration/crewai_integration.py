"""
CrewAI Integration for Finance Analyst AI Agent Framework
Provides integration with CrewAI for multi-agent collaboration
"""
import os
from typing import Dict, List, Any, Optional, Union, Callable
from dotenv import load_dotenv

# Import specialized agents
from agent_framework.agents.technical_analysis_agent import TechnicalAnalysisAgent
from agent_framework.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from agent_framework.agents.risk_analysis_agent import RiskAnalysisAgent
from agent_framework.agents.trading_agent import TradingAgent

# Import CrewAI components
try:
    from crewai import Agent as CrewAgent
    from crewai import Task, Crew, Process
    from crewai.tools.tool import Tool as CrewTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("CrewAI not available. Install with 'pip install crewai'")

# Load environment variables
load_dotenv()

class CrewAIIntegration:
    """
    Integrates CrewAI capabilities with the Finance Analyst AI Agent Framework
    """
    
    def __init__(self):
        """Initialize CrewAI integration"""
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not available. Install with 'pip install crewai'")
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize specialized agents
        self.technical_agent = TechnicalAnalysisAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.trading_agent = TradingAgent()
    
    def _convert_to_crew_tools(self, tools_dict: Dict[str, Callable]) -> List[CrewTool]:
        """Convert tools dictionary to CrewAI tools"""
        crew_tools = []
        
        for name, func in tools_dict.items():
            # Get function signature and docstring
            import inspect
            docstring = inspect.getdoc(func) or f"Tool for {name}"
            
            # Create CrewAI tool
            crew_tools.append(
                CrewTool(
                    name=name,
                    description=docstring,
                    func=func
                )
            )
        
        return crew_tools
    
    def create_technical_analyst(self) -> CrewAgent:
        """Create a CrewAI agent for technical analysis"""
        # Convert tools to CrewAI format
        tools = self._convert_to_crew_tools(self.technical_agent.tools)
        
        # Create CrewAI agent
        return CrewAgent(
            role="Technical Analyst",
            goal="Analyze stock price patterns and technical indicators to identify trading opportunities",
            backstory="""You are an expert technical analyst with years of experience in chart analysis 
            and technical indicators. You specialize in identifying chart patterns, support/resistance levels, 
            and technical signals that can predict future price movements.""",
            verbose=True,
            allow_delegation=True,
            tools=tools
        )
    
    def create_fundamental_analyst(self) -> CrewAgent:
        """Create a CrewAI agent for fundamental analysis"""
        # Convert tools to CrewAI format
        tools = self._convert_to_crew_tools(self.fundamental_agent.tools)
        
        # Create CrewAI agent
        return CrewAgent(
            role="Fundamental Analyst",
            goal="Analyze company financials and business fundamentals to determine intrinsic value",
            backstory="""You are a seasoned fundamental analyst with expertise in financial statement analysis 
            and business valuation. You excel at evaluating a company's financial health, competitive position, 
            and growth prospects to determine its true value.""",
            verbose=True,
            allow_delegation=True,
            tools=tools
        )
    
    def create_risk_manager(self) -> CrewAgent:
        """Create a CrewAI agent for risk analysis"""
        # Convert tools to CrewAI format
        tools = self._convert_to_crew_tools(self.risk_agent.tools)
        
        # Create CrewAI agent
        return CrewAgent(
            role="Risk Manager",
            goal="Evaluate investment risks and provide risk management strategies",
            backstory="""You are a meticulous risk manager with deep knowledge of portfolio risk metrics 
            and risk mitigation strategies. You focus on identifying potential risks, quantifying their impact, 
            and developing strategies to protect investments.""",
            verbose=True,
            allow_delegation=True,
            tools=tools
        )
    
    def create_trading_strategist(self) -> CrewAgent:
        """Create a CrewAI agent for trading strategy"""
        # Convert tools to CrewAI format
        tools = self._convert_to_crew_tools(self.trading_agent.tools)
        
        # Create CrewAI agent
        return CrewAgent(
            role="Trading Strategist",
            goal="Develop actionable trading strategies with specific entry/exit points",
            backstory="""You are a skilled trading strategist who combines technical and fundamental insights 
            to create practical trading plans. You excel at determining optimal entry and exit points, 
            position sizing, and risk-reward ratios for trades.""",
            verbose=True,
            allow_delegation=True,
            tools=tools
        )
    
    def create_comprehensive_analysis_crew(self, symbol: str) -> Crew:
        """
        Create a crew for comprehensive stock analysis
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            CrewAI Crew for comprehensive analysis
        """
        # Create agents
        technical_analyst = self.create_technical_analyst()
        fundamental_analyst = self.create_fundamental_analyst()
        risk_manager = self.create_risk_manager()
        trading_strategist = self.create_trading_strategist()
        
        # Create tasks
        technical_analysis_task = Task(
            description=f"Perform a comprehensive technical analysis of {symbol}. Include trend analysis, support/resistance levels, and key technical indicators (RSI, MACD, moving averages). Identify chart patterns and potential price targets.",
            agent=technical_analyst,
            expected_output="Detailed technical analysis report with chart patterns, indicators, and price targets."
        )
        
        fundamental_analysis_task = Task(
            description=f"Conduct a thorough fundamental analysis of {symbol}. Evaluate financial statements, key ratios, growth metrics, and competitive positioning. Determine a fair value estimate and growth outlook.",
            agent=fundamental_analyst,
            expected_output="Comprehensive fundamental analysis with financial health assessment, valuation metrics, and fair value estimate."
        )
        
        risk_assessment_task = Task(
            description=f"Assess the risk profile of {symbol} as an investment. Calculate volatility metrics, maximum drawdown, and risk-adjusted returns. Identify key risk factors and potential hedging strategies.",
            agent=risk_manager,
            expected_output="Risk assessment report with quantified risk metrics and risk management recommendations."
        )
        
        trading_strategy_task = Task(
            description=f"Develop a trading strategy for {symbol} based on the technical, fundamental, and risk analyses. Provide specific entry/exit points, position sizing recommendations, and risk management rules.",
            agent=trading_strategist,
            context=[technical_analysis_task, fundamental_analysis_task, risk_assessment_task],
            expected_output="Actionable trading strategy with entry/exit points, position sizing, and risk management rules."
        )
        
        # Create crew
        crew = Crew(
            agents=[technical_analyst, fundamental_analyst, risk_manager, trading_strategist],
            tasks=[technical_analysis_task, fundamental_analysis_task, risk_assessment_task, trading_strategy_task],
            verbose=2,
            process=Process.sequential
        )
        
        return crew
    
    def create_portfolio_optimization_crew(self, symbols: List[str]) -> Crew:
        """
        Create a crew for portfolio optimization
        
        Args:
            symbols: List of stock symbols in the portfolio
            
        Returns:
            CrewAI Crew for portfolio optimization
        """
        # Create agents
        technical_analyst = self.create_technical_analyst()
        fundamental_analyst = self.create_fundamental_analyst()
        risk_manager = self.create_risk_manager()
        
        # Create tasks
        market_analysis_task = Task(
            description=f"Analyze the current market conditions and how they might affect the portfolio of {', '.join(symbols)}. Consider market trends, sector performance, and macroeconomic factors.",
            agent=technical_analyst,
            expected_output="Market analysis report with insights on current market conditions and their impact on the portfolio."
        )
        
        stock_screening_task = Task(
            description=f"Evaluate each stock in the portfolio ({', '.join(symbols)}) based on fundamental metrics. Identify strengths and weaknesses of each position.",
            agent=fundamental_analyst,
            expected_output="Stock screening report with fundamental evaluation of each position in the portfolio."
        )
        
        portfolio_risk_task = Task(
            description=f"Analyze the risk characteristics of the portfolio containing {', '.join(symbols)}. Calculate portfolio volatility, correlation matrix, diversification metrics, and value at risk.",
            agent=risk_manager,
            expected_output="Portfolio risk analysis with diversification assessment and risk metrics."
        )
        
        optimization_task = Task(
            description=f"Optimize the portfolio allocation for {', '.join(symbols)} based on the risk and return characteristics. Recommend weight adjustments, potential additions or removals, and rebalancing strategy.",
            agent=risk_manager,
            context=[market_analysis_task, stock_screening_task, portfolio_risk_task],
            expected_output="Portfolio optimization recommendations with target allocations and rebalancing strategy."
        )
        
        # Create crew
        crew = Crew(
            agents=[technical_analyst, fundamental_analyst, risk_manager],
            tasks=[market_analysis_task, stock_screening_task, portfolio_risk_task, optimization_task],
            verbose=2,
            process=Process.sequential
        )
        
        return crew
    
    def run_comprehensive_analysis(self, symbol: str) -> str:
        """
        Run a comprehensive analysis for a stock symbol
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Comprehensive analysis report
        """
        # Create crew
        crew = self.create_comprehensive_analysis_crew(symbol)
        
        # Run crew
        result = crew.kickoff(inputs={"symbol": symbol})
        
        return result
    
    def run_portfolio_optimization(self, symbols: List[str]) -> str:
        """
        Run portfolio optimization for a list of symbols
        
        Args:
            symbols: List of stock symbols in the portfolio
            
        Returns:
            Portfolio optimization report
        """
        # Create crew
        crew = self.create_portfolio_optimization_crew(symbols)
        
        # Run crew
        result = crew.kickoff(inputs={"symbols": symbols})
        
        return result
