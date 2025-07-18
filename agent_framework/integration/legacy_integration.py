"""
Legacy Integration Module for Finance Analyst AI Agent Framework
Provides integration with the existing Finance Analyst AI Agent
"""
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dotenv import load_dotenv

# Add parent directory to path to allow imports from the original agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import original agent components
try:
    from finance_analyst_agent import FinanceAnalystReActAgent
except ImportError:
    print("Warning: Could not import original FinanceAnalystReActAgent")

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
logger = logging.getLogger("LegacyIntegration")

# Load environment variables
load_dotenv()

class LegacyIntegration:
    """
    Integrates the new agent framework with the existing Finance Analyst AI Agent
    """
    
    def __init__(self):
        """Initialize the legacy integration"""
        # Initialize the original agent
        try:
            self.legacy_agent = FinanceAnalystReActAgent()
            logger.info("Successfully initialized legacy FinanceAnalystReActAgent")
        except Exception as e:
            self.legacy_agent = None
            logger.warning(f"Failed to initialize legacy agent: {str(e)}")
        
        # Initialize the new agent framework components
        self.orchestrator = AgentOrchestrator()
        self.memory = AgentMemory()
        
        # Import legacy tools into the new framework
        if self.legacy_agent:
            self._import_legacy_tools()
    
    def _import_legacy_tools(self):
        """Import tools from the legacy agent into the new framework"""
        if not self.legacy_agent or not hasattr(self.legacy_agent, 'tools'):
            logger.warning("Legacy agent tools not available for import")
            return
        
        # Map of legacy tools to specialized agents
        tool_mapping = {
            # Technical analysis tools
            'get_stock_price': 'technical',
            'get_stock_history': 'technical',
            'calculate_rsi': 'technical',
            'calculate_macd': 'technical',
            'visualize_stock': 'technical',
            
            # Fundamental analysis tools
            'get_company_info': 'fundamental',
            'get_financial_ratios': 'fundamental',
            'get_financial_statements': 'fundamental',
            'get_company_profile': 'fundamental',
            
            # Risk analysis tools
            'calculate_volatility': 'risk',
            'calculate_var': 'risk',
            'calculate_beta': 'risk',
            'calculate_sharpe_ratio': 'risk',
            
            # Trading tools
            'get_trading_signals': 'trading',
            'backtest_strategy': 'trading',
            'simulate_trade': 'trading',
            
            # General tools
            'get_stock_news': 'fundamental'
        }
        
        # Import tools into the appropriate specialized agents
        imported_tools = 0
        for tool_name, tool_func in self.legacy_agent.tools.items():
            if tool_name in tool_mapping:
                agent_type = tool_mapping[tool_name]
                
                if agent_type == 'technical' and hasattr(self.orchestrator, 'technical_agent'):
                    self.orchestrator.technical_agent.register_tool(tool_name, tool_func)
                    imported_tools += 1
                
                elif agent_type == 'fundamental' and hasattr(self.orchestrator, 'fundamental_agent'):
                    self.orchestrator.fundamental_agent.register_tool(tool_name, tool_func)
                    imported_tools += 1
                
                elif agent_type == 'risk' and hasattr(self.orchestrator, 'risk_agent'):
                    self.orchestrator.risk_agent.register_tool(tool_name, tool_func)
                    imported_tools += 1
                
                elif agent_type == 'trading' and hasattr(self.orchestrator, 'trading_agent'):
                    self.orchestrator.trading_agent.register_tool(tool_name, tool_func)
                    imported_tools += 1
        
        logger.info(f"Imported {imported_tools} legacy tools into the new agent framework")
    
    def process_query_with_legacy(self, query: str) -> str:
        """
        Process a query using the legacy agent
        
        Args:
            query: User query string
            
        Returns:
            Response from the legacy agent
        """
        if not self.legacy_agent:
            return "Legacy agent not available"
        
        try:
            response = self.legacy_agent.process_query(query)
            return response
        except Exception as e:
            error_message = f"Error processing query with legacy agent: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message
    
    def process_query_with_new(self, query: str) -> str:
        """
        Process a query using the new agent framework
        
        Args:
            query: User query string
            
        Returns:
            Response from the new agent framework
        """
        try:
            response = self.orchestrator.process_query(query)
            
            # Store conversation in memory
            symbols = self.orchestrator.extract_symbols(query)
            analysis_types = self.orchestrator.classify_query(query)
            self.memory.add_conversation_entry(query, response, symbols, analysis_types)
            
            return response
        except Exception as e:
            error_message = f"Error processing query with new agent framework: {str(e)}"
            logger.error(error_message, exc_info=True)
            return error_message
    
    def process_query(self, query: str, use_legacy: bool = False) -> Dict[str, str]:
        """
        Process a query using both legacy and new agents for comparison
        
        Args:
            query: User query string
            use_legacy: Whether to include legacy agent response
            
        Returns:
            Dictionary with responses from both agents
        """
        results = {}
        
        # Process with new agent framework
        new_response = self.process_query_with_new(query)
        results['new'] = new_response
        
        # Process with legacy agent if requested
        if use_legacy and self.legacy_agent:
            legacy_response = self.process_query_with_legacy(query)
            results['legacy'] = legacy_response
        
        return results
    
    def migrate_legacy_memory(self):
        """Migrate memory from legacy agent to new memory system"""
        if not self.legacy_agent:
            logger.warning("Legacy agent not available for memory migration")
            return
        
        # Check if legacy agent has memory
        if not hasattr(self.legacy_agent, 'memory') or not self.legacy_agent.memory:
            logger.warning("Legacy agent does not have memory to migrate")
            return
        
        try:
            # Migrate conversation history
            if hasattr(self.legacy_agent.memory, 'conversation_history'):
                for entry in self.legacy_agent.memory.conversation_history:
                    if 'query' in entry and 'response' in entry:
                        self.memory.add_conversation_entry(
                            entry['query'],
                            entry['response'],
                            entry.get('symbols', []),
                            entry.get('analysis_types', [])
                        )
            
            # Migrate watched symbols
            if hasattr(self.legacy_agent.memory, 'watched_symbols'):
                for symbol in self.legacy_agent.memory.watched_symbols:
                    self.memory.add_watched_symbol(symbol)
            
            logger.info("Successfully migrated legacy memory to new memory system")
        except Exception as e:
            logger.error(f"Error migrating legacy memory: {str(e)}", exc_info=True)
    
    def get_tool_comparison(self) -> Dict[str, List[str]]:
        """
        Get a comparison of tools available in legacy and new agents
        
        Returns:
            Dictionary with tool lists for legacy and new agents
        """
        comparison = {
            'legacy_tools': [],
            'new_tools': {
                'technical': [],
                'fundamental': [],
                'risk': [],
                'trading': []
            }
        }
        
        # Get legacy tools
        if self.legacy_agent and hasattr(self.legacy_agent, 'tools'):
            comparison['legacy_tools'] = list(self.legacy_agent.tools.keys())
        
        # Get new tools
        if hasattr(self.orchestrator, 'technical_agent') and hasattr(self.orchestrator.technical_agent, 'tools'):
            comparison['new_tools']['technical'] = list(self.orchestrator.technical_agent.tools.keys())
        
        if hasattr(self.orchestrator, 'fundamental_agent') and hasattr(self.orchestrator.fundamental_agent, 'tools'):
            comparison['new_tools']['fundamental'] = list(self.orchestrator.fundamental_agent.tools.keys())
        
        if hasattr(self.orchestrator, 'risk_agent') and hasattr(self.orchestrator.risk_agent, 'tools'):
            comparison['new_tools']['risk'] = list(self.orchestrator.risk_agent.tools.keys())
        
        if hasattr(self.orchestrator, 'trading_agent') and hasattr(self.orchestrator.trading_agent, 'tools'):
            comparison['new_tools']['trading'] = list(self.orchestrator.trading_agent.tools.keys())
        
        return comparison
