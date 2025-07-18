#!/usr/bin/env python3
"""
Migration Script for Finance Analyst AI Agent
Helps users transition from the legacy agent to the new multi-agent framework
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
import time

# Import integration module
from agent_framework.integration.legacy_integration import LegacyIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Migration")

# Load environment variables
load_dotenv()

class AgentMigration:
    """Migration utility for Finance Analyst AI Agent"""
    
    def __init__(self):
        """Initialize the migration utility"""
        self.integration = LegacyIntegration()
    
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "="))
        print("=" * 80 + "\n")
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print("\n" + "-" * 80)
        print(f" {title} ".center(80, "-"))
        print("-" * 80 + "\n")
    
    def compare_tools(self):
        """Compare tools between legacy and new agents"""
        self.print_section("TOOL COMPARISON")
        
        tool_comparison = self.integration.get_tool_comparison()
        
        # Print legacy tools
        print("Legacy Agent Tools:")
        for tool in sorted(tool_comparison['legacy_tools']):
            print(f"  - {tool}")
        
        print("\nEnhanced Agent Tools:")
        for agent_type, tools in tool_comparison['new_tools'].items():
            print(f"  {agent_type.capitalize()} Agent:")
            for tool in sorted(tools):
                print(f"    - {tool}")
        
        print("\n")
    
    def migrate_memory(self):
        """Migrate memory from legacy agent to new memory system"""
        self.print_section("MEMORY MIGRATION")
        
        print("Migrating memory from legacy agent to new memory system...")
        start_time = time.time()
        self.integration.migrate_legacy_memory()
        elapsed_time = time.time() - start_time
        
        print(f"Memory migration completed in {elapsed_time:.2f} seconds")
        print("\n")
    
    def compare_responses(self, query: str):
        """Compare responses between legacy and new agents"""
        self.print_section(f"RESPONSE COMPARISON: {query}")
        
        print("Processing query with both agents...")
        start_time = time.time()
        responses = self.integration.process_query(query, use_legacy=True)
        elapsed_time = time.time() - start_time
        
        if 'legacy' in responses:
            print("\nLegacy Agent Response:")
            print(responses['legacy'])
        
        print("\nEnhanced Agent Response:")
        print(responses['new'])
        
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print("\n")
    
    def run_migration(self, compare_queries: List[str] = None):
        """Run the full migration process"""
        self.print_header("FINANCE ANALYST AI AGENT MIGRATION")
        print("This utility helps migrate from the legacy agent to the enhanced multi-agent framework")
        
        # Check for API keys
        if not os.getenv("GEMINI_API_KEY"):
            print("\nWARNING: GEMINI_API_KEY not found in environment variables.")
            print("The migration will not work without a valid API key.")
            return
        
        # Compare tools
        self.compare_tools()
        
        # Migrate memory
        self.migrate_memory()
        
        # Compare responses for sample queries
        if compare_queries:
            for query in compare_queries:
                self.compare_responses(query)
        
        self.print_header("MIGRATION COMPLETED")
        print("The migration process has completed successfully.")
        print("You can now use the enhanced Finance Analyst AI Agent for your financial analysis.")
        print("\nTo run the enhanced agent:")
        print("  python enhanced_finance_analyst.py -i")
        print("\nTo run the demo script:")
        print("  python demo_enhanced_agent.py --all")

def main():
    """Main entry point for the migration script"""
    parser = argparse.ArgumentParser(description="Finance Analyst AI Agent Migration")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare responses between legacy and new agents")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize the migration utility
    migration = AgentMigration()
    
    # Sample queries for comparison
    sample_queries = [
        "What's the current price of AAPL?",
        "Calculate the RSI for MSFT",
        "Give me a fundamental analysis of GOOGL",
        "What's the risk profile of a portfolio with AAPL, MSFT, and AMZN?",
        "Should I buy TSLA based on technical and fundamental factors?"
    ]
    
    # Run migration with or without response comparison
    if args.compare:
        migration.run_migration(sample_queries)
    else:
        migration.run_migration()

if __name__ == "__main__":
    main()
