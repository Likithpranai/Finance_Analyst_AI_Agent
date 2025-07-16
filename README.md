# Finance Analyst AI Agent

A ReAct agent for financial analysis built with Python, LangChain, LangGraph, and Gemini AI.

## Features

- Stock data analysis (prices, indicators, trends)
- Technical indicator calculations (RSI, moving averages, etc.)
- Market news tracking and summarization
- Question answering for financial insights
- ReAct pattern: Reason → Act → Observe → Loop

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: Entry point for the application
- `agent.py`: Main ReAct agent implementation
- `tools/`: Directory containing financial analysis tools
- `config.py`: Configuration settings
- `requirements.txt`: Project dependencies
